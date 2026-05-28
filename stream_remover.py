from __future__ import annotations

import argparse
from dataclasses import dataclass
from typing import Iterable, List, Optional, Sequence, Set, Tuple

import cv2
import numpy as np


@dataclass(frozen=True)
class TrackedInstance:
    track_id: int
    bbox: Tuple[float, float, float, float]
    mask: np.ndarray


def bbox_iou(a: Sequence[float], b: Sequence[float]) -> float:
    ax1, ay1, ax2, ay2 = a
    bx1, by1, bx2, by2 = b
    ix1, iy1 = max(ax1, bx1), max(ay1, by1)
    ix2, iy2 = min(ax2, bx2), min(ay2, by2)
    iw, ih = max(0.0, ix2 - ix1), max(0.0, iy2 - iy1)
    inter = iw * ih
    if inter <= 0:
        return 0.0
    area_a = max(0.0, ax2 - ax1) * max(0.0, ay2 - ay1)
    area_b = max(0.0, bx2 - bx1) * max(0.0, by2 - by1)
    union = area_a + area_b - inter
    return inter / union if union > 0 else 0.0


def match_selected_targets_to_ids(
    selected_boxes: Iterable[Sequence[float]],
    tracked_instances: Iterable[TrackedInstance],
    min_iou: float = 0.01,
) -> Set[int]:
    selected = list(selected_boxes)
    matched: Set[int] = set()
    for instance in tracked_instances:
        if any(bbox_iou(instance.bbox, box) >= min_iou for box in selected):
            matched.add(int(instance.track_id))
    return matched


def apply_background_replacement(
    frame: np.ndarray,
    background: np.ndarray,
    masks: Iterable[np.ndarray],
) -> Tuple[np.ndarray, np.ndarray]:
    out = frame.copy()
    repair_mask = np.zeros(frame.shape[:2], dtype=np.uint8)
    for m in masks:
        m_bool = m.astype(bool)
        out[m_bool] = background[m_bool]
        repair_mask[m_bool] = 255
    return out, repair_mask


class YoloSegTracker:
    def __init__(self, model_path: str, conf: float = 0.25, iou: float = 0.5) -> None:
        from ultralytics import YOLO  # type: ignore

        self.model = YOLO(model_path)
        self.conf = conf
        self.iou = iou

    def infer(self, frame: np.ndarray) -> List[TrackedInstance]:
        results = self.model.track(
            frame, persist=True, verbose=False, conf=self.conf, iou=self.iou
        )
        if not results:
            return []
        result = results[0]
        if result.boxes is None or result.boxes.xyxy is None:
            return []
        boxes = result.boxes.xyxy.cpu().numpy()
        ids = (
            result.boxes.id.cpu().numpy().astype(int)
            if result.boxes.id is not None
            else np.full((len(boxes),), -1, dtype=int)
        )
        if result.masks is None or result.masks.data is None:
            return []
        masks = result.masks.data.cpu().numpy() > 0.5
        h, w = frame.shape[:2]
        if masks.shape[1:] != (h, w):
            resized = []
            for m in masks:
                resized.append(
                    cv2.resize(m.astype(np.uint8), (w, h), interpolation=cv2.INTER_NEAREST)
                    > 0
                )
            masks = np.array(resized, dtype=bool)
        instances: List[TrackedInstance] = []
        for idx, box in enumerate(boxes):
            if idx >= len(masks):
                break
            track_id = int(ids[idx]) if idx < len(ids) else -1
            if track_id < 0:
                continue
            instances.append(
                TrackedInstance(
                    track_id=track_id, bbox=tuple(float(v) for v in box), mask=masks[idx]
                )
            )
        return instances


class StreamRemoverApp:
    MIN_HUD_Y_POSITION = 24

    def __init__(
        self,
        camera_index: int,
        tracker: YoloSegTracker,
        alpha: float = 0.02,
        window_name: str = "Camera Stream Remover",
    ) -> None:
        self.cap = cv2.VideoCapture(camera_index)
        self.tracker = tracker
        self.alpha = alpha
        self.window_name = window_name
        w = int(self.cap.get(cv2.CAP_PROP_FRAME_WIDTH)) or 640
        h = int(self.cap.get(cv2.CAP_PROP_FRAME_HEIGHT)) or 480
        self.fallback_frame_shape = (h, w, 3)
        self.paused = False
        self.learning_enabled = True
        self.current_frame: Optional[np.ndarray] = None
        self.paused_snapshot: Optional[np.ndarray] = None
        self.background_float: Optional[np.ndarray] = None
        self.pending_selected_boxes: List[Tuple[int, int, int, int]] = []
        self.erasing_target_ids: Set[int] = set()

    def _match_selected_targets_to_ids(
        self, tracked_instances: Iterable[TrackedInstance]
    ) -> None:
        if not self.pending_selected_boxes:
            return
        matched = match_selected_targets_to_ids(
            self.pending_selected_boxes, tracked_instances, min_iou=0.01
        )
        self.erasing_target_ids.update(matched)
        if matched:
            self.pending_selected_boxes.clear()

    def _update_background(self, frame: np.ndarray) -> None:
        if self.background_float is None:
            self.background_float = frame.astype(np.float32)
            return
        cv2.accumulateWeighted(frame.astype(np.float32), self.background_float, self.alpha)

    def _draw_hud(self, frame: np.ndarray) -> np.ndarray:
        out = frame.copy()
        status = "PAUSED" if self.paused else "STREAMING"
        mode = "LEARNING:ON" if self.learning_enabled else "LEARNING:OFF"
        cv2.putText(
            out,
            f"Status: {status} | {mode}",
            (12, 28),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.7,
            (20, 220, 20),
            2,
            cv2.LINE_AA,
        )
        cv2.putText(
            out,
            f"Erasing Targets Count: {len(self.erasing_target_ids)}",
            (12, 56),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.7,
            (20, 220, 20),
            2,
            cv2.LINE_AA,
        )
        cv2.putText(
            out,
            "Space: Pause/Resume  S: Select  C: Clear  L: Learn Toggle  Q/Esc: Quit",
            (12, max(self.MIN_HUD_Y_POSITION, out.shape[0] - 12)),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.52,
            (230, 230, 230),
            1,
            cv2.LINE_AA,
        )
        return out

    def _should_update_background(self) -> bool:
        return (
            self.learning_enabled
            and not self.erasing_target_ids
            and not self.pending_selected_boxes
        )

    def _select_rois(self) -> None:
        if self.paused_snapshot is None:
            return
        rois = cv2.selectROIs(self.window_name, self.paused_snapshot, False, False)
        for x, y, w, h in rois:
            if w > 0 and h > 0:
                self.pending_selected_boxes.append((x, y, x + w, y + h))

    def run(self) -> None:
        if not self.cap.isOpened():
            raise RuntimeError("Cannot open camera source.")
        cv2.namedWindow(self.window_name, cv2.WINDOW_NORMAL)
        while True:
            if not self.paused:
                ok, frame = self.cap.read()
                if not ok:
                    break
                self.current_frame = frame
                tracked = self.tracker.infer(frame)
                self._match_selected_targets_to_ids(tracked)

                if self._should_update_background():
                    self._update_background(frame)

                output = frame
                if self.background_float is not None and self.erasing_target_ids:
                    background = np.clip(self.background_float, 0, 255).astype(np.uint8)
                    target_masks = [
                        obj.mask for obj in tracked if obj.track_id in self.erasing_target_ids
                    ]
                    if target_masks:
                        replaced, repair = apply_background_replacement(
                            frame, background, target_masks
                        )
                        edge = cv2.morphologyEx(
                            repair, cv2.MORPH_GRADIENT, np.ones((3, 3), np.uint8)
                        )
                        output = cv2.inpaint(replaced, edge, 3, cv2.INPAINT_TELEA)
            else:
                if self.paused_snapshot is None and self.current_frame is not None:
                    self.paused_snapshot = self.current_frame.copy()
                output = (
                    self.paused_snapshot
                    if self.paused_snapshot is not None
                    else np.zeros(self.fallback_frame_shape, dtype=np.uint8)
                )

            cv2.imshow(self.window_name, self._draw_hud(output))
            key = cv2.waitKey(1) & 0xFF
            if key in (ord("q"), 27):
                break
            if key == ord(" "):
                self.paused = not self.paused
                if self.paused:
                    self.paused_snapshot = self.current_frame.copy() if self.current_frame is not None else None
                else:
                    self.paused_snapshot = None
            elif key == ord("s"):
                if not self.paused:
                    self.paused = True
                    self.paused_snapshot = self.current_frame.copy() if self.current_frame is not None else None
                self._select_rois()
            elif key == ord("c"):
                self.erasing_target_ids.clear()
                self.pending_selected_boxes.clear()
            elif key == ord("l"):
                self.learning_enabled = not self.learning_enabled

        self.cap.release()
        cv2.destroyAllWindows()


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Realtime camera stream target remover")
    parser.add_argument("--camera", type=int, default=0, help="Camera index")
    parser.add_argument(
        "--model",
        type=str,
        default="yolov8n-seg.pt",
        help="YOLO segmentation model path",
    )
    parser.add_argument("--conf", type=float, default=0.25, help="Detection confidence")
    parser.add_argument("--iou", type=float, default=0.5, help="NMS IoU threshold")
    parser.add_argument(
        "--bg-alpha",
        type=float,
        default=0.02,
        help="Background running average alpha",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    tracker = YoloSegTracker(model_path=args.model, conf=args.conf, iou=args.iou)
    app = StreamRemoverApp(camera_index=args.camera, tracker=tracker, alpha=args.bg_alpha)
    app.run()


if __name__ == "__main__":
    main()
