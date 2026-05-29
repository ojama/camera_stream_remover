from __future__ import annotations

import argparse
from dataclasses import dataclass
from typing import Iterable, List, Optional, Sequence, Set, Tuple

import cv2
import numpy as np

try:
    from ultralytics import YOLO  # type: ignore
except ImportError:  # pragma: no cover - depends on optional runtime dependency
    YOLO = None


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


def point_hits_instance(x: int, y: int, instance: TrackedInstance) -> bool:
    """Return True if pixel (x, y) falls inside the instance's segmentation mask.

    Falls back to the bounding box when the mask does not cover that pixel
    (e.g. after resizing artefacts or when the mask array is unexpectedly shaped).
    """
    mask = instance.mask
    if 0 <= y < mask.shape[0] and 0 <= x < mask.shape[1]:
        return bool(mask[y, x])
    # Fallback: bounding-box containment
    x1, y1, x2, y2 = instance.bbox
    return x1 <= x <= x2 and y1 <= y <= y2


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
        if YOLO is None:
            raise ImportError(
                "ultralytics is required for YoloSegTracker. "
                "Install it with `pip install ultralytics` or replace YoloSegTracker."
            )

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
    HUD_X_MARGIN = 12
    HUD_Y_OFFSET = 12
    HUD_STATUS_Y_POSITION = 28
    HUD_COUNT_Y_POSITION = 56
    HUD_HELP_FONT_SCALE = 0.52
    DEFAULT_FRAME_WIDTH = 640
    DEFAULT_FRAME_HEIGHT = 480
    MORPHOLOGY_KERNEL_SIZE = (3, 3)
    INPAINT_RADIUS = 3
    MASK_OVERLAY_ALPHA_DETECTED = 0.15  # dim overlay for every detected instance
    MASK_OVERLAY_ALPHA_ERASING = 0.45  # bright overlay for erasing targets

    def __init__(
        self,
        camera_index: int,
        tracker: YoloSegTracker,
        alpha: float = 0.02,
        window_name: str = "Camera Stream Remover",
        erasure_expand_pixels: int = 0,
        selected_target_alpha: float = 0.45,
    ) -> None:
        self.cap = cv2.VideoCapture(camera_index)
        self.tracker = tracker
        self.alpha = alpha
        self.window_name = window_name
        self.erasure_expand_pixels = max(0, erasure_expand_pixels)
        self.selected_target_alpha = max(0.0, selected_target_alpha)
        w = int(self.cap.get(cv2.CAP_PROP_FRAME_WIDTH)) or self.DEFAULT_FRAME_WIDTH
        h = int(self.cap.get(cv2.CAP_PROP_FRAME_HEIGHT)) or self.DEFAULT_FRAME_HEIGHT
        self.fallback_frame_shape = (h, w, 3)
        self.paused = False
        self.learning_enabled = True
        self.current_frame: Optional[np.ndarray] = None
        self.paused_snapshot: Optional[np.ndarray] = None
        self.background_float: Optional[np.ndarray] = None
        self.erasing_target_ids: Set[int] = set()
        self.last_tracked_instances: List[TrackedInstance] = []

    def _update_background(self, frame: np.ndarray) -> None:
        if self.background_float is None:
            self.background_float = frame.astype(np.float32)
            return
        cv2.accumulateWeighted(frame.astype(np.float32), self.background_float, self.alpha)

    def _should_update_background(self) -> bool:
        return self.learning_enabled and not self.erasing_target_ids

    def _overlay_tracked_masks(
        self, frame: np.ndarray, tracked: List[TrackedInstance]
    ) -> np.ndarray:
        """Overlay semi-transparent green masks on detected instances.

        All detected instances get a dim green hint so the user knows they are
        clickable. Instances selected for erasure can use a custom alpha, or
        hide the overlay entirely when alpha is 0.
        """
        out = frame.copy()
        green = np.array([0, 255, 0], dtype=np.float32)
        for inst in tracked:
            if inst.track_id in self.erasing_target_ids:
                alpha = self.selected_target_alpha
                if alpha <= 0.0:
                    continue
            else:
                alpha = self.MASK_OVERLAY_ALPHA_DETECTED
                
            m = inst.mask.astype(bool)
            if alpha >= 0.1:
                out[m] = np.clip(
                    out[m].astype(np.float32) * (1.0 - alpha) + green * alpha, 0, 255
                ).astype(np.uint8)
            else:
                out[m] = np.clip(
                    out[m].astype(np.float32), 0, 255
                ).astype(np.uint8)    
        return out

    def _on_mouse(self, event: int, x: int, y: int, _flags: int, _param: object) -> None:
        """Mouse callback: left-click toggles a target into / out of the erase set."""
        if event != cv2.EVENT_LBUTTONDOWN:
            return
        for inst in self.last_tracked_instances:
            if point_hits_instance(x, y, inst):
                if inst.track_id in self.erasing_target_ids:
                    self.erasing_target_ids.discard(inst.track_id)
                else:
                    self.erasing_target_ids.add(inst.track_id)
                return

    def _expand_mask(self, mask: np.ndarray) -> np.ndarray:
        if self.erasure_expand_pixels <= 0:
            return mask
        kernel_size = 2 * self.erasure_expand_pixels + 1
        kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (kernel_size, kernel_size))
        expanded = cv2.dilate(mask.astype(np.uint8), kernel, iterations=1)
        return expanded.astype(bool)

    def _draw_hud(self, frame: np.ndarray) -> np.ndarray:
        out = frame.copy()
        status = "PAUSED" if self.paused else "STREAMING"
        mode = "LEARNING:ON" if self.learning_enabled else "LEARNING:OFF"
        cv2.putText(
            out,
            f"Status: {status} | {mode}",
            (self.HUD_X_MARGIN, self.HUD_STATUS_Y_POSITION),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.7,
            (20, 220, 20),
            2,
            cv2.LINE_AA,
        )
        cv2.putText(
            out,
            f"Erasing Targets Count: {len(self.erasing_target_ids)}",
            (self.HUD_X_MARGIN, self.HUD_COUNT_Y_POSITION),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.7,
            (20, 220, 20),
            2,
            cv2.LINE_AA,
        )
        cv2.putText(
            out,
            "Click: Select/Deselect  Space: Pause/Resume  C: Clear  L: Learn Toggle  Q/Esc: Quit",
            (
                self.HUD_X_MARGIN,
                max(self.MIN_HUD_Y_POSITION, out.shape[0] - self.HUD_Y_OFFSET),
            ),
            cv2.FONT_HERSHEY_SIMPLEX,
            self.HUD_HELP_FONT_SCALE,
            (230, 230, 230),
            1,
            cv2.LINE_AA,
        )
        return out

    def _set_paused_snapshot_from_current(self) -> None:
        self.paused_snapshot = (
            self.current_frame.copy() if self.current_frame is not None else None
        )

    def run(self) -> None:
        if not self.cap.isOpened():
            raise RuntimeError("Cannot open camera source.")
        cv2.namedWindow(self.window_name, cv2.WINDOW_NORMAL)
        cv2.setMouseCallback(self.window_name, self._on_mouse)
        while True:
            if not self.paused:
                ok, frame = self.cap.read()
                if not ok:
                    break
                self.current_frame = frame
                tracked = self.tracker.infer(frame)
                self.last_tracked_instances = tracked

                if self._should_update_background():
                    self._update_background(frame)

                output = frame
                if self.background_float is not None and self.erasing_target_ids:
                    background = np.clip(self.background_float, 0, 255).astype(np.uint8)
                    target_masks = [
                        self._expand_mask(obj.mask)
                        for obj in tracked
                        if obj.track_id in self.erasing_target_ids
                    ]
                    if target_masks:
                        replaced, repair = apply_background_replacement(
                            frame, background, target_masks
                        )
                        edge = cv2.morphologyEx(
                            repair,
                            cv2.MORPH_GRADIENT,
                            np.ones(self.MORPHOLOGY_KERNEL_SIZE, np.uint8),
                        )
                        output = cv2.inpaint(
                            replaced, edge, self.INPAINT_RADIUS, cv2.INPAINT_TELEA
                        )

                output = self._overlay_tracked_masks(output, tracked)
            else:
                if self.paused_snapshot is None and self.current_frame is not None:
                    self.paused_snapshot = self.current_frame.copy()
                output = (
                    self.paused_snapshot
                    if self.paused_snapshot is not None
                    else np.zeros(self.fallback_frame_shape, dtype=np.uint8)
                )
                output = self._overlay_tracked_masks(output, self.last_tracked_instances)

            cv2.imshow(self.window_name, self._draw_hud(output))
            key = cv2.waitKey(1) & 0xFF
            if key in (ord("q"), ord("Q"), 27):
                break
            if key == ord(" "):
                self.paused = not self.paused
                if self.paused:
                    self._set_paused_snapshot_from_current()
                else:
                    self.paused_snapshot = None
            elif key in (ord("c"), ord("C")):
                self.erasing_target_ids.clear()
            elif key in (ord("l"), ord("L")):
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
    parser.add_argument(
        "--nms-iou",
        "--iou",
        dest="iou",
        type=float,
        default=0.5,
        help="NMS IoU threshold",
    )
    parser.add_argument(
        "--bg-alpha",
        type=float,
        default=0.02,
        help="Background running average alpha",
    )
    parser.add_argument(
        "--erasure-expand",
        type=int,
        default=0,
        help="Expand the erased mask by this many pixels",
    )
    parser.add_argument(
        "--selected-alpha",
        type=float,
        default=0.45,
        help="Alpha value for selected target overlay (0 disables it)",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    tracker = YoloSegTracker(model_path=args.model, conf=args.conf, iou=args.iou)
    app = StreamRemoverApp(
        camera_index=args.camera,
        tracker=tracker,
        alpha=args.bg_alpha,
        erasure_expand_pixels=args.erasure_expand,
        selected_target_alpha=args.selected_alpha,
    )
    app.run()


if __name__ == "__main__":
    main()
