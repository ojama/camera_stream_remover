import unittest

import numpy as np

from stream_remover import (
    TrackedInstance,
    apply_background_replacement,
    match_selected_targets_to_ids,
    point_hits_instance,
)


class StreamRemoverCoreTests(unittest.TestCase):
    def test_match_selected_targets_to_ids(self) -> None:
        selected = [(10, 10, 40, 40), (100, 100, 120, 120)]
        instances = [
            TrackedInstance(track_id=7, bbox=(12, 12, 30, 30), mask=np.zeros((2, 2), bool)),
            TrackedInstance(track_id=8, bbox=(80, 80, 90, 90), mask=np.zeros((2, 2), bool)),
            TrackedInstance(track_id=9, bbox=(101, 101, 119, 119), mask=np.zeros((2, 2), bool)),
        ]
        matched = match_selected_targets_to_ids(selected, instances, min_iou=0.01)
        self.assertEqual(matched, {7, 9})

    def test_apply_background_replacement(self) -> None:
        frame = np.zeros((3, 3, 3), dtype=np.uint8)
        background = np.full((3, 3, 3), 200, dtype=np.uint8)
        mask = np.array(
            [
                [0, 1, 0],
                [1, 1, 0],
                [0, 0, 0],
            ],
            dtype=bool,
        )
        out, repair = apply_background_replacement(frame, background, [mask])
        self.assertTrue(np.array_equal(out[mask], background[mask]))
        self.assertTrue(np.all(repair[mask] == 255))
        self.assertTrue(np.all(repair[~mask] == 0))


class PointHitsInstanceTests(unittest.TestCase):
    def _make_instance(self, mask: np.ndarray, track_id: int = 1) -> TrackedInstance:
        h, w = mask.shape[:2]
        return TrackedInstance(
            track_id=track_id,
            bbox=(0.0, 0.0, float(w), float(h)),
            mask=mask,
        )

    def test_hit_on_mask_pixel(self) -> None:
        mask = np.zeros((5, 5), dtype=bool)
        mask[2, 3] = True
        inst = self._make_instance(mask)
        self.assertTrue(point_hits_instance(3, 2, inst))

    def test_miss_on_empty_mask_pixel(self) -> None:
        mask = np.zeros((5, 5), dtype=bool)
        inst = self._make_instance(mask)
        self.assertFalse(point_hits_instance(2, 2, inst))

    def test_fallback_bbox_when_outside_mask_bounds(self) -> None:
        # mask covers only 3x3, but bbox covers 10x10
        mask = np.zeros((3, 3), dtype=bool)
        inst = TrackedInstance(track_id=1, bbox=(0.0, 0.0, 10.0, 10.0), mask=mask)
        # pixel (8, 8) is outside the 3x3 mask → falls back to bbox check
        self.assertTrue(point_hits_instance(8, 8, inst))

    def test_miss_outside_bbox(self) -> None:
        mask = np.zeros((3, 3), dtype=bool)
        inst = TrackedInstance(track_id=1, bbox=(0.0, 0.0, 10.0, 10.0), mask=mask)
        # pixel far outside bbox
        self.assertFalse(point_hits_instance(50, 50, inst))

    def test_click_toggles_erase_set(self) -> None:
        """Simulate the _on_mouse toggle logic used in StreamRemoverApp."""
        mask = np.ones((5, 5), dtype=bool)
        inst = TrackedInstance(track_id=42, bbox=(0.0, 0.0, 5.0, 5.0), mask=mask)
        instances = [inst]
        erasing: set = set()

        # First click: add to erase set
        for i in instances:
            if point_hits_instance(2, 2, i):
                if i.track_id in erasing:
                    erasing.discard(i.track_id)
                else:
                    erasing.add(i.track_id)
                break
        self.assertIn(42, erasing)

        # Second click on same target: remove from erase set
        for i in instances:
            if point_hits_instance(2, 2, i):
                if i.track_id in erasing:
                    erasing.discard(i.track_id)
                else:
                    erasing.add(i.track_id)
                break
        self.assertNotIn(42, erasing)


if __name__ == "__main__":
    unittest.main()
