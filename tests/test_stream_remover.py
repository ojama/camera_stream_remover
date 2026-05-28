import unittest

import numpy as np

from stream_remover import (
    TrackedInstance,
    apply_background_replacement,
    match_selected_targets_to_ids,
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


if __name__ == "__main__":
    unittest.main()
