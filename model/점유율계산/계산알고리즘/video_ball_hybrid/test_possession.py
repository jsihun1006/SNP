import unittest

import numpy as np
import pandas as pd

from possession import choose_video_balls, hold_video_balls, nearest_team, MISSING


class BallPossessionTests(unittest.TestCase):
    def test_candidate_selection_ignores_ref_and_filters(self):
        rows = pd.DataFrame([
            [1, 0, .50, .2, .3, 1, 1], [1, 0, .90, .4, .5, 1, 0],
            [1, 0, .70, .6, .7, 1, 1], [2, 1, .99, .5, .5, 1, 1],
            [3, 0, .05, .3, .3, 1, 1], [4, 0, .80, 1.2, .2, 1, 1],
        ], columns=["frame", "class_id", "confidence", "x_norm", "y_norm", "projection_valid", "in_pitch"])
        selected = choose_video_balls(rows, 0, n_frames=5)
        self.assertEqual(selected.frame.tolist(), [1])
        self.assertEqual(selected.x_norm.tolist(), [.6])

    def test_hold_has_no_backfill_or_unbounded_carry(self):
        selected = pd.DataFrame({"frame": [2], "confidence": [.7], "x_norm": [.3], "y_norm": [.5]})
        held, _, age = hold_video_balls(selected, n_frames=7, hold_frames=2)
        self.assertEqual(age.tolist(), [-1, -1, 0, 1, 2, -1, -1])
        self.assertTrue(np.isnan(held[1]).all())
        self.assertTrue(np.isnan(held[5]).all())

    def test_nearest_team_uses_physical_pitch_scaling(self):
        # Equal normalized coordinate gap is not equal metres on a non-square pitch.
        ball = np.array([.5, .5])
        groups = ["0.51,0.5,0", "0.5,0.513,0"]
        team, distance, slot = nearest_team(ball, groups, [0, 1])
        self.assertEqual((team, slot), (1, 1))
        self.assertAlmostEqual(distance, .013 * 68)
        self.assertEqual(nearest_team(None, groups, [0, 1])[0], MISSING)
        self.assertEqual(nearest_team(ball, ["NaN,NaN,NaN"] * 2, [0, 1])[0], MISSING)

    def test_invalid_frame_or_slot_count_fails(self):
        data = pd.DataFrame([[-1, 0, .5, .5, .5, 1, 1]], columns=["frame", "class_id", "confidence", "x_norm", "y_norm", "projection_valid", "in_pitch"])
        with self.assertRaises(ValueError): choose_video_balls(data, 0, n_frames=5)
        with self.assertRaises(ValueError): nearest_team(np.array([.5, .5]), [".5,.5,0"], [0, 1])


if __name__ == "__main__":
    unittest.main()
