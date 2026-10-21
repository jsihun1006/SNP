"""Tests for the geometry and missing-ball features."""

from __future__ import annotations

import unittest

import numpy as np

from possession_lab.data import TagTable, TrackingData
from possession_lab.features import extract_features


def make_tracking() -> TrackingData:
    n = 6
    ball = np.array(
        [
            [50.0, 34.0],
            [55.0, 34.0],
            [np.nan, np.nan],
            [np.nan, np.nan],
            [100.0, 30.0],
            [110.0, 30.0],
        ],
        dtype=np.float32,
    )
    present = np.array([True, True, False, False, True, True])
    players = np.full((n, 2, 2), np.nan, dtype=np.float32)
    players[:, 0] = np.array([50.0, 34.0])
    players[:, 1] = np.array([60.0, 34.0])
    players[1, 0] = [55.0, 34.0]
    players[1, 1] = [70.0, 34.0]
    players[2, 1] = [70.0, 34.0]
    players[3, 1] = [70.0, 34.0]
    players[4, 0] = [100.0, 30.0]
    players[4, 1] = [100.0, 40.0]
    speeds = np.ones((n, 2), dtype=np.float32)
    return TrackingData(
        ball=ball,
        ball_present=present,
        players=players,
        speeds=speeds,
        player_present=np.ones((n, 2), dtype=bool),
        teams=np.array([0, 1], dtype=np.int8),
        pitch=(105.3, 68.0),
        n_frames=n,
    )


class FeatureExtractionTest(unittest.TestCase):
    def test_distances_gap_edge_touch_and_out_of_play(self):
        tracking = make_tracking()
        tags = TagTable.empty(tracking.n_frames)
        f = extract_features(tracking, tags)

        np.testing.assert_array_equal(f.present, tracking.ball_present)
        np.testing.assert_allclose(f.d_home[:2], [0.0, 0.0])
        np.testing.assert_allclose(f.d_away[:2], [10.0, 15.0])
        np.testing.assert_array_equal(f.near_team[:2], [0, 0])
        np.testing.assert_allclose(f.near_distance[:2], [0.0, 0.0])
        np.testing.assert_array_equal(f.gap, [0, 0, 1, 2, 0, 0])
        self.assertAlmostEqual(float(f.edge[1]), 34.0)
        self.assertAlmostEqual(float(f.last_edge[3]), 34.0)
        self.assertAlmostEqual(float(f.edge[4]), 5.3, places=4)
        self.assertTrue(f.out_of_play[5])
        self.assertFalse(f.out_of_play[4])
        self.assertEqual(int(f.last_touch[4]), 0)
        self.assertEqual(int(f.last_touch_age[4]), 0)
        self.assertTrue(np.isfinite(f.d_home_last[2]))
        self.assertAlmostEqual(float(f.d_home_last[2]), 5.0)
        self.assertAlmostEqual(float(f.d_away_last[2]), 15.0)
        np.testing.assert_allclose(f.team_speed, 1.0)

    def test_team_speed_tracks_player_motion(self):
        tracking = make_tracking()
        tracking.speeds[:3] = 0.0
        f = extract_features(tracking, TagTable.empty(tracking.n_frames))

        self.assertLess(float(f.team_speed[0]), 1.0)
        self.assertGreater(float(f.team_speed[-1]), float(f.team_speed[0]))

    def test_observed_mask_creates_a_gap_without_touching_raw_data(self):
        tracking = make_tracking()
        tags = TagTable.empty(tracking.n_frames)
        mask = np.ones(tracking.n_frames, dtype=bool)
        mask[0] = False
        mask[1] = False
        f = extract_features(tracking, tags, observed_mask=mask)

        self.assertTrue(f.present[5])
        np.testing.assert_array_equal(f.gap[:5], [1, 2, 3, 4, 0])
        self.assertTrue(np.isnan(f.edge[0]))
        self.assertTrue(np.isnan(f.last_edge[3]))
        self.assertTrue(f.out_of_play[5])


if __name__ == "__main__":
    unittest.main()
