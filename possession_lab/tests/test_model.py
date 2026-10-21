"""Tests for emissions, Viterbi decoding, and confidence."""

from __future__ import annotations

import unittest
from types import SimpleNamespace

import numpy as np

from possession_lab.model import (
    DEFAULT_PARAMS,
    ModelParams,
    decode,
    emission_log_probs,
    forward_backward,
    merge_isolated_runs,
    viterbi,
)


def make_feature_namespace() -> SimpleNamespace:
    n = 7
    d = {
        "present": np.array([1, 1, 0, 0, 0, 1, 1], dtype=bool),
        "d_home": np.array([0.5, 0.5, np.nan, np.nan, np.nan, 0.5, 10.0]),
        "d_away": np.array([10.0, 10.0, np.nan, np.nan, np.nan, 10.0, 10.0]),
        "near_distance": np.array([0.5, 0.5, np.nan, np.nan, np.nan, 0.5, 10.0]),
        "near_team": np.array([0, 0, -1, -1, -1, 0, -1], dtype=np.int8),
        "second_distance": np.array([10.0] * n),
        "near_speed": np.ones(n),
        "team_speed": np.full(n, 2.0),
        "ball_speed": np.zeros(n),
        "ball_acceleration": np.zeros(n),
        "gap": np.array([0, 0, 1, 2, 3, 0, 0], dtype=np.int32),
        "edge": np.array([10.0, 10.0, np.nan, np.nan, np.nan, 10.0, np.nan]),
        "last_edge": np.array([10.0] * n),
        "out_of_play": np.array([0, 0, 0, 0, 0, 0, 1], dtype=bool),
        "last_touch": np.array([0, 0, 0, 0, 0, 0, 0], dtype=np.int8),
        "last_touch_age": np.array([0, 0, 1, 2, 3, 0, 0], dtype=np.int32),
        "d_home_last": np.array([0.5, 0.5, 0.5, 0.5, 0.5, 0.5, 10.0]),
        "d_away_last": np.array([10.0] * n),
        "tag_vote": np.array([0, 0, 0, 0, 1, 0, 0], dtype=np.int16),
        "tag_setpiece": np.zeros(n, dtype=np.int8),
        "tag_coverage": np.zeros(n, dtype=bool),
        "player_count": np.full(n, 20, dtype=np.int16),
    }
    return SimpleNamespace(**d)


class ModelTest(unittest.TestCase):
    def test_short_gaps_are_uniform_and_long_gaps_use_proximity(self):
        f = make_feature_namespace()
        params = ModelParams(
            gap_thresh=2,
            sigma_miss=4.0,
            p_stay=0.9,
            p_ha=0.05,
            p_hd=0.05,
            p_dh=0.05,
        )
        logp = emission_log_probs(f, params)

        np.testing.assert_allclose(logp[2], np.log(1.0 / 3.0), atol=1e-12)
        np.testing.assert_allclose(logp[3], np.log(1.0 / 3.0), atol=1e-12)
        self.assertGreater(logp[4, 0], logp[4, 1])
        self.assertGreater(logp[4, 0], logp[4, 2])

    def test_sparse_tracking_does_not_use_empty_team_shape(self):
        f = make_feature_namespace()
        f.player_count[4] = 2
        params = ModelParams(min_players=6, gap_thresh=2)
        logp = emission_log_probs(f, params)
        np.testing.assert_allclose(logp[4], np.log(1.0 / 3.0), atol=1e-12)

    def test_out_of_play_forces_dead_ball_emission(self):
        f = make_feature_namespace()
        logp = emission_log_probs(f, ModelParams())
        self.assertGreater(logp[6, 2], logp[6, 0])
        self.assertGreater(logp[6, 2], logp[6, 1])

    def test_viterbi_and_posterior_are_consistent(self):
        f = make_feature_namespace()
        params = ModelParams(
            gap_thresh=2,
            p_stay=0.9,
            p_ha=0.05,
            p_hd=0.05,
            p_dh=0.05,
        )
        logp = emission_log_probs(f, params)
        codes = viterbi(logp, params)
        posterior, confidence, entropy = forward_backward(logp, params)

        self.assertEqual(codes.shape, (7,))
        self.assertEqual(posterior.shape, (7, 3))
        np.testing.assert_allclose(posterior.sum(axis=1), 1.0, atol=1e-10)
        self.assertTrue(np.all(confidence >= 1.0 / 3.0 - 1e-12))
        self.assertTrue(np.all(confidence <= 1.0 + 1e-12))
        self.assertTrue(np.all(entropy >= -1e-12))
        self.assertEqual(int(codes[0]), 0)
        self.assertEqual(int(codes[6]), 2)

        result = decode(f, params)
        np.testing.assert_array_equal(result.codes, codes)
        np.testing.assert_allclose(result.posterior, posterior)

    def test_slow_players_support_the_dead_ball_state(self):
        f = make_feature_namespace()
        params = ModelParams(w_motion=1.0, w_dead=0.0)
        slow = SimpleNamespace(**{**f.__dict__, "team_speed": np.zeros(7)})
        fast = SimpleNamespace(**{**f.__dict__, "team_speed": np.full(7, 4.0)})

        slow_logp = emission_log_probs(slow, params)
        fast_logp = emission_log_probs(fast, params)
        self.assertGreater(slow_logp[0, 2], fast_logp[0, 2])

    def test_brief_flip_is_absorbed(self):
        codes = np.array([0] * 10 + [2] * 5 + [0] * 10 + [1] * 40, dtype=np.int8)
        merged = merge_isolated_runs(codes, min_frames=25)
        np.testing.assert_array_equal(merged, np.array([0] * 25 + [1] * 40, dtype=np.int8))

    def test_merge_keeps_real_runs_and_never_touches_x(self):
        codes = np.array(
            [3] * 3 + [0] * 10 + [2] * 30 + [0] * 10 + [1] * 10 + [3] * 3, dtype=np.int8
        )
        np.testing.assert_array_equal(merge_isolated_runs(codes, min_frames=25), codes)

    def test_loose_slow_ball_keeps_the_last_touch_team(self):
        f = make_feature_namespace()
        # ball far from both teams: no player within 7 m
        f.d_home[:] = 9.0
        f.d_away[:] = 11.0
        f.ball_speed[:] = 0.5
        f.last_touch[:] = 1
        f.last_touch_age[:] = 0
        logp = emission_log_probs(f, ModelParams())
        self.assertGreater(logp[0, 1], logp[0, 0])
        self.assertGreater(logp[0, 1], logp[0, 2])

    def test_loose_ball_without_a_touch_is_uniform(self):
        f = make_feature_namespace()
        f.d_home[:] = 9.0
        f.d_away[:] = 11.0
        f.ball_speed[:] = 0.5
        f.last_touch[:] = -1
        logp = emission_log_probs(f, ModelParams())
        np.testing.assert_allclose(logp[0], np.log(1.0 / 3.0), atol=1e-12)

    def test_default_parameters_are_the_documented_reference(self):
        self.assertEqual(DEFAULT_PARAMS.sigma, 3.5)
        self.assertEqual(DEFAULT_PARAMS.d_dead, 7.0)
        self.assertEqual(DEFAULT_PARAMS.miss_mode, "hybrid")
        self.assertEqual(DEFAULT_PARAMS.gap_thresh, 120)
        self.assertAlmostEqual(DEFAULT_PARAMS.p_stay, 0.995)
        self.assertAlmostEqual(DEFAULT_PARAMS.p_ha, 0.0005)
        self.assertAlmostEqual(DEFAULT_PARAMS.p_hd, 0.001)
        self.assertAlmostEqual(DEFAULT_PARAMS.p_dh, 0.01)
        self.assertAlmostEqual(DEFAULT_PARAMS.w_edge, 0.5)
        self.assertAlmostEqual(DEFAULT_PARAMS.w_motion, 1.0)
        self.assertAlmostEqual(DEFAULT_PARAMS.w_dead, 0.02)


if __name__ == "__main__":
    unittest.main()
