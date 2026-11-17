"""Synthetic checks of the constrained decoder (no match data needed).

    python model/phase2_possession/model1_dead_ball/test_model.py
"""
from pathlib import Path
import sys
import unittest

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))
import possession as ps  # noqa: E402


class TestConstraint(unittest.TestCase):
    def test_dead_frames_are_d_and_alive_frames_never_d(self):
        n = 400
        costs = np.zeros((n, 3))
        costs[:, ps.A] = 0.08
        graded = np.ones(n, dtype=bool)
        dead = np.zeros(n, dtype=bool)
        dead[100:200] = True
        labels = ps.viterbi(ps.constrained_costs(costs, graded, dead), 1.2, 0.8)
        self.assertTrue((labels[100:200] == ps.D).all())
        self.assertFalse((labels[np.r_[0:100, 200:n]] == ps.D).any())

    def test_ungraded_frames_are_unconstrained(self):
        n = 50
        costs = np.zeros((n, 3))
        costs[:, ps.H] = costs[:, ps.A] = 0.1
        labels = ps.viterbi(ps.constrained_costs(costs, np.zeros(n, bool), np.zeros(n, bool)), 1.2, 0.8)
        self.assertTrue((labels == ps.D).all())

    def test_ungraded_frames_are_written_as_x(self):
        graded = np.r_[np.zeros(5, bool), np.ones(10, bool), np.zeros(5, bool)]
        labels = np.r_[np.full(5, ps.H), np.full(10, ps.A), np.full(5, ps.D)]
        final = ps.final_labels(labels, graded)
        self.assertTrue((final[:5] == "X").all() and (final[5:15] == "A").all() and (final[15:] == "X").all())

    def test_long_missing_prefers_dead_without_constraint(self):
        obs = np.r_[np.zeros(50), -np.ones(200), np.zeros(50)].astype(np.int8)
        cfg = {"fps": ps.FPS, "long_missing_seconds": 3.0, "opposite_team_cost_per_second": 2.0,
               "dead_when_observed_cost_per_second": 2.0, "dead_in_short_gap_cost_per_second": 1.0,
               "alive_in_long_gap_cost_per_second": 0.5, "team_switch_cost": 1.2, "dead_switch_cost": 0.8}
        labels = ps.viterbi(ps.emission_costs(obs, cfg), 1.2, 0.8)
        self.assertTrue((labels[60:240] == ps.D).all())


if __name__ == "__main__":
    unittest.main()
