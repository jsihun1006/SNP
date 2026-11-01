"""Synthetic tests only: no provided-match GT, no parameter tuning."""
import itertools
import json
import unittest

import numpy as np

from common import HERE
from model import decode


class SequenceDecoderTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.config = json.loads((HERE / "config.json").read_text())

    def test_observation_patterns(self):
        cases = {
            "stable_home": ([0] * 200, [0] * 200),
            "brief_opponent": ([0] * 100 + [1] * 10 + [0] * 100, [0] * 210),
            "sustained_turnover": ([0] * 100 + [1] * 100, [0] * 100 + [1] * 100),
            "short_missing": ([0] * 100 + [-1] * 25 + [0] * 100, [0] * 225),
            "long_missing": ([0] * 100 + [-1] * 150 + [0] * 100, [0] * 100 + [2] * 150 + [0] * 100),
            "all_missing": ([-1] * 150, [2] * 150),
        }
        for name, (observations, expected) in cases.items():
            with self.subTest(name=name):
                labels, *_ = decode(observations, self.config)
                self.assertEqual(labels.tolist(), expected)

    def test_exact_optimum_against_exhaustive_search(self):
        team, dead = self.config["team_switch_cost"], self.config["dead_switch_cost"]
        transitions = np.array([[0, team, dead], [team, 0, dead], [dead, dead, 0]])
        for observations in itertools.product([-1, 0, 1], repeat=4):
            _, costs, _, _, objective = decode(observations, self.config)
            brute = min(sum(costs[t, state] for t, state in enumerate(path)) +
                        sum(transitions[path[t-1], path[t]] for t in range(1, 4))
                        for path in itertools.product(range(3), repeat=4))
            self.assertAlmostEqual(objective, brute, places=10)

    def test_invalid_and_empty_input(self):
        self.assertEqual(len(decode([], self.config)[0]), 0)
        with self.assertRaises(ValueError):
            decode([3], self.config)
        bad_config = dict(self.config, team_switch_cost=-1)
        with self.assertRaises(ValueError):
            decode([0], bad_config)


if __name__ == "__main__":
    unittest.main()
