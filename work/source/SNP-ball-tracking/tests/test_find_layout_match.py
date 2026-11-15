"""Synthetic cross-camera set matching, assignment and timeline-search tests only."""

from itertools import permutations
from pathlib import Path
import tempfile
import unittest

import numpy as np

from starter.find_layout_match import assignment_costs, find_matches, layout_error, read_positions


class LayoutMatchTests(unittest.TestCase):
    def test_rectangular_assignment_matches_brute_force(self):
        random = np.random.default_rng(35)
        for rows, columns in ((2, 3), (3, 2), (3, 3), (1, 4)):
            cost = random.uniform(0, 10, size=(rows, columns))
            small = cost if rows <= columns else cost.T
            expected = min(sum(small[row, column] for row, column in enumerate(order))
                           for order in permutations(range(small.shape[1]), small.shape[0]))
            self.assertAlmostEqual(float(assignment_costs(cost).sum()), expected)

    def test_team_sets_ignore_slot_order_but_not_duplicate_assignments(self):
        reference = np.array([[10, 10], [20, 20], [80, 40], [90, 50]], dtype=float)
        teams = ["H", "H", "A", "A"]
        candidate = reference[[1, 0, 3, 2]]
        self.assertEqual(layout_error(reference, candidate, teams, teams, minimum=2)["rmse_m"], 0)
        duplicated = reference.copy()
        duplicated[1] = duplicated[0]
        self.assertGreater(layout_error(reference, duplicated, teams, teams, minimum=2)["rmse_m"], 0)
        duplicated[1] = np.nan
        self.assertIsNone(layout_error(reference, duplicated, teams, teams, minimum=2))

    def test_recovers_query_frame_with_permuted_slots_and_reflection(self):
        teams = ["H", "H", "A", "A"]
        base = np.array([[10, 10], [20, 20], [70, 40], [80, 50]], dtype=np.float32)
        tactical = base[None] + np.arange(120)[:, None, None] * np.array([0.25, 0.1])
        broadcast = tactical[10:100, [1, 0, 3, 2]].copy()
        broadcast[:, :, 0] = 105.3 - broadcast[:, :, 0]
        matches = find_matches(tactical[26:35], 4, broadcast, teams, teams,
                               fps=2, minimum=2, events_per_flip=2)
        self.assertTrue(matches)
        self.assertEqual(matches[0]["broadcast_atd_frame0"], 20)
        self.assertEqual(matches[0]["broadcast_atd_frame_id"], 21)
        self.assertEqual(matches[0]["xy_reflection_index"], 1)
        self.assertLess(matches[0]["window_score_m"], 1e-8)

    def test_partial_reader_preserves_frame_slot_scaling_and_checks_full_index(self):
        with tempfile.TemporaryDirectory() as directory:
            raw = Path(directory) / "synthetic.txt"
            raw.write_text("1:0.1,0.2,0:NaN,NaN,NaN\n2:0.5,0.5,0:NaN,NaN,NaN\n"
                           "3:NaN,NaN,NaN:NaN,NaN,NaN\n")
            positions = read_positions(raw, ["H"], 3, start=1, stop=3)
            self.assertEqual(positions.shape, (2, 1, 2))
            np.testing.assert_allclose(positions[0, 0], [52.65, 34], atol=1e-5)
            self.assertTrue(np.isnan(positions[1]).all())
            raw.write_text("2:0.5,0.5,0:NaN,NaN,NaN\n")
            with self.assertRaises(ValueError):
                read_positions(raw, ["H"], 3, start=1, stop=3)


if __name__ == "__main__":
    unittest.main()
