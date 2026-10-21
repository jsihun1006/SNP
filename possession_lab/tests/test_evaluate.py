"""Tests for scoring and stress-mask helpers."""

from __future__ import annotations

import unittest

import numpy as np

from possession_lab.config import A, D, H, N_FRAMES, PUBLIC_START, PUBLIC_STOP, X
from possession_lab.evaluate import (
    evaluate_codes,
    full_match_stats,
    public_metrics,
    smooth_baseline,
    stress_masks,
)


class EvaluationTest(unittest.TestCase):
    def test_official_metrics_are_returned(self):
        gt = np.array([H] * 4 + [A] * 4 + [D] * 4 + [X] * 4, dtype=np.int8)
        pred = np.array([H] * 4 + [A] * 3 + [D] + [D] * 4 + [X] * 4, dtype=np.int8)
        result = evaluate_codes(gt, pred, bootstrap_n=0)
        self.assertIn("spell_f1_mean", result)
        self.assertIn("macro_f1", result)
        self.assertIn("per_class", result)
        self.assertEqual(np.array(result["confusion"]).shape, (4, 4))
        self.assertEqual(result["n_gt_spells"], 2)

    def test_public_metrics_ignores_frames_outside_the_public_window(self):
        gt = np.full(N_FRAMES, H, dtype=np.int8)
        pred = np.full(N_FRAMES, A, dtype=np.int8)
        mid = (PUBLIC_START + PUBLIC_STOP) // 2
        gt[PUBLIC_START:mid] = H
        gt[mid:PUBLIC_STOP] = A
        pred[PUBLIC_START:mid] = H
        pred[mid:PUBLIC_STOP] = A
        result = public_metrics(pred, gt, bootstrap_n=0)
        confusion = np.asarray(result["confusion"])
        self.assertEqual(confusion.shape, (4, 4))
        self.assertEqual(int(confusion[3].sum()), 0)
        self.assertEqual(result["n_frames"], PUBLIC_STOP - PUBLIC_START)
        self.assertEqual(result["n_pred_spells"], 2)

    def test_full_match_structure_respects_half_boundaries(self):
        pred = np.full(30, X, dtype=np.int8)
        pred[0:8] = H
        pred[8:10] = D
        pred[20:27] = A
        pred[27:30] = D
        stats = full_match_stats(pred, halves=[(0, 10), (20, 30)])
        self.assertEqual(stats["n_spells"], 2)
        self.assertAlmostEqual(stats["shares"]["H"], 8.0 / 20.0)
        self.assertAlmostEqual(stats["shares"]["A"], 7.0 / 20.0)
        self.assertAlmostEqual(stats["shares"]["D"], 5.0 / 20.0)
        self.assertGreater(stats["mean_duration_s"], 0)

    def test_stress_masks_only_remove_frames_inside_public_window(self):
        masks = stress_masks(n_frames=30, public=(10, 20), seed=3)
        self.assertTrue(masks["clean"].all())
        for name in ("random", "long"):
            self.assertTrue(masks[name][:10].all())
            self.assertTrue(masks[name][20:].all())
            self.assertGreaterEqual(int((~masks[name][10:20]).sum()), 2)
        self.assertGreaterEqual(int((~masks["long"][10:20]).sum()), 4)

    def test_smooth_baseline_returns_only_home_away_or_dead(self):
        codes = np.array([H, H, A, A, D, H, A], dtype=np.int8)
        out = smooth_baseline(codes, window=75)
        self.assertTrue(set(np.unique(out)).issubset({H, A, D}))


if __name__ == "__main__":
    unittest.main()
