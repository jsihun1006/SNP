"""Masked ball correlation and joint sequence matching with synthetic data only."""

import unittest
import numpy as np

from starter.find_joint_tracking_match import ball_curve, correlate_valid, find_joint


class JointTrackingMatchTests(unittest.TestCase):
    def test_fft_correlation_agrees_with_direct_sliding_dot_product(self):
        random = np.random.default_rng(19)
        signal, kernel = random.normal(size=37), random.normal(size=7)
        expected = np.array([np.dot(signal[index:index+7], kernel) for index in range(31)])
        np.testing.assert_allclose(correlate_valid(signal, kernel), expected, atol=1e-12)

    def test_masked_rmse_counts_only_jointly_observed_ball_positions(self):
        reference = np.array([[0, 0], [1, 1], [np.nan, np.nan], [3, 3], [4, 4]], dtype=float)
        broadcast = np.vstack([[[9, 9], [8, 8], [7, 7]], reference, [[5, 5], [6, 6]]])
        result = ball_curve(reference, broadcast, minimum_common=2, minimum_fraction=0.25)
        for start in range(len(broadcast)-len(reference)+1):
            sample = broadcast[start:start+len(reference)]
            valid = np.isfinite(reference).all(axis=1) & np.isfinite(sample).all(axis=1)
            self.assertEqual(result['common'][start], int(valid.sum()))
            if valid.sum() >= 2:
                expected = np.sqrt(np.sum((reference[valid]-sample[valid])**2)/valid.sum())
                self.assertAlmostEqual(result['rmse'][start], expected, places=6)
            else:
                self.assertTrue(np.isinf(result['rmse'][start]))
        self.assertAlmostEqual(result['rmse'][3], 0, places=6)

    def test_joint_search_recovers_shift_reflection_and_validates_later_motion(self):
        count = 200
        t = np.arange(count)
        players = np.column_stack([np.r_[np.linspace(.12,.36,6), np.linspace(.64,.88,6)],
                                   np.tile(np.linspace(.15,.85,6),2)])
        full = np.empty((count, 13, 2), dtype=np.float32)
        full[:, :12] = players
        for slot in range(12):
            full[:, slot, 0] += .025*np.sin(t/17+slot*.7)
            full[:, slot, 1] += .02*np.cos(t/19+slot*.4)
        full[:, -1, 0] = .5 + .25*np.sin(t/11) + .1*np.sin(t/23)
        full[:, -1, 1] = .5 + .22*np.cos(t/13) + .1*np.sin(t/7)
        reference = full[50:110].copy()
        order = [5,4,3,2,1,0,11,10,9,8,7,6,12]
        broadcast = full[20:180, order].copy()
        broadcast[:, :, 0] = 1-broadcast[:, :, 0]
        broadcast[35:39, -1] = np.nan
        teams = ['H']*6+['A']*6
        result = find_joint(reference, broadcast, teams, teams, window=30, fps=2,
                            previous_frame=0, events_per_flip=3)
        best = result['best_candidate']
        self.assertEqual(best['broadcast_atd_frame0'], 30)
        self.assertEqual(best['xy_reflection_index'], 1)
        self.assertEqual(best['ball_common_frames'], 26)
        self.assertLess(best['ball_rmse_m'], 1e-4)
        self.assertLess(best['player_score_m'], 1e-4)
        self.assertTrue(best['validation_ball_consistent'])
        self.assertGreater(result['previous_candidate_same_metric']['joint_score_m'], best['joint_score_m'])

    def test_missing_ball_is_not_replaced_with_a_fictitious_origin(self):
        reference = np.full((60, 13, 2), np.nan, dtype=np.float32)
        broadcast = np.full((100, 13, 2), .5, dtype=np.float32)
        teams = ['H']*6+['A']*6
        with self.assertRaises(ValueError):
            find_joint(reference, broadcast, teams, teams, window=30, fps=2)


if __name__ == '__main__':
    unittest.main()
