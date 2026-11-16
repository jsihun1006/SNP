"""Synthetic convergence, parallel translation, expansion and mask diagnostics."""

import unittest

import numpy as np

from model.영상추출.모델.tracking_alignment.team_motion_metrics import MotionConfig, filtered_velocities, measure_team, hull_area_rate


class TeamMotionMetricTests(unittest.TestCase):
    def circle(self):
        angle = np.arange(8)*2*np.pi/8
        centre = np.array([50.,34.])
        points = centre+10*np.column_stack((np.cos(angle),np.sin(angle)))
        return centre,points

    def test_inward_motion_recovers_focus_while_outward_motion_has_forward_rejection(self):
        centre,p = self.circle()
        inward=(centre-p)/5
        result=measure_team(np.stack([p,p]),np.stack([inward,-inward]),np.stack([centre,centre]),(105.3,68))
        np.testing.assert_allclose([result['focus_x_m'][0],result['focus_y_m'][0]],centre,atol=1e-8)
        self.assertEqual(result['focus_status'].tolist(),['ok','low_forward_support'])
        self.assertGreater(result['convergence_score'][0],.99)
        self.assertGreater(result['focus_confidence'][0],.99)
        self.assertLess(result['direction_coherence'][0],1e-8)
        self.assertLess(result['pair_distance_rate_mps'][0],0)
        self.assertGreater(result['pair_distance_rate_mps'][1],0)
        self.assertLess(result['hull_area_rate_m2ps'][0],0)
        self.assertGreater(result['hull_area_rate_m2ps'][1],0)
        self.assertTrue(np.isnan(result['focus_x_m'][1]))
        self.assertAlmostEqual(result['focus_to_ball_distance_m'][0],0)
        self.assertAlmostEqual(result['focus_median_line_eta_s'][0],5)

    def test_parallel_team_translation_is_not_convergence_or_compression(self):
        centre,p=self.circle()
        velocity=np.tile([1.,.5],(1,len(p),1))
        result=measure_team(p[None],velocity,centre[None],(105.3,68))
        self.assertEqual(result['focus_status'][0],'parallel_or_unstable')
        self.assertTrue(np.isnan(result['focus_x_m'][0]))
        self.assertTrue(np.isnan(result['convergence_score'][0]))
        self.assertAlmostEqual(result['direction_coherence'][0],1)
        self.assertAlmostEqual(result['mean_vx_mps'][0],1)
        self.assertAlmostEqual(result['mean_vy_mps'][0],.5)
        self.assertAlmostEqual(result['pair_distance_rate_mps'][0],0)
        self.assertAlmostEqual(result['hull_area_rate_m2ps'][0],0,places=8)

    def test_robust_focus_tolerates_one_distant_inconsistent_ray(self):
        centre,p=self.circle()
        v=(centre-p)/5
        p=np.vstack([p,[90,55]])
        v=np.vstack([v,[0,2]])
        result=measure_team(p[None],v[None],centre[None],(105.3,68))
        self.assertEqual(result['focus_status'][0],'ok')
        estimate=np.array([result['focus_x_m'][0],result['focus_y_m'][0]])
        self.assertLess(np.linalg.norm(estimate-centre),3)
        self.assertGreaterEqual(result['focus_support_count'][0],8)
        self.assertLess(result['focus_confidence'][0],1)

    def test_trailing_velocity_resets_at_x_half_missingness_and_large_step(self):
        positions=np.full((30,3,2),.3,dtype=np.float32)
        positions[:,0,0]=.2+np.arange(30)*.0002
        positions[10,0,0]+=.2
        active=np.ones(30,dtype=bool);active[15]=False
        half=np.r_[np.ones(16),np.full(14,2)]
        eligible=np.zeros(30,dtype=bool)
        eligible[1:]=active[1:]&active[:-1]&(half[1:]==half[:-1])
        result,rejected=filtered_velocities(positions,{'eligible':eligible},25,(105.3,68))
        self.assertTrue(rejected[10,0] and rejected[11,0])
        self.assertTrue(np.isnan(result[:10]).all())
        self.assertTrue(np.isnan(result[10:26,0]).all())
        np.testing.assert_allclose(result[26:,0,0],.5265,atol=1e-4)
        np.testing.assert_equal(result[10,1],[0,0])
        self.assertTrue(np.isnan(result[16:26]).all())

    def test_analytic_hull_rate_preserves_translation_and_detects_shrinking(self):
        p=np.array([[10.,10.],[14.,10.],[14.,12.],[10.,12.]])
        area,rate=hull_area_rate(p,-(p-p.mean(axis=0))*.1)
        self.assertAlmostEqual(area,8)
        self.assertAlmostEqual(rate,-1.6)
        self.assertAlmostEqual(hull_area_rate(p,np.tile([2.,1.],(4,1)))[1],0)
        area,rate=hull_area_rate(np.vstack([p,p[0]]),np.zeros((5,2)))
        self.assertAlmostEqual(area,8)
        self.assertTrue(np.isnan(rate))

    def test_no_motion_and_missing_motion_are_distinct(self):
        centre,p=self.circle()
        velocities=np.stack([np.zeros_like(p),np.full_like(p,np.nan)])
        result=measure_team(np.stack([p,p]),velocities,np.stack([centre,centre]),(105.3,68))
        self.assertEqual(result['n_valid_vectors'].tolist(),[8,0])
        self.assertEqual(result['n_stationary'].tolist(),[8,0])
        self.assertAlmostEqual(result['mean_vx_mps'][0],0)
        self.assertAlmostEqual(result['pair_distance_rate_mps'][0],0)
        self.assertTrue(np.isnan(result['mean_vx_mps'][1]))
        self.assertTrue(np.isnan(result['pair_distance_rate_mps'][1]))
        self.assertTrue(np.isnan(result['convergence_score']).all())


if __name__=='__main__':
    unittest.main()
