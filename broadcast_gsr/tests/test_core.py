import ast
import csv
import json
import os
from pathlib import Path
from types import SimpleNamespace
import tempfile
import unittest
import cv2
import numpy as np
from broadcast_gsr.geometry import Calibration, project, template_to_pitch_h
from broadcast_gsr.io import csv_write, load_config, new_run
from broadcast_gsr.matching import bbox_ious, linear_assignment
from broadcast_gsr.pipeline import Replay
from broadcast_gsr.refinement import split, incompatible

SOURCE=Path(os.environ.get('GSR_UPSTREAM',Path(__file__).resolve().parents[1]/'vendor/SoccernetGSR'))


class GeometryTests(unittest.TestCase):
    def test_four_point_direction_and_mtah_dimensions(self):
        pixels=np.array([[120.,80.],[1000.,100.],[950.,700.],[150.,650.]])
        template=np.array([[10.,5.],[115.,5.],[115.,73.],[10.,73.]])
        h,_=cv2.findHomography(pixels,template)
        pitch_to_image=template_to_pitch_h(h)
        expected=np.array([[0,0],[1,0],[1,1],[0,1.]])
        actual=project(np.linalg.inv(pitch_to_image),pixels)
        np.testing.assert_allclose(actual,expected,atol=1e-9)
        np.testing.assert_allclose(actual*[105.3,68],expected*[105.3,68],atol=1e-8)
        np.testing.assert_allclose(project(pitch_to_image,actual),pixels,atol=1e-8)

    def test_calibration_uses_explicit_frame_key_and_rejects_identity(self):
        with tempfile.TemporaryDirectory() as folder:
            path=Path(folder)/'h.csv'
            cols=['h%d%d'%(i,j) for i in range(3) for j in range(3)]
            matrix=np.array([[100.,0,20],[0,80,10],[0,0,1.]])
            rows=[dict(frame=i,valid=1,**dict(zip(cols,h.ravel()))) for i,h in [(9,matrix),(11,np.eye(3))]]
            csv_write(path,rows,['frame','valid']+cols)
            cal=Calibration(path)
            self.assertIsNone(cal.get(10)[0])
            self.assertEqual(cal.get(11)[2],'identity_homography')
            feet,_,flag=cal.feet(9,[60,10,20,40])
            self.assertFalse(flag)
            np.testing.assert_allclose(feet[2],[.5,.5])

    def test_projection_at_infinity_is_missing(self):
        self.assertTrue(np.isnan(project(np.zeros((3,3)),[[1,2]])).all())

    @unittest.skipUnless((SOURCE/'kpts.py').exists(),'Pinned upstream source not installed')
    def test_fixed_lm_accepts_four_correspondences(self):
        from broadcast_gsr.calibration import estimate_homography
        tree=ast.parse((SOURCE/'kpts.py').read_text())
        node=next(n for n in tree.body if isinstance(n,ast.FunctionDef) and n.name=='get_inv_residual_line_circle_points_middle_pair')
        env={'np':np};exec(compile(ast.Module(body=[node],type_ignores=[]),'<original residual>','exec'),env)
        source=SimpleNamespace(**env)
        a=np.array([[0.,0.,1.],[1.,0.,1.],[1.,1.,1.],[0.,1.,1.]])
        pairs=[(p,np.array([2*p[0]+3,4*p[1]+5,1.]),1.) for p in a]
        h=estimate_homography(source,[],[],np.eye(3),[],pairs)
        np.testing.assert_allclose(project(h,a[:,:2]),np.array([q[:2] for _,q,_ in pairs]),atol=1e-6)


class AssociationTests(unittest.TestCase):
    def test_inclusive_pixel_iou(self):
        self.assertAlmostEqual(bbox_ious([[0,0,9,9]],[[5,0,14,9]])[0,0],1/3)

    def test_gated_lap_and_empty_frames(self):
        matches,a,b=linear_assignment(np.array([[.15,.16],[.16,99.]]),.5)
        self.assertEqual(set(map(tuple,matches)),{(0,1),(1,0)})
        matches,a,b=linear_assignment(np.empty((2,0)),.5)
        self.assertEqual(matches.shape,(0,2));self.assertEqual(a,(0,1));self.assertEqual(b,())

    def test_lap_minimizes_cost_including_unmatched(self):
        rng=np.random.default_rng(5)
        for _ in range(60):
            cost=rng.random((2,2));threshold=.5
            matches,a,b=linear_assignment(cost,threshold)
            score=sum(cost[i,j] for i,j in matches)+threshold/2*(len(a)+len(b))
            possibilities=[threshold*2]
            possibilities += [cost[i,j]+threshold for i in range(2) for j in range(2) if cost[i,j]<=threshold]
            possibilities += [cost[0,j]+cost[1,1-j] for j in (0,1) if cost[0,j]<=threshold and cost[1,1-j]<=threshold]
            self.assertAlmostEqual(score,min(possibilities))

    @unittest.skipUnless((SOURCE/'yolox/EIoU_tracker/Deep_EIoU.py').exists(),'Pinned upstream source not installed')
    def test_actual_tracker_keeps_detection_indices_after_reordering_and_empty_frames(self):
        from broadcast_gsr.upstream import Tracker
        tracker=Tracker(SOURCE,25.)
        first=np.array([[20,20,40,70,.95],[200,20,220,70,.95]])
        features=np.eye(2)
        initial=tracker.update(first,features)
        self.assertEqual([(t,i) for t,i,_ in initial],[(1,0),(2,1)])
        swapped=tracker.update(first[::-1].copy(),features[::-1].copy())
        self.assertEqual([(t,i) for t,i,_ in swapped],[(1,1),(2,0)])
        tracker.update(np.empty((0,5)),np.empty((0,2)))
        self.assertEqual(tracker.impl.frame_id,3)
        recovered=tracker.update(first.copy(),features.copy())
        self.assertEqual({t for t,_,_ in recovered},{1,2})


class ContractTests(unittest.TestCase):
    def test_existing_output_is_preserved(self):
        with tempfile.TemporaryDirectory() as folder:
            out=Path(folder)/'existing';out.mkdir();(out/'keep.txt').write_text('keep')
            with self.assertRaises(FileExistsError):new_run({'output_root':folder},out)
            self.assertEqual((out/'keep.txt').read_text(),'keep')

    def test_output_cannot_contain_source_inputs(self):
        with tempfile.TemporaryDirectory() as folder:
            with self.assertRaises(ValueError):new_run({'output_root':folder,'video':str(Path(folder)/'video.mp4')},folder)

    def test_replay_is_explicit_and_malformed_features_fail(self):
        with tempfile.TemporaryDirectory() as folder:
            p=Path(folder)/'r.jsonl'
            p.write_text(json.dumps(dict(frame_idx=0,detections=[[1,1,4,6,.9]],features=[[0,0]]))+'\n')
            with self.assertRaises(ValueError):Replay(p)

    def test_split_uses_gaps_and_unknown_attributes_abstain(self):
        rows=[dict(frame_idx=i,role='Player',jersey=j,team_raw='') for i,j in [(1,8),(2,8),(30,9),(31,9)]]
        self.assertEqual(len(split(rows,25)),2)
        self.assertFalse(incompatible([dict(role='Player',jersey='',team_raw='')],rows[:1]))

    def test_refinement_preserves_observations_and_sorts_a_merge(self):
        from broadcast_gsr.refinement import refine
        from broadcast_gsr.io import json_write
        fields=['frame_idx','track_id','track_id_raw','camera_segment_id','role','jersey','team_raw',
                'homography_valid','pitch_x_m','pitch_y_m','feature_index','detection_score']
        rows=[]
        for i,(frame,tid) in enumerate([(1,'a'),(2,'a'),(4,'b'),(5,'b')]):
            rows.append(dict(zip(fields,[frame,tid,tid,0,'Player',8,'color:blue',1,frame*.1,3.,i,.82])))
        with tempfile.TemporaryDirectory() as folder:
            out=Path(folder)
            np.array([[1.,0.]]*4,dtype='<f4').tofile(out/'features.float32')
            json_write(out/'features.json',dict(count=4,dimension=2))
            refine(rows,out,{},25,fields)
            with (out/'tracks_refined.csv').open() as handle:actual=list(csv.DictReader(handle))
            self.assertEqual(len(actual),4)
            self.assertEqual(len({r['track_id'] for r in actual}),1)
            self.assertEqual([int(r['frame_idx']) for r in actual],[1,2,4,5])
            self.assertEqual({r['detection_score'] for r in actual},{'0.82'})


class IntegrationTests(unittest.TestCase):
    @unittest.skipUnless((SOURCE/'kpts.py').exists(),'Pinned upstream source not installed')
    def test_synthetic_video_full_pipeline_is_labeled_synthetic(self):
        from broadcast_gsr.smoke import smoke
        with tempfile.TemporaryDirectory() as folder:
            out=smoke(SOURCE,Path(folder)/'smoke')
            manifest=json.loads((out/'manifest.json').read_text())
            self.assertEqual(manifest['status'],'complete')
            self.assertEqual(manifest['processed_frames'],24)
            self.assertEqual(manifest['homography_valid_frames'],23)
            self.assertFalse(manifest['model_inference_executed'])
            with (out/'tracks.csv').open() as handle:rows=list(csv.DictReader(handle))
            self.assertEqual(len(rows),21)
            self.assertTrue(all(float(r['detection_score'])<=1 for r in rows))
            self.assertFalse(any(7<=int(r['frame_idx'])<=9 for r in rows))
            invalid=[r for r in rows if r['frame_idx']=='5'][0]
            self.assertEqual(invalid['homography_valid'],'0');self.assertEqual(invalid['pitch_x_m'],'')
            self.assertEqual({r['camera_segment_id'] for r in rows},{'0','1'})
            with (out/'tracks_refined.csv').open() as handle:refined=list(csv.DictReader(handle))
            self.assertEqual(len(refined),len(rows))
            cap=cv2.VideoCapture(str(out/'overlay.mp4'))
            self.assertEqual(int(cap.get(cv2.CAP_PROP_FRAME_COUNT)),24);cap.release()

    def test_comparison_reports_disagreement_and_unmatched(self):
        from broadcast_gsr.compare import compare
        with tempfile.TemporaryDirectory() as folder:
            folder=Path(folder)
            fields=['frame_idx','homography_valid','pitch_x_m','pitch_y_m','role']
            csv_write(folder/'tracks.csv',[
                dict(zip(fields,[0,1,10.,20.,'Player'])),dict(zip(fields,[0,1,90.,60.,'Player']))],fields)
            (folder/'atd.txt').write_text('1:'+f'{10/105.3},{20/68},0'+':nan,nan,nan\n')
            result=compare(folder/'tracks.csv',folder/'atd.txt',folder/'result')
            self.assertFalse(result['atd_is_ground_truth'])
            self.assertFalse(result['video_to_atd_alignment_verified'])
            self.assertEqual(result['matched'],1);self.assertEqual(result['predicted_unmatched'],1)


if __name__=='__main__':unittest.main()
