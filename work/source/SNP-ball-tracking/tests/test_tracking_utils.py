"""Synthetic shared tracking utilities and H.264 build availability; no match data."""

from pathlib import Path
import tempfile
import unittest

import cv2
import numpy as np

from starter.tracking_utils import (pack_source_frames, load_label_mask, source_state,
                                    track_teams, trailing_mean)


class TrackingUtilityTests(unittest.TestCase):
    def positions(self):
        return np.array([
            [[.10,.20],[.50,.50],[.50,.50]],
            [[.11,.22],[.50,.50],[.51,.50]],
            [[np.nan,.22],[.51,.50],[.50,.50]],
            [[.13,.24],[.52,.50],[.50,.50]],
            [[.14,.23],[.52,.50],[.50,.50]],
        ],dtype=np.float32)

    def test_vector_units_sign_missing_zero_and_ball_context(self):
        packed=pack_source_frames(self.positions(),np.array([0,1,3,4,-1]),25,(105.3,68))
        np.testing.assert_allclose(packed[1,0,2:],[26.325,34],atol=1e-4)
        np.testing.assert_equal(packed[1,1,2:],[0,0])
        self.assertTrue(np.isnan(packed[0,:,2:]).all())
        self.assertTrue(np.isnan(packed[2,0,2:]).all())
        np.testing.assert_allclose(packed[3,0,2:],[26.325,-17],atol=1e-4)
        self.assertTrue(np.isnan(packed[-1]).all())
        self.assertTrue(np.isnan(packed[:,-1,2:]).all())

    def test_original_source_adjacency_is_independent_of_display_sampling(self):
        packed=pack_source_frames(self.positions(),np.array([1,4]),25,(105.3,68))
        np.testing.assert_equal(packed[1,1,2:],[0,0])
        np.testing.assert_allclose(packed[1,0,2:],[26.325,-17],atol=1e-4)

    def test_invalid_frame_requests_rejected(self):
        for frames in (np.array([-2]),np.array([5]),np.array([1.5])):
            with self.assertRaises(ValueError):
                pack_source_frames(self.positions(),frames,25,(105.3,68))

    def test_label_mask_keeps_pending_and_d_frames_but_resets_at_x_and_half(self):
        with tempfile.TemporaryDirectory() as folder:
            path=Path(folder)/'labels.csv'
            path.write_text('frame,half_tag,label\n0,,X\n1,1st Half,\n2,1st Half,D\n3,1st Half,X\n4,2nd Half,\n5,2nd Half,H\n')
            active,halves=load_label_mask(path,6)
            self.assertEqual(active.tolist(),[False,True,True,False,True,True])
            self.assertEqual(halves.tolist(),[0,1,1,1,2,2])
            state=source_state(path,6)
            self.assertEqual(state['eligible'].tolist(),[False,False,True,False,False,True])
            self.assertEqual(state['labels'].tolist(),['X','','D','X','','H'])
            with self.assertRaises(ValueError):
                load_label_mask(path,7)

    def test_team_mapping_preserves_source_slot_order_and_rejects_unknown_team(self):
        with tempfile.TemporaryDirectory() as folder:
            path=Path(folder)/'meta.xml'
            path.write_text('<Root><Team id="home"/><Team id="away"/><Player teamId="away"/><Player teamId="home"/></Root>')
            self.assertEqual(track_teams(path),['A','H'])
            path.write_text('<Root><Team id="home"/><Team id="away"/><Player teamId="unknown"/></Root>')
            with self.assertRaises(ValueError):
                track_teams(path)

    def test_display_average_does_not_fill_a_missing_current_value(self):
        np.testing.assert_allclose(trailing_mean([1,3,np.nan,5,7],4),[np.nan,2,np.nan,3,5],equal_nan=True)

    def test_h264_encoder_writes_complete_decodable_video(self):
        with tempfile.TemporaryDirectory() as folder:
            path=Path(folder)/'synthetic_h264.mp4'
            writer=cv2.VideoWriter(str(path),cv2.VideoWriter_fourcc(*'avc1'),25,(320,192))
            try:
                self.assertTrue(writer.isOpened(),'The video workflow requires an OpenCV build with an H.264 (avc1) encoder.')
                for i in range(6):
                    image=np.zeros((192,320,3),dtype=np.uint8)
                    cv2.arrowedLine(image,(50,50),(100+i,80),(220,120,35),2)
                    writer.write(image)
            finally:
                writer.release()
            cap=cv2.VideoCapture(str(path))
            try:
                self.assertEqual(int(cap.get(cv2.CAP_PROP_FRAME_COUNT)),6)
                self.assertAlmostEqual(cap.get(cv2.CAP_PROP_FPS),25)
                cap.set(cv2.CAP_PROP_POS_FRAMES,5)
                ok,image=cap.read()
                self.assertTrue(ok)
                self.assertEqual(image.shape,(192,320,3))
            finally:
                cap.release()


if __name__=='__main__':
    unittest.main()
