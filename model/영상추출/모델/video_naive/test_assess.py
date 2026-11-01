import unittest
import tempfile
from pathlib import Path
import numpy as np
import pandas as pd
from assess import load_atd_selected, greedy_matches, select_detections


class AssessmentTests(unittest.TestCase):
    def test_atd_one_based_mapping_and_missing(self):
        with tempfile.TemporaryDirectory() as folder:
            path = Path(folder)/"sample.txt"
            path.write_text("1:0.1,0.2,0;NaN,NaN,NaN:0.5,0.5,0\n2:0.2,0.3,0:NaN,NaN,NaN\n")
            frames, meta = load_atd_selected(path, [0,1,3])
            self.assertEqual(set(frames), {0,1})
            np.testing.assert_allclose(frames[0]['players'], [[10.53,13.6]])
            self.assertTrue(np.isnan(frames[1]['ball']).all())
            self.assertEqual(meta['missing_requested_frames'], [3])

    def test_person_pairs_do_not_reuse_reference(self):
        pairs = greedy_matches([[0,0],[.1,0]], [[0,0]], radius=3)
        self.assertEqual(len(pairs), 1)
        self.assertEqual(pairs[0][:2], (0,0))

    def test_ball_selection_uses_confidence_not_reference(self):
        rows = [dict(frame=5,class_id=32,confidence=.3,x_m=1,y_m=1,projection_valid=True,in_pitch=True),
                dict(frame=5,class_id=32,confidence=.9,x_m=90,y_m=60,projection_valid=True,in_pitch=True)]
        chosen = select_detections(pd.DataFrame(rows))[5]
        np.testing.assert_allclose(chosen['ball'], [90,60])

    def test_empty_detections(self):
        self.assertEqual(select_detections(pd.DataFrame()), {})
        self.assertEqual(greedy_matches([], [[0,0]]), [])


if __name__ == '__main__':
    unittest.main()
