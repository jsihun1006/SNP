import contextlib
from types import SimpleNamespace
import unittest

import numpy as np

from ball_only.detector import BallDetector, candidate_from_box, merge_candidates, tile_windows


class _Data:
    def __init__(self, rows):
        self.rows = np.asarray(rows, dtype=float).reshape(-1, 6)

    def float(self):
        return self

    def cpu(self):
        return self

    def numpy(self):
        return self.rows


class _Model:
    def __init__(self):
        self.calls = []
        self.predictor = SimpleNamespace(model=SimpleNamespace(fp16=False))

    def predict(self, images, **kwargs):
        self.calls.append(dict(images=len(images), **kwargs))
        results = []
        for image in images:
            frame_id = int(image[0, 0, 0])
            if kwargs['imgsz'] == 1536:
                boxes = [] if frame_id == 1 else [[700, 400, 720, 420, .8, 32]]
            else:
                boxes = [[100, 120, 110, 130, .7, 32]]
            results.append(SimpleNamespace(boxes=SimpleNamespace(data=_Data(boxes))))
        return results


def fake_detector():
    value = object.__new__(BallDetector)
    value.full_imgsz, value.roi_imgsz = 1536, 960
    value.tile_size, value.tile_overlap = 640, .20
    value.confidence, value.device = .10, 0
    value.torch = SimpleNamespace(cuda=SimpleNamespace(OutOfMemoryError=type('CudaMemoryError', (RuntimeError,), {})))
    value.controller = SimpleNamespace(gpu_work=contextlib.nullcontext)
    value.model = _Model()
    return value


class DetectorTests(unittest.TestCase):
    def test_tiles_cover_all_original_pixels_and_edge_strips(self):
        coverage = np.zeros((1080, 1920), dtype=np.uint8)
        windows = tile_windows(1920, 1080)
        for x0, y0, x1, y1 in windows:
            coverage[y0:y1, x0:x1] += 1
        self.assertEqual(len(windows), 8)
        self.assertTrue(np.all(coverage > 0))

    def test_original_coordinates_remain_actual_observed_centers(self):
        value = candidate_from_box([100, 120, 110, 130, .7, 32], (512, 440, 1152, 1080), (1920, 1080), 'tile')
        self.assertEqual(value['bbox_xyxy'], [612, 560, 622, 570])
        self.assertEqual(value['point'], [617, 565])
        duplicate = dict(value, source='YOLO26x_ROI', confidence=.6)
        merged = merge_candidates([value, duplicate])
        self.assertEqual(len(merged), 1)
        self.assertEqual(merged[0]['point'], value['point'])
        self.assertEqual(merged[0]['confidence'], .7)
        self.assertEqual(merged[0]['independent_detector_families'], 1)

    def test_phases_explicitly_split_actual_lists_and_every_frame_gets_all_modes(self):
        detector = fake_detector()
        frames = [np.full((1080, 1920, 3), value, dtype=np.uint8) for value in range(3)]
        result = detector.run_chunk(frames, full_batch=2, roi_batch=4, tile_batch=8)
        calls = detector.model.calls
        self.assertEqual([(c['imgsz'], c['images']) for c in calls],
                         [(1536, 2), (1536, 1), (960, 3), (640, 8), (640, 8), (640, 8)])
        for call in calls:
            self.assertEqual(call['batch'], call['images'])
            self.assertEqual(call['quantize'], 32)
            self.assertEqual(call['classes'], [32])
            self.assertFalse(call['tile'])
        self.assertEqual(len(result), 3)
        self.assertEqual(result[1]['roi_centers'], [[960., 540.]])
        for row in result:
            self.assertEqual(len(row['tile']), 8)
            self.assertGreaterEqual(len(row['roi']), 1)
            self.assertTrue(all(c['independent_detector_families'] == 1 for c in row['candidates']))
        self.assertEqual(result[0]['roi_centers'], [[710., 410.]])
        self.assertEqual(result[0]['roi'][0]['point'], [495., 215.])

    def test_empty_and_invalid_chunks(self):
        detector = fake_detector()
        self.assertEqual(detector.run_chunk([]), [])
        with self.assertRaises(ValueError):
            detector.run_chunk([np.zeros((3, 3, 3), dtype=np.float32)])
        with self.assertRaises(ValueError):
            detector.run_chunk([np.zeros((3, 3, 3), dtype=np.uint8)], tile_batch=0)
        with self.assertRaises(ValueError):
            detector.run_chunk([np.zeros((3, 3, 3), dtype=np.uint8), np.zeros((4, 3, 3), dtype=np.uint8)])


if __name__ == '__main__':
    unittest.main()
