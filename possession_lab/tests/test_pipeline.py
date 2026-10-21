"""Small end-to-end smoke test for inference and submission output."""

from __future__ import annotations

import tempfile
import unittest
from pathlib import Path

import numpy as np
import pandas as pd

from possession_lab.config import A, H, X
from possession_lab.data import TagTable, TrackingData
from possession_lab.pipeline import predict_full, validate_codes, write_submission


def make_tracking() -> TrackingData:
    n = 30
    players = np.zeros((n, 2, 2), dtype=np.float32)
    players[:, 0] = [10.0, 34.0]
    players[:, 1] = [90.0, 34.0]
    ball = np.zeros((n, 2), dtype=np.float32)
    ball[:, 0] = 10.0
    ball[:, 1] = 34.0
    ball[20:] = [90.0, 34.0]
    return TrackingData(
        ball=ball,
        ball_present=np.ones(n, dtype=bool),
        players=players,
        speeds=np.ones((n, 2), dtype=np.float32),
        player_present=np.ones((n, 2), dtype=bool),
        teams=np.array([0, 1], dtype=np.int8),
        pitch=(105.3, 68.0),
        n_frames=n,
    )


class PipelineTest(unittest.TestCase):
    def test_predict_full_marks_ungraded_frames_x(self):
        tracking = make_tracking()
        tags = TagTable.empty(tracking.n_frames)
        pred = predict_full(
            tracking,
            tags,
            halves=[(0, 10), (20, 30)],
            context=3,
        )
        self.assertEqual(pred.shape, (30,))
        np.testing.assert_array_equal(pred[:10], np.full(10, H, dtype=np.int8))
        self.assertIn(int(pred[20]), (H, A))
        np.testing.assert_array_equal(pred[21:], np.full(9, A, dtype=np.int8))
        np.testing.assert_array_equal(pred[10:20], np.full(10, X, dtype=np.int8))

    def test_submission_writer_and_preflight_validator(self):
        pred = np.full(30, X, dtype=np.int8)
        pred[:10] = H
        pred[20:] = A
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "predictions.csv"
            write_submission(pred, path)
            table = pd.read_csv(path)
            self.assertEqual(list(table.columns), ["frame", "label"])
            self.assertEqual(len(table), 30)
            self.assertEqual(table["frame"].iloc[-1], 29)
            report = validate_codes(pred)
        self.assertTrue(report["ok"])
        self.assertEqual(report["counts"]["H"], 10)
        self.assertEqual(report["counts"]["A"], 10)
        self.assertEqual(report["counts"]["X"], 10)


if __name__ == "__main__":
    unittest.main()
