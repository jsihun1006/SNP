"""Tests for the local synchronized video reviewer."""

from __future__ import annotations

import json
import threading
import unittest
from pathlib import Path
from tempfile import TemporaryDirectory
from types import SimpleNamespace
from urllib.error import HTTPError
from urllib.request import Request, urlopen

import numpy as np

from possession_lab.config import A, D, H, X
from possession_lab.data import TagEvent
from possession_lab.viewer import build_viewer_data, make_server, parse_byte_range


class ViewerDataTest(unittest.TestCase):
    def test_builds_compact_labels_spells_disagreements_and_tags(self):
        pred = np.array([X, H, H, A, D, D, X], dtype=np.int8)
        gt = np.array([X, H, A, A, D, D, X], dtype=np.int8)
        tag = TagEvent(
            start_frame=2,
            stop_frame=4,
            code="ATTACK",
            team=0,
            type="Phase",
            labels={"Half": "1"},
        )
        data = build_viewer_data(
            pred,
            gt,
            tags=SimpleNamespace(events=(tag,)),
            metrics=None,
            fps=25,
            halves=((1, 6),),
            public=(1, 6),
            start_frame=None,
        )
        self.assertEqual(data["labels"], "XHHADDX")
        self.assertEqual(data["gt"], "XHAADDX")
        self.assertEqual(data["pred_spells"], [[1, 3, H], [3, 4, A]])
        self.assertEqual(data["gt_spells"], [[1, 2, H], [2, 4, A]])
        self.assertEqual(data["disagreements"], [2])
        self.assertEqual(data["start_frame"], 1)
        self.assertTrue(data["gt_available"])
        self.assertEqual(len(data["tags"]), 1)
        self.assertEqual(data["tags"][0]["code"], "ATTACK")
        self.assertEqual(data["tags"][0]["team"], 0)

    def test_prediction_spans_graded_halves_when_gt_is_public_only(self):
        pred = np.full(20, H, dtype=np.int8)
        gt = np.array([X] * 5 + [H] * 10 + [X] * 5, dtype=np.int8)
        data = build_viewer_data(
            pred,
            gt,
            tags=None,
            metrics=None,
            fps=25,
            halves=((0, 10), (12, 20)),
            public=(5, 15),
            start_frame=None,
        )
        self.assertEqual(data["pred_spells"], [[0, 10, H], [12, 20, H]])
        self.assertEqual(data["gt_spells"], [[5, 15, H]])
        self.assertEqual(data["pred_segments"], [[0, 10, H], [12, 20, H]])
        self.assertEqual(data["gt_segments"], [[5, 15, H]])

    def test_without_gt_uses_halves_and_marks_gt_unavailable(self):
        pred = np.full(10, H, dtype=np.int8)
        data = build_viewer_data(
            pred,
            None,
            tags=None,
            metrics=None,
            fps=25,
            halves=((2, 8),),
            public=(2, 8),
            start_frame=5,
        )
        self.assertFalse(data["gt_available"])
        self.assertIsNone(data["gt"])
        self.assertEqual(data["pred_spells"], [[2, 8, H]])
        self.assertEqual(data["disagreements"], [])
        self.assertEqual(data["start_frame"], 5)


class RangeTest(unittest.TestCase):
    def test_parses_normal_open_and_suffix_ranges(self):
        self.assertEqual(parse_byte_range("bytes=10-19", 100), (10, 19))
        self.assertEqual(parse_byte_range("bytes=90-", 100), (90, 99))
        self.assertEqual(parse_byte_range("bytes=-10", 100), (90, 99))

    def test_rejects_invalid_or_unsatisfiable_ranges(self):
        for header in ("not-a-range", "bytes=100-", "bytes=20-10", "bytes=0-1,3-4"):
            with self.subTest(header=header):
                with self.assertRaises(ValueError):
                    parse_byte_range(header, 100)


class ViewerServerTest(unittest.TestCase):
    def setUp(self):
        self.temp = TemporaryDirectory()
        root = Path(self.temp.name)
        (root / "index.html").write_text("<html><body>viewer</body></html>", encoding="utf-8")
        (root / "viewer.js").write_text("console.log('viewer');", encoding="utf-8")
        (root / "viewer.css").write_text("body{color:white}", encoding="utf-8")
        self.video = root / "video.mp4"
        self.video.write_bytes(bytes(range(100)))
        self.payload = json.dumps({"labels": "HHAA"}).encode("utf-8")
        self.server = make_server(
            "127.0.0.1",
            0,
            payload=self.payload,
            video_path=self.video,
            assets_dir=root,
        )
        self.thread = threading.Thread(target=self.server.serve_forever, daemon=True)
        self.thread.start()
        self.base = f"http://127.0.0.1:{self.server.server_address[1]}"

    def tearDown(self):
        self.server.shutdown()
        self.server.server_close()
        self.thread.join(timeout=5)
        self.temp.cleanup()

    def test_serves_index_and_api_payload(self):
        with urlopen(self.base + "/") as response:
            self.assertEqual(response.status, 200)
            self.assertIn(b"viewer", response.read())
        with urlopen(self.base + "/api/viewer-data") as response:
            self.assertEqual(response.status, 200)
            self.assertEqual(response.read(), self.payload)
            self.assertEqual(response.headers["Cache-Control"], "no-store")

    def test_range_request_returns_partial_content(self):
        request = Request(self.base + "/video", headers={"Range": "bytes=10-19"})
        with urlopen(request) as response:
            self.assertEqual(response.status, 206)
            self.assertEqual(response.read(), bytes(range(10, 20)))
            self.assertEqual(response.headers["Accept-Ranges"], "bytes")
            self.assertEqual(response.headers["Content-Range"], "bytes 10-19/100")

    def test_head_video_and_invalid_range(self):
        request = Request(self.base + "/video", method="HEAD")
        with urlopen(request) as response:
            self.assertEqual(response.status, 200)
            self.assertEqual(response.read(), b"")
            self.assertEqual(response.headers["Content-Length"], "100")
        request = Request(self.base + "/video", headers={"Range": "bytes=200-"})
        with self.assertRaises(HTTPError) as raised:
            urlopen(request)
        self.assertEqual(raised.exception.code, 416)

    def test_unknown_path_is_not_served(self):
        with self.assertRaises(HTTPError) as raised:
            urlopen(self.base + "/static/../index.html")
        self.assertEqual(raised.exception.code, 404)


if __name__ == "__main__":
    unittest.main()
