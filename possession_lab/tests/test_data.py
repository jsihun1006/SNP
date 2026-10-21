"""Tests for persistence and parsing edge cases in the data layer."""

from __future__ import annotations

import tempfile
import unittest
from pathlib import Path

import numpy as np

from possession_lab.data import load_homography, load_smart_tags, load_tracking


ATD_XML = """<?xml version="1.0"?>
<root>
  <Teams>
    <Team id="home"/>
    <Team id="away"/>
  </Teams>
  <Players>
    <Player id="p1" teamId="home"/>
    <Player id="p2" teamId="away"/>
  </Players>
</root>
"""

TAGS_XML = """<?xml version="1.0"?>
<file>
  <ALL_INSTANCES>
    <instance>
      <start>0.0</start><end>2.0</end><code>BUILD UP</code>
      <label><group>Team</group><text>Barcelona</text></label>
      <label><group>Type</group><text>Open Play</text></label>
    </instance>
    <instance>
      <start>1.0</start><end>3.0</end><code>SET PIECES</code>
      <label><group>Team</group><text>N/A</text></label>
      <label><group>Type</group><text>Throw In</text></label>
    </instance>
  </ALL_INSTANCES>
</file>
"""


class DataLoadingTest(unittest.TestCase):
    def test_tracking_parser_aligns_frame_ids_and_scales_to_metres(self):
        with tempfile.TemporaryDirectory() as tmp:
            raw = Path(tmp) / "atd.txt"
            meta = Path(tmp) / "atd.xml"
            raw.write_text(
                "1:0.1,0.2,1.0;NaN,NaN,NaN:0.5,0.5,NaN\n"
                "2:0.3,0.4,NaN;0.7,0.8,2.0:NaN,NaN,NaN\n",
                encoding="utf-8",
            )
            meta.write_text(ATD_XML, encoding="utf-8")
            tracking = load_tracking(raw, meta, pitch=(100.0, 50.0))

        self.assertEqual(tracking.n_frames, 2)
        self.assertEqual(tracking.n_players, 2)
        np.testing.assert_allclose(tracking.ball[0], [50.0, 25.0])
        self.assertTrue(tracking.ball_present[0])
        self.assertFalse(tracking.ball_present[1])
        np.testing.assert_allclose(tracking.players[0, 0], [10.0, 10.0])
        self.assertTrue(tracking.player_present[0, 0])
        self.assertFalse(tracking.player_present[0, 1])
        self.assertAlmostEqual(float(tracking.speeds[0, 0]), 1.0)
        np.testing.assert_allclose(tracking.players[1, 1], [70.0, 40.0])
        np.testing.assert_array_equal(tracking.teams, [0, 1])

    def test_smart_tags_are_half_open_and_team_vote_is_signed(self):
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "tags.xml"
            path.write_text(TAGS_XML, encoding="utf-8")
            tags = load_smart_tags(path, n_frames=100, fps=25)

        self.assertEqual(len(tags.events), 2)
        self.assertTrue(tags.coverage[0])
        self.assertEqual(int(tags.team_vote[25]), 1)
        self.assertFalse(tags.setpiece[24])
        self.assertTrue(tags.setpiece[25])
        self.assertTrue(tags.setpiece[50])
        self.assertEqual(int(tags.team_vote[60]), 0)
        self.assertTrue(tags.coverage[60])
        self.assertIn("BUILD UP", tags.code_names)
        self.assertEqual(float(tags.code_matrix[0].sum()), 1.0)

    def test_homography_loader_keeps_frame_index_and_validity(self):
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "homography.csv"
            path.write_text(
                "frame,h00,h01,h02,h10,h11,h12,h20,h21,h22,reprojection_error,valid\n"
                "0,1,0,0,0,1,0,0,0,1,0.1,1\n"
                "1,NaN,NaN,NaN,NaN,NaN,NaN,NaN,NaN,NaN,NaN,0\n",
                encoding="utf-8",
            )
            homography = load_homography(path)

        self.assertEqual(homography.matrices.shape, (2, 3, 3))
        np.testing.assert_array_equal(homography.valid, [True, False])
        self.assertEqual(homography.n_frames, 2)
        self.assertAlmostEqual(float(homography.reprojection_error[0]), 0.1)


if __name__ == "__main__":
    unittest.main()
