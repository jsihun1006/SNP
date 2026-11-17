"""Synthetic checkpoint/export contract tests; no private match data required."""
import json
from pathlib import Path
import subprocess
import sys
import tempfile
import unittest

import numpy as np
import pandas as pd

MODULE = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(MODULE))
from extract import extract
from validate import POSITION_FIELDS, sha256, validate_run, write_json


class CheckpointAdapterTests(unittest.TestCase):
    def setUp(self):
        self.temporary = tempfile.TemporaryDirectory()
        self.addCleanup(self.temporary.cleanup)
        self.repo = Path(self.temporary.name)
        source_dir = self.repo / "outputs/unified_tracking_v1"
        source_dir.mkdir(parents=True)
        self.source = source_dir / "tracking_all_frames.csv"
        self.roster = source_dir / "roster.json"
        self.config_path = self.repo / "synthetic_config.json"
        self.table = pd.DataFrame({"frame": np.arange(6),
            "tactical_frame": [-1, 1, 3, 5, 7, np.nan],
            "ball_source": ["RECORDED", "YOLO_RECORDED", "HERMITE_PREDICTION", "MISSING", "TACTICAL_RECORDED", "RECORDED"],
            "ball_is_prediction": [False, False, True, False, False, False]})
        for index, prefix in enumerate(["player_01", "player_02", "ball"]):
            x = np.linspace(.1, .6, 6) + index * .1
            y = np.linspace(.2, .7, 6)
            if prefix == "ball":
                x[3], y[3] = np.nan, np.nan
                x[4], y[4] = 1.2, -.1  # Off-pitch records must be preserved.
            if prefix == "player_02":
                x[2], y[2] = np.nan, np.nan
            for field, value in zip(POSITION_FIELDS, [x, y, x * 105, y * 68, [1, 2, np.nan, 4, 5, 6]]):
                self.table[prefix + "_" + field] = value
        write_json(self.roster, {"players": [
            {"column_prefix": "player_01", "team_code": "H"},
            {"column_prefix": "player_02", "team_code": "A"}]})
        self.config = {"mode": "accepted_checkpoint_export",
            "input_csv": "outputs/unified_tracking_v1/tracking_all_frames.csv",
            "roster": "outputs/unified_tracking_v1/roster.json",
            "source_frames": 6, "tactical_frames": 10, "fps": 25,
            "pitch_m": [105., 68.], "chunk_frames": 2, "validation_chunk_rows": 3,
            "roster_sha256": sha256(self.roster)}
        self.save_source()

    def save_source(self):
        self.table.to_csv(self.source, index=False, float_format="%.17g")
        self.config["source_sha256"] = sha256(self.source)
        write_json(self.config_path, self.config)

    def run_export(self, name="one"):
        return extract(self.repo, self.config_path, name)

    def output(self, name="one"):
        return self.repo / "outputs/phase1/ball_tracking" / name

    def test_export_preserves_positions_missing_prediction_and_bytes(self):
        original = self.source.read_bytes(), self.roster.read_bytes()
        manifest = self.run_export()
        result = pd.read_csv(self.output() / "tracking.csv", float_precision="round_trip")
        status = pd.read_csv(self.output() / "frame_status.csv")
        self.assertEqual(manifest["counts"]["tracking_rows"], 10)
        self.assertEqual(manifest["counts"]["unmapped_broadcast_frames"], 2)
        self.assertEqual(manifest["counts"]["unmapped_broadcast_ball_positions_not_assigned_to_tactical"], 2)
        self.assertEqual(list(status.frame), list(range(10)))
        self.assertEqual(list(result.frame.unique()), [1, 3, 5, 7])
        self.assertEqual(list(status[status.mapped].broadcast_frame), [1, 2, 3, 4])
        self.assertFalse(status.loc[5, "ball_position_available"])
        self.assertTrue(status["ball_world_live_reference"].eq("unknown").all())
        ball = result[result.track_id == "ball"]
        self.assertTrue(ball.team.isna().all())
        self.assertTrue(result.confidence.isna().all())
        self.assertTrue(ball[ball.frame == 3].is_prediction.iloc[0])
        self.assertFalse(ball[ball.frame == 3].observed.iloc[0])
        self.assertEqual(ball[ball.frame == 7].x_norm.iloc[0], 1.2)
        self.assertEqual(ball[ball.frame == 7].y_norm.iloc[0], -.1)
        self.assertEqual(original, (self.source.read_bytes(), self.roster.read_bytes()))
        self.assertTrue(validate_run(self.output(), self.repo)["passed"])
        self.run_export("two")
        for filename in ["tracking.csv", "frame_status.csv"]:
            self.assertEqual((self.output() / filename).read_bytes(), (self.output("two") / filename).read_bytes())

    def test_optional_source_flags_are_preserved(self):
        self.table["ball_world_live_reference"] = [True, False, True, True, False, False]
        self.table["broadcast_highlight_inferred"] = [False, True, False, False, True, False]
        self.save_source()
        self.run_export()
        status = pd.read_csv(self.output() / "frame_status.csv")
        self.assertEqual(status.loc[1, "ball_world_live_reference"], "false")
        self.assertEqual(status.loc[3, "ball_world_live_reference"], "true")
        self.assertEqual(status.loc[7, "broadcast_highlight_inferred"], "true")
        self.assertEqual(status.loc[0, "broadcast_highlight_inferred"], "unknown")

    def test_no_overwrite_and_unsafe_run_id(self):
        self.run_export()
        before = (self.output() / "tracking.csv").read_bytes()
        with self.assertRaisesRegex(ValueError, "overwrite"):
            self.run_export()
        self.assertEqual(before, (self.output() / "tracking.csv").read_bytes())
        for name in ["../escape", "/absolute", "", ".", "a/b"]:
            with self.subTest(name=name), self.assertRaisesRegex(ValueError, "single safe name"):
                self.run_export(name)

    def test_duplicate_or_invalid_mapping_is_rejected(self):
        for bad in [1, 3.5, 10, -2]:
            with self.subTest(mapping=bad):
                self.table.loc[2, "tactical_frame"] = bad
                self.save_source()
                with self.assertRaises(ValueError):
                    self.run_export()
                self.assertFalse(self.output().exists())

    def test_bad_input_hash_and_lfs_pointer(self):
        self.config["source_sha256"] = "0" * 64
        write_json(self.config_path, self.config)
        with self.assertRaisesRegex(ValueError, "SHA256 mismatch"):
            self.run_export()
        self.source.write_text("version https://git-lfs.github.com/spec/v1\noid sha256:000\nsize 999\n")
        with self.assertRaisesRegex(ValueError, "Git LFS pointer"):
            self.run_export()

    def test_source_changes_after_export_are_rejected(self):
        self.run_export()
        self.source.write_bytes(self.source.read_bytes() + b"\n")
        with self.assertRaisesRegex(ValueError, "SHA256 mismatch"):
            validate_run(self.output(), self.repo)

    def test_empty_tracking_does_not_invent_positions(self):
        for prefix in ["player_01", "player_02", "ball"]:
            for field in POSITION_FIELDS[:4]:
                self.table[prefix + "_" + field] = np.nan
        self.save_source()
        manifest = self.run_export()
        self.assertEqual(manifest["counts"]["tracking_rows"], 0)
        self.assertEqual(len(pd.read_csv(self.output() / "tracking.csv")), 0)
        self.assertTrue(validate_run(self.output(), self.repo)["passed"])

    def test_validator_detects_tampering_independent_of_output_hash(self):
        self.run_export()
        tracking_path = self.output() / "tracking.csv"
        tracking_path.write_bytes(tracking_path.read_bytes() + b"\n")
        with self.assertRaisesRegex(ValueError, "output SHA256"):
            validate_run(self.output(), self.repo)
        table = pd.read_csv(tracking_path, float_precision="round_trip")
        table.loc[0, "x_norm"] += .01
        table.loc[0, "x_m"] = table.loc[0, "x_norm"] * 105
        table.to_csv(tracking_path, index=False, float_format="%.17g")
        manifest_path = self.output() / "run_manifest.json"
        manifest = json.loads(manifest_path.read_text())
        manifest["output_sha256"]["tracking.csv"] = sha256(tracking_path)
        write_json(manifest_path, manifest)
        with self.assertRaisesRegex(ValueError, "values differ from the source"):
            validate_run(self.output(), self.repo)

    def test_real_extract_and_validate_cli(self):
        command = [sys.executable, str(MODULE / "extract.py"), "--repo-root", str(self.repo),
            "--config", str(self.config_path), "--input-csv", str(self.source),
            "--roster", str(self.roster), "--run-id", "cli"]
        completed = subprocess.run(command, capture_output=True, text=True)
        self.assertEqual(completed.returncode, 0, completed.stderr)
        checked = subprocess.run([sys.executable, str(MODULE / "validate.py"),
            "--repo-root", str(self.repo), "--run-id", "cli"], capture_output=True, text=True)
        self.assertEqual(checked.returncode, 0, checked.stderr)
        self.assertTrue(json.loads(checked.stdout)["passed"])
        overwritten = subprocess.run(command, capture_output=True, text=True)
        self.assertNotEqual(overwritten.returncode, 0)
        self.assertIn("overwrite", overwritten.stderr)


if __name__ == "__main__":
    unittest.main()
