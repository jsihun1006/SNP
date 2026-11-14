"""Small real-file fixtures for two-PC merge integrity and frame alignment."""
import hashlib
import json
from pathlib import Path
import tempfile
import unittest

from ball_only.merge import merge


class MergeTests(unittest.TestCase):
    def setUp(self):
        self.temporary = tempfile.TemporaryDirectory()
        self.root = Path(self.temporary.name)

    def tearDown(self):
        self.temporary.cleanup()

    def job(self, name, first, last, *, no_final_newline=False):
        folder = self.root / name
        folder.mkdir()
        config = dict(version="test", video=dict(sha256="a" * 64, bytes=1000, frames=100, fps=25., width=1920, height=1080),
                      weights=dict(sha256="b" * 64, bytes=200),
                      detector=dict(full_imgsz=1536, roi_imgsz=960, tile_size=640, precision="FP32"),
                      batches=dict(full=1, roi=4, tile=8, max_rois=2),
                      start_frame=first, end_frame=last, part_id=name, fingerprint=name)
        rows = [dict(frame_idx=f, source_frame=f, part_frame=f-first, video_seconds=f/25., part_id=name,
                     candidates=[dict(point=[float(f), 12.], confidence=.6)]) for f in range(first, last)]
        payload = "\n".join(json.dumps(row, separators=(",", ":")) for row in rows)
        if not no_final_newline:
            payload += "\n"
        encoded = payload.encode("utf-8")
        (folder / "candidates.jsonl").write_bytes(encoded)
        marker = dict(status="complete", start_frame=first, end_frame=last, frames=last-first,
                      output_path="Z:/old-other-machine/candidates.jsonl",
                      file_sha256=hashlib.sha256(encoded).hexdigest(), source_video_sha256="a" * 64, weights_sha256="b" * 64)
        self.write(folder, "run_config.json", config)
        self.write(folder, "job_complete.json", marker)
        return folder, rows

    @staticmethod
    def write(folder, name, value):
        (folder / name).write_text(json.dumps(value), encoding="utf-8")

    def rewrite_payload(self, folder, rows):
        encoded = ("\n".join(json.dumps(row) for row in rows)+"\n").encode()
        (folder / "candidates.jsonl").write_bytes(encoded)
        marker = json.loads((folder / "job_complete.json").read_text())
        marker["file_sha256"] = hashlib.sha256(encoded).hexdigest()
        self.write(folder, "job_complete.json", marker)

    def test_relocated_contiguous_parts_preserve_rows_and_source_ids(self):
        a, old = self.job("first_half", 2, 5, no_final_newline=True)
        b, new = self.job("second_half", 5, 8)
        result = merge(a, b, self.root / "merged")
        target = Path(result["output_path"])
        self.assertEqual([json.loads(line) for line in target.read_text().splitlines()], old+new)
        self.assertEqual(result["frames"], 6)
        self.assertEqual(result["gap_frames"], 0)
        self.assertEqual(hashlib.sha256(target.read_bytes()).hexdigest(), result["file_sha256"])

    def test_halftime_gap_is_recorded_and_never_filled(self):
        a, _ = self.job("first_half", 0, 3)
        b, _ = self.job("second_half", 8, 10)
        result = merge(a, b, self.root / "merged")
        self.assertEqual(result["frames"], 5)
        self.assertEqual(result["span_frames"], 10)
        self.assertEqual(result["gap_frames"], 5)
        rows = [json.loads(line)["source_frame"] for line in Path(result["output_path"]).read_text().splitlines()]
        self.assertEqual(rows, [0, 1, 2, 8, 9])

    def test_overlap_and_reversed_ranges_rejected(self):
        a, _ = self.job("first_half", 2, 6)
        b, _ = self.job("second_half", 5, 9)
        with self.assertRaisesRegex(ValueError, "overlap|reverse"):
            merge(a, b, self.root / "merged")
        with self.assertRaisesRegex(ValueError, "overlap|reverse"):
            merge(b, a, self.root / "reversed")

    def test_source_video_and_quality_mismatch_rejected(self):
        a, _ = self.job("first_half", 0, 2)
        b, _ = self.job("second_half", 2, 4)
        config = json.loads((b / "run_config.json").read_text())
        config["detector"]["full_imgsz"] = 640
        self.write(b, "run_config.json", config)
        with self.assertRaisesRegex(ValueError, "detector"):
            merge(a, b, self.root / "merged")

    def test_different_throughput_batches_are_allowed_and_recorded(self):
        a, _ = self.job("first_half", 0, 2)
        b, _ = self.job("second_half", 2, 4)
        config = json.loads((b / "run_config.json").read_text())
        config["batches"].update(roi=1, tile=2)
        self.write(b, "run_config.json", config)
        result = merge(a, b, self.root / "merged")
        self.assertTrue(result["batch_settings_differ"])
        self.assertEqual(result["source_ranges"][1]["batches"]["tile"], 2)

    def test_different_roi_search_limit_is_not_a_batch_optimization(self):
        a, _ = self.job("first_half", 0, 2)
        b, _ = self.job("second_half", 2, 4)
        config = json.loads((b / "run_config.json").read_text())
        config["batches"]["max_rois"] = 0
        self.write(b, "run_config.json", config)
        with self.assertRaisesRegex(ValueError, "max_rois"):
            merge(a, b, self.root / "merged")

    def test_corrupted_second_file_never_publishes_merged_output(self):
        a, _ = self.job("first_half", 0, 2)
        b, rows = self.job("second_half", 2, 4)
        rows[0]["candidates"][0]["confidence"] = .9
        # Update valid JSON but retain the completed job's original digest.
        (b / "candidates.jsonl").write_text("\n".join(json.dumps(row) for row in rows)+"\n")
        output = self.root / "merged"
        with self.assertRaisesRegex(ValueError, "SHA256"):
            merge(a, b, output)
        self.assertFalse((output / "candidates.jsonl").exists())
        self.assertFalse((output / "job_complete.json").exists())
        self.assertFalse(list(output.glob("*.tmp-*")))

    def test_duplicate_source_frame_rejected_even_with_updated_hash(self):
        a, _ = self.job("first_half", 0, 2)
        b, rows = self.job("second_half", 2, 4)
        rows[1]["source_frame"] = rows[0]["source_frame"]
        self.rewrite_payload(b, rows)
        with self.assertRaisesRegex(ValueError, "coverage"):
            merge(a, b, self.root / "merged")

    def test_wrong_part_offset_rejected(self):
        a, _ = self.job("first_half", 0, 2)
        b, rows = self.job("second_half", 2, 4)
        rows[0]["part_frame"] = 2
        self.rewrite_payload(b, rows)
        with self.assertRaisesRegex(ValueError, "part_frame"):
            merge(a, b, self.root / "merged")

    def test_incomplete_input_and_existing_output_rejected(self):
        a, _ = self.job("first_half", 0, 2)
        b, _ = self.job("second_half", 2, 4)
        result = merge(a, b, self.root / "merged")
        with self.assertRaises(FileExistsError):
            merge(a, b, self.root / "merged")
        marker = json.loads((b / "job_complete.json").read_text())
        marker["status"] = "stopped"
        self.write(b, "job_complete.json", marker)
        with self.assertRaisesRegex(ValueError, "not complete"):
            merge(a, b, self.root / "other")
        self.assertTrue(Path(result["output_path"]).exists())


if __name__ == "__main__":
    unittest.main()
