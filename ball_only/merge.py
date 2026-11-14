"""Verify and join two completed ball-candidate jobs without renumbering frames.

No detector, source video, ATD, homography, or player model is loaded here.
Candidate coverage is not a ground-truth ball-detection accuracy measurement.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import math
import os
from pathlib import Path
import time
import uuid


VERSION = "ball_only_two_part_merge_v1"


def _read(path):
    return json.loads(Path(path).read_text(encoding="utf-8-sig"))


def _integer(value, name, *, minimum=0):
    if isinstance(value, bool) or not isinstance(value, int) or value < minimum:
        raise ValueError(f"{name} must be an integer >= {minimum}")
    return value


def _sha_value(value, name):
    if not isinstance(value, str) or len(value) != 64:
        raise ValueError(f"{name} must be a SHA256 hex digest")
    try:
        bytes.fromhex(value)
    except ValueError as exc:
        raise ValueError(f"{name} must be a SHA256 hex digest") from exc
    return value.lower()


def _signature(path):
    path = Path(path).resolve(strict=True)
    info = path.stat()
    return dict(path=str(path), bytes=info.st_size, mtime_ns=info.st_mtime_ns)


def _atomic_json(path, value):
    path = Path(path)
    temporary = path.with_name(path.name + ".tmp-" + uuid.uuid4().hex)
    try:
        with temporary.open("w", encoding="utf-8") as handle:
            json.dump(value, handle, ensure_ascii=False, indent=2, allow_nan=False)
            handle.write("\n")
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(temporary, path)
    finally:
        if temporary.exists():
            temporary.unlink()


def _load_job(folder):
    folder = Path(folder).resolve(strict=True)
    config_path = folder / "run_config.json"
    complete_path = folder / "job_complete.json"
    config, complete = _read(config_path), _read(complete_path)
    if complete.get("status") != "complete":
        raise ValueError(f"Input job is not complete: {folder}")
    first = _integer(config.get("start_frame"), "start_frame")
    last = _integer(config.get("end_frame"), "end_frame")
    if last <= first:
        raise ValueError("Input frame ranges must be nonempty [start_frame, end_frame)")
    for key, expected in (("start_frame", first), ("end_frame", last), ("frames", last - first)):
        actual = _integer(complete.get(key), "job_complete." + key)
        if actual != expected:
            raise ValueError(f"Input completion marker and config disagree on {key}")
    video, weights = config.get("video"), config.get("weights")
    if not isinstance(video, dict) or not isinstance(weights, dict):
        raise ValueError("Input config must identify its video and weights")
    video_sha = _sha_value(video.get("sha256"), "video.sha256")
    weights_sha = _sha_value(weights.get("sha256"), "weights.sha256")
    if _sha_value(complete.get("source_video_sha256"), "source_video_sha256") != video_sha:
        raise ValueError("Input completion marker identifies a different source video")
    if _sha_value(complete.get("weights_sha256"), "weights_sha256") != weights_sha:
        raise ValueError("Input completion marker identifies different weights")
    video_identity = dict(sha256=video_sha)
    for name in ("bytes", "frames", "width", "height"):
        video_identity[name] = _integer(video.get(name), "video." + name, minimum=1)
    fps = video.get("fps")
    if isinstance(fps, bool) or not isinstance(fps, (int, float)) or not math.isfinite(fps) or fps <= 0:
        raise ValueError("video.fps must be finite and positive")
    video_identity["fps"] = float(fps)
    if last > video_identity["frames"]:
        raise ValueError("Input range extends beyond the source video")
    weight_identity = dict(sha256=weights_sha, bytes=_integer(weights.get("bytes"), "weights.bytes", minimum=1))
    detector, batches = config.get("detector"), config.get("batches")
    if not isinstance(detector, dict) or not detector or not isinstance(batches, dict) or not batches:
        raise ValueError("Input config must record detector and batch settings")
    part_id = config.get("part_id")
    if not isinstance(part_id, str) or not part_id:
        raise ValueError("Input part_id must be a nonempty string")
    # The recorded absolute output_path belongs to the machine that ran the
    # job. Copying a job directory must not require that old drive/path.
    payload = folder / "candidates.jsonl"
    return dict(folder=folder, config=config, complete=complete,
                first=first, last=last, part_id=part_id, fps=float(fps),
                video=video_identity, weights=weight_identity,
                detector=detector, batches=batches, payload=payload,
                expected_sha256=_sha_value(complete.get("file_sha256"), "file_sha256"),
                signatures=[_signature(p) for p in (config_path, complete_path, payload)])


def _reject_constant(value):
    raise ValueError("Non-finite JSON value in candidate output: " + value)


def _copy_verified(job, output, output_digest):
    digest = hashlib.sha256()
    expected_frame = job["first"]
    with job["payload"].open("rb") as handle:
        for raw in handle:
            digest.update(raw)
            if not raw.strip():
                raise ValueError("Candidate file contains an empty record")
            row = json.loads(raw.decode("utf-8-sig"), parse_constant=_reject_constant)
            if not isinstance(row, dict):
                raise ValueError("Candidate records must be JSON objects")
            frame = _integer(row.get("source_frame"), "source_frame")
            if frame != expected_frame or frame >= job["last"]:
                raise ValueError(f"Candidate source-frame coverage differs at {expected_frame}")
            if _integer(row.get("frame_idx"), "frame_idx") != frame:
                raise ValueError("frame_idx differs from source_frame")
            if _integer(row.get("part_frame"), "part_frame") != frame - job["first"]:
                raise ValueError("part_frame differs from the original part range")
            if row.get("part_id") != job["part_id"]:
                raise ValueError("Candidate part_id differs from its config")
            seconds = row.get("video_seconds")
            if (isinstance(seconds, bool) or not isinstance(seconds, (int, float))
                    or not math.isfinite(seconds)
                    or abs(seconds - frame / job["fps"]) > 1e-5):
                raise ValueError("Candidate video_seconds differs from the source timeline")
            encoded = raw if raw.endswith(b"\n") else raw + b"\n"
            output.write(encoded)
            output_digest.update(encoded)
            expected_frame += 1
    if expected_frame != job["last"]:
        raise ValueError("Candidate file ends before its declared range")
    if digest.hexdigest() != job["expected_sha256"]:
        raise ValueError("Candidate file SHA256 differs from the completed job")
    for signature in job["signatures"]:
        if _signature(signature["path"]) != signature:
            raise ValueError("Input job changed while merging")


def merge(first, second, out):
    """Merge two completed job directories; ranges must be ordered/nonoverlapping.

    A halftime gap is allowed and explicitly recorded. No records are added for
    that gap. Existing outputs are never overwritten. Source rows retain their
    original part_id, part_frame and global source_frame values.
    """
    jobs = [_load_job(first), _load_job(second)]
    left, right = jobs
    if left["folder"] == right["folder"]:
        raise ValueError("Two distinct input jobs are required")
    for key in ("video", "weights", "detector"):
        if left[key] != right[key]:
            raise ValueError(f"Input jobs have different {key} identities/settings")
    # Throughput batches may differ on GPUs with different VRAM. ROI count is
    # a search-quality setting, however, even if the runner records it beside
    # batch sizes; it must still agree.
    if left["batches"].get("max_rois") != right["batches"].get("max_rois"):
        raise ValueError("Input jobs have different max_rois search settings")
    batch_settings_differ = left["batches"] != right["batches"]
    if left["last"] > right["first"]:
        raise ValueError("Input ranges overlap or are in reverse order")
    out = Path(out).resolve()
    if out in (left["folder"], right["folder"]):
        raise ValueError("Merged output must be separate from both source jobs")
    target, marker = out / "candidates.jsonl", out / "job_complete.json"
    if target.exists() or marker.exists() or (out / "merge_config.json").exists():
        raise FileExistsError("Merged output already exists; choose a fresh output directory")
    out.mkdir(parents=True, exist_ok=True)
    temporary = target.with_name(target.name + ".tmp-" + uuid.uuid4().hex)
    started, output_digest = time.perf_counter(), hashlib.sha256()
    try:
        with temporary.open("wb") as handle:
            for job in jobs:
                _copy_verified(job, handle, output_digest)
            handle.flush()
            os.fsync(handle.fileno())
        # Recheck both sources after validating the SECOND file too. Nothing is
        # published under a final filename before all inputs have passed.
        for job in jobs:
            for signature in job["signatures"]:
                if _signature(signature["path"]) != signature:
                    raise ValueError("Input job changed before merge publication")
        gap = right["first"] - left["last"]
        ranges = [dict(part_id=j["part_id"], start_frame=j["first"], end_frame=j["last"],
                       frames=j["last"] - j["first"], input_folder=str(j["folder"]),
                       input_sha256=j["expected_sha256"], batches=j["batches"],
                       fingerprint=j["config"].get("fingerprint")) for j in jobs]
        gaps = ([dict(start_frame=left["last"], end_frame=right["first"], frames=gap)] if gap else [])
        config = dict(version=VERSION, video=left["video"], weights=left["weights"],
                      detector=left["detector"],
                      batches_by_part=[dict(part_id=j["part_id"], batches=j["batches"]) for j in jobs],
                      batch_settings_differ=batch_settings_differ,
                      source_ranges=ranges, gaps=gaps, source_frame_ids_preserved=True,
                      gap_frames_filled=False, candidate_outputs_are_not_ground_truth=True)
        completion = dict(status="complete", version=VERSION,
                          start_frame=left["first"], end_frame=right["last"],
                          frames=sum(r["frames"] for r in ranges),
                          span_frames=right["last"] - left["first"],
                          gap_frames=gap, gaps=gaps, source_ranges=ranges,
                          batch_settings_differ=batch_settings_differ,
                          fps=left["fps"], output_path=str(target),
                          file_sha256=output_digest.hexdigest(),
                          source_video_sha256=left["video"]["sha256"],
                          weights_sha256=left["weights"]["sha256"],
                          source_frame_ids_preserved=True, interpolation_performed=False,
                          source_files_modified=False, ground_truth_accuracy_validated=False,
                          elapsed_seconds=time.perf_counter() - started, finished_unix=time.time())
        os.replace(temporary, target)
        _atomic_json(out / "merge_config.json", config)
        _atomic_json(marker, completion)
        return completion
    finally:
        if temporary.exists():
            temporary.unlink()


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--first", required=True, help="Completed first-half job directory")
    parser.add_argument("--second", required=True, help="Completed second-half job directory")
    parser.add_argument("--out", required=True, help="New merged-output directory")
    args = parser.parse_args()
    print(json.dumps(merge(args.first, args.second, args.out), ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
