"""Check coordinate-export integrity, not detection correctness."""
import argparse
import hashlib
import json
from pathlib import Path

import numpy as np
import pandas as pd


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--out", type=Path, default=Path(__file__).resolve().parent / "outputs")
    args = parser.parse_args()
    manifest = json.loads((args.out / "run_manifest.json").read_text())
    assert manifest["status"] == "complete", "Extraction has not completed"
    frames = pd.read_csv(args.out / "frame_status.csv")
    detections = pd.read_csv(args.out / "detections.csv")
    config, video = manifest["config"], manifest["video"]
    expected = np.arange(config["start_frame"], config["end_frame_exclusive"], config["stride"])
    assert np.array_equal(frames.frame, expected), "Missing, duplicated, shifted or reordered source frames"
    assert len(frames) == manifest["sampled_frames"]
    assert np.allclose(frames.time_seconds, frames.frame / video["fps"])
    assert detections.frame.isin(frames.frame).all()
    assert not detections.duplicated(["frame", "detection_index"]).any()
    assert set(detections.class_id) <= {0, 1, 2, 3}
    assert detections.confidence.between(config["conf"] - 1e-7, 1).all()
    assert np.allclose(detections.time_seconds, detections.frame / video["fps"])
    boxes = detections[["x1_px", "y1_px", "x2_px", "y2_px"]].to_numpy()
    assert np.isfinite(boxes).all()
    assert (boxes[:, :2] >= -1e-5).all()
    assert (boxes[:, 2:] >= boxes[:, :2]).all()
    assert (boxes[:, [0, 2]] <= video["width"] + 1e-5).all()
    assert (boxes[:, [1, 3]] <= video["height"] + 1e-5).all()
    anchor = (boxes[:, :2] + boxes[:, 2:]) / 2
    anchor[detections.class_id.to_numpy() != 0, 1] = boxes[detections.class_id.to_numpy() != 0, 3]
    assert np.allclose(anchor, detections[["anchor_x_px", "anchor_y_px"]], atol=1e-6)
    projected = detections.projection_valid == 1
    xy = detections[["x_norm", "y_norm"]].to_numpy()
    assert np.isfinite(xy[projected]).all()
    assert np.allclose(xy[projected] * [105.3, 68.0],
                       detections.loc[projected, ["x_m", "y_m"]], atol=2e-6)
    in_pitch = projected & (xy >= 0).all(axis=1) & (xy <= 1).all(axis=1)
    assert np.array_equal(in_pitch, detections.in_pitch == 1)
    for name in ("person", "ball"):
        selected = detections.class_id != 0 if name == "person" else detections.class_id == 0
        for column, selection in ((f"n_{name}", selected),
                                  (f"n_{name}_in_pitch", selected & in_pitch)):
            counts = detections.loc[selection].groupby("frame").size().reindex(frames.frame, fill_value=0)
            assert np.array_equal(counts.to_numpy(), frames[column]), column
    for name, expected_hash in manifest["output_sha256"].items():
        actual = hashlib.sha256((args.out / name).read_bytes()).hexdigest()
        assert actual == expected_hash, f"Modified export: {name}"
    result = {"valid": True, "kind": "export_integrity_not_detection_accuracy",
              "sampled_frames": len(frames), "detections": len(detections),
              "source_frame_range": [int(frames.frame.min()), int(frames.frame.max())],
              "checks": ["exact source frame sequence and timebase", "class/confidence/box ranges",
                         "bbox anchor definitions", "pitch units and flags", "per-frame counts",
                         "output SHA256 matches extraction manifest"]}
    (args.out / "validation.json").write_text(json.dumps(result, indent=2) + "\n")
    print(json.dumps(result))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
