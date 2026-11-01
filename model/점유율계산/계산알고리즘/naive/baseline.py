"""Nearest-player baseline; fall back explicitly to the published raw predictions.

No ground truth is opened here. The algorithm and four fixed windows are reused
from starter/baseline.py. Video decoding and model training are not required.
"""
import argparse
from datetime import datetime, timezone
from pathlib import Path

import numpy as np

from common import (HERE, ROOT, N_FRAMES, PRIMARY, asset_paths, checked_labels,
                    official_baseline as reference, sha256, write_json)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input-mode", choices=("auto", "atd", "published-raw"), default="auto")
    parser.add_argument("--data-dir", type=Path)
    parser.add_argument("--atd-raw", type=Path)
    parser.add_argument("--atd-meta", type=Path)
    parser.add_argument("--raw-predictions", type=Path,
                        default=ROOT / "evaluation/results/baseline_raw.csv")
    parser.add_argument("--out", type=Path, default=HERE / "predictions")
    args = parser.parse_args()
    paths = asset_paths(data_dir=str(args.data_dir) if args.data_dir else None)
    raw = args.atd_raw or Path(paths["atd_raw"])
    meta = args.atd_meta or Path(paths["atd_meta"])
    explicit = args.atd_raw is not None or args.atd_meta is not None or args.data_dir is not None
    have_atd = raw.is_file() and meta.is_file()
    mode = args.input_mode
    if mode == "auto":
        # An explicit but broken ATD path must fail rather than silently use old predictions.
        mode = "atd" if have_atd or explicit else "published-raw"
    if mode == "atd":
        if not have_atd:
            parser.error(f"ATD TXT/XML required: {raw} ; {meta}")
        codes = reference.nearest_player_labels(
            str(raw), reference.team_of_each_track(str(meta)), N_FRAMES, (105.3, 68.0))
        inputs = [raw, meta]
        note = "Fresh nearest-player inference from tactical ATD; no ground truth used."
    else:
        _, published = checked_labels(args.raw_predictions)
        if np.any(published == 3):
            parser.error("Published raw predictions must contain only H/A/D, no X.")
        codes = np.where(published == 0, reference.HOME,
                         np.where(published == 1, reference.AWAY, reference.NOBODY))
        inputs = [args.raw_predictions]
        note = ("Reconstructed raw codes from organiser predictions; nearest-player inference "
                "was NOT rerun because ATD inputs are unavailable. No coordinates or video read.")
    args.out.mkdir(parents=True, exist_ok=True)
    outputs = []
    for name, window in reference.VARIANTS:
        labels = reference.smooth(codes, window)
        dest = args.out / f"{name}.csv"
        reference.write_submission(str(dest), labels)
        published = ROOT / "evaluation/results" / dest.name
        identical = sha256(dest) == sha256(published) if published.is_file() else None
        outputs.append({"name": name, "half_window_frames": window, "path": str(dest.resolve()),
                        "sha256": sha256(dest), "matches_published_bytes": identical,
                        "label_counts": {k: int(np.sum(labels == k)) for k in "HADX"}})
        print(f"{name}: {N_FRAMES} frames; identical to published = {identical}")
    write_json(args.out.parent / "run_manifest.json", {
        "created_at_utc": datetime.now(timezone.utc).isoformat(),
        "input_mode": mode, "note": note, "primary_model": PRIMARY,
        "primary_selection": "Published smooth50 reference; never chosen by public GT scores.",
        "inputs": [{"path": str(p.resolve()), "sha256": sha256(p)} for p in inputs],
        "expected_atd": {"raw": str(raw), "meta": str(meta), "available": have_atd},
        "reference_code": {str(ROOT / "starter/baseline.py"): sha256(ROOT / "starter/baseline.py")},
        "outputs": outputs,
    })
    print(f"input mode: {mode}\n{note}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
