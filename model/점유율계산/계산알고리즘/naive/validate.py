"""Validate baseline predictions with the official submission contract."""
import argparse
from pathlib import Path

import numpy as np

from common import HERE, N_FRAMES, FPS, metrics, sha256, write_json


def validate_file(path):
    row = {"file": str(path.resolve()), "sha256": sha256(path), "valid": False}
    try:
        table = metrics.read_label_csv(str(path))
        reason = metrics.validate_submission(table, N_FRAMES)
        if reason:
            row["reason"] = reason
            return row
        codes = metrics.label_codes(table)
        spells = metrics.possession_spells(codes, metrics.valid_runs(codes))
        alive = int(np.sum((codes == metrics.H) | (codes == metrics.A)))
        row.update(valid=True, rows=len(codes), first_frame=int(table.frame.iloc[0]),
                   last_frame=int(table.frame.iloc[-1]), reason=None,
                   label_counts={label: int(np.sum(codes == i)) for i, label in enumerate(metrics.LABELS)},
                   full_video_possessions=len(spells),
                   full_video_mean_possession_seconds=alive / FPS / len(spells) if spells else 0.0)
    except (ValueError, OSError, KeyError) as error:
        row["reason"] = str(error)
    return row


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("files", nargs="*", type=Path)
    parser.add_argument("--predictions", type=Path, default=HERE / "predictions")
    parser.add_argument("--out", type=Path, default=HERE / "validation.json")
    args = parser.parse_args()
    files = args.files or sorted(args.predictions.glob("*.csv"))
    if not files:
        parser.error("No prediction CSVs; run baseline.py first.")
    rows = [validate_file(path) for path in files]
    write_json(args.out, rows)
    for row in rows:
        print(f"{'PASS' if row['valid'] else 'FAIL'} {Path(row['file']).name}: "
              f"{row.get('rows', '?')} rows; {row.get('reason') or 'frame 0..199318, labels H/A/D/X'}")
    return 0 if all(row["valid"] for row in rows) else 1


if __name__ == "__main__":
    raise SystemExit(main())
