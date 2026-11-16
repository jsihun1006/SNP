"""Summarize predicted H/A possession shares without reading any ground truth.

Use the official games.py match-half frame blocks and the first five minutes
starting at the first half-block frame. H/(H+A), A/(H+A); D and X are excluded.
These are each algorithm's predicted shares, not verified match possession.
"""
import argparse
import csv
from pathlib import Path
import sys

import numpy as np


def summarize_labels(labels, scope, model, fps=25):
    """Summarize codes 0=H,1=A,2=D,3=X; zero denominator -> None."""
    codes = np.asarray(labels)
    if not np.isin(codes, [0, 1, 2, 3]).all():
        raise ValueError("Expected only H/A/D/X label codes")
    counts = {key: int(np.sum(codes == i)) for i, key in enumerate(("H", "A", "D", "X"))}
    live = counts["H"] + counts["A"]
    return {"model": model, "scope": scope, "n_frames": len(codes),
            "H_frames": counts["H"], "A_frames": counts["A"],
            "D_frames": counts["D"], "X_frames": counts["X"],
            "H_seconds": counts["H"] / fps, "A_seconds": counts["A"] / fps,
            "D_seconds": counts["D"] / fps, "X_seconds": counts["X"] / fps,
            "H_A_denominator_frames": live,
            "H_share_percent": 100 * counts["H"] / live if live else None,
            "A_share_percent": 100 * counts["A"] / live if live else None,
            "interpretation": "predicted_H_A_share_not_ground_truth"}


def discover_root(explicit=None):
    if explicit:
        candidates = [Path(explicit)]
    else:
        here = Path(__file__).resolve().parent
        candidates = [here, *here.parents, Path.cwd()]
    root = next((p.resolve() for p in candidates if (p / "games.py").is_file()), None)
    if root is None:
        raise ValueError("Cannot find repository games.py; supply --repo")
    return root


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--hybrid", type=Path, required=True, help="hybrid frame,label CSV")
    parser.add_argument("--out", type=Path, required=True, help="new aggregate CSV path")
    parser.add_argument("--repo", type=Path)
    parser.add_argument("--baseline", type=Path, help="optional override, e.g. synthetic tests")
    parser.add_argument("--model1", type=Path, help="optional override, e.g. synthetic tests")
    args = parser.parse_args()
    root = discover_root(args.repo)
    sys.path.insert(0, str(root))
    sys.path.insert(0, str(root / "evaluation"))
    import possession_metrics as pm
    from games import GAME, GAMES, half_video_blocks

    n_frames = GAMES[GAME]["n_video_frames"]
    fps = GAMES[GAME]["fps"]
    if fps != 25:
        raise ValueError(f"Expected tactical video fps 25; got {fps}")
    inputs = {
        "baseline_smooth50": args.baseline or root / "model/모델별결과csv/naive/predictions/baseline_smooth50.csv",
        "model1": args.model1 or root / "model/모델별결과csv/model1/predictions/model1.csv",
        "video_ball_hybrid": args.hybrid,
    }
    mask = np.zeros(n_frames, dtype=bool)
    blocks = half_video_blocks(GAME)
    for _half, first, last in blocks:
        if not (0 <= first <= last < n_frames) or mask[first:last + 1].any():
            raise ValueError("Invalid or overlapping match-half frame blocks")
        mask[first:last + 1] = True
    # The public window starts at the first graded half-block frame (11185).
    # The separate kickoff playback anchor is one frame later; do not conflate them.
    first_public = min(first for _half, first, _last in blocks)
    last_public_exclusive = first_public + 5 * 60 * fps
    if last_public_exclusive > n_frames:
        raise ValueError("First five-minute window exceeds the video")
    rows = []
    for name, path in inputs.items():
        if not path.is_file():
            parser.error(f"Missing prediction: {path}")
        table = pm.read_label_csv(str(path))
        reason = pm.validate_submission(table, n_frames)
        if reason:
            raise ValueError(f"{name}: {reason}")
        labels = pm.label_codes(table)
        match_row = summarize_labels(labels[mask], "match_halves", name, fps)
        match_row.update(start_frame=min(first for _, first, _ in blocks),
                         end_frame_inclusive=max(last for _, _, last in blocks), n_frame_blocks=len(blocks))
        rows.append(match_row)
        first_row = summarize_labels(labels[first_public:last_public_exclusive], "first_five_minutes", name, fps)
        first_row.update(start_frame=first_public, end_frame_inclusive=last_public_exclusive - 1, n_frame_blocks=1)
        rows.append(first_row)
    if args.out.exists():
        raise FileExistsError(f"Refusing to overwrite {args.out}")
    args.out.parent.mkdir(parents=True, exist_ok=True)
    with args.out.open("w", newline="", encoding="utf-8") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(rows[0]), lineterminator="\n")
        writer.writeheader()
        writer.writerows(rows)
    for row in rows:
        h, a = row["H_share_percent"], row["A_share_percent"]
        values = "undefined (no H or A frames)" if h is None else f"H={h:.2f}% A={a:.2f}%"
        print(f"{row['model']} {row['scope']}: {values}; "
              f"H/A/D/X={row['H_frames']}/{row['A_frames']}/{row['D_frames']}/{row['X_frames']}")
    print(f"Wrote {args.out}; predicted shares only, ground truth not read")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
