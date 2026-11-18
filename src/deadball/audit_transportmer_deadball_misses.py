"""Count the reference TranSPORTmer Model 1's missed DFL stoppages by situation."""

from pathlib import Path
import sys

_SRC = Path(__file__).resolve().parents[1]
sys.path[:0] = [str(_SRC), str(_SRC.parent)]

import argparse

import numpy as np
import pandas as pd

import dfl_labels

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
from evaluation import possession_metrics as pm  # noqa: E402

REFERENCE = ROOT / "experiments/deadball/transportmer/dfl_7fold/reference/by_half.csv"
STOP_EVENTS = {
    "BALL_OUT": "ball_out_restart_unknown",
    "FOUL_COMMITTED": "foul",
    "GENERIC:Offside": "offside",
}
IOU_COLUMNS = {threshold: f"miss_iou_{threshold:.1f}" for threshold in pm.SPELL_IOUS}


def classify(restart, candidates, first, last):
    if first and restart in ("kick_off", "other"):
        return "half_start"
    if last:
        return "half_end"
    if restart not in ("free_kick", "other"):
        return restart
    allowed = {"offside", "foul"} if restart == "free_kick" else set(STOP_EVENTS.values())
    if len(candidates) == 1 and candidates[0] in allowed:
        return candidates[0]
    return "free_kick_cause_unknown" if restart == "free_kick" else "unconfirmed"


def audit_half(tracking, events, predicted):
    truth = dfl_labels.dead_label(tracking).astype(bool)
    gt_codes = np.where(truth, pm.H, pm.D)
    pred_codes = np.where(predicted, pm.H, pm.D)
    runs = pm.valid_runs(gt_codes)
    actual = pm.possession_spells(gt_codes, runs)
    forecasts = pm.possession_spells(pred_codes, runs)
    overlaps = pm.spell_overlaps(actual, forecasts)
    flags = {threshold: pm.matched_gt_flags(overlaps, len(actual), threshold)
             for threshold in pm.SPELL_IOUS}
    score = np.mean([2 * match.sum() / max(1, len(actual) + len(forecasts))
                     for match in flags.values()])
    best_iou = np.zeros(len(actual))
    for iou, gt, _ in overlaps:
        best_iou[gt] = max(iou, best_iou[gt])

    tags = dfl_labels.dead_spells(tracking, events)
    np.testing.assert_array_equal(tags[["start", "stop"]].to_numpy(),
                                  np.array([(a, b) for a, b, _ in actual], dtype=int).reshape(-1, 2))
    starts = events.loc[events.event_type.isin(STOP_EVENTS), ["event_type", "timestamp_s"]]
    times = tracking.timestamp_s.to_numpy()
    frames = tracking.frame_id.to_numpy()
    output = []
    for i, tag in enumerate(tags.itertuples(index=False)):
        start, stop = tag.start, tag.stop
        # Referee/ball-out tags can be placed shortly before or after BallStatus flips.
        close = starts.loc[(starts.timestamp_s - times[start]).abs() <= 2]
        candidates = sorted({STOP_EVENTS[event] for event in close.event_type})
        output.append({
            "start_frame": frames[start], "end_frame_inclusive": frames[stop - 1],
            "start_s": times[start], "duration_s": tag.seconds,
            "situation": classify(tag.cause, candidates, start == 0, stop == len(tracking)),
            "restart": tag.cause, "nearby_stop_tags": ",".join(candidates),
            "best_iou": best_iou[i], "no_overlap": best_iou[i] == 0,
            **{column: not flags[threshold][i] for threshold, column in IOU_COLUMNS.items()},
        })
    return score, output


def csv_mask(data, tracking):
    frame = next((column for column in ("source_frame_id", "frame_id") if column in data), None)
    state = next((column for column in ("after_rules", "dead") if column in data), None)
    if frame is None or state is None:
        raise ValueError("Prediction CSV needs frame_id (or source_frame_id) and after_rules (or dead)")
    expected = tracking.frame_id.to_numpy()
    aligned = data.loc[data[frame].between(expected[0], expected[-1])].sort_values(frame)
    np.testing.assert_array_equal(aligned[frame].to_numpy(), expected, err_msg="Prediction CSV frame IDs")
    return aligned[state].to_numpy()


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--predictions", type=Path, required=True,
                        help="Combined CSV or directory of per-half restored_none predictions")
    parser.add_argument("--cache", type=Path, default=ROOT / "data/external/dfl")
    parser.add_argument("--reference", type=Path, default=REFERENCE,
                        help="by_half.csv from the same run as the predictions")
    parser.add_argument("--holdout", help="Audit both halves of a single held-out match")
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    reference = pd.read_csv(args.reference).query(
        "condition == 'restored' and event_mode == 'none' and stage == 'after_rules'")
    if args.holdout:
        expected = {f"{args.holdout}_H1", f"{args.holdout}_H2"}
        if len(reference) != 2 or set(reference.half_id) != expected:
            parser.error(f"Expected exactly the two halves of {args.holdout} in the selected run.")
    elif reference.half_id.nunique() != 14 or len(reference) != 14:
        parser.error("Expected all 14 half predictions or use --holdout for a single match.")
    combined = None
    files = {}
    if args.predictions.is_file() and args.predictions.suffix.lower() == ".csv":
        combined = pd.read_csv(args.predictions)
        match_column = next((column for column in ("match_id", "match") if column in combined), None)
        if match_column is None:
            parser.error("Combined prediction CSV needs a match_id or match column.")
    else:
        for half in reference.half_id:
            csv = args.predictions / f"{half}_restored_none.csv"
            files[half] = csv if csv.is_file() else csv.with_suffix(".npz")
        missing = [half for half, path in files.items() if not path.is_file()]
        if missing:
            parser.error(f"Missing reference predictions in {args.predictions} for {', '.join(missing)}")

    rows = []
    for row in reference.itertuples(index=False):
        match, half = row.half_id.split("_H")
        tracking = pd.read_parquet(args.cache / f"{match}_tracking.parquet",
                                   columns=["frame_id", "period", "timestamp_s", "ball_state"])
        tracking = tracking.loc[tracking.period == int(half)]
        events = pd.read_parquet(args.cache / f"{match}_events.parquet")
        events = events.loc[events.period_id == int(half)]
        if combined is not None:
            estimate = csv_mask(combined.loc[combined[match_column] == match], tracking)
        elif files[row.half_id].suffix == ".csv":
            estimate = csv_mask(pd.read_csv(files[row.half_id]), tracking)
        else:
            with np.load(files[row.half_id], allow_pickle=False) as pred:
                np.testing.assert_array_equal(pred["truth"], dfl_labels.dead_label(tracking),
                                              err_msg=f"DFL truth alignment on {row.half_id}")
                estimate = pred["after_rules"]
        if estimate.shape != (len(tracking),) or not np.isin(estimate, [0, 1]).all():
            raise ValueError(f"Invalid prediction mask for {row.half_id}")
        score, details = audit_half(tracking, events, estimate)
        np.testing.assert_allclose(score, row.dead_spell_f1, rtol=0, atol=1e-12,
                                   err_msg=f"Reference Spell F1 on {row.half_id}")
        rows.extend({"match": match, "half": int(half), **detail} for detail in details)

    details = pd.DataFrame(rows)
    by_situation = details.groupby("situation").agg(
        stoppages=("start_frame", "size"), no_overlap=("no_overlap", "sum"),
        **{name.replace(".", "_"): (name, "sum") for name in IOU_COLUMNS.values()},
    ).sort_values("miss_iou_0_5", ascending=False)
    args.output.mkdir(parents=True, exist_ok=True)
    details.to_csv(args.output / "stoppages.csv", index=False)
    by_situation.to_csv(args.output / "by_situation.csv")
    print(by_situation.to_string())


if __name__ == "__main__":
    main()
