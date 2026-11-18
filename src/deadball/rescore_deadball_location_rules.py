"""Compare fixed restart corrections using three ball-location flags on saved DFL scores."""

from pathlib import Path
import sys

_SRC = Path(__file__).resolve().parents[1]
sys.path[:0] = [str(_SRC), str(_SRC.parent)]

import argparse
import json
from time import perf_counter

import numpy as np
import pandas as pd

import compare_deadball_rule_signals as experiment
import deadball_rule_signals as signals
import dfl_load
import features
import final_dead_balls as fd
import simulate
from config import CONFIG

FPS = CONFIG["fps"]
SNAP_RADIUS = 15
PRIOR_OFFSETS = (0, 2, 4, 8)
MAX_PRE_KICK_SPEED = 1.5
PREPARATION_FRAMES = 5
METHODS = ("v2_six_rules", "motion_snap", "location_motion_snap", "location_motion_viterbi",
           "location_preparation_viterbi", "location_both_viterbi")


def restart_candidates(ball, pitch, movement):
    flags = signals.ball_location_flags(ball, pitch)
    recent = {name: np.any([signals.shift(value, shift) == 1 for shift in PRIOR_OFFSETS], axis=0)
              for name, value in flags.items()}
    foot = movement["kick"] & (signals.shift(movement["speed"], 5) <= MAX_PRE_KICK_SPEED)
    throw = movement["throw"]
    candidate = movement["restart"] & (foot | throw)
    allowed = (foot & (recent["ball_in_corner_area"] | recent["ball_in_goal_area"])) | (
        throw & recent["ball_near_touchline"])
    return candidate, candidate & allowed


def preparation_at_restart_location(ball, pitch, movement):
    flags = signals.ball_location_flags(ball, pitch)
    plausible_zone = np.any([value == 1 for value in flags.values()], axis=0)
    stationary = plausible_zone & movement["reliable"] & (movement["speed"] <= MAX_PRE_KICK_SPEED)
    # Five consecutive observed frames avoid treating a transient zone crossing
    # as ball placement. The decoder also requires its original preparation cue.
    frame = np.arange(len(ball))
    cumulative = np.r_[0, np.cumsum(stationary)]
    stable = cumulative[frame+1] - cumulative[np.maximum(0, frame+1-PREPARATION_FRAMES)]
    return (stable == PREPARATION_FRAMES) & ~movement["restart"]


def snap_restarts(dead, candidates, radius=SNAP_RADIUS):
    """Move an existing exit from dead, preserving the surrounding two segments."""
    output = dead.copy()
    ids = np.flatnonzero(candidates)
    spans = features.runs_of(dead)
    next_starts = [start for start, _ in spans[1:]] + [len(dead)]
    edits = []
    for (start, stop), next_start in zip(spans, next_starts, strict=True):
        if stop == len(dead):
            continue
        first = max(start + 1, stop - radius)
        last = min(next_start - 1, stop + radius)
        options = ids[np.searchsorted(ids, first):np.searchsorted(ids, last, side="right")]
        if not len(options):
            continue
        chosen = int(options[np.argmin(np.abs(options-stop))])
        if chosen == stop:
            continue
        output[min(stop, chosen):max(stop, chosen)] = chosen > stop
        edits.append((stop, chosen))
    return output, edits


def matching_context(source):
    rows = pd.read_csv(source / "restart_details.csv")
    rows = rows[(rows.variant == "v2_six_rules") & (rows.condition == "observed_ball")]
    return {
        (match, half): (dict(zip(part.index_in_half, part.cause)),
                        set(part.loc[part.near_corner_free_kick, "index_in_half"]))
        for (match, half), part in rows.groupby(["match", "half"])
    }


def score_half(part, number, index, match, source, baseline, causes, movement_features, decode, settings, output):
    rows = []
    per_event = []
    shifts = []
    path = source / f"{match}_H{number}_predictions.npz"
    with np.load(path, allow_pickle=False) as archive:
        original = {name: archive[name] for name in archive.files}
    if not np.array_equal(original["frame"], part.frame):
        raise ValueError(f"Stored predictions and DFL frames differ for {match} H{number}")
    ball_masked = simulate.ball_dropout(part.ball, np.random.default_rng(2027 + 2*index + number),
                                        baseline["masked_ball_target_fraction"], baseline["masked_ball_mean_gap_seconds"])
    stored = {name: original[name] for name in ("frame", "truth_dead", "v2_p_dead")}
    for condition, ball in zip(experiment.CONDITIONS, (part.ball, ball_masked), strict=True):
        tracked = type(part)(players=part.players, team=part.team, ball=ball,
                             pitch=part.pitch, period=part.period, frame=part.frame)
        geo = fd.ball_signals(tracked)
        crossings = fd.line_crossings(geo)
        movement = movement_features(tracked.players, tracked.team, tracked.ball, tracked.pitch)
        game = {"period": part.period, "conditions": {condition: (geo, crossings, [movement])}}
        base = original[f"{condition}_v2_six_rules_dead"]
        reproduced, _ = decode(original["v2_p_dead"], game, condition, experiment.prior.RULES, settings)
        if not np.array_equal(reproduced, base):
            raise ValueError(f"Re-decoded baseline differs from saved output for {match} H{number} {condition}")
        motion, location = restart_candidates(ball, part.pitch, movement)
        motion_state, motion_edits = snap_restarts(base, motion)
        location_state, location_edits = snap_restarts(base, location)
        # Reuse the decoder's directional restart-cost adapter with binary tracking evidence.
        boundary = np.column_stack((np.zeros(len(ball)), location.astype(float)))
        location_viterbi, _ = decode(original["v2_p_dead"], game, condition, experiment.prior.RULES,
                                     settings, boundary=boundary)
        preparation = preparation_at_restart_location(ball, part.pitch, movement)
        preparation_viterbi, prep_support = decode(
            original["v2_p_dead"], game, condition, experiment.prior.RULES, settings,
            location_preparation=preparation,
        )
        both_viterbi, _ = decode(original["v2_p_dead"], game, condition, experiment.prior.RULES,
                                 settings, boundary=boundary, location_preparation=preparation)
        restart_counts = {"motion_candidates": int(motion.sum()), "location_candidates": int(location.sum()),
                          "motion_snaps": len(motion_edits), "location_snaps": len(location_edits),
                          "location_preparation_frames": int(prep_support["location_preparation"].sum())}
        shifts.append({"match": match, "half": number, "condition": condition, **restart_counts})
        outcomes = (base, motion_state, location_state, location_viterbi, preparation_viterbi, both_viterbi)
        for method, predicted in zip(METHODS, outcomes, strict=True):
            base_row = {"match": match, "half": number, "condition": condition, "variant": method,
                        "ball_missing": 1 - float(geo["seen"].mean())}
            values, events = experiment.metric_rows(original["truth_dead"], predicted, part.period, *causes)
            rows.append({**base_row, **values})
            per_event.extend({**base_row, **event, "source_frame": int(part.frame[event["index_in_half"]])}
                             for event in events)
            stored[f"{condition}_{method}_dead"] = predicted
        stored[f"{condition}_location_restart_candidate"] = location
        stored[f"{condition}_location_preparation"] = prep_support["location_preparation"]
    np.savez_compressed(output / path.name, **stored)
    return rows, per_event, shifts


def run(source, output):
    start = perf_counter()
    manifest = json.loads((source / "manifest.json").read_text(encoding="utf-8"))
    causes = matching_context(source)
    movement, decode, settings = experiment.prior.load_notebook_rules()
    if {k: manifest["settings"][k] for k in settings} != settings:
        raise ValueError("Saved six-rule settings differ from current correction notebook.")
    settings["boundary_cut"] = manifest["settings"]["boundary_cut"]
    output.mkdir(parents=True, exist_ok=False)
    spec = {
        "source_run": str(source.resolve()), "methods": list(METHODS), "maximum_snap_frames": SNAP_RADIUS,
        "pre_kick_location_lookback_frames": list(PRIOR_OFFSETS), "max_pre_kick_speed_mps": MAX_PRE_KICK_SPEED,
        "location_preparation_frames_required": PREPARATION_FRAMES,
        "location_preparation_adjustment": "repeat the existing rule-4 stay-dead bonus and exit penalty while location and settled-ball observations support prep",
        "corner_radius_m": signals.CORNER_RADIUS_M, "touchline_band_each_side_m": signals.TOUCHLINE_BAND_M,
        "goal_area_depth_m": signals.GOAL_AREA_DEPTH_M, "goal_area_half_width_m": signals.GOAL_AREA_HALF_WIDTH_M,
        "decoder_settings": settings, "test_folds": manifest["folds"],
    }
    experiment.save_json(output / "manifest.json", spec)
    scores, results, edits = [], [], []
    for index, match in enumerate(dfl_load.MATCHES):
        tracking, _, meta = dfl_load.load_match(match)
        original = features.match_from_dfl(tracking, meta)
        simulated = simulate.simulate(original, seed=CONFIG["sim_seed"])
        for number in (1, 2):
            part = experiment.prior.slice_half(simulated, original.period == number)
            r, v, e = score_half(part, number, index, match, source, manifest, causes[(match, number)],
                                 movement, decode, settings, output)
            scores.extend(r)
            results.extend(v)
            edits.extend(e)
        print(f"SCORED {match} {index+1}/7 elapsed={perf_counter()-start:.1f}s", flush=True)
    frame = pd.DataFrame(scores)
    frame.to_csv(output / "by_half.csv", index=False)
    summary = experiment.summarize(frame)
    summary.to_csv(output / "summary.csv", index=False)
    pd.DataFrame(results).to_csv(output / "restart_details.csv", index=False)
    pd.DataFrame(edits).to_csv(output / "edits.csv", index=False)
    experiment.save_json(output / "timing.json", {"elapsed_seconds": perf_counter()-start})
    print(summary[["condition", "variant", "dead_spell_f1", "alive_spell_f1",
                   "restart_matched", "restart_spurious", "restart_mae_ms"]].round(4).to_string(index=False), flush=True)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source", required=True, type=Path)
    parser.add_argument("--output", required=True, type=Path)
    args = parser.parse_args()
    run(args.source, args.output)


if __name__ == "__main__":
    main()
