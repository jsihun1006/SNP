"""DFL leave-one-match-out comparison of shared rule signals and fixed corrections."""

from pathlib import Path
import sys

_SRC = Path(__file__).resolve().parents[1]
sys.path[:0] = [str(_SRC), str(_SRC.parent)]

import argparse
import hashlib
import json
from time import perf_counter

import lightgbm as lgb
import numpy as np
import pandas as pd
from scipy.ndimage import maximum_filter1d

import compare_deadball_half_folds as prior
import deadball_rule_signals as signals
import dfl_labels
import dfl_load
import features
import simulate
import smoothing
import train_dfl
from config import CONFIG

FPS = CONFIG["fps"]
STOP, RESTART = 0, 1
CONDITIONS = ("observed_ball", "masked_ball")
BASE = "v2"
MISSING_SHARE = 0.407
GAP_SECONDS = 6.0
BOUNDARY_CUT = 0.64
BOUNDARY_RADIUS = 4
TRANSITION_TOLERANCE = FPS
THREADS_DEFAULT = 4


def save_json(path, values):
    pending = path.with_suffix(".tmp")
    pending.write_text(json.dumps(values, indent=2, allow_nan=False) + "\n", encoding="utf-8")
    pending.replace(path)


def cached_v2(match_id, kind, seed, truth, period):
    path = Path(train_dfl.feature_dir()) / f"{match_id}_{kind}_{seed}.parquet"
    if not path.is_file():
        raise FileNotFoundError(f"Existing local v2 cache required: {path}")
    frame = pd.read_parquet(path)
    if not (np.array_equal(frame["dead"], truth) and np.array_equal(frame["period"], period)):
        raise ValueError(f"Existing v2 cache misaligned: {path}")
    columns = train_dfl.model_columns(frame.columns)
    if len(columns) != 264:
        raise ValueError(f"Unexpected v2 feature count in {path}: {len(columns)}")
    return frame[columns]


def boundary_targets(dead):
    previous, current = dead[:-1], dead[1:]
    changed = np.flatnonzero(previous != current) + 1
    y = np.zeros((len(dead), 2), dtype=np.int8)
    y[changed, np.where(dead[changed], STOP, RESTART)] = 1
    return maximum_filter1d(y, size=2*BOUNDARY_RADIUS+1, axis=0, mode="constant")


def prepared_match(match_id, index, movement):
    # Read local caches only. Event annotations are used for reporting strata.
    if not all(Path(path).is_file() for path in dfl_load.cache_paths(match_id)):
        raise FileNotFoundError(f"Local DFL tracking, events and metadata missing for {match_id}")
    tracking, events, meta = dfl_load.load_match(match_id)
    original = features.match_from_dfl(tracking, meta)
    truth = dfl_labels.dead_label(tracking)
    seed = CONFIG["sim_seed"]
    simulated = simulate.simulate(original, seed=seed)
    train = cached_v2(match_id, "rand", seed+index, truth, original.period)
    stride = CONFIG["train_stride"]
    restarts = dfl_labels.dead_spells(tracking, events)
    halves = {}
    for number in (1, 2):
        rows = np.flatnonzero(original.period == number)
        if not len(rows) or np.any(np.diff(rows) != 1) or np.any(np.diff(original.frame[rows]) != 1):
            raise ValueError(f"Expected a contiguous half at 25 fps for {match_id}, half {number}")
        half = prior.slice_half(simulated, rows)
        dead = truth[rows].astype(bool)
        targets = boundary_targets(dead)
        selected = (np.arange(len(rows)) % 7 == 0) | targets.any(axis=1)
        selected[:BOUNDARY_RADIUS+1] = False
        selected[-BOUNDARY_RADIUS-1:] = False
        ball = simulate.ball_dropout(half.ball, np.random.default_rng(2027+2*index+number),
                                     MISSING_SHARE, GAP_SECONDS)
        data = {}
        for condition, positions in zip(CONDITIONS, (half.ball, ball), strict=True):
            frame, generic, game = signals.tracking_condition(half, positions, movement)
            data[condition] = {"signals": frame, "game": {"period": half.period,
                               "conditions": {condition: game}},
                               "ball_missing": 1 - float(game[0]["seen"].mean())}
        cause = {int(row.stop)-rows[0]: str(row.cause) for row in restarts.itertuples()
                 if row.period == number and rows[0] < row.stop <= rows[-1]}
        near_corner_free = []
        for restart, kind in cause.items():
            if kind != "free_kick":
                continue
            at = half.ball[max(0, restart-FPS):min(len(half.ball), restart+5)]
            x, y = at.T
            distances = np.hypot(np.minimum(x, half.pitch[0]-x), np.minimum(y, half.pitch[1]-y))
            if np.any(distances <= 10.0):
                near_corner_free.append(restart)
        halves[number] = {
            "train": {BASE: train.iloc[rows[rows % stride == 0]].reset_index(drop=True)},
            "train_dead": truth[rows[rows % stride == 0]],
            "test_dead": dead, "frame": half.frame, "period": half.period, "inputs": data,
            "generic_columns": generic, "target": targets, "selected": selected,
            "restart_cause": cause, "near_corner_free": near_corner_free,
        }
    return halves


def fit_boundaries(halves, columns, threads):
    frames, targets = [], []
    for half in halves:
        selected = half["selected"]
        for condition in CONDITIONS:
            frames.append(half["inputs"][condition]["signals"].loc[selected, columns])
            targets.append(half["target"][selected])
    x = pd.concat(frames, ignore_index=True)
    y = np.concatenate(targets)
    params = {"objective": "binary", "learning_rate": 0.07, "num_leaves": 23,
              "min_data_in_leaf": 180, "num_threads": threads, "verbosity": -1, "seed": 17}
    return tuple(lgb.train(params, lgb.Dataset(x, label=target,
                                                weight=np.where(target, 5.0, 1.0)),
                           num_boost_round=110) for target in y.T)


def metric_rows(dead, predicted, period, causes, near_corner):
    pm = prior.scoring.pm
    truth_codes = np.where(dead, pm.D, pm.H)
    pred_codes = np.where(predicted, pm.D, pm.H)
    truth = pm.transitions(truth_codes, [(0, len(dead))])
    estimated = pm.transitions(pred_codes, [(0, len(dead))])
    pairs, missed, spurious = pm.match_transitions(truth, estimated, TRANSITION_TOLERANCE, typed="pair")
    scores = prior.spell_f1(dead, predicted, period)
    per_cause = []
    for name, destination in (("stop", pm.D), ("restart", pm.H)):
        deltas = np.array([int(p[0]-t[0]) for t, p in pairs if t[2] == destination], dtype=int)
        scores.update({f"{name}_matched": len(deltas),
                       f"{name}_missed": sum(t[2] == destination for t in missed),
                       f"{name}_spurious": sum(p[2] == destination for p in spurious),
                       f"{name}_abs_error_frames": int(np.abs(deltas).sum()),
                       f"{name}_signed_error_frames": int(deltas.sum())})
    matched = {t[0]: int(p[0]-t[0]) for t, p in pairs if t[2] == pm.H}
    for frame, _, destination in truth:
        if destination != pm.H:
            continue
        per_cause.append({"index_in_half": frame, "cause": causes.get(frame, "other"),
                          "near_corner_free_kick": frame in near_corner,
                          "matched": frame in matched,
                          "delta_frames": matched.get(frame)})
    return scores, per_cause


def evaluate_half(part, base_score, models, decode, settings, threads, identifiers, output):
    period = part["period"]
    scores = []
    details = []
    stored = {"frame": part["frame"], "truth_dead": part["test_dead"],
              "v2_p_dead": np.asarray(base_score, dtype=np.float32)}
    viterbi = smoothing.viterbi(base_score, period, prior.PENALTY, prior.BIAS).astype(bool)
    for condition, item in part["inputs"].items():
        states = {"v2_viterbi": viterbi}
        states["v2_six_rules"], _ = decode(base_score, item["game"], condition, prior.RULES, settings)
        for name, boundaries in models.items():
            table = item["signals"][boundaries[0].feature_name()]
            prediction = np.column_stack([model.predict(table, num_threads=threads) for model in boundaries])
            if not np.isfinite(prediction).all():
                raise ValueError(f"Nonfinite boundary scores {identifiers}, {condition}, {name}")
            states[name], used = decode(base_score, item["game"], condition, prior.RULES,
                                        settings, boundary=prediction)
            stored[f"{condition}_{name}_boundary"] = prediction.astype(np.float32)
            stored[f"{condition}_{name}_used"] = np.column_stack([used["learned_stop"], used["learned_restart"]])
        for name, outcome in states.items():
            row = {**identifiers, "condition": condition, "variant": name, "ball_missing": item["ball_missing"]}
            stats, events = metric_rows(part["test_dead"], outcome, period, part["restart_cause"],
                                        part["near_corner_free"])
            scores.append({**row, **stats})
            for event in events:
                row_id = event["index_in_half"]
                details.append({**row, "source_frame": int(part["frame"][row_id]), **event})
            stored[f"{condition}_{name}_dead"] = outcome
    path = output / f"{identifiers['match']}_H{identifiers['half']}_predictions.npz"
    pending = path.with_suffix(".tmp.npz")
    np.savez_compressed(pending, **stored)
    pending.replace(path)
    return scores, details


def summarize(scores):
    keys = ["condition", "variant"]
    spans = scores.groupby(["match", *keys])[prior.METRICS].mean().groupby(keys).mean()
    count_fields = [field for field in scores if field.startswith(("stop_", "restart_"))]
    grouped = scores.groupby(keys)[count_fields].sum()
    grouped = spans.join(grouped)
    for kind in ("stop", "restart"):
        matched = grouped[f"{kind}_matched"]
        grouped[f"{kind}_recall_at_1s"] = matched / (matched + grouped[f"{kind}_missed"])
        grouped[f"{kind}_precision_at_1s"] = matched / (matched + grouped[f"{kind}_spurious"])
        grouped[f"{kind}_mae_ms"] = 1000/FPS*grouped[f"{kind}_abs_error_frames"] / matched.replace(0, np.nan)
        grouped[f"{kind}_bias_ms"] = 1000/FPS*grouped[f"{kind}_signed_error_frames"] / matched.replace(0, np.nan)
    return grouped.reset_index()


def run(output, threads):
    start = perf_counter()
    movement, decode, settings = prior.load_notebook_rules()
    settings["boundary_cut"] = BOUNDARY_CUT
    train_dfl.set_feature_set("v2")
    matches = dfl_load.MATCHES
    output.mkdir(parents=True, exist_ok=False)
    spec = {
        "folds": [{"held_out_match": held, "training_matches": [m for m in matches if m != held]}
                  for held in matches],
        "state_features": "existing v2 264-column simulated caches and fixed LightGBM parameters",
        "corrections": list(prior.RULES), "settings": settings, "boundary_head_rounds": 110,
        "boundary_window_frames_each_side": BOUNDARY_RADIUS,
        "boundary_features": "both direction heads use test-available tracking signals only",
        "boundary_training": "observed and masked views, all ±4 frame positives plus 1/7 background",
        "location_flags": {
            "values": "1 inside region, 0 outside region, NaN for unobserved ball",
            "corner_radius_m": signals.CORNER_RADIUS_M,
            "touchline_band_each_side_m": signals.TOUCHLINE_BAND_M,
            "goal_area_depth_m": signals.GOAL_AREA_DEPTH_M,
            "goal_area_half_width_m": signals.GOAL_AREA_HALF_WIDTH_M,
        },
        "boundary_tolerance_evaluation_frames": TRANSITION_TOLERANCE,
        "masked_ball_target_fraction": MISSING_SHARE, "masked_ball_mean_gap_seconds": GAP_SECONDS,
        "ball_mask_is_label_independent": True, "threads": threads, "lightgbm": lgb.__version__,
        "source_sha256": {str(path.name): hashlib.sha256(path.read_bytes()).hexdigest() for path in (
            Path(__file__), Path(signals.__file__), prior.NOTEBOOK, Path(smoothing.__file__))},
    }
    save_json(output / "manifest.json", spec)
    prepared = {}
    for index, match in enumerate(matches):
        prepared[match] = prepared_match(match, index, movement)
        print(f"PREP {match} t={perf_counter()-start:.1f}s", flush=True)
    first = prepared[matches[0]][1]
    all_columns = first["inputs"][CONDITIONS[0]]["signals"].columns.tolist()
    location = list(signals.LOCATION_FLAGS)
    wide = [name for name in all_columns if name not in location]
    generic = first["generic_columns"]
    feature_sets = {"generic": generic, "generic_location": generic + location,
                    "shared": wide, "shared_location": wide + location}
    spec["feature_sets"] = feature_sets
    spec["preparation_seconds"] = perf_counter()-start
    save_json(output / "manifest.json", spec)
    summary_rows, restart_rows, timings = [], [], {}
    for held in matches:
        fold_start = perf_counter()
        training = [part for match in matches if match != held for part in prepared[match].values()]
        state = prior.fit_model(training, BASE, threads)
        models = {f"{name}_heads_six_rules": fit_boundaries(training, columns, threads)
                  for name, columns in feature_sets.items()}
        truth = np.concatenate([part["test_dead"] for part in prepared[held].values()])
        periods = np.concatenate([part["period"] for part in prepared[held].values()])
        test = cached_v2(held, "sim", CONFIG["sim_seed"], truth, periods)
        if state.feature_name() != test.columns.tolist():
            raise ValueError(f"v2 feature order differs in {held}")
        score = state.predict(test, num_threads=threads)
        del test
        position = 0
        for half, part in prepared[held].items():
            end = position+len(part["frame"])
            rows, events = evaluate_half(part, score[position:end], models, decode, settings, threads,
                                         {"match": held, "half": half}, output)
            summary_rows.extend(rows)
            restart_rows.extend(events)
            position = end
        timings[held] = perf_counter()-fold_start
        half_scores = pd.DataFrame(summary_rows)
        half_scores.to_csv(output / "by_half.csv", index=False)
        summarize(half_scores).to_csv(output / "summary.csv", index=False)
        pd.DataFrame(restart_rows).to_csv(output / "restart_details.csv", index=False)
        save_json(output / "timing.json", {"preparation_seconds": spec["preparation_seconds"],
                  "done_folds": timings, "elapsed_total_seconds": perf_counter()-start})
        print(f"FOLD {held} {len(timings)}/7 fold_s={timings[held]:.1f} "
              f"remaining_estimate_s={(7-len(timings))*np.median(list(timings.values())):.0f}", flush=True)
    print(summarize(pd.DataFrame(summary_rows)).to_string(index=False), flush=True)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--dfl-cache", type=Path, default=Path(CONFIG["dfl_cache"]))
    parser.add_argument("--threads", type=int, default=THREADS_DEFAULT)
    args = parser.parse_args()
    if args.threads < 1:
        parser.error("threads must be positive")
    CONFIG["dfl_cache"] = str(args.dfl_cache.resolve())
    run(args.output, args.threads)


if __name__ == "__main__":
    main()
