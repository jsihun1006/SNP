"""Evaluate a restart-event classifier followed by a dead-ball classifier."""

import argparse
import json
from pathlib import Path

import lightgbm as lgb
import numpy as np
import pandas as pd

import compare_deadball_half_folds as prior
import dfl_load
from compare_derived_vif import feature_matrix
from config import CONFIG

BASELINE = "vif50_plus_current_mean_370"
EVENTS = ("THROW_IN", "FREE_KICK", "GOAL_KICK", "CORNER_KICK", "KICK_OFF", "PENALTY")
EVENT_COLUMNS = tuple(f"event_{kind.lower()}" for kind in EVENTS)
WINDOW_SECONDS = 0.5


def label_events(timestamps, events):
    """Mark a symmetric window around each recorded restart in a single half."""
    labels = np.zeros(len(timestamps), dtype=np.int8)
    for timestamp, kind in events[["timestamp_s", "set_piece_type"]].itertuples(index=False, name=None):
        if kind not in EVENTS:
            continue
        if not np.isfinite(timestamp) or not timestamps[0] <= timestamp <= timestamps[-1]:
            raise ValueError(f"Restart outside its tracking half: {kind} at {timestamp}.")
        start = timestamps.searchsorted(timestamp - WINDOW_SECONDS, side="left")
        end = timestamps.searchsorted(timestamp + WINDOW_SECONDS, side="right")
        if labels[start:end].any():
            raise ValueError("Two restart windows overlap.")
        labels[start:end] = EVENTS.index(kind) + 1
    return labels


def attach_event_labels(data):
    stride = CONFIG["train_stride"]
    for match_id in dfl_load.MATCHES:
        tracking, events, _ = dfl_load.load_match(match_id)
        for half in (1, 2):
            rows = np.flatnonzero(tracking["period"].to_numpy() == half)
            times = tracking["timestamp_s"].to_numpy()[rows]
            part = data[f"{match_id}_H{half}"]
            truth = label_events(times, events.loc[events["period_id"] == half])
            part["test_event"] = truth
            part["train_event"] = truth[rows % stride == 0]
            if len(part["train_event"]) != len(part["train_dead"]) or len(truth) != len(part["test_dead"]):
                raise ValueError("Event and state labels must share the same frame indices.")


def add_event_columns(frame, values):
    columns = pd.DataFrame(values, columns=EVENT_COLUMNS, index=frame.index, dtype=np.float32)
    return pd.concat([frame, columns], axis=1)


def fit(features, target, threads, event_model=False):
    params = dict(CONFIG["lgbm_params"])
    rounds = params.pop("n_estimators")
    params.update(
        objective="multiclass" if event_model else "binary",
        bagging_fraction=params.pop("subsample"), bagging_freq=params.pop("subsample_freq"),
        feature_fraction=params.pop("colsample_bytree"), min_data_in_leaf=params.pop("min_child_samples"),
        seed=params.pop("random_state"), num_threads=threads,
    )
    if event_model:
        params["num_class"] = len(EVENTS) + 1
    return lgb.train(params, lgb.Dataset(features, label=target), num_boost_round=rounds)


def average_precision(truth, probabilities):
    if not truth.any():
        return np.nan
    order = np.argsort(-probabilities, kind="stable")
    scores = probabilities[order]
    ends = np.r_[np.flatnonzero(scores[1:] != scores[:-1]), len(scores) - 1]
    tp = truth[order].cumsum()[ends]
    return float(np.dot(np.diff(np.r_[0, tp]) / tp[-1], tp / (ends + 1)))


def run_folds(spec, data, decode, settings, threads, evaluation_events):
    use_probabilities = evaluation_events == "predicted"
    model_name = "event_probabilities_376" if use_probabilities else "event_training_only_376"
    state_rows, event_rows = [], []
    for fold in spec["folds"]:
        parts = [data[half] for half in fold["train"]]
        tracking = pd.concat([part["train"][BASELINE] for part in parts], ignore_index=True)
        event_target = np.concatenate([part["train_event"] for part in parts])
        dead_target = np.concatenate([part["train_dead"] for part in parts])
        event_model = fit(tracking, event_target, threads, event_model=True) if use_probabilities else None
        indicators = event_target[:, None] == np.arange(1, len(EVENTS) + 1)
        state_model = fit(add_event_columns(tracking, indicators), dead_target, threads)
        for half in fold["test"]:
            part = data[half]
            test = part["test"][BASELINE]
            if use_probabilities:
                predictions = event_model.predict(test, num_threads=threads)
                if (predictions.shape != (len(test), len(EVENTS) + 1)
                        or not np.isfinite(predictions).all()
                        or not np.allclose(predictions.sum(axis=1), 1)):
                    raise ValueError("Invalid event probabilities.")
                event_values = predictions[:, 1:]
            else:
                event_values = np.full((len(test), len(EVENT_COLUMNS)), np.nan, dtype=np.float32)
            state_score = state_model.predict(add_event_columns(test, event_values), num_threads=threads)
            state, _ = decode(state_score, part["game"], "original", prior.RULES, settings)
            scores = prior.spell_f1(part["test_dead"].astype(bool), state, part["game"]["period"])
            state_rows.append({"fold": fold["fold"], "half_id": half, "model": model_name, **scores})
            if use_probabilities:
                event_truth = part["test_event"]
                for idx, kind in enumerate(("ANY_RESTART", *EVENTS)):
                    truth = (event_truth == idx) if idx else event_truth != 0
                    probability = predictions[:, idx] if idx else 1 - predictions[:, 0]
                    event_rows.append({
                        "fold": fold["fold"], "half_id": half, "event": kind, "frames": len(truth),
                        "positive_frames": int(truth.sum()), "average_precision": average_precision(truth, probability),
                    })
        print(f"Scored fold {fold['fold']} of {spec['n_splits']}", flush=True)
    return pd.DataFrame(state_rows), pd.DataFrame(event_rows)


def matrices(match):
    result = feature_matrix(match)
    if result.shape[1] != 370:
        raise ValueError(f"Expected 370 tracking features, found {result.shape[1]}.")
    return {BASELINE: result}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--splits", type=Path, required=True)
    parser.add_argument("--baseline", type=Path, required=True, help="The previous experiment's by_half.csv")
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--dfl-cache", type=Path, required=True)
    parser.add_argument("--threads", type=int, default=6)
    parser.add_argument("--evaluation-events", choices=("predicted", "missing"), default="predicted")
    args = parser.parse_args()
    if args.threads < 1:
        parser.error("--threads must be positive")
    if args.output.exists() and any(args.output.iterdir()):
        parser.error("--output must be new or empty")
    spec = prior.load_folds(args.splits)
    baseline = pd.read_csv(args.baseline)
    baseline = baseline.loc[baseline["model"] == BASELINE].copy()
    expected = {(item["fold"], half) for item in spec["folds"] for half in item["test"]}
    actual = set(baseline[["fold", "half_id"]].itertuples(index=False, name=None))
    if len(baseline) != len(expected) or actual != expected:
        raise ValueError("Baseline scores do not match the requested splits.")

    movement, decode, settings = prior.load_notebook_rules()
    CONFIG["dfl_cache"] = str(args.dfl_cache)
    data = prior.prepare_halves(movement, build_matrices=matrices)
    attach_event_labels(data)
    state, events = run_folds(spec, data, decode, settings, args.threads, args.evaluation_events)
    model_name = state["model"].iat[0]
    by_half = pd.concat([baseline, state], ignore_index=True).sort_values(["fold", "half_id", "model"])
    by_fold = by_half.groupby(["fold", "model"])[prior.METRICS].mean().reset_index()
    summary = by_fold.groupby("model")[prior.METRICS].mean().loc[[BASELINE, model_name]]
    summary.loc["events_minus_baseline"] = summary.loc[model_name] - summary.loc[BASELINE]

    args.output.mkdir(parents=True, exist_ok=True)
    by_half.to_csv(args.output / "by_half.csv", index=False)
    by_fold.to_csv(args.output / "by_fold.csv", index=False)
    summary.to_csv(args.output / "summary.csv", index_label="model")
    if not events.empty:
        events.to_csv(args.output / "event_by_half.csv", index=False)
    manifest = {
        "splits": spec, "tracking_columns": 370, "state_event_columns": list(EVENT_COLUMNS),
        "event_model": "unweighted multiclass LightGBM including no restart" if args.evaluation_events == "predicted" else None,
        "event_label_window_seconds_each_side": WINDOW_SECONDS,
        "event_labels_source": "DFL set_piece_type and timestamp_s",
        "state_event_inputs": {"train": "binary event indicators", "test": "predicted event probabilities" if args.evaluation_events == "predicted" else "all event inputs NaN"},
        "evaluation_fps": prior.FPS, "rules": list(prior.RULES), "rule_settings": settings,
        "iou_thresholds": list(prior.scoring.pm.SPELL_IOUS), "baseline_csv": args.baseline.name,
        "aggregation": "score halves independently, then average equally over folds",
    }
    (args.output / "run_manifest.json").write_text(json.dumps(manifest, indent=2) + "\n", encoding="utf-8")
    print(summary.round(6).to_string())


if __name__ == "__main__":
    main()
