"""Three-state offline sequence decoder over noisy nearest-team observations.

H/A observations come from the organiser baseline, or from fresh ATD inference.
Raw baseline D is interpreted as MISSING, not as a ground-truth dead-ball state.
No ground truth is read here. No parameter estimation or score search is done.
"""
import argparse
from datetime import datetime, timezone
import json
from pathlib import Path

import numpy as np
import pandas as pd

from common import (HERE, N_FRAMES, FPS, ROOT, asset_paths, checked_labels,
                    official_baseline as baseline, sha256, write_json)

H, A, D = 0, 1, 2
MISSING = -1
STATES = np.array(["H", "A", "D"])


def missing_run_lengths(observations):
    """Total length of the containing missing run; zero at observed frames."""
    missing = np.asarray(observations) == MISSING
    edges = np.diff(np.r_[False, missing, False].astype(np.int8))
    starts, stops = np.flatnonzero(edges == 1), np.flatnonzero(edges == -1)
    lengths = np.zeros(len(observations), dtype=np.int32)
    for start, stop in zip(starts, stops):
        lengths[start:stop] = stop - start
    return lengths


def validate_config(config):
    keys = ("long_missing_seconds", "opposite_team_cost_per_second",
            "dead_when_observed_cost_per_second", "dead_in_short_gap_cost_per_second",
            "alive_in_long_gap_cost_per_second", "team_switch_cost", "dead_switch_cost")
    if config["fps"] != FPS:
        raise ValueError("Config fps must match the official tactical fps.")
    for key in keys:
        value = config[key]
        if not np.isfinite(value) or value <= 0:
            raise ValueError(f"{key} must be finite and positive")


def emission_costs(observations, config):
    """Non-probabilistic costs for each frame and latent H/A/D state."""
    observations = np.asarray(observations, dtype=np.int8)
    if not np.isin(observations, [MISSING, H, A]).all():
        raise ValueError("Observations must be H=0, A=1, missing=-1.")
    lengths = missing_run_lengths(observations)
    long_gap = lengths >= int(np.ceil(config["long_missing_seconds"] * FPS))
    costs = np.zeros((len(observations), 3), dtype=np.float64)
    costs[observations == H, A] = config["opposite_team_cost_per_second"]
    costs[observations == A, H] = config["opposite_team_cost_per_second"]
    costs[observations != MISSING, D] = config["dead_when_observed_cost_per_second"]
    costs[(observations == MISSING) & ~long_gap, D] = config["dead_in_short_gap_cost_per_second"]
    costs[long_gap, :2] = config["alive_in_long_gap_cost_per_second"]
    return costs / FPS, lengths, long_gap


def decode(observations, config):
    """Exact minimum of emission costs + state-change penalties via Viterbi DP.

    Minimise sum_t E[t, state[t]] + sum_t T[state[t-1], state[t]].
    Initial states have equal cost. Uses future context; this is not streaming.
    State ties are deterministic in H,A,D order, not calibrated confidence.
    """
    validate_config(config)
    costs, lengths, long_gap = emission_costs(observations, config)
    if not len(costs):
        return np.array([], dtype=np.int8), costs, lengths, long_gap, 0.0
    team, dead = config["team_switch_cost"], config["dead_switch_cost"]
    transitions = np.array([[0, team, dead], [team, 0, dead], [dead, dead, 0]], dtype=float)
    back = np.zeros((len(costs), 3), dtype=np.int8)
    score = costs[0].copy()
    for frame in range(1, len(costs)):
        candidates = score[:, None] + transitions
        parent = candidates.argmin(axis=0)
        back[frame] = parent
        score = costs[frame] + candidates[parent, np.arange(3)]
    labels = np.empty(len(costs), dtype=np.int8)
    labels[-1] = score.argmin()
    objective = float(score[labels[-1]])
    for frame in range(len(costs) - 1, 0, -1):
        labels[frame - 1] = back[frame, labels[frame]]
    return labels, costs, lengths, long_gap, objective


def read_observations(args):
    paths = asset_paths(data_dir=str(args.data_dir) if args.data_dir else None)
    raw, meta = args.atd_raw or Path(paths["atd_raw"]), args.atd_meta or Path(paths["atd_meta"])
    available = raw.is_file() and meta.is_file()
    explicit = any(x is not None for x in (args.data_dir, args.atd_raw, args.atd_meta))
    mode = args.input_mode
    if mode == "auto":
        mode = "atd" if available or explicit or raw.exists() or meta.exists() else "published-raw"
    if mode == "atd":
        if not available:
            raise ValueError(f"Both ATD TXT/XML required: {raw}; {meta}")
        observations = baseline.nearest_player_labels(
            str(raw), baseline.team_of_each_track(str(meta)), N_FRAMES, (105.3, 68.0))
        inputs = [raw, meta]
    else:
        _, published = checked_labels(args.raw_predictions)
        if (published == 3).any():
            raise ValueError("Raw nearest-team input cannot contain X.")
        observations = np.where(published == H, H, np.where(published == A, A, MISSING))
        inputs = [args.raw_predictions]
    return observations, mode, inputs


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input-mode", choices=("auto", "atd", "published-raw"), default="auto")
    parser.add_argument("--data-dir", type=Path)
    parser.add_argument("--atd-raw", type=Path)
    parser.add_argument("--atd-meta", type=Path)
    parser.add_argument("--raw-predictions", type=Path, default=ROOT / "evaluation/results/baseline_raw.csv")
    parser.add_argument("--config", type=Path, default=HERE / "config.json")
    parser.add_argument("--out", type=Path, default=HERE)
    args = parser.parse_args()
    config = json.loads(args.config.read_text())
    validate_config(config)
    observations, mode, inputs = read_observations(args)
    # This record is saved before inference or evaluation, with no GT file opened.
    manifest = {
        "created_at_utc": datetime.now(timezone.utc).isoformat(), "input_mode": mode,
        "inference": "3-state cost decoder; no GT access, fitting, or parameter search",
        "config": config, "config_sha256": sha256(args.config),
        "model_sha256": sha256(Path(__file__)),
        "inputs": [{"path": str(p.resolve()), "sha256": sha256(p)} for p in inputs],
        "limitations": ["Only nearest-team/missing observations in published-raw mode; no distances or speeds.",
                        "Long missing is only a dead-ball proxy; true control and dead ball are not directly observed.",
                        "Offline decoding uses future observations; public scores are pipeline checks only."],
    }
    args.out.mkdir(parents=True, exist_ok=True)
    write_json(args.out / "pre_evaluation_manifest.json", manifest)
    labels, costs, lengths, long_gap, objective = decode(observations, config)
    predictions = args.out / "predictions"
    predictions.mkdir(parents=True, exist_ok=True)
    dest = predictions / "model1.csv"
    baseline.write_submission(str(dest), STATES[labels])
    # Comparator is generated from the SAME observations, not a stale saved score.
    baseline.write_submission(str(predictions / "baseline_smooth50.csv"), baseline.smooth(observations, 50))
    folder = args.out / "diagnostics"
    folder.mkdir(parents=True, exist_ok=True)
    pd.DataFrame({"frame": np.arange(len(labels)),
                  "observation": np.where(observations == H, "H", np.where(observations == A, "A", "MISSING")),
                  "missing_run_frames": lengths, "long_missing_candidate": long_gap.astype(int),
                  "cost_H": costs[:, H], "cost_A": costs[:, A], "cost_D": costs[:, D],
                  "model1": STATES[labels]}).to_csv(folder / "inference_evidence.csv", index=False)
    manifest.update(prediction_sha256=sha256(dest), objective=objective,
                    label_counts={label: int(np.sum(labels == i)) for i, label in enumerate(STATES)},
                    n_frames=len(labels))
    write_json(args.out / "run_manifest.json", manifest)
    print(f"model1: {len(labels)} frames; input={mode}; objective={objective:.4f}")
    print(f"label counts: {manifest['label_counts']}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
