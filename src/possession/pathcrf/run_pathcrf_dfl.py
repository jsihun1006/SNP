"""Evaluate the official PathCRF checkpoint on its untouched DFL match."""

from __future__ import annotations

import argparse
import gzip
import hashlib
import json
from pathlib import Path
import shutil
import subprocess
import sys
from tempfile import TemporaryDirectory
from time import perf_counter

import numpy as np
import pandas as pd
import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from run_borghesi_tcn_dfl import ROOT, load_match, scores, write_json
from run_borghesi_transition_dfl import evaluate
from transportmer_ha_metrics import metrics

TRIAL = 140
CHECKPOINT = "state_dict_best_acc.pt"


def checkpoint_split(upstream, trial_args):
    match_ids = sorted(p.stem for p in (upstream / "data/sportec_synced").glob("*.parquet"))
    if len(match_ids) != 7 or trial_args.get("sources") != ["sportec"] or not trial_args.get("lomocv"):
        raise ValueError("Expected the official 7-match Sportec leave-one-match-out checkpoint.")
    test_index = trial_args["cv_fold_idx"]
    validation_index = (test_index - 1) % len(match_ids)
    return {
        "test": match_ids[test_index], "validation": match_ids[validation_index],
        "train": [match for i, match in enumerate(match_ids) if i not in (test_index, validation_index)],
    }


def load_official_tracking(raw_root, match_id):
    from datatools.sportec_data import SportecData

    def source(pattern):
        matches = sorted(raw_root.rglob(f"{pattern}*{match_id}.xml*"))
        if not matches:
            raise FileNotFoundError(f"Missing {pattern} XML for {match_id} in {raw_root}")
        return matches[0]

    raw_meta = source("DFL_02_01_matchinformation")
    raw_tracking = source("DFL_04_03_positions")
    with TemporaryDirectory() as temporary:
        meta = Path(temporary) / "metadata.xml"
        if raw_meta.suffix == ".gz":
            with gzip.open(raw_meta, "rb") as reader, meta.open("wb") as writer:
                shutil.copyfileobj(reader, writer)
        else:
            meta = raw_meta
        lineup = SportecData.load_lineup_data(str(meta))
        _, tracking = SportecData.load_tracking_data(str(raw_tracking), str(meta), lineup)

    return tracking.reset_index(drop=True)


def prepare_official_input(tracking):
    from datatools import utils

    tracking = tracking.copy()
    tracking["ball_state"] = tracking["ball_state"].astype(str).str.lower()
    tracking, _ = utils.label_frames_and_episodes(tracking)
    tracking, _ = utils.label_phases(tracking)
    tracking = utils.calculate_running_features(tracking)
    raw_states = tracking.set_index("frame_id")["ball_state"]
    public_columns = ["frame_id", "period_id", "timestamp", "phase_id", "episode_id"]
    model_columns = [name for name in tracking.columns if name.startswith(("home_", "away_"))]
    return tracking.loc[tracking["episode_id"] > 0, public_columns + model_columns], raw_states


def load_official_input(raw_root, match_id):
    return prepare_official_input(load_official_tracking(raw_root, match_id))


def team_predictions(half, micro):
    def team(nodes):
        return (nodes.fillna("").str.split("_", n=1).str[0]
                .map({"home": scores.H, "away": scores.A}))

    aligned = micro.reindex(half.frame)
    from_team = team(aligned["edge_src"])
    to_team = team(aligned["edge_dst"])
    team_evidence = from_team.fillna(to_team).to_numpy()
    missing_edge = (aligned["edge_src"].isna() | aligned["edge_dst"].isna()).to_numpy()
    hard = np.full(len(half.frame), scores.D, dtype=np.int8)
    for start, stop in half.episodes:
        evidence = pd.Series(team_evidence[start:stop]).ffill().bfill().fillna(scores.X)
        hard[start:stop] = evidence.to_numpy(dtype=np.int8)
    diagnostic = {
        "half": half.key,
        "inplay_frames": int(half.alive.sum()),
        "inplay_missing_decoded_edge": int((half.alive & missing_edge).sum()),
        "inplay_without_team_after_fill": int((half.alive & (hard == scores.X)).sum()),
    }
    return hard, aligned, diagnostic


def nearest_probability(half):
    distance = np.linalg.norm((half.xy[:, 1:] - half.xy[:, :1]) * np.asarray(half.pitch), axis=-1)
    distance = np.where(np.isfinite(distance), distance, np.inf)
    nearest_team = (distance[:, half.kind[1:] == scores.A].min(axis=1)
                    < distance[:, half.kind[1:] == scores.H].min(axis=1)).astype(float)
    observed = np.isfinite(distance.min(axis=1))
    for start, stop in half.episodes:
        known = np.flatnonzero(observed[start:stop])
        nearest_team[start:stop] = (
            np.interp(np.arange(stop - start), known, nearest_team[start + known])
            if known.size else 0.5
        )
    return nearest_team[:, None]


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--upstream", type=Path, required=True, help="Official PathCRF git checkout.")
    parser.add_argument("--cache", type=Path, default=ROOT / "data/external/dfl")
    parser.add_argument("--raw", type=Path, default=ROOT / "data/external/dfl/raw")
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--threads", type=int, default=4)
    args = parser.parse_args()
    upstream = args.upstream.resolve()
    if not (upstream / "inference.py").is_file():
        parser.error("--upstream must point to the official PathCRF source tree.")
    sys.path.insert(0, str(upstream))
    from inference import inference
    from models.utils import build_model, load_trial_args

    torch.set_num_threads(args.threads)
    started = perf_counter()
    trial_path = upstream / f"saved/{TRIAL}"
    trial_args = load_trial_args(str(trial_path))
    checkpoint = trial_path / "model" / CHECKPOINT
    split = checkpoint_split(upstream, trial_args)
    match = split["test"]
    args.output.mkdir(parents=True, exist_ok=True)
    manifest = {
        "pathcrf_commit": subprocess.check_output(
            ["git", "rev-parse", "HEAD"], cwd=upstream, text=True,
        ).strip(),
        "checkpoint_trial": TRIAL,
        "checkpoint_name": CHECKPOINT, "checkpoint_sha256": hashlib.sha256(checkpoint.read_bytes()).hexdigest(),
        "split": split, "trial_seed": trial_args["seed"], "device": "cpu",
        "input_fps": trial_args["target_fps"], "evaluation_fps": 25, "window_seconds": trial_args["window_seconds"],
        "deadball_mask": "DFL ground truth", "ball_as_input": False, "correct_episode_lasts": False,
        "team_projection": "source player's team, then destination if source is outside; fill within in-play run",
        "unresolved_policy": "X, scored as missing H/A",
    }
    write_json(args.output / "manifest.json", manifest)

    model = build_model(trial_args, torch.device("cpu"))
    model.load_state_dict(torch.load(checkpoint, map_location="cpu", weights_only=True))
    print(f"PathCRF held-out match {match}, loading local DFL XML and cache", flush=True)
    model_input, raw_states = load_official_input(args.raw, match)
    halves = [item.half for item in load_match(match, args.cache)]
    for half in halves:
        is_live_raw = raw_states.reindex(half.frame).eq("alive").to_numpy()
        if not np.array_equal(is_live_raw, half.alive):
            raise ValueError(f"Official and local DFL loaders disagree on in-play frames of {half.key}")
    preparation = perf_counter() - started
    print(f"Prepared in {preparation:.1f}s, inferring from player features", flush=True)
    inference_start = perf_counter()
    _, _, micro, _ = inference(
        model, model_input, use_crf=True, ball_feature=False,
        evaluate=False, correct_episode_lasts=False,
        window_seconds=trial_args["window_seconds"], sample_freq=round(25 / trial_args["target_fps"]),
    )
    inference_seconds = perf_counter() - inference_start
    if not {"edge_src", "edge_dst"}.issubset(micro.columns):
        raise RuntimeError("Official inference returned no decoded possession edges.")

    path_predictions, edges, coverage = zip(*(team_predictions(half, micro) for half in halves), strict=True)
    nearest_result, nearest_predictions = evaluate(
        halves, [nearest_probability(half) for half in halves], "A", 4.0,
    )
    path_result = metrics(halves, path_predictions)
    path_result["ha_frame_accuracy"] = float(
        sum(np.count_nonzero((pred == half.labels) & half.alive)
            for half, pred in zip(halves, path_predictions, strict=True))
        / sum(np.count_nonzero(half.alive & (half.labels >= 0)) for half in halves)
    )
    for half, path_pred, near_pred, aligned in zip(halves, path_predictions, nearest_predictions, edges, strict=True):
        pd.DataFrame({
            "frame": half.frame, "half": half.number, "truth": half.labels, "inplay": half.alive,
            "edge_src": aligned["edge_src"].to_numpy(), "edge_dst": aligned["edge_dst"].to_numpy(),
            "pathcrf": path_pred, "nearest_prob35": near_pred,
        }).to_parquet(args.output / f"{half.key}.parquet", index=False)
    result = {
        "match": match, "pathcrf": path_result, "nearest_prob35": nearest_result,
        "coverage": coverage, "prepare_seconds": preparation,
        "inference_seconds": inference_seconds, "total_seconds": perf_counter() - started,
    }
    write_json(args.output / "results.json", result)
    print(json.dumps(result, indent=2), flush=True)


if __name__ == "__main__":
    main()
