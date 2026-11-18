"""Prepare local DFL training targets and frame-aligned PathCRF episode tensors."""

from __future__ import annotations

import argparse
from dataclasses import dataclass
from pathlib import Path
import sys

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from run_pathcrf_dfl import ROOT, TRIAL, checkpoint_split, load_official_tracking
from transportmer_ha_data import contiguous_runs

WINDOW = 50
SAMPLE = 5
EVENT_RADIUS = 10
EVENT_TYPES = {"tackle", "interception", "ball_recovery"}
FEATURES = ("x", "y", "vx", "vy", "speed", "accel")
OUTSIDE = ("out_left", "out_right", "out_bottom", "out_top")
TEAM_CODE = {"home": 0, "away": 1}


def prepare_match(upstream, raw_root, match, cache, train):
    from datatools import utils

    output = cache / f"{match}.parquet"
    if output.exists():
        return output

    tracking = load_official_tracking(raw_root, match)
    tracking["ball_state"] = tracking["ball_state"].astype(str).str.lower()
    tracking, _ = utils.label_frames_and_episodes(tracking)
    tracking["episode_id"] = 0
    next_episode = 1
    for _, half in tracking.groupby("period_id", sort=True):
        alive = half["ball_state"].eq("alive").to_numpy()
        for start, stop in contiguous_runs(alive, half["frame_id"].to_numpy()):
            tracking.loc[half.index[start:stop], "episode_id"] = next_episode
            next_episode += 1
    if train:
        from datatools.preprocess import Preprocessor

        events = read_synced(upstream, match).drop(columns="episode_id")
        events = events.loc[events["frame_id"].isin(tracking["frame_id"])]
        tracking[["timestamp", "ball_x", "ball_y"]] = tracking[["timestamp", "ball_x", "ball_y"]].round(2)
        processor = Preprocessor(events, tracking)
        tracking = processor.merge_tracking_poss(processor.detect_touches())
    tracking, _ = utils.label_phases(tracking)
    tracking = utils.calculate_running_features(tracking)
    if not train:
        public = ["frame_id", "period_id", "timestamp", "phase_id", "episode_id", "ball_state"]
        players = [col for col in tracking if col.startswith(("home_", "away_"))]
        tracking = tracking[public + players]
    temporary = output.with_suffix(".tmp")
    tracking.to_parquet(temporary, index=False)
    temporary.replace(output)
    print(f"{match}: cached {len(tracking):,} frames at {output}", flush=True)
    return output


def read_synced(upstream, match):
    return pd.read_parquet(upstream / "data/sportec_synced" / f"{match}.parquet")


def read_truth(cache, match):
    data = pd.read_parquet(
        cache / f"{match}_tracking.parquet",
        columns=["frame_id", "period", "ball_state", "ball_owning_team"],
    )
    data["alive"] = data["ball_state"].eq(1)
    data["label"] = data["ball_owning_team"].map(TEAM_CODE).fillna(-1).astype(np.int8)
    data.loc[~data["alive"], "label"] = 2
    return data.set_index("frame_id", verify_integrity=True)


def transition_targets(labels, sample_rows):
    targets = np.full(len(sample_rows), -1, dtype=np.int64)
    for i, (start, stop) in enumerate(zip(sample_rows[:-1], sample_rows[1:], strict=True), 1):
        interval = labels[start:stop+1]
        if np.any(interval < 0):
            continue
        changes = np.count_nonzero(interval[1:] != interval[:-1])
        if changes <= 1:
            targets[i] = 0 if not changes else (1 if interval[-1] == 1 else 2)
    return targets


def event_weights(frames, labels, sample_rows, targets, synced):
    from evaluation.possession_metrics import match_transitions

    weights = np.ones(len(sample_rows), dtype=np.float32)
    changes = np.flatnonzero((labels[1:] != labels[:-1]) & (labels[1:] >= 0) & (labels[:-1] >= 0)) + 1
    actual = [(int(frames[row]), int(labels[row-1]), int(labels[row])) for row in changes]
    events = synced.loc[synced["frame_id"].between(frames[0], frames[-1])]
    proposed = [(int(row.frame_id), 1 - row.team, row.team) for row in events.itertuples(index=False)]
    matches, _, unmatched = match_transitions(actual, proposed, EVENT_RADIUS, typed="pair")
    for event, _ in matches:
        i = np.searchsorted(frames[sample_rows], event[0])
        if targets[i] > 0:
            weights[i] = 1.5
    for event in unmatched:
        row = event[0] - frames[0]
        if row < EVENT_RADIUS or row + EVENT_RADIUS >= len(frames):
            continue
        labels_near = labels[row-EVENT_RADIUS:row+EVENT_RADIUS+1]
        if labels_near[0] >= 0 and (labels_near == labels_near[0]).all():
            i = abs(frames[sample_rows] - event[0]).argmin()
            if targets[i] == 0:
                weights[i] = 1.5
    return weights


@dataclass
class Episode:
    frames: np.ndarray
    sample_rows: np.ndarray
    nodes: tuple
    features: np.ndarray
    edges: np.ndarray | None
    targets: np.ndarray | None
    weights: np.ndarray | None


def build_episodes(path, truth, synced=None):
    import inference as official
    from datatools import utils

    raw = pd.read_parquet(path).set_index("frame_id", verify_integrity=True)
    truth_aligned = truth.reindex(raw.index)
    if not np.array_equal(raw["ball_state"].eq("alive"), truth_aligned["alive"]):
        raise ValueError(f"Official and DFL cache ball status differ for {path.stem}")
    if not np.array_equal(truth.index[truth["alive"]], raw.index[raw["ball_state"].eq("alive")]):
        raise ValueError(f"Official and DFL cache in-play frame axes differ for {path.stem}")
    if synced is not None:
        synced = synced.loc[synced["spadl_type"].isin(EVENT_TYPES) & synced["frame_id"].notna()].copy()
        synced["team"] = synced["player_id"].str.split("_").str[0].map(TEAM_CODE)
        synced = synced.dropna(subset="team").astype({"team": int}).sort_values("frame_id")
    live = official._add_outside_nodes(raw.loc[raw["episode_id"] > 0])
    phase_keepers = {
        phase: utils.detect_keepers(rows, left_first=True)
        for phase, rows in live.groupby("phase_id", sort=False)
        if phase != 0
    }
    episodes = []
    skipped_frames = 0
    for (phase, _), rows in live.groupby(["phase_id", "episode_id"], sort=False):
        if phase == 0:
            if len(rows) < 2:
                skipped_frames += len(rows)
                continue
            left_gk, right_gk = utils.detect_keepers(rows, left_first=True)
        else:
            left_gk, right_gk = phase_keepers[phase]
        left, right = left_gk.split("_")[0], right_gk.split("_")[0]
        columns = rows.dropna(axis=1, how="all").columns
        left_nodes = sorted(col[:-2] for col in columns if col.startswith(f"{left}_") and col.endswith("_x"))
        right_nodes = sorted(col[:-2] for col in columns if col.startswith(f"{right}_") and col.endswith("_x"))
        names = left_nodes + right_nodes + list(OUTSIDE)
        cols = [f"{node}_{feature}" for node in names for feature in FEATURES]
        numbers = rows[cols].to_numpy(dtype=np.float32).reshape(len(rows), len(names), len(FEATURES))
        if not np.isfinite(numbers).all():
            skipped_frames += len(rows)
            continue
        frames = rows.index.to_numpy()
        if np.any(np.diff(frames) != 1):
            raise ValueError(f"Non-contiguous episode for {path.stem}")
        sample = np.arange(0, len(rows), SAMPLE)
        if sample[-1] != len(rows) - 1:
            sample = np.append(sample, len(rows) - 1)
        indicators = official._build_node_indicators(names, left, right, left_gk, right_gk)
        context = np.broadcast_to(indicators, (len(sample), len(names), 2))
        features = np.concatenate([context, numbers[sample]], axis=-1)
        features, _, names = official._pad_nodes(features, names, left, right, 11)

        edges, targets, weights = None, None, None
        if synced is not None:
            if len(frames) < WINDOW * SAMPLE or rows["player_id"].dropna().empty:
                continue
            node_index = {name: i for i, name in enumerate(names)}
            node_index["goal_left"] = node_index["out_left"]
            node_index["goal_right"] = node_index["out_right"]
            previous, following = official._downsample_possession(rows, sample, node_index)
            edges = np.stack([previous, following], axis=-1)
            edges = np.where(np.isfinite(edges), edges, -1).astype(np.int64)
            labels = truth_aligned.loc[frames, "label"].to_numpy()
            targets = transition_targets(labels, sample)
            weights = event_weights(frames, labels, sample, targets, synced)
        episodes.append(Episode(frames, sample, tuple(names), features, edges, targets, weights))
    return episodes, skipped_frames


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--upstream", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--raw", type=Path, default=ROOT / "data/external/dfl/raw")
    parser.add_argument("--split", choices=("development", "test"), default="development")
    args = parser.parse_args()
    args.output.mkdir(parents=True, exist_ok=True)
    sys.path.insert(0, str(args.upstream.resolve()))
    from models.utils import load_trial_args

    split = checkpoint_split(args.upstream, load_trial_args(str(args.upstream / f"saved/{TRIAL}")))
    matches = split["train"] + [split["validation"]] if args.split == "development" else [split["test"]]
    for match in matches:
        prepare_match(args.upstream, args.raw, match, args.output, match in split["train"])


if __name__ == "__main__":
    main()
