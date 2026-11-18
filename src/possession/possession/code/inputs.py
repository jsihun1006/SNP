"""Frame-aligned inputs, frozen predictions, and selection of refinement patches."""

from dataclasses import dataclass
from itertools import product
import json
from pathlib import Path
from time import perf_counter
from types import SimpleNamespace



import numpy as np
from scipy.signal import savgol_filter
import pandas as pd
import torch

from data_utils import contiguous_runs
from locations import ARTIFACT

CHECKPOINT = "state_dict_best_acc.pt"
TRIAL = 140
N_FEATURES = 14
HA = {"home": 0, "away": 1}
FEATURES = ("x", "y", "vx", "vy", "speed", "accel")


@dataclass
class Half:
    key: str
    match: str
    number: int
    frame: np.ndarray
    time: np.ndarray
    labels: np.ndarray
    xy: np.ndarray
    team: np.ndarray
    keeper: np.ndarray
    dead_prediction: np.ndarray
    coarse: dict
    signals: np.ndarray
    features: np.ndarray | None = None






def team_predictions(half, micro):
    def team(nodes):
        return (nodes.fillna("").str.split("_", n=1).str[0]
                .map({"home": 0, "away": 1}))

    aligned = micro.reindex(half.frame)
    from_team = team(aligned["edge_src"])
    to_team = team(aligned["edge_dst"])
    team_evidence = from_team.fillna(to_team).to_numpy()
    missing_edge = (aligned["edge_src"].isna() | aligned["edge_dst"].isna()).to_numpy()
    hard = np.full(len(half.frame), 2, dtype=np.int8)
    for start, stop in half.episodes:
        evidence = pd.Series(team_evidence[start:stop]).ffill().bfill().fillna(3)
        hard[start:stop] = evidence.to_numpy(dtype=np.int8)
    diagnostic = {
        "half": half.key,
        "inplay_frames": int(half.alive.sum()),
        "inplay_missing_decoded_edge": int((half.alive & missing_edge).sum()),
        "inplay_without_team_after_fill": int((half.alive & (hard == 3)).sum()),
    }
    return hard, aligned, diagnostic

def coarse_predictions(tracking, halves, alive, fallback, upstream, device, official_cache=None):
    """5 Hz official PathCRF using only the supplied in-play mask."""
    from datatools import utils
    from inference import inference
    from models.utils import build_model, load_trial_args

    cols = ["frame_id", "period_id", "timestamp"]
    names = [c for c in tracking if c.startswith(("home_", "away_")) and c.endswith("_x")]
    if official_cache is not None and official_cache.is_file():
        prepared = pd.read_parquet(official_cache, columns=cols + [
            f"{col[:-2]}_{feature}" for col in names for feature in FEATURES
        ])
    else:
        source = tracking.assign(ball_state="alive", episode_id=1, ball_owning_team_id="unknown")
        source, _ = utils.label_phases(source)
        prepared = utils.calculate_running_features(source).drop(columns=[
            "ball_state", "ball_owning_team_id", "phase_id", "episode_id",
        ])
        prepared = prepared[cols + [c for c in prepared if c.startswith(("home_", "away_"))]]
    prepared, _ = utils.label_phases(prepared.assign(ball_state="alive"))
    prepared["episode_id"] = 0
    index = pd.Index(prepared["frame_id"])
    count = 1
    scopes = []
    for (number, frame), keep in zip(halves, alive, strict=True):
        runs = contiguous_runs(keep, frame)
        for a, b in runs:
            rows = index.get_indexer(frame[a:b])
            prepared.loc[rows[rows >= 0], "episode_id"] = count
            count += 1
        scopes.append(SimpleNamespace(key=f"H{number}", frame=frame, alive=keep, episodes=runs))
    trial = upstream / f"saved/{TRIAL}"
    settings = load_trial_args(str(trial))
    model = build_model(settings, device)
    weights = torch.load(ARTIFACT / "weights/pathcrf_trial140.pt", map_location=device, weights_only=True)
    model.load_state_dict(weights)
    selected = prepared.loc[prepared["episode_id"] > 0].copy()
    # ATD observations can vanish during all in-play rows of an otherwise valid phase.
    # Such phases have no keeper/team input and must use the explicit fallback.
    for phase, group in selected.groupby("phase_id"):
        for prefix in ("home_", "away_"):
            columns = [c for c in group if c.startswith(prefix) and c.endswith("_x")]
            if not columns or not group[columns].notna().any().any():
                selected.loc[group.index, "phase_id"] = 0
                break
    with torch.inference_mode():
        carrier, _, edges, _ = inference(model, selected,
                                         evaluate=False, use_crf=True, ball_feature=False,
                                         correct_episode_lasts=False, window_seconds=10, sample_freq=5)
    if not {"edge_src", "edge_dst"}.issubset(edges.columns):
        edges = pd.DataFrame(columns=["edge_src", "edge_dst"], dtype=object)
    results = []
    for scope, fallback_half in zip(scopes, fallback, strict=True):
        labels, _, _ = team_predictions(scope, edges)
        use_fallback = scope.alive & ~np.isin(labels, (0, 1))
        labels[use_fallback] = fallback_half[use_fallback]
        # No H/A output from the D head on frames opened only by the oracle mask.
        labels[scope.alive & ~np.isin(labels, (0, 1))] = 0
        if carrier is None:
            values = pd.DataFrame(index=scope.frame)
        else:
            values = carrier.reindex(scope.frame).fillna(0)
        groups = [values.loc[:, values.columns.str.startswith(prefix)]
                  for prefix in ("home_", "away_")] if len(values.columns) else [values, values]
        masses = np.column_stack([group.sum(axis=1).to_numpy() for group in groups])
        mass = masses.sum(axis=1)
        margin = np.divide(abs(masses[:, 0] - masses[:, 1]), mass,
                           out=np.zeros(len(mass), dtype=np.float64), where=mass > 0)
        options = values[[c for c in values if c.startswith(("home_", "away_"))]]
        best = np.full(len(scope.frame), "", dtype="U32")
        if len(options.columns):
            probability = options.to_numpy()
            row = probability.argmax(axis=1)
            best = np.asarray(options.columns, dtype="U32")[row]
            best[probability.max(axis=1) <= 0] = ""
        results.append((labels, margin.astype(np.float32), best, int(use_fallback.sum())))
    return results


def contention(xy, names, team, carrier, frame, alive):
    """Nearest opponent to the predicted possessor, plus closing speed over 0.2 s."""
    lookup = {name: i for i, name in enumerate(names)}
    index = np.fromiter((lookup.get(name, -1) for name in carrier), np.intp, count=len(carrier))
    distances = np.full(len(xy), np.inf, dtype=np.float32)
    for lo in range(0, len(xy), 4096):
        hi = min(len(xy), lo + 4096)
        choice = index[lo:hi]
        points = xy[lo:hi, :-1]
        position = points[np.arange(hi - lo), choice]
        difference = np.linalg.norm(points - position[:, None], axis=-1)
        opponent = team[:-1] != team[choice, None]
        distances[lo:hi] = np.where(opponent & np.isfinite(difference), difference, np.inf).min(axis=1)
    distances[(index < 0) | ~alive] = np.inf
    closing = np.zeros(len(xy), np.float32)
    for lo, hi in contiguous_runs(alive, frame):
        if hi - lo <= 5:
            continue
        first, last = distances[lo:hi - 5], distances[lo + 5:hi]
        valid = np.isfinite(first) & np.isfinite(last) & (index[lo:hi - 5] == index[lo + 5:hi])
        np.subtract(first, last, out=closing[lo + 5:hi], where=valid)
    return distances, closing



def gate(half, threshold, margin_frames):
    margin, distance, closing = half.signals.T
    base = half.coarse["base"]
    active = base != 2
    uncertain, far, approach = threshold
    patch = np.zeros(len(base), dtype=bool)
    for first, end in contiguous_runs(active, half.frame):
        flags = ((margin[first:end] <= uncertain) |
                 ((distance[first:end] <= far) &
                  ((closing[first:end] >= approach) | (distance[first:end] <= 0.6 * far))))
        flips = np.flatnonzero(base[first + 1:end] != base[first:end - 1]) + 1
        flags[flips] = True
        anchors = np.flatnonzero(flags)
        cuts = np.zeros(end - first + 1, np.int32)
        np.add.at(cuts, np.maximum(0, anchors - margin_frames), 1)
        np.add.at(cuts, np.minimum(end - first, anchors + margin_frames + 1), -1)
        patch[first:end] = cuts[:-1].cumsum() > 0
    return patch


def motion(xy, frame, time, playable, filter_velocity, filter_acceleration):
    seen = np.isfinite(xy).all(axis=-1)
    velocity = np.full_like(xy, np.nan)
    acceleration = np.full(xy.shape[:-1], np.nan, dtype=np.float32)
    for node in range(xy.shape[1]):
        for first, stop in contiguous_runs(seen[:, node] & playable, frame):
            if stop - first < 2:
                continue
            dt = np.diff(time[first:stop]).astype(np.float32)
            speed = np.diff(xy[first:stop, node], axis=0) / dt[:, None]
            player = node != xy.shape[1] - 1
            if player and len(speed) >= filter_velocity:
                speed = savgol_filter(speed, filter_velocity, 2, axis=0)
            velocity[first + 1:stop, node] = speed
            if player and len(speed) > 1:
                change = np.diff(speed, axis=0) / dt[1:, None]
                if len(change) >= filter_acceleration:
                    change = savgol_filter(change, filter_acceleration, 2, axis=0)
                acceleration[first + 2:stop, node] = np.linalg.norm(change, axis=-1)
    return seen, velocity, acceleration


def make_features(half, config, training, hide_ball=False):
    playable = (half.labels != 2) if training else (half.coarse["base"] != 2)
    xy = half.xy.copy()
    report = None
    if training:
        raise ValueError("possession package is inference-only")
    if hide_ball:
        xy[:, -1] = np.nan
    seen, speed, accel = motion(xy, half.frame, half.time, playable,
                                config["player_velocity_filter_frames"], config["player_acceleration_filter_frames"])
    velocity_seen = np.isfinite(speed).all(axis=-1)
    accel_seen = np.isfinite(accel)
    norms = np.linalg.norm(speed, axis=-1)
    norms[:, -1] = 0
    shape = xy.shape[:-1]
    values = np.empty((*shape, N_FEATURES), dtype=np.float32)
    values[..., :2] = np.where(seen[..., None], xy, 0)
    values[..., 2:4] = np.nan_to_num(speed)
    values[..., 4] = np.nan_to_num(norms)
    values[..., 5] = np.nan_to_num(accel)
    values[..., 6] = seen
    values[..., 7] = velocity_seen
    values[..., 8] = accel_seen
    values[..., 9:12] = np.eye(3, dtype=np.float32)[half.team]
    values[..., 12] = half.keeper == 1
    values[..., 13] = half.keeper >= 0
    half.features = values
    return report
