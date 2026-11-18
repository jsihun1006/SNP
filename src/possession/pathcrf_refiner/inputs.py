"""Frame-aligned inputs, frozen predictions, and selection of refinement patches."""

from dataclasses import dataclass
from itertools import product
import json
from pathlib import Path
from time import perf_counter
from types import SimpleNamespace

from prepare_tracking import ROOT, prepare_tracking

from dfl_scene_data import EVENTS, Half as SceneHalf, make_scenes, scene_features
from features import match_from_dfl
from run_dfl_scene_transformers import SceneTransformer, predict, predictions, windows
from run_pathcrf_dfl import CHECKPOINT, TRIAL, team_predictions
from transportmer_ha_data import augment_ball, contiguous_runs, seed_for

import numpy as np
from scipy.signal import savgol_filter
import pandas as pd
import torch

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


def atomic_npz(path, **values):
    temporary = path.with_suffix(".tmp.npz")
    np.savez_compressed(temporary, **values)
    temporary.replace(path)


def frozen_predictions(halves, tracking, roster, checkpoint, fit_path, device):
    """Run the existing full-half model without using event or label inputs."""
    geometry = match_from_dfl(tracking, roster)
    fit = json.loads(fit_path.read_text())
    scenes_halves = []
    for number, frame in halves:
        use = geometry.period == number
        np.testing.assert_array_equal(geometry.frame[use], frame)
        coords = scene_features(geometry.ball[use], geometry.players[use], geometry.team,
                                geometry.pitch, frame, tracking.loc[use, "timestamp_s"].to_numpy())
        scenes_halves.append(SceneHalf(frozen_predictions.__name__, number, frame,
                                      np.full(len(frame), -1, np.int8), coords,
                                      {event: np.empty(0, dtype=np.int64) for event in EVENTS}))
    scene_model = SceneTransformer(3).to(device)
    scene_model.load_state_dict(torch.load(checkpoint, map_location=device, weights_only=True))
    scenes = make_scenes(scenes_halves, "full")
    prepared = windows(scenes, np.asarray(fit["normalization_mean"]), np.asarray(fit["normalization_scale"]))
    logits = predict(scene_model, scenes, prepared, device)
    return predictions(scenes_halves, scenes, logits, *fit["decoder"])


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
        source = tracking.assign(ball_state="alive", episode_id=1)
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
    weights = torch.load(trial / "model" / CHECKPOINT, map_location=device, weights_only=True)
    model.load_state_dict(weights)
    with torch.inference_mode():
        carrier, _, edges, _ = inference(model, prepared.loc[prepared["episode_id"] > 0],
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
                           out=np.ones(len(mass), dtype=np.float64), where=mass > 0)
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


def load_match(args, match, source_device, oracle=False):
    started = perf_counter()
    source = prepare_tracking(args.upstream, args.raw, args.sources, match)
    official = pd.read_parquet(source).set_index("frame_id", verify_integrity=True)
    track = pd.read_parquet(args.cache / f"{match}_tracking.parquet")
    roster = pd.read_parquet(args.cache / f"{match}_meta.parquet")
    players = sorted(c[:-2] for c in official if c.startswith(("home_", "away_")) and c.endswith("_x"))
    names = players + ["ball"]
    team = np.array([HA[name.split("_")[0]] for name in players] + [2], np.int8)
    confirmed = {}
    for row in roster.dropna(subset=["jersey"]).itertuples(index=False):
        pos = str(row.position).strip()
        name = f"{row.team}_{int(row.jersey)}"
        if pos.lower() not in {"", "unknown", "none", "nan"}:
            confirmed[name] = 1 if pos.lower() in {"gk", "goalkeeper"} else 0
    # An explicitly identified role scores as 0 or 1; other positions lack authority.
    # A half-by-half lineup change remains tied to the player's original track.
    keeper = np.asarray([confirmed.get(name, -1) for name in players] + [-1], np.int8)
    halves, labels, times, points = [], [], [], []
    for number in (1, 2):
        rows = track.loc[track["period"].eq(number)]
        frame = rows["frame_id"].to_numpy(np.int64)
        time = rows["timestamp_s"].to_numpy(np.float64)
        if len(frame) == 0 or (np.diff(frame) <= 0).any() or (np.diff(time) <= 0).any():
            raise ValueError(f"Invalid DFL axis {match} H{number}")
        aligned = official.reindex(frame)
        covered = aligned["period_id"].notna().to_numpy()
        if not (aligned.loc[covered, "period_id"] == number).all():
            raise ValueError(f"Official half differs from DFL cache for {match}")
        np.testing.assert_allclose(aligned.loc[covered, "timestamp"], time[covered], rtol=0, atol=1e-6)
        np.testing.assert_array_equal(aligned.loc[covered, "ball_state"].astype(str).str.lower().eq("alive"),
                                      rows.loc[covered, "ball_state"].eq(1))
        xy = aligned[[f"{name}_{axis}" for name in names for axis in ("x", "y")]]
        xy = xy.to_numpy(np.float32).reshape(len(frame), len(names), 2)
        inplay = rows["ball_state"].eq(1).to_numpy()
        truth = rows["ball_owning_team"].map(HA).fillna(-1).to_numpy(np.int8)
        truth[~inplay] = 2
        halves.append((number, frame))
        times.append(time)
        labels.append(truth)
        points.append(xy)

    args.frozen_cache.mkdir(parents=True, exist_ok=True)
    mode = "oracle" if oracle else "pred"
    path = args.frozen_cache / f"{match}_{mode}.npz"
    if not path.exists():
        run = perf_counter()
        full = frozen_predictions(halves, track, roster, args.dead_scene / "full/best.pt",
                                  args.dead_scene / "full/fit.json", source_device)
        alive = [truth != 2 for truth in labels] if oracle else [values != 2 for values in full]
        cache = args.official_prepared / f"{match}.parquet" if args.official_prepared else None
        coarse = coarse_predictions(official.reset_index(), halves, alive, full, args.upstream,
                                    source_device, cache)
        save = {}
        for (number, frame), full_half, (base, margin, carrier, fallback) in zip(halves, full, coarse, strict=True):
            for name, value in dict(frame=frame, full=full_half, base=base, margin=margin,
                                    carrier=carrier, fallback=np.asarray(fallback)).items():
                save[f"h{number}_{name}"] = value
        atomic_npz(path, **save)
        print(f"{match} {mode}: frozen D and 5fps {perf_counter() - run:.1f}s", flush=True)
    result = []
    with np.load(path) as saved:
        for (number, frame), time, truth, xy in zip(halves, times, labels, points, strict=True):
            def read(kind):
                return saved[f"h{number}_{kind}"]
            np.testing.assert_array_equal(read("frame"), frame)
            baseline = read("base")
            alive = baseline != 2
            distances, closing = contention(xy, names, team, read("carrier"), frame, alive)
            signals = np.column_stack((read("margin"), distances, closing))
            result.append(Half(f"{match}_H{number}", match, number, frame, time, truth, xy, team, keeper,
                               read("full") == 2, dict(base=baseline, fallback=int(read("fallback"))), signals))
    print(f"{match} {mode}: aligned {sum(len(h.frame) for h in result):,} frames in "
          f"{perf_counter() - started:.1f}s", flush=True)
    return result


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


def calibration(halves, config):
    active = sum(np.count_nonzero(h.coarse["base"] != 2) for h in halves)
    transitions = [(np.diff(h.labels, prepend=-1) != 0) & np.isin(h.labels, (0, 1)) &
                   np.r_[False, np.isin(h.labels[:-1], (0, 1)) & (np.diff(h.frame) == 1)] for h in halves]
    actual = sum(int(t.sum()) for t in transitions)
    limit = config["validation_gate_max_fraction"]
    scored = []
    for threshold in product((-1, .05, .1, .18, .25, .35, .50), (-1, 1.5, 2.5, 4, 6), (.12, .35, .65)):
        masks = [gate(h, threshold, config["gate_margin_frames"]) for h in halves]
        size = sum(int(mask.sum()) for mask in masks)
        hit = sum(int((mask & change).sum()) for mask, change in zip(masks, transitions, strict=True))
        scored.append({"threshold": threshold, "frames": size, "fraction": size / max(1, active),
                       "covered_transitions": hit, "total_transitions": actual})
    choices = [item for item in scored if item["fraction"] <= limit]
    if not choices:
        raise ValueError(f"No gate can meet {limit:.0%} validation coverage with the current margin")
    best = max(choices, key=lambda item: (item["covered_transitions"], -item["frames"]))
    return best, scored


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
        gaps = {k: 1 for k in range(config["synthetic_gap_min_frames"], config["synthetic_gap_max_frames"] + 1)}
        xy[:, -1], report = augment_ball(
            xy[:, -1], contiguous_runs(playable, half.frame),
            {"missing_fraction": config["ball_missing_fraction"], "gap_lengths_frames": gaps},
            np.random.default_rng(seed_for(config["seed"], half.key)), 0.0,
        )
        n = report["in_play"]
        report["realized_missing_fraction"] = (report["native_missing"] + report["masked"]) / max(1, n)
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
