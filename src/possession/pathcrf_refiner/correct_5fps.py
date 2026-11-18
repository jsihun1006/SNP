"""Compare six light corrections of a fixed 5 Hz DFL PathCRF."""

from __future__ import annotations

import argparse
from dataclasses import dataclass
import hashlib
import json
from pathlib import Path
import pickle
import sys
from time import perf_counter
from unittest.mock import patch

import numpy as np
import pandas as pd
from scipy.signal import find_peaks
import torch

from inputs import load_match
from run import ROOT, REVISION, digest, official_revision, write_json
from transportmer_ha_data import contiguous_runs
from transportmer_ha_metrics import metrics
from evaluation import possession_metrics as scores

TRAIN = ["J03WMX", "J03WN1", "J03WOH", "J03WOY", "J03WPY"]
VALIDATION, TEST = "J03WQQ", "J03WR9"
SEED = 100


@dataclass
class Episode:
    half: int
    rows: np.ndarray
    sampled: np.ndarray
    names: np.ndarray
    source: np.ndarray
    destination: np.ndarray
    incoming: np.ndarray
    allowed: np.ndarray
    event_type: np.ndarray
    ha_direction: np.ndarray
    emissions: np.ndarray
    previous: np.ndarray
    current: np.ndarray
    reference: np.ndarray
    features: np.ndarray | None = None
    supervision: np.ndarray | None = None
    event_supervision: np.ndarray | None = None


def team_of(names):
    return np.asarray([0 if str(n).startswith("home_") else 1 if str(n).startswith("away_") else -1
                       for n in names], dtype=np.int8)


def edge_categories(src, dst, incoming, allowed, teams):
    old_src, old_dst = src[incoming], dst[incoming]
    new_src = src[:, None]
    changed = incoming != np.arange(len(src))[:, None]
    kind = np.zeros(incoming.shape, np.int8)
    # 1 kick, 2 same-team receive, 3 opposing-team receive, 4 outside.
    kind[changed & (old_src == old_dst)] = 1
    reception = changed & (old_src != old_dst)
    both = (teams[old_src] >= 0) & (teams[new_src] >= 0)
    kind[reception & both & (teams[old_src] == teams[new_src])] = 2
    kind[reception & both & (teams[old_src] != teams[new_src])] = 3
    kind[changed & (teams[new_src] < 0)] = 4
    direction = np.zeros_like(kind)
    switch = changed & both & (teams[old_src] != teams[new_src])
    direction[switch] = np.broadcast_to(2 - teams[new_src], kind.shape)[switch]
    kind[~allowed] = 0
    direction[~allowed] = 0
    return kind, direction


def decode_many(ep, bias, event_bias=None):
    count = len(bias)
    length, states = ep.emissions.shape
    delta = np.broadcast_to(ep.emissions[0], (count, states)).copy()
    back = np.empty((length, count, states), np.int16)
    i = ep.incoming
    event_bias = np.zeros((count, 5), np.float32) if event_bias is None else event_bias
    event_cost = event_bias[:, ep.event_type]
    variant = np.arange(count)[:, None, None]
    state = np.arange(states)[None, :, None]
    for t in range(1, length):
        value = delta[:, i] + ep.previous[t - 1, i] + ep.current[t, None, :, None]
        value += bias[:, t, ep.ha_direction] + event_cost
        value = np.where(ep.allowed, value, -np.inf)
        pick = value.argmax(axis=-1)[..., None]
        delta = value[variant, state, pick].squeeze(-1) + ep.emissions[t]
        back[t] = i[state, pick].squeeze(-1)
    path = np.empty((count, length), np.int16)
    path[:, -1] = delta.argmax(axis=1)
    for t in range(length - 1, 0, -1):
        path[:, t - 1] = back[t, np.arange(count), path[:, t]]
    return path


class Capture:
    def __init__(self, official, halves, truth):
        self.official = official
        self.original_episode = official.inference_episode
        self.halves = halves
        self.truth = truth
        self.items = []

    def episode(self, *, model, ep_tracking, sample_freq, **kwargs):
        method = self.original_episode
        original_pad = self.official._pad_nodes
        original_decode = model.crf.decode
        names = None

        def capture_pad(*values, **options):
            nonlocal names
            output = original_pad(*values, **options)
            names = np.asarray(output[2])
            return output

        def capture_decode(emissions, embeddings, *, last_node_idx):
            if names is None or last_node_idx is not None:
                raise ValueError("Unexpected pinned PathCRF decoding interface")
            reference = original_decode(emissions, embeddings, last_node_idx=None)
            frames = ep_tracking.index.to_numpy()
            sample = np.arange(0, len(frames), sample_freq)
            if sample[-1] != len(frames) - 1:
                sample = np.r_[sample, len(frames) - 1]
            matches = [(index, pd.Index(h.frame).get_indexer(frames))
                       for index, h in enumerate(self.halves)
                       if int(ep_tracking["period_id"].iloc[0]) == h.number]
            index, rows = matches[0]
            if (rows < 0).any() or len(sample) != emissions.shape[1]:
                raise ValueError("Frame alignment differs from official PathCRF")
            crf = model.crf
            old_weights, new_weights = crf.linear.weight.split(crf.edge_embed_dim, dim=1)
            previous = torch.nn.functional.linear(embeddings[0], old_weights).squeeze(-1).cpu().numpy()
            current = torch.nn.functional.linear(embeddings[0], new_weights, crf.linear.bias).squeeze(-1).cpu().numpy()
            src, dst = crf.comp_src.cpu().numpy(), crf.comp_dst.cpu().numpy()
            incoming = crf.inc_idx.clamp_min(0).cpu().numpy()
            allowed = crf.inc_mask.cpu().numpy()
            kind, direction = edge_categories(src, dst, incoming, allowed, team_of(names))
            record = Episode(index, rows, sample, names, src, dst, incoming, allowed, kind, direction,
                             emissions[0].cpu().numpy(), previous, current, reference[0].cpu().numpy())
            if self.truth is not None:
                labelled = self.truth.reindex(frames)
                mapping = {name: n for n, name in enumerate(names)}
                mapping.update(goal_left=mapping["out_left"], goal_right=mapping["out_right"])
                a, b = self.official._downsample_possession(labelled, sample, mapping)
                states = {(int(u), int(v)): j for j, (u, v) in enumerate(zip(src, dst, strict=True))}
                record.event_supervision = np.asarray([
                    states.get((int(u), int(v)), -1) if np.isfinite(u) and np.isfinite(v) else -1
                    for u, v in zip(a, b, strict=True)
                ])
            self.items.append(record)
            return reference

        with patch.object(self.official, "_pad_nodes", capture_pad), patch.object(model.crf, "decode", capture_decode):
            return method(model=model, ep_tracking=ep_tracking, sample_freq=sample_freq, **kwargs)


def prepare(args, match):
    import inference as official
    from datatools import utils
    from models.utils import build_model, load_trial_args

    cached = args.output / "scores" / f"{match}.pkl"
    if cached.exists():
        with cached.open("rb") as f:
            return pickle.load(f)
    started = perf_counter()
    halves = load_match(args, match, torch.device("cpu"))
    raw = pd.read_parquet(args.sources / f"{match}.parquet")
    axes = ["frame_id", "period_id", "timestamp"]
    columns = [c for c in raw if c.startswith(("home_", "away_")) and c.endswith("_x")]
    columns = [f"{name[:-2]}_{feature}" for name in columns for feature in ("x", "y", "vx", "vy", "speed", "accel")]
    source = pd.read_parquet(args.official_prepared / f"{match}.parquet", columns=axes + columns)
    source, _ = utils.label_phases(source.assign(ball_state="alive"))
    source["episode_id"] = 0
    index = pd.Index(source["frame_id"])
    count = 1
    for half in halves:
        for a, b in contiguous_runs(half.coarse["base"] != 2, half.frame):
            loc = index.get_indexer(half.frame[a:b])
            source.loc[loc[loc >= 0], "episode_id"] = count
            count += 1
    trial = load_trial_args(str(args.upstream / "saved/140"))
    model = build_model(trial, torch.device("cpu"))
    model.load_state_dict(torch.load(args.upstream / "saved/140/model/state_dict_best_acc.pt",
                                    weights_only=True, map_location="cpu"))
    truth = None
    if match in TRAIN:
        truth = pd.read_parquet(args.official_prepared / f"{match}.parquet",
                                columns=["frame_id", "player_id"]).set_index("frame_id")
    capture = Capture(official, halves, truth)
    with patch.object(official, "inference_episode", capture.episode), torch.inference_mode():
        official.inference(model, source.loc[source["episode_id"] > 0], evaluate=False,
                           correct_episode_lasts=False, window_seconds=10, sample_freq=5)
    result = (halves, capture.items)
    cached.parent.mkdir(parents=True, exist_ok=True)
    temporary = cached.with_suffix(".tmp")
    with temporary.open("wb") as f:
        pickle.dump(result, f, protocol=5)
    temporary.replace(cached)
    print(f"{match}: captured {len(capture.items)} episodes in {perf_counter() - started:.1f}s", flush=True)
    return result


def parser():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("stage", choices=("prepare", "run"))
    p.add_argument("--upstream", type=Path, default=Path("/private/tmp/pathcrf-official-dfl"))
    p.add_argument("--cache", type=Path, default=ROOT / "data/external/dfl")
    p.add_argument("--raw", type=Path, default=ROOT / "data/external/dfl/raw")
    p.add_argument("--sources", type=Path, default=ROOT / "data/transition_final/source")
    p.add_argument("--official-prepared", type=Path, default=ROOT / "data/pathcrf_dfl/transition_head_v1/prepared")
    p.add_argument("--dead-scene", type=Path, default=ROOT / "data/pathcrf_dfl/scene_two_models_v1")
    p.add_argument("--output", type=Path, default=ROOT / "data/transition_final/correct_5fps_v1")
    p.add_argument("--threads", type=int, default=4)
    args = p.parse_args()
    args.frozen_cache = ROOT / "data/transition_final/frozen_7a9ebd31eba97ba5"
    args.output.mkdir(parents=True, exist_ok=True)
    torch.set_num_threads(args.threads)
    if official_revision(args.upstream) != REVISION:
        p.error("Pinned PathCRF source revision required")
    sys.path.insert(0, str(args.upstream))
    return args


def native_features(half):
    """Observed geometry only, with future context permitted for offline correction."""
    xy = half.xy
    ball = xy[:, -1]
    seen = np.isfinite(ball).all(axis=1)
    alive = half.coarse["base"] != 2
    before = np.full_like(ball, np.nan)
    after = np.full_like(ball, np.nan)
    for a, b in contiguous_runs(alive & seen, half.frame):
        if b - a >= 5:
            before[a + 2:b] = (ball[a + 2:b] - ball[a:b - 2]) / .08
            after[a:b - 2] = (ball[a + 2:b] - ball[a:b - 2]) / .08
    speed_before = np.linalg.norm(before, axis=1)
    speed_after = np.linalg.norm(after, axis=1)
    denom = speed_before * speed_after
    cosine = np.divide((before * after).sum(axis=1), denom,
                       out=np.ones(len(ball)), where=denom > 1e-6)
    angle = np.arccos(np.clip(cosine, -1, 1))
    deceleration = np.maximum(0, speed_before - speed_after)
    jerk = np.linalg.norm(after - before, axis=1)
    nearest, near_counts = [], []
    for team in (0, 1):
        nodes = xy[:, half.team == team]
        distances = np.linalg.norm(nodes - ball[:, None], axis=-1)
        distances = np.where(np.isfinite(distances), distances, np.inf)
        nearest.append(distances.min(axis=1))
        near_counts.append((distances < 3).sum(axis=1))
    contest = (nearest[0] < 3) & (nearest[1] < 3) & alive
    low_ball = np.minimum(nearest[0], nearest[1]) <= 1.5
    touch_score = np.nan_to_num(angle / np.pi + deceleration / 10)
    touch = low_ball & ((angle >= np.deg2rad(15)) | (deceleration >= 3)) & seen & alive
    touch &= np.isfinite(before).all(axis=1) & np.isfinite(after).all(axis=1)
    peaks = find_peaks(np.where(touch, touch_score, 0), distance=3)[0]
    is_peak = np.zeros(len(ball), bool)
    is_peak[peaks] = True
    base = half.coarse["base"]
    signed_margin = np.where(base == 1, half.signals[:, 0], -half.signals[:, 0])
    x = np.column_stack((ball, before, after, speed_before, speed_after, angle, deceleration, jerk,
                         seen, nearest[0], nearest[1], near_counts[0], near_counts[1], contest,
                         half.signals, signed_margin)).astype(np.float32)
    x[~np.isfinite(x)] = np.nan
    # Features from 0.2 seconds before and after the candidate.
    history = np.full_like(x, np.nan)
    future = np.full_like(x, np.nan)
    for a, b in contiguous_runs(alive, half.frame):
        if b - a > 5:
            history[a + 5:b] = x[a:b - 5]
            future[a:b - 5] = x[a + 5:b]
    return {"x": np.c_[x, history, future], "contest": contest, "touch": is_peak,
            "touch_score": touch_score, "nearest": np.column_stack(nearest)}


def episode_targets(half, ep):
    rows = ep.rows[ep.sampled]
    y = np.full(len(rows), -1, np.int64)
    for t, (a, b) in enumerate(zip(rows[:-1], rows[1:], strict=True), 1):
        known = half.labels[a:b + 1]
        if np.isin(known, (0, 1)).all() and np.count_nonzero(np.diff(known)) <= 1:
            y[t] = 0 if known[0] == known[-1] else 2 - known[-1]
    return y


def observed_event_counts(bundles):
    counts = np.ones(5, dtype=np.float64)
    for _, episodes in bundles:
        for ep in episodes:
            if ep.event_supervision is None:
                continue
            for old, new in zip(ep.event_supervision[:-1], ep.event_supervision[1:], strict=True):
                if old < 0 or new < 0:
                    continue
                entries = np.flatnonzero(ep.allowed[new] & (ep.incoming[new] == old))
                if len(entries):
                    kind = ep.event_type[new, entries[0]]
                    if kind:
                        counts[kind] += 1
    probability = counts[1:] / counts[1:].sum()
    costs = -np.log(probability)
    centered = costs - np.average(costs, weights=probability)
    return counts, np.r_[0., centered].astype(np.float32)


def transition_examples(half, pred, info):
    x, y, where = [], [], []
    for a, b in contiguous_runs(half.coarse["base"] != 2, half.frame):
        changes = np.flatnonzero(np.diff(pred[a:b])) + a + 1
        edges = np.r_[a, changes, b]
        for index, t in enumerate(changes, 1):
            end_left, end_right = edges[index - 1], edges[index + 1]
            valid = np.isin(half.labels[max(a, t - 25):min(b, t + 26)], (0, 1)).all()
            vector = np.r_[info["x"][t], (t - end_left) / 25, (end_right - t) / 25,
                           pred[t - 1], pred[t]]
            x.append(vector)
            where.append(t)
            y.append(-1 if not valid else 0)
        # Preserve one-to-one, direction-aware matching when labelling examples.
        if len(changes):
            real = []
            for run in contiguous_runs(np.isin(half.labels[a:b], (0, 1)), half.frame[a:b]):
                real.extend(scores.transitions(half.labels, [(a + run[0], a + run[1])]))
            predicted = [(int(t), int(pred[t - 1]), int(pred[t])) for t in changes]
            matched, _, _ = scores.match_transitions(real, predicted, 25, typed="pair")
            matched_rows = {right[0] for _, right in matched}
            for k in range(len(y) - len(changes), len(y)):
                if y[k] >= 0 and where[k] in matched_rows:
                    y[k] = 1
    shape = info["x"].shape[1] + 4
    return np.asarray(x, np.float32).reshape(-1, shape), np.asarray(y), np.asarray(where)


def fit_models(training, info, output):
    from lightgbm import LGBMClassifier

    team_x, team_y, event_x, event_y = [], [], [], []
    for halves, episodes in training:
        for ep in episodes:
            half = halves[ep.half]
            features = info[half.key]["x"][ep.rows[ep.sampled]]
            labels = episode_targets(half, ep)
            valid = labels >= 0
            team_x.append(features[valid])
            team_y.append(labels[valid])
        for half in halves:
            x, y, _ = transition_examples(half, half.coarse["base"], info[half.key])
            event_x.append(x[y >= 0])
            event_y.append(y[y >= 0])
    x = np.concatenate(team_x)
    y = np.concatenate(team_y)
    count = np.bincount(y, minlength=3)
    class_weight = np.sqrt(count.max() / np.maximum(1, count)).clip(1, 16)
    common = dict(n_estimators=150, learning_rate=.04, num_leaves=15, max_depth=4,
                  min_child_samples=40, reg_lambda=1, random_state=SEED, n_jobs=4, verbosity=-1)
    transition = LGBMClassifier(**common)
    transition.fit(x, y, sample_weight=class_weight[y])
    check_x, check_y = np.concatenate(event_x), np.concatenate(event_y)
    verifier = LGBMClassifier(**common)
    verifier.fit(check_x, check_y)
    transition.booster_.save_model(str(output / "transition_probability.txt"))
    verifier.booster_.save_model(str(output / "transition_verifier.txt"))
    write_json(output / "learned_model_training.json", {
        "transition_samples": len(y), "transition_classes": count.tolist(),
        "sampling_weight": class_weight.tolist(), "verifier_samples": len(check_y),
        "verifier_classes": np.bincount(check_y, minlength=2).tolist(),
        "parameters": common,
    })
    return transition, verifier, class_weight


def predicted_transition_bias(model, x, class_weight):
    probability = model.predict_proba(x)
    aligned = np.full((len(x), 3), 1e-8)
    aligned[:, model.classes_.astype(int)] = probability
    aligned /= class_weight
    aligned /= aligned.sum(axis=1, keepdims=True)
    log = np.log(np.clip(aligned, 1e-6, 1))
    result = log - log[:, :1]
    result[:, 0] = 0
    return result.astype(np.float32)


def reference_prediction(halves, episodes):
    result = [h.coarse["base"].copy() for h in halves]
    for ep in episodes:
        src = ep.source[ep.reference]
        dst = ep.destination[ep.reference]
        teams = team_of(ep.names)
        labels = teams[src]
        labels = np.where(labels >= 0, labels, teams[dst])
        labels = pd.Series(np.where(labels >= 0, labels, np.nan)).ffill().bfill().to_numpy()
        durations = np.diff(np.r_[ep.sampled, len(ep.rows)])
        result[ep.half][ep.rows] = np.repeat(labels, durations)
    for h, decoded in zip(halves, result, strict=True):
        np.testing.assert_array_equal(decoded, h.coarse["base"])
    return result


def render(halves, episodes, paths):
    result = [h.coarse["base"].copy() for h in halves]
    for ep, path in zip(episodes, paths, strict=True):
        teams = team_of(ep.names)
        labels = teams[ep.source[path]]
        labels = np.where(labels >= 0, labels, teams[ep.destination[path]])
        labels = pd.Series(np.where(labels >= 0, labels, np.nan)).ffill().bfill().to_numpy()
        duration = np.diff(np.r_[ep.sampled, len(ep.rows)])
        result[ep.half][ep.rows] = np.repeat(labels, duration)
    return result


def edge_predictions(halves, episodes, info, settings, type_cost, learned):
    outputs = [[] for _ in settings]
    for ep in episodes:
        rows = ep.rows[ep.sampled]
        scene = info[halves[ep.half].key]
        contest = scene["contest"][rows]
        dynamic = learned[(halves[ep.half].key, int(ep.rows[0]))]
        bias = np.zeros((len(settings), len(rows), 3), np.float32)
        event_bias = np.zeros((len(settings), 5), np.float32)
        for i, (contest_weight, type_weight, probability_weight) in enumerate(settings):
            bias[i, :, 1:] += contest_weight * np.where(contest[:, None], .5, -.5)
            bias[i] += probability_weight * dynamic
            event_bias[i] = -type_weight * type_cost
        paths = decode_many(ep, bias, event_bias)
        zero = [i for i, s in enumerate(settings) if s == (0, 0, 0)]
        for i in zero:
            np.testing.assert_array_equal(paths[i], ep.reference)
        for output, path in zip(outputs, paths, strict=True):
            output.append(path)
    return [render(halves, episodes, paths) for paths in outputs]


def snap(half, pred, info, radius):
    if not radius:
        return pred.copy()
    result = pred.copy()
    for a, b in contiguous_runs(pred != 2, half.frame):
        events = np.flatnonzero(np.diff(pred[a:b])) + a + 1
        candidates = np.flatnonzero(info["touch"][a:b]) + a
        previous = a
        for j, t in enumerate(events):
            next_t = events[j + 1] if j + 1 < len(events) else b
            available = candidates[(candidates > previous) & (candidates < next_t) & (abs(candidates - t) <= radius)]
            if len(available):
                new_team = pred[t]
                near = info["nearest"][available, new_team] <= 1.5
                available = available[near]
            if len(available):
                target = min(available, key=lambda row: (abs(int(row) - int(t)), -info["touch_score"][row]))
                if target < t:
                    result[target:t] = pred[t]
                else:
                    result[t:target] = pred[t - 1]
                previous = target
            else:
                previous = t
    return result


def prune(half, pred, info, model, threshold):
    if not threshold:
        return pred.copy()
    x, _, where = transition_examples(half, pred, info)
    result = pred.copy()
    if not len(x):
        return result
    probability = model.predict_proba(x)[:, np.flatnonzero(model.classes_ == 1)[0]]
    for p, t in sorted(zip(probability, where, strict=True)):
        if p >= threshold:
            break
        if result[t - 1] == result[t]:
            continue
        a, b = t - 1, t
        while a > 0 and half.frame[a] - half.frame[a - 1] == 1 and result[a - 1] == result[t - 1]:
            a -= 1
        while b + 1 < len(result) and half.frame[b + 1] - half.frame[b] == 1 and result[b + 1] == result[t]:
            b += 1
        if t - a <= b - t + 1:
            result[a:t] = result[t]
        else:
            result[t:b + 1] = result[t - 1]
    return result


DURATION_BINS = np.asarray([0, .2, .4, .6, 1, 1.5, 2, 3, 5, 8, 12, 20, 30, 60, 120, np.inf])


def duration_model(training):
    lengths = [[], []]
    for halves, _ in training:
        for half in halves:
            runs = contiguous_runs(half.labels >= 0, half.frame)
            for a, b, team in scores.possession_spells(half.labels, runs):
                lengths[team].append((b - a) / 25)
    result = []
    for values in lengths:
        density, _ = np.histogram(values, DURATION_BINS)
        density = (density + 1) / (len(values) + len(density))
        width = np.diff(DURATION_BINS)
        width[-1] = width[-2]
        result.append((np.log(density / width), float(np.mean(values))))
    return result


def semi_markov(half, pred, weight, prior):
    if not weight:
        return pred.copy()
    result = pred.copy()
    confidence = (.25 + .75 * half.signals[:, 0]) / 5
    for first, stop in contiguous_runs(pred != 2, half.frame):
        local = pred[first:stop]
        change = np.flatnonzero(np.diff(local)) + 1
        ends = np.unique(np.r_[0, np.arange(25, len(local), 25), change, len(local)]).astype(int)
        prefix = np.vstack([np.r_[0, np.cumsum(-confidence[first:stop] * (local != team))]
                            for team in (0, 1)])
        count = len(ends)
        dp = np.full((count, 2), -np.inf)
        dp[0] = 0
        back = np.zeros((count, 2), int)
        for end in range(1, count):
            start = np.arange(end)
            duration = (ends[end] - ends[:end]) / 25
            bins = np.searchsorted(DURATION_BINS, duration, side="right") - 1
            for team in (0, 1):
                log_density, mean = prior[team]
                # Compare the learned density with a memoryless duration model.
                duration_bonus = np.clip(log_density[bins] + np.log(mean) + duration / mean, -4, 4)
                value = dp[:end, 1 - team] + prefix[team, ends[end]] - prefix[team, ends[:end]]
                value += weight * duration_bonus
                best = value.argmax()
                dp[end, team], back[end, team] = value[best], best
        end, team = count - 1, int(dp[-1].argmax())
        while end:
            begin = back[end, team]
            result[first + ends[begin]:first + ends[end]] = team
            end, team = begin, 1 - team
    return result


def report(halves, predictions):
    for h, p in zip(halves, predictions, strict=True):
        h.alive = h.labels != 2
        np.testing.assert_array_equal(p == 2, h.coarse["base"] == 2)
    result = metrics(halves, predictions)
    valid = sum(int(np.isin(h.labels, (0, 1)).sum()) for h in halves)
    correct = sum(int(((h.labels == p) & np.isin(h.labels, (0, 1))).sum())
                  for h, p in zip(halves, predictions, strict=True))
    result["frame_accuracy"] = correct / max(1, valid)
    missed = spurious = 0
    latency = []
    for half, pred in zip(halves, predictions, strict=True):
        for run in contiguous_runs(np.isin(half.labels, (0, 1)), half.frame):
            real = scores.transitions(half.labels, [run])
            observed = [event for event in scores.transitions(pred, [run]) if {event[1], event[2]} == {0, 1}]
            matched, absent, extra = scores.match_transitions(real, observed, 25, typed="pair")
            missed += len(absent)
            spurious += len(extra)
            latency.extend(abs(half.time[a[0]] - half.time[b[0]]) * 1000 for a, b in matched)
    result.update(missed_transitions=missed, spurious_transitions=spurious, matched_transitions=len(latency),
                  transition_mae_ms=float(np.mean(latency)) if latency else None)
    return result


def select(halves, grid):
    measured = [{**entry, "metrics": report(halves, pred)} for entry, pred in grid]
    best = max(range(len(grid)), key=lambda i: (
        measured[i]["metrics"]["spellF1"],
        measured[i]["metrics"]["short_recall_iou_0_5"] or 0, -i))
    return grid[best][0], grid[best][1], measured


def run(args):
    started = perf_counter()
    manifest = {
        "split": {"train": TRAIN, "validation": VALIDATION, "test": TEST}, "seed": SEED,
        "base_fps": 5, "evaluation_fps": 25, "dead_mask": "fixed existing scene predicted D",
        "checkpoint_sha256": digest(args.upstream / "saved/140/model/state_dict_best_acc.pt"),
        "base_checkpoint_frozen": True, "selection": "validation Spell F1, then short-spell recall, then simpler",
        "test_role": "previously inspected holdout, retrospective diagnostic",
        "touch_role": "observed-trajectory candidate, not touch ground truth",
        "transition_weighting": "additive adjustments to existing learned sparse CRF scores",
    }
    write_json(args.output / "manifest.json", manifest)
    training = [prepare(args, match) for match in TRAIN]
    validation, val_episodes = prepare(args, VALIDATION)
    all_development = [h for halves, _ in training for h in halves] + validation
    info = {h.key: native_features(h) for h in all_development}
    models_start = perf_counter()
    transition, verifier, class_weights = fit_models(training, info, args.output)
    counts, type_cost = observed_event_counts(training)
    write_json(args.output / "event_prior.json", {"counts_with_laplace": counts.tolist(),
               "types": ["stay", "kick", "same_team_receive", "opponent_receive", "outside"],
               "centered_negative_log_probability": type_cost.tolist()})
    prior = duration_model(training)
    learned = {}
    for hset, episodes in training + [(validation, val_episodes)]:
        for ep in episodes:
            h = hset[ep.half]
            learned[(h.key, int(ep.rows[0]))] = predicted_transition_bias(
                transition, info[h.key]["x"][ep.rows[ep.sampled]], class_weights)
    base = reference_prediction(validation, val_episodes)
    grids, selected = {}, {}
    candidates = [(0, 0, 0)]
    candidates += [(x, 0, 0) for x in (.5, 1., 2., 4.)]
    candidates += [(0, x, 0) for x in (.1, .25, .5, 1.)]
    candidates += [(0, 0, x) for x in (.05, .1, .25, .5, 1.)]
    print("Validation: decoding transition penalties and learned probabilities", flush=True)
    edge = edge_predictions(validation, val_episodes, info, candidates, type_cost, learned)
    edge_map = dict(zip(candidates, edge, strict=True))
    for number, values in (
        (1, [(x, 0, 0) for x in (0, .5, 1., 2., 4.)]),
        (2, [(0, x, 0) for x in (0, .1, .25, .5, 1.)]),
        (5, [(0, 0, x) for x in (0, .05, .1, .25, .5, 1.)]),
    ):
        setting, _, grids[str(number)] = select(validation, [({"edge": list(s)}, edge_map[s]) for s in values])
        selected[str(number)] = setting
    post_grids = {
        3: [({"radius": r}, [snap(h, p, info[h.key], r) for h, p in zip(validation, base, strict=True)])
            for r in (0, 5, 10)],
        4: [({"threshold": t}, [prune(h, p, info[h.key], verifier, t) for h, p in zip(validation, base, strict=True)])
            for t in (0, .2, .4, .6, .8)],
        6: [({"duration_weight": w}, [semi_markov(h, p, w, prior) for h, p in zip(validation, base, strict=True)])
            for w in (0, .25, .5, 1., 2.)],
    }
    for number, grid in post_grids.items():
        setting, _, grids[str(number)] = select(validation, grid)
        selected[str(number)] = setting
    combined = tuple(sum(selected[str(n)]["edge"][i] for n in (1, 2, 5)) for i in range(3))
    combined_pred = edge_predictions(validation, val_episodes, info, [combined], type_cost, learned)[0]
    for i, half in enumerate(validation):
        combined_pred[i] = snap(half, combined_pred[i], info[half.key], selected["3"]["radius"])
        combined_pred[i] = prune(half, combined_pred[i], info[half.key], verifier, selected["4"]["threshold"])
        combined_pred[i] = semi_markov(half, combined_pred[i], selected["6"]["duration_weight"], prior)
    base_quality = report(validation, base)
    combined_quality = report(validation, combined_pred)
    choices = [("base", base_quality)] + [
        (f"correction_{key}", max(grid, key=lambda g: (
            g["metrics"]["spellF1"], g["metrics"]["short_recall_iou_0_5"] or 0))["metrics"])
        for key, grid in grids.items()
    ] + [("combined", combined_quality)]
    chosen_name = max(choices, key=lambda item: (
        item[1]["spellF1"], item[1]["short_recall_iou_0_5"] or 0))[0]
    write_json(args.output / "validation.json", {"base": base_quality, "grids": grids,
               "selected_parameters": selected, "combined": combined_quality, "selected_method": chosen_name})
    print(f"Selected by validation: {chosen_name}; settings={selected}", flush=True)
    selected_seconds = perf_counter() - models_start

    # Holdout labels are first scored after the saved validation settings are fixed.
    test, episodes = prepare(args, TEST)
    info.update({h.key: native_features(h) for h in test})
    for ep in episodes:
        h = test[ep.half]
        learned[(h.key, int(ep.rows[0]))] = predicted_transition_bias(
            transition, info[h.key]["x"][ep.rows[ep.sampled]], class_weights)
    predictions = {"base": reference_prediction(test, episodes)}
    edge_settings = [tuple(selected[str(n)]["edge"]) for n in (1, 2, 5)] + [combined]
    decoded = edge_predictions(test, episodes, info, edge_settings, type_cost, learned)
    for number, pred in zip((1, 2, 5), decoded[:3], strict=True):
        predictions[f"correction_{number}"] = pred
    predictions["correction_3"] = [snap(h, p, info[h.key], selected["3"]["radius"])
                                  for h, p in zip(test, predictions["base"], strict=True)]
    predictions["correction_4"] = [prune(h, p, info[h.key], verifier, selected["4"]["threshold"])
                                  for h, p in zip(test, predictions["base"], strict=True)]
    predictions["correction_6"] = [semi_markov(h, p, selected["6"]["duration_weight"], prior)
                                  for h, p in zip(test, predictions["base"], strict=True)]
    final = decoded[-1]
    for i, half in enumerate(test):
        final[i] = snap(half, final[i], info[half.key], selected["3"]["radius"])
        final[i] = prune(half, final[i], info[half.key], verifier, selected["4"]["threshold"])
        final[i] = semi_markov(half, final[i], selected["6"]["duration_weight"], prior)
    predictions["combined"] = final
    rows = [{"condition": name, **report(test, pred), "validation_selected": name == chosen_name}
            for name, pred in predictions.items()]
    pd.DataFrame(rows).to_csv(args.output / "holdout_comparison.csv", index=False)
    for i, half in enumerate(test):
        frame = pd.DataFrame({"match": half.match, "half": half.number, "frame": half.frame,
                              "seconds": half.time, "truth": half.labels, "dead": half.coarse["base"] == 2,
                              "contention_candidate": info[half.key]["contest"],
                              "touch_candidate": info[half.key]["touch"]})
        for name, pred in predictions.items():
            frame[name] = pred[i]
        frame.to_parquet(args.output / f"{half.key}.parquet", index=False)
        frame.to_csv(args.output / f"{half.key}.csv", index=False)
    write_json(args.output / "results.json", {"metrics": rows, "selected_method": chosen_name,
               "selected_parameters": selected, "selection_and_small_model_seconds": selected_seconds,
               "total_seconds": perf_counter() - started})
    print(pd.DataFrame(rows).to_string(index=False), flush=True)
    print(f"Saved results at {args.output}, wall {perf_counter() - started:.1f}s", flush=True)

if __name__ == "__main__":
    args = parser()
    if args.stage == "prepare":
        for match in TRAIN + [VALIDATION, TEST]:
            prepare(args, match)
    else:
        run(args)
