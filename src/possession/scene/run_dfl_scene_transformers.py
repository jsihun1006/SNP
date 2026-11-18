"""Train independent in-play H/A and full-half H/A/D scene Transformers."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import subprocess
import sys
from time import perf_counter
from types import SimpleNamespace

import numpy as np
import pandas as pd
from scipy.signal import find_peaks
import torch
from torch import nn
from torch.nn import functional as F

REPO = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(REPO))
sys.path.insert(0, str(REPO / "src"))

from dfl_scene_data import BOUNDARIES, EVENTS, FEATURE_NAMES, STEP, STRIDE, WINDOW, load_halves, make_scenes
from evaluation import possession_metrics as scores
from run_pathcrf_dfl import CHECKPOINT, ROOT, TRIAL, checkpoint_split, team_predictions, write_json
from run_pathcrf_geometry_dfl import UPSTREAM_REV
from run_pathcrf_transition_dfl import atomic_save, halves_from_truth
from run_pathcrf_transition_warmstart import saved_baseline
from transportmer_backbone import TranSPORTmer
from transportmer_ha_data import contiguous_runs

BATCH = 32
WIDTH = 64
HEADS = 4
LAYERS = 2
FFN = 128
SCOPES = ("inplay", "full")
DECODER_GRID = [(penalty, gain) for penalty in (0.0, 2.0, 4.0, 6.0, 8.0) for gain in (0.0, 1.0, 2.0)]


class SceneTransformer(nn.Module):
    def __init__(self, classes, width=WIDTH):
        super().__init__()
        self.position = nn.Parameter(torch.empty(WINDOW, width))
        nn.init.normal_(self.position, std=0.02)
        self.project = nn.Linear(len(FEATURE_NAMES), width)
        layer = nn.TransformerEncoderLayer(width, HEADS, FFN, dropout=0.1, activation="gelu", batch_first=True)
        self.encoder = nn.TransformerEncoder(layer, LAYERS, enable_nested_tensor=False)
        self.state = nn.Linear(width, classes)
        self.boundary = nn.Linear(width, 2 if classes == 2 else 4)
        self.event = nn.Linear(width, len(EVENTS))

    def forward(self, features, mask):
        x = self.encoder(self.project(features) + self.position, src_key_padding_mask=~mask)
        return self.state(x), self.boundary(x), self.event(x)


class SceneTranSPORTmer(nn.Module):
    def __init__(self, classes, width):
        super().__init__()
        self.project = nn.Linear(len(FEATURE_NAMES), width)
        self.encoder = TranSPORTmer("state", motion_input=True, width=width, heads=HEADS,
                                    ffn=FFN, window=WINDOW)
        self.state = nn.Linear(width, classes)
        self.boundary = nn.Linear(width, 2 if classes == 2 else 4)
        self.event = nn.Linear(width, len(EVENTS))

    def forward(self, features, mask, xy, motion, observed, kinds):
        x = self.encoder(xy, observed.long(), kinds.long(), motion=motion, padding=~mask,
                         frame_context=self.project(features))
        return self.state(x), self.boundary(x), self.event(x)


def normalizer(scenes):
    data = np.concatenate([scene.inputs for scene in scenes])
    mean = np.nanmean(data, axis=0)
    scale = np.nanstd(data, axis=0)
    mean = np.nan_to_num(mean)
    scale = np.where(np.isfinite(scale) & (scale > 1e-6), scale, 1)
    return mean, scale


def windows(scenes, mean, scale):
    """Keep overlapping 10s windows and a mask for shorter segments."""
    refs = []
    for index, scene in enumerate(scenes):
        length = len(scene.samples)
        starts = list(range(0, max(1, length-WINDOW+1), STRIDE))
        if length > WINDOW and starts[-1] != length-WINDOW:
            starts.append(length-WINDOW)
        refs.extend((index, a, min(length, a+WINDOW)) for a in starts)
    n = len(refs)
    data = np.zeros((n, WINDOW, len(FEATURE_NAMES)), dtype=np.float32)
    mask = np.zeros((n, WINDOW), dtype=bool)
    state = np.full((n, WINDOW), -1, dtype=np.int64)
    boundary = np.full((n, WINDOW, scenes[0].boundaries.shape[-1]), -1, dtype=np.float32)
    event = np.full((n, WINDOW, len(EVENTS)), -1, dtype=np.float32)
    for i, (index, first, last) in enumerate(refs):
        scene = scenes[index]
        size = last-first
        rows = slice(first, last)
        mask[i, :size] = True
        data[i, :size] = np.nan_to_num((scene.inputs[rows]-mean)/scale).clip(-8, 8)
        state[i, :size] = scene.states[rows]
        boundary[i, :size] = scene.boundaries[rows]
        boundary[i, 0] = -1
        event[i, :size] = scene.events[rows]
    values = (data, mask, state, boundary, event)
    if scenes[0].half.trajectories is not None:
        values += trajectory_windows(scenes, refs)
    return tuple(torch.from_numpy(value) for value in values), refs


def trajectory_windows(scenes, refs):
    active = []
    for idx, first, last in refs:
        scene = scenes[idx]
        trajectories = scene.half.trajectories
        rows = scene.samples[first:last]
        keep = trajectories.observed[rows].any(axis=0) | (trajectories.kinds == 2)
        active.append(np.flatnonzero(keep))
    shape = (len(refs), WINDOW, max(map(len, active)))
    xy = np.zeros((*shape, 2), dtype=np.float32)
    motion = np.zeros((*shape, 3), dtype=np.float32)
    observed = np.zeros(shape, dtype=bool)
    kinds = np.full((shape[0], shape[2]), 3, dtype=np.int8)
    for i, ((idx, first, last), players) in enumerate(zip(refs, active, strict=True)):
        scene = scenes[idx]
        rows = scene.samples[first:last]
        trajectories = scene.half.trajectories
        keep = np.ix_(rows, players)
        span = (i, slice(0, len(rows)), slice(0, len(players)))
        xy[span] = trajectories.xy[keep]
        motion[span] = trajectories.motion[keep]
        motion[i, :np.count_nonzero(rows - scene.begin < STEP), :len(players)] = 0
        observed[span] = trajectories.observed[keep]
        kinds[i, :len(players)] = trajectories.kinds[players]
    return xy, motion, observed, kinds


def model_inputs(batch, index, device):
    return (value[index].to(device) for value in (*batch[:2], *batch[5:]))


def positive_weights(targets):
    count = (targets > 0).sum(axis=(0, 1))
    return ((targets == 0).sum(axis=(0, 1)) / count.clamp_min(1)).sqrt().clamp(1, 32)


def sparse_loss(logits, target, weight):
    valid = target >= 0
    raw = F.binary_cross_entropy_with_logits(logits, target.clamp_min(0), pos_weight=weight, reduction="none")
    return raw[valid].sum() / valid.sum().clamp_min(1)


@torch.inference_mode()
def predict(model, scenes, prepared, device):
    model.eval()
    data, refs = prepared
    state_count = model.state.out_features
    n_scores = sum(layer.out_features for layer in (model.state, model.boundary, model.event))
    sums = [np.zeros((len(scene.samples), n_scores), dtype=np.float32) for scene in scenes]
    weights = [np.zeros(len(scene.samples), dtype=np.float32) for scene in scenes]
    for first in range(0, len(refs), BATCH):
        part = slice(first, first+BATCH)
        out = torch.cat(model(*model_inputs(data, part, device)), dim=-1).cpu().numpy()
        for y, (idx, begin, end) in zip(out, refs[part], strict=True):
            size = end-begin
            w = np.minimum(np.arange(1, size+1), np.arange(size, 0, -1))
            sums[idx][begin:end] += y[:size]*w[:, None]
            weights[idx][begin:end] += w
    result = []
    for total, weight in zip(sums, weights, strict=True):
        merged = torch.from_numpy(total / weight[:, None])
        state = F.log_softmax(merged[:, :state_count], dim=-1).numpy()
        auxiliary = torch.sigmoid(merged[:, state_count:]).numpy()
        result.append((state, auxiliary[:, :-len(EVENTS)], auxiliary[:, -len(EVENTS):]))
    return result


def decode_state(state, boundary, penalty, gain):
    count = state.shape[1]
    transition = np.full((count, count), -penalty, dtype=np.float32)
    np.fill_diagonal(transition, 0)
    back = np.zeros_like(state, dtype=np.int8)
    curr = state[0].copy()
    for row in range(1, len(state)):
        table = curr[:, None] + transition
        table[0, 1] += gain*boundary[row, 0]
        table[1, 0] += gain*boundary[row, 1]
        if count == 3:
            table[:2, 2] += gain*boundary[row, 2]
            table[2, :2] += gain*boundary[row, 3]
        last = table.argmax(axis=0)
        back[row] = last
        curr = state[row] + table[last, np.arange(count)]
    path = np.empty(len(state), dtype=np.int8)
    path[-1] = curr.argmax()
    for row in range(len(state)-1, 0, -1):
        path[row-1] = back[row, path[row]]
    return path


def predictions(halves, scenes, logits, penalty, gain):
    result = {half.key: np.full(len(half.frame), scores.D, dtype=np.int8) for half in halves}
    for scene, (state, boundary, _) in zip(scenes, logits, strict=True):
        pred = decode_state(state, boundary, penalty, gain)
        durations = np.diff(np.r_[scene.samples, scene.end])
        result[scene.half.key][scene.begin:scene.end] = np.repeat(pred, durations)
    return [result[half.key] for half in halves]


def metric_arrays(halves, pred):
    truth = np.concatenate([np.r_[np.where(h.truth >= 0, h.truth, scores.X), scores.X] for h in halves])
    estimate = np.concatenate([np.r_[p, scores.X] for p in pred])
    return truth, estimate


def spell_value(halves, pred):
    truth, estimate = metric_arrays(halves, pred)
    runs = scores.valid_runs(truth)
    return scores.spell_scores(truth, scores.fill_abstentions(estimate, runs), runs)["mean_f1"]


def spot_scores(scenes, logits, channel, threshold):
    counts = np.zeros(3, dtype=int)
    name = EVENTS[channel]
    for scene, (_, _, events) in zip(scenes, logits, strict=True):
        axis = scene.half.frame
        truth = scene.half.events[name]
        truth = truth[(truth >= axis[scene.begin]) & (truth <= axis[scene.end-1])]
        prob = events[:, channel]
        peaks, _ = find_peaks(np.r_[-1, prob, -1], height=threshold, distance=4)
        estimate = axis[scene.samples[peaks-1]]
        matches, missed, extra = scores.match_transitions(
            [(int(f), 0, 1) for f in truth], [(int(f), 0, 1) for f in estimate], 25, typed="pair")
        counts += list(map(len, (matches, missed, extra)))
    tp, fn, fp = map(int, counts)
    return {"support": tp+fn, "detected": tp+fp, "matched": tp, "f1": 2*tp/max(1, 2*tp+fn+fp)}


def summarize(halves, pred):
    truth, estimate = metric_arrays(halves, pred)
    result = scores.evaluate(truth, estimate, taus=[25], bootstrap_n=0, typed="pair")
    boundaries = {}
    for i, name in enumerate(BOUNDARIES[2:], 2):
        actual, detected = [], []
        for half, forecast in zip(halves, pred, strict=True):
            for labels, output in ((half.truth, actual), (forecast, detected)):
                changes = scores.transitions(labels, contiguous_runs(half.truth >= 0, half.frame))
                for row, frm, to in changes:
                    if (i == 2 and frm in (0, 1) and to == 2) or (i == 3 and frm == 2 and to in (0, 1)):
                        output.append((int(half.frame[row]), 0, 1))
        boundaries[name] = scores.transition_scores(actual, detected, 25, typed="none")
    return {
        "spell_f1": result["spell_f1_mean"], "macro_f1": result["macro_f1"], "accuracy": result["accuracy"],
        "dead_f1": result["per_class"]["D"]["f1"], "dead_boundaries_1s": boundaries,
        "confusion": result["confusion"],
    }


def fit(scope, training, validation, args, device):
    torch.manual_seed(args.seed)
    classes = 2 if scope == "inplay" else 3
    architecture = SceneTranSPORTmer if args.backbone == "transportmer" else SceneTransformer
    model = architecture(classes, args.width).to(device)
    train_scenes = make_scenes(training, scope)
    val_scenes = make_scenes(validation, scope)
    mean, scale = normalizer(train_scenes)
    train, refs = windows(train_scenes, mean, scale)
    val = windows(val_scenes, mean, scale)
    optimizer = torch.optim.AdamW(model.parameters(), lr=5e-4, weight_decay=0.01)
    bounds = positive_weights(train[3]).to(device)
    events = positive_weights(train[4]).to(device)
    folder = args.output / scope
    folder.mkdir(exist_ok=True)
    path = folder / "best.pt"
    info_path = folder / "fit.json"
    if info_path.exists():
        model.load_state_dict(torch.load(path, weights_only=True, map_location=device))
        return model, mean, scale, json.loads(info_path.read_text())
    history, best, best_epoch = [], -1, 0
    rng = torch.Generator().manual_seed(args.seed)
    last_path = folder / "last.pt"
    if last_path.exists():
        checkpoint = torch.load(last_path, weights_only=True, map_location="cpu")
        model.load_state_dict(checkpoint["model"])
        optimizer.load_state_dict(checkpoint["optimizer"])
        history, best, best_epoch = checkpoint["history"], checkpoint["best"], checkpoint["best_epoch"]
        rng.set_state(checkpoint["shuffle"])
    print(f"{scope}: {len(refs):,} train windows, {len(val[1]):,} validation windows", flush=True)
    for epoch in range(len(history)+1, args.epochs+1):
        if epoch-1-best_epoch >= args.patience:
            break
        tick = perf_counter()
        model.train()
        for batch_number, index in enumerate(torch.randperm(len(refs), generator=rng).split(BATCH), 1):
            target, target_b, target_e = (x[index].to(device) for x in train[2:5])
            state, boundary, event = model(*model_inputs(train, index, device))
            classification = F.cross_entropy(state.flatten(0, 1), target.flatten(), ignore_index=-1,
                                               reduction="sum") / (target >= 0).sum().clamp_min(1)
            loss = classification + 0.25*sparse_loss(boundary, target_b, bounds) + 0.1*sparse_loss(event, target_e, events)
            optimizer.zero_grad(set_to_none=True)
            loss.backward()
            nn.utils.clip_grad_norm_(model.parameters(), 5)
            optimizer.step()
            if args.backbone == "transportmer" and epoch == 1 and batch_number == 8:
                if device.type == "mps":
                    torch.mps.synchronize()
                print(f"{scope}: first 8 training batches in {perf_counter()-tick:.1f}s", flush=True)
        val_logits = predict(model, val_scenes, val, device)
        quality = spell_value(validation, predictions(validation, val_scenes, val_logits, 4, 1))
        history.append({"epoch": epoch, "spell_f1": quality, "seconds": perf_counter()-tick})
        if quality > best:
            best, best_epoch = quality, epoch
            atomic_save(path, model.state_dict())
        if args.backbone == "transportmer":
            atomic_save(last_path, {"model": model.state_dict(), "optimizer": optimizer.state_dict(),
                                    "history": history, "best": best, "best_epoch": best_epoch,
                                    "shuffle": rng.get_state()})
        print(f"{scope}: epoch {epoch}/{args.epochs}, validation spellF1 {quality:.4f}, {history[-1]['seconds']:.1f}s", flush=True)
        if epoch-best_epoch >= args.patience:
            break
    model.load_state_dict(torch.load(path, weights_only=True, map_location=device))
    val_logits = predict(model, val_scenes, val, device)
    grid = [spell_value(validation, predictions(validation, val_scenes, val_logits, *setting)) for setting in DECODER_GRID]
    best_setting = DECODER_GRID[int(np.argmax(grid))]
    thresholds = []
    for channel in range(len(EVENTS)):
        sweep = (0.15, 0.25, 0.4, 0.55, 0.7)
        thresholds.append(max(sweep, key=lambda t: spot_scores(val_scenes, val_logits, channel, t)["f1"]))
    info = {
        "best_epoch": best_epoch, "val_history": history, "validation_spell_f1": max(grid),
        "decoder": best_setting, "event_thresholds": thresholds, "training_windows": len(refs),
        "parameters": sum(p.numel() for p in model.parameters()),
        "normalization_mean": mean.tolist(), "normalization_scale": scale.tolist(),
    }
    write_json(info_path, info)
    return model, mean, scale, info


def pathcrf_with_predicted_dead(raw, halves, labels, trial, weights):
    """Rebuild PathCRF episodes using only the full-half model's estimated D mask."""
    from datatools import utils
    from inference import inference
    from models.utils import build_model

    axes = ["frame_id", "period_id", "timestamp"]
    table = pd.read_parquet(raw)
    tracking = table[axes + [col for col in table if col.startswith(("home_", "away_"))]].copy()
    del table
    # Use player observations for substitution phases, independent of the real ball-state labels.
    tracking, _ = utils.label_phases(tracking.assign(ball_state="alive"))
    tracking["episode_id"] = 0
    assert tracking["frame_id"].is_unique
    frame_index = tracking.set_index("frame_id").index
    counter = 1
    scopes = []
    for half, pred in zip(halves, labels, strict=True):
        alive = pred != scores.D
        runs = contiguous_runs(alive, half.frame)
        for start, stop in runs:
            loc = frame_index.get_indexer(half.frame[start:stop])
            available = loc >= 0
            if available.any():
                tracking.loc[loc[available], "episode_id"] = counter
                counter += 1
        scopes.append(SimpleNamespace(frame=half.frame, alive=alive, episodes=runs, key=half.key))
    assert not ({"ball_state", "ball_owning_team_id", "player_id"} & set(axes))
    model = build_model(trial, torch.device("cpu"))
    model.load_state_dict(torch.load(weights, weights_only=True, map_location="cpu"))
    with torch.inference_mode():
        _, _, edges, _ = inference(model, tracking[tracking["episode_id"] > 0], use_crf=True,
                                    evaluate=False, ball_feature=False, correct_episode_lasts=False,
                                    window_seconds=10, sample_freq=5)
    if not {"edge_src", "edge_dst"}.issubset(edges.columns):
        edges = pd.DataFrame(columns=["edge_src", "edge_dst"], dtype=object)
    output, coverage = [], []
    for half, original in zip(scopes, labels, strict=True):
        scored, _, report = team_predictions(half, edges)
        missing = scored == scores.X
        scored[missing] = original[missing]
        np.testing.assert_array_equal(scored == scores.D, original == scores.D)
        coverage.append({**report, "fallback_frames": int(missing.sum())})
        output.append(scored)
    return output, coverage


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--upstream", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--prepared-cache", type=Path, required=True)
    parser.add_argument("--baseline", type=Path, required=True)
    parser.add_argument("--cache", type=Path, default=ROOT / "data/external/dfl")
    parser.add_argument("--epochs", type=int, default=12)
    parser.add_argument("--patience", type=int, default=3)
    parser.add_argument("--backbone", choices=["transformer", "transportmer"], default="transformer")
    parser.add_argument("--width", type=int, default=WIDTH)
    parser.add_argument("--device", choices=["cpu", "mps", "cuda"], default="cpu")
    parser.add_argument("--seed", type=int, default=100)
    parser.add_argument("--threads", type=int, default=5)
    args = parser.parse_args()
    args.output.mkdir(parents=True, exist_ok=True)
    torch.set_num_threads(args.threads)
    if args.width <= 0 or args.width % HEADS or args.epochs < 1 or args.patience < 1:
        parser.error(f"width must be a positive multiple of {HEADS}, with positive epochs and patience")
    device = torch.device(args.device)
    individual_tracks = args.backbone == "transportmer"
    args.upstream = args.upstream.resolve()
    commit = subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=args.upstream, text=True).strip()
    if commit != UPSTREAM_REV:
        raise ValueError(f"Expected PathCRF revision {UPSTREAM_REV}")
    sys.path.insert(0, str(args.upstream))
    from models.utils import load_trial_args

    trial_folder = args.upstream / f"saved/{TRIAL}"
    trial = load_trial_args(str(trial_folder))
    split = checkpoint_split(args.upstream, trial)
    weights = trial_folder / "model" / CHECKPOINT
    manifest = {
        "split": split, "scopes": {"inplay": "ground-truth D boundaries, train and evaluate H/A",
                                    "full": "all half frames, predict H/A/D"},
        "feature_names": FEATURE_NAMES, "embedding": args.width, "attention_heads": HEADS,
        "layers": 6 if individual_tracks else LAYERS,
        "ffn": FFN, "input_fps": 5, "window_steps": WINDOW, "training_stride_steps": STRIDE,
        "event_labels": EVENTS, "boundary_labels": BOUNDARIES, "max_epochs": args.epochs,
        "early_stop_patience": args.patience, "batch": BATCH, "seed": args.seed, "device": str(device),
        "pathcrf_revision": commit, "pathcrf_trial": TRIAL,
    }
    if individual_tracks:
        manifest["backbone"] = args.backbone
        manifest["attention_order"] = ["temporal", "temporal", "social"] * 2
        manifest["trajectory_inputs"] = ["x", "y", "vx", "vy", "velocity_valid", "observed", "type"]
    mpath = args.output / "manifest.json"
    if mpath.exists() and json.loads(mpath.read_text()) != json.loads(json.dumps(manifest)):
        raise ValueError("Use a different output folder when changing experiment settings")
    write_json(mpath, manifest)
    started = perf_counter()
    training = [half for match in split["train"]
                for half in load_halves(args.cache, args.upstream, match, individual_tracks=individual_tracks)]
    validation = load_halves(args.cache, args.upstream, split["validation"], individual_tracks=individual_tracks)
    prepared = perf_counter()
    print(f"Prepared train and validation tracking features in {prepared-started:.1f}s", flush=True)
    trained = {scope: fit(scope, training, validation, args, device) for scope in SCOPES}
    learned = perf_counter()
    match = split["test"]
    test = load_halves(args.cache, args.upstream, match, individual_tracks=individual_tracks)
    result = {}
    outputs = {}
    for scope, (model, mean, scale, info) in trained.items():
        scenes = make_scenes(test, scope)
        logits = predict(model, scenes, windows(scenes, mean, scale), device)
        outputs[scope] = predictions(test, scenes, logits, *info["decoder"])
        result[scope] = {
            **summarize(test, outputs[scope]), "best_epoch": info["best_epoch"], "decoder": info["decoder"],
            "events_1s": {event: spot_scores(scenes, logits, i, info["event_thresholds"][i])
                          for i, event in enumerate(EVENTS)},
        }
        print(f"{scope}: test spellF1 {result[scope]['spell_f1']:.4f}", flush=True)
    outputs["pathcrf_predicted_dead"], coverage = pathcrf_with_predicted_dead(
        args.prepared_cache / f"{match}.parquet", test, outputs["full"], trial, weights)
    result["pathcrf_predicted_dead"] = {**summarize(test, outputs["pathcrf_predicted_dead"]), "coverage": coverage}
    reference_halves = halves_from_truth(
        pd.concat([pd.DataFrame({"period": h.period, "alive": h.truth != 2, "label": h.truth}, index=h.frame) for h in test]), match)
    _, outputs["pathcrf_oracle_dead"] = saved_baseline(args.baseline, reference_halves)
    result["pathcrf_oracle_dead"] = summarize(test, outputs["pathcrf_oracle_dead"])
    for i, half in enumerate(test):
        path = args.output / f"{half.key}.parquet"
        tmp = path.with_suffix(".tmp")
        pd.DataFrame({"frame": half.frame, "truth": half.truth,
                      **{name: pred[i] for name, pred in outputs.items()}}).to_parquet(tmp, index=False)
        tmp.replace(path)
    write_json(args.output / "results.json", {
        "match": match, "conditions": result, "seconds": {"preparation": prepared-started,
        "training": learned-prepared, "evaluation": perf_counter()-learned, "total": perf_counter()-started},
    })
    for scope, data in result.items():
        print(f"{scope}: spellF1={data['spell_f1']:.4f}, macroF1={data['macro_f1']:.4f}", flush=True)


if __name__ == "__main__":
    main()
