"""Run the short DFL transition-head experiment from the official PathCRF weights."""

from __future__ import annotations

import argparse
import fcntl
import hashlib
import json
from pathlib import Path
import subprocess
import sys
from time import perf_counter
from types import SimpleNamespace

import numpy as np
import pandas as pd
import torch
from torch.nn import functional as F

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from pathcrf_transition_data import build_episodes, prepare_match, read_synced, read_truth, transition_targets
from run_pathcrf_dfl import CHECKPOINT, ROOT, TRIAL, checkpoint_split, scores, write_json
from run_pathcrf_geometry_dfl import UPSTREAM_REV
from run_pathcrf_transition_dfl import (
    BATCH, GRID, TrainingWindows, atomic_save, evaluate_match, halves_from_truth, head_detection,
    make_model, summarize, update_batch, windows_for,
)
from transportmer_ha_data import contiguous_runs
from transportmer_ha_metrics import postprocess

HEAD_EPOCHS = 20
HEAD_PATIENCE = 4
HEAD_BATCH = 1024


def fit(args, model, data, device):
    rng = np.random.default_rng(args.seed)
    order = rng.permutation(len(data.refs))[:args.steps * BATCH]
    if len(order) != args.steps * BATCH:
        raise ValueError("Step budget exceeds the available training windows")
    flips = rng.integers(0, 2, (len(data.refs), 2), dtype=np.int8)
    optimizer = torch.optim.Adam([
        {"params": model.backbone.parameters(), "lr": 2e-5},
        {"params": model.head.parameters(), "lr": 1e-3},
    ], weight_decay=1e-4)
    class_weights = torch.as_tensor(data.class_weights, device=device)
    latest = args.output / "finetune.pt"
    elapsed, step, losses = 0., 0, []
    if latest.exists():
        state = torch.load(latest, map_location="cpu", weights_only=True)
        model.load_state_dict(state["model"])
        optimizer.load_state_dict(state["optimizer"])
        torch.set_rng_state(state["torch_rng"])
        if device.type == "mps":
            torch.mps.set_rng_state(state["device_rng"])
        elif device.type == "cuda":
            torch.cuda.set_rng_state(state["device_rng"], device=device)
        elapsed, step, losses = (state[k] for k in ("seconds", "step", "losses"))
    started = perf_counter()
    model.train()
    for i in range(step, args.steps):
        ids = order[i*BATCH:(i+1)*BATCH]
        losses.append(update_batch(model, optimizer, data, ids, flips, device, args.microbatch, class_weights))
        if (i+1) % 20 == 0 or i in (0, args.steps-1):
            if device.type == "mps":
                torch.mps.synchronize()
                device_rng = torch.mps.get_rng_state()
            elif device.type == "cuda":
                torch.cuda.synchronize(device)
                device_rng = torch.cuda.get_rng_state(device)
            else:
                device_rng = None
            seconds = elapsed + perf_counter() - started
            atomic_save(latest, {
                "model": model.state_dict(), "optimizer": optimizer.state_dict(), "step": i+1,
                "losses": losses, "seconds": seconds, "torch_rng": torch.get_rng_state(), "device_rng": device_rng,
            })
            print(f"step {i+1}/{args.steps} | loss {losses[-1]:.4f} | {seconds:.1f}s", flush=True)
    return elapsed + perf_counter() - started, losses


@torch.inference_mode()
def frozen_examples(args, model, match, device, train):
    import inference as official

    path = args.output / f"{match}_head_features.npz"
    if path.exists():
        with np.load(path) as cached:
            return {key: cached[key] for key in cached.files}
    started = perf_counter()
    truth = read_truth(args.cache, match)
    tracking = prepare_match(args.upstream, args.raw, match, args.prepared_cache, train=train)
    synced = read_synced(args.upstream, match) if train else None
    episodes, skipped = build_episodes(tracking, truth, synced)
    columns = {"features": [], "targets": [], "weights": []}
    counts = np.zeros(3, dtype=np.int64)
    captured = []
    model.eval()
    hook = model.head.register_forward_pre_hook(lambda _module, inputs: captured.append(inputs[0].cpu().numpy()))
    try:
        for ep in episodes:
            targets = ep.targets if train else transition_targets(truth.loc[ep.frames, "label"].to_numpy(), ep.sample_rows)
            events = ep.weights if train else np.ones(len(targets), dtype=np.float32)
            windows = windows_for(len(ep.features))
            window_weights = [official._window_weights(windows, i).numpy() for i in range(len(windows))]
            totals = np.zeros(len(targets), dtype=np.float32)
            for (first, last), weights in zip(windows, window_weights, strict=True):
                weights[0] = 0
                totals[first:last] += weights
            counts += np.bincount(targets[(targets >= 0) & (totals > 0)], minlength=3)
            lengths = np.array([last-first for first, last in windows])
            for length in np.unique(lengths):
                indices = np.flatnonzero(lengths == length)
                for start in range(0, len(indices), 4):
                    group = indices[start:start+4]
                    useful = [idx for idx in group if np.any(
                        (targets[slice(*windows[idx])] >= 0) & (window_weights[idx] > 0))]
                    if not useful:
                        continue
                    features = np.stack([ep.features[slice(*windows[idx])] for idx in useful])
                    home = torch.full((len(useful),), ep.nodes[0].startswith("home_"), dtype=torch.bool, device=device)
                    model(torch.as_tensor(features, device=device), home)
                    for values, idx in zip(captured.pop(), useful, strict=True):
                        rows = slice(*windows[idx])
                        keep = (targets[rows] >= 0) & (window_weights[idx] > 0)
                        columns["features"].append(values[keep])
                        columns["targets"].append(targets[rows][keep])
                        columns["weights"].append(events[rows][keep] * window_weights[idx][keep] / totals[rows][keep])
    finally:
        hook.remove()
    result = {key: np.concatenate(parts) for key, parts in columns.items()}
    result["counts"] = counts
    result["skipped"] = np.array(skipped)
    temporary = path.with_suffix(".tmp")
    with temporary.open("wb") as file:
        np.savez(file, **result)
    temporary.replace(path)
    print(f"{match}: {int(counts.sum()):,} supervised 5fps samples, {perf_counter()-started:.1f}s", flush=True)
    return result


def fit_frozen(args, model, split, device):
    path = args.output / "head.pt"
    report = args.output / "head_fit.json"
    source = {key: tensor.detach().cpu().clone() for key, tensor in model.backbone.state_dict().items()}
    model.backbone.requires_grad_(False)
    if report.exists():
        model.head.load_state_dict(torch.load(path, map_location=device, weights_only=True))
        return json.loads(report.read_text())

    extraction_start = perf_counter()
    matches = {match: frozen_examples(args, model, match, device, train=True) for match in split["train"]}
    validation = frozen_examples(args, model, split["validation"], device, train=False)
    counts = sum(data["counts"] for data in matches.values())
    if np.any(counts == 0):
        raise ValueError(f"An H/A direction has no training samples: {counts.tolist()}")
    class_weights = torch.as_tensor(np.minimum(10, np.sqrt(counts[0]/counts)), dtype=torch.float32, device=device)
    names = ("features", "targets", "weights")
    train = tuple(torch.from_numpy(np.concatenate([data[key] for data in matches.values()])) for key in names)
    dev = tuple(torch.from_numpy(validation[key]) for key in names)
    extraction_seconds = perf_counter() - extraction_start

    def loss(arrays):
        features, targets, weights = (value.to(device) for value in arrays)
        terms = F.cross_entropy(model.head(features), targets, weight=class_weights, reduction="none")
        return (terms * weights).sum() / (class_weights[targets] * weights).sum()

    train_start = perf_counter()
    optimizer = torch.optim.Adam(model.head.parameters(), lr=1e-3, weight_decay=1e-4)
    rng = torch.Generator().manual_seed(args.seed)
    best, best_epoch, history = float("inf"), 0, []
    for epoch in range(1, HEAD_EPOCHS+1):
        model.head.train()
        for ids in torch.randperm(len(train[0]), generator=rng).split(HEAD_BATCH):
            optimizer.zero_grad(set_to_none=True)
            objective = loss(tuple(value[ids] for value in train))
            objective.backward()
            optimizer.step()
        model.head.eval()
        with torch.inference_mode():
            validation_loss = float(loss(dev))
        if not np.isfinite(validation_loss):
            raise FloatingPointError("Non-finite validation loss during head-only training")
        history.append(validation_loss)
        if validation_loss < best:
            best, best_epoch = validation_loss, epoch
            atomic_save(path, model.head.state_dict())
        print(f"Head epoch {epoch}/{HEAD_EPOCHS}: validation loss {validation_loss:.5f}", flush=True)
        if epoch-best_epoch >= HEAD_PATIENCE:
            break
    model.head.load_state_dict(torch.load(path, map_location=device, weights_only=True))
    assert all(torch.equal(tensor.cpu(), source[key]) for key, tensor in model.backbone.state_dict().items())
    assert all(param.grad is None for param in model.backbone.parameters())
    result = {
        "mode": "frozen_backbone", "epochs": len(history), "best_epoch": best_epoch,
        "validation_loss": history, "backbone_unchanged": True,
        "train_sample_counts": {match: data["counts"].tolist() for match, data in matches.items()},
        "validation_sample_counts": validation["counts"].tolist(),
        "extraction_seconds": extraction_seconds, "head_fit_seconds": perf_counter()-train_start,
    }
    write_json(report, result)
    return result


def saved_baseline(folder, halves):
    predictions = []
    for half in halves:
        table = pd.read_parquet(folder / f"{half.key}.parquet", columns=["frame", "truth", "inplay", "pathcrf"])
        np.testing.assert_array_equal(table["frame"].to_numpy(), half.frame)
        np.testing.assert_array_equal(table["truth"].to_numpy(), half.labels)
        np.testing.assert_array_equal(table["inplay"].to_numpy(), half.alive)
        predictions.append(table["pathcrf"].to_numpy())
    result = summarize(halves, predictions)
    stored = json.loads((folder / "results.json").read_text())["pathcrf"]
    if result != stored:
        raise ValueError("Saved PathCRF metrics do not reproduce on the existing prediction files")
    return result, predictions


def smooth(halves, predictions):
    result = []
    for half, pred in zip(halves, predictions, strict=True):
        valid = half.alive & np.isin(pred, [scores.H, scores.A])
        axis = SimpleNamespace(frame=half.frame, episodes=contiguous_runs(valid, half.frame))
        output = postprocess(axis, pred, "prob35")
        output[~valid] = pred[~valid]
        result.append(output)
    return summarize(halves, result)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--upstream", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--prepared-cache", type=Path, required=True)
    parser.add_argument("--baseline", type=Path, required=True)
    parser.add_argument("--cache", type=Path, default=ROOT / "data/external/dfl")
    parser.add_argument("--raw", type=Path, default=ROOT / "data/external/dfl/raw")
    parser.add_argument("--device", default="mps")
    parser.add_argument("--steps", type=int, default=180)
    parser.add_argument("--microbatch", type=int, default=8)
    parser.add_argument("--seed", type=int, default=100)
    parser.add_argument("--freeze-backbone", action="store_true")
    args = parser.parse_args()
    args.upstream = args.upstream.resolve()
    args.output.mkdir(parents=True, exist_ok=True)
    args.prepared_cache.mkdir(parents=True, exist_ok=True)
    lock = (args.output / "run.lock").open("a")
    fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
    sys.path.insert(0, str(args.upstream))
    from models.utils import load_trial_args

    started = perf_counter()
    torch.set_num_threads(6)
    device = torch.device(args.device)
    commit = subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=args.upstream, text=True).strip()
    if commit != UPSTREAM_REV:
        raise ValueError(f"Expected official PathCRF commit {UPSTREAM_REV}")
    trial_path = args.upstream / f"saved/{TRIAL}"
    trial = load_trial_args(str(trial_path))
    checkpoint = trial_path / "model" / CHECKPOINT
    checkpoint_sha = hashlib.sha256(checkpoint.read_bytes()).hexdigest()
    split = checkpoint_split(args.upstream, trial)
    reference = json.loads((args.baseline / "manifest.json").read_text())
    if reference["split"] != split or reference["checkpoint_sha256"] != checkpoint_sha:
        raise ValueError("Existing predictions must be from the same split and starting checkpoint")
    manifest = {
        "split": split, "checkpoint_sha256": checkpoint_sha, "upstream_commit": commit,
        "width": 128, "attention_heads": 4, "seed": args.seed, "steps": args.steps,
        "batch_size": BATCH, "microbatch": args.microbatch, "device": str(device),
        "backbone_lr": 2e-5, "head_lr": 1e-3, "deadball": "DFL ground truth", "validation_grid": GRID,
        "baseline_saved_path": str(args.baseline.resolve()),
    }
    if args.freeze_backbone:
        manifest.update({
            "mode": "frozen_backbone", "steps": None, "batch_size": HEAD_BATCH, "microbatch": None,
            "backbone_lr": 0, "head_epochs": HEAD_EPOCHS, "head_patience": HEAD_PATIENCE,
            "train_features": "overlapping inference windows without pitch flips; window-weighted 5fps targets",
        })
    manifest_path = args.output / "manifest.json"
    if manifest_path.exists() and json.loads(manifest_path.read_text()) != json.loads(json.dumps(manifest)):
        raise ValueError("Resume requires the same model, data split, settings, and output directory")
    write_json(manifest_path, manifest)

    model = make_model(trial, "B", device, args.seed)
    model.backbone.load_state_dict(torch.load(checkpoint, map_location="cpu", weights_only=True))
    train_coverage = {}
    if args.freeze_backbone:
        print(f"Frozen PathCRF 128d / 4 heads on {device}; extracting events' inputs from player tracks.", flush=True)
        training = fit_frozen(args, model, split, device)
        fit_seconds = training["extraction_seconds"] + training["head_fit_seconds"]
    else:
        episodes = []
        for match in split["train"]:
            path = prepare_match(args.upstream, args.raw, match, args.prepared_cache, train=True)
            built, skipped = build_episodes(path, read_truth(args.cache, match), read_synced(args.upstream, match))
            train_coverage[match] = {"episodes": len(built), "feature_frames_skipped": skipped}
            episodes.extend(built)
        data = TrainingWindows(episodes, model.backbone)
        print(f"Warm-start 128d / 4 heads: {len(data.refs):,} windows, {args.steps} steps, {device}", flush=True)
        prep_ready = perf_counter()
        fit_seconds, losses = fit(args, model, data, device)
        training = {"steps": args.steps, "first_loss": losses[0], "final_loss": losses[-1]}
    ready = perf_counter()

    val_started = perf_counter()
    validation_path = args.output / "validation.json"
    if validation_path.exists():
        val = json.loads(validation_path.read_text())
    else:
        match = split["validation"]
        truth = read_truth(args.cache, match)
        path = prepare_match(args.upstream, args.raw, match, args.prepared_cache, train=False)
        validation, skipped = build_episodes(path, truth)
        val_metrics, _, _ = evaluate_match(model, validation, halves_from_truth(truth, match), GRID, device)
        choice = int(np.argmax([x["spellF1"] for x in val_metrics]))
        val = {"match": match, "grid": GRID, "metrics": val_metrics, "setting": GRID[choice],
               "feature_frames_skipped": skipped, "seconds": perf_counter()-val_started}
        write_json(validation_path, val)
    picked = tuple(val["setting"])
    print(f"Validation fixed gain {picked[0]:g} and threshold {picked[1]:g}; starting held-out evaluation.", flush=True)

    test_started = perf_counter()
    match = split["test"]
    test_path = prepare_match(args.upstream, args.raw, match, args.prepared_cache, train=False)
    test_truth = read_truth(args.cache, match)
    test, skipped = build_episodes(test_path, test_truth)
    halves = halves_from_truth(test_truth, match)
    prior, prior_preds = saved_baseline(args.baseline, halves)
    measurements, predictions, head = evaluate_match(model, test, halves, [picked, GRID[0]], device, save_predictions=True)
    test_seconds = perf_counter() - test_started
    prefix = "frozen" if args.freeze_backbone else "warmstart"
    with_head_key, zero_key = f"{prefix}_transition_head", f"{prefix}_zero_bonus"
    for half, old, with_head, zero_bonus in zip(halves, prior_preds, *predictions, strict=True):
        table = pd.DataFrame({"frame": half.frame, "inplay": half.alive, "truth": half.labels,
                              "previous_pathcrf": old, "transition_head": with_head, "zero_bonus": zero_bonus})
        path = args.output / f"{half.key}.parquet"
        temporary = path.with_suffix(".tmp")
        table.to_parquet(temporary, index=False)
        temporary.replace(path)
    result = {
        "match": match, "selected_setting": picked,
        "metrics": {"saved_pathcrf": prior, with_head_key: measurements[0], zero_key: measurements[1]},
        "prob35": {"saved_pathcrf": smooth(halves, prior_preds), with_head_key: smooth(halves, predictions[0]),
                   zero_key: smooth(halves, predictions[1])},
        "head_detection": head_detection(halves, head, picked[1]),
        "training_coverage": train_coverage, "test_feature_frames_skipped": skipped,
        "fit": training,
        "seconds": {"training_and_preparation": ready-started, "training": fit_seconds, "validation": val["seconds"],
                    "test_preparation_and_evaluation": test_seconds, "current_run_total": perf_counter()-started},
    }
    if args.freeze_backbone:
        result["zero_bonus_frames_changed"] = int(sum(
            np.count_nonzero(a != b) for a, b in zip(prior_preds, predictions[1], strict=True)))
    else:
        result["eligible_windows"] = len(data.refs)
        result["seconds"].pop("training_and_preparation")
        result["seconds"].pop("training")
        result["seconds"].update(train_preparation=prep_ready-started, fine_tuning=fit_seconds)
    write_json(args.output / "results.json", result)
    print(json.dumps(result["metrics"], indent=2), flush=True)


if __name__ == "__main__":
    main()
