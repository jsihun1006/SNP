"""Compare the DFL transition head and 128/192-dimensional PathCRF backbones."""

from __future__ import annotations

import argparse
import fcntl
import hashlib
import json
import math
from pathlib import Path
import subprocess
import sys
from time import perf_counter
from types import SimpleNamespace

import numpy as np
import pandas as pd
import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from pathcrf_transition_data import WINDOW, build_episodes, prepare_match, read_synced, read_truth
from pathcrf_transition_model import TransitionPathCRF, decode_paths, edge_projections
from run_pathcrf_dfl import ROOT, TRIAL, checkpoint_split, scores, team_predictions, write_json
from run_pathcrf_geometry_dfl import UPSTREAM_REV
from transportmer_ha_data import contiguous_runs
from transportmer_ha_metrics import metrics, postprocess

CONDITIONS = {"A": (128, False), "B": (128, True), "C": (192, True)}
SEEDS = (100, 200, 300)
BATCH = 32
EPOCHS = 30
GRID = [(0.0, 0.5)] + [(strength, threshold) for strength in (0.25, 0.5, 1.0, 2.0)
                                for threshold in (0.5, 0.3, 0.1)]


class TrainingWindows:
    def __init__(self, episodes, backbone):
        self.episodes = episodes
        self.refs = []
        count = np.zeros(3, dtype=np.int64)
        comp = backbone.orig2comp.cpu().numpy()
        allowed = np.zeros((backbone.crf.K, backbone.crf.K), dtype=bool)
        current, slot = np.nonzero(backbone.crf.inc_mask.cpu().numpy())
        previous = backbone.crf.inc_idx.cpu().numpy()[current, slot]
        allowed[previous, current] = True
        for i, ep in enumerate(episodes):
            edge_ids = comp[np.maximum(0, ep.edges[:, 0]) * 26 + np.maximum(0, ep.edges[:, 1])]
            invalid = (ep.edges < 0).any(axis=1) | (edge_ids < 0)
            forbidden = ~allowed[np.maximum(0, edge_ids[:-1]), np.maximum(0, edge_ids[1:])]
            candidates = np.arange(len(edge_ids) - WINDOW + 1)
            states = np.r_[0, np.cumsum(invalid)]
            links = np.r_[0, np.cumsum(forbidden)]
            stop = candidates + WINDOW
            starts = candidates[(states[stop] == states[candidates]) & (links[stop-1] == links[candidates])]
            if not len(starts):
                continue
            self.refs.extend((i, int(start)) for start in starts)
            covered = np.zeros(len(edge_ids) + 1, dtype=int)
            covered[starts+1] += 1
            covered[starts+WINDOW] -= 1
            valid_in_head = (np.cumsum(covered[:-1]) > 0) & (ep.targets >= 0)
            count += np.bincount(ep.targets[valid_in_head], minlength=3)
        if not self.refs or np.any(count == 0):
            raise ValueError(f"No train support for one or more transition classes: {count.tolist()}")
        self.counts = count.tolist()
        self.class_weights = np.minimum(10, np.sqrt(count[0] / count)).astype(np.float32)

    def batch(self, indices, flips, device):
        from datatools.config import PITCH_X, PITCH_Y

        features, home_first, edges, targets, weights = [], [], [], [], []
        for index in indices:
            episode, start = self.refs[index]
            ep = self.episodes[episode]
            rows = slice(start, start + WINDOW)
            x = ep.features[rows].copy()
            pair = ep.edges[rows].copy()
            home_left = ep.nodes[0].startswith("home_")
            flip_x, flip_y = flips[index]
            if flip_x:
                perm = np.r_[11:22, 0:11, 23, 22, 24, 25]
                observed = np.r_[x[0, :22, 0] != 0, np.ones(4, dtype=bool)]
                x[:, observed, 2] = PITCH_X - x[:, observed, 2]
                x[..., 4] *= -1
                x[:, :22, 0] *= -1
                x = x[:, perm]
                pair = perm[pair]
                home_left = not home_left
            if flip_y:
                perm = np.r_[0:24, 25, 24]
                observed = np.r_[x[0, :22, 0] != 0, np.ones(4, dtype=bool)]
                x[:, observed, 3] = PITCH_Y - x[:, observed, 3]
                x[..., 5] *= -1
                x = x[:, perm]
                pair = perm[pair]
            target = ep.targets[rows].copy()
            target[0] = -1
            features.append(x)
            home_first.append(home_left)
            edges.append(pair)
            targets.append(target)
            weights.append(ep.weights[rows])
        return tuple(torch.as_tensor(np.stack(values), device=device)
                     for values in (features, home_first, edges, targets, weights))


def make_model(trial, condition, device, seed):
    from models.utils import build_model

    torch.manual_seed(seed)
    width, has_head = CONDITIONS[condition]
    backbone = build_model({**trial, "seq_dim": width}, torch.device("cpu"))
    return TransitionPathCRF(backbone, width, has_head).to(device)


def update_batch(model, optimizer, data, indices, flips, device, microbatch, class_weights):
    optimizer.zero_grad(set_to_none=True)
    # Preserve the same normalization when an optimizer batch is split in memory.
    batch = data.batch(indices, flips, torch.device("cpu"))
    targets, weights = batch[-2:]
    class_weights_cpu = class_weights.detach().cpu()
    head_total = torch.where(targets >= 0, class_weights_cpu[targets.clamp_min(0)] * weights, 0).sum().to(device)
    loss_sum = torch.zeros((), device=device)
    for first in range(0, len(indices), microbatch):
        last = min(first + microbatch, len(indices))
        sub = tuple(t[first:last].to(device) for t in batch)
        loss = model.objective(*sub, class_weights, base_scale=(last-first)/len(indices), head_total=head_total)
        loss.backward()
        loss_sum += loss.detach()
    value = float(loss_sum)
    if not np.isfinite(value):
        raise FloatingPointError(f"Non-finite training loss: {value}")
    optimizer.step()
    return value


def windows_for(steps):
    if steps <= WINDOW:
        return [(0, steps)]
    starts = list(range(0, steps-WINDOW+1, WINDOW//2))
    return [(start, steps if i == len(starts)-1 else start+WINDOW) for i, start in enumerate(starts)]


@torch.inference_mode()
def episode_scores(model, episode, device):
    import inference as official

    backbone = model.backbone
    crf = backbone.crf
    features = episode.features
    steps = len(features)
    mode = torch.tensor([episode.nodes[0].startswith("home_")], dtype=torch.bool, device=device)
    sums = np.zeros((3, steps, crf.K), dtype=np.float32)
    head = np.zeros((steps, 3), dtype=np.float32)
    weights = np.zeros(steps, dtype=np.float32)
    head_weights = np.zeros(steps, dtype=np.float32)
    windows = windows_for(steps)
    for i, (start, stop) in enumerate(windows):
        x = torch.as_tensor(features[start:stop], device=device).unsqueeze(0)
        _, emission, embeddings, logits = model(x, mode)
        valid = backbone.valid_orig_edge_ids
        emission = emission[0].index_select(1, valid)
        previous, current = edge_projections(crf, embeddings[0].index_select(1, valid))
        w = official._window_weights(windows, i).numpy()
        sums[:, start:stop] += torch.stack([emission, previous, current]).float().cpu().numpy() * w[None, :, None]
        weights[start:stop] += w
        if logits is not None:
            head_w = w.copy()
            head_w[0] = 0
            head[start:stop] += logits[0].float().cpu().numpy() * head_w[:, None]
            head_weights[start:stop] += head_w
    if np.any(weights == 0):
        raise ValueError("An inference step has zero aggregate window weight")
    sums /= weights[None, :, None]
    head /= np.maximum(head_weights[:, None], np.finfo(np.float32).tiny)
    probability = torch.from_numpy(head).softmax(dim=-1).numpy()
    probability[head_weights == 0] = (1, 0, 0)
    return sums, probability


def halves_from_truth(truth, match_id):
    halves = []
    for period, half in truth.groupby("period", sort=True):
        frame = half.index.to_numpy()
        alive = half["alive"].to_numpy()
        halves.append(SimpleNamespace(key=f"{match_id}_H{period}", frame=frame, alive=alive,
                                      labels=half["label"].to_numpy(), episodes=contiguous_runs(alive, frame)))
    return halves


def summarize(halves, predictions):
    result = metrics(halves, predictions)
    total = sum((half.alive & (half.labels >= 0)).sum() for half in halves)
    correct = sum(((half.labels == pred) & half.alive).sum() for half, pred in zip(halves, predictions, strict=True))
    result["ha_frame_accuracy"] = float(correct / total)
    return result


def head_detection(halves, predictions, threshold):
    aggregate = {1: [0, 0, 0], 2: [0, 0, 0]}
    for half in halves:
        for start, stop in contiguous_runs(half.alive & (half.labels >= 0), half.frame):
            actual = scores.transitions(half.labels, [(start, stop)])
            available = predictions.loc[half.frame[start]:half.frame[stop-1]]
            frame = available.index.to_numpy()
            time_axis = np.arange(len(frame))
            if len(frame):
                time_axis[1:] += np.cumsum(np.diff(frame) > 5)
            for direction, (old, new) in enumerate(((0, 1), (1, 0)), 1):
                probability = available[direction].to_numpy()
                proposals = [(int(frame[a+probability[a:b].argmax()]), old, new)
                             for a, b in contiguous_runs(probability >= threshold, time_axis)]
                truth = [(int(half.frame[row]), src, dst) for row, src, dst in actual if (src, dst) == (old, new)]
                matches, missed, extra = scores.match_transitions(truth, proposals, 10, typed="pair")
                aggregate[direction] = [a+b for a, b in zip(aggregate[direction], map(len, (matches, missed, extra)))]
    return {
        name: {"matched": tp, "missed": fn, "spurious": fp, "f1": 2*tp / max(1, 2*tp+fp+fn)}
        for name, (tp, fn, fp) in zip(("H_to_A", "A_to_H"), aggregate.values(), strict=True)
    }


@torch.inference_mode()
def evaluate_match(model, episodes, halves, settings, device, save_predictions=False):
    backbone = model.backbone
    crf = backbone.crf
    incoming = crf.inc_idx.cpu().numpy().clip(min=0)
    allowed = crf.inc_mask.cpu().numpy()
    source, destination = backbone.comp_src.cpu().numpy(), backbone.comp_dst.cpu().numpy()
    parts = [[] for _ in settings]
    head_parts = []
    model.eval()
    for ep in episodes:
        names = np.asarray(ep.nodes)
        teams = np.array([0 if name.startswith("home_") and "_pad_" not in name else
                          1 if name.startswith("away_") and "_pad_" not in name else -1 for name in names])
        edge_teams = teams[source]
        direction = np.zeros(allowed.shape, dtype=np.int8)
        switches = (edge_teams[:, None] >= 0) & (edge_teams[incoming] >= 0) & (edge_teams[:, None] != edge_teams[incoming])
        direction[switches] = np.broadcast_to(2-edge_teams[:, None], direction.shape)[switches]
        projected, prob = episode_scores(model, ep, device)
        gains = np.stack([np.maximum(0, (prob-threshold)/(1-threshold)) * strength
                          for strength, threshold in settings])
        gains[:, :, 0] = 0
        paths = decode_paths(*projected, incoming, allowed, direction, gains)
        sampled_frames = ep.frames[ep.sample_rows]
        for slot, path in enumerate(paths):
            decoded = pd.DataFrame({"edge_src": names[source[path]], "edge_dst": names[destination[path]]},
                                   index=sampled_frames)
            parts[slot].append(decoded.reindex(ep.frames).ffill())
        if save_predictions and model.head is not None:
            head_parts.append(pd.DataFrame(prob, index=sampled_frames))
    tables = [pd.concat(chunks, verify_integrity=True) for chunks in parts]
    all_predictions = [[team_predictions(half, table)[0] for half in halves] for table in tables]
    results = [summarize(halves, predictions) for predictions in all_predictions]
    head = pd.concat(head_parts, verify_integrity=True).sort_index() if head_parts else None
    return results, all_predictions, head


def atomic_save(path, state):
    tmp = path.with_suffix(".tmp")
    torch.save(state, tmp)
    tmp.replace(path)


def train_one(args, trial, data, validation, val_halves, condition, seed, device):
    folder = args.output / f"{condition}_seed{seed}"
    folder.mkdir(exist_ok=True)
    best_path = folder / "best.pt"
    latest_path = folder / "latest.pt"
    if (folder / "complete.json").exists():
        return best_path
    model = make_model(trial, condition, device, seed)
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-3, weight_decay=1e-4)
    class_weights = torch.as_tensor(data.class_weights, device=device)
    epoch, cursor, running, elapsed = 1, 0, 0.0, 0.0
    best, best_epoch = -1.0, None
    if latest_path.exists():
        state = torch.load(latest_path, map_location="cpu", weights_only=True)
        model.load_state_dict(state["model"])
        optimizer.load_state_dict(state["optimizer"])
        torch.set_rng_state(state["torch_rng"])
        if device.type == "cuda":
            torch.cuda.set_rng_state(state["device_rng"])
        elif device.type == "mps":
            torch.mps.set_rng_state(state["device_rng"])
        epoch, cursor, running, elapsed = (state[key] for key in ("epoch", "cursor", "running", "elapsed"))
        best, best_epoch = state["best"], state["best_epoch"]
    start_time = perf_counter()

    def checkpoint(epoch, cursor, running):
        rng = torch.cuda.get_rng_state() if device.type == "cuda" else torch.mps.get_rng_state() if device.type == "mps" else None
        atomic_save(latest_path, {
            "model": model.state_dict(), "optimizer": optimizer.state_dict(), "epoch": epoch, "cursor": cursor,
            "running": running, "elapsed": elapsed + perf_counter() - start_time, "best": best, "best_epoch": best_epoch,
            "torch_rng": torch.get_rng_state(), "device_rng": rng,
        })

    print(f"{condition}/{seed}: starting at epoch {epoch}, window {cursor:,}", flush=True)
    while epoch <= args.epochs:
        epoch_start = perf_counter()
        rng = np.random.default_rng([seed, epoch])
        order = rng.permutation(len(data.refs))
        flips = rng.integers(0, 2, (len(order), 2), dtype=np.int8)
        for group in optimizer.param_groups:
            group["lr"] = 1e-3 * 0.5 ** ((epoch > 15) + (epoch > 25))
        model.train()
        for i in range(cursor, len(order), BATCH):
            ids = order[i:i+BATCH]
            running += len(ids) * update_batch(model, optimizer, data, ids, flips, device, args.microbatch, class_weights)
            end = i + len(ids)
            if i == 0 or ((i // BATCH) + 1) % args.checkpoint_steps == 0:
                checkpoint(epoch, end, running)
                print(f"{condition}/{seed} epoch {epoch}: {end:,}/{len(order):,} windows, loss {running/end:.4f}", flush=True)
        evaluated, _, _ = evaluate_match(model, validation, val_halves, [GRID[0]], device)
        score = evaluated[0]["spellF1"]
        if score > best:
            best, best_epoch = score, epoch
            atomic_save(best_path, model.state_dict())
        with (folder / "epochs.jsonl").open("a") as log:
            log.write(json.dumps({"epoch": epoch, "loss": running/len(order), "validation": evaluated[0],
                                  "seconds_since_resume": perf_counter()-epoch_start}) + "\n")
        print(f"{condition}/{seed} epoch {epoch}: validation Spell F1 {score:.6f}", flush=True)
        epoch, cursor, running = epoch + 1, 0, 0.0
        checkpoint(epoch, cursor, running)
    write_json(folder / "complete.json", {"best_epoch": best_epoch, "best_spellF1": best,
                                          "elapsed_seconds": elapsed + perf_counter()-start_time})
    return best_path


def benchmark(args, trial, data, validation, val_halves, device):
    rng = np.random.default_rng(100)
    refs = rng.permutation(len(data.refs))
    flips = rng.integers(0, 2, (len(refs), 2), dtype=np.int8)
    measurements = {}
    for condition in CONDITIONS:
        model = make_model(trial, condition, device, 100)
        optimizer = torch.optim.Adam(model.parameters(), lr=1e-3, weight_decay=1e-4)
        class_weights = torch.as_tensor(data.class_weights, device=device)
        timings, losses = [], []
        for batch in range(args.benchmark_batches + 1):
            start = perf_counter()
            ids = refs[batch*BATCH:(batch+1)*BATCH]
            assert len(ids) == BATCH
            losses.append(update_batch(model, optimizer, data, ids, flips, device, args.microbatch, class_weights))
            if device.type == "cuda":
                torch.cuda.synchronize(device)
            elif device.type == "mps":
                torch.mps.synchronize()
            timings.append(perf_counter() - start)
        start = perf_counter()
        predictions, _, _ = evaluate_match(model, validation, val_halves, [GRID[0]], device)
        validation_seconds = perf_counter() - start
        measurements[condition] = {
            "parameters": sum(p.numel() for p in model.parameters()), "losses": losses,
            "warmup_seconds": timings[0], "step_seconds": timings[1:],
            "validation_seconds": validation_seconds, "smoke_validation_metrics": predictions[0],
        }
        write_json(args.output / "benchmark.json", {
            "device": str(device), "windows": len(data.refs), "batch": BATCH, "microbatch": args.microbatch,
            "logical_steps_per_epoch": math.ceil(len(data.refs)/BATCH), "conditions": measurements,
        })
        print(f"{condition} smoke: {timings[1:]} s/step, validation {validation_seconds:.1f}s", flush=True)


def load_model(trial, condition, device, seed, path):
    model = make_model(trial, condition, device, seed)
    model.load_state_dict(torch.load(path, map_location=device, weights_only=True))
    return model


def finish(args, trial, split, paths, validation, val_halves, device):
    selected_path = args.output / "selected.json"
    if selected_path.exists():
        selected = json.loads(selected_path.read_text())
    else:
        by_seed = {}
        for seed in args.seeds:
            model = load_model(trial, "B", device, seed, paths["B", seed])
            values, _, _ = evaluate_match(model, validation, val_halves, GRID, device)
            by_seed[str(seed)] = values
        mean = [float(np.mean([runs[i]["spellF1"] for runs in by_seed.values()])) for i in range(len(GRID))]
        choice = int(np.argmax(mean))
        selected = {"setting": GRID[choice], "grid": GRID, "mean_validation_spellF1": mean, "by_seed": by_seed}
        write_json(selected_path, selected)
    setting = tuple(selected["setting"])
    print(f"Validation selected {setting}. Preparing frozen holdout {split['test']}.", flush=True)
    test_path = prepare_match(args.upstream, args.raw, split["test"], args.prepared_cache, train=False)
    test_truth = read_truth(args.cache, split["test"])
    test, skipped = build_episodes(test_path, test_truth)
    test_halves = halves_from_truth(test_truth, split["test"])
    results = {}
    for (condition, seed), path in paths.items():
        key = f"{condition}_seed{seed}"
        path_out = args.output / key / "test.json"
        if path_out.exists():
            results[key] = json.loads(path_out.read_text())
            continue
        model = load_model(trial, condition, device, seed, path)
        settings = [setting, GRID[0]] if condition == "B" else [GRID[0] if condition == "A" else setting]
        measured, decoded, head = evaluate_match(model, test, test_halves, settings, device, save_predictions=True)
        main_pred = decoded[0]
        smoothed = []
        for half, pred in zip(test_halves, main_pred, strict=True):
            valid = half.alive & np.isin(pred, (scores.H, scores.A))
            filtered = postprocess(SimpleNamespace(episodes=contiguous_runs(valid, half.frame), frame=half.frame),
                                   pred, "prob35")
            filtered[~valid] = pred[~valid]
            smoothed.append(filtered)
        result = {"raw": measured[0], "prob35": summarize(test_halves, smoothed)}
        if condition == "B":
            result["with_zero_bonus"] = measured[1]
        if head is not None:
            result["head_detection"] = head_detection(test_halves, head, setting[1])
        for i, half in enumerate(test_halves):
            table = pd.DataFrame({"frame": half.frame, "inplay": half.alive, "truth": half.labels,
                                  "prediction": main_pred[i], "prob35": smoothed[i]})
            if condition == "B":
                table["with_zero_bonus"] = decoded[1][i]
            out = args.output / key / f"{half.key}.parquet"
            temporary = out.with_suffix(".tmp")
            table.to_parquet(temporary, index=False)
            temporary.replace(out)
        write_json(path_out, result)
        results[key] = result
    deltas = {}
    for baseline, treatment in (("A", "B"), ("B", "C")):
        differences = [results[f"{treatment}_seed{s}"]["raw"]["spellF1"] -
                       results[f"{baseline}_seed{s}"]["raw"]["spellF1"] for s in args.seeds]
        deltas[f"{treatment}_minus_{baseline}"] = {"by_seed": differences, "mean": float(np.mean(differences)),
                                                   "sd": float(np.std(differences, ddof=1)) if len(differences)>1 else None}
    write_json(args.output / "results.json", {"match": split["test"], "setting": setting, "skipped_frames": skipped,
                                              "runs": results, "spellF1_deltas": deltas})


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--upstream", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--cache", type=Path, default=ROOT / "data/external/dfl")
    parser.add_argument("--raw", type=Path, default=ROOT / "data/external/dfl/raw")
    parser.add_argument("--prepared-cache", type=Path)
    parser.add_argument("--device", default="auto")
    parser.add_argument("--threads", type=int, default=6)
    parser.add_argument("--microbatch", type=int, default=4)
    parser.add_argument("--checkpoint-steps", type=int, default=20)
    parser.add_argument("--epochs", type=int, default=EPOCHS)
    parser.add_argument("--seeds", nargs="+", type=int, default=SEEDS)
    parser.add_argument("--mode", choices=("benchmark", "run"), default="run")
    parser.add_argument("--benchmark-batches", type=int, default=2)
    args = parser.parse_args()
    if not 1 <= args.microbatch <= BATCH or min(args.epochs, args.benchmark_batches, args.checkpoint_steps) < 1:
        parser.error("Expected positive epoch and batch settings and microbatch <= 32")
    args.upstream = args.upstream.resolve()
    sys.path.insert(0, str(args.upstream))
    from models.utils import build_model, load_trial_args

    automatic = "cuda" if torch.cuda.is_available() else "mps" if torch.backends.mps.is_available() else "cpu"
    device = torch.device(automatic if args.device == "auto" else args.device)
    torch.set_num_threads(args.threads)
    commit = subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=args.upstream, text=True).strip()
    if commit != UPSTREAM_REV:
        parser.error(f"Expected official PathCRF commit {UPSTREAM_REV}")
    trial = load_trial_args(str(args.upstream / f"saved/{TRIAL}"))
    split = checkpoint_split(args.upstream, trial)
    args.output.mkdir(parents=True, exist_ok=True)
    run_lock = (args.output / "run.lock").open("a")
    try:
        fcntl.flock(run_lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
    except BlockingIOError:
        parser.error(f"Another PathCRF process is using {args.output}")
    args.prepared_cache = args.prepared_cache or args.output / "prepared"
    args.prepared_cache.mkdir(parents=True, exist_ok=True)
    start = perf_counter()
    print(f"Preparing development tensors on {device}; optimizer batch {BATCH}, microbatch {args.microbatch}.", flush=True)
    episodes, coverage = [], {}
    for match in split["train"] + [split["validation"]]:
        train = match in split["train"]
        path = prepare_match(args.upstream, args.raw, match, args.prepared_cache, train)
        truth = read_truth(args.cache, match)
        built, skipped = build_episodes(path, truth, read_synced(args.upstream, match) if train else None)
        coverage[match] = {"episodes": len(built), "skipped_feature_frames": skipped}
        if train:
            episodes.extend(built)
        else:
            validation, val_halves = built, halves_from_truth(truth, match)
    data = TrainingWindows(episodes, build_model(trial, torch.device("cpu")))
    manifest = {
        "split": split, "conditions": CONDITIONS, "upstream_commit": commit, "seeds": list(args.seeds),
        "epochs": args.epochs, "batch": BATCH, "microbatch": args.microbatch, "device": str(device),
        "coverage": coverage, "windows": len(data.refs), "class_counts": data.counts, "grid": GRID,
        "source_sha256": {name: hashlib.sha256((ROOT / "src" / name).read_bytes()).hexdigest()
                          for name in ("pathcrf_transition_data.py", "pathcrf_transition_model.py",
                                       "run_pathcrf_transition_dfl.py")},
        "reference_plan_sha256": hashlib.sha256((ROOT / "experiments/possession/pathcrf/dfl/experiment_plan.md").read_bytes()).hexdigest(),
        "deadball_mask": "DFL ground truth", "learning_rate": 0.001, "learning_rate_halving_after_epochs": [15, 25],
        "window_index_sha256": hashlib.sha256(np.asarray(data.refs, dtype=np.int64).tobytes()).hexdigest(),
    }
    manifest_path = args.output / "manifest.json"
    canonical = json.loads(json.dumps(manifest))
    if manifest_path.exists():
        old = json.loads(manifest_path.read_text())
        if old != canonical:
            raise ValueError("Run configuration changed. Use a fresh output directory to change the experiment.")
    else:
        write_json(manifest_path, manifest)
    print(f"{len(data.refs):,} eligible train windows prepared in {perf_counter()-start:.1f}s; classes {data.counts}.", flush=True)

    if args.mode == "benchmark":
        benchmark(args, trial, data, validation, val_halves, device)
        return
    paths = {(condition, seed): train_one(args, trial, data, validation, val_halves, condition, seed, device)
             for seed in args.seeds for condition in CONDITIONS}
    finish(args, trial, split, paths, validation, val_halves, device)


if __name__ == "__main__":
    main()
