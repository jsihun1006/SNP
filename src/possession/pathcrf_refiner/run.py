"""Prepare, benchmark, train, and evaluate the DFL 5 Hz + 25 Hz possession model."""

import argparse
import csv
from dataclasses import dataclass
import hashlib
import json
import math
from pathlib import Path
import platform
import subprocess
import sys
from time import perf_counter

from inputs import (N_FEATURES, ROOT, calibration, gate, load_match, make_features)
from refiner import Refiner, decode, loss_sums

import numpy as np
import pandas as pd
import torch

from evaluation import possession_metrics as scores
from run_pathcrf_dfl import CHECKPOINT, TRIAL, checkpoint_split
from transportmer_ha_data import contiguous_runs
from transportmer_ha_metrics import metrics

REVISION = "1b7c6f85a6b78c92e4543fde72ca2b86a933f997"
FOLDER = Path(__file__).resolve().parent
CONFIGS = ROOT / "experiments/possession/pathcrf/candidate25_dfl"
DEFAULT_OUTPUT = ROOT / "data/transition_final"


def write_json(path, data):
    temporary = path.with_suffix(".tmp.json")
    temporary.write_text(json.dumps(data, indent=2, ensure_ascii=False) + "\n")
    temporary.replace(path)


def save_checkpoint(path, data):
    temporary = path.with_suffix(".tmp.pt")
    torch.save(data, temporary)
    temporary.replace(path)


def read_config(path):
    config = {}
    with path.open(encoding="utf-8-sig", newline="") as source:
        for row in csv.DictReader(source):
            try:
                value = json.loads(row["value"])
            except json.JSONDecodeError:
                value = row["value"]
            config[row["parameter"]] = value
    return config


@dataclass(frozen=True, order=True)
class Window:
    half: int
    start: int
    stop: int


def train_windows(halves, masks, config):
    candidates, ordinary = [], []
    width = config["refine_window_frames"]
    stride = config["train_stride_seconds"] * config["refine_fps"]
    for half_idx, (half, patch) in enumerate(zip(halves, masks, strict=True)):
        for first, stop in contiguous_runs(half.labels != scores.D, half.frame):
            last = max(first, stop - width)
            for start in sorted({first + math.floor(k * stride)
                                 for k in range(math.ceil((last - first) / stride))}
                                | {last}):
                stop_window = min(start + width, stop)
                if not np.isin(half.labels[start:stop_window], (0, 1)).any():
                    continue
                l = candidates if patch[(start + stop_window - 1) // 2] else ordinary
                l.append(Window(half_idx, start, stop_window))
    if not candidates or not ordinary:
        raise ValueError("Training requires at least one candidate and one ordinary window.")
    return candidates, ordinary


def epoch_sizes(candidates, config):
    fraction = config["candidate_train_fraction"]
    cap = config["epoch_window_cap"]
    count = min(len(candidates), math.floor(cap * fraction)) if cap else len(candidates)
    return count, round(count * (1 - fraction) / fraction)


def epoch_windows(candidates, ordinary, config, rng):
    n_candidate, n_ordinary = epoch_sizes(candidates, config)
    result = [candidates[i] for i in rng.choice(len(candidates), n_candidate, replace=False)]
    result.extend(ordinary[i] for i in rng.choice(len(ordinary), n_ordinary,
                                               replace=n_ordinary > len(ordinary)))
    rng.shuffle(result)
    return result


def inference_windows(halves, masks, config):
    width, step = config["refine_window_frames"], config["infer_stride_frames"]
    refs, patches = set(), []
    for index, (half, mask) in enumerate(zip(halves, masks, strict=True)):
        for begin, end in contiguous_runs(half.coarse["base"] != scores.D, half.frame):
            for local_a, local_b in contiguous_runs(mask[begin:end], half.frame[begin:end]):
                a, b = local_a + begin, local_b + begin
                limit = max(begin, end - width)
                patches.append((index, a, b, begin, end))
                target_end = min(end, b + 1)
                start = max(begin, min((a + target_end - width) // 2, limit))
                if target_end - a > width:
                    start = max(begin, min(a - width // 2, limit))
                refs.add(Window(index, start, min(end, start + width)))
                while start + width < target_end:
                    start = min(start + step, limit)
                    refs.add(Window(index, start, min(end, start + width)))
    return sorted(refs), patches


def collate(halves, refs, config, rng=None):
    spans, members = [], []
    for win in refs:
        half = halves[win.half]
        values = half.features[win.start:win.stop]
        seen = values[..., 6].any(axis=0)
        spans.append(values)
        members.append([np.flatnonzero(seen & (half.team == code)) for code in (0, 1)])
    side = max(1, max(len(s) for sides in members for s in sides))
    length = max(map(len, spans))
    values = np.zeros((len(refs), length, 2 * side + 1, N_FEATURES), dtype=np.float32)
    seen = np.zeros(values.shape[:-1], dtype=bool)
    labels = np.full((len(refs), length), -1, dtype=np.int64)
    trans = np.full_like(labels, -1)
    for i, (win, span, teams) in enumerate(zip(refs, spans, members, strict=True)):
        last = len(span)
        for code, player_ids in enumerate(teams):
            values[i, :last, code * side:code * side + len(player_ids)] = span[:, player_ids]
        values[i, :last, -1] = span[:, -1]
        seen[i, :last] = values[i, :last, :, 6] > 0
        seen[i, :last, -1] = True  # retain the missing-ball indicator
        if rng is not None:
            if config["pitch_flip"]:
                for axis, size in enumerate((config["pitch_x"], config["pitch_y"])):
                    if rng.random() < .5:
                        present = values[i, :last, :, 6] > 0
                        values[i, :last, :, axis][present] = size - values[i, :last, :, axis][present]
                        values[i, :last, :, axis + 2] *= -1
            label = halves[win.half].labels[win.start:win.stop].astype(np.int64)
            label[~np.isin(label, (0, 1))] = -1
            labels[i, :last] = label
            left, right = label[:-1], label[1:]
            known = (left >= 0) & (right >= 0)
            trans[i, 1:last][known] = np.where(left[known] == right[known], 0, 2 - right[known])
    return tuple(torch.from_numpy(x) for x in (values, seen, labels, trans))


def sync(device):
    if device.type == "mps":
        torch.mps.synchronize()
    if device.type == "cuda":
        torch.cuda.synchronize(device)


def weights(halves, config, device):
    counts = np.zeros(3, dtype=np.int64)
    for half in halves:
        for a, b in contiguous_runs(np.isin(half.labels, (0, 1)), half.frame):
            left, right = half.labels[a:b - 1], half.labels[a + 1:b]
            counts += np.bincount(np.where(left == right, 0, 2 - right), minlength=3)
    group = np.sqrt(counts[0] / np.maximum(1, counts)).clip(1, config["max_transition_class_weight"])
    return torch.tensor(group, dtype=torch.float32, device=device), counts


def train_step(model, optimizer, halves, refs, microbatch, config, rng, weight, device):
    batches = [collate(halves, refs[i:i + microbatch], config, rng)
               for i in range(0, len(refs), microbatch)]
    valid = sum(int((parts[2] >= 0).sum()) for parts in batches)
    counts = sum((torch.bincount(parts[3].flatten().clamp_min(0), weights=(parts[3].flatten() >= 0).float(),
                                minlength=3) for parts in batches), start=torch.zeros(3))
    sizes = torch.stack((weight.new_tensor(valid), (counts.to(device) * weight).sum(),
                         weight.new_tensor(valid))).clamp_min(1)
    coefficients = weight.new_tensor([config[f"{name}_loss_weight"] for name in ("state", "transition", "crf")])
    optimizer.zero_grad(set_to_none=True)
    loss = weight.new_zeros(3)
    for parts in batches:
        x, mask, y, transition = (a.to(device) for a in parts)
        state, change = model(x, mask)
        normal = loss_sums(state, change, y, transition, weight) / sizes
        (normal * coefficients).sum().backward()
        loss += normal.detach()
    optimizer.step()
    result = loss.cpu().numpy()
    if not np.isfinite(result).all():
        raise ValueError(f"Non-finite training losses: {result.tolist()}")
    return result


@torch.inference_mode()
def inference_batch(model, halves, refs, config, device):
    values, visible, _, _ = collate(halves, refs, config)
    state, change = model(values.to(device), visible.to(device))
    return torch.cat((state, change), dim=-1).cpu().numpy()


@torch.inference_mode()
def infer(model, halves, refs, patches, batch_size, config, device):
    model.eval()
    total = [np.zeros((len(h.frame), 5), np.float32) for h in halves]
    count = [np.zeros(len(h.frame), np.float32) for h in halves]
    for first in range(0, len(refs), batch_size):
        batch = refs[first:first + batch_size]
        out = inference_batch(model, halves, batch, config, device)
        for ref, values in zip(batch, out, strict=True):
            size = ref.stop - ref.start
            w = np.minimum(np.arange(1, size + 1), np.arange(size, 0, -1))
            total[ref.half][ref.start:ref.stop] += values[:size] * w[:, None]
            count[ref.half][ref.start:ref.stop] += w
    result = [h.coarse["base"].copy() for h in halves]
    for i, a, b, first, stop in patches:
        with_right = b < stop
        end = b + int(with_right)
        if not (count[i][a:end] > 0).all():
            raise AssertionError(f"Hole inside refinement patch {halves[i].key} {a}:{end}")
        scores_ = total[i][a:end] / count[i][a:end, None]
        coarse = halves[i].coarse["base"]
        result[i][a:b] = decode(scores_[:b - a, :2], scores_[:b - a, 2:],
                                int(coarse[a - 1]) if a > first else None,
                                int(coarse[b]) if with_right else None,
                                scores_[-1, 2:] if with_right else None)
    return result


def evaluation(halves, refined, masks):
    for half, refined_half, patch in zip(halves, refined, masks, strict=True):
        np.testing.assert_array_equal(refined_half[~patch], half.coarse["base"][~patch])
        np.testing.assert_array_equal(refined_half == scores.D, half.coarse["base"] == scores.D)
        half.alive = half.labels != scores.D
    baseline = [h.coarse["base"] for h in halves]
    results = {"base": metrics(halves, baseline), "refined": metrics(halves, refined)}
    active = sum(np.count_nonzero(p != scores.D) for p in baseline)
    results["coverage"] = sum(np.count_nonzero(p) for p in masks) / max(1, active)
    for key, outputs in (("base", baseline), ("refined", refined)):
        bucket = {"candidate": {"missed": 0, "spurious": 0}, "ordinary": {"missed": 0, "spurious": 0}}
        latency = []
        for h, pred, patch in zip(halves, outputs, masks, strict=True):
            for run in contiguous_runs(np.isin(h.labels, (0, 1)), h.frame):
                real = scores.transitions(h.labels, [run])
                estimated = [e for e in scores.transitions(pred, [run]) if {e[1], e[2]} == {0, 1}]
                paired, missed, spurious = scores.match_transitions(real, estimated, 25, typed="pair")
                latency.extend(abs(h.time[left[0]] - h.time[right[0]]) * 1000 for left, right in paired)
                for field, events in (("missed", missed), ("spurious", spurious)):
                    for row, _, _ in events:
                        bucket["candidate" if patch[row] else "ordinary"][field] += 1
        results[key].update(
            transitions_by_scope=bucket,
            missed_transitions=sum(item["missed"] for item in bucket.values()),
            spurious_transitions=sum(item["spurious"] for item in bucket.values()),
            matched_transitions=len(latency),
            transition_mae_ms=float(np.mean(latency)) if latency else None,
        )
    return results


def prepare(args, config, split, source_device):
    tick = perf_counter()
    validation = load_match(args, split["validation"], source_device)
    selected, grid = calibration(validation, config)
    print(f"Gate: {selected}", flush=True)
    write_json(args.output / "gate.json", {"selected": selected, "grid": grid})
    aug = {}
    training = []
    for match in split["train"]:
        for half in load_match(args, match, source_device):
            aug[half.key] = make_features(half, config, training=True)
            half.xy = None
            training.append(half)
    write_json(args.output / "ball_mask.json", aug)
    print("Prepared all train windows input arrays", flush=True)
    for half in validation:
        make_features(half, config, training=False)
    threshold = selected["threshold"]
    choose = lambda halves: [gate(h, threshold, config["gate_margin_frames"]) for h in halves]
    candidates, ordinary = train_windows(training, choose(training), config)
    val_masks = choose(validation)
    val_refs, patches = inference_windows(validation, val_masks, config)
    timing = perf_counter() - tick
    print(f"Preparation {timing:.1f}s, train windows {len(candidates):,} candidate + "
          f"{len(ordinary):,} ordinary, validation {len(val_refs):,} windows", flush=True)
    return training, validation, candidates, ordinary, val_masks, val_refs, patches, timing


def timed(device, fn):
    sync(device)
    start = perf_counter()
    value = fn()
    sync(device)
    return perf_counter() - start, value


def benchmark(args, config, device, parts, model, optimizer, weight, prep_time):
    training, validation, candidates, ordinary, val_masks, val_refs, patches, _ = parts
    rng = np.random.default_rng(config["seed"])
    refs = epoch_windows(candidates, ordinary, config, rng)
    size = config["batch"]
    n = args.bench_steps
    if len(refs) < size or not val_refs:
        raise ValueError("Benchmark needs a full training batch and at least one validation window.")
    train_times, infer_times = [], []
    for step in range(n + 1):
        start = (step * size) % (len(refs) - size + 1)
        model.train()
        elapsed, value = timed(device, lambda: train_step(model, optimizer, training, refs[start:start + size],
                                                           args.microbatch, config, rng, weight, device))
        if step:
            train_times.append(elapsed)
        print(f"Training {'timed' if step else 'warmup'} batch: {elapsed:.2f}s, losses={value.tolist()}", flush=True)
    model.eval()
    for step in range(n + 1):
        size_ = min(len(val_refs), args.microbatch)
        start = (step * size_) % (len(val_refs) - size_ + 1)
        elapsed, out = timed(device, lambda: inference_batch(model, validation, val_refs[start:start + size_],
                                                            config, device))
        if step:
            infer_times.append(elapsed)
        if not np.isfinite(out).all():
            raise ValueError("Non-finite inference outputs in benchmark.")
    if args.full_validation_probe:
        evaluation_seconds, summary = timed(device, lambda: evaluation(
            validation, infer(model, validation, val_refs, patches, args.microbatch, config, device), val_masks))
        write_json(args.output / "smoke_validation.json", summary)
    else:
        evaluation_seconds = None
    infer_median = float(np.median(infer_times))
    train_steps = math.ceil(len(refs) / size)
    val_batches = math.ceil(len(val_refs) / args.microbatch)
    val_time = evaluation_seconds if evaluation_seconds is not None else infer_median * val_batches
    seconds_per_epoch = [float(np.min(train_times)) * train_steps + val_time,
                         float(np.median(train_times)) * train_steps + val_time,
                         float(np.max(train_times)) * train_steps + val_time]
    report = {
        "device": str(device), "platform": platform.platform(), "torch": str(torch.__version__),
        "threads": torch.get_num_threads(), "microbatch": args.microbatch, "effective_batch": size,
        "model_parameters": sum(p.numel() for p in model.parameters()),
        "measured_prepare_this_invocation_s": prep_time, "train_step_seconds": train_times,
        "inference_batch_seconds": infer_times, "inference_batch_windows": size_,
        "validation_probe_measured_s": evaluation_seconds, "validation_windows": len(val_refs),
        "validation_batches": val_batches, "planned_candidate_windows": len(candidates),
        "ordinary_window_pool": len(ordinary), "windows_per_epoch": len(refs),
        "optimizer_steps_per_epoch": train_steps, "projected_epoch_s_min_median_max": seconds_per_epoch,
        "projected_max_epochs": config["max_epochs"],
        "projected_training_validation_s_min_median_max": [x * config["max_epochs"] for x in seconds_per_epoch],
    }
    if device.type == "cuda":
        report["device_name"] = torch.cuda.get_device_name(device)
        report["peak_device_bytes"] = torch.cuda.max_memory_allocated(device)
    else:
        import resource
        report["peak_process_rss_bytes"] = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss * (
            1 if sys.platform == "darwin" else 1024)
    write_json(args.output / "benchmark.json", report)
    print(json.dumps(report, indent=2), flush=True)


def train(args, config, device, parts, model, optimizer, weight):
    training, validation, candidates, ordinary, val_masks, val_refs, patches, _ = parts
    best = args.output / "best.pt"
    last = args.output / "last.pt"
    history, no_gain, best_score, lr_wait = [], 0, -1., 0
    rng = np.random.default_rng(config["seed"])
    if last.is_file():
        saved = torch.load(last, map_location="cpu", weights_only=False)
        model.load_state_dict(saved["model"])
        optimizer.load_state_dict(saved["optimizer"])
        rng.bit_generator.state = saved["rng"]
        torch.set_rng_state(saved["torch_rng"])
        if device.type == "cuda":
            torch.cuda.set_rng_state(saved["cuda_rng"], device)
        if device.type == "mps":
            torch.mps.set_rng_state(saved["mps_rng"])
        history, no_gain, best_score, lr_wait = (saved[k] for k in ("history", "no_gain", "best_score", "lr_wait"))
    for epoch in range(len(history) + 1, config["max_epochs"] + 1):
        if no_gain >= config["early_stop"]:
            break
        start = perf_counter()
        refs = epoch_windows(candidates, ordinary, config, rng)
        model.train()
        losses = []
        for i in range(0, len(refs), config["batch"]):
            refs_ = refs[i:i + config["batch"]]
            losses.append(train_step(model, optimizer, training, refs_, args.microbatch,
                                     config, rng, weight, device))
            if i == 0 or (i // config["batch"] + 1) % 100 == 0:
                print(f"Epoch {epoch} batch {i // config['batch'] + 1}/{math.ceil(len(refs)/config['batch'])}",
                      flush=True)
        val = evaluation(validation, infer(model, validation, val_refs, patches, args.microbatch, config, device),
                         val_masks)
        quality = val["refined"]["spellF1"]
        if quality > best_score:
            best_score, no_gain, lr_wait = quality, 0, 0
            save_checkpoint(best, model.state_dict())
            write_json(args.output / "best_validation.json", {"epoch": epoch, "scores": val})
        else:
            no_gain += 1
            lr_wait += 1
            if lr_wait >= config["lr_patience"]:
                for group in optimizer.param_groups:
                    group["lr"] = max(config["min_lr"], group["lr"] * .5)
                lr_wait = 0
        row = {"epoch": epoch, "validation_spellF1": quality, "lr": optimizer.param_groups[0]["lr"],
               "seconds": perf_counter() - start, "train_windows": len(refs),
               "losses": np.mean(losses, axis=0).tolist()}
        history.append(row)
        checkpoint = {"model": model.state_dict(), "optimizer": optimizer.state_dict(), "history": history,
                      "no_gain": no_gain, "lr_wait": lr_wait, "best_score": best_score,
                      "rng": rng.bit_generator.state, "torch_rng": torch.get_rng_state()}
        if device.type == "cuda":
            checkpoint["cuda_rng"] = torch.cuda.get_rng_state(device)
        if device.type == "mps":
            checkpoint["mps_rng"] = torch.mps.get_rng_state()
        save_checkpoint(last, checkpoint)
        write_json(args.output / "training.json", history)
        print(json.dumps(row), flush=True)


def run_test(args, config, split, model, device, source_device):
    model.load_state_dict(torch.load(args.output / "best.pt", map_location=device, weights_only=True))
    threshold = json.loads((args.output / "gate.json").read_text())["selected"]["threshold"]
    results = {}
    for oracle in (False, True):
        start = perf_counter()
        mode = "oracle_D_diagnostic" if oracle else "estimated_D_primary"
        halves = load_match(args, split["test"], source_device, oracle=oracle)
        masks = [gate(h, threshold, config["gate_margin_frames"]) for h in halves]
        refs, patches = inference_windows(halves, masks, config)
        for h in halves:
            make_features(h, config, training=False)
        pred = infer(model, halves, refs, patches, args.microbatch, config, device)
        result = evaluation(halves, pred, masks)
        if not oracle:
            for h in halves:
                make_features(h, config, training=False, hide_ball=True)
            masked = infer(model, halves, refs, patches, args.microbatch, config, device)
            result["same_checkpoint_ball_hidden"] = evaluation(halves, masked, masks)["refined"]
        for h, p, mask in zip(halves, pred, masks, strict=True):
            columns = dict(match=h.match, half=h.number, frame=h.frame, seconds=h.time,
                           truth=h.labels, dead=h.coarse["base"] == 2, predicted_dead=h.dead_prediction,
                           base=h.coarse["base"], refined=p, candidate=mask)
            pd.DataFrame(columns).to_parquet(args.output / f"{mode}_{h.key}.parquet", index=False)
        results[mode] = {**result, "inference_windows": len(refs),
                         "pathcrf_fallback_frames": sum(h.coarse["fallback"] for h in halves),
                         "seconds": perf_counter() - start}
    write_json(args.output / "holdout.json", results)
    print(json.dumps(results, indent=2), flush=True)


def digest(path):
    with path.open("rb") as f:
        return hashlib.file_digest(f, "sha256").hexdigest()


def official_revision(upstream):
    if (upstream / ".git").exists():
        return subprocess.check_output(
            ["git", "-C", str(upstream), "rev-parse", "HEAD"], text=True,
        ).strip()
    record = json.loads((upstream / "source_manifest.json").read_text())
    if record["revision"] != REVISION:
        raise ValueError("The packaged PathCRF revision differs from the pinned source.")
    for relative, expected in record["sha256"].items():
        if digest(upstream / relative) != expected:
            raise ValueError(f"Packaged PathCRF file changed: {relative}")
    return record["revision"]


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("stage", choices=("benchmark", "train", "test"))
    p.add_argument("--config", type=Path, default=CONFIGS / "parameters.csv")
    p.add_argument("--upstream", type=Path, default=Path("/private/tmp/pathcrf-official-dfl"))
    p.add_argument("--cache", type=Path, default=ROOT / "data/external/dfl")
    p.add_argument("--raw", type=Path, default=ROOT / "data/external/dfl/raw")
    p.add_argument("--sources", type=Path, default=DEFAULT_OUTPUT / "source")
    p.add_argument("--official-prepared", type=Path, default=None)
    p.add_argument("--dead-scene", type=Path, default=ROOT / "data/pathcrf_dfl/scene_two_models_v1")
    p.add_argument("--output", type=Path, default=DEFAULT_OUTPUT / "run")
    p.add_argument("--device", choices=("auto", "cpu", "mps", "cuda"), default="auto")
    p.add_argument("--source-device", choices=("cpu", "cuda"), default="cpu")
    p.add_argument("--threads", type=int, default=4)
    p.add_argument("--microbatch", type=int, default=32)
    p.add_argument("--bench-steps", type=int, default=4)
    p.add_argument("--full-validation-probe", action="store_true")
    p.add_argument("--epochs", type=int)
    p.add_argument("--epoch-window-cap", type=int)
    args = p.parse_args()
    config = read_config(args.config)
    if args.epochs is not None:
        config["max_epochs"] = args.epochs
    if args.epoch_window_cap is not None:
        config["epoch_window_cap"] = args.epoch_window_cap
    if (args.microbatch < 1 or args.microbatch > config["batch"] or args.bench_steps < 1 or
            config["max_epochs"] < 1 or config["epoch_window_cap"] not in (0,) and config["epoch_window_cap"] < 4):
        p.error("Batch sizes, epoch budget, and benchmark counts must be valid.")
    if config["base_fps"] != 5 or config["refine_fps"] != 25 or config["validation_every_epochs"] != 1:
        p.error("This experiment reuses the pinned 5 Hz PathCRF and native 25 Hz DFL, validating every epoch.")
    if args.device == "auto":
        args.device = "cuda" if torch.cuda.is_available() else "mps" if torch.backends.mps.is_available() else "cpu"
    device, source_device = torch.device(args.device), torch.device(args.source_device)
    torch.set_num_threads(args.threads)
    torch.manual_seed(config["seed"])
    rev = official_revision(args.upstream)
    if rev != REVISION:
        p.error(f"Expected official PathCRF revision {REVISION}, got {rev}.")
    sys.path.insert(0, str(args.upstream.resolve()))
    official = args.upstream / f"saved/{TRIAL}"
    trial = json.loads((official / "args.json").read_text())
    split = checkpoint_split(args.upstream, trial)
    expected = {"train": config["train_matches"].split(),
                "validation": config["validation_match"], "test": config["test_match"]}
    if split != expected or split != json.loads((args.dead_scene / "manifest.json").read_text())["split"]:
        p.error("The PathCRF and dead-ball checkpoint splits must match the requested split.")
    identity = {
        "pathcrf_commit": rev, "pathcrf_checkpoint_sha256": digest(official / "model" / CHECKPOINT),
        "dead_scene_checkpoint_sha256": digest(args.dead_scene / "full/best.pt"),
        "dead_scene_decoder_sha256": digest(args.dead_scene / "full/fit.json"),
        "candidate_feature_cache": str(args.official_prepared.resolve()) if args.official_prepared else None,
    }
    key = hashlib.sha256(json.dumps(identity, sort_keys=True).encode()).hexdigest()[:16]
    args.frozen_cache = args.sources.parent / f"frozen_{key}"
    args.output.mkdir(parents=True, exist_ok=True)
    manifest = {"split": split, "inputs": identity, "parameters": config,
                "microbatch": args.microbatch, "device": str(device), "source_device": str(source_device)}
    model = Refiner(config, N_FEATURES).to(device)
    if args.stage == "test":
        prior = json.loads((args.output / "manifest.json").read_text())
        if (prior["split"], prior["inputs"], prior["parameters"]) != (split, identity, config):
            p.error("Prediction inputs and settings differ from the trained model.")
        run_test(args, config, split, model, device, source_device)
        return
    if args.stage == "train":
        existing = args.output / "manifest.json"
        if existing.exists() and json.loads(existing.read_text()) != manifest:
            p.error("Use a new output directory after changing training settings or inputs.")
        write_json(existing, manifest)
    start = perf_counter()
    parts = prepare(args, config, split, source_device)
    torch.manual_seed(config["seed"])
    optimizer = torch.optim.Adam(model.parameters(), lr=config["lr"], weight_decay=config["weight_decay"])
    weight, counts = weights(parts[0], config, device)
    write_json(args.output / "transition_weights.json",
               {"counts": counts.tolist(), "weights": weight.cpu().numpy().tolist()})
    if args.stage == "benchmark":
        write_json(args.output / "benchmark_manifest.json", manifest)
        benchmark(args, config, device, parts, model, optimizer, weight, perf_counter() - start)
    else:
        train(args, config, device, parts, model, optimizer, weight)


if __name__ == "__main__":
    main()
