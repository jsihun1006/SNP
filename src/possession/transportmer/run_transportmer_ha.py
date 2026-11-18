"""DFL-only six-match OOF and one-match B/C possession experiment."""

import argparse
from collections import Counter, defaultdict
from dataclasses import dataclass
import hashlib
import json
import math
from pathlib import Path
import platform
import sys
from time import perf_counter

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(ROOT))

from transportmer_ha_data import FPS, STEP, WINDOW, ball_profile, collate, load_halves, seed_for, split_matches, windows
from transportmer_ha_metrics import METHODS, frame_table, metrics, mismatches, postprocess

BATCH = 24
WARM_EPOCHS = 5
EXTRA_EPOCHS = 15


def save_json(path, payload):
    temporary = path.with_suffix(".tmp")
    temporary.write_text(json.dumps(payload, indent=2, allow_nan=False) + "\n", encoding="utf-8")
    temporary.replace(path)


def read_json(path):
    return json.loads(path.read_text(encoding="utf-8"))


def setup(args):
    args.output.mkdir(parents=True, exist_ok=True)
    split = split_matches(args.seed)
    code = ("run_transportmer_ha.py", "transportmer_ha_data.py", "transportmer_ha_metrics.py",
            "transportmer_backbone.py")
    spec = {
        "split": split, "device_request": args.device, "threads": args.threads,
        "interpolation_fraction_assumed": args.interpolate_fraction,
        "warm_epochs": WARM_EPOCHS, "warm_lr": 2e-4, "extra_epochs": EXTRA_EPOCHS,
        "extra_lr": 1e-4, "batch": BATCH, "fps": FPS, "sample_step": STEP, "window": WINDOW,
        "atd": str(args.atd.resolve()), "dead": str(args.dead.resolve()), "cache": str(args.cache.resolve()),
        "sha256": {name: hashlib.sha256((ROOT / "src" / name).read_bytes()).hexdigest() for name in code},
    }
    manifest = args.output / "manifest.json"
    if manifest.exists() and read_json(manifest) != spec:
        raise ValueError("Run settings or source changed. Use a new output directory.")
    save_json(manifest, spec)
    profile_path = args.output / "ball_profile.json"
    if not profile_path.exists():
        save_json(profile_path, ball_profile(args.atd, args.dead))
    print(f"SPLIT holdout={split['holdout']} development={','.join(split['development'])}", flush=True)
    return split, read_json(profile_path)


def dev_data(args, split, profile):
    halves = load_halves(split["development"], args.cache, profile, args.seed, args.interpolate_fraction)
    report = {half.key: {"rows": len(half.frame), "episodes": len(half.episodes),
                         "first_frame": int(half.frame[0]), "last_frame": int(half.frame[-1]),
                         "home_labels": int((half.labels == 0).sum()), "away_labels": int((half.labels == 1).sum()),
                         **half.augmentation} for half in halves}
    save_json(args.output / "development_halves.json", report)
    return halves


@dataclass
class Engine:
    torch: object
    device: object
    model_type: object
    amp: bool

    @classmethod
    def create(cls, args):
        import torch
        from transportmer_backbone import TranSPORTmer

        torch.set_num_threads(args.threads)
        name = args.device
        if name == "auto":
            name = "cuda" if torch.cuda.is_available() else ("mps" if torch.backends.mps.is_available() else "cpu")
        device = torch.device(name)
        amp = device.type == "cuda" and torch.cuda.is_bf16_supported()
        return cls(torch, device, TranSPORTmer, amp)

    def about(self):
        torch = self.torch
        name = torch.cuda.get_device_name(self.device) if self.device.type == "cuda" else platform.machine()
        return {"device": str(self.device), "hardware": name, "threads": torch.get_num_threads(),
                "torch": torch.__version__, "precision": "bf16-autocast" if self.amp else "float32"}

    def model_optimizer(self, seed, lr):
        self.torch.manual_seed(seed)
        model = self.model_type("possession", motion_input=True).to(self.device)
        return model, self.torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=0.01)

    def forward(self, model, blocks):
        xy, seen, kind, motion, pad, restart, truth = (
            self.torch.as_tensor(x, device=self.device) for x in blocks)
        with self.torch.autocast("cuda", dtype=self.torch.bfloat16, enabled=self.amp):
            logits = model(xy, seen.long(), kind, motion=motion, padding=pad, restart=restart).squeeze(-1)
        return logits.float(), truth

    def train_batch(self, model, optim, blocks):
        optim.zero_grad(set_to_none=True)
        logits, truth = self.forward(model, blocks)
        supervised = truth >= 0
        loss = self.torch.nn.functional.binary_cross_entropy_with_logits(
            logits[supervised], truth[supervised].float())
        loss.backward()
        self.torch.nn.utils.clip_grad_norm_(model.parameters(), 5.0)
        optim.step()
        result = float(loss.detach())
        if not math.isfinite(result):
            raise ValueError("Nonfinite H/A training loss.")
        return result

    def probability(self, model, halves):
        model.eval()
        batches = windows(halves, training=False)
        total = [np.zeros(len(half.frame), dtype=np.float64) for half in halves]
        count = [np.zeros(len(half.frame), dtype=np.int32) for half in halves]
        with self.torch.inference_mode():
            for pos in range(0, len(batches), BATCH):
                batch = batches[pos:pos+BATCH]
                blocks, index = collate(halves, batch)
                logits = self.forward(model, blocks)[0].cpu().numpy()
                for j, (i, _, _, _) in enumerate(batch):
                    valid = ~blocks[4][j]
                    row = index[j, valid]
                    total[i][row] += logits[j, valid]
                    count[i][row] += 1
        probabilities = []
        for half, sums, counts in zip(halves, total, count, strict=True):
            probabilities.append(np.full(len(half.frame), np.nan, dtype=np.float32))
            for first, stop in half.episodes:
                known = np.flatnonzero(counts[first:stop]) + first
                expected = np.arange(first, stop, STEP)
                if not np.array_equal(known, expected):
                    raise ValueError(f"Incomplete sampled timeline for {half.key} at {first}")
                logit = np.interp(np.arange(first, stop), known, sums[known] / counts[known])
                probabilities[-1][first:stop] = 1 / (1 + np.exp(-np.clip(logit, -80, 80)))
            if not np.isfinite(probabilities[-1][half.alive]).all():
                raise ValueError(f"Nonfinite H/A prediction for {half.key}")
        return probabilities


def model_state(engine, model, optim, epoch, seconds, counts):
    return {"model": model.state_dict(), "optimizer": optim.state_dict(), "epoch": epoch,
            "seconds": seconds, "counts": dict(counts)}


def resampled_batch(available, order, step, rng, hard, counts):
    size = BATCH if hard is None else BATCH//2
    samples = [available[index] for index in order[step*size:(step+1)*size]]
    if len(samples) < size:
        samples.extend(available[index] for index in rng.integers(len(available), size=size-len(samples)))
    if hard is not None:
        kinds = sorted(hard)
        for _ in range(BATCH//2):
            kind = rng.choice(kinds)
            error_id, options = hard[kind][rng.integers(len(hard[kind]))]
            samples.append(options[rng.integers(len(options))])
            counts[f"error:{error_id}"] += 1
            counts[f"type:{kind}"] += 1
        rng.shuffle(samples)
    return samples


def fit(engine, halves, available, output, seed, epochs, lr, warm=None, hard=None):
    model, optim = engine.model_optimizer(seed, lr)
    completed, seconds, counts = 0, 0.0, Counter()
    if output.is_file() or warm is not None:
        state = engine.torch.load(output if output.is_file() else warm, map_location=engine.device, weights_only=True)
        model.load_state_dict(state["model"])
        optim.load_state_dict(state["optimizer"])
        if output.is_file():
            completed, seconds, counts = state["epoch"], state["seconds"], Counter(state["counts"])
    for group in optim.param_groups:
        group["lr"] = lr
    steps = math.ceil(len(available) / BATCH)
    for epoch in range(completed, epochs):
        started = perf_counter()
        model.train()
        rng = np.random.default_rng(seed_for(seed, f"epoch:{epoch}"))
        order = rng.permutation(len(available))
        losses = []
        for step in range(steps):
            samples = resampled_batch(available, order, step, rng, hard, counts)
            losses.append(engine.train_batch(model, optim, collate(halves, samples)[0]))
        seconds += perf_counter() - started
        temporary = output.with_suffix(".tmp")
        engine.torch.save(model_state(engine, model, optim, epoch+1, seconds, counts), temporary)
        temporary.replace(output)
        print(f"TRAIN {output.stem} epoch={epoch+1}/{epochs} steps={steps} loss={np.mean(losses):.5f} cumulative_s={seconds:.1f}", flush=True)
    return model, {"seconds": seconds, "steps": steps * epochs, "hard_draws": dict(counts)}


def subsets(split, halves):
    for fold in split["folds"]:
        yield fold["oof"], [half for half in halves if half.match in fold["train"]], [
            half for half in halves if half.match == fold["oof"]]


def benchmark(engine, args, split, halves, prep):
    t = engine.torch
    available = windows(halves, training=True)
    rng = np.random.default_rng(args.seed)
    samples = [available[i] for i in rng.permutation(len(available))[:BATCH*5]]
    checks, _ = collate(halves, samples[:BATCH])
    if np.any(checks[-1][checks[4]] != -1) or np.any(checks[3][checks[4]]):
        raise AssertionError("Padding must carry no target or velocity.")
    model, optim = engine.model_optimizer(args.seed, 2e-4)
    model.train()
    times = []
    for a in range(0, len(samples), BATCH):
        tic = perf_counter()
        engine.train_batch(model, optim, collate(halves, samples[a:a+BATCH])[0])
        times.append(perf_counter()-tic)
    model.eval()
    inference = []
    available_infer = windows(halves, training=False)
    samples = [available_infer[i] for i in rng.permutation(len(available_infer))[:BATCH*5]]
    with t.inference_mode():
        for a in range(0, len(samples), BATCH):
            tic = perf_counter()
            engine.forward(model, collate(halves, samples[a:a+BATCH])[0])[0].cpu()
            inference.append(perf_counter()-tic)
    train = times[1:]
    inference = inference[1:]
    steps = {left: math.ceil(len(windows(learn, training=True))/BATCH)*(WARM_EPOCHS+EXTRA_EPOCHS)
             for left, learn, _ in subsets(split, halves)}
    refit = math.ceil(len(available)/BATCH)
    train_ops = sum(steps.values()) + refit*(WARM_EPOCHS+2*EXTRA_EPOCHS)
    infer_ops = sum(math.ceil(len(windows(out, training=False))/BATCH) for _, _, out in subsets(split, halves))
    # Holdout frame data remain sealed. Approximate its window count by the development mean.
    infer_total = infer_ops + 2*math.ceil(infer_ops/6)
    low = train_ops*min(train) + infer_total*min(inference) + prep
    high = train_ops*max(train) + infer_total*max(inference) + prep
    scoring_start = perf_counter()
    representative = [half for half in halves if half.match == split["development"][0]]
    # Fixed dummy scores exercise the scoring/ledger code without choosing any setting.
    dummy = [np.where(half.alive, 0.5, np.nan) for half in representative]
    realistic_error_volume = 0
    for method in METHODS:
        estimated = [postprocess(half, value, method) for half, value in zip(representative, dummy)]
        metrics(representative, estimated)
    for half, prediction in zip(representative, estimated):
        realistic_error_volume += sum(1 for _ in mismatches(half, prediction, prediction))
    analysis_per_match = perf_counter() - scoring_start
    result = {"runtime": engine.about(), "train_batch_seconds": train, "inference_batch_seconds": inference,
              "preparation_seconds_measured_this_run": prep,
              "estimated_training_steps": train_ops,
              "training_steps_oof": steps, "training_steps_shared_warmup": refit*WARM_EPOCHS,
              "training_steps_per_refit": refit*EXTRA_EPOCHS,
              "inference_batches_oof": infer_ops, "estimated_total_inference_batches": infer_total,
              "estimated_oof_train_and_inference_seconds": sum(steps.values())*float(np.median(train)) + infer_ops*float(np.median(inference)),
              "estimated_shared_warmup_seconds": refit*WARM_EPOCHS*float(np.median(train)),
              "estimated_bc_train_and_inference_seconds": 2*refit*EXTRA_EPOCHS*float(np.median(train)) + (infer_total-infer_ops)*float(np.median(inference)),
              "analysis_benchmark_seconds_per_match": analysis_per_match,
              "analysis_benchmark_error_rows": realistic_error_volume,
              "estimated_oof_analysis_seconds": 6*analysis_per_match,
              "estimated_total_seconds_excluding_io": [low+8*analysis_per_match, high+8*analysis_per_match],
              "estimated_compute_and_preparation_seconds": [low, high],
              "estimate_excludes": "checkpoint and prediction I/O; holdout windows approximated"}
    save_json(args.output / "benchmark.json", result)
    print(f"BENCHMARK {engine.device} steps={train_ops} projected_hours={low/3600:.2f}..{high/3600:.2f}", flush=True)


def load_or_predict(engine, model, around, folder):
    paths = [folder / (half.key + ".npz") for half in around]
    if all(path.is_file() for path in paths):
        values = []
        for half, path in zip(around, paths, strict=True):
            with np.load(path, allow_pickle=False) as loaded:
                if not np.array_equal(half.frame, loaded["frame"]):
                    raise ValueError(f"Prediction frame index changed: {half.key}")
                values.append(loaded["probability"])
        return values
    values = engine.probability(model, around)
    folder.mkdir(exist_ok=True)
    for half, value, path in zip(around, values, paths, strict=True):
        temporary = path.with_suffix(".tmp.npz")
        np.savez_compressed(temporary, frame=half.frame, probability=value)
        temporary.replace(path)
    return values


def collect_oof(engine, args, split, halves):
    saved = {}
    timing = {}
    for left, learn, out in subsets(split, halves):
        started = perf_counter()
        train = windows(learn, training=True)
        warm = args.output / f"oof_{left}_warm.pt"
        final = args.output / f"oof_{left}_full.pt"
        seed = int(seed_for(args.seed, f"model:{left}").generate_state(1)[0])
        _, warm_log = fit(engine, learn, train, warm, seed, WARM_EPOCHS, 2e-4)
        model, main_log = fit(engine, learn, train, final, seed+1, EXTRA_EPOCHS, 1e-4, warm=warm)
        measured_inference = perf_counter()
        values = load_or_predict(engine, model, out, args.output / "oof_probabilities")
        saved[left] = (out, values)
        timing[left] = {"warm": warm_log, "main": main_log, "inference_or_load": perf_counter()-measured_inference,
                        "wall_this_invocation": perf_counter()-started}
        save_json(args.output / "oof_timing.json", timing)
        print(f"OOF complete {left} train_s={warm_log['seconds']+main_log['seconds']:.1f} inference_or_load_s={timing[left]['inference_or_load']:.1f}", flush=True)
        bench = read_json(args.output / "benchmark.json")
        steps_done = sum(item["warm"]["steps"] + item["main"]["steps"] for item in timing.values())
        seconds_done = sum(item["warm"]["seconds"] + item["main"]["seconds"] for item in timing.values())
        remaining_steps = bench["estimated_training_steps"] - steps_done
        seconds = remaining_steps * seconds_done / steps_done
        left_matches = (8 - len(timing)) / 6
        seconds += left_matches * (bench["inference_batches_oof"] * float(np.median(bench["inference_batch_seconds"]))
                                   + bench["estimated_oof_analysis_seconds"])
        save_json(args.output / "remaining.json", {"completed_oof_folds": len(timing),
                  "remaining_training_steps": remaining_steps, "estimated_remaining_seconds": seconds})
        print(f"ETA after {len(timing)}/6 OOF folds remaining_hours={seconds/3600:.2f}", flush=True)
    return saved, timing


def analyze_oof(args, saved):
    started = perf_counter()
    summary = []
    tables = {}
    for match, (halves, probabilities) in saved.items():
        predictions = {method: [postprocess(half, prob, method) for half, prob in zip(halves, probabilities)]
                       for method in METHODS}
        tables[match] = predictions
        for method, values in predictions.items():
            summary.append({"match": match, "method": method, **metrics(halves, values)})
        for i, half in enumerate(halves):
            table = frame_table(half, probabilities[i], {key: values[i] for key, values in predictions.items()})
            table.to_csv(args.output / "oof_probabilities" / f"{half.key}.csv.gz", index=False)
    comparison = pd.DataFrame(summary)
    average = comparison.groupby("method", sort=False)[["spellF1", "short_recall_iou_0_5"]].mean()
    ranked = average.sort_values(["spellF1", "short_recall_iou_0_5"], ascending=False, kind="stable")
    selected = str(ranked.index[0])
    comparison.to_csv(args.output / "oof_smoothing_by_match.csv", index=False)
    ranked.to_csv(args.output / "oof_smoothing_mean.csv")
    errors = []
    for match, (halves, _) in saved.items():
        for i, half in enumerate(halves):
            errors.extend(mismatches(half, tables[match]["raw"][i], tables[match][selected][i]))
    for index, error in enumerate(errors):
        error["error_id"] = index
    pd.DataFrame(errors).to_csv(args.output / "error_ledger.csv", index=False)
    save_json(args.output / "oof_selection.json", {
        "method": selected, "seconds": perf_counter()-started,
        "counts": dict(Counter(row["stage"] for row in errors)),
        "context_rule": "both teams within 2m of observed/simulated ball; diagnostic only",
    })
    print(f"OOF selected {selected}, {len(errors)} error intervals", flush=True)
    return selected, errors


def hard_pool(halves, errors, normal):
    by_half = {(half.match, half.number): i for i, half in enumerate(halves)}
    by_index = defaultdict(list)
    for window in normal:
        by_index[window[0]].append(window)
    grouped = defaultdict(list)
    for row in errors:
        if row["stage"] != "persistent":
            continue
        index = by_half[row["match"], row["half"]]
        first, stop = row["start_row"], row["stop_row_exclusive"]
        options = [window for window in by_index[index] if window[2] <= first < window[3]
                   and window[1] < stop and window[1]+STEP*(WINDOW-1) >= first]
        if options:
            grouped[row["error_type"]].append((row["error_id"], options))
    if not grouped:
        raise ValueError("OOF generated no eligible persistent error windows.")
    return grouped


def compare(engine, args, split, profile, halves, method, errors, oof_times):
    train = windows(halves, training=True)
    pool = hard_pool(halves, errors, train)
    warm = args.output / "refit_warm.pt"
    seed = int(seed_for(args.seed, "refit").generate_state(1)[0])
    _, warm_log = fit(engine, halves, train, warm, seed, WARM_EPOCHS, 2e-4)
    models, report = {}, []
    for name, hard in (("B", None), ("C", pool)):
        model, info = fit(engine, halves, train, args.output / f"refit_{name}.pt",
                          seed+1, EXTRA_EPOCHS, 1e-4, warm=warm, hard=hard)
        models[name] = model
        report.append({"condition": name, **info})
    assert report[0]["steps"] == report[1]["steps"]
    save_json(args.output / "locked_holdout_recipe.json", {"seed": seed, "method": method,
              "training": report, "warm": warm_log, "ordinary_windows": len(train),
              "eligible_error_counts": {key: len(value) for key, value in pool.items()}})
    holdout = load_halves([split["holdout"]], args.cache, profile, args.seed, args.interpolate_fraction)
    save_json(args.output / "holdout_augmentation.json", {half.key: half.augmentation for half in holdout})
    scores = []
    for item in report:
        name = item["condition"]
        start = perf_counter()
        probability = load_or_predict(engine, models[name], holdout, args.output / f"holdout_{name}")
        pred = [postprocess(half, value, method) for half, value in zip(holdout, probability)]
        scores.append({"match": split["holdout"], "condition": name, "method": method,
                       **metrics(holdout, pred)})
        for half, value, labels in zip(holdout, probability, pred, strict=True):
            frame_table(half, value, {"final": labels}).to_csv(
                args.output / f"holdout_{name}" / f"{half.key}.csv.gz", index=False)
        item["prediction_and_evaluation_seconds"] = perf_counter()-start
    save_json(args.output / "timing.json", {"oof": oof_times, "refit_warm": warm_log, "refit": report})
    pd.DataFrame(scores).to_csv(args.output / "holdout_comparison.csv", index=False)
    print(pd.DataFrame(scores).to_string(index=False), flush=True)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--stage", choices=("prepare", "benchmark", "run"), default="run")
    parser.add_argument("--device", default="auto")
    parser.add_argument("--threads", type=int, default=4)
    parser.add_argument("--seed", type=int, default=2027)
    parser.add_argument("--interpolate-fraction", type=float, default=0.25)
    parser.add_argument("--cache", type=Path, default=ROOT / "data/external/dfl")
    parser.add_argument("--atd", type=Path, default=ROOT / "data/tactical/BAR-ATH_tactical_atd.txt")
    parser.add_argument("--dead", type=Path, default=ROOT / "data/dead_balls/final_model/frames.csv")
    args = parser.parse_args()
    if args.threads < 1 or args.seed < 0 or not 0 <= args.interpolate_fraction < 1:
        parser.error("Need positive threads, nonnegative seed and interpolation fraction in [0,1).")
    started = perf_counter()
    split, profile = setup(args)
    halves = dev_data(args, split, profile)
    prep = perf_counter() - started
    print(f"DATA development_halves={len(halves)} prepared_s={prep:.1f}", flush=True)
    if args.stage == "prepare":
        return
    engine = Engine.create(args)
    if args.stage == "benchmark" or not (args.output / "benchmark.json").is_file():
        benchmark(engine, args, split, halves, prep)
    if args.stage == "run":
        saved, times = collect_oof(engine, args, split, halves)
        selected, errors = analyze_oof(args, saved)
        compare(engine, args, split, profile, halves, selected, errors, times)


if __name__ == "__main__":
    main()
