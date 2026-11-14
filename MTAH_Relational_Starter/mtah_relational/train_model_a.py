"""Local A-model training/evaluation. No ball/tag/GT/network/reporting services."""
from __future__ import annotations

import argparse
import csv
import hashlib
import importlib.metadata
import json
import os
os.environ.setdefault("PYTORCH_ENABLE_MPS_FALLBACK", "1")
from pathlib import Path
import platform
import random
import subprocess
import sys
import time

import numpy as np
import torch

from .corruption import SpanMasker
from .model_a import SpatialBiLSTM, save_checkpoint, load_checkpoint
from .model_a_data import build_feature_cache, FeatureWindowDataset, sampled_indices, batch_arrays
from .reconstruction import masked_huber, ReconstructionMetrics
from .reconstruction_baselines import reconstruct_baselines
from .spatial_schema import FEATURE_NAMES, TARGET_NAMES
from .splits import SplitManifest, assert_disjoint_support
from .temporal_cli import open_temporal_dataset


def _json(path, value):
    Path(path).write_text(json.dumps(value, indent=2, ensure_ascii=False, allow_nan=False)+"\n")


def set_seed(seed, threads=4):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.set_num_threads(threads)
    torch.use_deterministic_algorithms(True, warn_only=True)


def _sync(device):
    if str(device) == "mps":
        torch.mps.synchronize()


def select_device(requested="auto"):
    if requested not in ("auto", "cpu", "mps"):
        raise ValueError("Device must be auto/cpu/mps")
    available = torch.backends.mps.is_available()
    choice = "mps" if requested != "cpu" and available else "cpu"
    info = dict(mps_built=torch.backends.mps.is_built(), mps_available=available,
                requested=requested, device=choice, fallback_reason=None,
                mps_operator_cpu_fallback_enabled=os.environ.get("PYTORCH_ENABLE_MPS_FALLBACK") == "1")
    if requested != "cpu" and not available:
        info["fallback_reason"] = "MPS unavailable in this process"
    started = time.perf_counter()
    try:
        model = SpatialBiLSTM().to(choice)
        optimizer = torch.optim.Adam(model.parameters(), lr=.001)
        x = torch.randn(2, 151, 53, device=choice)
        pred, _ = model(x)
        pred.square().mean().backward()
        optimizer.step()
        padding = torch.zeros(2, 151, dtype=torch.bool, device=choice)
        padding[0, :20], padding[1, -20:] = True, True
        with torch.no_grad():
            model(x, padding)
        _sync(choice)
    except (RuntimeError, NotImplementedError) as exc:
        if choice != "mps":
            raise
        info.update(device="cpu", fallback_reason=f"MPS preflight failed: {type(exc).__name__}: {str(exc)[:400]}")
    info["synthetic_probe_seconds"] = time.perf_counter()-started
    return torch.device(info["device"]), info


def train_update(model, optimizer, x, padding, targets, device, *, delta=1., feature_weights=None, grad_clip=1.):
    fw = np.ones(13) if feature_weights is None else np.asarray(feature_weights)
    eligible = targets.loss_mask & np.isfinite(targets.values) & (fw > 0)
    if not eligible.any():
        return dict(skipped=True, loss=None, weight_sum=0., valid_cells=0, gradient_norm=None)
    optimizer.zero_grad(set_to_none=True)
    prediction, _ = model(torch.as_tensor(x, device=device), torch.as_tensor(padding, device=device))
    result = masked_huber(prediction, torch.as_tensor(targets.values, device=device),
                           torch.as_tensor(targets.observed, device=device),
                           torch.as_tensor(targets.quality_weight, device=device),
                           torch.as_tensor(targets.artificial_mask, device=device), feature_weights, delta)
    if result.loss is None:
        return dict(skipped=True, loss=None, weight_sum=0., valid_cells=0, gradient_norm=None)
    result.loss.backward()
    norm = torch.nn.utils.clip_grad_norm_(model.parameters(), grad_clip, error_if_nonfinite=True)
    optimizer.step()
    return dict(skipped=False, loss=float(result.loss.detach().cpu()), weight_sum=result.weight_sum,
                valid_cells=result.valid_cells, gradient_norm=float(norm.detach().cpu()))


class EarlyStopping:
    def __init__(self, patience, min_delta):
        if patience < 1 or min_delta < 0:
            raise ValueError("Invalid early stopping policy")
        self.patience, self.min_delta = patience, min_delta
        self.best, self.bad_epochs = float("inf"), 0

    def step(self, value):
        if value is None or not np.isfinite(value):
            raise ValueError("No finite validation reconstruction objective")
        improved = value < self.best-self.min_delta
        if improved:
            self.best, self.bad_epochs = value, 0
        else:
            self.bad_epochs += 1
        return improved, self.bad_epochs >= self.patience


def evaluate(model, data, positions, batch_size, device, *, delta=1., feature_weights=None, baselines=True):
    model.eval()
    names = ("persistence", "interpolation", "bilstm") if baselines else ("bilstm",)
    metrics = {name: ReconstructionMetrics(delta) for name in names}
    fallback_cells = 0
    with torch.inference_mode():
        for start in range(0, len(positions), batch_size):
            x, padding, target, _ = batch_arrays(data, positions[start:start+batch_size])
            pred, _ = model(torch.as_tensor(x, device=device), torch.as_tensor(padding, device=device))
            predictions = {"bilstm": pred.cpu().numpy()}
            if baselines:
                other, fallback = reconstruct_baselines(x, padding)
                predictions.update(other)
                fallback_cells += int((fallback & target.loss_mask).sum())
            for name in names:
                metrics[name].update(predictions[name], target.values, target.observed, target.quality_weight,
                                     target.artificial_mask, feature_weights)
    result = {name: m.report(TARGET_NAMES, data.base.normalizer.scale[:13]) for name, m in metrics.items()}
    if len({m["valid_cells"] for m in result.values()}) != 1:
        raise AssertionError("Reconstruction comparisons used different target cells")
    return dict(windows=len(positions), baseline_no_anchor_target_cells=fallback_cells, metrics=result)


def _code_versions():
    root = Path(__file__).resolve().parent
    signatures = {p.name: hashlib.sha256(p.read_bytes()).hexdigest() for p in sorted(root.glob("*.py"))}
    try:
        head = subprocess.run(["git", "rev-parse", "HEAD"], capture_output=True, text=True, check=True).stdout.strip()
    except (OSError, subprocess.CalledProcessError):
        head = None
    return dict(git_head=head, python_source_sha256=signatures)


def export_embeddings(model, base, cache, output, batch_size, device, variance_threshold=1e-6):
    data = FeatureWindowDataset(base, cache)
    n = len(data)
    embedding = np.lib.format.open_memmap(output/"embeddings.npy", mode="w+", dtype="float32", shape=(n, 64))
    frame_ids = np.asarray(base.index.records["center_frame"], dtype=np.int64)
    quality = np.asarray(cache.quality_valid[base.index.records["center_row"]])
    np.save(output/"embedding_frame_ids.npy", frame_ids, allow_pickle=False)
    np.save(output/"embedding_quality_valid.npy", quality, allow_pickle=False)
    model.eval()
    with torch.inference_mode():
        for start in range(0, n, batch_size):
            positions = np.arange(start, min(start+batch_size, n))
            x, padding, _, _ = batch_arrays(data, positions, with_targets=False)
            _, state = model(torch.as_tensor(x, device=device), torch.as_tensor(padding, device=device))
            embedding[start:start+len(positions)] = state[:, base.index.spec.radius].cpu().numpy()
            if start and start//batch_size % 400 == 0:
                print(f"Exported {start:,}/{n:,} unmasked center embeddings", flush=True)
    embedding.flush()
    reports = {}
    for label, mask in (("all", np.ones(n, dtype=bool)), ("quality_valid", quality)):
        a = np.asarray(embedding[mask], dtype=np.float64)
        if not len(a) or not np.isfinite(a).all():
            raise FloatingPointError("Missing or nonfinite exported embeddings")
        var = a.var(axis=0)
        eigen = np.maximum(np.linalg.eigvalsh(np.cov(a, rowvar=False)), 0)
        probabilities = eigen/eigen.sum() if eigen.sum() > 0 else np.zeros(64)
        effective_rank = float(np.exp(-(probabilities[probabilities > 0]*np.log(probabilities[probabilities > 0])).sum())) \
            if eigen.sum() else 0.
        reports[label] = dict(frames=len(a), dimension_variances=var.tolist(), mean_variance=float(var.mean()),
                              minimum_variance=float(var.min()), low_variance_dimensions=int((var < variance_threshold).sum()),
                              variance_threshold=variance_threshold, effective_rank=effective_rank,
                              interpretation="variance/rank diagnostics, not proof of semantic non-collapse")
    del embedding
    return reports


def run_training(temporal_dir, feature_cache_dir, output_dir, config):
    output = Path(output_dir).resolve()
    if output.exists():
        raise FileExistsError("A training output already exists; refusing overwrite")
    temporal_dir = Path(temporal_dir).resolve()
    if config["epochs"] < 1 or config["batch_size"] < 1 or config["learning_rate"] <= 0:
        raise ValueError("Invalid training settings")
    if len(config["feature_weights"]) != 13 or not np.isfinite(config["feature_weights"]).all() \
            or min(config["feature_weights"]) < 0 or not max(config["feature_weights"]) > 0:
        raise ValueError("Invalid feature loss weights")
    set_seed(config["seed"], config["cpu_threads"])
    bases = {s: open_temporal_dataset(temporal_dir, split=s) for s in ("train", "validation", "test")}
    for base in bases.values():
        base.masker = SpanMasker(seed=config["seed"])
    manifest = SplitManifest.load(temporal_dir/"split_manifest.json")
    support = assert_disjoint_support([d.index for d in bases.values()], manifest)
    cache = build_feature_cache(bases["train"], feature_cache_dir, progress=True)
    data = {s: FeatureWindowDataset(base, cache) for s, base in bases.items()}
    # Verify the optimized adapter against the existing public interface BEFORE training.
    for role, ds in data.items():
        for i in (0, len(ds)//2, len(ds)-1):
            old, new = ds.base[i], ds[i]
            if not np.array_equal(old.inputs.dense_features(), new.dense):
                raise AssertionError("A adapter changed the existing 53-feature input schema")
            expected, actual = ds.base.get_targets(i, window=old), ds.get_targets(i, window=new)
            for field in ("values", "observed", "quality_weight", "loss_mask", "artificial_mask"):
                if not np.array_equal(getattr(expected, field), getattr(actual, field)):
                    raise AssertionError("A adapter changed clean loss-only targets")
    positions = {s: sampled_indices(ds.base.index, config["train_stride"] if s == "train" else config["evaluation_stride"])
                 for s, ds in data.items()}
    device, device_info = select_device(config["device"])
    set_seed(config["seed"], config["cpu_threads"])  # probe must not change initialization
    model = SpatialBiLSTM().to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=config["learning_rate"])
    output.mkdir(parents=True)
    _json(output/"config.json", config)
    versions = _code_versions()
    environment = dict(python=sys.version, interpreter=sys.executable, platform=platform.platform(),
                       torch=str(torch.__version__), numpy=np.__version__, device=device_info,
                       seed=config["seed"], deterministic_algorithms_warn_only=True, code=versions)
    _json(output/"environment.json", environment)
    (output/"LOCAL_ONLY.txt").write_text("Local match-trained weights/embeddings/metrics. No uploads, public push or cloud services.\n")
    parameter_count = sum(p.numel() for p in model.parameters())
    print(f"Model A: {parameter_count:,} parameters; device={device}; train windows={len(positions['train']):,}", flush=True)
    stopper = EarlyStopping(config["patience"], config["min_delta"])
    history = []
    started_all = time.perf_counter()
    best_epoch = 0
    for epoch in range(config["epochs"]):
        model.train()
        data["train"].set_epoch(epoch)
        rng = np.random.default_rng(config["seed"]+epoch)
        order = rng.permutation(positions["train"])
        summed, weights, skipped, updates = 0., 0., 0, 0
        started = time.perf_counter()
        for start in range(0, len(order), config["batch_size"]):
            x, padding, target, _ = batch_arrays(data["train"], order[start:start+config["batch_size"]])
            result = train_update(model, optimizer, x, padding, target, device, delta=config["huber_delta"],
                                  feature_weights=config["feature_weights"], grad_clip=config["gradient_clip"])
            skipped += int(result["skipped"])
            if not result["skipped"]:
                summed += result["loss"]*result["weight_sum"]
                weights += result["weight_sum"]
                updates += 1
        _sync(device)
        validation = evaluate(model, data["validation"], positions["validation"], config["batch_size"], device,
                               delta=config["huber_delta"], feature_weights=config["feature_weights"], baselines=False)
        value = validation["metrics"]["bilstm"]["huber"]
        improved, stop = stopper.step(value)
        record = dict(epoch=epoch+1, train_huber=summed/weights if weights else None,
                       validation_huber=value, validation_mae=validation["metrics"]["bilstm"]["mae_standardized"],
                       skipped_batches=skipped, optimizer_updates=updates, train_target_weight=weights,
                       seconds=time.perf_counter()-started)
        history.append(record)
        if improved:
            best_epoch = epoch+1
            save_checkpoint(output/"best.pt", model, epoch=best_epoch, seed=config["seed"],
                             validation_huber=value, config=config, source_fingerprint=bases["train"].source.fingerprint,
                             split_fingerprint=manifest.fingerprint, code=versions)
        with (output/"history.csv").open("w", newline="") as handle:
            writer = csv.DictWriter(handle, fieldnames=list(record)); writer.writeheader(); writer.writerows(history)
        train_display = "no eligible targets" if record["train_huber"] is None else f"{record['train_huber']:.6f}"
        print(f"Epoch {epoch+1}: train={train_display}, validation={value:.6f}, "
              f"updates={updates}, skipped={skipped}, seconds={record['seconds']:.1f}", flush=True)
        if stop:
            break
    training_seconds = time.perf_counter()-started_all
    model, metadata = load_checkpoint(output/"best.pt", device)
    evaluation = {}
    for split in ("validation", "test"):
        evaluation[split] = {}
        for label, span in (("8-13", None), ("25", 25), ("50", 50)):
            base = open_temporal_dataset(temporal_dir, split=split)
            if span is not None:
                base.masker = SpanMasker(seed=config["seed"], minimum=span, maximum=span)
            else:
                base.masker = SpanMasker(seed=config["seed"])
            result = evaluate(model, FeatureWindowDataset(base, cache), positions[split], config["batch_size"], device,
                               delta=config["huber_delta"], feature_weights=config["feature_weights"])
            evaluation[split][label] = result
            print(f"{split}, mask={label}: " + ", ".join(f"{k} MAE={v['mae_standardized']:.6f}" for k,v in result['metrics'].items()), flush=True)
    embedding = export_embeddings(model, open_temporal_dataset(temporal_dir, split=None, mode="inference"),
                                  cache, output, config["embedding_batch_size"], device,
                                  config["collapse_variance_threshold"])
    report = dict(version="model-a-run-v1", parameter_count=parameter_count, architecture="53-64-64 / BiLSTM32x2 / 64-64-13",
                   feature_names=list(FEATURE_NAMES), target_names=list(TARGET_NAMES), environment=environment,
                   temporal_data=str(temporal_dir), config=config, support_audit=support,
                   full_window_counts={s: len(d) for s,d in data.items()},
                   used_window_counts={s: len(v) for s,v in positions.items()}, split_blocks=[vars(b) for b in manifest.blocks],
                   best_epoch=best_epoch, best_validation_huber=metadata["validation_huber"],
                   training_seconds=training_seconds, epochs_completed=len(history), history=history,
                   evaluation=evaluation, embedding_diagnostics=embedding,
                   interpretation="Reconstruction only; no evidence of contest accuracy or H/A/D performance",
                   graph_fusion_clustering_possession_implemented=False)
    _json(output/"report.json", report)
    with (output/"feature_metrics.csv").open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=["split","mask","method","feature","mae_standardized","mae_schema_units","huber","valid_cells","weight_sum"])
        writer.writeheader()
        for split, by_mask in evaluation.items():
            for label, result in by_mask.items():
                for method, metrics in result["metrics"].items():
                    for name, values in metrics["per_feature"].items():
                        writer.writerow(dict(split=split, mask=label, method=method, feature=name, **values))
    return report


def main():
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--data", default="data/derived/relational_temporal_v1_20261109")
    parser.add_argument("--feature-cache", default="data/derived/model_a_feature_cache_v1")
    parser.add_argument("--out-dir", default="data/derived/model_a_run_v1")
    parser.add_argument("--config", default="MTAH_Relational_Starter/configs/model_a_small.json")
    args = parser.parse_args()
    config = json.loads(Path(args.config).read_text())
    run_training(args.data, args.feature_cache, args.out_dir, config)


if __name__ == "__main__":
    main()
