"""Local-only temporal data preparation and audits, WITHOUT neural training.

From the official root:
PYTHONPATH="$PWD/MTAH_Relational_Starter:$PWD" python -m mtah_relational.temporal_cli build
"""
from __future__ import annotations

import argparse
from dataclasses import asdict, fields
import json
import os
from pathlib import Path
import sys
import tempfile

import numpy as np

from .corruption import SpanMasker
from .data_quality import ObservationPolicy, PlayerFrame
from .frame_store import build_frame_cache, open_frame_cache
from .normalization import fit_train_normalizer, TrainOnlyNormalizer
from .splits import SPLITS, SplitManifest, make_split_manifest, assert_disjoint_support
from .windows import (WindowDataset, WindowSpec, audit_window_index, build_window_index,
                      collate_inputs, collate_targets, load_window_index)


class _CounterfactualSource:
    """Audit-only in-memory substitutions. Never write a corrected tracking file."""

    def __init__(self, source, hidden_frames, *, forbid_reads=False):
        self.source, self.hidden = source, frozenset(int(f) for f in hidden_frames)
        self.forbid_reads = forbid_reads

    def __getattr__(self, key):
        return getattr(self.source, key)

    def __len__(self):
        return len(self.source)

    def frame(self, row):
        if int(self.source.frame_ids[row]) in self.hidden:
            if self.forbid_reads:
                raise AssertionError("Hidden raw observation was accessed by model input generation")
            n = len(self.source.team_codes)
            xy = np.full((n, 2), [999., -999.])
            flags = np.zeros(n, dtype=bool)
            return PlayerFrame(int(self.source.frame_ids[row]), int(self.source.periods[row]), xy,
                               self.source.team_codes, flags, flags, flags)
        return self.source.frame(row)


def _equal_inputs(left, right):
    for field in fields(left):
        a, b = getattr(left, field.name), getattr(right, field.name)
        if isinstance(a, tuple):
            equal = all(np.array_equal(x, y) for x, y in zip(a, b, strict=True))
        else:
            equal = np.array_equal(a, b)
        if not equal:
            raise AssertionError("Hidden raw perturbation changed a corrupted model input")


def audit_materialized_samples(source, index, normalizer, manifest, policy, masker):
    ds = WindowDataset(source, index, normalizer, manifest, policy=policy, masker=masker)
    # First/middle/last of each half's role block: fixed before seeing x/y/Q.
    sample_indices = []
    for block in np.unique(index.records["block_index"]):
        positions = np.flatnonzero(index.records["block_index"] == block)
        sample_indices.extend(int(positions[i]) for i in sorted({0, len(positions)//2, len(positions)-1}))
    windows, targets = [], []
    cells, empty_targets = 0, 0
    for i in sample_indices:
        window = ds[i]
        target = ds.get_targets(i, window=window)
        if window.inputs.values.shape != (index.spec.length, 24) \
                or target.values.shape != (index.spec.length, 13) \
                or not np.isfinite(window.inputs.dense_features()).all() or not np.isfinite(target.values).all() \
                or not np.array_equal(target.artificial_mask, window.inputs.artificial_mask):
            raise AssertionError("Materialized input/target shape, mask or finiteness failed")
        hidden = window.inputs.artificial_mask
        if not 8 <= hidden.sum() <= 13 or not (np.diff(np.flatnonzero(hidden)) == 1).all() \
                or window.inputs.quality[hidden].any() or window.inputs.feature_valid[hidden].any() \
                or window.inputs.team_structure_valid[hidden].any() \
                or any(len(window.inputs.raw_xy_m[t]) for t in np.flatnonzero(hidden)):
            raise AssertionError("Raw and engineered hidden-time masking contract failed")
        windows.append(window)
        targets.append(target)
        cells += int(target.loss_mask.sum())
        empty_targets += int(not target.loss_mask.any())
    if windows:
        batch, _ = collate_inputs(windows[:2])
        loss_only = collate_targets(targets[:2])
        if batch.values.shape[1:] != (index.spec.length, 24) or loss_only.values.shape[1:] != (index.spec.length, 13):
            raise AssertionError("Batch collation failed")
        i = sample_indices[len(sample_indices)//2]
        original = ds[i]
        hidden_frames = original.routing.frame_ids[original.inputs.artificial_mask]
        altered = _CounterfactualSource(source, hidden_frames)
        changed = WindowDataset(altered, index, normalizer, manifest, policy=policy, masker=masker)[i]
        _equal_inputs(original.inputs, changed.inputs)
        guarded = _CounterfactualSource(source, hidden_frames, forbid_reads=True)
        WindowDataset(guarded, index, normalizer, manifest, policy=policy, masker=masker)[i]
    return dict(materialized_windows=len(windows), eligible_target_cells=cells,
                zero_loss_sample_windows=empty_targets, hidden_sentinel_passed=True,
                hidden_raw_read_guard_passed=True, all_windows_materialized=False)


def build_temporal_data(*, meta_path, raw_path, output_dir, pitch, n_frames, blocks,
                        expected_slots=46, policy=ObservationPolicy(), fractions=(.7, .15, .15),
                        spec=WindowSpec(), seed=2027, progress=False):
    output = Path(output_dir).resolve()
    if output.exists():
        raise FileExistsError("Temporal output already exists; choose a new local directory")
    manifest = make_split_manifest(blocks, fractions)
    masker = SpanMasker(seed=seed)
    output.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.TemporaryDirectory(prefix=".temporal_data_", dir=output.parent) as temp:
        dest = Path(temp)
        source = build_frame_cache(meta_path=meta_path, raw_path=raw_path, output_dir=dest/"raw_cache",
                                   pitch=pitch, n_frames=n_frames, blocks=blocks,
                                   expected_slots=expected_slots, progress=progress)
        manifest.save(dest/"split_manifest.json")
        indices = {s: build_window_index(source, manifest, split=s, spec=spec) for s in SPLITS}
        if any(not len(i) for i in indices.values()):
            raise ValueError("At least one split has no full centered windows; data blocks are too short")
        supports = assert_disjoint_support(indices.values(), manifest)
        routing_audits = {s: audit_window_index(source, i) for s, i in indices.items()}
        for split, index in indices.items():
            index.save(dest/f"{split}_windows.npy")
        deployment_spec = WindowSpec(length=spec.length, fps=spec.fps, stride=1)
        deployment = build_window_index(source, manifest, mode="inference", spec=deployment_spec)
        deployment_audit = audit_window_index(source, deployment)
        deployment.save(dest/"deployment_windows.npy")
        if progress:
            print("All temporal indices aligned; cross-split raw-support overlap is zero", flush=True)
        normalizer = fit_train_normalizer(source, manifest, policy, progress=progress)
        normalizer.save(dest/"train_normalizer.json")
        samples = {s: audit_materialized_samples(source, index, normalizer, manifest, policy, masker)
                   for s, index in indices.items()}
        inference_ds = WindowDataset(source, deployment, normalizer, manifest, policy=policy)
        inference_samples = []
        for bi in np.unique(deployment.records["block_index"]):
            locations = np.flatnonzero(deployment.records["block_index"] == bi)
            for item in (int(locations[0]), int(locations[-1])):
                window = inference_ds[item]
                if window.inputs.artificial_mask.any() or window.routing.frame_ids[spec.radius] \
                        != window.routing.center_frame or not window.inputs.padding_mask.any():
                    raise AssertionError("Deployment boundary padding failed")
                inference_samples.append(item)
        report = dict(version="temporal-data-v1", source_records=len(source), fps=source.fps,
                      source_fingerprint=source.fingerprint, window=asdict(spec), mask=asdict(masker),
                      split_manifest_fingerprint=manifest.fingerprint, observation_policy=asdict(policy),
                      reconstruction_indices=routing_audits, support_audit=supports,
                      normalizer=dict(fit_role="train_only_unique_frames", fit_frames=normalizer.fit_frames,
                                      requested_train_frames=normalizer.requested_train_frames,
                                      feature_observation_counts=normalizer.counts.tolist()),
                      materialization_audit=samples, deployment_index=deployment_audit,
                      deployment_boundary_samples=len(inference_samples),
                      excluded_inputs=["ball", "speed", "smart_tagging", "broadcast", "public_gt"],
                      graph_encoder_implemented=False, bilstm_implemented=False, training_performed=False,
                      runtime=dict(interpreter=sys.executable, python=sys.version.split()[0], numpy=np.__version__))
        (dest/"temporal_report.json").write_text(json.dumps(report, indent=2, allow_nan=False)+"\n")
        (dest/"dataset_config.json").write_text(json.dumps(dict(window=asdict(spec), masker=asdict(masker),
                        policy=asdict(policy), feature_count=24, target_count=13, team_structure_shape=[2, 10],
                        cache="raw_cache", inputs_targets_separate=True,
                        precomputed_temporal_features_accepted=False), indent=2)+"\n")
        (dest/"LOCAL_ONLY.txt").write_text(
            "Licensed match-derived cache, indices and normalization statistics: keep local.\n"
            "Do not upload/push to public GitHub or cloud LLMs. No training or manual labels.\n")
        if output.exists():
            raise FileExistsError("Temporal output appeared during generation")
        os.rename(dest, output)
    return report


def open_temporal_dataset(path, *, split="train", mode="reconstruction", epoch=0):
    """Reopen immutable artifacts without fitting anything on validation/test.

    mode='inference', split=None uses the saved full-half deployment index.
    Split-restricted inference is rebuilt cheaply with its own boundary padding.
    """
    path = Path(path)
    config = json.loads((path/"dataset_config.json").read_text())
    manifest = SplitManifest.load(path/"split_manifest.json")
    policy = ObservationPolicy(**config["policy"])
    source = open_frame_cache(path/"raw_cache")
    normalizer = TrainOnlyNormalizer.load(path/"train_normalizer.json", manifest=manifest, policy=policy)
    if mode == "reconstruction" and split in SPLITS:
        index = load_window_index(path/f"{split}_windows.npy", source=source, manifest=manifest)
    elif mode == "inference" and split is None:
        index = load_window_index(path/"deployment_windows.npy", source=source, manifest=manifest)
    elif mode == "inference" and split in SPLITS:
        index = build_window_index(source, manifest, split=split, mode=mode,
                                    spec=WindowSpec(length=config["window"]["length"]))
    else:
        raise ValueError("Invalid saved dataset split/mode")
    if index.mode != mode or index.split != split:
        raise ValueError("Saved index role/mode disagrees with the requested dataset")
    return WindowDataset(source, index, normalizer, manifest, policy=policy,
                          masker=SpanMasker(**config["masker"]), epoch=epoch)


def main():
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    sub = parser.add_subparsers(dest="command", required=True)
    build = sub.add_parser("build", help="Local raw cache, chronological indices and masking/scaler audits")
    build.add_argument("--out-dir", default="data/derived/relational_temporal_v1")
    build.add_argument("--stride", type=int, default=1)
    build.add_argument("--seed", type=int, default=2027)
    args = parser.parse_args()
    import games
    paths = games.asset_paths(source="tactical")
    report = build_temporal_data(meta_path=paths["atd_meta"], raw_path=paths["atd_raw"],
                                 output_dir=args.out_dir, pitch=games.GAMES[games.GAME]["pitch"],
                                 n_frames=games.source_frames("tactical"), blocks=games.half_video_blocks(games.GAME),
                                 spec=WindowSpec(stride=args.stride), seed=args.seed, progress=True)
    print("Completed local temporal-data audits:", report["support_audit"]["window_counts"])
    print("No neural model or training was run.")


if __name__ == "__main__":
    main()
