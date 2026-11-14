"""Fast A-model adapter preserving the existing WindowDataset/masking contract.

Only frame-local derived inputs are cached. A query chooses raw-time corruption
FIRST, reads cached inputs ONLY for visible rows, and obtains clean targets via
a separate loss-only accessor. No temporal features/scalers are fitted here.
"""
from dataclasses import asdict, dataclass
import hashlib
import json
import os
from pathlib import Path
import tempfile

import numpy as np

from .data_quality import assess_observations
from .preprocessing import spatial_input
from .windows import LossTargets, WindowDataset


def normalizer_signature(base):
    h = hashlib.sha256(base.normalizer.manifest_fingerprint.encode())
    for x in (base.normalizer.median, base.normalizer.scale, base.normalizer.counts):
        h.update(x.tobytes())
    h.update(json.dumps(asdict(base.policy), sort_keys=True).encode())
    return h.hexdigest()


class FeatureFrameCache:
    def __init__(self, path, base):
        self.path = Path(path)
        metadata = json.loads((self.path/"cache.json").read_text())
        if metadata.get("version") != "model-a-frame-cache-v1" \
                or metadata.get("source_fingerprint") != base.source.fingerprint \
                or metadata.get("normalizer_signature") != normalizer_signature(base):
            raise ValueError("A feature cache source/scaler mismatch")
        for name, shape in {
            "input_values": (len(base.source), 24), "input_valid": (len(base.source), 24),
            "quality": (len(base.source), 4), "quality_valid": (len(base.source),),
            "target_values": (len(base.source), 13), "target_observed": (len(base.source), 13),
        }.items():
            a = np.load(self.path/f"{name}.npy", mmap_mode="r", allow_pickle=False)
            if a.shape != shape:
                raise ValueError("A feature cache shape mismatch")
            setattr(self, name, a)


def build_feature_cache(base, output_dir, *, progress=False):
    output = Path(output_dir).resolve()
    if output.exists():
        return FeatureFrameCache(output, base)
    output.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.TemporaryDirectory(prefix=".a_features_", dir=output.parent) as temp:
        dest = Path(temp)
        arrays = {}
        for name, shape, dtype in (
            ("input_values", (len(base.source), 24), "float32"),
            ("input_valid", (len(base.source), 24), "bool"),
            ("quality", (len(base.source), 4), "float32"),
            ("quality_valid", (len(base.source),), "bool"),
            ("target_values", (len(base.source), 13), "float32"),
            ("target_observed", (len(base.source), 13), "bool"),
        ):
            arrays[name] = np.lib.format.open_memmap(dest/f"{name}.npy", mode="w+", dtype=dtype, shape=shape)
        for row in range(len(base.source)):
            frame = base.source.frame(row)
            visible = spatial_input(frame, base.source.pitch, base.policy)
            _, q = assess_observations(frame, base.source.pitch, base.policy)
            # Match the existing WindowDataset's input float32 rounding exactly.
            arrays["input_values"][row] = base.normalizer.transform(visible.values.astype(np.float32), visible.feature_valid)
            arrays["input_valid"][row] = visible.feature_valid
            arrays["quality"][row] = visible.quality
            arrays["quality_valid"][row] = q["quality_valid"]
            # Loss-only targets preserve the original clean-summary precision.
            arrays["target_values"][row] = base.normalizer.transform(visible.values[:13], visible.feature_valid[:13])
            arrays["target_observed"][row] = visible.feature_valid[:13]
            if progress and (row+1) % 50000 == 0:
                print(f"Cached {row+1:,} independent spatial feature frames", flush=True)
        for a in arrays.values():
            a.flush()
        del a
        arrays.clear()
        (dest/"cache.json").write_text(json.dumps(dict(version="model-a-frame-cache-v1",
            source_fingerprint=base.source.fingerprint, normalizer_signature=normalizer_signature(base),
            frame_local_only=True, clean_targets_loss_only=True), indent=2)+"\n")
        if output.exists():
            raise FileExistsError("A feature cache appeared during generation")
        os.rename(dest, output)
    return FeatureFrameCache(output, base)


@dataclass(frozen=True)
class AInputWindow:
    dense: np.ndarray
    padding_mask: np.ndarray
    artificial_mask: np.ndarray
    routing: object


class FeatureWindowDataset:
    def __init__(self, base: WindowDataset, cache: FeatureFrameCache):
        self.base, self.cache = base, cache

    def __len__(self):
        return len(self.base)

    def set_epoch(self, epoch):
        self.base.set_epoch(epoch)

    def __getitem__(self, item):
        routing, rows = self.base._routing(item)
        hidden = self.base._mask(routing)
        padding = routing.frame_ids < 0
        x = np.zeros((self.base.index.spec.length, 53), dtype=np.float32)
        times = np.flatnonzero(~padding)
        visible = ~hidden[times]
        times, rows = times[visible], rows[visible]
        # Never read a hidden row from any clean feature/validity/quality array.
        x[times, :24] = self.cache.input_values[rows]
        x[times, 24:48] = self.cache.input_valid[rows]
        x[times, 48:52] = self.cache.quality[rows]
        x[:, 52] = hidden
        return AInputWindow(x, padding, hidden, routing)

    def get_targets(self, item, *, window):
        if self.base.index.mode == "inference":
            raise ValueError("Inference has no reconstruction target")
        current, _ = self.base._routing(item)
        if not np.array_equal(current.frame_ids, window.routing.frame_ids):
            raise ValueError("Mismatched A-model target routing")
        hidden = self.base._mask(window.routing)
        if not np.array_equal(hidden, window.artificial_mask):
            raise ValueError("Mismatched A-model target mask")
        length = self.base.index.spec.length
        times = np.flatnonzero(hidden)
        rows = np.searchsorted(self.base.source.frame_ids, window.routing.frame_ids[times])
        values = np.zeros((length, 13), dtype=np.float32)
        observed = np.zeros((length, 13), dtype=bool)
        weights = np.zeros((length, 13), dtype=np.float32)
        values[times] = self.cache.target_values[rows]
        observed[times] = self.cache.target_observed[rows]
        weights[times] = self.cache.quality_valid[rows, None] * (self.base.normalizer.counts[:13] > 0)
        return LossTargets(values, observed, weights, observed & hidden[:, None] & (weights > 0), hidden)


def sampled_indices(index, stride):
    if not isinstance(stride, int) or stride <= 0:
        raise ValueError("Sampling stride must be positive")
    return np.concatenate([np.flatnonzero(index.records["block_index"] == b)[::stride]
                           for b in np.unique(index.records["block_index"])])


def batch_arrays(dataset, indices, *, with_targets=True):
    windows = [dataset[int(i)] for i in indices]
    x = np.stack([w.dense for w in windows])
    padding = np.stack([w.padding_mask for w in windows])
    targets = None
    if with_targets:
        from .windows import collate_targets
        targets = collate_targets([dataset.get_targets(int(i), window=w) for i, w in zip(indices, windows)])
    return x, padding, targets, windows
