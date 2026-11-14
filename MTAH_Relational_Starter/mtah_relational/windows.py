"""151-frame split-safe inputs and separate loss-only targets; no neural models."""
from __future__ import annotations

from dataclasses import dataclass, fields
import json
from pathlib import Path

import numpy as np

from .corruption import SpanMasker
from .data_quality import ObservationPolicy, assess_observations
from .features import geometry_features
from .preprocessing import spatial_input
from .spatial_schema import symmetric_summary
from .splits import SPLITS, TimeBlock


@dataclass(frozen=True)
class WindowSpec:
    length: int = 151
    fps: int = 25
    stride: int = 1

    def __post_init__(self):
        if not isinstance(self.length, int) or self.length < 13 or self.length % 2 != 1 \
                or self.fps != 25 or not isinstance(self.stride, int) or self.stride <= 0:
            raise ValueError("Window must have odd length >=13, positive stride and 25 FPS")

    @property
    def radius(self):
        return self.length//2


RECORD_DTYPE = np.dtype([
    ("center_frame", "int64"), ("center_row", "int64"), ("first_row", "int64"), ("last_row", "int64"),
    ("support_first", "int64"), ("support_last", "int64"), ("period", "int8"), ("block_index", "int32"),
])


@dataclass(frozen=True)
class WindowIndex:
    records: np.ndarray
    blocks: tuple
    spec: WindowSpec
    mode: str
    split: str | None
    manifest_fingerprint: str

    def __len__(self):
        return len(self.records)

    def save(self, path):
        from dataclasses import asdict
        path = Path(path)
        if path.suffix != ".npy":
            raise ValueError("Index path must end in .npy")
        header = path.with_suffix(".json")
        if path.exists() or header.exists():
            raise FileExistsError("Temporal index already exists")
        np.save(path, self.records, allow_pickle=False)
        header.write_text(json.dumps(dict(version="window-index-v1", mode=self.mode, split=self.split,
                    spec=asdict(self.spec), blocks=[asdict(b) for b in self.blocks],
                    manifest_fingerprint=self.manifest_fingerprint), indent=2)+"\n")


def load_window_index(path, *, source, manifest):
    path = Path(path)
    info = json.loads(path.with_suffix(".json").read_text())
    if info.get("version") != "window-index-v1" or info.get("manifest_fingerprint") != manifest.fingerprint \
            or info.get("mode") not in ("inference", "reconstruction") \
            or (info.get("split") not in SPLITS and not (info.get("mode") == "inference" and info.get("split") is None)):
        raise ValueError("Unrecognized or mismatched window index header")
    index = WindowIndex(np.load(path, mmap_mode="r", allow_pickle=False),
                        tuple(TimeBlock(**b) for b in info["blocks"]), WindowSpec(**info["spec"]),
                        info["mode"], info["split"], info["manifest_fingerprint"])
    if index.split is not None and index.blocks != manifest.blocks:
        raise ValueError("Saved window blocks disagree with split manifest")
    if index.split is None:
        expected = tuple(TimeBlock(p, "deployment", min(b.first for b in manifest.blocks if b.period == p),
                                   max(b.last for b in manifest.blocks if b.period == p))
                         for p in sorted({b.period for b in manifest.blocks}))
        if index.blocks != expected:
            raise ValueError("Saved deployment half boundaries disagree with manifest")
    audit_window_index(source, index)
    return index


def build_window_index(source, manifest, *, split=None, mode="reconstruction", spec=WindowSpec()):
    if mode not in ("reconstruction", "inference") or (split is not None and split not in SPLITS):
        raise ValueError("Invalid window mode or split")
    if mode == "reconstruction" and split is None:
        raise ValueError("Reconstruction requires a chronological split")
    if mode == "inference" and split is None:
        # Deployment context may cross a train/validation boundary, never a half.
        blocks = tuple(TimeBlock(p, "deployment", min(b.first for b in manifest.blocks if b.period == p),
                                 max(b.last for b in manifest.blocks if b.period == p))
                       for p in sorted({b.period for b in manifest.blocks}))
    else:
        blocks = manifest.blocks
    for b in blocks:
        inside = (source.frame_ids >= b.first) & (source.frame_ids <= b.last)
        if (source.periods[inside] != b.period).any():
            raise ValueError("Official half metadata and source periods disagree")
    records = []
    radius = spec.radius
    for block_index, block in enumerate(blocks):
        if split is not None and block.split != split:
            continue
        for period, run_first, run_last in source.runs:
            if period != block.period:
                continue
            first_row = max(run_first, int(np.searchsorted(source.frame_ids, block.first)))
            last_row = min(run_last, int(np.searchsorted(source.frame_ids, block.last, side="right"))-1)
            if first_row > last_row:
                continue
            first, last = int(source.frame_ids[first_row]), int(source.frame_ids[last_row])
            centers = (range(first+radius, last-radius+1, spec.stride) if mode == "reconstruction"
                       else range(first, last+1, spec.stride))
            for center in centers:
                left, right = max(first, center-radius), min(last, center+radius)
                center_row = first_row+center-first
                records.append((center, center_row, first_row+left-first, first_row+right-first,
                                left, right, period, block_index))
    array = np.array(records, dtype=RECORD_DTYPE)
    array.setflags(write=False)
    return WindowIndex(array, blocks, spec, mode, split, manifest.fingerprint)


@dataclass(frozen=True)
class WindowRouting:
    frame_ids: np.ndarray  # real IDs; -1 only for padding
    requested_frame_ids: np.ndarray  # ideal centered timeline, routing only
    center_frame: int
    center_index: int
    period: int
    split: str
    block_index: int
    mask_epoch: int = 0


@dataclass(frozen=True)
class ModelInputs:
    values: np.ndarray
    feature_valid: np.ndarray
    quality: np.ndarray
    quality_valid: np.ndarray  # visible eligibility only; hidden/padding=False
    team_structure: np.ndarray
    team_structure_valid: np.ndarray
    raw_xy_m: tuple  # [T] of [N_t,2], or [B] of [T] ragged tuples after collation
    raw_team: tuple
    artificial_mask: np.ndarray
    padding_mask: np.ndarray

    def dense_features(self):
        """The A-model's 53 channels; padding stays a separate temporal mask."""
        return np.concatenate([self.values, self.feature_valid.astype(np.float32), self.quality,
                               self.artificial_mask[..., None].astype(np.float32)], axis=-1)


@dataclass(frozen=True)
class InputWindow:
    inputs: ModelInputs
    routing: WindowRouting


@dataclass(frozen=True)
class LossTargets:
    values: np.ndarray
    observed: np.ndarray
    quality_weight: np.ndarray
    loss_mask: np.ndarray
    artificial_mask: np.ndarray


class WindowDataset:
    """NumPy map-style dataset, ready for a later A/B/C model adapter.

    __getitem__ NEVER returns clean targets. get_targets is a separate loss-only
    call. Mask choices depend only on immutable routing/seed/epoch, not quality.
    Inference has no artificial masks/targets and pads at each contiguous run.
    """

    def __init__(self, source, index, normalizer, manifest, *, policy=ObservationPolicy(),
                 masker=SpanMasker(), epoch=0):
        if index.manifest_fingerprint != manifest.fingerprint:
            raise ValueError("Window index split manifest mismatch")
        if index.split is not None and index.blocks != manifest.blocks:
            raise ValueError("Window index raw-time blocks disagree with manifest")
        normalizer.validate(manifest, policy, source.pitch)
        audit_window_index(source, index)
        self.source, self.index, self.normalizer = source, index, normalizer
        self.policy, self.masker = policy, masker
        self.set_epoch(epoch)

    def __len__(self):
        return len(self.index)

    def set_epoch(self, epoch):
        if not isinstance(epoch, int) or epoch < 0:
            raise ValueError("epoch must be a nonnegative integer")
        self.epoch = epoch

    def _routing(self, item):
        rec = self.index.records[item]
        spec = self.index.spec
        center = int(rec["center_frame"])
        requested = np.arange(center-spec.radius, center+spec.radius+1, dtype=np.int64)
        actual = requested.copy()
        actual[(actual < rec["support_first"]) | (actual > rec["support_last"])] = -1
        valid = actual >= 0
        rows = np.arange(int(rec["first_row"]), int(rec["last_row"])+1)
        if len(rows) != valid.sum() or not np.array_equal(self.source.frame_ids[rows], actual[valid]) \
                or not (self.source.periods[rows] == rec["period"]).all():
            raise ValueError("Window support no longer matches source frame IDs or period")
        block = self.index.blocks[int(rec["block_index"])]
        routing = WindowRouting(actual, requested, center, spec.radius, int(rec["period"]),
                                block.split, int(rec["block_index"]), self.epoch if block.split == "train" else 0)
        return routing, rows

    def _mask(self, routing):
        if self.index.mode == "inference":
            return np.zeros(self.index.spec.length, dtype=bool)
        return self.masker.sample(self.index.spec.length, center_frame=routing.center_frame,
                                  period=routing.period, split=routing.split, epoch=routing.mask_epoch)

    def __getitem__(self, item):
        routing, rows = self._routing(item)
        length = self.index.spec.length
        artificial = self._mask(routing)
        padding = routing.frame_ids < 0
        values = np.zeros((length, 24), dtype=np.float32)
        feature_valid = np.zeros((length, 24), dtype=bool)
        quality = np.zeros((length, 4), dtype=np.float32)
        quality_valid = np.zeros(length, dtype=bool)
        ordered = np.zeros((length, 2, 10), dtype=np.float32)
        ordered_valid = np.zeros((length, 2, 10), dtype=bool)
        raw_xy = [np.empty((0, 2), dtype=np.float32) for _ in range(length)]
        raw_team = [np.empty(0, dtype=np.int8) for _ in range(length)]
        for t, row in zip(np.flatnonzero(~padding), rows):
            if artificial[t]:
                # Do not fetch hidden raw data; no clean counts/masks/topology.
                continue
            frame = self.source.frame(int(row))
            visible = spatial_input(frame, self.source.pitch, self.policy)
            _, q = assess_observations(frame, self.source.pitch, self.policy)
            values[t], feature_valid[t] = visible.values, visible.feature_valid
            quality[t], quality_valid[t] = visible.quality, q["quality_valid"]
            ordered[t], ordered_valid[t] = visible.team_structure, visible.team_structure_valid
            raw_xy[t], raw_team[t] = visible.xy_m.copy(), visible.team.copy()
        values = self.normalizer.transform(values, feature_valid)
        values[artificial | padding] = 0
        inputs = ModelInputs(values, feature_valid, quality, quality_valid, ordered, ordered_valid,
                             tuple(raw_xy), tuple(raw_team), artificial, padding)
        return InputWindow(inputs, routing)

    def get_targets(self, item, *, window=None):
        if self.index.mode == "inference":
            raise ValueError("No reconstruction targets for inference windows")
        routing, _ = self._routing(item)
        if window is not None:
            if not isinstance(window, InputWindow) or not np.array_equal(window.routing.frame_ids, routing.frame_ids) \
                    or window.routing.center_frame != routing.center_frame or window.routing.split != routing.split:
                raise ValueError("Target request does not match the supplied input window")
            # Epoch snapshot makes an already materialized sample replayable.
            routing = window.routing
        artificial = self._mask(routing)
        if window is not None and not np.array_equal(window.inputs.artificial_mask, artificial):
            raise ValueError("Input and loss-only target masks disagree")
        length = self.index.spec.length
        values = np.zeros((length, 13), dtype=np.float64)
        observed = np.zeros((length, 13), dtype=bool)
        weight = np.zeros((length, 13), dtype=np.float32)
        for t in np.flatnonzero(artificial):
            frame = self.source.frame(self.source.position(int(routing.frame_ids[t])))
            xy, q = assess_observations(frame, self.source.pitch, self.policy)
            f = geometry_features(xy[frame.team_codes == 0], xy[frame.team_codes == 1])
            clean, valid = symmetric_summary(f, self.source.pitch)
            observed[t] = valid[:13]
            values[t] = np.where(valid[:13], clean[:13], 0)
            weight[t] = float(q["quality_valid"])*(self.normalizer.counts[:13] > 0)
        scaled = self.normalizer.transform(values, observed)
        loss_mask = artificial[:, None] & observed & (weight > 0)
        return LossTargets(scaled, observed, weight, loss_mask, artificial)


def collate_inputs(windows):
    if not windows or not all(isinstance(w, InputWindow) for w in windows):
        raise ValueError("Collate input windows only; never include loss targets")
    items = [w.inputs for w in windows]
    kwargs = {}
    for f in fields(ModelInputs):
        values = [getattr(x, f.name) for x in items]
        kwargs[f.name] = tuple(values) if f.name in ("raw_xy_m", "raw_team") else np.stack(values)
    return ModelInputs(**kwargs), tuple(w.routing for w in windows)


def collate_targets(targets):
    if not targets or not all(isinstance(t, LossTargets) for t in targets):
        raise ValueError("Collate loss targets separately")
    return LossTargets(**{f.name: np.stack([getattr(x, f.name) for x in targets]) for f in fields(LossTargets)})


def audit_window_index(source, index):
    """Vectorized audit of EVERY index entry, without materializing all windows."""
    r = index.records
    if not len(r):
        return dict(windows=0, padded_windows=0, original_frame_alignment=True, no_record_gap_crossing=True)
    if r.dtype != RECORD_DTYPE or (r["first_row"] < 0).any() or (r["last_row"] >= len(source)).any() \
            or (r["first_row"] > r["center_row"]).any() or (r["center_row"] > r["last_row"]).any() \
            or (r["block_index"] < 0).any() or (r["block_index"] >= len(index.blocks)).any():
        raise ValueError("Invalid window index row or block references")
    segment = np.cumsum(np.r_[True, (np.diff(source.frame_ids) != 1) | (np.diff(source.periods) != 0)])
    first, center, last = r["first_row"], r["center_row"], r["last_row"]
    if (source.frame_ids[center] != r["center_frame"]).any() \
            or (source.frame_ids[first] != r["support_first"]).any() \
            or (source.frame_ids[last] != r["support_last"]).any() \
            or (segment[first] != segment[last]).any() or (source.periods[center] != r["period"]).any() \
            or (r["period"] == 0).any() or (r["center_frame"]-r["support_first"] > index.spec.radius).any() \
            or (r["support_last"]-r["center_frame"] > index.spec.radius).any():
        raise ValueError("Window crosses a record gap/period or lost original-frame alignment")
    for bi in np.unique(r["block_index"]):
        part = r[r["block_index"] == bi]
        block = index.blocks[int(bi)]
        if (part["support_first"] < block.first).any() or (part["support_last"] > block.last).any() \
                or (part["period"] != block.period).any() \
                or (index.split is not None and block.split != index.split):
            raise ValueError("Window crossed its raw-time block boundary")
    span = r["support_last"]-r["support_first"]+1
    if index.mode == "reconstruction" and ((span != index.spec.length).any()
            or (r["center_frame"]-r["support_first"] != index.spec.radius).any()):
        raise ValueError("Reconstruction windows must have full centered context")
    return dict(windows=len(r), padded_windows=int((span < index.spec.length).sum()),
                original_frame_alignment=True, no_record_gap_crossing=True)
