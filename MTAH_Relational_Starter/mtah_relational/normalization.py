"""Frozen median imputation/IQR scaling fitted on unique TRAIN observations only."""
from dataclasses import asdict, dataclass
import json
from pathlib import Path

import numpy as np

from .data_quality import ObservationPolicy, assess_observations
from .features import geometry_features
from .spatial_schema import FEATURE_NAMES, SCHEMA_VERSION, symmetric_summary


@dataclass(frozen=True)
class TrainOnlyNormalizer:
    median: np.ndarray
    scale: np.ndarray
    counts: np.ndarray
    fit_frames: int
    requested_train_frames: int
    manifest_fingerprint: str
    source_fingerprint: str
    policy: dict
    pitch: tuple

    def __post_init__(self):
        for name, dtype in (("median", float), ("scale", float), ("counts", np.int64)):
            a = np.array(getattr(self, name), dtype=dtype, copy=True)
            if a.shape != (len(FEATURE_NAMES),) or not np.isfinite(a).all():
                raise ValueError("Invalid normalizer statistics")
            a.setflags(write=False)
            object.__setattr__(self, name, a)
        if (self.scale <= 0).any() or (self.counts < 0).any() or self.fit_frames <= 0:
            raise ValueError("Invalid training normalizer counts or scales")
        object.__setattr__(self, "pitch", tuple(self.pitch))

    def validate(self, manifest, policy, pitch=None):
        if self.manifest_fingerprint != manifest.fingerprint or self.policy != asdict(policy) \
                or (pitch is not None and tuple(pitch) != self.pitch):
            raise ValueError("Normalizer's train split, policy or pitch does not match")

    def transform(self, values, valid):
        """Last dimension may be 24 inputs or the first 13 reconstruction targets.

        Unavailable values are imputed with TRAIN medians, hence zero after
        scaling. Artificial hidden/padded data are reset by the caller too.
        No state updates, ffill/bfill, time smoothing or interpolation.
        """
        values = np.asarray(values, dtype=float)
        valid = np.asarray(valid, dtype=bool)
        if values.shape != valid.shape or values.ndim == 0 or values.shape[-1] not in (13, 24):
            raise ValueError("Invalid values/mask for normalizer")
        n = values.shape[-1]
        mask = valid & np.isfinite(values)
        filled = np.where(mask, values, self.median[:n])
        result = (filled-self.median[:n])/self.scale[:n]
        return np.where(mask, result, 0).astype(np.float32)

    def save(self, path):
        data = dict(version="train-normalizer-v1", feature_schema=SCHEMA_VERSION,
                    feature_names=list(FEATURE_NAMES), median=self.median.tolist(), scale=self.scale.tolist(),
                    counts=self.counts.tolist(), fit_frames=self.fit_frames,
                    requested_train_frames=self.requested_train_frames,
                    manifest_fingerprint=self.manifest_fingerprint, source_fingerprint=self.source_fingerprint,
                    policy=self.policy, pitch=list(self.pitch), fit_role="train_only_unique_frames")
        Path(path).write_text(json.dumps(data, indent=2, allow_nan=False)+"\n")

    @classmethod
    def load(cls, path, *, manifest, policy=ObservationPolicy()):
        data = json.loads(Path(path).read_text())
        if data.get("version") != "train-normalizer-v1" or data.get("feature_schema") != SCHEMA_VERSION \
                or data.get("feature_names") != list(FEATURE_NAMES) \
                or data.get("fit_role") != "train_only_unique_frames":
            raise ValueError("Unrecognized train-only normalizer")
        keys = ("median", "scale", "counts", "fit_frames", "requested_train_frames",
                "manifest_fingerprint", "source_fingerprint", "policy", "pitch")
        result = cls(**{k: data[k] for k in keys})
        result.validate(manifest, policy)
        return result


def fit_train_normalizer(source, manifest, policy=ObservationPolicy(), *, progress=False):
    """The only fitting API selects chronological TRAIN raw frames itself.

    Each observed frame is counted once, not once per overlapping window. It is
    legitimate for frozen train distribution statistics to be reused by every
    sample; counterfactual hidden-input tests hold these fitted parameters fixed.
    """
    select = np.zeros(len(source), dtype=bool)
    for b in manifest.blocks:
        if b.split != "train":
            continue
        inside = (source.frame_ids >= b.first) & (source.frame_ids <= b.last)
        if (source.periods[inside] != b.period).any():
            raise ValueError("Train raw-time block disagrees with source periods")
        select |= inside
    rows = np.flatnonzero(select)
    values = np.full((len(rows), len(FEATURE_NAMES)), np.nan)
    eligible = 0
    for i, row in enumerate(rows):
        frame = source.frame(int(row))
        xy, q = assess_observations(frame, source.pitch, policy)
        if q["quality_valid"]:
            f = geometry_features(xy[source.team_codes == 0], xy[source.team_codes == 1])
            v, valid = symmetric_summary(f, source.pitch)
            values[i] = np.where(valid, v, np.nan)
            eligible += 1
        if progress and (i+1) % 25000 == 0:
            print(f"Inspected {i+1:,} unique TRAIN observations for scaling", flush=True)
    if eligible == 0:
        raise ValueError("No quality-eligible training observations for normalization")
    median, scale, counts = np.zeros(24), np.ones(24), np.zeros(24, dtype=np.int64)
    for k in range(24):
        v = values[np.isfinite(values[:, k]), k]
        counts[k] = len(v)
        if len(v):
            median[k] = np.median(v)
            q25, q75 = np.percentile(v, [25, 75])
            scale[k] = q75-q25 if q75 > q25 else 1.0
    return TrainOnlyNormalizer(median, scale, counts, eligible, len(rows), manifest.fingerprint,
                               source.fingerprint, asdict(policy), source.pitch)
