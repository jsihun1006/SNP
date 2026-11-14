"""Chronological raw-time blocks; no random-window split, labels or GT."""
from dataclasses import asdict, dataclass
import hashlib
import json
from pathlib import Path

import numpy as np

SPLITS = ("train", "validation", "test")


@dataclass(frozen=True)
class TimeBlock:
    period: int
    split: str
    first: int
    last: int


@dataclass(frozen=True)
class SplitManifest:
    blocks: tuple[TimeBlock, ...]
    fractions: tuple[float, float, float]
    fps: int = 25

    def __post_init__(self):
        if self.fps != 25 or len(self.fractions) != 3 or not np.isfinite(self.fractions).all() \
                or min(self.fractions) <= 0 or not np.isclose(sum(self.fractions), 1):
            raise ValueError("Invalid chronological split fractions or FPS")
        last = -1
        seen = set()
        for b in self.blocks:
            if b.period not in (1, 2) or b.split not in SPLITS or not 0 <= b.first <= b.last \
                    or b.first <= last or (b.period, b.split) in seen:
                raise ValueError("Invalid, overlapping or repeated raw-time blocks")
            last = b.last
            seen.add((b.period, b.split))

    @property
    def fingerprint(self):
        return hashlib.sha256(json.dumps(asdict(self), sort_keys=True).encode()).hexdigest()

    def split_of(self, frame):
        return next((b.split for b in self.blocks if b.first <= frame <= b.last), None)

    def save(self, path):
        Path(path).write_text(json.dumps(dict(version="temporal-split-v1", **asdict(self),
                                             fingerprint=self.fingerprint), indent=2)+"\n")

    @classmethod
    def load(cls, path):
        data = json.loads(Path(path).read_text())
        if data.get("version") != "temporal-split-v1":
            raise ValueError("Unrecognized split manifest")
        result = cls(tuple(TimeBlock(**b) for b in data["blocks"]), tuple(data["fractions"]), data["fps"])
        if result.fingerprint != data.get("fingerprint"):
            raise ValueError("Split manifest fingerprint mismatch")
        return result


def make_split_manifest(half_blocks, fractions=(.7, .15, .15)):
    fractions = tuple(float(x) for x in fractions)
    # Validate fractions even when there are no half blocks.
    SplitManifest((), fractions)
    result = []
    for period, first, last in half_blocks:
        n = last-first+1
        sizes = (int(n*fractions[0]), int(n*fractions[1]))
        sizes += (n-sum(sizes),)
        if min(sizes) < 1:
            raise ValueError("Half too short to allocate three positive raw-time blocks")
        start = first
        for split, size in zip(SPLITS, sizes):
            result.append(TimeBlock(period, split, start, start+size-1))
            start += size
    return SplitManifest(tuple(result), fractions)


def assert_disjoint_support(indices, manifest):
    """Audit actual raw INPUT+TARGET supports, not merely center IDs.

    Same-split overlapping windows are expected. Deployed full-half inference is
    deliberately excluded from cross-validation and is rejected by this audit.
    """
    intervals = {s: [] for s in SPLITS}
    counts = {s: 0 for s in SPLITS}
    for index in indices:
        if index.mode != "reconstruction" or index.split not in SPLITS:
            raise ValueError("Audit requires split-restricted reconstruction indices")
        if index.manifest_fingerprint != manifest.fingerprint:
            raise ValueError("Window split manifest mismatch")
        for block_index in np.unique(index.records["block_index"]):
            b = manifest.blocks[int(block_index)]
            r = index.records[index.records["block_index"] == block_index]
            if b.split != index.split or (r["period"] != b.period).any() \
                    or (r["support_first"] < b.first).any() or (r["support_last"] > b.last).any() \
                    or (r["support_last"]-r["support_first"] != index.spec.length-1).any() \
                    or (r["center_frame"]-r["support_first"] != index.spec.radius).any():
                raise ValueError("A window's raw support escaped its chronological block")
            for first, last in zip(r["support_first"], r["support_last"]):
                intervals[index.split].append((int(first), int(last)))
            counts[index.split] += len(r)
    merged = {}
    for split, ranges in intervals.items():
        union = []
        for first, last in sorted(ranges):
            if union and first <= union[-1][1]+1:
                union[-1] = (union[-1][0], max(last, union[-1][1]))
            else:
                union.append((first, last))
        merged[split] = union
    for i, left in enumerate(SPLITS):
        for right in SPLITS[i+1:]:
            if any(max(a, c) <= min(b, d) for a, b in merged[left] for c, d in merged[right]):
                raise ValueError("Cross-split raw frame overlap")
    return dict(cross_split_overlap_frames=0, window_counts=counts,
                raw_frames_used={s: sum(b-a+1 for a, b in ranges) for s, ranges in merged.items()},
                raw_support_unions={s: [list(r) for r in ranges] for s, ranges in merged.items()})
