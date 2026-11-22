"""Stateless, observation-independent span selection (no model or loss)."""
from dataclasses import dataclass

import numpy as np

from .splits import SPLITS


@dataclass(frozen=True)
class SpanMasker:
    seed: int = 2027
    minimum: int = 8
    maximum: int = 13

    def __post_init__(self):
        if not isinstance(self.seed, int) or self.seed < 0 or not 1 <= self.minimum <= self.maximum:
            raise ValueError("Invalid masking seed or span length")

    def sample(self, length, *, center_frame, period, split, epoch=0):
        if split not in SPLITS or length < self.maximum or epoch < 0 or center_frame < 0 or period not in (1, 2):
            raise ValueError("Invalid reconstruction mask routing")
        # No Python hash(): stable across processes and worker order. No x/y/Q.
        sequence = np.random.SeedSequence([self.seed, int(epoch) if split == "train" else 0,
                                          int(center_frame), int(period), SPLITS.index(split), length])
        rng = np.random.default_rng(sequence)
        span = int(rng.integers(self.minimum, self.maximum+1))
        first = int(rng.integers(0, length-span+1))
        mask = np.zeros(length, dtype=bool)
        mask[first:first+span] = True
        return mask
