from dataclasses import dataclass
from typing import Protocol, Sequence

import numpy as np


@dataclass(frozen=True)
class Detection:
    bbox: tuple[float, float, float, float]
    confidence: float
    class_id: int = 0

    def __post_init__(self):
        x1, y1, x2, y2 = self.bbox
        if not np.isfinite([*self.bbox, self.confidence]).all():
            raise ValueError("Detection contains NaN/Inf")
        if x2 <= x1 or y2 <= y1 or not 0 <= self.confidence <= 1:
            raise ValueError("Invalid bbox or confidence")


class Detector(Protocol):
    def predict(self, image: np.ndarray, confidence: float) -> Sequence[Detection]: ...


class PoseEstimator(Protocol):
    def predict(self, image: np.ndarray, boxes: Sequence[Detection]) -> np.ndarray:
        """Return original-image pixel coordinates, shape (N, 2, 2)."""
        ...
