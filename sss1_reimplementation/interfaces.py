"""Extension contracts only: none of these are implemented or claimed as SSS-1."""
from typing import Protocol, Sequence
import numpy as np


class Tracker(Protocol):
    def update(self, frame: int, detections: Sequence[dict]) -> Sequence[dict]: ...


class TeamClassifier(Protocol):
    def classify(self, image: np.ndarray, bbox: tuple) -> str: ...


class DynamicCalibration(Protocol):
    def estimate(self, image: np.ndarray, frame: int): ...


class SceneCutDetector(Protocol):
    def is_cut(self, previous: np.ndarray, current: np.ndarray) -> bool: ...


class ATDAlignment(Protocol):
    def align(self, world_positions: np.ndarray, timestamp: float) -> np.ndarray: ...


class GSRIntegration(Protocol):
    def consume(self, frames: Sequence[dict]) -> dict: ...


class PossessionJudgement(Protocol):
    def predict(self, aligned_tracks: Sequence[dict]) -> str: ...
