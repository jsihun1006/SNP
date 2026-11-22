"""Frame-local, corruption-safe model input contract; no neural model or training."""
from dataclasses import dataclass

import numpy as np

from .data_quality import ObservationPolicy, assess_observations
from .features import geometry_features
from .spatial_schema import FEATURE_NAMES, symmetric_summary, team_structures


@dataclass(frozen=True)
class SpatialInput:
    frame: int  # routing only, do not encode absolute match time
    period: int
    xy_m: np.ndarray  # variable-N visible nodes only
    team: np.ndarray
    values: np.ndarray  # [24], unavailable values sanitized to zero
    feature_valid: np.ndarray  # [24]
    quality: np.ndarray  # [4], symmetric, observed context only
    team_structure: np.ndarray  # [2,10], ordered H/A; unavailable values zero
    team_structure_valid: np.ndarray  # [2,10]
    hidden: bool


def spatial_input(frame, pitch, policy=ObservationPolicy(), hidden=False):
    """Hide raw data BEFORE any graph/statistic/quality calculation.

    Targets must be built separately from clean frames. Nothing here contains
    clean hidden counts, topology, quality or target-validity bits.
    """
    if hidden:
        return SpatialInput(frame.frame, frame.period, np.empty((0, 2)), np.empty(0, dtype=np.int8),
                            np.zeros(len(FEATURE_NAMES)), np.zeros(len(FEATURE_NAMES), dtype=bool),
                            np.zeros(4), np.zeros((2, 10)), np.zeros((2, 10), dtype=bool), True)
    xy, q = assess_observations(frame, pitch, policy)
    mask = np.isfinite(xy).all(axis=1)
    team = frame.team_codes[mask].copy()
    xy = xy[mask]
    f = geometry_features(xy[team == 0], xy[team == 1])
    values, valid = symmetric_summary(f, pitch)
    ordered = team_structures(f)
    ordered_valid = np.isfinite(ordered)
    for t in (0, 1):
        if (team == t).sum() < 2:
            ordered_valid[t, 2:] = False
    quality = np.array([min(f["n_home"], f["n_away"])/11,
                        max(f["n_home"], f["n_away"])/11,
                        q["duplicate_suspect_players"]/max(1, len(xy)),
                        q["outside_pitch_slots"]/max(1, len(frame.xy_m))])
    return SpatialInput(frame.frame, frame.period, xy.copy(), team, np.where(valid, values, 0), valid,
                        quality, np.where(ordered_valid, ordered, 0), ordered_valid, False)


def spatial_inputs(frames, pitch, hidden_frames=(), policy=ObservationPolicy()):
    """Frame-local recomputation: no precomputed temporal deltas/caches are accepted."""
    hidden = frozenset(hidden_frames)
    return [spatial_input(f, pitch, policy, f.frame in hidden) for f in frames]
