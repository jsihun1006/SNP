"""Read existing player-only geometry; raw parsing stays in mtah_relational."""
from dataclasses import asdict

import numpy as np
import pandas as pd

from mtah_relational.data_quality import ObservationPolicy, assess_observations, period_of
from mtah_relational.features import geometry_features

POLICY = ObservationPolicy()
GEOMETRY_KEYS = tuple(geometry_features(np.empty((0, 2)), np.empty((0, 2))))
# All possible geometry keys, including nonempty-only cross_mean/nearest fields.
GEOMETRY_KEYS = tuple(dict.fromkeys((*GEOMETRY_KEYS, *geometry_features([[0, 0]], [[1, 1]]))))
QUALITY_KEYS = (
    "raw_observed", "missing_slots", "partial_missing_slots", "invalid_slots",
    "outside_pitch_slots", "excluded_bounds_slots", "duplicate_suspect_pairs",
    "duplicate_suspect_players", "coverage_ok", "quality_valid", "n_home_valid", "n_away_valid",
)
INPUT_COLUMNS = ("frame", "period", *GEOMETRY_KEYS, *QUALITY_KEYS)


def observation_row(frame, pitch):
    """Pure player-only adapter. Does not connect slots across time."""
    xy, quality = assess_observations(frame, pitch, POLICY)
    f = geometry_features(xy[frame.team_codes == 0], xy[frame.team_codes == 1])
    return dict(frame=frame.frame, period=frame.period, **f, **quality)


def read_features(path, n_frames, blocks):
    """Allowlist excludes legacy future deltas and any ball/tag/label decoys."""
    table = pd.read_csv(path, usecols=list(INPUT_COLUMNS))
    validate_table(table, n_frames, blocks)
    return table


def validate_table(table, n_frames, blocks):
    missing = set(INPUT_COLUMNS) - set(table)
    if missing:
        raise ValueError(f"Missing player geometry columns: {sorted(missing)}")
    if len(table) != n_frames or not np.array_equal(table.frame, np.arange(n_frames)):
        raise ValueError("Expected every original frame in order, without duplicates or gaps")
    expected = np.array([period_of(i, blocks) for i in range(n_frames)])
    if not np.array_equal(table.period, expected):
        raise ValueError("Period must come from official blocks")
    for col in ("quality_valid", "cross_valid", "coverage_ok"):
        if table[col].dtype != bool:
            raise ValueError(f"Expected boolean {col}")
    counts = table[["n_home", "n_away"]].to_numpy()
    if not np.isfinite(counts).all() or (counts < 0).any() or not np.equal(counts, counts.astype(int)).all():
        raise ValueError("Observed counts must be nonnegative integers")
    if not np.array_equal(counts, table[["n_home_valid", "n_away_valid"]]):
        raise ValueError("Geometry/quality counts disagree")
    for r in (3, 5, 10):
        valid = table.cross_valid.to_numpy()
        pairs = table[f"cross_pairs_{r}m"].to_numpy()
        if not np.isfinite(pairs[valid]).all() or (pairs[valid] < 0).any():
            raise ValueError("Invalid global pair count")


def quality_fields(row):
    """Mathematical availability is independent of strict eligibility."""
    pair = bool(row["n_home"] > 0 and row["n_away"] > 0 and row["cross_valid"])
    keys = ("cross_min_m", "cross_nearest_mean_m", "cross_contact_fraction_3m",
            "cross_cluster_fraction_3m")
    metric = pair and bool(np.isfinite([row[k] for k in keys]).all())
    strict = bool(row["quality_valid"])
    reasons = []
    if min(row["n_home"], row["n_away"]) < POLICY.min_team_players:
        reasons.append("low_players")
    if max(row["n_home"], row["n_away"]) > POLICY.max_team_players:
        reasons.append("overcount")
    for key, reason in (("duplicate_suspect_pairs", "duplicate_suspect"),
                        ("outside_pitch_slots", "out_of_bounds"),
                        ("partial_missing_slots", "partial_missing"), ("invalid_slots", "invalid_coordinate")):
        if row[key]:
            reasons.append(reason)
    if not metric:
        reasons.append("undefined_pair")
    return dict(strict_quality_valid=strict, metric_valid=metric,
                quality_tier="strict_valid" if strict and metric else "low" if metric else "insufficient",
                reason_list=reasons)


def policy_dict():
    return asdict(POLICY)
