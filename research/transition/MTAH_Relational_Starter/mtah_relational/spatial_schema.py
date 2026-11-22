"""Versioned invariant summaries; ordered home/away geometry is kept separately."""
from __future__ import annotations

import numpy as np

SCHEMA_VERSION = "relational-phase1-v1"
FEATURE_NAMES = (
    "cross_density_3m", "cross_density_5m", "cross_density_10m",
    "cross_min", "cross_nearest_mean", "cross_cluster_fraction", "centroid_gap",
    "width_sum", "width_abs_diff", "length_sum", "length_abs_diff",
    "radius_sum", "radius_abs_diff", "team_mix",
    "cross_pairs_3m", "cross_pairs_5m", "cross_pairs_10m", "cross_cluster_count",
    "centroid_mid_x", "centroid_mid_y", "bbox_area_sum", "bbox_area_abs_diff",
    "within_nearest_sum", "within_nearest_abs_diff",
)
TARGET_NAMES = FEATURE_NAMES[:13]
TEAM_STRUCTURE_NAMES = (
    "cx", "cy", "width", "length", "width_p90_p10", "length_p90_p10",
    "radius", "bbox_area", "density_per_m2", "nearest_teammate_m",
)


def symmetric_summary(features: dict, pitch):
    """Return 24 values and observability masks, never a contest/possession label.

    All 24 need BOTH teams. Shape statistics need >=2 observations/team;
    nearest teammate statistics need >=2 too. This is mathematical validity,
    not the stricter minimum-eight/duplicate-free index quality policy.
    """
    f = features
    L, W = pitch
    diag = float(np.hypot(L, W))
    nh, na = f["n_home"], f["n_away"]
    n = nh + na
    values = [f[f"cross_density_{r}m"] for r in (3, 5, 10)]
    values += [f["cross_min_m"]/diag, f["cross_nearest_mean_m"]/diag,
               f["cross_cluster_fraction_3m"], f["centroid_sep_m"]/diag]
    for name, scale in (("width_p90_p10", W), ("length_p90_p10", L), ("radius", diag)):
        h, a = f[f"home_{name}"], f[f"away_{name}"]
        values.extend(((h+a)/(2*scale), abs(h-a)/scale))
    values.append(2*min(nh, na)/n if n else np.nan)
    values.extend(f[f"cross_pairs_{r}m"]/121 for r in (3, 5, 10))
    values += [f["max_cross_cluster_3m"]/22,
               (f["home_cx"]+f["away_cx"])/(2*L), (f["home_cy"]+f["away_cy"])/(2*W)]
    for name, scale in (("bbox_area", L*W), ("nearest_teammate_m", diag)):
        h, a = f[f"home_{name}"], f[f"away_{name}"]
        values.extend(((h+a)/(2*scale), abs(h-a)/scale))
    values = np.asarray(values, dtype=np.float64)
    valid = np.isfinite(values) & bool(nh and na)
    if min(nh, na) < 2:
        valid[7:13] = False
        valid[20:24] = False
    values[~valid] = np.nan
    return values, valid


def team_structures(features: dict):
    """[2,10], H first and A second; swapping teams swaps these rows."""
    return np.array([[features[f"{team}_{name}"] for name in TEAM_STRUCTURE_NAMES]
                     for team in ("home", "away")], dtype=np.float64)
