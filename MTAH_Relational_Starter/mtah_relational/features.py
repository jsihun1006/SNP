"""Pure geometry/graph statistics from player locations (metres). No ball, no labels.

This module is intentionally independent from the official MTAH dataset and can be
unit-tested with synthetic coordinates.
"""
from __future__ import annotations

import numpy as np


def _team_stats(p: np.ndarray, prefix: str) -> dict[str, float]:
    n = len(p)
    if n == 0:
        return {
            f"{prefix}_cx": np.nan, f"{prefix}_cy": np.nan,
            f"{prefix}_width": np.nan, f"{prefix}_length": np.nan,
            f"{prefix}_radius": np.nan, f"{prefix}_bbox_area": np.nan,
            f"{prefix}_width_p90_p10": np.nan, f"{prefix}_length_p90_p10": np.nan,
            f"{prefix}_compactness_m": np.nan, f"{prefix}_density_per_m2": np.nan,
            f"{prefix}_nearest_teammate_m": np.nan,
        }
    center = np.mean(p, axis=0)
    extent = np.max(p, axis=0) - np.min(p, axis=0)
    robust_extent = np.percentile(p, 90, axis=0) - np.percentile(p, 10, axis=0)
    area = float(extent[0]*extent[1])
    radius = float(np.mean(np.linalg.norm(p-center, axis=1)))
    within = np.linalg.norm(p[:, None] - p[None, :], axis=-1)
    np.fill_diagonal(within, np.inf)
    return {
        f"{prefix}_cx": float(center[0]), f"{prefix}_cy": float(center[1]),
        f"{prefix}_length": float(extent[0]), f"{prefix}_width": float(extent[1]),
        f"{prefix}_radius": radius,
        f"{prefix}_bbox_area": area,
        f"{prefix}_width_p90_p10": float(robust_extent[1]),
        f"{prefix}_length_p90_p10": float(robust_extent[0]),
        f"{prefix}_compactness_m": radius,
        f"{prefix}_density_per_m2": n/area if area > 0 else np.nan,
        f"{prefix}_nearest_teammate_m": float(within.min(axis=1).mean()) if n > 1 else np.nan,
    }


def max_bipartite_cluster_size(mask: np.ndarray) -> int:
    """Size of largest connected component with >=1 cross-team edge.

    Only H-A edges are included: two players from the same team do not become
    adjacent simply because they are close. Thus components are a conservative
    *opposition-contact* proxy, not an actual duel label.
    """
    mask = np.asarray(mask, dtype=bool)
    if mask.ndim != 2 or not mask.any():
        return 0
    h, a = mask.shape
    seen = np.zeros(h + a, dtype=bool)
    best = 0
    for seed in np.flatnonzero(mask.any(axis=1)):
        if seen[seed]:
            continue
        stack = [int(seed)]
        seen[seed] = True
        count = 0
        while stack:
            node = stack.pop()
            count += 1
            if node < h:
                neighbors = h + np.flatnonzero(mask[node, :])
            else:
                neighbors = np.flatnonzero(mask[:, node-h])
            for neighbor in neighbors:
                neighbor = int(neighbor)
                if not seen[neighbor]:
                    seen[neighbor] = True
                    stack.append(neighbor)
        best = max(best, count)
    return int(best)


def geometry_features(home: np.ndarray, away: np.ndarray) -> dict[str, float | int]:
    """Geometric features for one frame, independent of player ordering."""
    h = np.asarray(home, dtype=float).reshape(-1, 2)
    a = np.asarray(away, dtype=float).reshape(-1, 2)
    h = h[np.isfinite(h).all(axis=1)]
    a = a[np.isfinite(a).all(axis=1)]
    n_h, n_a = len(h), len(a)
    out: dict[str, float | int] = {
        "n_home": n_h, "n_away": n_a, "n_players": n_h+n_a,
        "low_coverage": int(n_h < 8 or n_a < 8),
        "overcount": int(n_h+n_a > 22),
    }
    out.update(_team_stats(h, "home"))
    out.update(_team_stats(a, "away"))
    if n_h and n_a:
        d = np.linalg.norm(h[:, None, :] - a[None, :, :], axis=-1)
        out.update({
            "cross_min_m": float(d.min()),
            "cross_mean_m": float(d.mean()),
            "cross_pairs_3m": int(np.count_nonzero(d <= 3.0)),
            "cross_pairs_5m": int(np.count_nonzero(d <= 5.0)),
            "cross_pairs_10m": int(np.count_nonzero(d <= 10.0)),
            "cross_density_3m": float(np.count_nonzero(d <= 3.0) / (n_h*n_a)),
            "cross_density_5m": float(np.count_nonzero(d <= 5.0) / (n_h*n_a)),
            "cross_density_10m": float(np.count_nonzero(d <= 10.0) / (n_h*n_a)),
            "cross_nearest_mean_m": float(np.concatenate([d.min(axis=1), d.min(axis=0)]).mean()),
            "cross_nearest_median_m": float(np.median(np.concatenate([d.min(axis=1), d.min(axis=0)]))),
            "cross_contact_fraction_3m": float(((d <= 3).any(axis=1).sum() +
                                                (d <= 3).any(axis=0).sum()) / (n_h+n_a)),
            "cross_cluster_fraction_3m": max_bipartite_cluster_size(d <= 3.0)/(n_h+n_a),
            "cross_valid": True,
            "max_cross_cluster_3m": max_bipartite_cluster_size(d <= 3.0),
            "centroid_sep_m": float(np.linalg.norm(h.mean(axis=0)-a.mean(axis=0))),
        })
    else:
        out.update({
            "cross_min_m": np.nan, "cross_mean_m": np.nan,
            "cross_pairs_3m": 0, "cross_pairs_5m": 0, "cross_pairs_10m": 0,
            "cross_density_3m": np.nan, "max_cross_cluster_3m": 0,
            "centroid_sep_m": np.nan,
            "cross_density_5m": np.nan, "cross_density_10m": np.nan,
            "cross_nearest_mean_m": np.nan, "cross_nearest_median_m": np.nan,
            "cross_contact_fraction_3m": np.nan, "cross_cluster_fraction_3m": np.nan,
            "cross_valid": False,
        })
    return out


def add_temporal_features(df, fps: int = 25):
    """Add symmetric 1s deltas; never cross the published half boundary.

    These are *descriptive features*, not labels. The +-1s context is for an
    offline, bidirectional model. Discard undefined deltas at window edges.
    """
    df = df.copy()
    ids = df["frame"].to_numpy()
    if len(ids) and (not np.isfinite(ids).all() or not np.equal(ids, ids.astype(np.int64)).all()
                     or (np.diff(ids) <= 0).any()):
        raise ValueError("frame must be unique, integral and strictly increasing")
    if not isinstance(fps, int) or fps <= 0:
        raise ValueError("fps must be a positive integer")
    period = df["period"]
    # Row shifts are meaningful only when every intermediate record is contiguous.
    runs = ((df["frame"].diff() != 1) | period.ne(period.shift())).cumsum()
    for name in ["cross_pairs_3m", "max_cross_cluster_3m", "centroid_sep_m",
                 "home_radius", "away_radius", "home_cx", "away_cx"]:
        before = df[name].shift(fps)
        after = df[name].shift(-fps)
        valid_before = ((period > 0) & (runs == runs.shift(fps)) &
                        (df["frame"] - df["frame"].shift(fps) == fps))
        valid_after = ((period > 0) & (runs == runs.shift(-fps)) &
                       (df["frame"].shift(-fps) - df["frame"] == fps))
        df[f"{name}_delta_prev_1s"] = (df[name] - before).where(valid_before)
        df[f"{name}_delta_next_1s"] = (after - df[name]).where(valid_after)
    return df
