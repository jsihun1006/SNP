"""Fixed causal baseline plus a Train-fitted threshold and causal run gate."""
from collections import deque
import hashlib
import json

import numpy as np
import pandas as pd

from mtah_relational.candidate_baseline import CausalContestedIndex
from .quality import quality_fields

STATES = ("candidate", "not_candidate", "insufficient_observation")
DELTA_NAMES = ("cross_pairs_3m", "max_cross_cluster_3m", "centroid_sep_m",
               "home_cx", "home_cy", "away_cx", "away_cy", "home_width", "away_width",
               "home_length", "away_length", "home_compactness_m", "away_compactness_m")


def compute_evidence(table, manifest):
    """Exact whole-half baseline retained; routing evidence resets at split edges.

    No future values; missing record, invalid current observation, period 0,
    half change or split change prevents temporal derivatives across it.
    """
    original, routed = CausalContestedIndex(25), CausalContestedIndex(25)
    history, records = deque(maxlen=26), []
    previous = None
    # Only named player geometry fields are ever used, not arbitrary extra columns.
    for values in table.itertuples(index=False, name=None):
        row = dict(zip(table.columns, values))
        frame, period = int(row["frame"]), int(row["period"])
        role = manifest.split_of(frame) or "outside"
        route_key = (period, role)
        boundary = previous is None or frame != previous[0]+1 or route_key != previous[1] or period == 0
        baseline = original.step(frame, period, row, row["quality_valid"])
        if boundary:
            routed = CausalContestedIndex(25)
            history.clear()
        index = routed.step(frame, period, row, row["quality_valid"])
        q = quality_fields(row)
        out = dict(row, split=role, in_official_block=period in (1, 2),
                   **{k: v for k, v in q.items() if k != "reason_list"},
                   baseline_score=baseline["contested_index"],
                   baseline_temporal_available=baseline["temporal_available"],
                   candidate_score=index["contested_index"],
                   contact_player_fraction=row["cross_contact_fraction_3m"],
                   cross_cluster_fraction=row["cross_cluster_fraction_3m"],
                   **{k: v for k, v in index.items() if k not in ("contested_index", "index_valid")},
                   record_boundary=boundary, count_change_prev_frame=False,
                   count_change_available=False, count_stable_1s=False)
        # An undefined opposing set is not an observed zero-pair/zero-cluster frame.
        for radius in (3, 5, 10):
            if not q["metric_valid"]:
                out[f"cross_pairs_{radius}m"] = np.nan
            out[f"valid_cross_pairs_{radius}m"] = q["metric_valid"]
        if not q["metric_valid"]:
            out["max_cross_cluster_3m"] = np.nan
        for name in ("cross_min_m", "cross_nearest_mean_m", "cross_density_3m", "cross_density_5m",
                     "cross_density_10m", "contact_player_fraction", "cross_cluster_fraction", "max_cross_cluster_3m"):
            out[f"valid_{name}"] = q["metric_valid"] and bool(np.isfinite(out[name]))
        for team in ("home", "away"):
            for name in ("cx", "cy", "width", "length", "width_p90_p10", "length_p90_p10",
                         "radius", "compactness_m", "bbox_area", "density_per_m2", "nearest_teammate_m"):
                key = f"{team}_{name}"
                minimum = 1 if name in ("cx", "cy") else 2
                out[f"valid_{key}"] = row[f"n_{team}"] >= minimum and bool(np.isfinite(out[key]))
                if not out[f"valid_{key}"]:
                    out[key] = np.nan
        for name in DELTA_NAMES:
            out[f"{name}_delta_prev_1s"] = np.nan
        if history and not boundary:
            out["count_change_available"] = True
            out["count_change_prev_frame"] = any(row[k] != history[-1][k] for k in ("n_home", "n_away"))
        history.append(row)
        if index["temporal_available"]:
            for name in DELTA_NAMES:
                out[f"{name}_delta_prev_1s"] = row[name]-history[0][name]
            out["count_stable_1s"] = all((p["n_home"], p["n_away"]) == (row["n_home"], row["n_away"])
                                         for p in history)
        reasons = q["reason_list"]
        if period == 0:
            reasons.append("outside_official_block")
        elif boundary:
            reasons.append("history_boundary")
        if not index["temporal_available"] and index["index_valid"]:
            reasons.append("spatial_fallback")
            reasons.append("temporal_gap" if len(history) == 26 else "history_warmup")
        if not index["index_valid"]:
            reasons.append("strict_observation_required")
        out["reason_codes"] = "|".join(reasons)
        records.append(out)
        previous = (frame, route_key)
    return pd.DataFrame(records)


def fit_threshold(evidence, quantile=.9):
    if not 0 < quantile < 1:
        raise ValueError("Quantile must be between zero and one")
    selected = evidence.loc[(evidence.split == "train") & evidence.strict_quality_valid &
                            evidence.in_official_block & np.isfinite(evidence.candidate_score),
                            ["frame", "candidate_score"]]
    if not len(selected) or selected.frame.duplicated().any():
        raise ValueError("Threshold fit requires unique eligible Train raw frames")
    return dict(threshold=float(np.quantile(selected.candidate_score, quantile, method="linear")),
                quantile=quantile, fit_frames=len(selected), fit_split="train",
                fit_frame_sha256=hashlib.sha256(selected.frame.to_numpy(dtype="int64").tobytes()).hexdigest(),
                comparison=">=", interpolation="linear")


def classify(evidence, threshold, min_run=5):
    """Online causal gate: first min_run-1 frames remain pending/noncandidate.

    No backdating after the fifth observation. No bridges through unknown frames.
    Candidate emitted segment duration can therefore be shorter than min_run/25.
    """
    if not np.isfinite(threshold) or not isinstance(min_run, int) or min_run < 1:
        raise ValueError("Invalid fixed rule")
    scores = evidence.candidate_score.to_numpy()
    valid = (evidence.strict_quality_valid & evidence.metric_valid & evidence.in_official_block).to_numpy() & np.isfinite(scores)
    ids, periods = evidence.frame.to_numpy(), evidence.period.to_numpy()
    roles = evidence.split.to_numpy()
    states = np.full(len(evidence), STATES[2], dtype=object)
    pending = np.zeros(len(evidence), bool)
    runs = np.zeros(len(evidence), dtype=int)
    run = 0
    for i in range(len(evidence)):
        if i == 0 or ids[i] != ids[i-1]+1 or periods[i] != periods[i-1] or roles[i] != roles[i-1]:
            run = 0
        if not valid[i]:
            run = 0
            continue
        run = run+1 if scores[i] >= threshold else 0
        states[i] = STATES[0] if run >= min_run else STATES[1]
        pending[i] = 0 < run < min_run
        runs[i] = run
    return states, pending, runs


def apply_policy(evidence, fitted, min_run=5):
    result = evidence.copy()
    state, pending, run = classify(evidence, fitted["threshold"], min_run)
    result["candidate_state"], result["candidate_pending"], result["above_threshold_run_frames"] = state, pending, run
    result["reason_codes"] = ["|".join(filter(None, [s, "minimum_run_pending" if p else "", "threshold_pass" if c == STATES[0] else ""]))
                              for s, p, c in zip(result.reason_codes, pending, state)]
    return result


def fingerprint(config):
    return hashlib.sha256(json.dumps(config, sort_keys=True, allow_nan=False).encode()).hexdigest()
