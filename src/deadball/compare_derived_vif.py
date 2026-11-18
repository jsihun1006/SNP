"""Build DFL tracking features and remove columns with VIF greater than 50."""

from pathlib import Path
import sys

_SRC = Path(__file__).resolve().parents[1]
sys.path[:0] = [str(_SRC), str(_SRC.parent)]

import argparse
import json

import numpy as np
import pandas as pd
from scipy.linalg import solve
from threadpoolctl import threadpool_limits

import derived_deadball_features
import dfl_load
import features
import simulate
import train_dfl
from config import CONFIG


LOW_SPEED_CONTEXT_EXCLUDED = ('both_teams_slow_window@-0.4s', 'both_teams_slow_window@+0.4s')
VIF50_EXCLUDED = (
    "home_center_distance_median",
    "away_speed_mean@+0.4s",
    "nn_mean",
    "home_speed_mean@-0.4s",
    "away_center_distance_median",
    "home_speed_mean@+0.4s",
    "away_speed_mean@-1s",
    "away_opponent_median",
    "away_speed_mean@+1s",
    "centroid_distance",
    "home_width",
    "both_teams_slow_duration",
    "home_speed_mean@-1s",
    "nn_mean@-0.4s",
    "home_center_distance_median@+0.4s",
    "home_speed_mean@+2s",
    "nn_mean@+1s",
    "away_width",
    "home_length@-1s",
    "home_length@+1s",
    "away_speed_mean@-0.4s",
    "home_opponent_median",
    "away_speed_max",
    "home_speed_mean@-2s",
    "away_center_distance_median@-0.4s",
    "home_speed_max",
    "home_center_distance_median@-1s",
    "away_length@-1s",
    "away_length@+1s",
    "away_speed_mean@+4s",
    "away_speed_mean@-4s",
    "all_speed_p50",
    "away_center_distance_median@+1s",
    "home_speed_mean@+1s",
    "home_speed_mean@-10s",
    "away_speed_mean@+10s",
    "home_width@+1s",
    "all_speed_p90",
    "both_teams_slow_duration@-0.4s",
    "both_teams_slow_window",
    "centroid_distance@+0.4s",
    "home_goalline_fraction",
    "both_teams_slow_duration@+1s",
    "away_opponent_median@+0.4s",
)


def feature_matrix(match, apply_vif_filter=True):
    base = features.base_features(match)
    excluded = {"touchline_min", "goalline_min", "all_speed_mean", "all_speed_p25", "all_speed_p75"}
    original = [column for column in train_dfl.model_columns(base.columns) if column not in excluded]
    base = base[["period", *original]].astype(np.float32)
    extended = pd.concat([base, derived_deadball_features.player_features(match)], axis=1)
    shorter_context = [
        f"{side}_{metric}"
        for side in ("home", "away")
        for metric in ("accelerating_fraction", "decelerating_fraction", "opponent_rate")
    ]
    shape_columns = ("home_length", "home_width", "away_length", "away_width")
    columns = extended.columns.drop("period")
    context = features.add_context(extended, columns)
    replacement = {}
    for seconds in (-5, 5):
        shift = -seconds * CONFIG["fps"]
        values = extended.groupby("period", sort=False)[shorter_context].shift(shift)
        for name in shorter_context:
            old_column = f"{name}@{seconds * 2:+g}s"
            replacement[old_column] = f"{name}@{seconds:+g}s"
            context[old_column] = values[name]
    excluded_shape_columns = [
        f"{name}@{seconds:+g}s"
        for name in shape_columns
        for seconds in CONFIG["context_offsets_s"]
        if abs(seconds) not in (1, 2)
    ]
    matrix = context.rename(columns=replacement).drop(
        columns=["period", *excluded_shape_columns, *LOW_SPEED_CONTEXT_EXCLUDED]
    )
    if apply_vif_filter:
        matrix = matrix.drop(columns=list(VIF50_EXCLUDED))
    return matrix


def vif_select(frame, limit=10.0):
    # Median-fill for the VIF calculation only. LightGBM sees the original missing values.
    medians = frame.median().fillna(0)
    values = frame.fillna(medians).to_numpy(dtype=np.float64)
    values -= values.mean(axis=0)
    norms = np.linalg.norm(values, axis=0)
    varying = norms > 1e-10
    indexes = np.flatnonzero(varying)
    dropped = [{"column": frame.columns[i], "vif": None, "reason": "constant"} for i in np.flatnonzero(~varying)]
    values = values[:, varying] / norms[varying]
    correlation = values.T @ values
    # Small ridge keeps exactly redundant columns invertible. Recompute the
    # final VIF on the retained columns after removing the redundant ones.
    precision = solve(correlation + np.eye(len(indexes)) * 1e-10, np.eye(len(indexes)), assume_a="pos")
    initial = {frame.columns[i]: float(vif) for i, vif in zip(indexes, precision.diagonal())}
    while len(indexes):
        vifs = precision.diagonal()
        worst = int(vifs.argmax())
        if vifs[worst] <= limit:
            break
        dropped.append({"column": frame.columns[indexes[worst]], "vif": float(vifs[worst]), "reason": "VIF"})
        retain = np.arange(len(indexes)) != worst
        term = precision[retain, worst]
        precision = precision[np.ix_(retain, retain)] - np.outer(term, term) / vifs[worst]
        indexes = indexes[retain]
    retained = list(frame.columns[indexes])
    relative = np.searchsorted(np.flatnonzero(varying), indexes)
    final_vif = np.linalg.inv(correlation[np.ix_(relative, relative)]).diagonal()
    assert final_vif.max() <= limit + 1e-6
    return retained, {"initial_vif": initial, "removed": dropped,
                      "final_vif": dict(zip(retained, final_vif.tolist()))}


def feature_cache(cache, match_id):
    return cache / f"{match_id}_player_context_5s_no_line_min_no_all_speed_mean_no_pair_median_shape_1s_2s_speed_p10_p50_p90_no_slow_window_04s"


def prepare(cache, match_id):
    target = feature_cache(cache, match_id)
    target.mkdir(parents=True, exist_ok=True)
    if (target / "ready").exists():
        return
    track, _, meta = dfl_load.load_match(match_id)
    match = features.match_from_dfl(track, meta)
    index = dfl_load.MATCHES.index(match_id)
    seed = CONFIG["sim_seed"]
    settings = train_dfl.random_settings(np.random.default_rng(seed + index))
    train = simulate.simulate(match, seed=seed + index, **settings)

    train_data = feature_matrix(train, apply_vif_filter=False).iloc[::CONFIG["train_stride"]].copy()

    train_data.to_parquet(target / "train.parquet", index=False)


    (target / "ready").touch()
    print("Prepared", match_id, len(track), train_data.shape[1], flush=True)



def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--cache", type=Path, required=True)
    parser.add_argument("--report", type=Path, required=True)
    args = parser.parse_args()
    args.cache.mkdir(parents=True, exist_ok=True)
    CONFIG["dfl_cache"] = str(args.cache)
    with threadpool_limits(limits=4, user_api="blas"):
        for match_id in dfl_load.MATCHES:
            prepare(args.cache, match_id)
        frames = [pd.read_parquet(feature_cache(args.cache, match_id) / "train.parquet")
                  for match_id in dfl_load.MATCHES]
        data = pd.concat(frames, ignore_index=True).drop(columns="dead", errors="ignore")
        retained, report = vif_select(data, limit=50)
    report.update(rows=len(data), columns_before=len(data.columns), columns_after=len(retained),
                  retained=retained, limit=50)
    args.report.parent.mkdir(parents=True, exist_ok=True)
    args.report.write_text(json.dumps(report, indent=2, allow_nan=False) + "\n", encoding="utf-8")
    print("Rows:", len(data), "Columns:", len(data.columns), "->", len(retained),
          "Max final VIF:", max(report["final_vif"].values()), flush=True)


if __name__ == "__main__":
    main()
