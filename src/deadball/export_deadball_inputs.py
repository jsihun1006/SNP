"""Export identical DFL and ATD feature schemas for two dead-ball models.

LightGBM gets feature rows. TranSPORTmer gets one trajectory per match, with
time windows sliced on demand to avoid duplicating overlapping observations.
"""

from pathlib import Path
import sys

_SRC = Path(__file__).resolve().parents[1]
sys.path[:0] = [str(_SRC), str(_SRC.parent)]

import argparse
import json

import numpy as np
import pandas as pd
from scipy.linalg import eigh
from threadpoolctl import threadpool_limits

import competition_data
import dfl_labels
import dfl_load
import features
import simulate
import train_dfl
from compare_derived_vif import feature_matrix
from config import CONFIG
from games import GAME, GAMES, half_video_blocks, require, source_frames
from starter.baseline import team_of_each_track


def load_atd(root):
    metadata, raw = require(["atd_meta", "atd_raw"], data_dir=root)
    teams = team_of_each_track(metadata)
    n_frames = source_frames("tactical")
    pitch = GAMES[GAME]["pitch"]
    players, ball = competition_data.read_tracking(raw, len(teams), n_frames, *pitch)
    period = np.zeros(n_frames, dtype=np.int8)
    for half, first, last in half_video_blocks(GAME):
        period[first:last + 1] = half
    return features.Match(players, teams, ball, period, pitch, np.arange(n_frames))


def export_match(output, name, match, stride, dead=None):
    matrix = feature_matrix(match).iloc[::stride].reset_index(drop=True)
    columns = matrix.columns.tolist()
    matrix.insert(0, "period", match.period[::stride])
    matrix.insert(0, "frame", match.frame[::stride])
    if dead is not None:
        matrix["dead"] = dead[::stride]
    path = output / "lightgbm" / f"{name}.parquet"
    matrix.to_parquet(path, index=False)

    observed = np.isfinite(match.players).all(axis=-1)
    normalized = match.players / np.asarray(match.pitch, dtype=np.float32)
    arrays = {
        "positions": np.where(observed[..., None], normalized, 0).astype(np.float32),
        "observed": observed,
        "team": np.asarray(match.team, dtype=np.int8),
        "frame": np.asarray(match.frame, dtype=np.int32),
        "period": np.asarray(match.period, dtype=np.int8),
        "fps": np.int16(CONFIG["fps"]),
        "pitch_m": np.asarray(match.pitch, dtype=np.float32),
    }
    if dead is not None:
        arrays["dead"] = dead
    np.savez_compressed(output / "transportmer" / f"{name}.npz", **arrays)
    print(f"{name}: {len(matrix):,} LightGBM rows, sequence shape {normalized.shape}", flush=True)
    return columns, matrix[columns] if dead is not None else None


def vif_diagnostics(frame, limit=10):
    """Use column medians for missing values in VIF only."""
    filled = frame.fillna(frame.median()).fillna(0)
    values = filled.to_numpy(dtype=np.float64, copy=True)
    values -= values.mean(axis=0)
    norms = np.linalg.norm(values, axis=0)
    varying = norms > 1e-10
    values = values[:, varying] / norms[varying]
    correlation = values.T @ values
    eigenvalues, vectors = eigh(correlation)
    threshold = max(eigenvalues[-1] * 1e-12, 1e-14)
    nonzero = eigenvalues > threshold
    coefficients = vectors[:, nonzero] ** 2
    vif = coefficients @ (1 / eigenvalues[nonzero])
    redundancy = (vectors[:, ~nonzero] ** 2).sum(axis=1) > 1e-10
    rows = []
    finite_index = 0
    for column, valid in zip(frame.columns, varying):
        if not valid:
            rows.append({"feature": column, "vif": None, "status": "constant_or_all_missing"})
            continue
        if redundancy[finite_index]:
            rows.append({"feature": column, "vif": None, "status": "exact_linear_dependence"})
        else:
            score = float(vif[finite_index])
            status = "above_threshold" if score > limit else "ok"
            rows.append({"feature": column, "vif": score, "status": status})
        finite_index += 1
    return sorted(rows, key=lambda row: (row["status"] == "exact_linear_dependence", row["vif"] or 0), reverse=True)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dfl-cache", type=Path, required=True)
    parser.add_argument("--atd-root", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--vif-report", type=Path, required=True)
    args = parser.parse_args()
    CONFIG["dfl_cache"] = str(args.dfl_cache)
    for name in ("lightgbm", "transportmer"):
        (args.output / name).mkdir(parents=True, exist_ok=True)

    training = []
    for index, match_id in enumerate(dfl_load.MATCHES):
        tracking, _, meta = dfl_load.load_match(match_id)
        match = features.match_from_dfl(tracking, meta)
        seed = CONFIG["sim_seed"] + index
        settings = train_dfl.random_settings(np.random.default_rng(seed))
        match = simulate.simulate(match, seed=seed, **settings)
        columns, train = export_match(
            args.output, match_id, match, CONFIG["train_stride"], dfl_labels.dead_label(tracking)
        )
        training.append(train)

    atd_columns, _ = export_match(args.output, GAME, load_atd(args.atd_root), stride=1)
    if columns != atd_columns or any(list(frame.columns) != columns for frame in training):
        raise ValueError("DFL and ATD LightGBM features do not match.")
    (args.output / "lightgbm" / "feature_columns.json").write_text(
        json.dumps(columns, indent=2) + "\n", encoding="utf-8"
    )
    with threadpool_limits(limits=4, user_api="blas"):
        data = pd.concat(training, ignore_index=True)
        rows = vif_diagnostics(data)
    summary = {
        "rows": len(data),
        "columns": len(columns),
        "removed": 0,
        "threshold": 10,
        "above_10": sum(row["status"] == "above_threshold" for row in rows),
        "exact_linear_dependence": sum(row["status"] == "exact_linear_dependence" for row in rows),
        "constant_or_all_missing": sum(row["status"] == "constant_or_all_missing" for row in rows),
        "max_finite_vif": max(row["vif"] for row in rows if row["vif"] is not None),
        "features": rows,
    }
    args.vif_report.parent.mkdir(parents=True, exist_ok=True)
    args.vif_report.write_text(json.dumps(summary, indent=2) + "\n", encoding="utf-8")
    print(json.dumps({key: value for key, value in summary.items() if key != "features"}))


if __name__ == "__main__":
    main()
