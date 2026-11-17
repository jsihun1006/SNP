"""Validate a Phase 1 export against its retained unified-tracking checkpoint.

This checks the export contract and source preservation. It does not evaluate
ball-position accuracy, generate observations, or read possession ground truth.
"""
import argparse
import hashlib
import json
from pathlib import Path
import re

import numpy as np
import pandas as pd


HERE = Path(__file__).resolve().parent
REPO_ROOT = HERE.parents[2]
MODEL = "ball_tracking"
TRACKING_COLUMNS = [
    "frame", "time_seconds", "track_id", "id_kind", "object_type", "team",
    "x_norm", "y_norm", "x_m", "y_m", "confidence", "observed",
    "is_prediction", "source", "broadcast_frame", "s_kph",
]
POSITION_FIELDS = ["x_norm", "y_norm", "x_m", "y_m", "s_kph"]
OPTIONAL_FLAGS = ["ball_world_live_reference", "broadcast_highlight_inferred"]


def require(condition, message):
    if not condition:
        raise ValueError(message)


def sha256(path):
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for chunk in iter(lambda: handle.read(8 * 1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def write_json(path, value):
    Path(path).write_text(json.dumps(value, ensure_ascii=False, indent=2,
        sort_keys=True, allow_nan=False) + "\n", encoding="utf-8")


def resolve_path(repo_root, value):
    path = Path(value)
    return path.resolve() if path.is_absolute() else (Path(repo_root) / path).resolve()


def validate_run_id(value):
    require(bool(re.fullmatch(r"[A-Za-z0-9][A-Za-z0-9_.-]{0,63}", value)),
            "run-id must be a single safe name, not a path")
    return value


def checked_config(value):
    require(value.get("mode") == "accepted_checkpoint_export",
            "only accepted_checkpoint_export mode is supported")
    for key in ["source_frames", "tactical_frames", "fps", "chunk_frames", "validation_chunk_rows"]:
        require(type(value.get(key)) is int and value[key] > 0,
                f"{key} must be a positive integer")
    pitch = np.asarray(value.get("pitch_m"), dtype=float)
    require(pitch.shape == (2,) and np.isfinite(pitch).all() and (pitch > 0).all(),
            "pitch_m must contain two positive finite dimensions")
    for key in ["source_sha256", "roster_sha256"]:
        require(bool(re.fullmatch(r"[0-9a-f]{64}", value.get(key, ""))),
                f"{key} must pin the exact checkpoint input")
    require(isinstance(value.get("input_csv"), str) and isinstance(value.get("roster"), str),
            "input_csv and roster paths are required")
    return value


def read_config(path):
    return checked_config(json.loads(Path(path).read_text(encoding="utf-8")))


def reject_lfs_pointer(path):
    with Path(path).open("rb") as handle:
        first = handle.read(200)
    require(not first.startswith(b"version https://git-lfs.github.com/spec/v1"),
            f"{path} is a Git LFS pointer; materialize the actual checkpoint file first")


def integer_array(values, name, minimum, maximum):
    array = pd.to_numeric(values, errors="raise").to_numpy(dtype=float)
    require(np.isfinite(array).all() and (array == np.floor(array)).all(),
            f"{name} must contain finite integer indices")
    require(((array >= minimum) & (array <= maximum)).all(), f"{name} is out of range")
    return array.astype(np.int64)


def boolean_array(values, name):
    normalized = values.astype(str).str.strip().str.lower()
    require(normalized.isin(["true", "false", "1", "0"]).all(),
            f"{name} must contain explicit booleans")
    return normalized.isin(["true", "1"]).to_numpy(dtype=bool)


def optional_flag(table, name):
    """Preserve existing flags, explicitly mark an unavailable flag unknown."""
    if name not in table:
        return np.full(len(table), "unknown", dtype=object)
    values = table[name].astype("string").str.strip().str.lower().fillna("unknown")
    require(values.isin(["true", "false", "1", "0", "unknown"]).all(),
            f"invalid optional source flag: {name}")
    return values.replace({"1": "true", "0": "false"}).to_numpy(dtype=object)


def read_checkpoint(input_csv, roster_path, config):
    for path, key in [(input_csv, "source_sha256"), (roster_path, "roster_sha256")]:
        require(Path(path).is_file(), f"missing checkpoint input: {path}")
        reject_lfs_pointer(path)
        require(sha256(path) == config[key], f"checkpoint SHA256 mismatch: {path}")
    roster = json.loads(Path(roster_path).read_text(encoding="utf-8"))
    players = roster.get("players", [])
    require(isinstance(players, list) and len(players) > 0, "roster needs player slots")
    names = []
    for player in players:
        prefix = player.get("column_prefix", "")
        require(bool(re.fullmatch(r"player_[0-9]{2,}", prefix)), "invalid roster column_prefix")
        require(player.get("team_code") in ["H", "A"], "invalid player team code")
        names.append(prefix)
    require(len(names) == len(set(names)), "duplicate roster track slot")
    required = ["frame", "tactical_frame", "ball_source", "ball_is_prediction"]
    required += [prefix + "_" + field for prefix in names + ["ball"] for field in POSITION_FIELDS]
    table = pd.read_csv(input_csv, low_memory=False, float_precision="round_trip")
    require(set(required).issubset(table.columns), "checkpoint is missing required wide columns")
    require(len(table) == config["source_frames"], "checkpoint source frame count mismatch")
    frames = integer_array(table["frame"], "source frame", 0, len(table) - 1)
    require(np.array_equal(frames, np.arange(len(table))), "source frames must be complete and ordered")
    raw_mapping = pd.to_numeric(table["tactical_frame"], errors="raise").to_numpy(dtype=float)
    require(not np.isinf(raw_mapping).any(), "tactical_frame contains infinity")
    absent = np.isnan(raw_mapping) | (raw_mapping == -1)
    require(((raw_mapping[~absent] >= 0) &
             (raw_mapping[~absent] < config["tactical_frames"])).all(), "tactical_frame is out of range")
    require((raw_mapping[~absent] == np.floor(raw_mapping[~absent])).all(),
            "tactical_frame must contain integer indices or -1/NaN")
    mapped = np.full(len(table), -1, dtype=np.int64)
    mapped[~absent] = raw_mapping[~absent].astype(np.int64)
    require((np.diff(mapped[~absent]) > 0).all(),
            "tactical mappings must be unique and increase with the source frames")
    predictions = boolean_array(table["ball_is_prediction"], "ball_is_prediction")
    for field in OPTIONAL_FLAGS:
        optional_flag(table, field)
    pitch = np.asarray(config["pitch_m"])
    for prefix in names + ["ball"]:
        fields = [prefix + "_" + field for field in POSITION_FIELDS]
        values = table[fields].to_numpy(dtype=float)
        valid = np.isfinite(values[:, :2]).all(axis=1)
        require(np.isfinite(values[valid, :4]).all(), f"{prefix}: finite XY needs finite metre XY")
        require(np.allclose(values[valid, 2:4], values[valid, :2] * pitch,
                            rtol=0, atol=1e-9), f"{prefix}: metre coordinate units mismatch")
    return table, players, mapped, predictions


def expected_status(table, players, mapped, predictions, config):
    count = config["tactical_frames"]
    frames = np.arange(count, dtype=np.int64)
    status = pd.DataFrame({"frame": frames, "time_seconds": frames / config["fps"],
        "mapped": np.zeros(count, bool), "broadcast_frame": np.full(count, np.nan),
        "player_position_count": np.zeros(count, np.int64),
        "ball_position_available": np.zeros(count, bool),
        "ball_is_prediction": np.zeros(count, bool),
        "object_position_count": np.zeros(count, np.int64)})
    source_ids = np.flatnonzero(mapped >= 0)
    target_ids = mapped[source_ids]
    counts = np.zeros(len(table), dtype=np.int64)
    for player in players:
        prefix = player["column_prefix"]
        counts += np.isfinite(table[[prefix + "_x_norm", prefix + "_y_norm"]].to_numpy(float)).all(axis=1)
    ball_valid = np.isfinite(table[["ball_x_norm", "ball_y_norm"]].to_numpy(float)).all(axis=1)
    status.loc[target_ids, "mapped"] = True
    status.loc[target_ids, "broadcast_frame"] = source_ids
    status.loc[target_ids, "player_position_count"] = counts[source_ids]
    status.loc[target_ids, "ball_position_available"] = ball_valid[source_ids]
    status.loc[target_ids, "ball_is_prediction"] = ball_valid[source_ids] & predictions[source_ids]
    status.loc[target_ids, "object_position_count"] = counts[source_ids] + ball_valid[source_ids]
    for field in OPTIONAL_FLAGS:
        status[field] = np.full(count, "unknown", dtype=object)
        status.loc[target_ids, field] = optional_flag(table, field)[source_ids]
    return status


def same_values(actual, expected, name):
    actual, expected = np.asarray(actual), np.asarray(expected)
    numeric = actual.dtype.kind in "biufc" and expected.dtype.kind in "biufc"
    require(np.array_equal(actual, expected, equal_nan=numeric),
            f"exported values differ from the source: {name}")


def validate_run(run_dir, repo_root=REPO_ROOT, write=True):
    run_dir = Path(run_dir).resolve()
    manifest = json.loads((run_dir / "run_manifest.json").read_text(encoding="utf-8"))
    config = checked_config(manifest["config"])
    source = resolve_path(repo_root, manifest["inputs"]["checkpoint"]["path"])
    roster = resolve_path(repo_root, manifest["inputs"]["roster"]["path"])
    require(manifest["inputs"]["checkpoint"]["sha256"] == config["source_sha256"], "manifest source hash disagrees")
    require(manifest["inputs"]["roster"]["sha256"] == config["roster_sha256"], "manifest roster hash disagrees")
    table, players, mapped, predictions = read_checkpoint(source, roster, config)
    for name in ["tracking.csv", "frame_status.csv"]:
        require(sha256(run_dir / name) == manifest["output_sha256"][name], f"output SHA256 mismatch: {name}")
    expected = expected_status(table, players, mapped, predictions, config)
    status = pd.read_csv(run_dir / "frame_status.csv", float_precision="round_trip",
                         dtype={field: str for field in OPTIONAL_FLAGS})
    require(list(status.columns) == list(expected.columns), "frame_status columns differ")
    require(len(status) == config["tactical_frames"], "incomplete Tactical frame status")
    for field in expected.columns:
        same_values(status[field], expected[field], "frame_status." + field)
    slots = {p["column_prefix"]: p for p in players}
    order = {name: index for index, name in enumerate(list(slots) + ["ball"])}
    player_count = np.zeros(len(status), np.int64)
    ball_count = np.zeros(len(status), np.int64)
    predicted_count = np.zeros(len(status), np.int64)
    last_key = -1
    rows = 0
    chunks = pd.read_csv(run_dir / "tracking.csv", chunksize=config["validation_chunk_rows"],
        float_precision="round_trip", keep_default_na=True,
        dtype={"track_id": str, "id_kind": str, "object_type": str, "team": str, "source": str})
    try:
        for chunk in chunks:
            require(list(chunk.columns) == TRACKING_COLUMNS, "tracking CSV columns differ")
            if len(chunk) == 0:
                continue
            frames = integer_array(chunk.frame, "tracking frame", 0, len(status) - 1)
            broadcasts = integer_array(chunk.broadcast_frame, "broadcast_frame", 0, len(table) - 1)
            require(chunk.track_id.isin(order).all(), "unknown track_id")
            keys = frames * len(order) + chunk.track_id.map(order).to_numpy(np.int64)
            require(keys[0] > last_key and (np.diff(keys) > 0).all(),
                    "tracking rows must be ordered and unique by frame/track_id")
            last_key = int(keys[-1])
            same_values(mapped[broadcasts], frames, "frame mapping")
            require(np.allclose(chunk.time_seconds, frames / config["fps"], rtol=0, atol=1e-12),
                    "tracking time_seconds mismatch")
            xy = chunk[["x_norm", "y_norm", "x_m", "y_m"]].to_numpy(float)
            require(np.isfinite(xy).all(), "tracking rows require finite positions")
            require(np.allclose(xy[:, 2:], xy[:, :2] * np.asarray(config["pitch_m"]), rtol=0, atol=1e-9),
                    "tracking metre units mismatch")
            require(chunk.confidence.isna().all(), "unknown source confidence must remain missing")
            observed = boolean_array(chunk.observed, "observed")
            predicted = boolean_array(chunk.is_prediction, "is_prediction")
            require(np.array_equal(observed, ~predicted), "observed/prediction flags conflict")
            for name, positions in chunk.groupby("track_id", sort=False).indices.items():
                ids = np.asarray(positions, dtype=int)
                part = chunk.iloc[ids]
                src = broadcasts[ids]
                fields = [name + "_" + field for field in POSITION_FIELDS]
                same_values(part[["x_norm", "y_norm", "x_m", "y_m", "s_kph"]].to_numpy(float),
                            table.iloc[src][fields].to_numpy(float), name)
                if name == "ball":
                    require(part.object_type.eq("ball").all() and part.id_kind.eq("ball").all(), "ball ID semantics mismatch")
                    require(part.team.isna().all(), "ball export must not invent a possession team")
                    same_values(predicted[ids], predictions[src], "ball prediction flag")
                    require(np.array_equal(part.source.fillna("").to_numpy(),
                        table.iloc[src].ball_source.fillna("").astype(str).to_numpy()), "ball provenance changed")
                    np.add.at(ball_count, frames[ids], 1)
                    np.add.at(predicted_count, frames[ids], predicted[ids].astype(np.int64))
                else:
                    require(part.object_type.eq("player").all() and part.id_kind.eq("track_slot").all(), "player ID semantics mismatch")
                    require(part.team.eq(slots[name]["team_code"]).all(), "player team changed")
                    require(part.source.eq("TACTICAL_ATD_RECORDED").all(), "player provenance changed")
                    require(not predicted[ids].any(), "player coordinates must not become predictions")
                    np.add.at(player_count, frames[ids], 1)
            rows += len(chunk)
    finally:
        chunks.close()
    same_values(player_count, status.player_position_count.to_numpy(np.int64), "player counts")
    same_values(ball_count, status.ball_position_available.to_numpy(np.int64), "ball counts")
    same_values(predicted_count, status.ball_is_prediction.to_numpy(np.int64), "prediction counts")
    require(rows == manifest["counts"]["tracking_rows"], "manifest row count mismatch")
    require(rows == int(status.object_position_count.sum()), "missing or additional exported objects")
    source_ball_valid = np.isfinite(table[["ball_x_norm", "ball_y_norm"]].to_numpy(float)).all(axis=1)
    expected_counts = {"source_frames": len(table), "tactical_frames": len(status),
        "mapped_broadcast_frames": int((mapped >= 0).sum()),
        "unmapped_broadcast_frames": int((mapped < 0).sum()),
        "unmapped_tactical_frames": int((~status.mapped).sum()),
        "unmapped_broadcast_ball_positions_not_assigned_to_tactical": int((source_ball_valid & (mapped < 0)).sum()),
        "tracking_rows": rows, "exported_player_positions": int(player_count.sum()),
        "exported_ball_positions": int(ball_count.sum()),
        "exported_ball_predictions": int(predicted_count.sum())}
    require(manifest["counts"] == expected_counts, "manifest counts differ from source and exported rows")
    require(sha256(source) == config["source_sha256"] and sha256(roster) == config["roster_sha256"],
            "checkpoint inputs changed during validation")
    result = {"passed": True, "mode": "accepted_checkpoint_export", "tactical_frames": len(status),
        "tracking_rows": rows, "mapped_tactical_frames": int(status.mapped.sum()),
        "exported_ball_positions": int(ball_count.sum()), "exported_ball_predictions": int(predicted_count.sum()),
        "source_position_and_speed_values_preserved": True, "source_mapping_preserved": True,
        "all_tactical_status_rows_present": True, "no_new_coordinates": True,
        "no_clipping_or_interpolation": True, "input_sha256_unchanged": True,
        "position_accuracy_evaluated": False, "possession_ground_truth_used": False}
    if write:
        write_json(run_dir / "validation.json", result)
    return result


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run-id", default="submission")
    parser.add_argument("--repo-root", type=Path, default=REPO_ROOT)
    args = parser.parse_args()
    try:
        run_id = validate_run_id(args.run_id)
        result = validate_run(args.repo_root / "outputs" / "phase1" / MODEL / run_id,
                              args.repo_root)
    except (OSError, ValueError, KeyError) as error:
        parser.exit(1, f"ball_tracking validation failed: {error}\n")
    print(json.dumps({"passed": result["passed"], "tracking_rows": result["tracking_rows"]}))


if __name__ == "__main__":
    main()
