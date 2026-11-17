"""Export the retained unified checkpoint to the Tactical Phase 1 contract.

This adapter reindexes existing records only. It does not rerun the historical
detector/correction chain, fill gaps, change positions, or infer possession.
"""
import argparse
import json
from pathlib import Path
import shutil
import tempfile

import numpy as np
import pandas as pd

from validate import (
    HERE, REPO_ROOT, MODEL, TRACKING_COLUMNS, POSITION_FIELDS, expected_status,
    read_checkpoint, read_config, require, resolve_path, sha256, validate_run,
    validate_run_id, write_json,
)


def portable_path(path, repo_root):
    try:
        return str(Path(path).relative_to(repo_root)).replace("\\", "/")
    except ValueError:
        return str(Path(path))


def export_chunk(table, players, mapped, predictions, fps):
    pieces = []
    for order, (prefix, team) in enumerate([(p["column_prefix"], p["team_code"]) for p in players] + [("ball", "")]):
        values = table[[prefix + "_" + field for field in POSITION_FIELDS]].to_numpy(float)
        available = (mapped >= 0) & np.isfinite(values[:, :2]).all(axis=1)
        if not available.any():
            continue
        is_ball = prefix == "ball"
        frames = mapped[available]
        pred = predictions[available] if is_ball else np.zeros(int(available.sum()), bool)
        source = table.ball_source.fillna("").astype(str).to_numpy()[available] if is_ball else "TACTICAL_ATD_RECORDED"
        piece = pd.DataFrame({"frame": frames, "time_seconds": frames / fps,
            "track_id": prefix, "id_kind": "ball" if is_ball else "track_slot",
            "object_type": "ball" if is_ball else "player", "team": team,
            "x_norm": values[available, 0], "y_norm": values[available, 1],
            "x_m": values[available, 2], "y_m": values[available, 3],
            "confidence": np.nan, "observed": ~pred, "is_prediction": pred,
            "source": source, "broadcast_frame": table.frame.to_numpy(np.int64)[available],
            "s_kph": values[available, 4], "_order": order})
        pieces.append(piece)
    if not pieces:
        return pd.DataFrame(columns=TRACKING_COLUMNS)
    return pd.concat(pieces, ignore_index=True).sort_values(["frame", "_order"], kind="stable")[TRACKING_COLUMNS]


def extract(repo_root, config_path=HERE / "config.json", run_id="submission", input_csv=None, roster=None):
    repo_root = Path(repo_root).resolve()
    run_id = validate_run_id(run_id)
    config_path = Path(config_path).resolve()
    config = read_config(config_path)
    source = resolve_path(repo_root, input_csv if input_csv is not None else config["input_csv"])
    roster_path = resolve_path(repo_root, roster if roster is not None else config["roster"])
    out = repo_root / "outputs" / "phase1" / MODEL / run_id
    require(not out.exists(), f"refusing to overwrite existing run: {out}")
    table, players, mapped, predictions = read_checkpoint(source, roster_path, config)
    status = expected_status(table, players, mapped, predictions, config)
    ball_valid = np.isfinite(table[["ball_x_norm", "ball_y_norm"]].to_numpy(float)).all(axis=1)
    counts = {"source_frames": len(table), "tactical_frames": len(status),
        "mapped_broadcast_frames": int((mapped >= 0).sum()),
        "unmapped_broadcast_frames": int((mapped < 0).sum()),
        "unmapped_tactical_frames": int((~status.mapped).sum()),
        "unmapped_broadcast_ball_positions_not_assigned_to_tactical": int((ball_valid & (mapped < 0)).sum()),
        "tracking_rows": int(status.object_position_count.sum()),
        "exported_player_positions": int(status.player_position_count.sum()),
        "exported_ball_positions": int(status.ball_position_available.sum()),
        "exported_ball_predictions": int(status.ball_is_prediction.sum())}
    out.parent.mkdir(parents=True, exist_ok=True)
    temporary = Path(tempfile.mkdtemp(prefix="." + run_id + "-", dir=out.parent))
    try:
        target = temporary / "tracking.csv"
        with target.open("w", newline="", encoding="utf-8") as stream:
            pd.DataFrame(columns=TRACKING_COLUMNS).to_csv(stream, index=False, lineterminator="\n")
            for start in range(0, len(table), config["chunk_frames"]):
                stop = min(len(table), start + config["chunk_frames"])
                chunk = export_chunk(table.iloc[start:stop], players, mapped[start:stop],
                                     predictions[start:stop], config["fps"])
                chunk.to_csv(stream, index=False, header=False, float_format="%.17g", lineterminator="\n")
        status.to_csv(temporary / "frame_status.csv", index=False, float_format="%.17g", lineterminator="\n")
        require(sha256(source) == config["source_sha256"] and sha256(roster_path) == config["roster_sha256"],
                "checkpoint inputs changed during export")
        manifest = {"model": MODEL, "phase": 1, "run_id": run_id,
            "mode": "accepted_checkpoint_export", "checkpoint_reconstruction_from_raw_video": False,
            "inputs": {"checkpoint": {"path": portable_path(source, repo_root), "sha256": sha256(source)},
                       "roster": {"path": portable_path(roster_path, repo_root), "sha256": sha256(roster_path)}},
            "config": config, "config_sha256": sha256(config_path),
            "adapter_code_sha256": {name: sha256(HERE / name) for name in ["extract.py", "validate.py"]},
            "output_sha256": {name: sha256(temporary / name) for name in ["tracking.csv", "frame_status.csv"]},
            "counts": counts, "target_timeline": "zero-based Tactical video frame",
            "source_timeline": "zero-based Broadcast video frame",
            "coordinate_units": {"normalized": True, "pitch_m": config["pitch_m"], "s_kph": "recorded speed, not height"},
            "observed_definition": "recorded coordinate rather than a generated prediction; ATD records are not claimed direct new detector observations",
            "player_identity": "anonymous source track slot, not a verified player or shirt number",
            "confidence_policy": "not supplied by the unified checkpoint, left missing",
            "source_flags_policy": "existing live-reference/highlight flags retained in frame_status; unavailable source flags or unmapped frames are unknown; reindexing does not verify replay-time accuracy",
            "missing_policy": "no fallback, clipping, interpolation or coordinate generation",
            "new_tracking_or_correction": False, "possession_inference": False,
            "position_accuracy_evaluated": False}
        write_json(temporary / "run_manifest.json", manifest)
        del table, status
        validate_run(temporary, repo_root, write=True)
        require(not out.exists(), f"output appeared during export: {out}")
        temporary.rename(out)
    finally:
        if temporary.exists():
            shutil.rmtree(temporary)
    return manifest


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, default=HERE / "config.json")
    parser.add_argument("--input-csv", type=Path)
    parser.add_argument("--roster", type=Path)
    parser.add_argument("--run-id", default="submission")
    parser.add_argument("--repo-root", type=Path, default=REPO_ROOT)
    args = parser.parse_args()
    try:
        manifest = extract(args.repo_root, args.config, args.run_id, args.input_csv, args.roster)
    except (OSError, ValueError, KeyError) as error:
        parser.exit(1, f"ball_tracking export failed: {error}\n")
    print(json.dumps({"model": MODEL, "run_id": args.run_id, "tracking_rows": manifest["counts"]["tracking_rows"]}))


if __name__ == "__main__":
    main()
