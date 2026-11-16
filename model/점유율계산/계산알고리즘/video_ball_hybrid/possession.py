"""Hybrid video-ball + ATD-player possession; no possession labels are read.

Input detector CSV uses original zero-based video frame IDs and normalized pitch
coordinates. The ATD TXT is 1-based and the metadata maps each slot to a team.
Within every frame, prefer a confidence-selected video ball for at most 12 frames,
then fall back to the provided ATD ball. Team evidence is nearest tracked player;
the existing fixed model1 sequence decoder turns evidence into H/A/D.
"""
import argparse
import hashlib
import importlib.util
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd

FPS = 25
N_FRAMES = 199319
PITCH_M = np.array([105.3, 68.0])
MIN_CONFIDENCE = 0.10  # Fixed from the chosen ball detector author's recommendation, not GT tuning.
MAX_VIDEO_HOLD_FRAMES = 12
HOME, AWAY, MISSING = 0, 1, -1
LABELS = np.array(["H", "A", "D"])


def file_hash(path):
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for chunk in iter(lambda: handle.read(4 * 1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def normalized_flags(series):
    return series.astype(str).str.strip().str.lower().isin(["1", "1.0", "true"])


def choose_video_balls(detections, class_id, n_frames=N_FRAMES):
    """Highest confidence valid in-pitch candidate; selection never reads ATD."""
    needed = {"frame", "class_id", "confidence", "x_norm", "y_norm", "projection_valid", "in_pitch"}
    if missing := needed - set(detections.columns):
        raise ValueError(f"Missing detection columns: {sorted(missing)}")
    data = detections.copy()
    for col in ("frame", "class_id", "confidence", "x_norm", "y_norm"):
        data[col] = pd.to_numeric(data[col], errors="coerce")
    frame = data.frame.to_numpy(dtype=float)
    if not (np.isfinite(frame).all() and np.equal(frame, np.floor(frame)).all()
            and ((frame >= 0) & (frame < n_frames)).all()):
        raise ValueError("Detector frame IDs must be integer, zero-based and within the video")
    mask = ((data.class_id == class_id) & (data.confidence >= MIN_CONFIDENCE)
            & (data.confidence <= 1) & normalized_flags(data.projection_valid)
            & normalized_flags(data.in_pitch) & data.x_norm.between(0, 1) & data.y_norm.between(0, 1))
    data = data.loc[mask].sort_values(["frame", "confidence"], ascending=[True, False], kind="stable")
    return data.drop_duplicates("frame", keep="first")[["frame", "confidence", "x_norm", "y_norm"]]


def hold_video_balls(accepted, n_frames=N_FRAMES, hold_frames=MAX_VIDEO_HOLD_FRAMES):
    """Forward hold last observed video point for 0..hold_frames; never backfill."""
    ball = np.full((n_frames, 2), np.nan)
    confidence = np.full(n_frames, np.nan)
    if len(accepted):
        frame = accepted.frame.to_numpy(dtype=int)
        ball[frame] = accepted[["x_norm", "y_norm"]].to_numpy(dtype=float)
        confidence[frame] = accepted.confidence.to_numpy(dtype=float)
    frames = np.arange(n_frames)
    last = np.maximum.accumulate(np.where(np.isfinite(confidence), frames, -1))
    age = frames - last
    available = (last >= 0) & (age <= hold_frames)
    held_ball = np.full_like(ball, np.nan)
    held_confidence = np.full(n_frames, np.nan)
    held_ball[available] = ball[last[available]]
    held_confidence[available] = confidence[last[available]]
    return held_ball, held_confidence, np.where(available, age, -1)


def parse_xy(group):
    fields = group.split(",")
    try:
        xy = np.array([float(fields[0]), float(fields[1])], dtype=float)
    except (ValueError, IndexError):
        return None
    return xy if np.isfinite(xy).all() else None


def nearest_team(ball, player_groups, team_of_track):
    """Nearest player's team in real-pitch metres; no fabricated player ID."""
    if ball is None:
        return MISSING, np.nan, -1
    if len(player_groups) != len(team_of_track):
        raise ValueError(f"ATD slot count {len(player_groups)} != metadata {len(team_of_track)}")
    best_distance = float("inf")
    best_team, best_slot = MISSING, -1
    for slot, group in enumerate(player_groups):
        player = parse_xy(group)
        if player is None:
            continue
        distance = float(np.linalg.norm((player - ball) * PITCH_M))
        if distance < best_distance:
            best_distance, best_team, best_slot = distance, int(team_of_track[slot]), slot
    return best_team, best_distance if best_slot >= 0 else np.nan, best_slot


def infer_observations(raw_path, team_of_track, held_ball, ages, n_frames=N_FRAMES):
    """Return raw team observations and diagnostics; strict 1:1 ATD alignment."""
    obs = np.full(n_frames, MISSING, dtype=np.int8)
    source = np.full(n_frames, "missing", dtype="<U12")
    distance = np.full(n_frames, np.nan)
    nearest_slot = np.full(n_frames, -1, dtype=np.int16)
    atd_ball_present = np.zeros(n_frames, dtype=bool)
    n_records = 0
    with Path(raw_path).open() as stream:
        for line in stream:
            n_records += 1
            parts = line.strip().split(":")
            if len(parts) != 3 or int(parts[0]) != n_records or n_records > n_frames:
                raise ValueError(f"Malformed, duplicate or non-contiguous ATD record at line {n_records}")
            frame = n_records - 1
            atd_ball = parse_xy(parts[2])
            atd_ball_present[frame] = atd_ball is not None
            if ages[frame] >= 0:
                ball = held_ball[frame]
                source[frame] = "video" if ages[frame] == 0 else "video_hold"
            else:
                ball = atd_ball
                if ball is not None:
                    source[frame] = "atd_ball"
            obs[frame], distance[frame], nearest_slot[frame] = nearest_team(
                ball, parts[1].split(";"), team_of_track)
    if n_records != n_frames:
        raise ValueError(f"ATD has {n_records} records; expected {n_frames}")
    return obs, source, distance, nearest_slot, atd_ball_present


def discover_root(explicit=None):
    if explicit:
        candidates = [Path(explicit)]
    else:
        here = Path(__file__).resolve().parent
        candidates = [here, *here.parents, Path.cwd()]
    return next((p.resolve() for p in candidates if (p / "games.py").is_file()), None)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--detections", required=True, type=Path)
    parser.add_argument("--frame-status", type=Path)
    parser.add_argument("--ball-class", type=int, default=32)
    parser.add_argument("--repo", type=Path)
    parser.add_argument("--out", type=Path, required=True)
    args = parser.parse_args()
    root = discover_root(args.repo)
    if root is None:
        parser.error("Cannot locate repository games.py; provide --repo")
    model1 = root / "model/점유율계산/계산알고리즘/model1"
    sys.path.insert(0, str(root))
    sys.path.insert(0, str(model1))
    from starter import baseline
    spec = importlib.util.spec_from_file_location("_ball_hybrid_model1", model1 / "model.py")
    prior = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(prior)
    config_path = model1 / "config.json"
    config = json.loads(config_path.read_text())
    atd_txt = root / "data/tactical/BAR-ATH_tactical_atd.txt"
    atd_xml = root / "data/tactical/BAR-ATH_tactical_atd.xml"
    for path in (args.detections, atd_txt, atd_xml, config_path):
        if not path.is_file():
            parser.error(f"Missing required local file: {path}")
    if args.frame_status is not None and not args.frame_status.is_file():
        parser.error(f"Missing frame status: {args.frame_status}")
    selected_fields = ["frame", "class_id", "confidence", "x_norm", "y_norm", "projection_valid", "in_pitch"]
    detections = pd.read_csv(args.detections, usecols=selected_fields)
    if args.frame_status:
        status = pd.read_csv(args.frame_status, usecols=["frame"])
        if status.frame.duplicated().any() or not status.frame.is_monotonic_increasing:
            raise ValueError("Sampled frames must be unique and sorted")
        if not detections.frame.isin(status.frame).all():
            raise ValueError("Detection frames must occur in the sampling status")
    accepted = choose_video_balls(detections, args.ball_class)
    held_ball, held_confidence, ages = hold_video_balls(accepted)
    slots = baseline.team_of_each_track(str(atd_xml))
    args.out.mkdir(parents=True, exist_ok=True)
    for name in ("prediction.csv", "frame_diagnostics.csv", "run_manifest.json"):
        if (args.out / name).exists():
            raise FileExistsError(f"Refusing to overwrite existing {args.out / name}")
    inputs = [args.detections, atd_txt, atd_xml, config_path, model1 / "model.py"]
    if args.frame_status:
        inputs.append(args.frame_status)
    manifest = {
        "model": "video_ball_atd_players_model1_decoder", "status": "before_inference",
        "fixed_policy": {"ball_class_id": args.ball_class, "minimum_confidence": MIN_CONFIDENCE,
                         "maximum_forward_hold_frames": MAX_VIDEO_HOLD_FRAMES,
                         "selection": "highest confidence valid in-pitch video ball candidate before consulting ATD",
                         "fallback": "fresh provided ATD ball if no current/recent video detection",
                         "team": "nearest tracked player using ATD team slot metadata and metre distances",
                         "decoder": "existing model1 fixed H/A/D cost decoder and config",
                         "dead_ball": "model1 long-observation-gap heuristic, not a verified dead-ball detector"},
        "inputs": [{"path": str(path.resolve()), "sha256": file_hash(path)} for path in inputs],
        "code_sha256": file_hash(Path(__file__)), "ground_truth_used": False,
    }
    (args.out / "pre_evaluation_manifest.json").write_text(json.dumps(manifest, ensure_ascii=False, indent=2)+"\n")
    obs, source, distance, nearest_slot, atd_present = infer_observations(atd_txt, slots, held_ball, ages)
    states, _, _, _, objective = prior.decode(obs, config)
    submission = pd.DataFrame({"frame": np.arange(N_FRAMES), "label": LABELS[states]})
    submission.to_csv(args.out / "prediction.csv", index=False, lineterminator="\n")
    pd.DataFrame({"frame": np.arange(N_FRAMES), "ball_source": source, "video_age_frames": ages,
                  "video_confidence": held_confidence, "atd_ball_present": atd_present.astype(int),
                  "nearest_slot_zero_based": nearest_slot, "nearest_player_distance_m": distance,
                  "team_observation": np.where(obs == HOME, "H", np.where(obs == AWAY, "A", "MISSING")),
                  "label": LABELS[states]}).to_csv(args.out / "frame_diagnostics.csv", index=False)
    manifest.update(status="complete", sampled_detections=len(detections), accepted_video_ball_frames=len(accepted),
                    source_counts={name: int(np.sum(source == name)) for name in ("video", "video_hold", "atd_ball", "missing")},
                    observed_teams={str(name): int(np.sum(obs == i)) for i, name in ((0, "H"), (1, "A"), (-1, "MISSING"))},
                    label_counts={str(name): int(np.sum(states == i)) for i, name in enumerate(LABELS)},
                    objective=objective, prediction_sha256=file_hash(args.out / "prediction.csv"))
    (args.out / "run_manifest.json").write_text(json.dumps(manifest, ensure_ascii=False, indent=2)+"\n")
    print(json.dumps({key:manifest[key] for key in ("accepted_video_ball_frames", "source_counts", "observed_teams", "label_counts")},ensure_ascii=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
