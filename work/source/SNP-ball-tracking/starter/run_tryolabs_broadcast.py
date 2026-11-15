"""Local, instrumented Tryolabs demo; no ATD, labels, uploads or submission writes.

See docs/TRYOLABS_BROADCAST.md for setup, provenance and interpretation.
Run with the separate data/derived/tryolabs_broadcast/.venv/bin/python.
"""

import argparse
from collections import Counter
import csv
import functools
import hashlib
import importlib.metadata
import json
import os
from pathlib import Path
import subprocess
import sys
import time
from tryolabs_runtime import configure_cache, resolve_device, DetectorPair, log_event

ROOT = Path(__file__).resolve().parents[1]
ASSETS = ROOT / "data/derived/tryolabs_broadcast"
configure_cache()

import cv2
import numpy as np
import torch


class EvidenceRun:
    """Diagnostic only: 20 uninterrupted fresh, unique-ball observations.

    This is observable proximity evidence, never a controlled-possession label.
    Missing/ambiguous ball, stale player, unknown kit or distance >45px resets it.
    """

    def __init__(self):
        self.kit = None
        self.count = 0

    def update(self, kit):
        self.count = self.count + 1 if kit is not None and kit == self.kit else int(kit is not None)
        self.kit = kit
        return kit if self.count >= 20 else None


def import_upstream():
    sys.path.insert(0, str(ASSETS / "upstream"))
    sys.path.insert(0, str(ASSETS / "yolov5"))
    from inference import Converter, HSVClassifier, InertiaClassifier
    from inference.colors import black, blue, green, white, yellow
    from norfair import Tracker
    from norfair.camera_motion import MotionEstimator
    from norfair.distances import mean_euclidean
    from run_utils import get_main_ball, update_motion_estimator
    from soccer import Match, Player, Team
    from soccer.ball import Ball
    return locals()


def load_model(path, device="cpu"):
    from models.common import AutoShape
    from models.experimental import attempt_load

    # These two specific public, hashed legacy checkpoints contain full models.
    # Restrict the compatibility override to checkpoint loading, then restore it.
    path = Path(path)
    if not path.is_file():
        raise FileNotFoundError(f"Model weights missing: {path}. Prepare local assets before running.")
    selected = resolve_device(device)
    original_load = torch.load
    try:
        torch.load = functools.partial(original_load, weights_only=False)
        model = AutoShape(attempt_load(path, device=torch.device("cpu")))
        model = model.to(selected).eval()
    finally:
        torch.load = original_load
    return model


def new_state(u):
    teams = [u["Team"](name="BLUE", abbreviation="BLU", color=(255, 120, 70)),
             u["Team"](name="LIGHT", abbreviation="LGT", color=(210, 210, 255))]
    # Generic upstream HSV ranges, with no match-specific tuning or H/A mapping.
    filters = [dict(name="BLUE", colors=[u["blue"]]),
               dict(name="LIGHT", colors=[u["white"]]),
               dict(name="OTHER", colors=[u["black"], u["green"], u["yellow"]])]
    match = u["Match"](home=teams[0], away=teams[1], fps=25)
    match.team_possession = teams[1]  # Preserve run.py's arbitrary away default.
    return dict(
        teams=teams, match=match,
        classifier=u["InertiaClassifier"](u["HSVClassifier"](filters=filters), inertia=20),
        player_tracker=u["Tracker"](distance_function=u["mean_euclidean"], distance_threshold=250,
                                    initialization_delay=3, hit_counter_max=90),
        ball_tracker=u["Tracker"](distance_function=u["mean_euclidean"], distance_threshold=150,
                                  initialization_delay=20, hit_counter_max=2000),
        motion=u["MotionEstimator"](), evidence=EvidenceRun(),
    )


def digest(path):
    h = hashlib.sha256()
    with path.open("rb") as f:
        for block in iter(lambda: f.read(1024 * 1024), b""):
            h.update(block)
    return h.hexdigest()


def caption(image, text, xy, color=(230, 230, 230), scale=.62):
    cv2.putText(image, text, xy, cv2.FONT_HERSHEY_SIMPLEX, scale, color, 1, cv2.LINE_AA)


def box(image, detection, color, label, stale=False):
    points = np.rint(detection.points * .5).astype(int)
    x1, y1 = points[0]; x2, y2 = points[1]
    if stale:
        for x in range(x1, x2, 10):
            cv2.line(image, (x, y1), (min(x+5, x2), y1), color, 1)
            cv2.line(image, (x, y2), (min(x+5, x2), y2), color, 1)
        for y in range(y1, y2, 10):
            cv2.line(image, (x1, y), (x1, min(y+5, y2)), color, 1)
            cv2.line(image, (x2, y), (x2, min(y+5, y2)), color, 1)
    else:
        cv2.rectangle(image, (x1, y1), (x2, y2), color, 1)
    caption(image, label, (max(0, x1), max(12, y1-4)), color, .37)


def render(frame, players, raw_balls, ball, row):
    raw = cv2.resize(frame, (960, 540))
    overlay = raw.copy()
    colors = {"BLUE": (255, 150, 80), "LIGHT": (225, 225, 255), "OTHER": (150, 150, 150)}
    for player in players:
        d = player.detection
        age = row["source_frame0"] - d.data["source_frame0"]
        kit = d.data.get("classification", "OTHER")
        box(overlay, d, (0, 170, 255) if age else colors[kit],
            f'{kit} #{d.data["id"]}' + (f" age{age}" if age else ""), bool(age))
    for d in raw_balls:
        box(overlay, d, (80, 255, 80), f'ball? {d.data["p"]:.2f}')
        center = tuple(np.rint(d.points.mean(axis=0)*.5).astype(int))
        cv2.circle(overlay, center, 12, (80, 255, 80), 1)
    if ball.detection is not None:
        age = row["main_ball_age_frames"]
        box(overlay, ball.detection, (0, 170, 255), f"legacy main ball age{age}", bool(age))
    canvas = np.full((800, 1920, 3), 22, np.uint8)
    canvas[42:582, :960] = raw
    canvas[42:582, 960:] = overlay
    caption(canvas, "BROADCAST SOURCE", (18, 29))
    caption(canvas, "TRYOLABS DETECTION / TRACKING (instrumented)", (978, 29))
    caption(canvas, f'Segment {row["segment"]} | source frame {row["source_frame0"]} | file time {row["source_seconds"]:.2f}s | 25fps', (18, 612))
    caption(canvas, f'Persons detected {row["raw_persons"]} / tracks {row["player_tracks"]} | ball candidates {row["raw_balls"]} | legacy ball age {row["main_ball_age_frames"]}', (18, 645))
    caption(canvas, f'Legacy owner {row["legacy_owner"]} (initial default LIGHT) | accumulated BLUE {row["legacy_blue_frames"]/25:.2f}s / LIGHT {row["legacy_light_frames"]/25:.2f}s', (18, 678), (100, 190, 255))
    caption(canvas, f'Fresh proximity evidence: {row["fresh_evidence_kit"] or "UNKNOWN"} | consecutive {row["fresh_run"]}/20 | {row["reason"]}', (18, 711), (100, 230, 180))
    caption(canvas, "Green: current ball candidate (unverified) | Orange/dashed: legacy or stale data | BLUE/LIGHT: HSV kit guesses, no H/A identity", (18, 744), scale=.57)
    caption(canvas, "Local debug only. Candidate detection rate is not accuracy. No dead-ball/replay model; no controlled-possession claim.", (18, 779), scale=.57)
    return canvas


def self_check(u):
    from norfair import Detection
    from types import SimpleNamespace
    r = EvidenceRun()
    assert all(r.update("BLUE") is None for _ in range(19))
    assert r.update("BLUE") == "BLUE"
    assert r.update(None) is None and r.count == 0
    assert r.update("LIGHT") is None and r.count == 1
    state = new_state(u); match = state["match"]
    ball = u["Ball"](Detection(points=np.array([[0, 0], [2, 2]]), data={}))
    try:
        match.update([], ball)
    except ValueError as e:
        assert "empty" in str(e)
    else:
        raise AssertionError("Expected upstream min([]) failure")
    match.possession_counter = 7
    before = match.duration
    match.update([], u["Ball"](None))
    assert match.duration == before + 1 and match.possession_counter == 7
    d = Detection(points=np.array([[1, 2], [3, 4]]), data={})
    obj = SimpleNamespace(live_points=np.array([True, True]), last_detection=d,
                          id=1, estimate=np.array([[100, 200], [300, 400]]))
    converted = u["Converter"].TrackedObjects_to_Detections([obj])[0]
    assert np.array_equal(converted.points, d.points) and not np.array_equal(converted.points, obj.estimate)
    print("SELF_CHECK_OK: strict reset, upstream empty-player failure, missing-ball accumulation/counter carry, last_detection conversion", flush=True)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--video", type=Path, default=ROOT / "data/broadcast/BAR-ATH_broadcast-001.mp4")
    parser.add_argument("--starts", type=float, nargs="+", default=[600, 1800, 3600])
    parser.add_argument("--seconds", type=float, default=10)
    parser.add_argument("--output", type=Path, default=ASSETS / "preview")
    parser.add_argument("--self-check", action="store_true")
    parser.add_argument("--device", default="cpu", help="cpu (legacy default), auto, cuda, cuda:0 or mps")
    parser.add_argument("--model-residency", choices=["auto", "single", "both"], default="auto")
    args = parser.parse_args()
    u = import_upstream()
    if args.self_check:
        self_check(u)
        return
    if args.seconds <= 0 or not np.isfinite(args.seconds) or any(s < 0 or not np.isfinite(s) for s in args.starts):
        parser.error("Starts must be finite and >=0; seconds must be finite and >0")
    out = args.output.resolve()
    if not out.is_relative_to(ASSETS.resolve()):
        parser.error("Output must stay under the git-ignored local analysis directory")
    out.mkdir(parents=True, exist_ok=False)
    cap = cv2.VideoCapture(str(args.video))
    fps = cap.get(cv2.CAP_PROP_FPS)
    source_count = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    if not cap.isOpened() or abs(fps-25) > .001 or (int(cap.get(3)), int(cap.get(4))) != (1920, 1080):
        raise ValueError("Expected decodable 1920x1080 25fps broadcast; preserve original 45px geometry")
    frames_per_segment = round(args.seconds * fps)
    if frames_per_segment < 1 or any(round(s*fps)+frames_per_segment > source_count for s in args.starts):
        raise ValueError("Requested segment outside video or shorter than one frame")
    source_stat = args.video.stat()
    torch.set_num_threads(4)
    start_time = time.monotonic()
    try:
        selected_device = resolve_device(args.device)
        detectors = DetectorPair(load_model, ASSETS, selected_device, args.model_residency, out / "runtime_events.jsonl")
    except Exception as exc:
        log_event(out / "runtime_events.jsonl", "run_failed", error=str(exc))
        cap.release()
        raise
    import imageio_ffmpeg
    command = [imageio_ffmpeg.get_ffmpeg_exe(), "-nostdin", "-v", "error", "-n", "-f", "rawvideo", "-pix_fmt", "bgr24",
               "-s", "1920x800", "-r", "25", "-i", "pipe:0", "-an", "-c:v", "libx264", "-preset", "veryfast", "-crf", "21",
               "-pix_fmt", "yuv420p", "-movflags", "+faststart", str(out / "broadcast_tryolabs.mp4")]
    rows, details, events = [], [], []
    encoder_log = (out / "encoder.log").open("w")
    encoder = subprocess.Popen(command, stdin=subprocess.PIPE, stderr=encoder_log)
    cvt = u["Converter"]
    try:
        for segment, seconds in enumerate(args.starts, 1):
            state = new_state(u)
            start_frame = round(seconds*fps)
            cap.set(cv2.CAP_PROP_POS_FRAMES, start_frame)
            for local_frame in range(frames_per_segment):
                frame0 = start_frame + local_frame
                ok, frame = cap.read()
                if not ok or abs(cap.get(cv2.CAP_PROP_POS_FRAMES)-(frame0+1)) > .1:
                    raise ValueError(f"Exact source decode failed at {frame0}")
                t0 = time.monotonic()
                # Preserve upstream BGR input and inference size, thresholds and NMS.
                pp, bb = detectors.predict([frame], [frame0])
                pdf, bdf = pp[0], bb[0]
                pdf = pdf[(pdf["name"] == "person") & (pdf["confidence"] > .35)]
                bdf = bdf[bdf["confidence"] > .3]
                raw_people = cvt.DataFrame_to_Detections(pdf)
                raw_balls = cvt.DataFrame_to_Detections(bdf)
                for d in raw_people + raw_balls:
                    d.data["source_frame0"] = frame0
                motion_error = None
                try:
                    transform = u["update_motion_estimator"](state["motion"], raw_people+raw_balls, frame)
                except (cv2.error, ValueError, np.linalg.LinAlgError) as e:
                    motion_error = f"{type(e).__name__}: {e}"
                    events.append(dict(frame0=frame0, kind="motion_reset", error=motion_error))
                    state = new_state(u)
                    transform = None
                pts = state["player_tracker"].update(detections=raw_people, coord_transformations=transform)
                bts = state["ball_tracker"].update(detections=raw_balls, coord_transformations=transform)
                pd = cvt.TrackedObjects_to_Detections(pts)
                bd = cvt.TrackedObjects_to_Detections(bts)
                pd = state["classifier"].predict_from_detections(pd, frame)
                players = u["Player"].from_detections(pd, teams=state["teams"])
                ball = u["get_main_ball"](bd)
                match = state["match"]
                empty_guard = bool(ball.detection is not None and not players)
                if empty_guard:
                    events.append(dict(frame0=frame0, kind="upstream_min_empty_players_guard"))
                match.update(players, u["Ball"](None) if empty_guard else ball)
                age = frame0-ball.detection.data["source_frame0"] if ball.detection is not None else None
                kit, distance = None, None
                if len(raw_balls) == 0:
                    reason = "NO_FRESH_BALL"
                elif len(raw_balls) > 1:
                    reason = "MULTIPLE_BALL_CANDIDATES"
                else:
                    fresh_players = [p for p in players if p.detection.data["source_frame0"] == frame0]
                    if not fresh_players:
                        reason = "NO_FRESH_PLAYER_TRACK"
                    else:
                        current_ball = u["Ball"](raw_balls[0])
                        nearest = min(fresh_players, key=lambda p: p.distance_to_ball(current_ball))
                        distance = float(nearest.distance_to_ball(current_ball))
                        if distance > 45:
                            reason = "BALL_PLAYER_DISTANCE_GT_45PX"
                        elif nearest.team is None:
                            reason = "UNKNOWN_KIT"
                        else:
                            kit = nearest.team.name
                            reason = "PROXIMITY_ONLY_NOT_CONTROL"
                evidence = state["evidence"].update(kit)
                row = dict(segment=segment, output_frame0=len(rows), source_frame0=frame0, source_seconds=frame0/fps,
                           raw_persons=len(raw_people), player_tracks=len(players), raw_balls=len(raw_balls),
                           ball_tracks=len(bd), main_ball_age_frames=age,
                           legacy_owner=match.team_possession.name,
                           legacy_blue_frames=state["teams"][0].possession,
                           legacy_light_frames=state["teams"][1].possession,
                           legacy_counter=match.possession_counter, nearest_fresh_distance_px=distance,
                           fresh_evidence_kit=evidence, fresh_run=state["evidence"].count, reason=reason,
                           empty_players_guard=empty_guard, motion_error=motion_error,
                           inference_pipeline_seconds=time.monotonic()-t0)
                canvas = render(frame, players, raw_balls, ball, row)
                encoder.stdin.write(canvas.tobytes())
                rows.append(row)
                details.append(dict(source_frame0=frame0, balls=bdf.to_dict("records"), persons=pdf.to_dict("records")))
                if local_frame in (0, frames_per_segment//2, frames_per_segment-1):
                    cv2.imwrite(str(out / f"segment_{segment}_frame_{frame0}.jpg"), canvas)
                if local_frame % 25 == 0:
                    print(json.dumps({k:row[k] for k in ["segment", "source_frame0", "raw_persons", "raw_balls", "main_ball_age_frames", "reason"]}), flush=True)
    finally:
        cap.release()
        detectors.close()
        encoder.stdin.close()
        encoder.wait()
        encoder_log.close()
        if rows:
            with (out / "frames.csv").open("w", encoding="utf-8", newline="") as f:
                writer = csv.DictWriter(f, fieldnames=list(rows[0]))
                writer.writeheader(); writer.writerows(rows)
        with (out / "detections.jsonl").open("w") as f:
            for detail in details:
                f.write(json.dumps(detail)+"\n")
        (out / "events.json").write_text(json.dumps(events, indent=2))
    if encoder.returncode:
        raise RuntimeError("H264 encoder failed; see encoder.log")
    verification = cv2.VideoCapture(str(out / "broadcast_tryolabs.mp4"))
    decoded = 0
    while True:
        ok, _ = verification.read()
        if not ok:
            break
        decoded += 1
    output_fps = verification.get(cv2.CAP_PROP_FPS)
    verification.release()
    if decoded != len(rows) or abs(output_fps-25) > .001:
        raise ValueError("Output frame count or fps mismatch")
    final_stat = args.video.stat()
    assert (source_stat.st_size, source_stat.st_mtime_ns) == (final_stat.st_size, final_stat.st_mtime_ns)
    segments = []
    for segment in range(1, len(args.starts)+1):
        rr = [r for r in rows if r["segment"] == segment]
        segments.append(dict(segment=segment, source_start_frame0=rr[0]["source_frame0"],
                             source_end_frame0=rr[-1]["source_frame0"], frames=len(rr),
                             raw_ball_candidate_frames=sum(r["raw_balls"] > 0 for r in rr),
                             multiple_ball_candidate_frames=sum(r["raw_balls"] > 1 for r in rr),
                             legacy_stale_ball_frames=sum(r["main_ball_age_frames"] is not None and r["main_ball_age_frames"] > 0 for r in rr),
                             fresh_evidence_frames=sum(r["fresh_evidence_kit"] is not None for r in rr),
                             reasons=dict(Counter(r["reason"] for r in rr)),
                             mean_person_detections=float(np.mean([r["raw_persons"] for r in rr]))))
    summary = dict(upstream_commit="b00c7c75a1d52bd5ca183521a35fc76fbd89cc0e",
                   yolov5_commit="d3ea0df8b9f923685ce5f2555c303b8eddbf83fd", inference_size=640,
                   torch=str(torch.__version__), packages={name:importlib.metadata.version(name) for name in
                   ["numpy", "opencv-python", "norfair", "pandas", "Pillow", "torchvision", "imageio-ffmpeg"]},
                   checkpoint_sha256={name:digest(ASSETS/name) for name in ["ball.pt", "yolov5x.pt"]},
                   source_video=str(args.video.resolve()), source_size_bytes=source_stat.st_size,
                   source_mtime_ns=source_stat.st_mtime_ns, source_unchanged=True,
                   device=selected_device, model_residency=detectors.residency, decoded_output_frames=decoded, output_fps=output_fps,
                   elapsed_seconds=time.monotonic()-start_time, events=len(events), segments=segments,
                   scope="Local video-only diagnostic, not accuracy evaluation or H/A/D submission")
    (out / "summary.json").write_text(json.dumps(summary, indent=2)+"\n")
    print(json.dumps(summary, indent=2), flush=True)


if __name__ == "__main__":
    main()
