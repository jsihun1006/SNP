"""Local YOLO11n baseline: video pixels -> boxes -> pitch-plane coordinates.

No ATD positions, team labels or possession GT are read during extraction.
The supplied per-frame homography is used for calibration only.
"""
import argparse
import csv
from datetime import datetime, timezone
import hashlib
import json
import os
from pathlib import Path
import time

HERE = Path(__file__).resolve().parent
ROOT = next((path for path in HERE.parents if (path / "games.py").is_file()), None)
if ROOT is None:
    raise RuntimeError("Could not find repository root containing games.py")
LOCAL_CONFIG = HERE / ".ultralytics"
LOCAL_CONFIG.mkdir(parents=True, exist_ok=True)
os.environ["YOLO_CONFIG_DIR"] = str(LOCAL_CONFIG)
os.environ["YOLO_OFFLINE"] = "true"
os.environ["YOLO_AUTOINSTALL"] = "false"
os.environ["PYTORCH_ENABLE_MPS_FALLBACK"] = "1"

import cv2
import numpy as np
import pandas as pd
import torch
import ultralytics
from ultralytics import YOLO, settings

from geometry import box_anchors, image_to_pitch

settings.update({"sync": False, "hub": False})

DETECTION_COLUMNS = (
    "frame", "time_seconds", "detection_index", "class_id", "class_name", "confidence",
    "x1_px", "y1_px", "x2_px", "y2_px", "anchor_x_px", "anchor_y_px",
    "x_norm", "y_norm", "x_m", "y_m", "projection_valid", "in_pitch", "homography_error",
)
FRAME_COLUMNS = ("frame", "time_seconds", "n_person", "n_ball", "n_person_in_pitch", "n_ball_in_pitch")


def sha256(path):
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for block in iter(lambda: handle.read(4 * 1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def write_json(path, value):
    Path(path).write_text(json.dumps(value, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")


def finite(value):
    return round(float(value), 8) if np.isfinite(value) else ""


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--video", type=Path, default=ROOT / "mp4/BAR-ATH_tactical-001.mp4")
    parser.add_argument("--homography", type=Path, default=ROOT / "data/tactical/BAR-ATH_tactical_homography.csv")
    parser.add_argument("--weights", type=Path, default=HERE / "weights/yolo11n.pt")
    parser.add_argument("--out", type=Path, default=HERE / "outputs")
    parser.add_argument("--stride", type=int, default=25, help="25 = 1fps from a 25fps video; 1 = every frame")
    parser.add_argument("--start-frame", type=int, default=0)
    parser.add_argument("--end-frame", type=int, help="exclusive; defaults to video length")
    parser.add_argument("--imgsz", type=int, default=1280)
    parser.add_argument("--conf", type=float, default=.15)
    parser.add_argument("--iou", type=float, default=.7)
    parser.add_argument("--batch", type=int, default=8)
    parser.add_argument("--device", choices=("auto", "cpu", "mps"), default="auto")
    args = parser.parse_args()
    if args.stride <= 0 or args.batch <= 0 or args.start_frame < 0:
        parser.error("stride and batch must be positive; start-frame must be nonnegative")
    for path in (args.video, args.homography, args.weights):
        if not path.is_file():
            parser.error(f"Missing local input: {path}")
    if (args.out / "detections.csv").exists():
        parser.error("Output already exists; use a new --out to preserve previous results")
    args.out.mkdir(parents=True, exist_ok=True)
    cap = cv2.VideoCapture(str(args.video), cv2.CAP_FFMPEG)
    if not cap.isOpened():
        raise RuntimeError("Could not open tactical video with OpenCV/FFmpeg")
    video = {"fps": float(cap.get(cv2.CAP_PROP_FPS)),
             "n_frames": int(cap.get(cv2.CAP_PROP_FRAME_COUNT)),
             "width": int(cap.get(cv2.CAP_PROP_FRAME_WIDTH)),
             "height": int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))}
    if video != {"fps": 25.0, "n_frames": 199319, "width": 1920, "height": 1080}:
        raise ValueError(f"Unexpected tactical video geometry/timebase; verify calibration first: {video}")
    end = video["n_frames"] if args.end_frame is None else min(args.end_frame, video["n_frames"])
    if end <= args.start_frame:
        parser.error("end-frame must exceed start-frame")
    matrices = pd.read_csv(args.homography)
    if len(matrices) != video["n_frames"] or not np.array_equal(matrices.frame, np.arange(video["n_frames"])):
        raise ValueError("Homography rows do not exactly match original frame IDs")
    hcols = [f"h{r}{c}" for r in range(3) for c in range(3)]
    homographies = matrices[hcols].to_numpy(float).reshape(-1, 3, 3)
    valid = matrices.valid.to_numpy(bool)
    error = matrices.reprojection_error.to_numpy(float)
    device = ("mps" if torch.backends.mps.is_available() else "cpu") if args.device == "auto" else args.device
    if device == "mps" and not torch.backends.mps.is_available():
        raise RuntimeError("MPS requested but unavailable")
    torch.set_num_threads(4)
    cv2.setNumThreads(2)
    config = {"model": "YOLO11n COCO pretrained", "classes": {"0": "person", "32": "sports ball"},
              "imgsz": args.imgsz, "conf": args.conf, "iou": args.iou, "batch": args.batch,
              "stride": args.stride, "device": device, "start_frame": args.start_frame,
              "end_frame_exclusive": end, "fps_sampled": video["fps"] / args.stride,
              "person_anchor": "bbox bottom centre", "ball_anchor": "bbox centre",
              "pitch_m": [105.3, 68.0], "tracking": "none; detection_index is local to each frame",
              "team_classification": "none"}
    manifest = {"created_at_utc": datetime.now(timezone.utc).isoformat(), "status": "running",
                "config": config, "video": video,
                "inputs": {"video": str(args.video.resolve()), "video_sha256": sha256(args.video),
                           "homography": str(args.homography.resolve()), "homography_sha256": sha256(args.homography),
                           "weights": str(args.weights.resolve()), "weights_sha256": sha256(args.weights)},
                "versions": {"ultralytics": ultralytics.__version__, "torch": torch.__version__,
                             "opencv": cv2.__version__, "numpy": np.__version__, "pandas": pd.__version__},
                "code_sha256": sha256(Path(__file__)),
                "source_video_frame_ids_preserved": True, "uploads": "none; local model, sync and hub disabled"}
    write_json(args.out / "run_manifest.json", manifest)
    model = YOLO(str(args.weights))
    if model.names[0] != "person" or model.names[32] != "sports ball":
        raise ValueError("Weights do not use the expected COCO class mapping")
    started = time.monotonic()
    sampled = 0
    inference_seconds = 0.0
    last_report = started
    batch_frames, batch_ids = [], []
    expected = len(range(args.start_frame, end, args.stride))

    with (args.out / "detections.csv").open("w", newline="") as det_handle, \
            (args.out / "frame_status.csv").open("w", newline="") as frame_handle:
        det_writer = csv.DictWriter(det_handle, fieldnames=DETECTION_COLUMNS)
        frame_writer = csv.DictWriter(frame_handle, fieldnames=FRAME_COLUMNS)
        det_writer.writeheader()
        frame_writer.writeheader()

        def process_batch():
            nonlocal sampled, inference_seconds, last_report
            if not batch_frames:
                return
            tick = time.monotonic()
            results = model.predict(batch_frames, imgsz=args.imgsz, conf=args.conf, iou=args.iou,
                                    classes=[0, 32], device=device, verbose=False, save=False,
                                    max_det=100, rect=True)
            inference_seconds += time.monotonic() - tick
            for frame_id, result in zip(batch_ids, results):
                boxes = result.boxes.xyxy.cpu().numpy()
                classes = result.boxes.cls.cpu().numpy().astype(int)
                confidences = result.boxes.conf.cpu().numpy()
                anchors = box_anchors(boxes, classes)
                projected = image_to_pitch(homographies[frame_id], anchors, valid=bool(valid[frame_id]))
                inside = projected["in_pitch"]
                for index, (box, cls, confidence, anchor) in enumerate(zip(boxes, classes, confidences, anchors)):
                    normalized, metres = projected["xy_normalized"][index], projected["xy_metres"][index]
                    det_writer.writerow(dict(zip(DETECTION_COLUMNS, (
                        frame_id, frame_id / video["fps"], index, int(cls), model.names[int(cls)], finite(confidence),
                        *[finite(v) for v in box], *[finite(v) for v in anchor],
                        *[finite(v) for v in normalized], *[finite(v) for v in metres],
                        int(projected["projection_valid"][index]), int(inside[index]), finite(error[frame_id]),
                    ))))
                frame_writer.writerow({"frame": frame_id, "time_seconds": frame_id / video["fps"],
                                       "n_person": int(np.sum(classes == 0)), "n_ball": int(np.sum(classes == 32)),
                                       "n_person_in_pitch": int(np.sum((classes == 0) & inside)),
                                       "n_ball_in_pitch": int(np.sum((classes == 32) & inside))})
                sampled += 1
            batch_frames.clear()
            batch_ids.clear()
            now = time.monotonic()
            if now - last_report >= 20 or sampled == expected:
                elapsed = now - started
                progress = {"sampled_frames": sampled, "expected_frames": expected,
                            "elapsed_seconds": round(elapsed, 2), "inference_seconds": round(inference_seconds, 2)}
                write_json(args.out / "progress.json", progress)
                det_handle.flush()
                frame_handle.flush()
                print(f"processed {sampled}/{expected} sampled frames; elapsed {elapsed:.1f}s", flush=True)
                last_report = now

        if args.start_frame:
            cap.set(cv2.CAP_PROP_POS_FRAMES, args.start_frame)
            if abs(cap.get(cv2.CAP_PROP_POS_FRAMES) - args.start_frame) > .5:
                raise RuntimeError("Decoder could not seek to the exact start frame")
        for frame_id in range(args.start_frame, end):
            if not cap.grab():
                raise RuntimeError(f"Video ended early at original frame {frame_id}")
            if (frame_id - args.start_frame) % args.stride:
                continue
            ok, frame = cap.retrieve()
            if not ok:
                raise RuntimeError(f"Could not decode original frame {frame_id}")
            batch_frames.append(frame)
            batch_ids.append(frame_id)
            if len(batch_frames) == args.batch:
                process_batch()
        process_batch()
    cap.release()
    if sampled != expected:
        raise RuntimeError(f"Sample count {sampled} != {expected}")
    manifest.update(status="complete", sampled_frames=sampled, inference_seconds=inference_seconds,
                    total_seconds=time.monotonic() - started,
                    output_sha256={name: sha256(args.out / name) for name in ("detections.csv", "frame_status.csv")})
    write_json(args.out / "run_manifest.json", manifest)
    print(f"Complete: {sampled} sampled frames, {manifest['total_seconds']:.1f}s; {args.out}", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
