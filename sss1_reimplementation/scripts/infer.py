import argparse
import csv
import json
import time
from pathlib import Path
import cv2
from ..detection import YOLODetector, DetectionEngine
from ..geometry import load_camera
from ..pipeline import Pipeline, FIELDS, coco_predictions
from ..visualization import draw_predictions
from ..image_io import read_image, write_image


def build_parser():
    p = argparse.ArgumentParser(description="Independent frame inference: detection IDs are NOT track IDs")
    p.add_argument("--input", required=True, type=Path)
    p.add_argument("--detector", default="yolo26l.pt")
    p.add_argument("--person-class", type=int, default=0)
    p.add_argument("--mode", choices=["full", "sahi", "baat"], default="baat")
    p.add_argument("--device", default="cpu")
    p.add_argument("--confidence", type=float, default=.25)
    p.add_argument("--pose-config", type=Path, default=Path(__file__).resolve().parents[1]/"configs/rtmpose_x_two_points.py")
    p.add_argument("--pose-checkpoint", type=Path)
    p.add_argument("--camera", type=Path, help="Static camera JSON or raw SynLoc camera directory")
    p.add_argument("--coco", type=Path, help="COCO image IDs/file names and optional per-image SynLoc calibration")
    p.add_argument("--image-id", type=int, default=0, help="Single-image ID; use --coco for evaluation")
    p.add_argument("--output", type=Path, required=True)
    p.add_argument("--frame-step", type=int, default=1)
    p.add_argument("--max-frames", type=int)
    return p


def run(args):
    if args.frame_step < 1 or (args.max_frames is not None and args.max_frames < 1):
        raise ValueError("Positive frame step and frame limit required")
    if not args.input.exists():
        raise FileNotFoundError(args.input)
    engine = DetectionEngine(YOLODetector(args.detector, args.device, person_class=args.person_class),
                             mode=args.mode, confidence=args.confidence)
    pose = None
    if args.pose_checkpoint:
        from ..pose.inference import MMPoseEstimator
        pose = MMPoseEstimator(args.pose_config, args.pose_checkpoint, args.device)
    pipeline = Pipeline(engine, pose)
    camera = load_camera(args.camera) if args.camera else None
    args.output.mkdir(parents=True, exist_ok=True)
    visualization = args.output / "visualizations"
    visualization.mkdir(exist_ok=True)
    timings, results = [], []
    start = time.perf_counter()
    video = None
    def frames():
        nonlocal video
        if args.coco:
            doc = json.loads(args.coco.read_text(encoding="utf-8"))
            from ..geometry import SynLocCamera
            for im in doc["images"]:
                path = args.input / im["file_name"]
                image_camera = camera
                if all(k in im for k in ("camera_matrix", "dist_poly", "undist_poly")):
                    image_camera = SynLocCamera(**{k: im[k] for k in ("width", "height", "camera_matrix", "dist_poly", "undist_poly")})
                yield read_image(path), im["id"], 0, image_camera
        elif args.input.suffix.lower() in {".jpg", ".jpeg", ".png", ".bmp", ".webp"}:
            yield read_image(args.input), args.image_id, 0, camera
        else:
            video = cv2.VideoCapture(str(args.input))
            if not video.isOpened():
                raise ValueError("Cannot open video")
            index = 0
            while True:
                ok, image = video.read()
                if not ok:
                    break
                if index % args.frame_step == 0:
                    yield image, index, index, camera
                index += 1
    try:
        with (args.output / "predictions.csv").open("w", encoding="utf-8", newline="") as file:
            writer = csv.DictWriter(file, fieldnames=FIELDS)
            writer.writeheader()
            for count, (image, image_id, frame, image_camera) in enumerate(frames()):
                if args.max_frames is not None and count >= args.max_frames:
                    break
                if image is None:
                    raise ValueError(f"Cannot read image {image_id}")
                rows, timing = pipeline.infer(image, image_id, frame, image_camera)
                writer.writerows(rows)
                results.extend(coco_predictions(rows))
                timings.append(dict(image_id=image_id, frame=frame, **timing))
                path = visualization / f"{image_id}_{frame}.jpg"
                if not write_image(path, draw_predictions(image, rows)):
                    raise OSError(f"Cannot write {path}")
    finally:
        if video is not None:
            video.release()
    if not timings:
        raise ValueError("No input frames decoded")
    elapsed = time.perf_counter()-start
    (args.output/"results.json").write_text(json.dumps(results, ensure_ascii=False, allow_nan=False, indent=2), encoding="utf-8")
    report = dict(mode=args.mode, pose_available=pose is not None, temporal_tracking=False,
                  frames=len(timings), total_io_inference_seconds=elapsed, total_fps=len(timings)/elapsed,
                  model_initialization_excluded=True, warmup_included=True, timings=timings)
    (args.output/"timing.json").write_text(json.dumps(report, allow_nan=False, indent=2), encoding="utf-8")
    print(json.dumps({k:v for k,v in report.items() if k != "timings"}, indent=2))


def main():
    run(build_parser().parse_args())


if __name__ == "__main__":
    main()
