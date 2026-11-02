"""Make local-only annotated previews from saved algorithmic detections.

This script does not train, annotate truth, upload images, or run a detector.
It renders already generated boxes at their actual sampled source frame IDs.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd

HERE = Path(__file__).resolve().parent
ROOT = next((path for path in HERE.parents if (path / "games.py").is_file()), None)
if ROOT is None:
    raise RuntimeError("Could not find repository root containing games.py")


def elapsed_time(frame: int, fps: float) -> str:
    millis = round(int(frame) * 1000 / fps)
    hours, rest = divmod(millis, 3_600_000)
    minutes, rest = divmod(rest, 60_000)
    seconds, millis = divmod(rest, 1000)
    return f"{hours:02}:{minutes:02}:{seconds:02}.{millis:03}"


def validate_status(status: pd.DataFrame, frame_count: int) -> pd.DataFrame:
    required = {"frame", "time_seconds", "n_person", "n_ball", "n_person_in_pitch", "n_ball_in_pitch"}
    missing = required - set(status.columns)
    if missing:
        raise ValueError(f"frame_status.csv missing columns: {sorted(missing)}")
    if status.empty:
        raise ValueError("frame_status.csv has no sampled frames")
    frames = status["frame"]
    if not np.issubdtype(frames.dtype, np.integer):
        raise ValueError("Sampled frame IDs must be integers")
    if frames.duplicated().any() or (frames < 0).any() or (frames >= frame_count).any():
        raise ValueError("Sampled frame IDs must be unique and inside the video")
    return status.sort_values("frame").reset_index(drop=True)


def select_preview_frames(status: pd.DataFrame, start: int, duration_seconds: float,
                          source_fps: float, output_fps: float) -> list[int]:
    """Use only real sampled IDs, selecting at most one per requested output interval."""
    stop = start + duration_seconds * source_fps
    frames = status.loc[(status.frame >= start) & (status.frame < stop), "frame"].to_numpy()
    if not len(frames):
        raise ValueError(f"No sampled frames in requested source interval {start}..{int(stop)-1}")
    targets = np.arange(start, stop, source_fps / output_fps)
    result = []
    for target in targets:
        index = int(np.searchsorted(frames, target, side="left"))
        # Select the next recorded sample within this output interval. Never invent a frame.
        if index < len(frames) and frames[index] < target + source_fps / output_fps:
            frame = int(frames[index])
            if not result or frame != result[-1]:
                result.append(frame)
    if not result:
        raise ValueError("No recorded samples align with the requested preview intervals")
    return result


def read_exact(cap, frame: int, cv2) -> np.ndarray:
    """Check the decoder's reported 0-based position before/after a seek+read."""
    if not cap.set(cv2.CAP_PROP_POS_FRAMES, int(frame)):
        raise RuntimeError(f"Video backend rejected seek to source frame {frame}")
    before = cap.get(cv2.CAP_PROP_POS_FRAMES)
    if not np.isfinite(before) or abs(before - frame) > 0.25:
        raise RuntimeError(f"Seek position mismatch: wanted {frame}, backend reports {before}")
    success, image = cap.read()
    after = cap.get(cv2.CAP_PROP_POS_FRAMES)
    if not success or image is None:
        raise RuntimeError(f"Could not decode source frame {frame}")
    if not np.isfinite(after) or abs(after - (frame + 1)) > 0.25:
        raise RuntimeError(f"Read position mismatch: wanted next={frame+1}, backend reports {after}")
    return image


def annotate(image: np.ndarray, frame: int, fps: float, detections: pd.DataFrame,
             frame_status: pd.Series, cv2) -> np.ndarray:
    canvas = image.copy()
    height, width = canvas.shape[:2]
    scale = max(width / 1920, 0.5)
    for row in detections.itertuples(index=False):
        xy = np.asarray([row.x1_px, row.y1_px, row.x2_px, row.y2_px,
                         row.anchor_x_px, row.anchor_y_px], dtype=float)
        if not np.isfinite(xy).all():
            continue
        x1, y1, x2, y2 = np.rint(xy[:4]).astype(int)
        x1, x2 = np.clip([x1, x2], 0, width - 1)
        y1, y2 = np.clip([y1, y2], 0, height - 1)
        if x2 <= x1 or y2 <= y1:
            continue
        is_ball = int(row.class_id) == 0
        color = (60, 220, 80) if is_ball else (255, 180, 50)
        thickness = max(1, round(2 * scale))
        cv2.rectangle(canvas, (x1, y1), (x2, y2), color, thickness)
        anchor = (int(np.clip(round(xy[4]), 0, width-1)),
                  int(np.clip(round(xy[5]), 0, height-1)))
        cv2.circle(canvas, anchor, max(2, round(3 * scale)), color, -1)
        name = "ball candidate" if is_ball else str(row.class_name)
        label = f"{name} {float(row.confidence):.2f}"
        cv2.putText(canvas, label, (x1, max(16, y1 - 5)), cv2.FONT_HERSHEY_SIMPLEX,
                    0.48 * scale, color, max(1, round(scale)), cv2.LINE_AA)
    cv2.rectangle(canvas, (0, 0), (width, round(85 * scale)), (15, 15, 15), -1)
    lines = [f"source frame {frame} | elapsed {elapsed_time(frame, fps)} | sampled preview",
             f"person={int(frame_status.n_person)}  ball candidates={int(frame_status.n_ball)}"
             " | football classes, no team or identity"]
    for i, text in enumerate(lines):
        cv2.putText(canvas, text, (round(14*scale), round((30+i*32)*scale)),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.7*scale, (245, 245, 245),
                    max(1, round(scale)), cv2.LINE_AA)
    return canvas


def open_writer(path: Path, fps: float, size: tuple[int, int], cv2):
    """Prefer H264 MP4, then use the broadly available MPEG-4 Part 2 fallback."""
    for codec in ("avc1", "mp4v"):
        writer = cv2.VideoWriter(str(path), cv2.VideoWriter_fourcc(*codec), fps, size)
        if writer.isOpened():
            return writer, codec
        writer.release()
        if path.exists():
            path.unlink()
    raise RuntimeError("Neither avc1 nor mp4v MP4 encoding is available in OpenCV")


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--video", type=Path, default=ROOT / "mp4/BAR-ATH_tactical-001.mp4")
    parser.add_argument("--out", type=Path, default=HERE / "outputs")
    parser.add_argument("--start-frame", type=int, default=11250)
    parser.add_argument("--seconds", type=float, default=60.0)
    parser.add_argument("--preview-fps", type=float, default=1.0)
    args = parser.parse_args()
    if args.start_frame < 0 or args.seconds <= 0 or args.preview_fps <= 0:
        parser.error("start-frame must be nonnegative; seconds and preview-fps must be positive")
    import cv2

    detections = pd.read_csv(args.out / "detections.csv")
    expected = {"frame", "class_id", "confidence", "x1_px", "y1_px", "x2_px", "y2_px",
                "anchor_x_px", "anchor_y_px"}
    if expected - set(detections.columns):
        parser.error(f"detections.csv missing columns: {sorted(expected-set(detections.columns))}")
    status = pd.read_csv(args.out / "frame_status.csv")
    cap = cv2.VideoCapture(str(args.video))
    if not cap.isOpened():
        parser.error(f"Cannot open video: {args.video}")
    fps = float(cap.get(cv2.CAP_PROP_FPS))
    frame_count = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    width, height = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH)), int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    if not np.isfinite(fps) or fps <= 0 or width <= 0 or height <= 0:
        parser.error("Video metadata contains invalid fps or dimensions")
    status = validate_status(status, frame_count)
    if not np.allclose(status.time_seconds.to_numpy(), status.frame.to_numpy() / fps, atol=0.001):
        parser.error("frame_status time_seconds disagrees with source frame/fps")
    if not set(detections.frame.unique()).issubset(set(status.frame)):
        parser.error("Some detection frames have no frame_status row")
    selected = select_preview_frames(status, args.start_frame, args.seconds, fps, args.preview_fps)
    status_by_frame = status.set_index("frame")
    grouped = {int(frame): rows for frame, rows in detections.groupby("frame", sort=False)}
    empty = detections.iloc[:0]
    args.out.mkdir(parents=True, exist_ok=True)
    video_path = args.out / "preview.mp4"
    writer, codec = open_writer(video_path, args.preview_fps, (width, height), cv2)
    try:
        for frame in selected:
            image = read_exact(cap, frame, cv2)
            canvas = annotate(image, frame, fps, grouped.get(frame, empty), status_by_frame.loc[frame], cv2)
            writer.write(canvas)
    finally:
        writer.release()

    # Eight samples spread across all available detector output, not hand-selected outcomes.
    indices = np.unique(np.rint(np.linspace(0, len(status)-1, min(8, len(status)))).astype(int))
    contact_frames = status.iloc[indices].frame.astype(int).tolist()
    thumb_w = 640
    thumb_h = round(height * thumb_w / width)
    columns = 2
    rows = int(np.ceil(len(contact_frames) / columns))
    sheet = np.zeros((rows * thumb_h, columns * thumb_w, 3), dtype=np.uint8)
    try:
        for i, frame in enumerate(contact_frames):
            image = read_exact(cap, frame, cv2)
            canvas = annotate(image, frame, fps, grouped.get(frame, empty), status_by_frame.loc[frame], cv2)
            thumb = cv2.resize(canvas, (thumb_w, thumb_h), interpolation=cv2.INTER_AREA)
            row, col = divmod(i, columns)
            sheet[row*thumb_h:(row+1)*thumb_h, col*thumb_w:(col+1)*thumb_w] = thumb
    finally:
        cap.release()
    contact_path = args.out / "contact_sheet.jpg"
    if not cv2.imwrite(str(contact_path), sheet, [cv2.IMWRITE_JPEG_QUALITY, 90]):
        raise RuntimeError(f"Failed to write {contact_path}")
    check = cv2.VideoCapture(str(video_path))
    encoded_count = int(check.get(cv2.CAP_PROP_FRAME_COUNT))
    encoded_fps = float(check.get(cv2.CAP_PROP_FPS))
    check.release()
    if encoded_count != len(selected) or abs(encoded_fps - args.preview_fps) > 0.01:
        raise RuntimeError(f"Preview encode verification failed: {encoded_count} frames at {encoded_fps} fps")
    manifest = {
        "source_video": str(args.video.resolve()), "source_fps": fps,
        "source_frame_count": frame_count, "source_dimensions": [width, height],
        "preview_frame_ids": selected, "preview_fps": args.preview_fps,
        "preview_duration_seconds": len(selected) / args.preview_fps,
        "preview_source_start_frame": selected[0], "preview_source_last_frame": selected[-1],
        "preview_codec": codec, "preview_path": str(video_path.resolve()),
        "contact_sheet_frame_ids": contact_frames, "contact_sheet_path": str(contact_path.resolve()),
        "seek_check": "OpenCV CAP_PROP_POS_FRAMES checked before/after each read; no independent visual alignment verification",
        "scope": "Local rendering of algorithmic detections only; no manual truth labels, no accuracy scoring, no uploads.",
    }
    (args.out / "preview_manifest.json").write_text(json.dumps(manifest, indent=2) + "\n")
    print(f"Saved local preview: {len(selected)} sampled frames, codec={codec}, duration={len(selected)/args.preview_fps:.1f}s")
    print(f"Saved local contact sheet: {len(contact_frames)} timeline samples")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
