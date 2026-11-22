"""View local tactical video alongside frame-aligned team width/length CSV values.

    python starter/view_team_extents.py
    python starter/view_team_extents.py --start-frame 12000
    python starter/view_team_extents.py --video /local/path/to/tactical.mp4

Starts paused. Space: play/pause; A/D: previous/next frame; J/L: -/+ one second;
B: toggle projected team bounds; Q/Esc: quit. Stepping pauses playback.
Drag the Frame bar below the video to seek and pause at any point in the video.
Team rectangles use second-smallest(x) to second-largest(x), with full min/max(y).
This trims one observation at each x end; it does not identify the goalkeeper.
Tactical homography maps their normalised corners to each video frame.
All video/CSV processing stays local. No GT, labels, uploads or frame exports.
"""

import argparse
from pathlib import Path
import sys
import time

import cv2
import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from games import GAME, GAMES, asset_paths, source_frames  # noqa: E402
from starter.load_homography import MATRIX_COLUMNS  # noqa: E402
from starter.team_extents import LENGTH_METHOD  # noqa: E402

WINDOW = "Tactical team width / length"
SEEK_BAR = "Frame"
COLORS = {"H": (220, 120, 35), "A": (45, 60, 220)}  # BGR
COLUMNS = ["frame"] + [
    side + "_" + metric
    for side in ("H", "A")
    for metric in ("width_m", "length_m", "n_tracks", "x_min", "x_max", "y_min", "y_max")
]


class ViewerError(Exception):
    """Safe diagnostic text that contains no private data rows."""


def load_metrics(path, expected_frames=None):
    if expected_frames is None:
        expected_frames = source_frames("tactical")
    try:
        table = pd.read_csv(path, usecols=COLUMNS + ["length_method"])
        if not table["length_method"].eq(LENGTH_METHOD).all():
            raise ViewerError("CSV uses a different length formula. Run starter/team_extents.py again.")
        table[COLUMNS] = table[COLUMNS].apply(pd.to_numeric, errors="raise")
    except (OSError, ValueError, pd.errors.ParserError):
        raise ViewerError("Cannot read metrics CSV. Run starter/team_extents.py first.") from None
    if len(table) != expected_frames or not np.array_equal(
        table["frame"].to_numpy(), np.arange(expected_frames)
    ):
        raise ViewerError("CSV must contain consecutive tactical frame numbers from zero.")
    for side in ("H", "A"):
        counts = table[side + "_n_tracks"].to_numpy(dtype=float)
        if not np.all(np.isfinite(counts) & (counts >= 0) & (counts == np.floor(counts))):
            raise ViewerError("CSV observation counts must be non-negative integers.")
        for metric in ("width_m", "length_m"):
            minimum_tracks = 4 if metric == "length_m" else 2
            values = table[side + "_" + metric].to_numpy(dtype=float)
            if not np.all(np.isnan(values) | (np.isfinite(values) & (values >= 0))):
                raise ViewerError("CSV extents must be non-negative finite values or empty.")
            if not np.array_equal(np.isnan(values), counts < minimum_tracks):
                raise ViewerError("CSV width needs two tracks; trimmed x length needs four.")
        for axis, metric, scale in (("x", "length_m", GAMES[GAME]["pitch"][0]),
                                    ("y", "width_m", GAMES[GAME]["pitch"][1])):
            low = table[side + "_" + axis + "_min"].to_numpy(dtype=float)
            high = table[side + "_" + axis + "_max"].to_numpy(dtype=float)
            known = counts >= (4 if axis == "x" else 2)
            if not (np.all(np.isnan(low[~known]) & np.isnan(high[~known]))
                    and np.all(np.isfinite(low[known]) & np.isfinite(high[known])
                               & (low[known] <= high[known]))
                    and np.allclose((high[known] - low[known]) * scale,
                                    table.loc[known, side + "_" + metric], atol=2e-6)):
                raise ViewerError("CSV bounds do not agree with extents. Run starter/team_extents.py again.")
    return table


def load_homographies(path, expected_frames):
    """Validate frame indices before positional access; never log private rows."""
    try:
        table = pd.read_csv(path, usecols=["frame", "valid"] + MATRIX_COLUMNS)
        table = table.apply(pd.to_numeric, errors="raise")
    except (OSError, ValueError, pd.errors.ParserError):
        raise ViewerError("Cannot read tactical homography CSV.") from None
    if len(table) != expected_frames or not np.array_equal(table["frame"], np.arange(expected_frames)):
        raise ViewerError("Homography frame indices must match the tactical CSV.")
    if not table["valid"].isin([0, 1]).all():
        raise ViewerError("Homography valid flag must be zero or one.")
    matrices = table[MATRIX_COLUMNS].to_numpy(dtype=float).reshape(-1, 3, 3)
    valid = table["valid"].to_numpy(dtype=bool)
    valid &= np.isfinite(matrices).all(axis=(1, 2))
    return matrices, valid


def projected_bounds(row, side, matrix):
    """Return pixel corners in L,W,L,W edge order, or None if unsafe to project."""
    bounds = np.array([row[side + "_" + name]
                       for name in ("x_min", "x_max", "y_min", "y_max")], dtype=float)
    if int(row[side + "_n_tracks"]) < 4 or not np.isfinite(bounds).all():
        return None
    if not np.isfinite(matrix).all() or np.linalg.matrix_rank(matrix) < 3:
        return None
    xmin, xmax, ymin, ymax = bounds
    corners = np.array([[xmin, ymin, 1], [xmax, ymin, 1],
                        [xmax, ymax, 1], [xmin, ymax, 1]])
    homogeneous = corners @ matrix.T
    denominator = homogeneous[:, 2]
    # Crossing the projective horizon makes a finite closed rectangle misleading.
    if not (np.all(denominator > 1e-10) or np.all(denominator < -1e-10)):
        return None
    pixels = homogeneous[:, :2] / denominator[:, None]
    if not np.isfinite(pixels).all() or np.any(np.abs(pixels) > 2**30):
        return None
    return pixels


def draw_bounds(image, row, matrix):
    """Draw clipped edges, preserving perspective and without altering source pixels."""
    overlay = image.copy()
    height, width = image.shape[:2]
    for side in ("H", "A"):
        points = projected_bounds(row, side, matrix)
        if points is None:
            continue
        points = np.rint(points).astype(np.int32)
        for index in range(4):
            start = tuple(map(int, points[index]))
            end = tuple(map(int, points[(index + 1) % 4]))
            visible, clipped_start, clipped_end = cv2.clipLine((0, 0, width, height), start, end)
            if not visible:
                continue
            cv2.line(overlay, clipped_start, clipped_end, COLORS[side], 2, cv2.LINE_AA)
            if index in (0, 1):
                metric, label = ("length_m", "L") if index == 0 else ("width_m", "W")
                midpoint = (points[index].astype(float) + points[(index + 1) % 4]) / 2
                if 0 <= midpoint[0] < width and 0 <= midpoint[1] < height:
                    if metric == "length_m":
                        label = "L(trim1)"
                    text = f"{side} {label}: {row[side + '_' + metric]:.1f}m"
                    x, y = map(int, midpoint)
                    text_width = cv2.getTextSize(text, cv2.FONT_HERSHEY_SIMPLEX, 0.6, 2)[0][0]
                    position = (max(0, min(x, width - text_width - 2)), max(20, min(y - 8, height - 3)))
                    cv2.putText(overlay, text, position, cv2.FONT_HERSHEY_SIMPLEX,
                                0.6, (0, 0, 0), 5, cv2.LINE_AA)
                    cv2.putText(overlay, text, position, cv2.FONT_HERSHEY_SIMPLEX,
                                0.6, COLORS[side], 2, cv2.LINE_AA)
    return overlay


def default_video():
    configured = Path(asset_paths(source="tactical")["video"])
    if configured.is_file():
        return configured
    # The existing local possession-overlay script uses this bundle filename.
    alternative = configured.with_name(configured.stem + "-001.mp4")
    return alternative if alternative.is_file() else configured


def read_frame(cap, target):
    """Sequential read where possible; verify requested position on explicit seeks."""
    current = int(round(cap.get(cv2.CAP_PROP_POS_FRAMES)))
    if current != target:
        if not cap.set(cv2.CAP_PROP_POS_FRAMES, target):
            raise ViewerError("Video backend cannot seek to the requested frame.")
        if int(round(cap.get(cv2.CAP_PROP_POS_FRAMES))) != target:
            raise ViewerError("Video seek position does not match the requested frame.")
    ok, image = cap.read()
    if not ok or int(round(cap.get(cv2.CAP_PROP_POS_FRAMES))) != target + 1:
        raise ViewerError("Video frame decoding or frame-position check failed.")
    return image


def format_metres(value):
    return "N/A" if pd.isna(value) else f"{value:.2f} m"


def render(image, row, frame, fps, playing, homography=None, lines_visible=True):
    """Put values above the image so the panel does not obscure the pitch."""
    height, width = image.shape[:2]
    panel_height = 198
    canvas = np.zeros((height + panel_height, max(width, 1100), 3), dtype=np.uint8)
    canvas[:panel_height] = (25, 25, 25)
    display = draw_bounds(image, row, homography) if lines_visible and homography is not None else image
    canvas[panel_height:panel_height + height, :width] = display

    def text(message, y, color=(235, 235, 235), size=0.72):
        cv2.putText(canvas, message, (20, y), cv2.FONT_HERSHEY_SIMPLEX,
                    size, color, 2, cv2.LINE_AA)

    elapsed = frame / fps
    line_status = "OFF" if not lines_visible else ("ON" if homography is not None else "UNAVAILABLE")
    text(f"Frame {frame} (0-based) | Video {int(elapsed // 60):02d}:{elapsed % 60:05.2f}"
         f" | {'PLAYING' if playing else 'PAUSED'} | Lines: {line_status}", 29)
    for side, name, y in (("H", "BARCELONA", 66), ("A", "ATHLETIC", 103)):
        count = int(row[side + "_n_tracks"])
        text(f"{side} {name} | Width: {format_metres(row[side + '_width_m'])}"
             f" | Length(trim1): {format_metres(row[side + '_length_m'])} | Tracks: {count}",
             y, COLORS[side])
    text("Drag Frame bar: seek | Space: play/pause | A/D: -/+1 frame | J/L: -/+1 sec | B: lines | Q: quit",
         145, size=0.55)
    text("L(trim1) = 2nd-largest x - 2nd-smallest x | N/A: W <2 tracks, L <4 tracks",
         180, (175, 175, 175), size=0.55)
    return canvas


def run(video, table, start_frame, homographies, homography_valid):
    if not video.is_file():
        raise ViewerError("Tactical video not found. Set --video to its local path.")
    if not 0 <= start_frame < len(table):
        raise ViewerError("--start-frame is outside the tactical video.")
    cap = cv2.VideoCapture(str(video))
    window_open = False
    try:
        if not cap.isOpened():
            raise ViewerError("Cannot open tactical video.")
        fps = cap.get(cv2.CAP_PROP_FPS)
        if not np.isfinite(fps) or not np.isclose(fps, GAMES[GAME]["fps"], atol=0.01):
            raise ViewerError("Video FPS must match the tactical timeline (25 fps).")
        if int(cap.get(cv2.CAP_PROP_FRAME_COUNT)) != len(table):
            raise ViewerError("Video frame count differs from tactical metrics CSV.")
        cv2.namedWindow(WINDOW, cv2.WINDOW_NORMAL)
        window_open = True
        cv2.resizeWindow(WINDOW, 1280, 850)
        # Poll after waitKeyEx processes GUI events. The callback intentionally does
        # nothing: updating the bar during playback must not trigger a manual seek.
        cv2.createTrackbar(SEEK_BAR, WINDOW, start_frame, len(table) - 1, lambda position: None)
        frame = start_frame
        decoded_frame = None
        playing = False
        lines_visible = True
        dirty = True
        next_tick = time.monotonic()
        while True:
            if dirty:
                if frame != decoded_frame:
                    image = read_frame(cap, frame)
                    decoded_frame = frame
                matrix = homographies[frame] if homography_valid[frame] else None
                cv2.imshow(WINDOW, render(image, table.iloc[frame], frame, fps, playing,
                                          matrix, lines_visible))
                if cv2.getTrackbarPos(SEEK_BAR, WINDOW) != frame:
                    cv2.setTrackbarPos(SEEK_BAR, WINDOW, frame)
                dirty = False
            key = cv2.waitKeyEx(10)
            if cv2.getWindowProperty(WINDOW, cv2.WND_PROP_VISIBLE) < 1:
                break
            if key in (27, ord("q"), ord("Q")):
                break
            selected_frame = cv2.getTrackbarPos(SEEK_BAR, WINDOW)
            if selected_frame != frame:
                frame = min(len(table) - 1, max(0, selected_frame))
                playing = False
                next_tick = time.monotonic() + 1 / fps
                dirty = True
                continue
            if key == ord(" "):
                playing = not playing
                next_tick = time.monotonic() + 1 / fps
                dirty = True
            elif key in (ord("b"), ord("B")):
                lines_visible = not lines_visible
                dirty = True
            elif key in tuple(map(ord, "aAdDjJlL")):
                delta = {"a": -1, "d": 1, "j": -round(fps), "l": round(fps)}[chr(key).lower()]
                frame = min(len(table) - 1, max(0, frame + delta))
                playing = False
                dirty = True
            elif playing and time.monotonic() >= next_tick:
                if frame == len(table) - 1:
                    playing = False
                else:
                    frame += 1
                # Never skip frames to catch up when decoding/rendering is slow.
                next_tick = time.monotonic() + 1 / fps
                dirty = True
    finally:
        cap.release()
        if window_open:
            cv2.destroyAllWindows()


def main():
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--csv", type=Path,
                        default=ROOT / "data" / "derived" / "tactical_team_extents.csv")
    parser.add_argument("--video", type=Path, default=default_video())
    parser.add_argument("--homography", type=Path,
                        default=Path(asset_paths(source="tactical")["homography"]))
    parser.add_argument("--start-frame", type=int, default=11185)
    args = parser.parse_args()
    try:
        table = load_metrics(args.csv)
        matrices, valid = load_homographies(args.homography, len(table))
        run(args.video, table, args.start_frame, matrices, valid)
    except ViewerError as error:
        print(str(error), file=sys.stderr)
        return 1
    except cv2.error:
        print("OpenCV video/display operation failed. Run locally in a GUI session.", file=sys.stderr)
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
