"""Play tactical/broadcast ATD coordinates side by side, with a frame seek bar.

    python model/영상추출/모델/tracking_alignment/compare_tracking_player.py
    python model/영상추출/모델/tracking_alignment/compare_tracking_player.py --export
    python model/영상추출/모델/tracking_alignment/compare_tracking_player.py --half-map
    python model/영상추출/모델/tracking_alignment/compare_tracking_player.py --half-map --export

Default diagnostic pairing: broadcast 7150 + k versus tactical 11150 + k.
Stop at the last available broadcast ATD frame (or earlier if tactical ends).
These user-selected starts are a VISUAL DIAGNOSTIC ONLY, not synchronization
anchors for possession inference. No tracking, labels or alignment tables change.
No source video or homography is used: both panels show raw normalized tracking
on the same pitch. Broadcast MP4/ATD frame correspondence remains unverified.

Space: play/pause; A/D: -/+1 frame; J/L: -/+1 s; I: slot labels; Q/Esc: quit.
Drag Broadcast frame to seek and pause. Starts paused; no frames are skipped.
--export renders every compared frame to a private local 25-fps MP4, without GUI.
Original finite out-of-pitch points are counted; points beyond panel bounds cannot
be displayed. Missing positions are omitted, with no interpolation/correction.
Slot numbers/team tags come from each source independently, not real player IDs.
All data/artifacts stay under model/영상추출/추출csv/tracking_alignment/. Never upload the output video.

--half-map loads the automatically generated per-half mapping, skips the official
halftime in the playback timeline and displays all official evaluation frames.
Unmapped/exhausted broadcast frames show an empty UNMATCHED panel, never a repeat
of another frame. Original tactical and broadcast ATD indices remain on screen.
"""

import argparse
import csv
import json
import math
import os
from pathlib import Path
import sys
import time
import xml.etree.ElementTree as ET

import cv2
import numpy as np

ROOT = next(parent for parent in Path(__file__).resolve().parents
            if (parent / "games.py").is_file())
OUTPUT_ROOT = ROOT / "model/영상추출/추출csv/tracking_alignment"
sys.path.insert(0, str(ROOT))
from games import GAME, GAMES, asset_paths, source_frames, half_video_blocks  # noqa: E402
from model.영상추출.모델.tracking_alignment.tracking_utils import track_teams, read_tracking  # noqa: E402

WINDOW = "Tracking comparison - constant offset diagnostic"
BAR = "Broadcast frame"
COLORS = {"H": (220, 120, 35), "A": (45, 60, 220)}
SIZE = (1280, 580)
OUTPUT_DIR = OUTPUT_ROOT / "sync" / "tracking_comparison_t011150_b007150"
HALF_MAP = OUTPUT_ROOT / 'sync/half_start_alignment/mapping.csv'


def comparison_length(tactical_count, broadcast_count, tactical_start, broadcast_start):
    if not (0 <= tactical_start < tactical_count and 0 <= broadcast_start < broadcast_count):
        raise ValueError("Comparison starts must lie inside the two ATD timelines.")
    return min(tactical_count - tactical_start, broadcast_count - broadcast_start)




def read_half_map(path, tactical_count, broadcast_count, blocks):
    """Validate the full mapping, then concatenate the official half frames only."""
    mapping = np.full(tactical_count, -1, dtype=np.int32)
    half_ids = np.zeros(tactical_count, dtype=np.int8)
    for half, first, last in blocks:
        half_ids[first:last + 1] = half
    seen = 0
    with open(path, newline='', encoding='utf-8') as handle:
        reader = csv.DictReader(handle)
        if not {'tactical_frame', 'broadcast_atd_frame0', 'half', 'status'}.issubset(reader.fieldnames or []):
            raise ValueError('Invalid half-map schema.')
        for frame, row in enumerate(reader):
            if frame >= tactical_count or int(row['tactical_frame']) != frame or int(row['half']) != half_ids[frame]:
                raise ValueError('Half-map frame/half indices differ from official configuration.')
            value = row['broadcast_atd_frame0'].strip()
            if value:
                target = int(value)
                if (not 0 <= target < broadcast_count or not half_ids[frame]
                        or row['status'] != 'mapped_by_half_anchor_assumption'):
                    raise ValueError('Invalid mapped broadcast frame/status.')
                mapping[frame] = target
            elif row['status'] == 'mapped_by_half_anchor_assumption':
                raise ValueError('Mapped status lacks a broadcast frame.')
            seen += 1
    if seen != tactical_count:
        raise ValueError('Half-map must preserve every original tactical row.')
    for half, _, _ in blocks:
        valid = np.flatnonzero((half_ids == half) & (mapping >= 0))
        if len(valid) > 1 and not np.all(np.diff(mapping[valid]) == np.diff(valid)):
            raise ValueError('Half-map does not have a constant offset within each half.')
    valid = np.flatnonzero(mapping >= 0)
    if len(valid) > 1 and not np.all(np.diff(mapping[valid]) > 0):
        raise ValueError('Half-map reuses/reverses broadcast frames across halves.')
    frames = np.flatnonzero(half_ids > 0)
    return frames, mapping[frames], half_ids[frames]


def load_half_comparison(path, data_dir=None):
    t_frames, b_frames, halves = read_half_map(path, source_frames('tactical'),
                                              source_frames('broadcast'), half_video_blocks(GAME))
    arrays, teams = {}, {}
    for source in ('tactical', 'broadcast'):
        paths = asset_paths(source=source, data_dir=data_dir)
        teams[source] = track_teams(paths['atd_meta'])
        full = read_tracking(paths['atd_raw'], teams[source], source_frames(source), 0, source_frames(source))
        if source == 'tactical':
            arrays[source] = full[t_frames]
        else:
            arrays[source] = np.full((len(b_frames), full.shape[1], 2), np.nan, dtype=np.float32)
            usable = b_frames >= 0
            arrays[source][usable] = full[b_frames[usable]]
        del full
    return arrays, teams, t_frames, b_frames, halves


def panel_geometry(panel):
    length, width = GAMES[GAME]["pitch"]
    scale = min(576 / length, 402 / width)
    left = panel * 640 + (640 - scale * length) / 2
    top = 122 + (402 - scale * width) / 2
    return left, top, scale


def pitch_point(x, y, panel):
    left, top, scale = panel_geometry(panel)
    return int(round(left + x * scale)), int(round(top + y * scale))


def make_template():
    canvas = np.full((SIZE[1], SIZE[0], 3), 28, dtype=np.uint8)
    length, width = GAMES[GAME]["pitch"]
    white = (210, 225, 210)
    for panel in (0, 1):
        a, b = pitch_point(0, 0, panel), pitch_point(length, width, panel)
        cv2.rectangle(canvas, a, b, (48, 108, 48), -1)
        cv2.rectangle(canvas, a, b, white, 1, cv2.LINE_AA)
        cv2.line(canvas, pitch_point(length / 2, 0, panel), pitch_point(length / 2, width, panel), white, 1)
        cv2.circle(canvas, pitch_point(length / 2, width / 2, panel),
                   round(9.15 * panel_geometry(panel)[2]), white, 1, cv2.LINE_AA)
        for side in (0, 1):
            for depth, box_width in ((16.5, 40.3), (5.5, 18.3)):
                x0, x1 = (0, depth) if side == 0 else (length - depth, length)
                cv2.rectangle(canvas, pitch_point(x0, (width - box_width) / 2, panel),
                              pitch_point(x1, (width + box_width) / 2, panel), white, 1)
    cv2.line(canvas, (640, 0), (640, SIZE[1]), (80, 80, 80), 1)
    return canvas


def timestamp(frame, fps):
    seconds = frame / fps
    return f"{int(seconds // 60):02d}:{seconds % 60:05.2f}"


def render(template, tactical, broadcast, t_teams, b_teams, t_frame, b_frame,
           fps=25, playing=False, labels=True, comparison_note=None):
    canvas = template.copy()
    length, width = GAMES[GAME]["pitch"]
    for panel, points, teams, name, frame in (
        (0, tactical, t_teams, "TACTICAL", t_frame),
        (1, broadcast, b_teams, "BROADCAST ATD", b_frame),
    ):
        if frame < 0:
            points = np.full_like(points, np.nan)
        origin = panel * 640 + 18
        def text(message, y, size=0.58, color=(235, 235, 235)):
            cv2.putText(canvas, message, (origin, y), cv2.FONT_HERSHEY_SIMPLEX,
                        size, color, 1, cv2.LINE_AA)
        text(name, 28, 0.75)
        if frame < 0:
            text('UNMATCHED: no broadcast ATD frame', 57, color=(70, 190, 245))
        else:
            text(f"Frame {frame} | ATD id {frame + 1} | {timestamp(frame, fps)}", 57)
        finite = np.isfinite(points).all(axis=1)
        team_array = np.asarray(teams)
        n_home = int((finite[:-1] & (team_array == "H")).sum())
        n_away = int((finite[:-1] & (team_array == "A")).sum())
        outside = int((finite & ((points[:, 0] < 0) | (points[:, 0] > 1)
                                 | (points[:, 1] < 0) | (points[:, 1] > 1))).sum())
        text(f"Seen H:{n_home} A:{n_away} | Ball:{'yes' if finite[-1] else 'missing'} | Outside:{outside}", 84, 0.5)
        for index in np.flatnonzero(finite):
            x, y = points[index]
            # Guard against overflow from a malformed but finite export position.
            if abs(float(x)) > 100 or abs(float(y)) > 100:
                continue
            px, py = pitch_point(float(x) * length, float(y) * width, panel)
            if not (panel * 640 + 4 <= px < (panel + 1) * 640 - 4 and 104 <= py < 532):
                continue
            ball = index == len(teams)
            color = (250, 250, 250) if ball else COLORS[teams[index]]
            cv2.circle(canvas, (px, py), 6, (0, 0, 0), -1, cv2.LINE_AA)
            cv2.circle(canvas, (px, py), 4, color, -1, cv2.LINE_AA)
            if labels:
                label = "BALL" if ball else f"{teams[index]}:{index + 1}"
                position = (min(px + 7, (panel + 1) * 640 - 54), max(116, py - 5))
                cv2.putText(canvas, label, position, cv2.FONT_HERSHEY_SIMPLEX, 0.36, (0, 0, 0), 3, cv2.LINE_AA)
                cv2.putText(canvas, label, position, cv2.FONT_HERSHEY_SIMPLEX, 0.36, color, 1, cv2.LINE_AA)
    mode = "PLAYING" if playing else "PAUSED"
    cv2.putText(canvas, f"{mode} | Space: play/pause | A/D: 1 frame | J/L: 1 sec | I: IDs | Q/Esc: quit",
                (18, 550), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (235, 235, 235), 1, cv2.LINE_AA)
    note = comparison_note or f"FIXED OFFSET DIAGNOSTIC: T = B + {t_frame - b_frame} | cuts/replays/halftime are not aligned"
    cv2.putText(canvas, note,
                (18, 573), cv2.FONT_HERSHEY_SIMPLEX, 0.49, (70, 200, 245), 1, cv2.LINE_AA)
    return canvas


def render_index(template, tactical, broadcast, t_teams, b_teams, index, t_start, b_start,
                 fps, playing, labels=True, t_frames=None, b_frames=None, half_ids=None,
                 anchor_tactical_frame=None):
    if t_frames is None:
        return render(template, tactical[index], broadcast[index], t_teams, b_teams,
                      t_start + index, b_start + index, fps, playing, labels)
    tf, bf = int(t_frames[index]), int(b_frames[index])
    half = int(half_ids[index])
    state = f'T = B + {tf - bf}' if bf >= 0 else 'UNMATCHED BROADCAST'
    note = f'HALF {half} | {state} | automatic start + constant offset assumption; NOT verified sync'
    if anchor_tactical_frame is not None:
        direction = ('BACKWARD EXTRAPOLATION' if tf < anchor_tactical_frame else
                     'REFERENCE ANCHOR' if tf == anchor_tactical_frame else 'AFTER REFERENCE ANCHOR')
        note = f'HALF {half} | {direction} | {state} | assumed timing, NOT verified sync'
    return render(template, tactical[index], broadcast[index], t_teams, b_teams,
                  tf, bf, fps, playing, labels, comparison_note=note)


def view(tactical, broadcast, t_teams, b_teams, t_start, b_start, fps,
         t_frames=None, b_frames=None, half_ids=None, start_index=0, anchor_tactical_frame=None):
    count = len(broadcast)
    if not 0 <= start_index < count:
        raise ValueError('Initial comparison position is outside the loaded range.')
    template = make_template()
    cv2.namedWindow(WINDOW, cv2.WINDOW_NORMAL)
    try:
        cv2.resizeWindow(WINDOW, *SIZE)
        bar_name = 'Comparison step (halves)' if t_frames is not None else BAR
        bar_start = 0 if t_frames is not None else b_start
        cv2.createTrackbar(bar_name, WINDOW, bar_start + start_index, bar_start + count - 1, lambda position: None)
        cv2.setTrackbarMin(bar_name, WINDOW, bar_start)
        index, playing, labels, dirty = start_index, False, True, True
        tick = time.monotonic()
        while True:
            if dirty:
                cv2.imshow(WINDOW, render_index(template, tactical, broadcast, t_teams, b_teams,
                                                index, t_start, b_start, fps, playing, labels,
                                                t_frames, b_frames, half_ids, anchor_tactical_frame))
                if cv2.getTrackbarPos(bar_name, WINDOW) != bar_start + index:
                    cv2.setTrackbarPos(bar_name, WINDOW, bar_start + index)
                dirty = False
            key = cv2.waitKeyEx(10)
            if cv2.getWindowProperty(WINDOW, cv2.WND_PROP_VISIBLE) < 1 or key in (27, ord('q'), ord('Q')):
                break
            requested = min(count - 1, max(0, cv2.getTrackbarPos(bar_name, WINDOW) - bar_start))
            if requested != index:
                index, playing, dirty = requested, False, True
                continue
            if key == ord(' '):
                playing, dirty = not playing, True
                tick = time.monotonic() + 1 / fps
            elif key in (ord('i'), ord('I')):
                labels, dirty = not labels, True
            elif key in tuple(map(ord, 'aAdDjJlL')):
                delta = {'a': -1, 'd': 1, 'j': -round(fps), 'l': round(fps)}[chr(key).lower()]
                index, playing, dirty = min(count - 1, max(0, index + delta)), False, True
            elif playing and time.monotonic() >= tick:
                if index == count - 1:
                    playing = False
                else:
                    index += 1
                dirty = True
                tick = time.monotonic() + 1 / fps
    finally:
        cv2.destroyAllWindows()


def export_video(tactical, broadcast, t_teams, b_teams, t_start, b_start, fps, output,
                 t_frames=None, b_frames=None, half_ids=None, anchor_tactical_frame=None):
    output = Path(output)
    output.parent.mkdir(parents=True, exist_ok=True)
    temporary = output.with_name(output.stem + '.partial.mp4')
    writer = cv2.VideoWriter(str(temporary), cv2.VideoWriter_fourcc(*'mp4v'), fps, SIZE)
    if not writer.isOpened():
        writer.release()
        raise ValueError('Cannot open local MP4 writer.')
    succeeded = False
    try:
        template = make_template()
        count = len(broadcast)
        for index in range(count):
            writer.write(render_index(template, tactical, broadcast, t_teams, b_teams, index,
                                      t_start, b_start, fps, True, True, t_frames, b_frames, half_ids,
                                      anchor_tactical_frame))
            if (index + 1) % 5000 == 0:
                print(f'Rendered {index + 1}/{count} comparison frames.', flush=True)
        succeeded = True
    finally:
        writer.release()
        if not succeeded:
            temporary.unlink(missing_ok=True)
    # Reopen to verify stored frame count and FPS before publishing the complete file.
    cap = cv2.VideoCapture(str(temporary))
    try:
        valid = (cap.isOpened() and int(cap.get(cv2.CAP_PROP_FRAME_COUNT)) == len(broadcast)
                 and math.isclose(cap.get(cv2.CAP_PROP_FPS), fps, abs_tol=0.01))
    finally:
        cap.release()
    if not valid:
        temporary.unlink(missing_ok=True)
        raise ValueError('Exported MP4 metadata verification failed.')
    os.replace(temporary, output)


def main():
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument('--tactical-start', type=int, default=11150)
    parser.add_argument('--broadcast-start', type=int, default=7150)
    parser.add_argument('--data-dir', help='Local input bundle root.')
    parser.add_argument('--export', action='store_true', help='Render all compared frames to MP4, without opening a window.')
    parser.add_argument('--half-map', type=Path, nargs='?', const=HALF_MAP,
                        help='Use automatic per-half mapping; omit path for the standard mapping.csv.')
    args = parser.parse_args()
    try:
        if args.half_map is not None:
            arrays, teams, t_frames, b_frames, halves = load_half_comparison(args.half_map, args.data_dir)
            print('Loaded local per-half comparison arrays; no coordinate rows logged.', flush=True)
            if args.export:
                folder = OUTPUT_ROOT / 'sync/half_aligned_comparison'
                export_video(arrays['tactical'], arrays['broadcast'], teams['tactical'], teams['broadcast'],
                             0, 0, GAMES[GAME]['fps'], folder / 'tracking_comparison.mp4',
                             t_frames, b_frames, halves)
                manifest = {'frames': len(t_frames), 'fps': GAMES[GAME]['fps'],
                            'mapped_frames': int((b_frames >= 0).sum()),
                            'unmatched_frames_shown_blank': int((b_frames < 0).sum()),
                            'halftime_excluded_from_playback': True, 'verified_synchronization': False,
                            'mapping': str(args.half_map)}
                (folder / 'manifest.json').write_text(json.dumps(manifest, indent=2) + '\n', encoding='utf-8')
                print('Saved private local per-half comparison MP4 and manifest.', flush=True)
            else:
                view(arrays['tactical'], arrays['broadcast'], teams['tactical'], teams['broadcast'],
                     0, 0, GAMES[GAME]['fps'], t_frames, b_frames, halves)
            return 0
        count = comparison_length(source_frames('tactical'), source_frames('broadcast'),
                                  args.tactical_start, args.broadcast_start)
        arrays, teams = {}, {}
        for source, start in (('tactical', args.tactical_start), ('broadcast', args.broadcast_start)):
            paths = asset_paths(data_dir=args.data_dir, source=source)
            teams[source] = track_teams(paths['atd_meta'])
            arrays[source] = read_tracking(paths['atd_raw'], teams[source], source_frames(source), start, count)
        print('Loaded local coordinate arrays. No coordinate rows logged.', flush=True)
        fps = GAMES[GAME]['fps']
        folder = OUTPUT_ROOT / 'sync' / f'tracking_comparison_t{args.tactical_start:06d}_b{args.broadcast_start:06d}'
        if args.export:
            export_video(arrays['tactical'], arrays['broadcast'], teams['tactical'], teams['broadcast'],
                         args.tactical_start, args.broadcast_start, fps, folder / 'tracking_comparison.mp4')
            manifest = {'broadcast_start_atd_frame0': args.broadcast_start,
                        'broadcast_end_atd_frame0': args.broadcast_start + count - 1,
                        'tactical_start_atd_frame0': args.tactical_start,
                        'tactical_end_atd_frame0': args.tactical_start + count - 1,
                        'frames': count, 'fps': fps, 'verified_synchronization': False,
                        'pairing': 'constant_offset_visual_diagnostic_only'}
            (folder / 'manifest.json').write_text(json.dumps(manifest, indent=2) + '\n', encoding='utf-8')
            print('Saved private local tracking_comparison.mp4 and manifest.json.', flush=True)
        else:
            view(arrays['tactical'], arrays['broadcast'], teams['tactical'], teams['broadcast'],
                 args.tactical_start, args.broadcast_start, fps)
    except (OSError, ValueError, cv2.error, ET.ParseError):
        print('Local tracking comparison failed. Check source files and display/codec support; no private rows logged.',
              file=sys.stderr)
        return 1
    return 0


if __name__ == '__main__':
    raise SystemExit(main())
