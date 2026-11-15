"""Render ALL non-X frames to one H.264 vector MP4 and an offline frame player.

    python starter/view_all_movement_vectors.py

Reads full source label tables, final alignment and raw tracking; no event catalog.
Use the original adjacent source frames for coordinate velocities, retaining only
derivatives whose two frames are non-X in the same half. Excluded intervals are
removed from the playback timeline without bridging their positions/velocities.
Tactical non-X rows define playback; mapped broadcast X rows appear unmatched.
Require every non-X broadcast frame to be represented, so neither source loses
requested data silently. Raw source labels/tracking and existing event views stay
unchanged. Slot IDs are not verified player IDs; no clipping/correction or GT.

Private data/derived/full_movement_vectors/: full_movement_vectors.mp4,
full_movement_vector_player.html, frame_timeline.csv, report.json.
The HTML embeds the local MP4 and offers frame stepping and original-frame seeks.
Arrow length = coordinate velocity * display_seconds, not a position forecast.
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

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from games import GAME, GAMES, asset_paths, source_frames  # noqa: E402
from starter.tracking_utils import track_teams, read_mapping, source_state, pack_source_frames  # noqa: E402
from starter.compare_tracking_player import read_tracking, make_template, panel_geometry, pitch_point, SIZE, COLORS, timestamp  # noqa: E402

FOLDER = ROOT/'data/derived/full_movement_vectors'


def select_timeline(mapping, states):
    tf = np.flatnonzero(states['tactical']['active']).astype(np.int32)
    if not len(tf):
        raise ValueError('No non-X tactical frames.')
    bf = mapping[tf].copy()
    valid = bf >= 0
    bf[valid & ~states['broadcast']['active'][np.maximum(bf, 0)]] = -1
    mapped = bf[bf >= 0]
    if not np.array_equal(mapped, np.flatnonzero(states['broadcast']['active'])):
        raise ValueError('Non-X broadcast coverage is incomplete, reused or out of order.')
    halves = states['tactical']['halves'][tf]
    if np.any(halves == 0):
        raise ValueError('Non-X frames must have explicit half tags.')
    if not np.array_equal(halves[bf >= 0], states['broadcast']['halves'][bf[bf >= 0]]):
        raise ValueError('Mapped non-X source half tags differ.')
    return tf, bf, halves


def timeline_segments(tf, bf, halves):
    """Compact mapping, split on every original-index gap or half change."""
    split = ((np.diff(tf) != 1) | (np.diff(halves) != 0)
             | ((bf[1:] >= 0) != (bf[:-1] >= 0))
             | ((bf[1:] >= 0) & (bf[:-1] >= 0) & (np.diff(bf) != 1)))
    firsts = np.r_[0, np.flatnonzero(split)+1]
    ends = np.r_[firsts[1:], len(tf)]
    return [{'video_first': int(first), 'count': int(end-first), 'tactical_first': int(tf[first]),
             'broadcast_first': int(bf[first]) if bf[first] >= 0 else None,
             'half': '1st Half' if halves[first] == 1 else '2nd Half'} for first, end in zip(firsts, ends)]


def pack_non_x(positions, frames, state, fps, pitch):
    valid = frames >= 0
    if np.any(~state['active'][frames[valid]]):
        raise ValueError('An X frame was selected for full movement playback.')
    packed = pack_source_frames(positions, frames, fps, pitch)
    eligible = np.zeros(len(frames), dtype=bool)
    eligible[valid] = state['eligible'][frames[valid]]
    packed[~eligible, :, 2:] = np.nan
    return packed


def render_frame(template, packed, teams, tf, bf, half, video_frame, count, fps, display_seconds):
    canvas = template.copy()
    pitch_length, pitch_width = GAMES[GAME]['pitch']
    for panel, source, frame in [(0, 'tactical', tf), (1, 'broadcast', bf)]:
        points = packed[source]
        origin = panel*640+18
        def text(message, y, scale=.52, color=(235, 235, 235)):
            cv2.putText(canvas, message, (origin, y), cv2.FONT_HERSHEY_SIMPLEX, scale, color, 1, cv2.LINE_AA)
        text(source.upper()+' | '+('1st Half' if half == 1 else '2nd Half'), 28, .7)
        if frame < 0:
            text('UNMATCHED BROADCAST', 58, color=(70, 190, 245))
            continue
        text(f'Frame {frame} | ATD id {frame+1} | {timestamp(frame, fps)}', 57)
        finite = np.isfinite(points[:, :2]).all(axis=1)
        vector_valid = np.isfinite(points[:-1, 2:]).all(axis=1)
        team_array = np.asarray(teams[source])
        nh = int((finite[:-1] & (team_array == 'H')).sum())
        na = int((finite[:-1] & (team_array == 'A')).sum())
        text(f'Seen H:{nh} A:{na} | Vectors:{int(vector_valid.sum())} | Ball:{"yes" if finite[-1] else "missing"}', 84, .48)
        scale = panel_geometry(panel)[2]
        bounds = (panel*640+4, 104, 632, 428)
        for slot in np.flatnonzero(finite):
            nx, ny, vx, vy = points[slot]
            if abs(float(nx)) > 100 or abs(float(ny)) > 100:
                continue
            x, y = pitch_point(float(nx)*pitch_length, pitch_width-float(ny)*pitch_width, panel)
            if not panel*640+4 <= x < (panel+1)*640-4 or not 104 <= y < 532:
                continue
            ball = slot == len(teams[source])
            color = (250, 250, 250) if ball else COLORS[teams[source][slot]]
            if not ball and vector_valid[slot] and (vx != 0 or vy != 0):
                ex, ey = x+float(vx)*display_seconds*scale, y-float(vy)*display_seconds*scale
                if math.isfinite(ex) and math.isfinite(ey) and max(abs(ex), abs(ey)) < 1e9:
                    ok, start, end = cv2.clipLine(bounds, (x, y), (round(ex), round(ey)))
                    if ok and start != end:
                        distance = math.hypot(end[0]-start[0], end[1]-start[1])
                        cv2.arrowedLine(canvas, start, end, color, 1, cv2.LINE_AA,
                                        tipLength=min(.45, 6/max(distance, 1)))
            cv2.circle(canvas, (x, y), 5, (8, 20, 8), -1, cv2.LINE_8)
            cv2.circle(canvas, (x, y), 3, color, -1, cv2.LINE_8)
            label = 'BALL' if ball else f'{teams[source][slot]}:{slot+1}'
            at = (min(x+6, (panel+1)*640-48), max(116, y-5))
            cv2.putText(canvas, label, at, cv2.FONT_HERSHEY_SIMPLEX, .32, (8, 20, 8), 2, cv2.LINE_8)
            cv2.putText(canvas, label, at, cv2.FONT_HERSHEY_SIMPLEX, .32, color, 1, cv2.LINE_8)
    cv2.putText(canvas, f'Video frame {video_frame}/{count-1} | Non-X only | Arrow = coordinate velocity x {display_seconds:g} s',
                (18, 550), cv2.FONT_HERSHEY_SIMPLEX, .48, (235, 235, 235), 1, cv2.LINE_AA)
    cv2.putText(canvas, 'Original adjacent-frame derivative | gaps/half boundaries reset vectors | source slots; full sync unverified',
                (18, 573), cv2.FONT_HERSHEY_SIMPLEX, .43, (70, 200, 245), 1, cv2.LINE_AA)
    return canvas


def export_movie(positions, teams, states, tf, bf, halves, output, fps=25, display_seconds=1., chunk_frames=2500,
                 packed_transform=None, frame_overlay=None, output_size=SIZE):
    output.parent.mkdir(parents=True, exist_ok=True)
    temporary = output.with_name(output.stem+'.partial.mp4')
    writer = cv2.VideoWriter(str(temporary), cv2.VideoWriter_fourcc(*'avc1'), fps, output_size)
    if not writer.isOpened():
        writer.release()
        raise ValueError('H.264 writer unavailable.')
    template = make_template()
    success = False
    started = time.monotonic()
    try:
        for first in range(0, len(tf), chunk_frames):
            end = min(first+chunk_frames, len(tf))
            packed = {source: pack_non_x(positions[source], frames[first:end], states[source], fps, GAMES[GAME]['pitch'])
                      for source, frames in [('tactical', tf), ('broadcast', bf)]}
            if packed_transform is not None:
                packed = packed_transform(packed, first, end)
            for i, index in enumerate(range(first, end)):
                frame = render_frame(template, {s: packed[s][i] for s in packed}, teams,
                                     int(tf[index]), int(bf[index]), int(halves[index]), index, len(tf), fps, display_seconds)
                if frame_overlay is not None:
                    frame = frame_overlay(frame, index)
                if frame.shape[:2] != (output_size[1], output_size[0]):
                    raise ValueError('Video overlay dimensions differ from configured output size.')
                writer.write(frame)
            elapsed = time.monotonic()-started
            print(f'Rendered {end}/{len(tf)} non-X frames ({end/max(elapsed,.001):.1f} frames/s).', flush=True)
        success = True
    finally:
        writer.release()
        if not success:
            temporary.unlink(missing_ok=True)
    cap = cv2.VideoCapture(str(temporary))
    try:
        if (not cap.isOpened() or int(cap.get(cv2.CAP_PROP_FRAME_COUNT)) != len(tf)
                or not math.isclose(cap.get(cv2.CAP_PROP_FPS), fps, abs_tol=.01)):
            raise ValueError('Rendered video frame count/FPS mismatch.')
        for index in sorted(set([0, len(tf)-1, *np.flatnonzero(np.diff(halves) != 0)+1])):
            cap.set(cv2.CAP_PROP_POS_FRAMES, int(index))
            ok, frame = cap.read()
            if not ok or frame.shape[:2] != (output_size[1], output_size[0]):
                raise ValueError('Rendered video sample did not decode.')
        codec = int(cap.get(cv2.CAP_PROP_FOURCC))
    finally:
        cap.release()
    os.replace(temporary, output)
    return {'frames': len(tf), 'fps': fps, 'duration_seconds': len(tf)/fps,
            'size': list(output_size), 'fourcc': ''.join(chr((codec>>(8*i))&255) for i in range(4)),
            'render_seconds': round(time.monotonic()-started, 2)}


HTML = r'''<!doctype html><html lang="ko"><meta charset="utf-8"><meta name="viewport" content="width=device-width,initial-scale=1">
<title>전체 non-X 선수 이동 벡터</title><style>
body{background:#151e2c;color:#eef4fc;font:15px system-ui,sans-serif;margin:20px}h1{font-size:23px}.muted{color:#b8c9dd;font-size:13px;line-height:1.8}
video{display:block;width:100%;max-height:72vh;background:#111}.controls{display:flex;gap:10px;flex-wrap:wrap;align-items:center;margin:14px 0}
button,select,input[type=number]{font:inherit;background:#293c54;color:#fff;border:1px solid #596f8b;border-radius:5px;padding:7px}button{cursor:pointer}
input[type=number]{width:115px}input[type=range]{width:100%}a{color:#8ccaff}#status{min-height:28px;font-variant-numeric:tabular-nums}
</style><h1>전체 프레임 선수 이동 벡터 · X 제외</h1><div class="muted" id="summary"></div>
<video id="video" controls preload="metadata" playsinline><source src="full_movement_vectors.mp4" type="video/mp4"></video>
<div class="controls"><button id="prev">◀ 1프레임</button><button id="play">재생 / 정지</button><button id="next">1프레임 ▶</button>
<button id="half1">전반 시작</button><button id="half2">후반 시작</button>
<label>속도 <select id="speed"><option value=".25">0.25×</option><option value=".5">0.5×</option><option value="1" selected>1×</option><option value="2">2×</option></select></label>
<label>Tactical 프레임 <input id="jump" type="number" step="1"></label><button id="go">이동</button></div>
<input id="seek" type="range" min="0" step="1" value="0" aria-label="영상 프레임 위치"><div class="muted" id="status"></div><div class="muted" id="message"></div>
<p class="muted">파랑 Home · 빨강 Away · 흰색 공 · 화살표는 좌표 차분 속도(m/s) × 표시 초입니다. 공은 위치만 표시합니다.<br>
하프타임과 X 프레임은 재생에서 제외했습니다. 화면의 원래 프레임 번호가 뛰어도, 그 경계를 잇는 벡터는 만들지 않습니다.<br>
←/→ 또는 A/D: 한 프레임 · J/L: 1초 · Space: 재생/정지 · Home/End: 처음/끝<br>
슬롯은 고정 선수 ID가 아니며, 재할당·좌표 오차로 화살표가 급변할 수 있습니다. 방송 매핑은 기존 일정 시차 가정입니다.<br>
HTML과 MP4를 같은 폴더에 보관하세요. 외부 통신 없이 로컬 영상만 재생합니다.<br>
<a href="full_movement_vectors.mp4" download>MP4 저장</a> · <a href="frame_timeline.csv">영상 ↔ 원래 프레임 번호 CSV</a></p>
<script>
'use strict';const meta=__META__,el=id=>document.getElementById(id),video=el('video');let current=0;
el('seek').max=meta.count-1;el('summary').textContent=`${meta.count.toLocaleString()}프레임 · ${meta.fps}fps · ${(meta.count/meta.fps/60).toFixed(2)}분 · 두 영상의 트래킹을 나란히 표시`;
function segment(index){return meta.segments.find(s=>s.video_first<=index&&index<s.video_first+s.count);}
function show(index){current=Math.max(0,Math.min(meta.count-1,Math.trunc(index)));const s=segment(current),step=current-s.video_first;
 el('seek').value=current;el('jump').value=s.tactical_first+step;
 el('status').textContent=`영상 ${current} / ${meta.count-1} · ${s.half} · Tactical ${s.tactical_first+step} · Broadcast ${s.broadcast_first===null?'미매칭':s.broadcast_first+step} · 영상 경과 ${(current/meta.fps).toFixed(2)}초`;}
function seek(index){video.pause();const target=Math.max(0,Math.min(meta.count-1,Math.trunc(index)));video.currentTime=(target+.25)/meta.fps;show(target);el('message').textContent='';}
function toggle(){if(video.paused)video.play().catch(()=>{el('message').textContent='MP4가 같은 폴더에 있는지 확인해 주세요.';});else video.pause();}
el('seek').oninput=()=>seek(Number(el('seek').value));el('prev').onclick=()=>seek(current-1);el('next').onclick=()=>seek(current+1);el('play').onclick=toggle;
for(const [id,half] of [['half1','1st Half'],['half2','2nd Half']])el(id).onclick=()=>{const s=meta.segments.find(s=>s.half===half);if(s)seek(s.video_first);};
el('go').onclick=()=>{const frame=Number(el('jump').value);if(!Number.isInteger(frame))return;
 const s=meta.segments.find(s=>s.tactical_first<=frame&&frame<s.tactical_first+s.count);
 if(s)seek(s.video_first+frame-s.tactical_first);else el('message').textContent='해당 프레임은 X로 제외됐거나 범위 밖입니다.';};
el('speed').onchange=()=>{video.playbackRate=Number(el('speed').value);};
video.addEventListener('timeupdate',()=>{if(!video.requestVideoFrameCallback)show(Math.floor(video.currentTime*meta.fps+1e-5));});
video.addEventListener('seeked',()=>show(Math.floor(video.currentTime*meta.fps+1e-5)));
video.addEventListener('ended',()=>show(meta.count-1));video.addEventListener('error',()=>{el('message').textContent='로컬 MP4를 읽을 수 없습니다. 영상과 HTML을 같은 폴더에 보관해 주세요.';});
if(video.requestVideoFrameCallback){const update=(now,info)=>{show(Math.round(info.mediaTime*meta.fps));video.requestVideoFrameCallback(update);};video.requestVideoFrameCallback(update);}
document.addEventListener('keydown',e=>{if(e.target.tagName==='SELECT'||(e.target.tagName==='INPUT'&&e.target.type!=='range'))return;if(e.target.tagName==='BUTTON'&&e.code==='Space')return;
 const steps={ArrowLeft:-1,ArrowRight:1,a:-1,d:1,j:-meta.fps,l:meta.fps};if(e.key in steps){e.preventDefault();seek(current+steps[e.key]);}else if(e.code==='Space'){e.preventDefault();toggle();}else if(e.key==='Home'){e.preventDefault();seek(0);}else if(e.key==='End'){e.preventDefault();seek(meta.count-1);}});
show(0);
</script></html>'''


def write_player(path, tf, bf, halves, fps):
    meta = {'count': len(tf), 'fps': fps, 'segments': timeline_segments(tf, bf, halves)}
    path.write_text(HTML.replace('__META__', json.dumps(meta, separators=(',', ':'))), encoding='utf-8')
    return meta


def main():
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument('--data-dir')
    parser.add_argument('--display-seconds', type=float, default=1.)
    args = parser.parse_args()
    try:
        if not math.isfinite(args.display_seconds) or args.display_seconds <= 0:
            raise ValueError('Arrow display seconds must be positive.')
        states = {s: source_state(ROOT/f'data/derived/sync/final_tracking_{s}_labels.csv', source_frames(s))
                  for s in ('tactical', 'broadcast')}
        mapping = read_mapping(ROOT/'data/derived/sync/final_tracking_alignment.csv', source_frames('tactical'), source_frames('broadcast'))
        tf, bf, halves = select_timeline(mapping, states)
        print(json.dumps({'selected_non_X_frames': len(tf), 'duration_seconds': len(tf)/GAMES[GAME]['fps']}), flush=True)
        positions, teams = {}, {}
        for source in ('tactical', 'broadcast'):
            paths = asset_paths(source=source, data_dir=args.data_dir)
            teams[source] = track_teams(paths['atd_meta'])
            positions[source] = read_tracking(paths['atd_raw'], teams[source], source_frames(source), 0, source_frames(source))
        FOLDER.mkdir(parents=True, exist_ok=True)
        fps = GAMES[GAME]['fps']
        movie = export_movie(positions, teams, states, tf, bf, halves, FOLDER/'full_movement_vectors.mp4', fps, args.display_seconds)
        meta = write_player(FOLDER/'full_movement_vector_player.html', tf, bf, halves, fps)
        with (FOLDER/'frame_timeline.csv').open('w', newline='', encoding='utf-8') as handle:
            writer = csv.writer(handle)
            writer.writerow(['video_frame', 'video_time_seconds', 'tactical_frame', 'broadcast_atd_frame0', 'half_tag'])
            writer.writerows((i, f'{i/fps:.6f}', int(t), int(b) if b >= 0 else '', '1st Half' if half == 1 else '2nd Half')
                             for i, (t, b, half) in enumerate(zip(tf, bf, halves)))
        report = {**movie, 'segments': meta['segments'], 'X_frames_excluded': {s: int((~states[s]['active']).sum()) for s in states},
                  'all_non_X_broadcast_covered': True, 'event_catalog_used': False, 'possession_gt_used': False,
                  'vector_definition': 'adjacent original source coordinate difference * pitch * fps, both non-X and same half',
                  'vector_units': 'm/s', 'arrow_display_seconds': args.display_seconds, 'gap_policy': 'No interpolation or derivatives across removed intervals.',
                  'source_slots_not_verified_player_IDs': True, 'full_half_synchronization_verified': False}
        (FOLDER/'report.json').write_text(json.dumps(report, indent=2)+'\n', encoding='utf-8')
    except (OSError, ValueError, KeyError, IndexError, TypeError, ET.ParseError, csv.Error, cv2.error):
        print('Full non-X movement export failed; no private tracking rows logged.', file=sys.stderr)
        return 1
    print(json.dumps({'video': movie, 'segments': meta['segments']}), flush=True)
    print('Saved one complete non-X H.264 MP4, local frame player and source frame timeline.', flush=True)
    return 0


if __name__ == '__main__':
    raise SystemExit(main())
