"""Render exactly one full-match video from the saved unified tracking ATD.

Broadcast context on the left; Tactical players and retained V3 ball together
on the right. No coordinate edits, repeated last points or new predictions.
"""
from pathlib import Path
import argparse
import json
import subprocess
import sys

import cv2
import numpy as np
import pandas as pd
from PIL import Image, ImageDraw, ImageFont

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from starter.render_broadcast_atd_from_csv import clean, digest, font, project

BASE = ROOT / 'outputs/unified_tracking_v1'
ASSETS = BASE / 'render_assets'
SOURCE = ROOT / 'data/broadcast/BAR-ATH_broadcast-001.mp4'
HFILE = ROOT / 'data/broadcast/BAR-ATH_broadcast_homography.csv'
TARGET = BASE / 'unified_tracking.mp4'
NATIVE = ASSETS / 'render_native'
N = 164487
FPS = 25
W, H = 1920, 900
SCALE = 920 / 165
ORIGIN = np.array([980 + 30 * SCALE, 110 + 26 * SCALE])
BOUNDS = np.array([[-30., -26.], [135., 94.]])
COLORS = {0: (220, 220, 220), 1: (250, 250, 250), 4: (255, 219, 82)}
SOURCE_LABELS = {0: '공: 현재 좌표 결측', 1: '공: 현재 유지한 V3 좌표', 4: '공: V3에 이미 포함된 추정 좌표'}
STATE_LABELS = {0: 'Tactical 선수 시간 대응 없음 · 기존 공 좌표만 유지',
    1: 'Tactical 선수 + 현재 공 좌표를 같은 시간축에 표시',
    2: 'Tactical 선수 좌표만 있음 · 공 결측은 그대로 표시',
    3: '선수 좌표 결측 · 현재 공 기록만 표시', 4: '현재 프레임은 선수와 공 좌표 모두 결측',
    5: '하이라이트: 중계 장면과 경기장 좌표 시점이 다를 수 있음'}
PROJECTION_LABELS = {0: '중계 화면의 공 원은 기존 H로 지면 재투영 · 직접 검출이나 공 높이 표시가 아님',
    1: '공 좌표는 있음 · H/화면 위치가 유효하지 않아 중계 화면의 공 원만 숨김',
    2: '하이라이트 중계 장면에는 공 원을 겹치지 않음 · 오른쪽에 저장 좌표 표시',
    3: '현재 공 좌표 결측 · 선수는 같은 프레임 Tactical 기록 좌표로 표시'}


def save(name, value):
    (BASE / name).write_text(json.dumps(clean(value), ensure_ascii=False, indent=2,
        allow_nan=False) + '\n', encoding='utf-8')


def pitch(xy):
    return ORIGIN + np.asarray(xy) * SCALE


def assets():
    im = Image.new('RGBA', (W, H), (21, 24, 30, 255))
    d = ImageDraw.Draw(im)
    d.text((18, 10), 'BAR–ATH 통합 트래킹 V1 · 중계영상', font=font(27), fill=(241, 241, 241))
    d.text((980, 10), 'Tactical 선수 + 현재 V3 공 좌표', font=font(27), fill=(241, 241, 241))
    for x, label, color in [(18, '파랑 Barcelona', (56, 143, 252)), (240, '빨강 Athletic Club', (255, 87, 82)),
                            (506, '흰 공 좌표', COLORS[1]), (690, '노랑 기존 추정', COLORS[4])]:
        d.text((x, 50), label, font=font(20), fill=color)
    d.text((18, 78), '전체 164,487프레임 · 25FPS · 중계 원본 시간/음성 유지', font=font(16), fill=(208, 214, 222))
    d.text((980, 48), '105 × 68m · 선수 46개 트랙 슬롯 · 현재 프레임 위치', font=font(19), fill=(225, 232, 227))
    d.text((980, 71), '프레임', font=font(17), fill=(238, 238, 238))
    d.text((1188, 71), '시간', font=font(17), fill=(238, 238, 238))
    d.rectangle((980, 110, 1900, 110 + 120 * SCALE), fill=(25, 53, 38, 255))
    def p(x, y): return tuple(pitch([x, y]))
    d.rectangle((*p(0, 0), *p(105, 68)), fill=(30, 87, 49), outline=(214, 224, 217), width=2)
    d.line([p(52.5, 0), p(52.5, 68)], fill=(214, 224, 217), width=2)
    for a, b in [(0, 16.5), (105, 88.5)]:
        d.line([p(a, 13.84), p(b, 13.84), p(b, 54.16), p(a, 54.16)], fill=(214, 224, 217), width=2)
    for a, b in [(0, 5.5), (105, 99.5)]:
        d.line([p(a, 24.84), p(b, 24.84), p(b, 43.16), p(a, 43.16)], fill=(214, 224, 217), width=2)
    c = pitch([52.5, 34]); r = 9.15 * SCALE
    d.ellipse((c[0] - r, c[1] - r, c[0] + r, c[1] + r), outline=(214, 224, 217), width=2)
    for x in [11, 52.5, 94]:
        a, b = p(x, 34); d.ellipse((a - 2, b - 2, a + 2, b + 2), fill=(214, 224, 217))
    d.text((992, 122), '경기장 밖 좌표도 보존 · 원본 XY 방향 유지', font=font(17), fill=(225, 232, 227))
    d.text((18, 746), '선수 번호 = 트래킹 슬롯 · 반투명 = 원본 속도 s 결측', font=font(20), fill=(220, 227, 233))
    d.text((18, 781), '공·선수 모두 저장 좌표 그대로 표시 · 추가 평활화 없음', font=font(20), fill=(220, 227, 233))
    d.text((992, 801), '● 필드 선수 슬롯   ■ 골키퍼 슬롯   흰/노랑 큰 점 = 공', font=font(19), fill=(225, 232, 227))
    d.text((980, 844), '공 X(m)', font=font(20), fill=(238, 238, 238))
    d.text((1470, 844), '공 Y(m)', font=font(20), fill=(238, 238, 238))
    d.text((992, 879), '시각화는 통합 tracking_atd.txt의 좌표와 동일', font=font(15), fill=(207, 217, 223))
    im.paste((0, 0, 0, 0), (0, 100, 960, 640))
    im.save(ASSETS / 'background.png')
    for prefix, labels, size, height in [('source', SOURCE_LABELS, 22, 34),
        ('state', STATE_LABELS, 20, 30), ('projection', PROJECTION_LABELS, 17, 28)]:
        for code, label in labels.items():
            tile = Image.new('RGBA', (930, height), (21, 24, 30, 255))
            color = COLORS[code] if prefix == 'source' else (222, 228, 233)
            ImageDraw.Draw(tile).text((0, 1), label, font=font(size), fill=color)
            tile.save(ASSETS / f'{prefix}_{code}.png')
    mono = ImageFont.truetype('/System/Library/Fonts/Menlo.ttc', 20)
    for char in '0123456789:+-. ':
        tile = Image.new('RGBA', (17, 28), (21, 24, 30, 255))
        ImageDraw.Draw(tile).text((1, 0), char, font=mono, fill=(242, 242, 242))
        tile.save(ASSETS / f'glyph_{ord(char)}.png')
    people = json.loads((BASE / 'roster.json').read_text())['players']
    for person in people:
        tile = Image.new('RGBA', (34, 18), (0, 0, 0, 0))
        ImageDraw.Draw(tile).text((1, 0), person['display_label'], font=font(11),
            fill=(250, 250, 250), stroke_width=1, stroke_fill=(8, 20, 18))
        tile.save(ASSETS / f"slot_{person['slot']:02d}.png")
    (ASSETS / 'player_styles.json').write_text(json.dumps(people, separators=(',', ':')), encoding='utf-8')


def expected():
    values = np.load(ASSETS / 'tracking_values.npy', mmap_mode='r')
    assert values.shape == (N, 47, 3)
    table = pd.read_csv(BASE / 'tracking_all_frames.csv', float_precision='round_trip',
        usecols=['frame', 'alignment_half', 'tactical_frame', 'players_time_aligned', 'player_position_count',
            'ball_position_available', 'ball_is_prediction', 'ball_world_live_reference', 'broadcast_highlight_inferred',
            'ball_x_norm', 'ball_y_norm', 'ball_x_m', 'ball_y_m'])
    assert np.array_equal(values[:, 46, :2], table[['ball_x_norm', 'ball_y_norm']], equal_nan=True)
    norm = values[:, 46, :2]
    world = norm * [105., 68.]
    valid_ball = np.isfinite(world).all(axis=1)
    valid_player = np.isfinite(values[:, :46, :2]).all(axis=2)
    htable = pd.read_csv(HFILE, float_precision='round_trip')
    assert len(htable) == N + 1 and np.array_equal(htable.frame, np.arange(N + 1))
    h = htable[[f'h{i}{j}' for i in range(3) for j in range(3)]].to_numpy().reshape(-1, 3, 3)[:N]
    hg = htable.valid.iloc[:N].eq(1).to_numpy() & htable.reprojection_error.iloc[:N].le(5).to_numpy()
    hg &= np.isfinite(h).all(axis=(1, 2))
    image, depth = project(h, norm)
    center = np.einsum('nij,j->ni', h, [.5, .5, 1.])[:, 2]
    visible = valid_ball & hg & np.isfinite(image).all(axis=1) & (depth * center > 0)
    visible &= ((image >= [0, 0]) & (image < [1920, 1080])).all(axis=1)
    highlight = table.broadcast_highlight_inferred.to_numpy(bool)
    visible &= ~highlight
    image = image * .5 + [0, 100]
    image[~visible] = np.nan
    ground = pitch(world); ground[~valid_ball] = np.nan
    assert (((world >= BOUNDS[0]) & (world <= BOUNDS[1])).all(axis=1)[valid_ball]).all()
    pp = values[:, :46, :2] * [105., 68.]
    assert (((pp >= BOUNDS[0]) & (pp <= BOUNDS[1])).all(axis=2)[valid_player]).all()
    players = np.empty((N, 46, 3), dtype='<f4')
    players[:, :, :2] = pitch(pp)
    players[:, :, :2][~valid_player] = np.nan
    players[:, :, 2] = np.isfinite(values[:, :46, 2])
    source = np.where(~valid_ball, 0, np.where(table.ball_is_prediction, 4, 1))
    state = np.ones(N, int)
    state[~valid_ball & valid_player.any(axis=1)] = 2
    state[valid_ball & ~valid_player.any(axis=1)] = 3
    state[~valid_ball & ~valid_player.any(axis=1)] = 4
    state[highlight & table.players_time_aligned.to_numpy(bool)] = 5
    state[~table.players_time_aligned.to_numpy(bool)] = 0
    projection = np.where(~valid_ball, 3, np.where(highlight, 2, np.where(~visible, 1, 0)))
    rows = np.c_[np.arange(N), source, state, np.full((N, 4), np.nan), image, ground, world, projection]
    return table, rows, players


def prepare():
    assets()
    table, rows, players = expected()
    wire = [[int(v) if j in [0, 1, 2, 13] else (float(v) if np.isfinite(v) else None)
        for j, v in enumerate(row)] for row in rows]
    (ASSETS / 'ball_rows.json').write_text(json.dumps(wire, separators=(',', ':'), allow_nan=False))
    players.tofile(ASSETS / 'player_positions.f32')
    table['source_code'] = rows[:, 1].astype(int)
    table['state_code'] = rows[:, 2].astype(int)
    table['projection_code'] = rows[:, 13].astype(int)
    table.to_csv(ASSETS / 'overlay_frame_metadata.csv', index=False)
    save('video_manifest.json', dict(status='ASSETS_READY', video=TARGET.name, frames=N, fps=FPS,
        size=[W, H], duration_seconds=N / FPS, only_one_video=True, all_source_frames_in_order=True,
        source_stat=dict(bytes=SOURCE.stat().st_size, mtime_ns=SOURCE.stat().st_mtime_ns),
        input_hashes={str(p.relative_to(ROOT)): digest(p) for p in [BASE / 'tracking_atd.txt',
            BASE / 'tracking_atd.xml', BASE / 'tracking_all_frames.csv', BASE / 'roster.json',
            ASSETS / 'tracking_values.npy', ASSETS / 'ball_rows.json', ASSETS / 'player_positions.f32', HFILE, NATIVE]},
        plot_bounds_m=BOUNDS, pitch_origin_canvas=ORIGIN, pitch_pixels_per_m=SCALE,
        source_colors=SOURCE_LABELS, state_labels=STATE_LABELS, projection_labels=PROJECTION_LABELS,
        player_marker_colors=dict(Barcelona=[56, 143, 252], Athletic_Club=[255, 87, 82]),
        highlighted_broadcast_ball_markers_hidden=True, players_on_pitch_only=True,
        no_coordinate_or_source_data_changes=True))
    verify_metadata()


def verify_metadata():
    table, rows, expected_players = expected()
    actual = np.array(json.loads((ASSETS / 'ball_rows.json').read_text()), dtype=float)
    players = np.memmap(ASSETS / 'player_positions.f32', dtype='<f4', mode='r', shape=(N, 46, 3))
    assert actual.shape == (N, 14) and np.array_equal(actual, rows, equal_nan=True)
    assert np.array_equal(players, expected_players, equal_nan=True)
    assert np.isnan(actual[~table.ball_position_available.to_numpy(bool), 7:11]).all()
    assert np.isnan(players[~table.players_time_aligned.to_numpy(bool), :, :2]).all()
    assert np.isnan(actual[table.broadcast_highlight_inferred.to_numpy(bool), 7:9]).all()
    save('overlay_verification.json', dict(passed=True, frames=N, player_slots=46,
        all_player_positions_exactly_match_saved_tracking_at_float32_display_precision=True,
        all_ball_positions_exactly_match_saved_tracking=True, missing_positions_not_held=True,
        unmatched_tactical_frames_not_invented=True, highlight_image_ball_not_drawn=True))


def render():
    if TARGET.exists(): raise FileExistsError(TARGET)
    cmd = [str(NATIVE), str(SOURCE), str(ASSETS / 'ball_rows.json'), str(ASSETS / 'background.png'),
        str(ASSETS / 'player_positions.f32'), str(TARGET)]
    with (ASSETS / 'render.log').open('w') as log:
        subprocess.run(cmd, stdout=log, stderr=subprocess.STDOUT, check=True, cwd=ROOT)


def color_count(patch, team):
    b, g, r = [patch[:, :, j].astype(float) for j in range(3)]
    if team == 'H': selected = (b > r + 35) & (b > g + 20) & (b > 110)
    elif team == 'A': selected = (r > b + 35) & (r > g + 25) & (r > 130) & (g < 180)
    elif team == 'ball_white': selected = (r > 215) & (g > 215) & (b > 215)
    else: selected = (r > 175) & (g > 150) & (b < 150)
    return int(selected.sum())


def verify_video():
    verify_metadata()
    rows = np.array(json.loads((ASSETS / 'ball_rows.json').read_text()), dtype=float)
    players = np.memmap(ASSETS / 'player_positions.f32', dtype='<f4', mode='r', shape=(N, 46, 3))
    people = json.loads((BASE / 'roster.json').read_text())['players']
    video = cv2.VideoCapture(str(TARGET)); source = cv2.VideoCapture(str(SOURCE))
    assert video.isOpened() and source.isOpened()
    assert int(video.get(7)) == N and video.get(5) == FPS
    assert video.get(3) == W and video.get(4) == H
    indices = set(np.linspace(0, N - 1, 40).astype(int).tolist())
    for col, codes in [(1, SOURCE_LABELS), (2, STATE_LABELS), (13, PROJECTION_LABELS)]:
        for code in codes:
            found = np.flatnonzero(rows[:, col] == code)
            if len(found): indices.update(found[np.linspace(0, len(found) - 1, min(4, len(found))).astype(int)].tolist())
    for boundary in [7171, 86638, 86823, 163279]:
        indices.update(range(boundary - 1, boundary + 2))
    glyphs = {c: cv2.imread(str(ASSETS / f'glyph_{ord(c)}.png')) for c in '0123456789'}
    sprites = {prefix: {c: cv2.imread(str(ASSETS / f'{prefix}_{c}.png')) for c in labels}
        for prefix, labels in [('source', SOURCE_LABELS), ('state', STATE_LABELS), ('projection', PROJECTION_LABELS)]}
    samples = []
    verified_players = dict(H=0, A=0)
    verified_balls = 0
    for f in sorted(indices):
        video.set(1, f); ok, frame = video.read(); assert ok and frame.shape == (H, W, 3)
        source.set(1, f); ok, raw = source.read(); assert ok
        small_a = cv2.resize(frame[100:640, :960], (64, 36), interpolation=cv2.INTER_AREA)
        small_b = cv2.resize(raw, (64, 36), interpolation=cv2.INTER_AREA)
        mae = float(np.abs(small_a.astype(float) - small_b.astype(float)).mean())
        assert mae < 12, (f, 'source frame', mae)
        digits = ''
        for j in range(6):
            patch = frame[68:96, 1050 + j * 17:1050 + (j + 1) * 17]
            digits += min(glyphs, key=lambda c: np.abs(patch.astype(float) - glyphs[c].astype(float)).mean())
        assert digits == f'{f:06d}', (f, digits)
        code, state, projection = map(int, rows[f, [1, 2, 13]])
        for prefix, value, y, height in [('source', code, 655, 34), ('state', state, 697, 30), ('projection', projection, 846, 28)]:
            error = np.abs(frame[y:y + height, 18:948].astype(float) - sprites[prefix][value].astype(float)).mean()
            assert error < 10, (f, prefix, error)
        visible = np.isfinite(players[f, :, :2]).all(axis=1)
        tested = {'H': 0, 'A': 0}
        for slot in np.flatnonzero(visible):
            team = people[slot]['team_code']
            if tested[team] >= 3: continue
            point = players[f, slot, :2]
            others = players[f, visible, :2]
            distances = np.linalg.norm(others - point, axis=1)
            if np.sum(distances < 15) > 1: continue
            if np.isfinite(rows[f, 9]) and np.linalg.norm(rows[f, 9:11] - point) < 19: continue
            x, y = np.rint(point).astype(int)
            n = color_count(frame[y - 7:y + 8, x - 7:x + 8], team)
            assert n >= 10, (f, slot, team, n)
            tested[team] += 1; verified_players[team] += 1
        if not visible.any():
            # The field lines are neutral; no stale team markers should survive.
            field = frame[130:775, 990:1890].copy()
            if np.isfinite(rows[f, 9]):
                x, y = np.rint(rows[f, 9:11]).astype(int) - [990, 130]
                field[max(0, y - 14):min(field.shape[0], y + 15),
                      max(0, x - 14):min(field.shape[1], x + 15)] = [49, 87, 30]
            assert color_count(field, 'H') == 0 and color_count(field, 'A') == 0, (f, 'stale player marker')
        if np.isfinite(rows[f, 9]):
            x, y = np.rint(rows[f, 9:11]).astype(int)
            n = color_count(frame[y - 8:y + 9, x - 8:x + 9], 'ball_white' if code == 1 else 'ball_yellow')
            assert n >= 25, (f, 'ball marker', n)
            verified_balls += 1
        if projection == 2: assert np.isnan(rows[f, 7:9]).all()
        samples.append(dict(frame=f, source_frame_mae=mae, frame_glyph=digits,
            player_markers_checked=tested, ball_source_code=code, state_code=state, projection_code=projection))
    video.release(); source.release()
    assert verified_players['H'] > 30 and verified_players['A'] > 30 and verified_balls > 10
    progress = json.loads(TARGET.with_suffix('.mp4.progress.json').read_text())
    assert progress['status'] == 'complete' and progress['frames'] == N and progress['audio_preserved']
    assert progress['source_segments'] == 1 and abs(progress['output_duration_seconds'] - N / FPS) < .1
    manifest = json.loads((BASE / 'video_manifest.json').read_text())
    assert all(digest(ROOT / p) == h for p, h in manifest['input_hashes'].items())
    assert dict(bytes=SOURCE.stat().st_size, mtime_ns=SOURCE.stat().st_mtime_ns) == manifest['source_stat']
    save('video_verification.json', dict(passed=True, frames=N, fps=FPS, size=[W, H], duration_seconds=N / FPS,
        audio_preserved=True, continuous_source_timeline=True, sampled_frames=len(samples), samples=samples,
        player_markers_checked=verified_players, ball_markers_checked=verified_balls,
        video_sha256=digest(TARGET), full_frame_data_and_overlay_metadata_verified=True))
    finish()


def finish():
    for filename in ['validation.json', 'overlay_verification.json', 'video_verification.json']:
        assert json.loads((BASE / filename).read_text())['passed']
    manifest = json.loads((BASE / 'tracking_manifest.json').read_text())
    assert all(digest(ROOT / p) == h for p, h in manifest['input_hashes'].items())
    manifest.update(status='COMPLETE', all_data_and_video_checks_passed=True)
    save('tracking_manifest.json', manifest)
    vm = json.loads((BASE / 'video_manifest.json').read_text())
    vm.update(status='COMPLETE', verified=True)
    save('video_manifest.json', vm)
    report = json.loads((BASE / 'video_verification.json').read_text())
    signatures = {TARGET.name: report['video_sha256']}
    signatures.update({p.name: digest(p) for p in BASE.iterdir()
        if p.is_file() and p.name not in signatures and p.name != 'artifact_integrity.json'})
    save('artifact_integrity.json', dict(algorithm='sha256', files=signatures,
        source_files={str(p.relative_to(ROOT)): digest(p) for p in [Path(__file__),
            ROOT / 'starter/build_unified_tracking_v1.py', ROOT / 'starter/render_unified_tracking_native_v1.m']}))


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('stage', choices=['prepare', 'render', 'verify'])
    args = parser.parse_args()
    if args.stage == 'prepare': prepare()
    elif args.stage == 'render': render()
    else: verify_video()
