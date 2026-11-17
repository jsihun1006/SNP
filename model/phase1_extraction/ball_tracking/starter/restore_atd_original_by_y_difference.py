"""New ATD export: restore original ball XY when signed delta-Y/68 <= -0.1.

Existing ATD and homography matrices are immutable. Render one full-match
before/after comparison using the same original H for both coordinate tracks.
"""
from pathlib import Path
import argparse
import itertools
import json
import shutil
import subprocess
import sys

import cv2
import numpy as np
import pandas as pd
from PIL import Image, ImageDraw, ImageFont

from render_broadcast_atd_from_csv import (
    ROOT, SOURCE, HFILE, PRIMARY, NATIVE, N, FPS, W, H, SCALE, ORIGIN, BOUNDS,
    ball_text, clean, digest, font, pitch, project,
)

BEFORE_BASE = ROOT / 'outputs/broadcast_atd_from_csv_v1'
ORIGINAL = ROOT / 'data/broadcast/BAR-ATH_broadcast_atd.txt'
BEFORE = BEFORE_BASE / ORIGINAL.name
OUT = ROOT / 'outputs/broadcast_atd_y_minus_0p1_original_v1'
AFTER = OUT / ORIGINAL.name
VIDEO = OUT / 'comparison_full_match.mp4'
THRESHOLD = -.1
COLORS = {0: (220, 220, 220), 1: (76, 230, 99), 3: (64, 209, 250)}


def save(name, value):
    (OUT / name).write_text(json.dumps(clean(value), ensure_ascii=False, indent=2,
                                    allow_nan=False) + '\n', encoding='utf-8')


def read(path):
    return pd.read_csv(path, float_precision='round_trip', low_memory=False)


def metrics(delta, comparable):
    v = delta[comparable]
    return dict(frames=len(v), y_rmse_m=np.sqrt(np.mean(v[:, 1] ** 2)),
                xy_rms_difference_m=np.sqrt(np.mean(np.sum(v ** 2, axis=1))),
                y_mean_absolute_difference_m=np.mean(np.abs(v[:, 1])),
                y_absolute_p95_m=np.quantile(np.abs(v[:, 1]), .95))


def export():
    OUT.mkdir(parents=True, exist_ok=True)
    if AFTER.exists():
        raise FileExistsError(f'Refusing to overwrite {AFTER}')
    paths = [ORIGINAL, BEFORE, HFILE, PRIMARY, ORIGINAL.with_suffix('.xml'),
             BEFORE_BASE / 'corrected_ball_coordinates.csv',
             BEFORE_BASE / 'coordinate_difference_analysis/y_only_normalized_view/y_difference_all_frames.csv']
    signatures = {str(p.relative_to(ROOT)): digest(p) for p in paths}
    old = ball_text(ORIGINAL)
    before = ball_text(BEFORE)
    assert old.shape == before.shape == (N + 1, 3)
    comparable = np.isfinite(old[:, :2]).all(axis=1) & np.isfinite(before[:, :2]).all(axis=1)
    delta = (before[:, :2] - old[:, :2]) * [105., 68.]
    # Match exactly the previously visualized definition (subtract meters first).
    delta[:, 1] = before[:, 1] * 68. - old[:, 1] * 68.
    norm_y = delta[:, 1] / 68.
    norm_y[~comparable] = np.nan
    selected = comparable & (norm_y <= THRESHOLD)
    assert not selected[-1] and not np.any(selected & ~np.isfinite(old[:, :2]).all(axis=1))
    previous_y = read(paths[-1])
    assert np.array_equal(norm_y, previous_y.signed_delta_y_over_68.to_numpy(), equal_nan=True)
    assert np.array_equal(selected, comparable & ((before[:, 1] - old[:, 1]) <= THRESHOLD))
    records = dict(rows=0, player_and_frame_prefixes_preserved=True,
                   speed_tokens_preserved=True, line_endings_preserved=True,
                   unselected_rows_byte_identical=True, selected_xy_tokens_exact_original=True)
    with ORIGINAL.open('rb') as original_stream, BEFORE.open('rb') as before_stream, AFTER.open('wb') as output:
        for f, pair in enumerate(itertools.zip_longest(original_stream, before_stream)):
            original_line, before_line = pair
            assert original_line is not None and before_line is not None
            old_body = original_line.rstrip(b'\r\n')
            before_body = before_line.rstrip(b'\r\n')
            ending = before_line[len(before_body):]
            prefix, tokens = before_body.rsplit(b':', 1)
            before_tokens = tokens.split(b',')
            old_tokens = old_body.rsplit(b':', 1)[1].split(b',')
            assert int(prefix.split(b':', 1)[0]) == f + 1
            new_tokens = before_tokens.copy()
            if selected[f]:
                new_tokens[:2] = old_tokens[:2]
                final_line = prefix + b':' + b','.join(new_tokens) + ending
            else:
                final_line = before_line
            output.write(final_line)
            final_body = final_line.rstrip(b'\r\n')
            final_prefix, final_tokens = final_body.rsplit(b':', 1)
            final_tokens = final_tokens.split(b',')
            records['player_and_frame_prefixes_preserved'] &= final_prefix == prefix
            records['speed_tokens_preserved'] &= final_tokens[2] == before_tokens[2]
            records['line_endings_preserved'] &= final_line[len(final_body):] == ending
            records['unselected_rows_byte_identical'] &= selected[f] or final_line == before_line
            records['selected_xy_tokens_exact_original'] &= not selected[f] or final_tokens[:2] == old_tokens[:2]
            records['rows'] += 1
    assert records['rows'] == N + 1 and all(v for k, v in records.items() if k != 'rows')
    after = ball_text(AFTER)
    expected = before.copy()
    expected[selected, :2] = old[selected, :2]
    assert np.array_equal(after, expected, equal_nan=True)
    assert np.array_equal(after[:, 2], before[:, 2], equal_nan=True)
    assert np.array_equal(np.isfinite(after[:, :2]).all(axis=1), np.isfinite(before[:, :2]).all(axis=1))
    shutil.copyfile(ORIGINAL.with_suffix('.xml'), AFTER.with_suffix('.xml'))
    audit = read(BEFORE_BASE / 'corrected_ball_coordinates.csv')
    result = pd.DataFrame(dict(frame=np.arange(N + 1), atd_frame_id=np.arange(1, N + 2),
        time_sec=np.arange(N + 1) / FPS, original_x=old[:, 0], original_y=old[:, 1],
        before_x=before[:, 0], before_y=before[:, 1], final_x=after[:, 0], final_y=after[:, 1],
        original_x_m=old[:, 0] * 105., original_y_m=old[:, 1] * 68.,
        before_x_m=before[:, 0] * 105., before_y_m=before[:, 1] * 68.,
        final_x_m=after[:, 0] * 105., final_y_m=after[:, 1] * 68.,
        before_delta_y_m=delta[:, 1], before_signed_delta_y_over_68=norm_y,
        final_delta_y_m=after[:, 1] * 68. - old[:, 1] * 68.,
        final_signed_delta_y_over_68=(after[:, 1] * 68. - old[:, 1] * 68.) / 68.,
        comparable=comparable, original_xy_valid=np.isfinite(old[:, :2]).all(axis=1),
        final_xy_valid=np.isfinite(after[:, :2]).all(axis=1), original_xy_restored=selected,
        before_source_kind=audit.source_kind, before_source_is_prediction=audit.source_is_prediction,
        final_source=np.where(selected, 'ORIGINAL_ATD_Y_THRESHOLD', audit.source_kind),
        final_source_is_prediction=np.where(selected, False, audit.source_is_prediction),
        replacement_reason=np.where(selected, 'SIGNED_DELTA_Y_OVER_68_LE_MINUS_0_1_RESTORE_ORIGINAL_XY',
            np.where(~comparable, 'NO_COMPARABLE_ORIGINAL_KEEP_PREVIOUS', 'THRESHOLD_NOT_MET_KEEP_PREVIOUS'))))
    result.to_csv(OUT / 'corrected_ball_coordinates.csv', index=False)
    result.loc[selected].to_csv(OUT / 'replacement_log.csv', index=False)
    starts = np.flatnonzero(selected & ~np.r_[False, selected[:-1]])
    ends = np.flatnonzero(selected & ~np.r_[selected[1:], False])
    pd.DataFrame([dict(first_frame=int(a), last_frame=int(b), frames=int(b - a + 1),
                       start_video_time_sec=a / FPS, end_video_time_sec=(b + 1) / FPS,
                       duration_sec=(b - a + 1) / FPS,
                       minimum_delta_y_normalized=float(np.min(norm_y[a:b + 1])))
                  for a, b in zip(starts, ends)]).to_csv(OUT / 'replacement_segments.csv', index=False)
    final_delta = (after[:, :2] - old[:, :2]) * [105., 68.]
    summary = dict(status='COORDINATES_COMPLETE', threshold=THRESHOLD,
        threshold_equation='(previous_corrected_y_m-original_ATD_y_m)/68 <= -0.1',
        threshold_m=-6.8, exact_equality_frames=int(np.sum(comparable & (norm_y == THRESHOLD))),
        interpretation='The requested -0.1 is treated as a signed threshold, <= -0.1, not exact float equality.',
        replacement='Restore BOTH ball X and Y from original ATD for selected frames; do not swap H matrices.',
        reason_for_coordinate_interpretation='Previous corrections changed ball XY only; both tracks already use the unchanged original broadcast homography.',
        replacement_frames=int(selected.sum()), contiguous_replacement_segments=len(starts),
        selected_before_source_counts=audit.loc[selected, 'source_kind'].value_counts().to_dict(),
        total_ATD_rows=N + 1, video_frames=N, fps=FPS, valid_final_ball_frames=int(result.final_xy_valid.sum()),
        original_missing_filled_frames_retained=int((~np.isfinite(old[:, :2]).all(axis=1) & result.final_xy_valid).sum()),
        delta_Y_normalization='Signed difference divided by fixed 68m pitch width; not minmax amplitude.',
        after_selected_xy_equals_original=True, unselected_coordinates_equal_previous=True,
        before_against_original=metrics(delta, comparable),
        after_against_original=metrics(final_delta, comparable),
        reference_agreement_reduction_is_not_actual_accuracy_improvement=True,
        original_ATD_modified=False, previous_corrected_ATD_modified=False, homography_matrices_modified=False,
        player_coordinates_and_speed_preserved=True, extra_final_ATD_row_preserved=True,
        no_new_detection_prediction_or_height_estimation=True, input_hashes=signatures,
        output_ATD=AFTER.name, all_frames_csv='corrected_ball_coordinates.csv',
        comparison_video=VIDEO.name)
    save('correction_summary.json', summary)
    assert all(digest(ROOT / p) == signature for p, signature in signatures.items())
    save('coordinate_verification.json', dict(passed=True, byte_contract=records,
        selection_matches_previously_visualized_Y_difference=True,
        selected_fully_restored_to_original_xy=True, unselected_rows_and_missing_states_preserved=True,
        exact_equality_count=int(np.sum(comparable & (norm_y == THRESHOLD))),
        selected_count=int(selected.sum()), preserved_input_hashes=True,
        output_ATD_sha256=digest(AFTER), metadata_XML_identical=digest(AFTER.with_suffix('.xml')) == digest(ORIGINAL.with_suffix('.xml'))))
    (OUT / 'README.md').write_text(f'''# 정규화 Y차이 -0.1 이하: 원본 ATD 공 좌표 복원

기존 보정 ATD에서 `(보정 Y − 원본 Y)/68 ≤ -0.1`, 즉 `ΔY ≤ -6.8m`인 **{int(selected.sum()):,}프레임**의 공 **XY 쌍**을 원본 Broadcast ATD로 복원했습니다.
정확히 -0.1인 프레임은 0개이므로 요청의 -0.1을 음수 임계값으로 해석했습니다. 이번 정규화는 경기장 폭 68m 기준이며 절대 차이 minmax 0–1 점수가 아닙니다.

기존 보정에서는 H 행렬 자체를 바꾸지 않았습니다. 원본/기존 보정 공 좌표가 같은 원본 Broadcast H를 사용하므로,
이번에는 사용자가 말한 원본 homography 대체를 **선택 프레임의 원본 ATD 공 XY 복원**으로 구현했습니다. H 행렬은 새로 추정하거나 바꾸지 않았습니다.
원본 ATD가 없으면 기존 보정 좌표를 유지하며, 선택하지 않은 모든 행은 이전 보정 TXT와 바이트 단위로 같습니다.
선수·속도(s)·프레임·구분자·줄끝·추가 마지막 ATD 1행을 보존했습니다. 원본과 기존 산출물은 덮어쓰지 않았습니다.

[전체 비교 영상](comparison_full_match.mp4) · [새 ATD TXT](BAR-ATH_broadcast_atd.txt) · [전체 좌표 CSV](corrected_ball_coordinates.csv) · [교체 프레임](replacement_log.csv) · [교체 연속 구간](replacement_segments.csv)

영상은 전체 원본 {N:,}프레임, 25FPS, 원본 음성을 유지합니다. 왼쪽 중계영상과 오른쪽 105×68m 경기장 도면입니다.
회색 빈 원은 대체 전 보정 좌표, 하늘색은 원본 ATD로 대체한 최종 좌표, 초록색은 대체 없이 유지한 최종 좌표입니다.
중계영상의 점은 같은 원본 H로 재투영하며 실제 영상 검출 자체가 아닙니다. H가 없거나 화면 밖이면 영상 점을 숨기고 경기장 좌표는 유지합니다.
좌표가 없는 프레임에는 점을 만들거나 직전 프레임의 점을 유지하지 않습니다.

원본 ATD와의 Y RMSE는 {summary['before_against_original']['y_rmse_m']:.6f}m → {summary['after_against_original']['y_rmse_m']:.6f}m입니다.
원본값으로 되돌렸기 때문에 원본과 차이가 줄어든 수치이며, 실제 공 위치 정확도 개선을 증명하지 않습니다.
''', encoding='utf-8')
    return summary


def make_assets():
    directory = OUT / 'assets'
    directory.mkdir(exist_ok=True)
    (OUT / 'logs').mkdir(exist_ok=True)
    im = Image.new('RGBA', (W, H), (21, 24, 30, 255))
    d = ImageDraw.Draw(im)
    d.text((18, 10), 'Y 차이 -0.1 이하 · 원본 ATD 좌표 대체 전/후', font=font(26), fill=(241, 241, 241))
    d.text((980, 10), '대체 전 / 대체 후 경기장 좌표', font=font(24), fill=(241, 241, 241))
    d.text((18, 51), '회색 ○ 대체 전', font=font(20), fill=(220, 220, 220))
    d.text((268, 51), '하늘색 원본 XY 적용', font=font(20), fill=COLORS[3])
    d.text((608, 51), '초록 기존 보정 유지', font=font(20), fill=COLORS[1])
    d.text((18, 77), f'전체 {N:,}프레임 · 원본 25FPS / 시간 / 음성 유지', font=font(15), fill=(200, 206, 214))
    d.text((980, 44), '원본 ATD XY 대체 · 같은 원본 H 사용', font=font(18), fill=(225, 225, 225))
    d.text((980, 71), '프레임', font=font(17), fill=(238, 238, 238))
    d.text((1188, 71), '시간', font=font(17), fill=(238, 238, 238))
    d.rectangle((980, 110, 1580, 110 + 120 * SCALE), fill=(25, 53, 38, 255))
    def p(x, y): return tuple(pitch([x, y]))
    d.rectangle((*p(0, 0), *p(105, 68)), fill=(30, 87, 49), outline=(214, 224, 217), width=2)
    d.line([p(52.5, 0), p(52.5, 68)], fill=(214, 224, 217), width=2)
    for a, b in [(0, 16.5), (105, 88.5)]:
        d.line([p(a, 13.84), p(b, 13.84), p(b, 54.16), p(a, 54.16)], fill=(214, 224, 217), width=2)
    for a, b in [(0, 5.5), (105, 99.5)]:
        d.line([p(a, 24.84), p(b, 24.84), p(b, 43.16), p(a, 43.16)], fill=(214, 224, 217), width=2)
    center = pitch([52.5, 34]); r = 9.15 * SCALE
    d.ellipse((center[0] - r, center[1] - r, center[0] + r, center[1] + r), outline=(214, 224, 217), width=2)
    for x in [11, 52.5, 94]:
        a, b = p(x, 34)
        d.ellipse((a - 2, b - 2, a + 2, b + 2), fill=(214, 224, 217))
    d.text((990, 119), '105 × 68m · 경기장 밖 좌표도 그대로 표시', font=font(16), fill=(225, 232, 227))
    d.text((18, 655), '선택 기준: (이전 보정 Y - 원본 Y) / 68 <= -0.1', font=font(21), fill=(229, 233, 237))
    d.text((18, 686), '선택 프레임은 원본 공 X·Y 함께 적용 · 나머지는 유지', font=font(20), fill=(229, 233, 237))
    d.text((980, 660), '최종 X', font=font(18), fill=(235, 235, 235))
    d.text((1270, 660), '최종 Y', font=font(18), fill=(235, 235, 235))
    d.text((980, 704), '미터 좌표 · 새 ATD TXT에 실제 저장한 값', font=font(18), fill=(226, 232, 238))
    d.text((980, 733), 'H 행렬·선수·속도·원본 파일 보존', font=font(16), fill=(215, 218, 224))
    im.paste((0, 0, 0, 0), (0, 100, 960, 640))
    im.save(directory / 'background.png')
    source_text = {0: '대체 후: 공 좌표 없음', 1: '대체 후: 이전 보정 좌표 유지', 3: '대체 후: 원본 ATD 공 XY 적용'}
    state_text = {0: '두 데이터 모두 공 좌표 없음', 1: '조건 해당: ΔY/68 <= -0.1 (ΔY <= -6.8m)',
                  2: '원본 ATD 결측: 이전 보정 좌표 유지', 3: '조건 미해당: 이전 보정 좌표 유지'}
    for code, text in source_text.items():
        im = Image.new('RGBA', (600, 34), (21, 24, 30, 255))
        ImageDraw.Draw(im).text((0, 1), text, font=font(22), fill=COLORS[code])
        im.save(directory / f'source_{code}.png')
    for code, text in state_text.items():
        im = Image.new('RGBA', (600, 30), (21, 24, 30, 255))
        ImageDraw.Draw(im).text((0, 1), text, font=font(19), fill=(228, 232, 236))
        im.save(directory / f'state_{code}.png')
    for code, text in {0: '영상 점: 같은 원본 H로 재투영한 공 XY · 실제 영상 검출 위치와 다를 수 있음',
                       1: '경기장 좌표는 있음 · H 사용 불가/화면 밖이라 영상 점만 숨김'}.items():
        im = Image.new('RGBA', (930, 28), (21, 24, 30, 255))
        ImageDraw.Draw(im).text((0, 1), text, font=font(17), fill=(230, 218, 176) if code else (208, 214, 222))
        im.save(directory / f'projection_{code}.png')
    mono = ImageFont.truetype('/System/Library/Fonts/Menlo.ttc', 20)
    for char in '0123456789:+-. ':
        im = Image.new('RGBA', (17, 28), (21, 24, 30, 255))
        ImageDraw.Draw(im).text((1, 0), char, font=mono, fill=(242, 242, 242))
        im.save(directory / f'glyph_{ord(char)}.png')


def prepare_video():
    make_assets()
    before = ball_text(BEFORE)[:N, :2]
    after = ball_text(AFTER)[:N, :2]
    old = ball_text(ORIGINAL)[:N, :2]
    d = read(OUT / 'corrected_ball_coordinates.csv').iloc[:N].copy()
    assert np.array_equal(d[['final_x', 'final_y']], after, equal_nan=True)
    assert np.array_equal(d[['before_x', 'before_y']], before, equal_nan=True)
    htable = read(HFILE)
    assert np.array_equal(htable.frame, np.arange(N + 1))
    h = htable[[f'h{i}{j}' for i in range(3) for j in range(3)]].to_numpy(float).reshape(-1, 3, 3)[:N]
    good_h = np.isfinite(h).all(axis=(1, 2)) & htable.valid.iloc[:N].eq(1).to_numpy() & htable.reprojection_error.iloc[:N].le(5).to_numpy()
    center = np.einsum('nij,j->ni', h, [.5, .5, 1.])[:, 2]
    def points(norm):
        valid = np.isfinite(norm).all(axis=1)
        world = norm * [105., 68.]
        q, depth = project(h, norm)
        visible = valid & good_h & np.isfinite(q).all(axis=1) & (depth * center > 0)
        visible &= ((q >= [0, 0]) & (q < [1920, 1080])).all(axis=1)
        image = q * .5 + [0, 100]
        image[~visible] = np.nan
        pitch_visible = valid & ((world >= BOUNDS[0]) & (world <= BOUNDS[1])).all(axis=1)
        assert np.array_equal(pitch_visible, valid)
        field = pitch(world)
        field[~valid] = np.nan
        return image, field, visible, valid
    bi, bp, biv, bv = points(before)
    ai, ap, aiv, av = points(after)
    restored = d.original_xy_restored.to_numpy(bool)
    code = np.where(~av, 0, np.where(restored, 3, 1))
    state = np.where(~av, 0, np.where(restored, 1, np.where(~np.isfinite(old).all(axis=1), 2, 3)))
    values = np.c_[np.arange(N), code, state, bi, bp, ai, ap, after * [105., 68.], av & ~aiv]
    rows = [[int(v) if j in [0, 1, 2, 13] else (float(v) if np.isfinite(v) else None)
             for j, v in enumerate(row)] for row in values]
    (OUT / 'assets/full_rows.json').write_text(json.dumps(rows, separators=(',', ':'), allow_nan=False), encoding='utf-8')
    d['before_image_visible'] = biv
    d['final_image_visible'] = aiv
    d['final_pitch_visible'] = av
    d.to_csv(OUT / 'overlay_frame_metadata.csv', index=False)
    assert np.allclose((ap[av] - ORIGIN) / SCALE, after[av] * [105., 68.], atol=1e-10)
    assert np.allclose((bp[bv] - ORIGIN) / SCALE, before[bv] * [105., 68.], atol=1e-10)
    assert np.array_equal(after[restored], old[restored], equal_nan=True)
    assert np.array_equal(after[~restored], before[~restored], equal_nan=True)
    assert np.array_equal(np.asarray(rows, float)[:, 11:13], after * [105., 68.], equal_nan=True)
    save('video_manifest.json', dict(status='ASSETS_READY', full_video=VIDEO.name,
        source_frames=N, output_frames=N, fps=FPS, size=[W, H], duration_seconds=N / FPS,
        full_original_timeline=True, source_audio_requested=True, restored_frames=int(restored.sum()),
        before_marker='Grey hollow ring: previous corrected ATD',
        after_markers={'green': 'Previous corrected XY retained', 'cyan': 'Original ATD XY restored'},
        all_points_read_from_actual_saved_TXT=True, same_original_H_for_both_tracks=True,
        metadata_sha256=digest(OUT / 'assets/full_rows.json'), renderer_sha256=digest(NATIVE),
        source_video_stat=dict(bytes=SOURCE.stat().st_size, mtime_ns=SOURCE.stat().st_mtime_ns)))
    save('overlay_metadata_verification.json', dict(passed=True, checked_frames=N,
        frame_mapping_exact=True, both_tracks_from_actual_TXT=True, all_projected_and_field_points_verified=True,
        selection_and_colors_verified=True, absent_coordinates_are_null=True))


def render():
    assert not VIDEO.exists()
    cmd = [str(NATIVE), str(SOURCE), str(OUT / 'assets/full_rows.json'),
           str(OUT / 'assets/background.png'), str(VIDEO)]
    with (OUT / 'logs/render.log').open('w') as stream:
        subprocess.run(cmd, cwd=ROOT, stdout=stream, stderr=subprocess.STDOUT, check=True)


def verify():
    if str(ROOT) not in sys.path:
        sys.path.insert(0, str(ROOT))
    from starter.ball_tracking_video_utils import color_pixels
    rows = np.array(json.loads((OUT / 'assets/full_rows.json').read_text()), float)
    assert len(rows) == N and np.array_equal(rows[:, 0], np.arange(N))
    cap = cv2.VideoCapture(str(VIDEO))
    assert cap.isOpened() and int(cap.get(7)) == N and cap.get(5) == FPS
    assert cap.get(3) == W and cap.get(4) == H
    original = cv2.VideoCapture(str(SOURCE))
    assert original.isOpened()
    indices = set(np.linspace(0, N - 1, 30).astype(int).tolist())
    # Cover both sides of replacements and missing transitions, plus severe and near-threshold examples.
    selected = rows[:, 1] == 3
    edges = np.flatnonzero(selected != np.r_[False, selected[:-1]])
    for f in edges[np.linspace(0, len(edges) - 1, min(20, len(edges))).astype(int)]:
        indices.update([max(0, int(f) - 1), int(f), min(N - 1, int(f) + 1)])
    for code in [0, 1, 3]:
        ids = np.flatnonzero(rows[:, 1] == code)
        indices.update(ids[np.linspace(0, len(ids) - 1, min(6, len(ids))).astype(int)].tolist())
    gaps = np.flatnonzero(~np.isfinite(rows[:, 9]) & np.r_[False, np.isfinite(rows[:-1, 9])])
    for f in gaps[np.linspace(0, len(gaps) - 1, min(8, len(gaps))).astype(int)]:
        indices.update([int(f) - 1, int(f)])
    y = read(OUT / 'corrected_ball_coordinates.csv').iloc[:N]
    indices.add(int(y.before_signed_delta_y_over_68.idxmin()))
    selected_ids = np.flatnonzero(selected)
    near = selected_ids[np.argsort(-y.before_signed_delta_y_over_68.to_numpy()[selected_ids])[:3]]
    indices.update(near.tolist())
    glyphs = {c: cv2.imread(str(OUT / 'assets' / f'glyph_{ord(c)}.png')) for c in '0123456789'}
    source_sprites = {code: cv2.imread(str(OUT / 'assets' / f'source_{code}.png')) for code in [0, 1, 3]}
    state_sprites = {code: cv2.imread(str(OUT / 'assets' / f'state_{code}.png')) for code in range(4)}
    samples = []
    for f in sorted(indices):
        cap.set(1, f)
        ok, frame = cap.read()
        assert ok and frame.shape == (H, W, 3)
        original.set(1, f)
        ok, src = original.read()
        assert ok
        left = cv2.resize(frame[100:640, :960], (64, 36), interpolation=cv2.INTER_AREA)
        ref = cv2.resize(src, (64, 36), interpolation=cv2.INTER_AREA)
        source_mae = float(np.abs(left.astype(float) - ref.astype(float)).mean())
        assert source_mae < 12, (f, 'source frame mismatch', source_mae)
        digits = ''
        for j in range(6):
            patch = frame[68:96, 1050 + j * 17:1050 + (j + 1) * 17]
            scores = {c: float(np.abs(patch.astype(float) - glyphs[c].astype(float)).mean()) for c in glyphs}
            digits += min(scores, key=scores.get)
        assert digits == f'{f:06d}', (f, digits)
        code = int(rows[f, 1]); state = int(rows[f, 2])
        status_mae = float(np.abs(frame[560:594, 980:1580].astype(float) - source_sprites[code].astype(float)).mean())
        state_mae = float(np.abs(frame[602:632, 980:1580].astype(float) - state_sprites[state].astype(float)).mean())
        assert status_mae < 10 and state_mae < 10, (f, 'status not aligned', status_mae, state_mae)
        marker = None
        if np.isfinite(rows[f, 9]):
            x, v = np.rint(rows[f, 9:11]).astype(int)
            marker = color_pixels(frame[v - 9:v + 10, x - 9:x + 10], code)
            assert marker >= 12, (f, 'pitch marker', marker)
        elif f and np.isfinite(rows[f - 1, 9]):
            x, v = np.rint(rows[f - 1, 9:11]).astype(int)
            assert color_pixels(frame[v - 2:v + 3, x - 2:x + 3], int(rows[f - 1, 1])) < 8, (f, 'stale marker')
        if np.isfinite(rows[f, 7]):
            x, v = np.rint(rows[f, 7:9]).astype(int)
            if 13 < x < 947 and 113 < v < 627:
                assert color_pixels(frame[v - 13:v + 14, x - 13:x + 14], code) >= 10, (f, 'image marker')
        if code == 3:
            x, v = np.rint(rows[f, 5:7]).astype(int)
            # Before marker has a larger hollow radius than the final field dot.
            patch = frame[v - 11:v + 12, x - 11:x + 12]
            gray = ((patch.max(axis=2) - patch.min(axis=2) < 30) & (patch.mean(axis=2) > 150)).sum()
            assert gray >= 8, (f, 'before marker missing', gray)
        samples.append(dict(frame=f, restored_original_xy=code == 3, source_frame_mae=source_mae,
            frame_digits=digits, status_mae=status_mae, state_mae=state_mae, final_marker_color_pixels=marker))
    cap.release(); original.release()
    progress = json.loads(VIDEO.with_suffix('.mp4.progress.json').read_text())
    assert progress['status'] == 'complete' and progress['frames'] == N
    assert abs(progress['output_duration_seconds'] - N / FPS) < .1 and progress['audio_preserved']
    summary = json.loads((OUT / 'correction_summary.json').read_text())
    assert all(digest(ROOT / p) == signature for p, signature in summary['input_hashes'].items())
    manifest = json.loads((OUT / 'video_manifest.json').read_text())
    assert digest(OUT / 'assets/full_rows.json') == manifest['metadata_sha256']
    assert digest(NATIVE) == manifest['renderer_sha256']
    assert dict(bytes=SOURCE.stat().st_size, mtime_ns=SOURCE.stat().st_mtime_ns) == manifest['source_video_stat']
    report = dict(passed=True, frames=N, fps=FPS, size=[W, H], duration_seconds=N / FPS,
        source_audio_preserved=True, sampled_frames=len(samples), samples=samples,
        full_source_timeline_preserved=True, frame_digits_source_video_status_and_markers_verified=True,
        absent_marker_not_carried=True, existing_original_corrected_files_and_H_hashes_preserved=True,
        video_sha256=digest(VIDEO))
    save('video_verification.json', report)
    manifest.update(status='COMPLETE', verification_passed=True, sampled_frames=len(samples),
                    source_audio_preserved=True, video_sha256=report['video_sha256'])
    save('video_manifest.json', manifest)
    summary.update(status='COMPLETE', comparison_video_verified=True, video_duration_seconds=N / FPS)
    save('correction_summary.json', summary)
    save('artifact_integrity.json', {p.name: digest(p) for p in sorted(OUT.iterdir())
                                   if p.is_file() and p.name != 'artifact_integrity.json'})
    return summary


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('stage', choices=['prepare', 'render', 'verify'])
    args = parser.parse_args()
    if args.stage == 'prepare':
        export()
        prepare_video()
    elif args.stage == 'render':
        render()
    else:
        verify()
