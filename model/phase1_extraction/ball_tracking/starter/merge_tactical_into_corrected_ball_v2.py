"""Use original-CSV missingness to replace ball records IN corrected temporal V2.

The mask input is never the coordinate destination. Start with the corrected
track and preserve it exactly wherever no available tactical replacement applies.
"""
from pathlib import Path
import itertools
import json
import shutil
import sys

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from starter.ball_tactical_fusion_common import read, metadata, scan_atd, no_observed_players, alignment
from starter.render_broadcast_atd_from_csv import clean, digest
from starter.tracking_utils import read_tracking, track_teams

MASK_CSV = ROOT / 'data/derived/ball_temporal_ai_20261117/ball_homography_all_frames.csv'
DESTINATION = ROOT / 'outputs/broadcast_atd_temporal_v2/ball_homography_all_frames.csv'
DEST_ATD = DESTINATION.with_name('BAR-ATH_broadcast_atd.txt')
BROADCAST = ROOT / 'data/broadcast/BAR-ATH_broadcast_atd.txt'
HFILE = BROADCAST.with_name('BAR-ATH_broadcast_homography.csv')
TACTICAL = ROOT / 'data/tactical/BAR-ATH_tactical_atd.txt'
MAPFILE = ROOT / 'data/derived/sync/final_tracking_alignment.csv'
OUT = ROOT / 'outputs/broadcast_tactical_ball_merge_v2'
REJECTED = ROOT / 'outputs/broadcast_tactical_ball_merge_v1'
N = 164487
PITCH = np.array([105., 68.])


def save(name, value):
    (OUT / name).write_text(json.dumps(clean(value), ensure_ascii=False, indent=2,
                                     allow_nan=False) + '\n', encoding='utf-8')


def apply_replacements(destination_xy, original_missing_mask, tactical_xy, tactical_available, highlight):
    current = np.asarray(destination_xy, float)
    original_missing_mask = np.asarray(original_missing_mask, bool)
    tactical = np.asarray(tactical_xy, float)
    available = np.asarray(tactical_available, bool)
    highlight = np.asarray(highlight, bool)
    assert current.shape == tactical.shape == (len(original_missing_mask), 2)
    assert not np.any(available & ~np.isfinite(tactical).all(axis=1))
    requested = original_missing_mask | highlight
    use = requested & available
    final = current.copy()
    final[use] = tactical[use]
    reason = np.full(len(current), 'KEEP_CORRECTED_DESTINATION', dtype=object)
    reason[requested & ~available] = 'TACTICAL_UNAVAILABLE_KEEP_CORRECTED_DESTINATION'
    reason[use & original_missing_mask & ~highlight] = 'ORIGINAL_CSV_MISSING_REPLACE_CORRECTED_DESTINATION'
    reason[use & ~original_missing_mask & highlight] = 'NO_PLAYERS_REPLACE_CORRECTED_DESTINATION'
    reason[use & original_missing_mask & highlight] = 'ORIGINAL_CSV_MISSING_AND_NO_PLAYERS_REPLACE_CORRECTED_DESTINATION'
    return final, use, requested, reason


def policy_tests():
    corrected = np.array([[.11, .22], [.12, .23], [.13, .24], [.14, .25],
                          [np.nan, np.nan], [np.nan, np.nan], [0., 0.], [.2, .3]])
    tac = np.tile([.8, .9], (8, 1))
    original_missing = np.array([False, True, True, False, True, False, False, True])
    tactical_available = np.array([True, True, False, True, True, True, True, False])
    highlight = np.array([False, False, False, True, False, False, False, True])
    final, use, requested, _ = apply_replacements(corrected, original_missing, tac, tactical_available, highlight)
    assert np.array_equal(use, [False, True, False, True, True, False, False, False])
    assert np.array_equal(final[[0, 2, 6, 7]], corrected[[0, 2, 6, 7]])
    assert np.array_equal(final[[1, 3, 4]], tac[[1, 3, 4]])
    assert np.isnan(final[5]).all(), 'Destination missingness must not silently replace the original input mask'
    assert requested[1] and requested[2] and requested[7]
    assert no_observed_players([0, 0, 1], [0, 1, 0]).tolist() == [True, False, False]
    return dict(passed=True, cases=[
        'KEEP_Y_CORRECTED_COORDINATE_WHEN_ORIGINAL_PRESENT',
        'ORIGINAL_MISSING_REPLACES_ALREADY_VALID_CORRECTED_DESTINATION',
        'TACTICAL_UNAVAILABLE_PRESERVES_VALID_CORRECTED_DESTINATION',
        'HIGHLIGHT_OVERRIDES_REGARDLESS_OF_ORIGINAL_MISSINGNESS',
        'ORIGINAL_MISSING_FILLS_MISSING_DESTINATION',
        'CURRENT_MISSINGNESS_DOES_NOT_REDEFINE_ORIGINAL_MASK',
        'ZERO_XY_IS_VALID', 'BOTH_TRIGGERS_TACTICAL_UNAVAILABLE_RETAINS_DESTINATION',
        'GK_REFERENCE_NOT_A_PLAYER_OBSERVATION', 'H_INVALID_ALONE_NOT_A_HIGHLIGHT'])


def export_json(table):
    info = dict(video_frames=N, fps=25, pitch_m=[105, 68],
        missing_mask_source=str(MASK_CSV.relative_to(ROOT)),
        corrected_coordinate_destination=str(DESTINATION.relative_to(ROOT)),
        policy='final starts as corrected destination; replace only when (original_missing OR no_players) AND aligned_tactical_ball_valid',
        no_new_interpolation_or_AI=True, tactical_reference_is_not_broadcast_image_detection=True)
    with (OUT / 'ball_homography_all_frames.json').open('w', encoding='utf-8') as stream:
        stream.write('{"metadata":' + json.dumps(info, ensure_ascii=False, separators=(',', ':')) + ',"frames":[')
        for i, r in enumerate(table.itertuples(index=False)):
            row = dict(frame=int(r.frame), atd_frame_id=int(r.atd_frame_id), video_seconds=r.video_seconds,
                missing_in_original_csv=bool(r.missing_in_original_csv),
                corrected_target_x=r.merge_before_x, corrected_target_y=r.merge_before_y,
                tactical_frame=int(r.tactical_frame) if np.isfinite(r.tactical_frame) else None,
                tactical_atd_frame_id=int(r.tactical_atd_frame_id) if np.isfinite(r.tactical_atd_frame_id) else None,
                tactical_used=bool(r.tactical_used), highlight_inferred=bool(r.highlight_inferred),
                pitch_x_normalized=r.final_x, pitch_y_normalized=r.final_y,
                pitch_x_m=r.pitch_x_m, pitch_y_m=r.pitch_y_m, ball_s_kph=r.final_ball_s_kph,
                final_source=r.final_source, replacement_reason=r.replacement_reason,
                is_prediction=bool(r.is_prediction), ball_observed=bool(r.ball_observed),
                ball_position_available=bool(r.ball_position_available))
            if i: stream.write(',')
            stream.write(json.dumps(clean(row), ensure_ascii=False, separators=(',', ':'), allow_nan=False))
        stream.write(']}\n')


def main():
    assert not REJECTED.exists(), 'The rejected merge results must be deleted first'
    OUT.mkdir(parents=True, exist_ok=True)
    target_atd = OUT / DEST_ATD.name
    if target_atd.exists(): raise FileExistsError(f'Refusing to overwrite {target_atd}')
    paths = [MASK_CSV, DESTINATION, DEST_ATD, DEST_ATD.with_suffix('.xml'),
        BROADCAST, BROADCAST.with_suffix('.xml'), HFILE, TACTICAL, TACTICAL.with_suffix('.xml'), MAPFILE]
    hashes = {str(p.relative_to(ROOT)): digest(p) for p in paths}
    tests = policy_tests()
    original = read(MASK_CSV); corrected = read(DESTINATION)
    assert len(original) == len(corrected) == N
    assert np.array_equal(original.source_frame, np.arange(N)) and np.array_equal(corrected.frame, np.arange(N))
    original_xy = original[['pitch_x_normalized', 'pitch_y_normalized']].to_numpy(float)
    original_valid = np.isfinite(original_xy).all(axis=1)
    flags = original.pitch_usable.to_numpy(bool) & original.ball_position_available.to_numpy(bool) & original.homography_valid.to_numpy(bool)
    assert np.array_equal(original_valid, flags)
    original_missing = ~original_valid
    current = corrected[['final_x', 'final_y']].to_numpy(float)
    current_valid = np.isfinite(current).all(axis=1)
    assert np.array_equal(current_valid, corrected.ball_position_available.to_numpy(bool))
    assert np.array_equal(current, corrected[['pitch_x_normalized', 'pitch_y_normalized']], equal_nan=True)
    assert np.allclose(current * PITCH, corrected[['pitch_x_m', 'pitch_y_m']], atol=1e-12, equal_nan=True)
    bm, tm = metadata(BROADCAST.with_suffix('.xml')), metadata(TACTICAL.with_suffix('.xml'))
    assert bm['fps'] == tm['fps'] == 25 and bm['pitch'] == tm['pitch'] == [105., 68.]
    assert bm['speed_unit'] == tm['speed_unit'] == 'kph'
    assert bm['team_first_half_directions'] == tm['team_first_half_directions']
    destination_ball, _, players = scan_atd(DEST_ATD, bm, presence=True)
    tactical_ball, tactical_tokens, _ = scan_atd(TACTICAL, tm)
    assert destination_ball.shape == (N + 1, 3) and tactical_ball.shape == (199319, 3)
    assert np.array_equal(current, destination_ball[:N, :2], equal_nan=True)
    mapped, alignment_half, specs = alignment(len(tactical_ball))
    has_mapping = mapped >= 0
    aligned = np.full((N, 3), np.nan)
    aligned[has_mapping] = tactical_ball[mapped[has_mapping]]
    available = has_mapping & np.isfinite(aligned[:, :2]).all(axis=1)
    highlight = no_observed_players(players[:N, 1], players[:N, 3])
    final, use, requested, reason = apply_replacements(current, original_missing, aligned[:, :2], available, highlight)
    final_valid = np.isfinite(final).all(axis=1)
    expected_ball = destination_ball.copy()
    expected_ball[:N, :2] = final
    expected_ball[:N, 2][use] = aligned[use, 2]
    output = corrected.copy(deep=True)
    output['merge_before_x'] = current[:, 0]; output['merge_before_y'] = current[:, 1]
    output['merge_before_source'] = corrected.final_source
    output['merge_before_valid'] = current_valid
    output['original_mask_x'] = original_xy[:, 0]; output['original_mask_y'] = original_xy[:, 1]
    output['missing_in_original_csv'] = original_missing
    output['alignment_half'] = alignment_half
    output['tactical_frame'] = np.where(has_mapping, mapped, np.nan)
    output['tactical_atd_frame_id'] = np.where(has_mapping, mapped + 1, np.nan)
    output['alignment_status'] = np.where(has_mapping, 'HALF_OFFSET_OUTFIELD_VALIDATED', 'UNMAPPED_OUTSIDE_KNOWN_HALVES')
    output['tactical_ball_available'] = available
    output['tactical_x'] = aligned[:, 0]; output['tactical_y'] = aligned[:, 1]
    output['tactical_ball_s_kph'] = aligned[:, 2]
    output['tactical_used'] = use
    output['replacement_requested'] = requested
    output['replacement_reason'] = reason
    output['fusion_action'] = np.where(use, np.where(current_valid, 'REPLACE_VALID_CORRECTED_TARGET', 'FILL_CORRECTED_TARGET_GAP'), 'KEEP_CORRECTED_TARGET')
    output['highlight_inferred'] = highlight
    output['player_count'] = players[:N, 0]; output['outfield_player_count'] = players[:N, 1]
    output['goalkeeper_count'] = players[:N, 2]; output['goalkeeper_with_finite_speed_count'] = players[:N, 3]
    output['reliable_player_count'] = players[:N, 1] + players[:N, 3]
    output['highlight_reason'] = np.where(highlight, np.where(players[:N, 0] == 0, 'NO_PLAYER_XY', 'ONLY_XML_GK_REFERENCES_WITH_MISSING_SPEED'), 'PLAYER_TRACKS_PRESENT')
    h = read(HFILE)
    assert len(h) == N + 1 and np.array_equal(h.frame, np.arange(N + 1))
    output['broadcast_homography_valid'] = h.valid.iloc[:N].eq(1).to_numpy() & np.isfinite(h.iloc[:N][[f'h{i}{j}' for i in range(3) for j in range(3)]].to_numpy()).all(axis=1)
    output['broadcast_homography_error'] = h.reprojection_error.iloc[:N].to_numpy()
    for col, values in [('final_x', final[:, 0]), ('final_y', final[:, 1]),
        ('pitch_x_normalized', final[:, 0]), ('pitch_y_normalized', final[:, 1]),
        ('final_x_m', final[:, 0] * 105.), ('final_y_m', final[:, 1] * 68.),
        ('pitch_x_m', final[:, 0] * 105.), ('pitch_y_m', final[:, 1] * 68.)]:
        output.loc[use, col] = values[use]
    output.loc[use, 'final_source'] = 'TACTICAL_ATD'
    output.loc[use, 'temporal_action'] = 'TACTICAL_ATD_REPLACEMENT'
    for col in ['final_source_is_prediction', 'is_prediction', 'new_temporal_estimate',
                'actual_recovered_detection', 'final_actual_observation', 'ball_observed', 'ball_detected']:
        output.loc[use, col] = False
    for col in ['left_anchor_frame', 'right_anchor_frame', 'candidate_index', 'image_x', 'image_y']:
        output.loc[use, col] = np.nan
    output.loc[use, ['source_detection_origin', 'detection_origin']] = ''
    output.loc[use, 'image_coordinate_kind'] = 'TACTICAL_GROUND_REFERENCE_NO_BROADCAST_DETECTION'
    output.loc[use, 'ball_position_available'] = True
    output['pitch_usable'] = final_valid
    output['source_is_prediction'] = output.is_prediction
    output['final_ball_s_kph'] = expected_ball[:N, 2]
    output['speed_source'] = np.where(use, 'TACTICAL_ATD_S', 'CORRECTED_TARGET_ATD_S_PRESERVED')
    assert output.loc[~use, corrected.columns].equals(corrected.loc[~use]), 'Non-replaced destination fields changed'
    output = pd.concat([output, corrected.add_prefix('target_'), original.add_prefix('mask_input_')], axis=1)
    output.to_csv(OUT / 'ball_homography_all_frames.csv', index=False)
    logcols = ['frame', 'atd_frame_id', 'tactical_frame', 'tactical_atd_frame_id',
        'missing_in_original_csv', 'merge_before_valid', 'merge_before_x', 'merge_before_y',
        'final_x', 'final_y', 'tactical_x', 'tactical_y', 'final_ball_s_kph',
        'highlight_inferred', 'highlight_reason', 'fusion_action', 'replacement_reason']
    output.loc[use, logcols].to_csv(OUT / 'replacement_log.csv', index=False)
    output[['frame', 'atd_frame_id', 'alignment_half', 'tactical_frame', 'tactical_atd_frame_id', 'alignment_status']].to_csv(OUT / 'frame_alignment.csv', index=False)
    output[['frame', 'missing_in_original_csv', 'highlight_inferred', 'player_count',
        'outfield_player_count', 'goalkeeper_count', 'goalkeeper_with_finite_speed_count',
        'reliable_player_count', 'highlight_reason', 'broadcast_homography_valid',
        'broadcast_homography_error', 'tactical_ball_available', 'tactical_used']].to_csv(OUT / 'replacement_masks.csv', index=False)
    output.loc[requested & ~available, ['frame', 'alignment_status', 'missing_in_original_csv',
        'highlight_inferred', 'merge_before_valid', 'ball_position_available', 'replacement_reason']].to_csv(OUT / 'tactical_unavailable_target_retained.csv', index=False)
    untouched_rows = 0
    with DEST_ATD.open('rb') as source, target_atd.open('wb') as destination:
        for f, line in enumerate(source):
            if f < N and use[f]:
                body = line.rstrip(b'\r\n'); prefix = body.rsplit(b':', 1)[0]
                new = prefix + b':' + b','.join(tactical_tokens[mapped[f]]) + line[len(body):]
            else:
                new = line; untouched_rows += 1
            destination.write(new)
    shutil.copyfile(DEST_ATD.with_suffix('.xml'), target_atd.with_suffix('.xml'))
    actual_ball, _, actual_players = scan_atd(target_atd, bm, presence=True)
    assert np.array_equal(actual_ball, expected_ball, equal_nan=True)
    assert np.array_equal(actual_players, players)
    with DEST_ATD.open('rb') as source, target_atd.open('rb') as destination:
        for f, (a, b) in enumerate(itertools.zip_longest(source, destination)):
            assert a is not None and b is not None
            aa = a.rstrip(b'\r\n'); bb = b.rstrip(b'\r\n')
            assert aa.rsplit(b':', 1)[0] == bb.rsplit(b':', 1)[0]
            assert a[len(aa):] == b[len(bb):]
            if f == N or not use[f]: assert a == b
            else: assert bb.rsplit(b':', 1)[1].split(b',') == tactical_tokens[mapped[f]]
    parsed = read_tracking(target_atd, track_teams(target_atd.with_suffix('.xml')), N + 1, 0, N + 1)
    assert np.array_equal(parsed[:, -1], expected_ball[:, :2].astype(np.float32), equal_nan=True)
    del parsed
    assert np.array_equal(final[~use], current[~use], equal_nan=True)
    assert np.array_equal(final[use], tactical_ball[mapped[use], :2])
    assert not (use & ~has_mapping).any()
    assert final_valid[current_valid].all()
    export_json(output)
    summary = dict(status='COMPLETE', missing_mask_csv=str(MASK_CSV.relative_to(ROOT)),
        corrected_coordinate_destination=str(DESTINATION.relative_to(ROOT)),
        coordinate_base_is_corrected_destination_not_missing_mask=True,
        video_frames=N, ATD_rows=N + 1, original_mask_missing_frames=int(original_missing.sum()),
        corrected_destination_initial_valid_frames=int(current_valid.sum()),
        total_tactical_replacements=int(use.sum()),
        existing_corrected_coordinates_replaced=int((use & current_valid).sum()),
        corrected_destination_gaps_filled=int((use & ~current_valid).sum()),
        original_missing_rule_replacements=int((use & original_missing).sum()),
        original_missing_rule_replaced_already_valid_target=int((use & original_missing & current_valid).sum()),
        additional_highlight_rule_replacements=int((use & highlight & ~original_missing).sum()),
        tactical_highlight_total=int((use & highlight).sum()),
        tactical_highlight_replaced_existing_target=int((use & highlight & current_valid).sum()),
        original_missing_tactical_unavailable_valid_target_preserved=int((original_missing & ~available & current_valid).sum()),
        untouched_destination_video_frames=int((~use).sum()), untouched_ATD_rows_including_extra=untouched_rows,
        valid_destination_coordinates_retained=int((current_valid & ~use).sum()),
        final_valid_frames=int(final_valid.sum()), final_missing_frames=int((~final_valid).sum()),
        mapped_frames=int(has_mapping.sum()), unmapped_frames=int((~has_mapping).sum()), alignment=specs,
        prior_Y_stage_rows_retained=int((corrected.prior_original_xy_restored & ~use).sum()),
        prior_Y_stage_rows_replaced_by_explicit_rule=int((corrected.prior_original_xy_restored & use).sum()),
        prior_new_AI_estimates_retained=int((corrected.new_temporal_estimate & ~use).sum()),
        prior_new_AI_estimates_replaced_by_explicit_rule=int((corrected.new_temporal_estimate & use).sum()),
        prior_actual_candidate_recoveries_retained=int((corrected.actual_recovered_detection & ~use).sum()),
        source_counts=output.final_source.value_counts().to_dict(),
        highlight_rule='No outfield XY and no XML GK with finite XY+speed. GK-only missing-speed reference positions are not detections.',
        H_invalid_alone_not_used_as_highlight=True, tactical_time_is_live_half_offset_not_replay_match=True,
        pixel_level_video_synchronization_not_claimed=True, normalized_pitch_and_meters=[105, 68],
        tactical_s_policy='Copy tactical x,y,s when used; otherwise preserve destination row byte-for-byte.',
        no_new_AI_prediction_or_detection=True, corrected_destination_and_all_sources_unmodified=True,
        source_hashes=hashes, rejected_previous_output_deleted=not REJECTED.exists(),
        output_ATD=target_atd.name, full_CSV='ball_homography_all_frames.csv', full_JSON='ball_homography_all_frames.json')
    assert all(digest(ROOT / rel) == signature for rel, signature in hashes.items())
    assert not REJECTED.exists()
    save('merge_summary.json', summary)
    save('test_results.json', dict(passed=True, policy_cases=tests,
        missing_mask_from_original_CSV_only=True, all_coordinates_start_from_corrected_destination=True,
        all_nonreplacement_destination_columns_identical=True,
        all_nonreplacement_ATD_rows_byte_identical=True,
        masked_valid_destination_overwritten_when_tactical_available=True,
        masked_valid_destination_preserved_when_tactical_unavailable=True,
        target_valid_coordinates_never_lost=True, target_and_mask_source_columns_archived_in_full=True,
        all_tactical_XY_and_s_match_mapped_original_tokens=True,
        player_frame_prefix_speed_and_newline_policy_verified=True,
        no_tactical_player_slots_copied=True, extra_ATD_row_unchanged=True,
        native_MTAH_parser_accepts_all_rows=True, no_unmapped_negative_index_wrap=True,
        source_hashes_unchanged=True, rejected_output_directory_absent=True))
    deletions = Path('/private/tmp/corrected_tactical_fusion_v2_deletions.json')
    if deletions.exists(): shutil.copyfile(deletions, OUT / 'deleted_rejected_results.json')
    audit = Path('/private/tmp/corrected_tactical_fusion_policy_audit.json')
    if audit.exists(): shutil.copyfile(audit, OUT / 'independent_expected_policy.json')
    (OUT / 'README.md').write_text(f'''# 보정된 공 좌표에 Tactical ATD 대체 적용 V2

**결측 판정 입력과 좌표를 수정하는 대상을 분리했습니다.**

- 결측 판정만: `{MASK_CSV.relative_to(ROOT)}`
- 좌표의 실제 시작점: `{DESTINATION.relative_to(ROOT)}`

먼저 원래 CSV에서 경기장 공 좌표가 없는 프레임을 찾고 그 프레임의 **보정된 목적지 좌표**를 Tactical로 바꿉니다.
따라서 목적지에 이미 원본 ATD fallback이나 AI 보정 좌표가 있어도 원래 CSV가 결측이고 Tactical이 유효하면 교체합니다.
그다음 선수 미검출 하이라이트 프레임도 같은 목적지에서 Tactical로 교체합니다.
Tactical이 없거나 시간 대응이 없으면 목적지 값을 그대로 유지합니다. 원래 CSV 값으로 되돌리거나 목적지의 유효 좌표를 지우지 않습니다.

이번에 Tactical {summary['total_tactical_replacements']:,}프레임을 적용했습니다.
현재 보정본의 유효 좌표 교체 {summary['existing_corrected_coordinates_replaced']:,}, 현재 보정본의 결측 신규 보완 {summary['corrected_destination_gaps_filled']:,}프레임입니다.
나머지 {summary['untouched_destination_video_frames']:,}프레임은 목적지의 모든 기존 필드가 같습니다.
원래 CSV는 결측이지만 목적지에는 좌표가 있고 Tactical이 없는 {summary['original_missing_tactical_unavailable_valid_target_preserved']:,}프레임을 보존했습니다.
최종 유효 {summary['final_valid_frames']:,}, 결측 {summary['final_missing_frames']:,}프레임입니다.

기존 Y 단계 대상은 {summary['prior_Y_stage_rows_retained']:,}프레임 유지했고 명시적 Tactical 규칙에 해당한 {summary['prior_Y_stage_rows_replaced_by_explicit_rule']:,}프레임만 교체했습니다.
이전 새 AI 추정은 {summary['prior_new_AI_estimates_retained']:,}프레임 유지, 규칙에 해당한 {summary['prior_new_AI_estimates_replaced_by_explicit_rule']:,}프레임 교체입니다.
원래 두 입력 CSV의 모든 필드는 `target_`, `mask_input_` 접두사로 보존했습니다.

선수 유무는 XML 슬롯 정의와 목적지 ATD 선수 XY/s로 계산했습니다. GK만 남고 속도가 NaN인 참조 위치는 실제 검출로 세지 않았습니다.
Broadcast H의 valid/오차도 저장했으며 H 불량만으로 하이라이트라고 판정하지 않았습니다. 이 하이라이트 표시는 요청한 자동 규칙이며 실제 영상 리플레이 정답 라벨은 아닙니다.
시간은 전반 T=B+4014(B7171–86637), 후반 T=B+26919(B86823–163278)로 맞췄습니다. 좌표 반전은 없으며 하이라이트도 같은 실시간에 대응합니다.
전체 픽셀 시각까지 정확한 동기화를 주장하지 않습니다. Tactical 경기장 좌표를 중계 리플레이 장면의 직접 이미지 관측으로 표시하지 않았습니다.

[새 통합 ATD](BAR-ATH_broadcast_atd.txt) · [전체 CSV](ball_homography_all_frames.csv) · [전체 JSON](ball_homography_all_frames.json) · [교체 기록](replacement_log.csv) · [교체 마스크](replacement_masks.csv)

CSV/JSON은 영상164,487프레임, TXT는164,488행이며 추가 마지막 행은 원문 그대로입니다.
좌표는 정규화 XY와105×68m를 명확히 구분했습니다. Tactical은 이미 경기장 좌표이므로 H를 다시 적용하지 않았습니다.
교체 프레임의 공 x/y/s만 Tactical에서 복사했고, 나머지 ATD 행은 목적지와 바이트 단위로 같습니다.
목적지 원본·원래 CSV·Broadcast/Tactical 원본·XML·H·기존 시간 정렬표는 변경하지 않았습니다.
잘못 만든 이전 `broadcast_tactical_ball_merge_v1` 결과와 그 영상은 삭제했습니다. 새 결과는 혼동을 막기 위해 별도 V2 폴더에 저장했습니다.
''', encoding='utf-8')
    save('artifact_integrity.json', dict(algorithm='sha256', files={p.name: digest(p)
        for p in sorted(OUT.iterdir()) if p.is_file() and p.name != 'artifact_integrity.json'},
        exporter_source_sha256=digest(Path(__file__))))


if __name__ == '__main__': main()
