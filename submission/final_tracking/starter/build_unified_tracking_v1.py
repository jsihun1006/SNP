"""One Broadcast-timeline dataset: retained V3 ball + aligned Tactical players.

No coordinate correction, smoothing, time extrapolation or new detections.
The supplied ATD player and ball payloads are retained byte for byte.
"""
from pathlib import Path
import csv
import json
import sys
import xml.etree.ElementTree as ET
from datetime import datetime
from zoneinfo import ZoneInfo

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from starter.ball_tactical_fusion_common import alignment, metadata
from starter.render_broadcast_atd_from_csv import clean, digest

BALL = ROOT / 'outputs/broadcast_tactical_ball_smoothed_v3'
TAC = ROOT / 'data/tactical/BAR-ATH_tactical_atd.txt'
MAP = ROOT / 'data/derived/sync/final_tracking_alignment.csv'
HFILE = ROOT / 'data/broadcast/BAR-ATH_broadcast_homography.csv'
OUT = ROOT / 'outputs/unified_tracking_v1'
N = 164487
FPS = 25
PITCH = np.array([105., 68.])


def save(name, value):
    (OUT / name).write_text(json.dumps(clean(value), ensure_ascii=False, indent=2,
        allow_nan=False) + '\n', encoding='utf-8')


def roster():
    tree = ET.parse(TAC.with_suffix('.xml'))
    r = tree.getroot()
    teams = {t.get('id'): t.findtext('Name') for t in r.iter('Team')}
    score = r.find('.//Score')
    home, away = score.get('idLocalTeam'), score.get('idVisitingTeam')
    counters = {home: 0, away: 0}
    rows = []
    for slot, p in enumerate(r.iter('Player')):
        team = p.get('teamId')
        counters[team] += 1
        keeper = any(x.findtext('Name') == 'position_type' and
            (x.findtext('Value') or '').lower() == 'goalkeeper' for x in p.iter('ProviderParameter'))
        rows.append(dict(slot=slot, column_prefix=f'player_{slot + 1:02d}',
            original_player_id=p.get('id'), original_track_name=p.findtext('Name'),
            team_id=team, team_name=teams[team], team_code='H' if team == home else 'A',
            team_slot=counters[team], display_label=('H' if team == home else 'A') + f'{counters[team]:02d}',
            goalkeeper_slot=keeper, label_is_tracking_slot_not_verified_jersey=True))
    assert len(rows) == 46
    return tree, rows


def write_xml(tree, specs):
    root = tree.getroot()
    config = root.find('.//GlobalConfig')
    config.find('FileName').text = 'BAR-ATH_unified_tracking_v1'
    config.find('FileDate').text = datetime.now(ZoneInfo('Asia/Seoul')).isoformat()
    config.find('ProviderName').text = 'Local unified tracking (Tactical players + retained ball V3)'
    params = config.find('ProviderGlobalParameters')
    values = {p.findtext('Name'): p.find('Value') for p in params.findall('ProviderParameter')}
    updates = dict(tracking_source='Tactical player ATD + retained corrected ball V3; Broadcast timeline',
        version='unified_v1', frame_count=str(N), first_half_start=str(specs[0]['broadcast_first']),
        first_half_end=str(specs[0]['broadcast_last'] + 1), second_half_start=str(specs[1]['broadcast_first']),
        second_half_end=str(specs[1]['broadcast_last'] + 1),
        half_range_semantics='Known aligned live ranges; zero-based start inclusive/end exclusive',
        position_units='Normalized; x*105m and y*68m; no reflection',
        speed_channel_semantics='s is recorded speed_kph, never height',
        ball_source='outputs/broadcast_tactical_ball_smoothed_v3/ball_homography_all_frames.csv')
    for key, text in updates.items():
        if key in values: values[key].text = text
        else:
            element = ET.SubElement(params, 'ProviderParameter')
            ET.SubElement(element, 'Name').text = key
            ET.SubElement(element, 'Value').text = text
    for spec in root.iter('DataFormatSpecification'):
        spec.set('startFrame', '1')
        spec.set('endFrame', str(N))
    tree.write(OUT / 'tracking_atd.xml', encoding='utf-8', xml_declaration=True)


def main():
    if OUT.exists(): raise FileExistsError(f'Refusing to overwrite {OUT}')
    OUT.mkdir(parents=True)
    (OUT / 'render_assets').mkdir()
    required = [BALL / 'ball_homography_all_frames.csv', BALL / 'BAR-ATH_broadcast_atd.txt',
        BALL / 'BAR-ATH_broadcast_atd.xml', TAC, TAC.with_suffix('.xml'), MAP, HFILE]
    hashes = {str(p.relative_to(ROOT)): digest(p) for p in required}
    d = pd.read_csv(required[0], float_precision='round_trip', low_memory=False)
    assert len(d) == N and np.array_equal(d.frame, np.arange(N))
    tree, people = roster()
    mapped, half, specs = alignment(199319)
    assert np.array_equal(d.tactical_frame.fillna(-1).to_numpy(int), mapped)
    assert np.array_equal(d.alignment_half.to_numpy(int), half)
    assert metadata(TAC.with_suffix('.xml'))['pitch'] == [105., 68.]
    values = np.lib.format.open_memmap(OUT / 'render_assets/tracking_values.npy', mode='w+',
        dtype=np.float64, shape=(N, 47, 3))
    values[:] = np.nan
    missing_players = b';'.join([b'NaN,NaN,NaN'] * 46)
    mapped_lines = 0
    with ((BALL / 'BAR-ATH_broadcast_atd.txt').open('rb') as balls, TAC.open('rb') as tactical,
          (OUT / 'tracking_atd.txt').open('wb') as dest):
        current_t = -1
        tline = None
        for frame in range(N):
            bline = next(balls)
            bparts = bline.rstrip(b'\r\n').split(b':')
            assert len(bparts) == 3 and int(bparts[0]) == frame + 1
            ball_payload = bparts[2]
            values[frame, 46] = np.fromstring(ball_payload.decode('ascii'), sep=',')
            if mapped[frame] >= 0:
                while current_t < mapped[frame]:
                    tline = next(tactical)
                    current_t += 1
                parts = tline.rstrip(b'\r\n').split(b':')
                assert len(parts) == 3 and int(parts[0]) == current_t + 1
                player_payload = parts[1]
                row = np.fromstring(player_payload.replace(b';', b',').decode('ascii'), sep=',')
                assert len(row) == 138
                values[frame, :46] = row.reshape(46, 3)
                mapped_lines += 1
            else: player_payload = missing_players
            dest.write(str(frame + 1).encode('ascii') + b':' + player_payload + b':' + ball_payload + b'\n')
        # Broadcast ATD has one non-video tail record. Do not invent a video frame.
        assert len(next(balls).rstrip(b'\r\n').split(b':')) == 3
        assert next(balls, None) is None
    values.flush()
    assert np.array_equal(values[:, 46, :2], d[['final_x', 'final_y']], equal_nan=True)
    assert np.array_equal(values[:, 46, 2], d.final_ball_s_kph, equal_nan=True)
    assert np.isnan(values[mapped < 0, :46]).all()
    valid_p = np.isfinite(values[:, :46, :2]).all(axis=2)
    speed_p = valid_p & np.isfinite(values[:, :46, 2])
    valid_b = np.isfinite(values[:, 46, :2]).all(axis=1)
    team_h = np.array([r['team_code'] == 'H' for r in people])
    table = pd.DataFrame(dict(frame=np.arange(N), atd_frame_id=np.arange(1, N + 1),
        broadcast_seconds=np.arange(N) / FPS, alignment_half=half, tactical_frame=mapped,
        tactical_atd_frame_id=np.where(mapped >= 0, mapped + 1, np.nan), players_time_aligned=mapped >= 0,
        player_position_count=valid_p.sum(axis=1), player_with_finite_speed_count=speed_p.sum(axis=1),
        barcelona_position_count=valid_p[:, team_h].sum(axis=1),
        athletic_position_count=valid_p[:, ~team_h].sum(axis=1),
        ball_position_available=valid_b, ball_source=d.final_source.to_numpy(),
        ball_is_prediction=d.is_prediction.to_numpy(bool),
        ball_world_live_reference=d.world_live_reference.to_numpy(bool),
        broadcast_highlight_inferred=d.highlight_inferred.to_numpy(bool),
        ball_x_norm=values[:, 46, 0], ball_y_norm=values[:, 46, 1],
        ball_x_m=values[:, 46, 0] * 105., ball_y_m=values[:, 46, 1] * 68., ball_s_kph=values[:, 46, 2]))
    extra = {}
    for i, person in enumerate(people):
        key = person['column_prefix']
        for suffix, data in [('x_norm', values[:, i, 0]), ('y_norm', values[:, i, 1]),
            ('x_m', values[:, i, 0] * 105), ('y_m', values[:, i, 1] * 68),
            ('s_kph', values[:, i, 2])]: extra[f'{key}_{suffix}'] = data
    table = pd.concat([table, pd.DataFrame(extra)], axis=1)
    table.to_csv(OUT / 'tracking_all_frames.csv', index=False)
    write_xml(tree, specs)
    save('roster.json', dict(players=people, ball=dict(array_slot=46, CSV_prefix='ball'),
        coordinate_units=dict(x_y='normalized', metres=[105, 68], s='kph'),
        track_labels_are_not_verified_shirt_numbers=True))
    save('tracking_manifest.json', dict(status='DATA_READY', dataset_name='BAR-ATH unified tracking V1',
        frames=N, fps=FPS, duration_seconds=N / FPS, timeline='Broadcast source frame zero-based',
        players='Unmodified Tactical ATD, all 46 XML slots',
        ball=str((BALL / 'ball_homography_all_frames.csv').relative_to(ROOT)),
        rejected_V4_used=False, new_smoothing_or_predictions=False,
        player_slots=46, team_slots=dict(Barcelona=int(team_h.sum()), Athletic_Club=int((~team_h).sum())),
        ball_valid_frames=int(valid_b.sum()), ball_missing_frames=int((~valid_b).sum()),
        aligned_player_frames=mapped_lines, unmapped_frames=int((mapped < 0).sum()),
        frames_with_any_player_xy=int(valid_p.any(axis=1).sum()),
        frames_with_both_ball_and_player_xy=int((valid_b & valid_p.any(axis=1)).sum()),
        preserved_existing_ball_prediction_frames=int((valid_b & d.is_prediction.to_numpy(bool)).sum()),
        alignment=specs, field_m=[105, 68], reflection='NONE',
        retained_ball_error_values_not_corrected_further=True,
        discarded_original_ATD_nonvideo_tail_records=1, all_video_frames_preserved=True,
        data_files=['tracking_all_frames.csv', 'tracking_atd.txt', 'tracking_atd.xml', 'roster.json'],
        video='unified_tracking.mp4', input_hashes=hashes))
    validate(values, d, mapped, half)
    (OUT / 'README.md').write_text('''# BAR–ATH 통합 트래킹 V1

현재 유지한 V3 공 좌표와 Tactical 선수 좌표를 중계영상 시간축 하나로 합쳤습니다.
폐기한 V4는 사용하지 않았고 공 좌표에 추가 평활화·보정·예측을 하지 않았습니다.

- `tracking_all_frames.csv`: 전체 164,487프레임의 공 + 선수 46개 트랙 슬롯. 정규화 XY와 105×68m 좌표, 속도(kph), 공 출처/기존 추정 여부, Tactical 대응 프레임을 포함합니다.
- `tracking_atd.txt` + `tracking_atd.xml`: 같은 통합 데이터의 ATD 형식. 선수 46개 XYZ가 아니라 XYs이며 마지막은 공 XYs입니다. s는 속도(kph), 높이가 아닙니다.
- `roster.json`: CSV 선수 열과 원본 Tactical XML 트랙·팀·골키퍼 슬롯의 대응입니다. 표시 번호는 확인된 유니폼 번호가 아닌 트래킹 슬롯입니다.
- `unified_tracking.mp4`: 단 하나의 전체 경기 영상. 왼쪽 중계영상, 오른쪽 통합 경기장 좌표입니다.

시간 정렬은 검증된 기존 대응을 그대로 사용했습니다. 전반 T=B+4014, 후반 T=B+26919이며 각각의 유효 범위만 적용합니다.
대응이 없는 8,564프레임에는 선수 좌표를 새로 넣지 않았습니다. 모든 결측은 그대로이며 직전 좌표를 유지하지 않습니다.
현재 공 좌표의 기존 오차·예측 출처·결측도 그대로입니다. Tactical 선수 필드와 V3 공 필드는 원본 토큰을 보존했습니다.
원본 Broadcast의 영상 밖 추가 ATD 1행은 이 영상 프레임 기준 데이터에서만 제외했으며 원본 파일은 유지했습니다.
XML half 범위는 정렬이 확인된 라이브 범위이며 0-based 시작 포함/끝 제외입니다.

하이라이트 중계 장면은 오른쪽 라이브 경기장 좌표와 시점이 다를 수 있습니다. 선수는 경기장 도식에 표시합니다.
원본 H가 유효하고 라이브 시점이 확인되는 경우에만 공 지면좌표를 중계영상에 재투영합니다. 직접 이미지 검출 위치나 공중 높이를 뜻하지 않습니다.
파랑 Barcelona, 빨강 Athletic Club, 흰 공은 유지한 공 좌표, 노란 공은 기존 추정 좌표입니다.
반투명 선수 표시는 원본 속도 s가 결측인 기록 위치입니다. 새로 선수 관측을 추정한 것이 아닙니다.

입력/원본 파일은 변경하지 않았습니다. 선수와 공의 실제 위치 정답 정확도를 새로 검증한 결과는 아닙니다.
''', encoding='utf-8')


def validate(values, ball, mapped, half):
    # Independently read the finished ATD and the two source payload streams.
    tac_frame = -1
    count = 0
    with (TAC.open('rb') as tac, (BALL / 'BAR-ATH_broadcast_atd.txt').open('rb') as bsrc,
          (OUT / 'tracking_atd.txt').open('rb') as exported):
        for frame, line in enumerate(exported):
            parts = line.rstrip(b'\r\n').split(b':')
            assert len(parts) == 3 and int(parts[0]) == frame + 1
            original_ball = next(bsrc).rstrip(b'\r\n').rsplit(b':', 1)[1]
            assert parts[2] == original_ball
            if mapped[frame] >= 0:
                while tac_frame < mapped[frame]:
                    original_tac = next(tac)
                    tac_frame += 1
                assert parts[1] == original_tac.rstrip(b'\r\n').split(b':')[1]
            raw = np.fromstring(parts[1].replace(b';', b',').decode('ascii'), sep=',').reshape(46, 3)
            assert np.array_equal(raw, values[frame, :46], equal_nan=True)
            assert np.array_equal(np.fromstring(parts[2].decode('ascii'), sep=','), values[frame, 46], equal_nan=True)
            count += 1
    assert count == N
    csv = pd.read_csv(OUT / 'tracking_all_frames.csv', float_precision='round_trip', low_memory=False)
    assert np.array_equal(csv.frame, np.arange(N)) and np.array_equal(csv.atd_frame_id, np.arange(1, N + 1))
    assert np.array_equal(csv[['ball_x_norm', 'ball_y_norm']], ball[['final_x', 'final_y']], equal_nan=True)
    assert np.array_equal(csv.ball_is_prediction, ball.is_prediction)
    assert np.array_equal(csv.ball_source, ball.final_source)
    for slot in range(46):
        prefix = f'player_{slot + 1:02d}'
        assert np.array_equal(csv[[prefix + '_x_norm', prefix + '_y_norm', prefix + '_s_kph']], values[:, slot], equal_nan=True)
        assert np.allclose(csv[[prefix + '_x_m', prefix + '_y_m']], values[:, slot, :2] * PITCH,
            rtol=0, atol=1e-12, equal_nan=True)
    assert np.isnan(values[mapped < 0, :46]).all()
    manifest = json.loads((OUT / 'tracking_manifest.json').read_text())
    assert all(digest(ROOT / p) == h for p, h in manifest['input_hashes'].items())
    assert metadata(OUT / 'tracking_atd.xml')['players'] == 46
    assert metadata(OUT / 'tracking_atd.xml')['pitch'] == [105, 68]
    xml = ET.parse(OUT / 'tracking_atd.xml').getroot()
    specs = list(xml.iter('DataFormatSpecification'))
    assert len(specs) == 1 and specs[0].get('startFrame') == '1' and specs[0].get('endFrame') == str(N)
    save('validation.json', dict(passed=True, frames=N, player_slots=46,
        all_ball_payloads_byte_identical_to_retained_V3=True,
        all_mapped_player_payloads_byte_identical_to_raw_Tactical=True,
        all_CSV_normalized_coordinates_and_speed_channels_exact=True,
        metre_units_105_by_68_verified=True, full_frame_mapping_verified=True,
        all_unmapped_player_coordinates_missing=True, no_added_predictions_or_smoothing=True,
        ball_source_prediction_and_missing_flags_preserved=True, original_files_unchanged=True,
        XML_player_slot_order_and_count_preserved=True, source_extra_nonvideo_ATD_row_not_duplicated=True))


if __name__ == '__main__': main()
