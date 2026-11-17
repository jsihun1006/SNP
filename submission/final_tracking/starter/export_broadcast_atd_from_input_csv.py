"""Transfer the specified full input track into a NEW broadcast ATD TXT.

All usable CSV pitch coordinates are authoritative for this export, including
the CSV's existing SNP observations and predictions, with provenance retained.
No new ball detection, ground/air filtering or prediction is performed.
Only ball x/y text tokens can change; player fields and ball speed stay exact.
"""
from pathlib import Path
import hashlib
import json
import shutil
import xml.etree.ElementTree as ET
import numpy as np
import pandas as pd

ROOT=Path(__file__).resolve().parents[1]
CSV=ROOT/'data/derived/ball_temporal_ai_20261117/ball_homography_all_frames.csv'
RAW=ROOT/'data/broadcast/BAR-ATH_broadcast_atd.txt'
META=RAW.with_suffix('.xml')
HFILE=RAW.with_name('BAR-ATH_broadcast_homography.csv')
OUT=ROOT/'outputs/broadcast_atd_from_csv_v1'
TARGET=OUT/RAW.name
PITCH=np.array([105.,68.])


def digest(path):
    result=hashlib.sha256()
    with Path(path).open('rb') as stream:
        for block in iter(lambda:stream.read(8*1024*1024),b''):result.update(block)
    return result.hexdigest()


def clean(value):
    if isinstance(value,dict):return {str(k):clean(v) for k,v in value.items()}
    if isinstance(value,(tuple,list)):return [clean(v) for v in value]
    if isinstance(value,np.ndarray):return clean(value.tolist())
    if isinstance(value,np.integer):return int(value)
    if isinstance(value,np.bool_):return bool(value)
    if isinstance(value,(float,np.floating)):return float(value) if np.isfinite(value) else None
    return value


def save(name,data):(OUT/name).write_text(json.dumps(clean(data),ensure_ascii=False,indent=2,allow_nan=False)+'\n',encoding='utf-8')


def project(h,xy):
    q=np.einsum('nij,nj->ni',h,np.c_[xy,np.ones(len(xy))])
    with np.errstate(divide='ignore',invalid='ignore'):return q[:,:2]/q[:,2:]


def patch(line,xy):
    body=line.rstrip(b'\r\n');ending=line[len(body):]
    prefix,ball=body.rsplit(b':',1);tokens=ball.split(b',');assert len(tokens)==3
    tokens[:2]=[format(float(value),'.17g').encode('ascii') for value in xy]
    return prefix+b':'+b','.join(tokens)+ending


def main():
    OUT.mkdir(parents=True,exist_ok=True)
    if TARGET.exists():raise FileExistsError('Refusing to overwrite an existing export: '+str(TARGET))
    signatures={str(p.relative_to(ROOT)):digest(p) for p in [CSV,RAW,META,HFILE]}
    save('source_hashes.json',signatures)
    source=pd.read_csv(CSV,float_precision='round_trip',low_memory=False)
    n=len(source);assert n==164487 and np.array_equal(source.source_frame,np.arange(n))
    assert np.allclose(source.video_seconds,np.arange(n)/25)
    normalized=source[['pitch_x_normalized','pitch_y_normalized']].to_numpy(float)
    legacy=source[['pitch_x_m','pitch_y_m']].to_numpy(float)
    eligible=np.isfinite(normalized).all(axis=1)&np.isfinite(legacy).all(axis=1)
    eligible&=source.pitch_usable.to_numpy(bool)&source.homography_valid.to_numpy(bool)&source.ball_position_available.to_numpy(bool)
    assert np.array_equal(eligible,np.isfinite(legacy).all(axis=1)),'Finite source coordinates unexpectedly fail source validity flags'
    assert np.allclose(normalized[eligible],legacy[eligible]/[105.3,68.],atol=1e-14)
    tree=ET.parse(META).getroot();slots=len(list(tree.iter('Player')))
    assert float(tree.findtext('.//FrameRate'))==25
    assert [float(tree.findtext('.//FieldSize/Width')),float(tree.findtext('.//FieldSize/Height'))]==[105.,68.]
    speed_channel=next(channel for channel in tree.iter('Channel') if channel.get('id')=='s')
    assert speed_channel.findtext('Unit')=='kph'
    htable=pd.read_csv(HFILE,float_precision='round_trip',low_memory=False)
    assert len(htable)==n+1 and np.array_equal(htable.frame,np.arange(n+1))
    h=htable[[f'h{i}{j}' for i in range(3) for j in range(3)]].to_numpy(float).reshape(-1,3,3)
    ids=np.flatnonzero(eligible)
    assert np.isfinite(h[ids]).all() and htable.valid.iloc[ids].eq(1).all()
    image=source.loc[eligible,['image_x','image_y']].to_numpy(float)
    roundtrip=np.linalg.norm(project(h[ids],normalized[eligible])-image,axis=1)
    assert np.isfinite(roundtrip).all() and np.max(roundtrip)<1e-6
    rows=[];byte_counts=dict(lines=0,player_prefix_unchanged=True,ball_s_tokens_unchanged=True,
        untouched_rows_byte_identical=True,line_endings_unchanged=True,frame_ids_unchanged=True)
    missing_before=0;missing_after=0;changed_count=0;filled=0;replaced_existing=0
    with RAW.open('rb') as input_stream,TARGET.open('wb') as output_stream:
        for frame,line in enumerate(input_stream):
            body=line.rstrip(b'\r\n');parts=body.split(b':');assert len(parts)==3 and int(parts[0])==frame+1
            assert len(parts[1].split(b';'))==slots
            values=parts[2].split(b',');assert len(values)==3
            original=np.array([float(v) for v in values],float);original_valid=np.isfinite(original[:2]).all()
            use_csv=frame<n and bool(eligible[frame])
            final=normalized[frame].copy() if use_csv else original[:2].copy()
            changed=use_csv and not np.array_equal(original[:2],final,equal_nan=True)
            output_line=patch(line,final) if changed else line
            output_stream.write(output_line)
            new_body=output_line.rstrip(b'\r\n');new_prefix,new_ball=new_body.rsplit(b':',1);new_values=new_ball.split(b',')
            original_prefix=body.rsplit(b':',1)[0]
            byte_counts['player_prefix_unchanged']&=new_prefix==original_prefix
            byte_counts['ball_s_tokens_unchanged']&=new_values[2]==values[2]
            byte_counts['line_endings_unchanged']&=output_line[len(new_body):]==line[len(body):]
            byte_counts['frame_ids_unchanged']&=new_prefix.split(b':',1)[0]==parts[0]
            if not changed:byte_counts['untouched_rows_byte_identical']&=output_line==line
            assert np.array_equal(np.array([float(v) for v in new_values[:2]]),final,equal_nan=True)
            byte_counts['lines']+=1;changed_count+=int(changed)
            missing_before+=int(not original_valid);missing_after+=int(not np.isfinite(final).all())
            filled+=int(use_csv and not original_valid);replaced_existing+=int(changed and original_valid)
            origin=source.detection_origin.iloc[frame] if frame<n else ''
            predicted=bool(source.is_prediction.iloc[frame]) if frame<n else False
            if use_csv:
                kind='EXISTING_CSV_PREDICTION' if predicted else ('CSV_ORIGINAL_SNP_OBSERVATION' if origin=='original_snp' else 'CSV_YOLO26X_OBSERVATION')
                reason='APPLY_VALID_COORDINATE_FROM_SPECIFIED_CSV'
            else:
                kind='ORIGINAL_BROADCAST_ATD';reason='CSV_POSITION_MISSING_KEEP_ORIGINAL_ATD' if frame<n else 'EXTRA_ATD_ROW_OUTSIDE_CSV_KEEP_ORIGINAL'
            rows.append(dict(frame=frame,atd_frame_id=frame+1,csv_source_frame=frame if frame<n else np.nan,
                csv_coordinate_used=use_csv,coordinate_changed=changed,source_kind=kind,source_is_prediction=predicted if use_csv else False,
                csv_detection_origin=origin if frame<n else '',original_atd_valid=original_valid,
                original_ball_x=original[0],original_ball_y=original[1],original_ball_s=original[2],
                csv_ball_x=normalized[frame,0] if frame<n else np.nan,csv_ball_y=normalized[frame,1] if frame<n else np.nan,
                corrected_ball_x=final[0],corrected_ball_y=final[1],corrected_ball_s=original[2],
                original_ball_x_m=original[0]*105,original_ball_y_m=original[1]*68,
                corrected_ball_x_m=final[0]*105,corrected_ball_y_m=final[1]*68,
                replacement_reason=reason,coordinate_units='normalized',speed_units='kph_preserved'))
    assert len(rows)==164488 and byte_counts['lines']==len(rows)
    assert all(value for key,value in byte_counts.items() if key!='lines')
    result=pd.DataFrame(rows);result.to_csv(OUT/'corrected_ball_coordinates.csv',index=False)
    result[['frame','atd_frame_id','csv_coordinate_used','coordinate_changed','source_kind','source_is_prediction',
        'csv_detection_origin','original_atd_valid','replacement_reason']].to_csv(OUT/'replacement_log.csv',index=False)
    # An identical metadata copy is convenient for existing MTAH parsers.
    shutil.copyfile(META,OUT/META.name);assert digest(OUT/META.name)==signatures[str(META.relative_to(ROOT))]
    from starter.tracking_utils import read_tracking,track_teams
    teams=track_teams(META)
    before=read_tracking(RAW,teams,len(result),0,len(result))
    after=read_tracking(TARGET,teams,len(result),0,len(result))
    players_preserved=np.array_equal(before[:,:-1],after[:,:-1],equal_nan=True)
    assert players_preserved
    expected=result[['corrected_ball_x','corrected_ball_y']].to_numpy(float)
    assert np.allclose(after[:,-1],expected,equal_nan=True,atol=6e-8)
    assert np.array_equal(np.isfinite(expected).all(axis=1),
        result.original_atd_valid.to_numpy(bool)|result.csv_coordinate_used.to_numpy(bool))
    del before,after
    changed_sources=[relative for relative,signature in signatures.items() if digest(ROOT/relative)!=signature]
    assert not changed_sources
    source_used=result.csv_coordinate_used.to_numpy(bool)
    assert np.array_equal(expected[:n][eligible],normalized[eligible])
    assert np.array_equal(result.loc[~source_used,['original_ball_x','original_ball_y']].to_numpy(float),expected[~source_used],equal_nan=True)
    summary=dict(status='COMPLETE',source_csv=str(CSV.relative_to(ROOT)),source_atd=str(RAW.relative_to(ROOT)),
        output_atd=str(TARGET.relative_to(ROOT)),csv_frames=n,atd_frames=len(result),players_per_frame=slots,fps=25,
        csv_coordinate_frames_applied=int(eligible.sum()),numerically_changed_xy_frames=changed_count,
        existing_atd_xy_changed_frames=replaced_existing,atd_missing_frames_filled_from_csv=filled,
        final_valid_ball_coordinate_frames=int(np.isfinite(expected).all(axis=1).sum()),
        original_missing_ball_coordinate_frames=missing_before,final_missing_ball_coordinate_frames=missing_after,
        csv_existing_prediction_frames_applied=int(result.loc[source_used,'source_is_prediction'].sum()),
        applied_source_counts=result.loc[source_used,'source_kind'].value_counts().to_dict(),
        original_atd_retained_frames=int((~source_used).sum()),extra_atd_rows_preserved=1,
        normalized_xy_transferred_directly=True,source_csv_pitch_m=[105.3,68.],atd_pitch_m=[105.,68.],
        common_meter_conversion='CSV x_m*105/105.3; y_m unchanged. TXT stores normalized xy.',
        original_speed_s_kph_preserved=True,player_coordinates_preserved=True,
        original_atd_modified=False,original_homography_modified=False,original_csv_modified=False,
        source_csv_SNP_not_relabelled_as_YOLO=True,new_prediction_or_airborne_classification=False,
        existing_csv_predictions_copied_as_provided=True,actual_position_accuracy_improvement_proven=False,
        maximum_CSV_H_roundtrip_error_px=float(np.max(roundtrip)))
    save('summary.json',summary)
    save('verification.json',dict(passed=True,txt_byte_contract=byte_counts,
        original_source_hashes_preserved=True,source_hash_mismatches=changed_sources,
        existing_MTAH_parser_accepts=True,all_player_tracks_equal=True,
        every_valid_CSV_coordinate_applied=True,unavailable_CSV_rows_preserve_original=True,
        no_original_finite_coordinate_lost=True,metadata_copy_identical=True,
        normalized_and_105_3_to_105_conversion_verified=True,actual_position_accuracy_tested=False,
        output_sha256=digest(TARGET)))
    (OUT/'README.md').write_text(f'''# 지정 CSV 기준 Broadcast ATD 공 x/y 보정본

[보정 ATD TXT]({TARGET.name}) · [좌표 및 출처](corrected_ball_coordinates.csv) · [검증](verification.json)

지정한 `{CSV.relative_to(ROOT)}`의 사용 가능한 경기장 좌표 **{int(eligible.sum()):,}프레임**을 적용했습니다.
수치가 변경된 공 x/y는 **{changed_count:,}프레임**이며, 이 중 원본 ATD 결측을 채운 프레임은 **{filled:,}개**입니다.
CSV 좌표가 없는 프레임은 원래 ATD를 유지했습니다. 원본 ATD의 164,488행과 마지막 추가 1행을 보존했습니다.

TXT의 x/y는 정규화 좌표입니다. CSV의 `pitch_x_normalized/pitch_y_normalized`를 직접 옮겼습니다.
CSV의 105.3×68m 좌표를 105×68m ATD로 옮길 때 생기는 X 크기 차이도 검증했습니다.
선수 좌표·선수 속도·공 속도(s, kph)·프레임 ID·구분자·줄끝·결측 표현과, 교체하지 않는 모든 행을 원문대로 유지했습니다.
원본 영상·ATD·XML·Homography·지정 CSV는 변경하지 않았습니다. 같은 구조의 XML을 이 폴더에 동일하게 복사했습니다.

이번에는 GROUND/공중볼 판정이나 순수 YOLO 한정 필터를 새로 적용하지 않았습니다.
**지정 CSV 전체를 기준**으로, 원래 SNP·YOLO 관측과 CSV에 이미 있던 예측 **{summary['csv_existing_prediction_frames_applied']:,}프레임**도 출처를 구분하여 옮겼습니다.
예측을 새로 생성하거나 SNP를 실제 YOLO 검출로 표시하지 않았습니다.
경기장 밖의 유효한 CSV 좌표도 임의로 잘라내지 않았습니다. 공중볼의 실제 높이/지면 위치 정답이나 정확도 개선을 주장하지 않습니다.

`corrected_ball_coordinates.csv`의 기본 x/y는 정규화 값이고 `_m` 열은 105×68m 미터 좌표입니다.
`replacement_log.csv`에서 프레임별 반영 여부와 SNP/YOLO/기존 예측 출처를 확인할 수 있습니다.
원본과 결과를 실제 MTAH 파서로 다시 읽어 모든 선수 좌표와 프레임·채널 구조를 검증했습니다.
''',encoding='utf-8')
    save('artifact_integrity.json',dict(algorithm='sha256',files={p.name:digest(p) for p in sorted(OUT.iterdir()) if p.is_file() and p.name!='artifact_integrity.json'},
        exporter_source_sha256=digest(Path(__file__))))
    print(json.dumps(clean(summary),ensure_ascii=False),flush=True)


if __name__=='__main__':main()
