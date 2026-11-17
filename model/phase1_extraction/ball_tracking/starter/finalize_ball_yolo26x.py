"""Publish an X-only redetection result using verified completed detector work.

The request remains the selected uniform/penalty-mark frames across both halves.
Reuse is explicit: the detector has already completed these exact source frames.
The classifier is applied anew to the actual X boxes, rather than the old S boxes.
"""
from pathlib import Path
import argparse
import hashlib
import json
import shutil
import time

import cv2
import numpy as np
import pandas as pd
from PIL import Image,ImageDraw,ImageFont

ROOT=Path(__file__).resolve().parents[1]
AUDIT=ROOT/'data/derived/ball_yolo26_model_audit_20261116'
REQUEST=ROOT/'data/derived/ball_targeted_redetection_20261116'
OUT=ROOT/'data/derived/ball_yolo26x_redetection_20261116'
VIDEO=ROOT/'data/broadcast/BAR-ATH_broadcast-001.mp4'
KEY=['source_frame','candidate_index']
BOX=['bbox_x1','bbox_y1','bbox_x2','bbox_y2']
COLORS={'model_ball':(90,230,70),'model_nonball':(65,65,245),'uncertain':(30,225,255)}


def read(path):return pd.read_csv(path,float_precision='round_trip',low_memory=False)
def sha(path):return hashlib.sha256(Path(path).read_bytes()).hexdigest()
def write(name,data):(OUT/name).write_text(json.dumps(data,ensure_ascii=False,indent=2)+'\n')


def prepare():
    OUT.mkdir(exist_ok=True)
    source=AUDIT/'yolo26x_recheck';metadata=json.loads((source/'run_metadata.json').read_text())
    assert metadata['status']=='complete' and metadata['completed_frames']==11092
    request=metadata['request'];assert request['source']=='YOLO26x_FULL_REDETECT'
    assert request['manifest_sha256']==sha(REQUEST/'frames_to_redetect.csv')
    assert request['model_sha256']==sha(AUDIT/'models/yolo26x.pt')
    assert metadata['runtime']['actual_parameter_dtype']=='torch.float16'
    for a,b in [(source/'raw_ball_candidates.csv','raw_ball_candidates.csv'),
                (source/'frame_results.csv','frame_results.csv'),
                (source/'run_metadata.json','source_detection_metadata.json'),
                (source/'invalid_model_boxes.jsonl','invalid_model_boxes.jsonl'),
                (REQUEST/'frames_to_redetect.csv','frames_to_redetect.csv'),
                (REQUEST/'frame_intervals.csv','frame_intervals.csv')]:
        if (OUT/b).exists():assert sha(a)==sha(OUT/b)
        else:shutil.copyfile(a,OUT/b)
    raw=read(OUT/'raw_ball_candidates.csv');frames=read(OUT/'frame_results.csv');queue=read(OUT/'frames_to_redetect.csv')
    assert len(raw)==metadata['ball_candidates'] and np.array_equal(frames.source_frame,queue.source_frame)
    assert raw.source.eq('YOLO26x_FULL_REDETECT').all()
    write('request.json',dict(scope='Previously selected uniform/penalty-mark/boundary frames across both halves',
        requested_source_frames=11092,model='YOLO26x',detector_stage='reused_verified_complete',
        reused_detection_folder=str(source.relative_to(ROOT)),raw_detection_sha256=sha(source/'raw_ball_candidates.csv'),
        reused_source_frames=11092,new_detector_inference_required=0,
        candidate_verifier_stage='new scoring of every actual X box; no old S candidate substituted',
        raw_confidence=.05,positive_detection_threshold=.3,imgsz=1280,full_frame=True,roi=False,tiles=False,
        original_tracking_modified=False,homography_modified=False))
    print('Prepared verified X-only detections:',len(queue),'frames,',len(raw),'candidates',flush=True)


def finalize():
    raw=read(OUT/'raw_ball_candidates.csv').sort_values(KEY).reset_index(drop=True)
    scores=read(OUT/'verification_scores/scores.csv').sort_values(KEY).reset_index(drop=True)
    assert np.array_equal(raw[KEY],scores[KEY]);np.testing.assert_array_equal(raw[BOX],scores[BOX])
    parameters=json.loads((AUDIT/'decision_parameters.json').read_text())
    low=parameters['negative_binary_threshold'];high=parameters['positive_binary_threshold'];x=parameters['x_threshold']
    result=raw.copy();result['binary_ball_score']=scores.binary_ball_score
    result['verdict']=np.select([result.binary_ball_score<=low,(result.binary_ball_score>=high)&(result.confidence>=x)],
        ['model_nonball','model_ball'],default='uncertain')
    result['verdict_ko']=result.verdict.map(dict(model_ball='공 추정',model_nonball='비공 추정',uncertain='판단 유보'))
    result.to_csv(OUT/'all_candidates.csv',index=False)
    for value,name in [('model_ball','ball_candidates.csv'),('model_nonball','nonball_candidates.csv'),('uncertain','review_candidates.csv')]:
        result[result.verdict.eq(value)].to_csv(OUT/name,index=False)
    queue=read(OUT/'frames_to_redetect.csv')[['source_frame','video_seconds','half','interval_id']]
    counts=result.groupby(['source_frame','verdict']).size().unstack(fill_value=0).reindex(columns=list(COLORS),fill_value=0)
    queue=queue.merge(counts.reset_index(),on='source_frame',how='left').fillna(0)
    queue[list(COLORS)]=queue[list(COLORS)].astype(int);queue.to_csv(OUT/'frame_summary.csv',index=False)
    write('decision_parameters.json',parameters)
    write('summary.json',dict(status='complete',detector='YOLO26x',source_frames=len(queue),
        raw_candidates=len(raw),verdict_counts=result.verdict.value_counts().to_dict(),
        frames_without_candidate=int((queue[list(COLORS)].sum(axis=1)==0).sum()),
        detector_work_reused=True,all_actual_x_candidates_reverified=True,
        original_tracking_modified=False,homography_modified=False,
        scores_are_calibrated_probabilities=False,model_verdicts_are_not_ground_truth=True))
    print((OUT/'summary.json').read_text(),flush=True)


def render():
    data=read(OUT/'all_candidates.csv');queue=read(OUT/'frames_to_redetect.csv')
    with np.load(ROOT/'data/derived/ball_fp_reduction_20261116/fp_inputs.npz') as z:point=z['pixel_xy']
    groups={int(f):g.to_dict('records') for f,g in data.groupby('source_frame',sort=False)}
    header=Image.new('RGB',(1920,660),(21,24,30));draw=ImageDraw.Draw(header)
    font='/System/Library/Fonts/AppleSDGothicNeo.ttc';title=ImageFont.truetype(font,29);small=ImageFont.truetype(font,18)
    draw.text((18,8),'기존 SNP 트래킹 후보',font=title,fill=(255,179,35))
    draw.text((978,8),'YOLO26X 재탐지 · 후보 판정',font=title,fill=(90,230,70))
    draw.text((18,634),'초록: 공 추정   빨강 X: 비공 추정   노랑: 판단 유보 · 모델 판정이며 정답 확정 아님',font=small,fill=(238,238,238))
    template=cv2.cvtColor(np.array(header),cv2.COLOR_RGB2BGR)
    writer=cv2.VideoWriter(str(OUT/'yolo26x_comparison.mp4'),cv2.CAP_FFMPEG,cv2.VideoWriter_fourcc(*'avc1'),25,(1920,660));assert writer.isOpened()
    cap=cv2.VideoCapture(str(VIDEO));assert cap.isOpened();current=-1;start=time.monotonic();last=start
    pd.DataFrame(dict(output_frame=np.arange(len(queue)),source_frame=queue.source_frame)).to_csv(OUT/'video_frames.csv',index=False)
    snapshots=OUT/'snapshots';snapshots.mkdir(exist_ok=True)
    try:
        for i,row in enumerate(queue.itertuples()):
            f=int(row.source_frame)
            if current<0 or f-current>75:assert cap.set(cv2.CAP_PROP_POS_FRAMES,f);current=f-1
            while current<f:assert cap.grab();current+=1
            ok,image=cap.retrieve();assert ok
            image=cv2.resize(image,(960,540),interpolation=cv2.INTER_AREA)
            canvas=template.copy();canvas[90:630,:960]=image;canvas[90:630,960:]=image
            if np.isfinite(point[f]).all():cv2.circle(canvas,tuple(np.rint(point[f]*.5+[0,90]).astype(int)),10,(35,179,255),3,cv2.LINE_AA)
            rows=groups.get(f,[]);priority={'uncertain':0,'model_nonball':1,'model_ball':2}
            for r in sorted(rows,key=lambda r:priority[r['verdict']]):
                a=np.rint([r['bbox_x1']*.5+960,r['bbox_y1']*.5+90]).astype(int)
                b=np.rint([r['bbox_x2']*.5+960,r['bbox_y2']*.5+90]).astype(int);color=COLORS[r['verdict']]
                cv2.rectangle(canvas,tuple(a),tuple(b),color,3 if r['verdict']=='model_ball' else 1,cv2.LINE_AA)
                center=tuple(np.rint([(r['image_x']*.5+960),(r['image_y']*.5+90)]).astype(int))
                cv2.drawMarker(canvas,center,color,cv2.MARKER_TILTED_CROSS if r['verdict']=='model_nonball' else cv2.MARKER_CROSS,9,2,cv2.LINE_AA)
            counts={v:sum(r['verdict']==v for r in rows) for v in COLORS}
            clock=f'{f//90000:02d}:{f//1500%60:02d}:{f//25%60:02d}.{f%25*4:02d}'
            cv2.putText(canvas,f'SOURCE f{f} | {clock} | {int(row.half)}H',(18,72),0,.57,(238,238,238),1,cv2.LINE_AA)
            cv2.putText(canvas,f"SAME f{f} | BALL {counts['model_ball']} | NONBALL {counts['model_nonball']} | HOLD {counts['uncertain']}",(978,72),0,.57,(238,238,238),1,cv2.LINE_AA)
            writer.write(canvas)
            if f in [41901,69850,125032,126300]:cv2.imwrite(str(snapshots/f'frame_{f}.jpg'),canvas)
            if time.monotonic()-last>=30:
                print(json.dumps(dict(written_frames=i+1,total_frames=len(queue))),flush=True);last=time.monotonic()
    finally:cap.release();writer.release()
    write('video_manifest.json',dict(status='complete',frames=len(queue),fps=25,size=[1920,660],duration_seconds=len(queue)/25,
        all_requested_frames_included=True,source_mapping='video_frames.csv'))


def verify():
    queue=read(OUT/'frames_to_redetect.csv');data=read(OUT/'all_candidates.csv');raw=read(OUT/'raw_ball_candidates.csv')
    assert len(queue)==11092 and len(data)==len(raw)==34784 and not data.duplicated(KEY).any()
    np.testing.assert_array_equal(data[KEY+BOX+['image_x','image_y','confidence']],raw.sort_values(KEY)[KEY+BOX+['image_x','image_y','confidence']])
    parameters=json.loads((OUT/'decision_parameters.json').read_text())
    expected=np.select([data.binary_ball_score<=parameters['negative_binary_threshold'],
        (data.binary_ball_score>=parameters['positive_binary_threshold'])&(data.confidence>=parameters['x_threshold'])],['model_nonball','model_ball'],default='uncertain')
    np.testing.assert_array_equal(data.verdict,expected)
    scores=read(OUT/'verification_scores/scores.csv').sort_values(KEY).reset_index(drop=True)
    np.testing.assert_array_equal(data[KEY+BOX],scores[KEY+BOX])
    np.testing.assert_array_equal(data.binary_ball_score,scores.binary_ball_score)
    metadata=json.loads((OUT/'verification_scores/run_metadata.json').read_text())
    assert metadata['status']=='complete' and metadata['completed_candidates']==len(raw)
    assert metadata['request']['source_candidates_sha256']==sha(OUT/'raw_ball_candidates.csv')
    assert metadata['request']['model_sha256']==sha(AUDIT/'models/yolo26s-cls.pt')
    assert metadata['request']['head_sha256']==sha(AUDIT/'binary_probe/binary_head.npz')
    mapping=read(OUT/'video_frames.csv');np.testing.assert_array_equal(mapping.source_frame,queue.source_frame)
    cap=cv2.VideoCapture(str(OUT/'yolo26x_comparison.mp4'));assert cap.isOpened()
    assert cap.get(5)==25 and cap.get(3)==1920 and cap.get(4)==660;count=0
    wanted=set(range(0,11092,200));wanted.add(11091);sampled={}
    while True:
        ok,image=cap.read()
        if not ok:break
        assert image.shape==(660,1920,3)
        if count in wanted:sampled[count]=cv2.resize(image[90:630,:960],(48,27),interpolation=cv2.INTER_AREA)
        count+=1
    cap.release();assert count==11092
    original=cv2.VideoCapture(str(VIDEO));errors=[]
    for i,actual in sampled.items():
        assert original.set(cv2.CAP_PROP_POS_FRAMES,int(mapping.source_frame.iloc[i]))
        ok,image=original.read();assert ok
        expected=cv2.resize(image,(48,27),interpolation=cv2.INTER_AREA)
        error=float(np.abs(actual.astype(float)-expected.astype(float)).mean());assert error<12.,(i,error)
        errors.append(error)
    original.release()
    integrity=json.loads((ROOT/'data/derived/ball_fp_reduction_20261116/source_integrity.json').read_text())
    assert all(sha(ROOT/name)==value for name,value in integrity.items())
    write('verification.json',dict(passed=True,source_frames=11092,actual_x_candidates=34784,video_frames_decoded=count,
        all_coordinates_preserved=True,all_verdicts_verified=True,source_video_samples=len(errors),
        maximum_source_background_mae=max(errors),original_sources_and_homography_unchanged=True))
    (OUT/'README.md').write_text('# YOLO26X 재탐지 완료\n\n선별한 전·후반 11,092프레임의 YOLO26X 전수 실행 결과를 재사용하고, 실제 X 후보 34,784개를 다시 판별했습니다. 이전 S 후보를 대신 사용하지 않았습니다.\n\n- [비교 영상 · 7분 24초](yolo26x_comparison.mp4)\n- [전체 X 후보와 판정](all_candidates.csv)\n- [공 추정 후보](ball_candidates.csv)\n- [비공 추정 후보](nonball_candidates.csv)\n- [판단 유보 후보](review_candidates.csv)\n\n기존 동일 설정의 완료된 X 검출을 재사용한 사실은 request.json과 source_detection_metadata.json에 기록했습니다. 전체 164,487프레임 중 기존에 지정한 유니폼·페널티마크·흰선 구간만 대상입니다. 최종 단일 공 궤적이나 인플레이 상태를 자동 확정하지 않았고 원본 좌표·homography를 변경하지 않았습니다.\n')
    print(json.dumps(json.loads((OUT/'summary.json').read_text()),ensure_ascii=False),flush=True)


def main():
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('stage',choices=['prepare','finalize','render','verify'])
    a=p.parse_args();globals()[a.stage]()


if __name__=='__main__':main()
