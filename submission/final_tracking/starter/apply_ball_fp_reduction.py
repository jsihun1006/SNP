"""Apply visually approved non-ball rejections while preserving every other SNP point."""
from pathlib import Path
import argparse
import hashlib
import json
import sys

import cv2
import numpy as np
import pandas as pd

ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT))
from starter.compare_broadcast_ball_coordinates import VIDEO,stamp,PITCH

OUT=ROOT/'data/derived/ball_fp_reduction_20261116'
STATE=['outside_export','missing','review_candidate','observed','interpolated']


def arrays(path):
    with np.load(path) as z:return {k:z[k] for k in z.files}


def approved_mask(records,n):
    mask=np.zeros(n,bool);reasons=np.full(n,'',dtype=object)
    for record in records:
        if 'frames' in record:frames=np.array(record['frames'],int)
        elif 'source_frame' in record:frames=np.array([record['source_frame']],int)
        else:frames=np.arange(record['start_frame'],record['end_frame']+1,dtype=int)
        assert ((frames>=0)&(frames<n)).all()
        mask[frames]=True;reasons[frames]=record['reason']
    return mask,reasons


def apply():
    d=arrays(OUT/'fp_inputs.npz');n=len(d['source_frame'])
    manifest=json.loads((OUT/'approved_rejections.json').read_text())
    if isinstance(manifest,dict) and 'source_input_sha256' in manifest:
        assert hashlib.sha256((OUT/'fp_inputs.npz').read_bytes()).hexdigest()==manifest['source_input_sha256']
    records=manifest['records'] if isinstance(manifest,dict) else manifest
    proposed,reasons=approved_mask(records,n)
    available=np.isfinite(d['pixel_xy']).all(axis=1)
    rejected=proposed&available
    before=d['pixel_xy'].copy();after=before.copy();pitch=d['pitch_xy'].copy()
    after[rejected]=np.nan;pitch[rejected]=np.nan
    scores=np.full(n,np.nan)
    if (OUT/'appearance_scores.csv').exists():
        ranked=pd.read_csv(OUT/'appearance_scores.csv')
        scores[ranked.source_frame.to_numpy(int)]=ranked.nonball_score
    decision=np.array(['outside_export','source_missing','source_review','kept_observation','kept_interpolated'],dtype=object)[d['status_codes']]
    decision[rejected]='rejected_nonball'
    reason=np.where(rejected,reasons,np.where(scores>=.85,'appearance_review_only; observation retained','source observation unchanged'))
    result=pd.DataFrame(dict(source_frame=d['source_frame'],half=d['half'],source_status=np.array(STATE)[d['status_codes']],
        source_pixel_x=before[:,0],source_pixel_y=before[:,1],source_pitch_x_m=d['pitch_xy'][:,0],source_pitch_y_m=d['pitch_xy'][:,1],
        pixel_x=after[:,0],pixel_y=after[:,1],pitch_x_m=pitch[:,0],pitch_y_m=pitch[:,1],
        pitch_x_normalized=pitch[:,0]/PITCH[0],pitch_y_normalized=pitch[:,1]/PITCH[1],decision=decision,
        reason=reason,nonball_score=scores,reviewed_rejection=rejected,source_confidence=d['confidence']))
    result.to_csv(OUT/'filtered_ball_coordinates.csv',index=False)
    base=ROOT/'data/derived/ball_tracking_homography_20261116/sources'
    for half,filename in [(1,'first_half_corrected_ball_coordinates.csv'),(2,'second_half_ball_object_audited.csv')]:
        original=pd.read_csv(base/filename,low_memory=False)
        frames=original.source_frame.to_numpy(int);drop=rejected[frames]
        original['fp_source_status']=original.status
        original['fp_reviewed_rejection']=drop
        original['fp_rejection_reason']=np.where(drop,reasons[frames],'')
        original['fp_nonball_score']=scores[frames]
        original.loc[drop,'status']='rejected_nonball'
        if half==1:
            for field in ['pixel_x','pixel_y','pitch_x_normalized','pitch_y_normalized','pitch_x_m','pitch_y_m']:
                original.loc[drop,field]=np.nan
        else:
            for field in ['image_x','image_y','bbox_x1','bbox_y1','bbox_x2','bbox_y2','detector_confidence']:
                original.loc[drop,field]=np.nan
            original['pitch_x_m']=pitch[frames,0];original['pitch_y_m']=pitch[frames,1]
            original['pitch_x_normalized']=pitch[frames,0]/PITCH[0];original['pitch_y_normalized']=pitch[frames,1]/PITCH[1]
        original.to_csv(OUT/f'half{half}_false_positive_reduced.csv',index=False)
    result[rejected].to_csv(OUT/'removed_observations.csv',index=False)
    result[(scores>=.85)&~rejected].to_csv(OUT/'unresolved_appearance_proposals.csv',index=False)
    assert np.array_equal(np.isnan(before[~rejected]),np.isnan(after[~rejected]))
    np.testing.assert_allclose(before[~rejected],after[~rejected],rtol=0,atol=0,equal_nan=True)
    summary=dict(frames=n,original_image_points=int(available.sum()),filtered_image_points=int(np.isfinite(after).all(1).sum()),
        newly_rejected_observations=int(rejected.sum()),rejected_observed=int((rejected&(d['status_codes']==3)).sum()),
        rejected_existing_interpolation=int((rejected&(d['status_codes']==4)).sum()),
        retained_observations=int(((d['status_codes']==3)&~rejected).sum()),
        original_pitch_points=int(np.isfinite(d['pitch_xy']).all(1).sum()),filtered_pitch_points=int(np.isfinite(pitch).all(1).sum()),
        per_half={str(h):dict(rejected=int((rejected&(d['half']==h)).sum()),
            retained_observed=int(((d['status_codes']==3)&~rejected&(d['half']==h)).sum())) for h in [1,2]},
        all_other_coordinates_exactly_preserved=True,new_observations=0,new_interpolation=0,
        homography_changed=False,atd_used_for_rejection=False,
        method='Appearance ranking followed by exact-observation/sequence visual approval; uncertain proposals retained',
        not_a_full_match_accuracy_benchmark=True)
    (OUT/'summary.json').write_text(json.dumps(summary,indent=2)+'\n')
    print(json.dumps(summary,indent=2))


def build_player():
    """Reuse the ball-only display: old/new here mean raw/filtered SNP, not ATD."""
    import base64
    source=arrays(OUT/'fp_inputs.npz');filtered=pd.read_csv(OUT/'filtered_ball_coordinates.csv',float_precision='round_trip')
    n=len(filtered);values=np.full((n,9),np.nan)
    values[:,:2]=source['pitch_xy'];values[:,2:4]=filtered[['pitch_x_m','pitch_y_m']]
    values[:,4:6]=source['pixel_xy'];values[:,6:8]=filtered[['pixel_x','pixel_y']]
    both=np.isfinite(values[:,:4]).all(1);values[both,8]=np.linalg.norm(values[both,:2]-values[both,2:4],axis=1)
    state=source['status_codes'].copy();state[filtered.reviewed_rejection]=5
    encode=lambda a:base64.b64encode(np.ascontiguousarray(a).tobytes()).decode()
    fields=['before_x_m','before_y_m','after_x_m','after_y_m','before_pixel_x','before_pixel_y','after_pixel_x','after_pixel_y','retained_distance_m']
    payload=dict(count=n,fps=25,fields=fields,values=encode(values.astype('<f8')),snp_state=encode(state.astype(np.uint8)),half=encode(source['half'].astype(np.uint8)))
    (OUT/'player_data.js').write_text('window.SNP_BALL_COMPARISON='+json.dumps(payload,separators=(',',':'))+';\n')
    template=(ROOT/'starter/ball_tracking_comparison_player.template.html').read_text()
    replacements={
        '전체 공 트래킹 좌표 비교':'SNP 공 오탐 감소 전후 비교',
        '기존 broadcast ATD':'보정 전 SNP','기존 ATD':'보정 전 SNP','SNP 공 트래킹':'오탐 감소 후 SNP',
        '원본 ATD 공은 청록색 원, SNP 공은 주황색 십자':'원본 SNP는 주황색 원, 오탐 감소 후 SNP는 녹색 십자',
        'ATD':'원본 SNP',
        "atd:'#00c8ef',snp:'#ffb323'":"atd:'#ffb323',snp:'#40d688'",
        "['export 밖','결측','검토 후보','관측','보간']":"['export 밖','결측','검토 후보','유지된 관측','기존 보간','확인된 비공 물체 → 보류']",
        '두 공의 거리':'유지된 좌표의 이동량',
        '<span class="cross">＋</span> SNP':'<span class="cross">＋</span> 오탐 감소 후 SNP',
        '경기장 좌표 · 105.3 × 68.0m':'공 좌표 유지 / 확인된 오탐 보류',
        '.dot { color: #00b8df;':'.dot { color: #ffb323;',
        '.cross { color: #e39318;':'.cross { color: #40d688;',
    }
    # Text-only ATD label replacements must not alter JavaScript identifiers (which are lowercase).
    for old,new in replacements.items():template=template.replace(old,new)
    template=template.replace('두 공의 거리: —','유지된 좌표의 이동량: —')
    (OUT/'player.html').write_text(template)
    print('Full-frame before/after player generated')


def render():
    table=pd.read_csv(OUT/'filtered_ball_coordinates.csv',float_precision='round_trip').set_index('source_frame')
    removed=table.index[table.reviewed_rejection].to_numpy()
    assert len(removed)
    starts=np.r_[0,np.flatnonzero(np.diff(removed)>50)+1];ends=np.r_[starts[1:],len(removed)]
    clips=[(max(0,int(removed[a])-20),min(len(table),int(removed[b-1])+21),'nonball rejection') for a,b in zip(starts,ends)]
    for frame in [33329,46249,104158,125032,145898,150694]:clips.append((frame-20,frame+21,'protected real ball retained'))
    cap=cv2.VideoCapture(str(VIDEO));writer=cv2.VideoWriter(str(OUT/'before_after_review.mp4'),cv2.CAP_FFMPEG,cv2.VideoWriter_fourcc(*'avc1'),25,(1280,420))
    assert writer.isOpened();snapshots=OUT/'before_after';snapshots.mkdir(exist_ok=True);count=0
    for first,last,label in clips:
        cap.set(cv2.CAP_PROP_POS_FRAMES,first)
        for frame in range(first,last):
            ok,image=cap.read();assert ok
            small=cv2.resize(image,(640,360),interpolation=cv2.INTER_AREA)
            canvas=np.full((420,1280,3),22,np.uint8);canvas[60:,:640]=small;canvas[60:,640:]=small
            row=table.loc[frame]
            for side,columns,color in [(0,['source_pixel_x','source_pixel_y'],(0,170,255)),(1,['pixel_x','pixel_y'],(90,230,80))]:
                point=row[columns].to_numpy(float)
                if np.isfinite(point).all():
                    p=np.rint(point/3).astype(int)
                    cv2.circle(canvas,(int(p[0])+640*side,int(p[1])+60),8,color,2,cv2.LINE_AA)
            action='REJECTED NON-BALL' if row.reviewed_rejection else row.decision
            for side,text in [(0,f'BEFORE SNP | f{frame} | {stamp(frame)}'),(1,f'AFTER | f{frame} | {action}')]:
                cv2.putText(canvas,text,(10+side*640,22),0,.46,(235,235,235),1,cv2.LINE_AA)
                cv2.putText(canvas,label,(10+side*640,45),0,.44,(210,210,210),1,cv2.LINE_AA)
            writer.write(canvas);count+=1
            if row.reviewed_rejection and (frame==first+20 or frame%25==0):cv2.imwrite(str(snapshots/f'frame_{frame}.jpg'),canvas)
    cap.release();writer.release()
    (OUT/'review_video_manifest.json').write_text(json.dumps(dict(frames=count,fps=25,duration_seconds=count/25,
        clips=[dict(first_frame=a,last_frame_exclusive=b,reason=c) for a,b,c in clips],all_rejected_frames_included=True),indent=2)+'\n')
    print(f'Review video: {count} frames / {count/25:.2f}s')


def main():
    parser=argparse.ArgumentParser(description=__doc__);parser.add_argument('stage',choices=['apply','player','render'])
    a=parser.parse_args()
    {'apply':apply,'player':build_player,'render':render}[a.stage]()


if __name__=='__main__':main()
