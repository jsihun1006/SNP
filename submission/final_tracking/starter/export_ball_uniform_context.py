"""Export pose-assisted non-ball suppression without changing retained coordinates."""
from pathlib import Path
import argparse
import base64
import json
import sys

import cv2
import numpy as np
import pandas as pd

ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT))
from starter.ball_uniform_context import BASE,OUT,VIDEO,arrays
from starter.compare_broadcast_ball_coordinates import stamp
PITCH=np.array([105.3,68.])


def apply():
    source=arrays(BASE/'fp_inputs.npz')
    old=pd.read_csv(BASE/'filtered_ball_coordinates.csv',float_precision='round_trip')
    context=pd.read_csv(OUT/'context_decisions.csv',float_precision='round_trip')
    assert np.array_equal(old.source_frame,context.source_frame)
    manual=old.reviewed_rejection.to_numpy(bool);automatic=context.model_rejected.to_numpy(bool)
    rejected=manual|automatic;new=automatic&~manual
    result=old.copy()
    result['model_rejected']=automatic;result['rejected']=rejected
    result['rejection_method']=np.select([manual&automatic,manual,automatic],
        ['manual_v1_and_pose_uniform','manual_v1','pose_uniform'],default='retained')
    for name in ['pixel_x','pixel_y','pitch_x_m','pitch_y_m','pitch_x_normalized','pitch_y_normalized']:
        result.loc[rejected,name]=np.nan
    result.loc[rejected,'decision']='rejected_nonball'
    result.loc[new,'reason']=context.loc[new,'reason']
    result.loc[manual&automatic,'reason']=result.loc[manual&automatic,'reason']+'; confirmed by pretrained pose and temporal attachment'
    for name in ['pose_inferred','torso_u','torso_v','temporal_support','temporal_span_frames','temporal_uv_dispersion']:
        result[name]=context[name]
    result.to_csv(OUT/'filtered_ball_coordinates.csv',index=False)
    result.loc[new].to_csv(OUT/'new_uniform_rejections.csv',index=False)
    result.loc[rejected].to_csv(OUT/'removed_observations.csv',index=False)
    for half in [1,2]:
        compatible=pd.read_csv(BASE/f'half{half}_false_positive_reduced.csv',low_memory=False,float_precision='round_trip')
        frames=compatible.source_frame.to_numpy(int);drop=automatic[frames]
        compatible['fp_model_rejection']=drop;compatible['fp_rejected']=rejected[frames]
        compatible.loc[drop,'status']='rejected_nonball'
        compatible.loc[drop,'fp_rejection_reason']=context.loc[frames[drop],'reason'].to_numpy()
        columns=['pixel_x','pixel_y'] if half==1 else ['image_x','image_y','bbox_x1','bbox_y1','bbox_x2','bbox_y2','detector_confidence']
        columns+=['pitch_x_m','pitch_y_m','pitch_x_normalized','pitch_y_normalized']
        compatible.loc[drop,columns]=np.nan
        compatible.to_csv(OUT/f'half{half}_false_positive_reduced.csv',index=False)
    kept=~rejected
    np.testing.assert_allclose(result.loc[kept,['pixel_x','pixel_y']],source['pixel_xy'][kept],rtol=0,atol=0,equal_nan=True)
    np.testing.assert_allclose(result.loc[kept,['pitch_x_m','pitch_y_m']],source['pitch_xy'][kept],rtol=0,atol=0,equal_nan=True)
    summary=dict(frames=len(result),original_observations=int((source['status_codes']==3).sum()),
        original_image_points=int(np.isfinite(source['pixel_xy']).all(1).sum()),
        previous_manual_rejections=int(manual.sum()),automatic_uniform_rejections=int(automatic.sum()),
        overlapping_manual_and_model=int((manual&automatic).sum()),new_uniform_rejections=int(new.sum()),
        total_rejections=int(rejected.sum()),remaining_image_points=int(np.isfinite(result[['pixel_x','pixel_y']]).all(1).sum()),
        original_pitch_points=int(np.isfinite(source['pitch_xy']).all(1).sum()),
        remaining_pitch_points=int(np.isfinite(result[['pitch_x_m','pitch_y_m']]).all(1).sum()),
        new_uniform_rejections_with_pitch=int((new&np.isfinite(source['pitch_xy']).all(1)).sum()),
        per_half={str(h):dict(new_uniform_rejections=int((new&(source['half']==h)).sum()),
            total_rejections=int((rejected&(source['half']==h)).sum())) for h in [1,2]},
        all_retained_coordinates_exactly_preserved=True,new_interpolation=0,new_ball_coordinates=0,
        homography_changed=False,atd_used=False,
        limitation='Missing points remain missing. This suppresses jersey-like false observations; it does not reacquire the true ball or correct aerial-ball ground projection.')
    (OUT/'summary.json').write_text(json.dumps(summary,indent=2)+'\n');print(json.dumps(summary,indent=2))


def player():
    source=arrays(BASE/'fp_inputs.npz');evidence=arrays(OUT/'context_evidence.npz')
    table=pd.read_csv(OUT/'filtered_ball_coordinates.csv',float_precision='round_trip')
    n=len(table);values=np.full((n,9),np.nan)
    values[:,:2]=source['pitch_xy'];values[:,2:4]=table[['pitch_x_m','pitch_y_m']]
    values[:,4:6]=source['pixel_xy'];values[:,6:8]=table[['pixel_x','pixel_y']]
    both=np.isfinite(values[:,:4]).all(1);values[both,8]=np.linalg.norm(values[both,:2]-values[both,2:4],axis=1)
    state=source['status_codes'].copy();state[table.reviewed_rejection]=5;state[table.model_rejected]=6
    encode=lambda a:base64.b64encode(np.ascontiguousarray(a).tobytes()).decode()
    torso={str(f):np.round(evidence['torso_polygon'][f],2).tolist()
        for f in np.flatnonzero(np.isfinite(evidence['torso_uv']).all(1))}
    fields=['before_x_m','before_y_m','after_x_m','after_y_m','before_pixel_x','before_pixel_y','after_pixel_x','after_pixel_y','retained_distance_m']
    payload=dict(count=n,fps=25,fields=fields,values=encode(values.astype('<f8')),
        snp_state=encode(state.astype(np.uint8)),half=encode(source['half'].astype(np.uint8)),torso=torso)
    (OUT/'player_data.js').write_text('window.SNP_BALL_COMPARISON='+json.dumps(payload,separators=(',',':'))+';\n')
    template=(BASE/'player.html').read_text().replace('SNP 공 오탐 감소 전후 비교','사람 포즈 모델 적용 · 공 오탐 감소 전후')
    template=template.replace("['export 밖','결측','검토 후보','유지된 관측','기존 보간','확인된 비공 물체 → 보류']",
        "['export 밖','결측','검토 후보','유지된 관측','기존 보간','기존 수동 검수 → 보류','포즈·외형·시간 근거 → 유니폼·신체 오탐 보류']")
    template=template.replace('영상 공 표시</label>','영상 공 표시</label><label><input id="show-torso" type="checkbox" checked> 모델이 찾은 몸통</label>')
    template=template.replace("overlay.clearRect(0,0,960,540);", """overlay.clearRect(0,0,960,540);
        const polygon=data.torso[String(frame)];
        if(polygon&&byId('show-torso').checked){
          overlay.save();overlay.strokeStyle='#df75ff';overlay.lineWidth=2;overlay.beginPath();
          polygon.forEach((p,i)=>{if(i)overlay.lineTo(p[0]/2,p[1]/2);else overlay.moveTo(p[0]/2,p[1]/2)});
          overlay.closePath();overlay.stroke();overlay.restore();
        }""")
    template=template.replace("render(0);", "byId('show-torso').addEventListener('change',()=>render(frame));\n      render(0);")
    template=template.replace('영상의 두 표시는 같은 프레임의 공 위치입니다. 영상에 경기장 선은 투영하지 않습니다.',
        '주황색 원은 기존 검출, 녹색 십자는 유지된 공입니다. 보라색 사각형은 모델이 찾은 몸통입니다. 몸통과 겹쳐도 외형·시간 근거가 부족하면 공을 유지합니다.')
    cases=[(41901,'새로 걸러낸 선수 로고'),(13716,'새로 걸러낸 골키퍼 로고'),(33329,'공중볼 유지'),(30442,'손에 든 공 유지'),(125032,'경합 공 유지')]
    for f,label in cases:
        template=template.replace('</select></label>\n    <label><input id="show-points"',f'<option value="{f}">{label}</option></select></label>\n    <label><input id="show-points"',1)
    (OUT/'player.html').write_text(template)
    print('Full-frame player generated')


def render():
    table=pd.read_csv(OUT/'filtered_ball_coordinates.csv',float_precision='round_trip')
    evidence=arrays(OUT/'context_evidence.npz');removed=np.flatnonzero(table.rejected)
    starts=np.r_[0,np.flatnonzero(np.diff(removed)>30)+1];ends=np.r_[starts[1:],len(removed)]
    clips=[(max(0,int(removed[a])-12),min(len(table),int(removed[b-1])+13),'non-ball suppressed') for a,b in zip(starts,ends)]
    for frame in [33329,30442,104158,125032,145898,150694]:clips.append((frame-15,frame+16,'real ball retained'))
    cap=cv2.VideoCapture(str(VIDEO));writer=cv2.VideoWriter(str(OUT/'before_after_model_review.mp4'),cv2.CAP_FFMPEG,cv2.VideoWriter_fourcc(*'avc1'),25,(1280,420))
    assert writer.isOpened();count=0;snaps=OUT/'before_after';snaps.mkdir(exist_ok=True)
    for first,last,label in clips:
        cap.set(cv2.CAP_PROP_POS_FRAMES,first)
        for frame in range(first,last):
            ok,img=cap.read();assert ok
            small=cv2.resize(img,(640,360),interpolation=cv2.INTER_AREA)
            canvas=np.full((420,1280,3),22,np.uint8);canvas[60:,:640]=small;canvas[60:,640:]=small
            row=table.iloc[frame]
            poly=evidence['torso_polygon'][frame]
            if np.isfinite(poly).all():
                poly=np.rint(poly/3+[640,60]).astype(np.int32)
                cv2.polylines(canvas,[poly],True,(255,110,230),1,cv2.LINE_AA)
            for side,columns,color in [(0,['source_pixel_x','source_pixel_y'],(0,170,255)),(1,['pixel_x','pixel_y'],(90,230,80))]:
                point=row[columns].to_numpy(float)
                if np.isfinite(point).all():
                    p=np.rint(point/3).astype(int);cv2.circle(canvas,(int(p[0])+640*side,int(p[1])+60),8,color,2,cv2.LINE_AA)
            action='MODEL: NONBALL REMOVED' if row.model_rejected else ('MANUAL V1: NONBALL REMOVED' if row.reviewed_rejection else 'OBSERVATION RETAINED')
            for side,text in [(0,f'RAW SNP | f{frame} | {stamp(frame)}'),(1,f'AFTER | {action}')]:
                cv2.putText(canvas,text,(10+side*640,22),0,.46,(235,235,235),1,cv2.LINE_AA)
                cv2.putText(canvas,label,(10+side*640,45),0,.44,(210,210,210),1,cv2.LINE_AA)
            writer.write(canvas);count+=1
            if frame in [first+12,128172,108841,33329,30442,125032]:cv2.imwrite(str(snaps/f'frame_{frame}.jpg'),canvas)
    cap.release();writer.release()
    (OUT/'review_video_manifest.json').write_text(json.dumps(dict(frames=count,fps=25,duration_seconds=count/25,
        clips=[dict(first_frame=a,last_frame_exclusive=b,reason=c) for a,b,c in clips],all_rejected_frames_included=True),indent=2))
    print(f'Review video: {count} frames / {count/25:.2f}s')


def main():
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('stage',choices=['apply','player','render'])
    a=p.parse_args();{'apply':apply,'player':player,'render':render}[a.stage]()


if __name__=='__main__':main()
