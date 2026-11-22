"""Original broadcast ATD versus the newly exported CSV-based ATD, in video.

Read both actual TXT files. Draw only ball positions on the source footage and
a separate pitch schematic. No coordinates or existing files are modified.
"""
from pathlib import Path
import argparse
import json
import subprocess
import time
import cv2
import numpy as np
import pandas as pd
from PIL import Image,ImageDraw,ImageFont

ROOT=Path(__file__).resolve().parents[1]
BASE=ROOT/'outputs/broadcast_atd_from_csv_v1'
OUT=BASE/'review_video'
SOURCE=ROOT/'data/broadcast/BAR-ATH_broadcast-001.mp4'
ORIGINAL=ROOT/'data/broadcast/BAR-ATH_broadcast_atd.txt'
CORRECTED=BASE/'BAR-ATH_broadcast_atd.txt'
HFILE=ROOT/'data/broadcast/BAR-ATH_broadcast_homography.csv'
PRIMARY=ROOT/'data/derived/ball_temporal_ai_20261117/ball_homography_all_frames.csv'
N=164487;FPS=25;W=1600;H=760;SCALE=600/165
ORIGIN=np.array([980+30*SCALE,110+26*SCALE]);BOUNDS=np.array([[-30.,-26.],[135.,94.]])
NATIVE=ROOT/'outputs/ball_tracking_runtime/render_atd_native'
COLORS={0:(220,220,220),1:(76,230,99),2:(255,163,59),3:(64,209,250),4:(255,219,82)}


def read(path):return pd.read_csv(path,float_precision='round_trip',low_memory=False)


def digest(path):
    import hashlib
    h=hashlib.sha256()
    with Path(path).open('rb') as stream:
        for block in iter(lambda:stream.read(8*1024*1024),b''):h.update(block)
    return h.hexdigest()


def clean(value):
    if isinstance(value,dict):return {str(k):clean(v) for k,v in value.items()}
    if isinstance(value,(list,tuple)):return [clean(v) for v in value]
    if isinstance(value,np.ndarray):return clean(value.tolist())
    if isinstance(value,np.integer):return int(value)
    if isinstance(value,np.bool_):return bool(value)
    if isinstance(value,(float,np.floating)):return float(value) if np.isfinite(value) else None
    return value


def save(name,value):(OUT/name).write_text(json.dumps(clean(value),ensure_ascii=False,indent=2,allow_nan=False)+'\n',encoding='utf-8')


def font(size):return ImageFont.truetype('/System/Library/Fonts/AppleSDGothicNeo.ttc',size)


def pitch(xy):return ORIGIN+np.asarray(xy)*SCALE


def project(h,xy):
    q=np.einsum('nij,nj->ni',h,np.c_[xy,np.ones(len(xy))])
    with np.errstate(divide='ignore',invalid='ignore'):return q[:,:2]/q[:,2:],q[:,2]


def ball_text(path):
    rows=[]
    with path.open('rb') as stream:
        for f,line in enumerate(stream):
            prefix,ball=line.rstrip(b'\r\n').rsplit(b':',1)
            assert int(prefix.split(b':',1)[0])==f+1
            row=np.fromstring(ball.decode('ascii'),sep=',');assert len(row)==3
            rows.append(row)
    result=np.asarray(rows);assert len(result)==N+1
    return result


def background(review):
    im=Image.new('RGBA',(W,H),(21,24,30,255));d=ImageDraw.Draw(im)
    d.text((18,9),'Broadcast ATD 보정값 · 중계영상 + 경기장 좌표',font=font(27),fill=(241,241,241))
    d.text((980,10),'원본 ATD / 보정 ATD 비교',font=font(25),fill=(241,241,241))
    d.text((18,51),'초록 CSV 관측 반영',font=font(20),fill=COLORS[1])
    d.text((283,51),'노랑 CSV 기존 예측 반영',font=font(20),fill=COLORS[4])
    d.text((604,51),'하늘 원본 ATD 유지',font=font(20),fill=COLORS[3])
    d.text((980,44),'회색 원본 / 색상 보정 · 105 × 68m',font=font(19),fill=(225,225,225))
    d.text((980,71),'프레임',font=font(17),fill=(238,238,238));d.text((1188,71),'시간',font=font(17),fill=(238,238,238))
    d.text((18,77),'검토 구간 모음 · 구간 사이 원본 시간 점프' if review else '전체 164,487프레임 · 원본 25FPS / 시간 유지',font=font(15),fill=(200,206,214))
    d.rectangle((980,110,1580,110+120*SCALE),fill=(25,53,38,255))
    def p(x,y):return tuple(pitch([x,y]))
    d.rectangle((*p(0,0),*p(105,68)),fill=(30,87,49),outline=(214,224,217),width=2)
    d.line([p(52.5,0),p(52.5,68)],fill=(214,224,217),width=2)
    for a,b in [(0,16.5),(105,88.5)]:d.line([p(a,13.84),p(b,13.84),p(b,54.16),p(a,54.16)],fill=(214,224,217),width=2)
    for a,b in [(0,5.5),(105,99.5)]:d.line([p(a,24.84),p(b,24.84),p(b,43.16),p(a,43.16)],fill=(214,224,217),width=2)
    center=pitch([52.5,34]);r=9.15*SCALE
    d.ellipse((center[0]-r,center[1]-r,center[0]+r,center[1]+r),outline=(214,224,217),width=2)
    for x in [11,52.5,94]:
        a,b=p(x,34);d.ellipse((a-2,b-2,a+2,b+2),fill=(214,224,217))
    d.text((990,119),'경기장 밖 좌표 포함 · Y방향은 원본 ATD 좌표계',font=font(15),fill=(225,232,227))
    d.text((18,655),'회색 ○ 원본 ATD  /  색 점 보정 ATD',font=font(21),fill=(220,220,220))
    d.text((18,686),'공 x/y만 반영 · 선수·속도(s)·H·원본 파일 보존',font=font(20),fill=(229,233,237))
    d.text((980,660),'X (m)',font=font(20),fill=(235,235,235));d.text((1270,660),'Y (m)',font=font(20),fill=(235,235,235))
    d.text((980,704),'색 점: 새 ATD TXT에 저장된 최종 공 좌표',font=font(18),fill=(226,232,238))
    d.text((980,733),'새 탐지·예측·높이 추정 없이 저장값만 시각화',font=font(16),fill=(215,218,224))
    im.paste((0,0,0,0),(0,100,960,640));return im


def assets():
    (OUT/'assets').mkdir(parents=True,exist_ok=True);(OUT/'logs').mkdir(exist_ok=True)
    for review,name in [(False,'background_full.png'),(True,'background_review.png')]:background(review).save(OUT/'assets'/name)
    sources={0:'보정 ATD: 최종 공 좌표 없음',1:'보정 ATD: CSV 관측 좌표 반영',2:'보정 ATD: CSV 좌표 반영',3:'보정 ATD: CSV 결측 · 원본 ATD 유지',4:'보정 ATD: CSV에 있던 예측 좌표 반영'}
    states={0:'출처: 두 데이터 모두 좌표 없음',1:'출처: CSV 원래 SNP 관측',2:'출처: CSV 실제 YOLO26x 관측',3:'출처: CSV 기존 예측 (새 예측 생성 아님)',4:'출처: 원본 MTAH Broadcast ATD'}
    for code,text in sources.items():
        im=Image.new('RGBA',(600,34),(21,24,30,255));ImageDraw.Draw(im).text((0,1),text,font=font(22),fill=COLORS[code]);im.save(OUT/'assets'/f'source_{code}.png')
    for code,text in states.items():
        im=Image.new('RGBA',(600,30),(21,24,30,255));ImageDraw.Draw(im).text((0,1),text,font=font(19),fill=(228,232,236));im.save(OUT/'assets'/f'state_{code}.png')
    for code,text in {0:'중계영상 점은 ATD 좌표의 기존 H 재투영 위치 · 실제 이미지 검출과 다를 수 있음',
        1:'경기장 공 좌표는 있음 · H 사용 불가/화면 밖이어서 영상 점은 표시하지 않음'}.items():
        im=Image.new('RGBA',(930,28),(21,24,30,255));ImageDraw.Draw(im).text((0,1),text,font=font(17),fill=(230,218,176) if code else (208,214,222));im.save(OUT/'assets'/f'projection_{code}.png')
    mono=ImageFont.truetype('/System/Library/Fonts/Menlo.ttc',20)
    for char in '0123456789:+-. ':
        im=Image.new('RGBA',(17,28),(21,24,30,255));ImageDraw.Draw(im).text((1,0),char,font=mono,fill=(242,242,242));im.save(OUT/'assets'/f'glyph_{ord(char)}.png')


def prepare():
    assets();d=read(BASE/'corrected_ball_coordinates.csv').iloc[:N].copy()
    assert np.array_equal(d.frame,np.arange(N))
    old=ball_text(ORIGINAL);new=ball_text(CORRECTED)
    assert np.array_equal(old[:N,:2],d[['original_ball_x','original_ball_y']],equal_nan=True)
    assert np.array_equal(new[:N,:2],d[['corrected_ball_x','corrected_ball_y']],equal_nan=True)
    assert np.array_equal(old[:,2],new[:,2],equal_nan=True)
    htable=read(HFILE);h=htable[[f'h{i}{j}' for i in range(3) for j in range(3)]].to_numpy(float).reshape(-1,3,3)[:N]
    finite_h=np.isfinite(h).all(axis=(1,2));hgood=finite_h&htable.valid.iloc[:N].eq(1).to_numpy()&htable.reprojection_error.iloc[:N].le(5).to_numpy()
    center=np.einsum('nij,j->ni',h,[.5,.5,1.])[:,2]
    oldworld=old[:N,:2]*[105.,68.];newworld=new[:N,:2]*[105.,68.]
    oldvalid=np.isfinite(oldworld).all(axis=1);newvalid=np.isfinite(newworld).all(axis=1)
    def image_points(norm,valid):
        point,depth=project(h,norm)
        visible=valid&hgood&np.isfinite(point).all(axis=1)&(depth*center>0)&((point>=[0,0])&(point<[1920,1080])).all(axis=1)
        point=point*.5+[0,100];point[~visible]=np.nan;return point,visible
    def pitch_points(world,valid):
        visible=valid&((world>=BOUNDS[0])&(world<=BOUNDS[1])).all(axis=1)
        point=pitch(world);point[~visible]=np.nan;return point,visible
    oi,oiv=image_points(old[:N,:2],oldvalid);ni,niv=image_points(new[:N,:2],newvalid)
    op,opv=pitch_points(oldworld,oldvalid);np_,npv=pitch_points(newworld,newvalid)
    assert opv[oldvalid].all() and npv[newvalid].all()
    applied=d.csv_coordinate_used.to_numpy(bool);pred=d.source_is_prediction.to_numpy(bool)
    code=np.where(~newvalid,0,np.where(applied,np.where(pred,4,1),3))
    provenance=np.where(~newvalid,0,np.where(~applied,4,np.where(pred,3,np.where(d.source_kind.eq('CSV_ORIGINAL_SNP_OBSERVATION'),1,2))))
    values=np.c_[d.frame,code,provenance,oi,op,ni,np_,newworld,(newvalid&~niv).astype(int)]
    rows=[[int(v) if j in [0,1,2,13] else (float(v) if np.isfinite(v) else None) for j,v in enumerate(row)] for row in values]
    (OUT/'assets/full_rows.json').write_text(json.dumps(rows,separators=(',',':'),allow_nan=False),encoding='utf-8')
    delta=np.linalg.norm(newworld-oldworld,axis=1)
    d['difference_m']=delta;d['original_image_visible']=oiv;d['corrected_image_visible']=niv;d['corrected_pitch_visible']=npv
    d.to_csv(OUT/'overlay_frame_metadata.csv',index=False)
    selected=[]
    def choose(mask,maximum,rank=None):
        ids=np.flatnonzero(mask)
        if rank is not None:ids=ids[np.argsort(-np.nan_to_num(rank[ids],nan=-1))]
        elif len(ids):ids=ids[np.unique(np.linspace(0,len(ids)-1,min(len(ids),maximum*30)).astype(int))]
        chosen=[]
        for f in ids:
            if any(abs(int(f)-q)<1500 for q in selected+chosen):continue
            chosen.append(int(f))
            if len(chosen)>=maximum:break
        selected.extend(chosen)
    visible=oiv&niv
    choose(applied&~pred&visible&(delta>.3)&(delta<5),2)
    choose(applied&~pred&visible&(delta>=5)&(delta<=25),3,delta)
    choose(applied&~oldvalid&~pred&niv,2)
    choose(applied&pred&niv,2)
    choose((code==3)&niv,1)
    gaps=~newvalid;starts=np.flatnonzero(gaps&~np.r_[False,gaps[:-1]]);ends=np.flatnonzero(gaps&~np.r_[gaps[1:],False])
    for a,b in zip(starts,ends):
        if 1<=b-a+1<=50 and a>7000 and a>0 and b+1<N and newvalid[a-1] and newvalid[b+1]:
            middle=int((a+b)//2)
            if all(abs(middle-q)>=1500 for q in selected):selected.append(middle);break
    intervals=[]
    for f in sorted(selected):
        a=max(0,f-100);b=min(N,f+101)
        if intervals and a<=intervals[-1][1]:intervals[-1][1]=max(intervals[-1][1],b)
        else:intervals.append([a,b])
    review_frames=np.concatenate([np.arange(a,b) for a,b in intervals])
    (OUT/'assets/review_rows.json').write_text(json.dumps([rows[f] for f in review_frames],separators=(',',':'),allow_nan=False),encoding='utf-8')
    pd.DataFrame(dict(output_frame=np.arange(len(review_frames)),source_frame=review_frames)).to_csv(OUT/'review_frame_mapping.csv',index=False)
    save('video_manifest.json',dict(status='ASSETS_READY',frames=N,fps=25,size=[W,H],full_duration_seconds=N/25,
        review_frames=len(review_frames),review_duration_seconds=len(review_frames)/25,review_intervals=intervals,
        pitch_origin=ORIGIN,pitch_scale=SCALE,pitch_world_bounds=BOUNDS,
        source_codes={'MISSING':0,'CSV_OBSERVATION_APPLIED':1,'ORIGINAL_ATD_RETAINED':3,'EXISTING_CSV_PREDICTION_APPLIED':4},
        provenance_codes={'MISSING':0,'CSV_SNP_OBSERVATION':1,'CSV_YOLO26X_OBSERVATION':2,'CSV_EXISTING_PREDICTION':3,'ORIGINAL_ATD':4},
        input_hashes={str(p.relative_to(ROOT)):digest(p) for p in [ORIGINAL,CORRECTED,PRIMARY,HFILE,BASE/'corrected_ball_coordinates.csv',BASE/'summary.json',NATIVE]},
        original_valid_ball_frames=int(oldvalid.sum()),corrected_valid_ball_frames=int(newvalid.sum()),
        corrected_image_marker_frames=int(niv.sum()),original_image_marker_frames=int(oiv.sum()),
        corrected_pitch_marker_frames=int(npv.sum()),source_video_stat=dict(bytes=SOURCE.stat().st_size,mtime_ns=SOURCE.stat().st_mtime_ns),
        metadata_read_from_actual_original_and_corrected_TXT=True,all_saved_xy_checked_against_export_CSV=True,
        pitch_lines_drawn_only_on_separate_schematic=True,new_coordinates_or_predictions=False,
        frame_mapping='Full output frame == original zero-based video frame; ATD row ID == frame+1',
        original_extra_ATD_row_excluded_from_video=1))
    return rows


def render(which):
    target=OUT/('corrected_atd_full_match.mp4' if which=='full' else 'corrected_atd_review.mp4')
    assert NATIVE.exists() and not target.exists()
    cmd=[str(NATIVE),str(SOURCE),str(OUT/'assets'/('full_rows.json' if which=='full' else 'review_rows.json')),
        str(OUT/'assets'/('background_full.png' if which=='full' else 'background_review.png')),str(target)]
    with (OUT/'logs'/f'render_{which}.log').open('w') as stream:subprocess.run(cmd,cwd=ROOT,stdout=stream,stderr=subprocess.STDOUT,check=True)


def verify_metadata():
    rows=json.loads((OUT/'assets/full_rows.json').read_text());arr=np.array(rows,float)
    d=read(BASE/'corrected_ball_coordinates.csv').iloc[:N];new=ball_text(CORRECTED)[:N,:2];old=ball_text(ORIGINAL)[:N,:2]
    assert len(rows)==N and np.array_equal(arr[:,0],np.arange(N))
    valid=np.isfinite(new).all(axis=1);oldvalid=np.isfinite(old).all(axis=1)
    assert np.array_equal(arr[:,11:13],new*[105.,68.],equal_nan=True)
    assert np.array_equal(np.isfinite(arr[:,9:11]).all(axis=1),valid)
    assert np.allclose((arr[valid,9:11]-ORIGIN)/SCALE,new[valid]*[105.,68.],atol=1e-10)
    assert np.allclose((arr[oldvalid,5:7]-ORIGIN)/SCALE,old[oldvalid]*[105.,68.],atol=1e-10)
    applied=d.csv_coordinate_used.to_numpy(bool);pred=d.source_is_prediction.to_numpy(bool)
    expected_code=np.where(~valid,0,np.where(applied,np.where(pred,4,1),3))
    assert np.array_equal(arr[:,1],expected_code)
    review=json.loads((OUT/'assets/review_rows.json').read_text());mapping=read(OUT/'review_frame_mapping.csv')
    assert len(review)==len(mapping) and all(row==rows[int(mapping.source_frame.iloc[i])] for i,row in enumerate(review))
    report=dict(passed=True,all_164487_frames_checked=True,corrected_points_match_actual_TXT=True,
        original_points_match_original_TXT=True,normalized_to_105x68_conversion_verified=True,
        no_predicted_or_synthetic_points_added=True,review_mapping_exact=True)
    save('overlay_metadata_verification.json',report);return report


def verify_video(which):
    from starter.ball_tracking_video_utils import color_pixels
    rows=json.loads((OUT/'assets'/('full_rows.json' if which=='full' else 'review_rows.json')).read_text())
    path=OUT/('corrected_atd_full_match.mp4' if which=='full' else 'corrected_atd_review.mp4')
    cap=cv2.VideoCapture(str(path));assert cap.isOpened()
    count=int(cap.get(7));fps=cap.get(5);assert count==len(rows) and fps==25 and cap.get(3)==W and cap.get(4)==H
    indices=set(np.unique(np.linspace(0,count-1,min(count,40)).astype(int)).tolist())
    for code in [0,1,3,4]:
        ids=np.array([i for i,row in enumerate(rows) if row[1]==code and (code==0 or row[9] is not None)])
        if len(ids):indices.update(ids[np.unique(np.linspace(0,len(ids)-1,min(5,len(ids))).astype(int))].tolist())
    gaps=np.array([f for f in range(1,count) if rows[f][9] is None and rows[f-1][9] is not None])
    for f in gaps[np.unique(np.linspace(0,len(gaps)-1,min(8,len(gaps))).astype(int))] if len(gaps) else []:indices.update([int(f)-1,int(f)])
    glyphs={c:cv2.imread(str(OUT/'assets'/f'glyph_{ord(c)}.png')) for c in '0123456789'}
    status={code:cv2.imread(str(OUT/'assets'/f'source_{code}.png')) for code in [0,1,3,4]}
    source=cv2.VideoCapture(str(SOURCE));assert source.isOpened();samples=[]
    for index in sorted(indices):
        cap.set(1,int(index));ok,frame=cap.read();assert ok and frame.shape==(H,W,3)
        row=rows[index];f=int(row[0]);source.set(1,f);yes,image=source.read();assert yes
        a=cv2.resize(frame[100:640,:960],(64,36),interpolation=cv2.INTER_AREA);b=cv2.resize(image,(64,36),interpolation=cv2.INTER_AREA)
        error=float(abs(a.astype(float)-b.astype(float)).mean());assert error<12,(which,index,'source frame image',error)
        recognized=''
        for j in range(6):
            patch=frame[68:96,1050+j*17:1050+(j+1)*17]
            scores={c:float(abs(patch.astype(float)-glyphs[c].astype(float)).mean()) for c in glyphs};recognized+=min(scores,key=scores.get)
        assert recognized==f'{f:06d}',(which,index,recognized,f)
        source_error=float(abs(frame[560:594,980:1580].astype(float)-status[row[1]].astype(float)).mean())
        assert source_error<10,(which,index,'wrong source status',source_error)
        marker=None
        if row[9] is not None:
            x,y=np.rint(row[9:11]).astype(int);patch=frame[y-9:y+10,x-9:x+10]
            marker=color_pixels(patch,row[1]);assert marker>=12,(which,index,'pitch marker',marker)
        elif index and rows[index-1][9] is not None:
            x,y=np.rint(rows[index-1][9:11]).astype(int);patch=frame[y-2:y+3,x-2:x+3]
            assert color_pixels(patch,rows[index-1][1])<8,(which,index,'marker carried into missing frame')
        if row[7] is not None:
            x,y=np.rint(row[7:9]).astype(int)
            if 13<x<947 and 113<y<627:
                patch=frame[y-13:y+14,x-13:x+14]
                assert color_pixels(patch,row[1])>=10,(which,index,'image marker')
        samples.append(dict(output_frame=int(index),source_frame=f,source_code=row[1],source_image_mae=error,
            frame_digits=recognized,source_status_mae=source_error,pitch_marker_color_pixels=marker))
    cap.release();source.release();progress=json.loads(path.with_suffix('.mp4.progress.json').read_text())
    assert progress['status']=='complete' and progress['frames']==count and abs(progress['output_duration_seconds']-count/25)<.1
    manifest=json.loads((OUT/'video_manifest.json').read_text())
    for relative,signature in manifest['input_hashes'].items():assert digest(ROOT/relative)==signature,relative
    report=dict(passed=True,file=path.name,frames=count,fps=25,size=[W,H],duration_seconds=count/25,
        source_audio_preserved=progress['audio_preserved'],sampled_frames=len(samples),samples=samples,
        source_frame_and_overlay_synchronization_checked=True,no_marker_carried_across_missing_frames=True,
        coordinate_source_colors_checked=True,immutable_input_hashes_unchanged=True,video_sha256=digest(path))
    save(f'{which}_video_verification.json',report);return report


def finish():
    metadata=verify_metadata();full=verify_video('full')
    reviewed=OUT/'review_video_verification.json'
    review=json.loads(reviewed.read_text()) if reviewed.exists() else None
    if review is None or not review.get('passed') or review.get('video_sha256')!=digest(OUT/'corrected_atd_review.mp4'):review=verify_video('review')
    manifest=json.loads((OUT/'video_manifest.json').read_text());manifest.update(status='COMPLETE',
        full_video='corrected_atd_full_match.mp4',review_video='corrected_atd_review.mp4',
        metadata_verification=metadata,full_verification=full,review_verification=review,
        renderer_source_sha256=digest(Path(__file__)))
    save('video_manifest.json',manifest)
    (OUT/'README.md').write_text(f'''# CSV 기준 Broadcast ATD 보정값 영상

- [검토 구간 모음](corrected_atd_review.mp4): {manifest['review_duration_seconds']:.2f}초. 좌표 이동·원본 결측 채움·기존 예측 반영·원본 유지의 자동 선정 예시입니다.
- [전체 프레임 영상](corrected_atd_full_match.mp4): 164,487프레임, 25FPS, 1시간 49분 39.48초. 모든 원본 프레임과 음성을 유지했습니다.

왼쪽 중계영상, 오른쪽 105×68m 경기장 좌표입니다. 회색 빈 원은 원본 ATD, 색 점은 새로 저장한 보정 ATD TXT의 공 x/y입니다.
초록은 CSV 관측(SNP·YOLO) 반영, 노랑은 CSV에 이미 있던 예측 반영, 하늘색은 CSV 결측으로 원본 ATD를 유지한 좌표입니다.
각 프레임의 숫자·원본 시간·미터 좌표와 정확한 관측 출처를 표시했습니다.

좌표는 두 실제 TXT 파일에서 직접 읽고 저장된 CSV와 전체 행을 대조했습니다. 원본 ATD·보정 ATD·H·지정 CSV는 이 영상 제작에서 변경하지 않았습니다.
중계영상 점은 기존 H로 재투영한 위치이며 원래 이미지 공 위치와 다를 수 있습니다. H가 없거나 화면 밖이면 영상 점만 숨기고 경기장 점은 유지합니다.
최종 결측 프레임에서는 색 점이 남지 않습니다. 경기장 선은 별도 도면에만 있습니다. 새 탐지·예측·공중볼/높이 보정은 하지 않았습니다.
검토 모음은 구간 사이 원본 시간이 건너뜁니다. 전체 영상은 시간 점프가 없습니다.

전체 프레임 좌표 대조는 `overlay_metadata_verification.json`, 영상 샘플 검증은 `full_video_verification.json`과 `review_video_verification.json`에 있습니다.
''',encoding='utf-8')


def main():
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('stage',choices=['assets','review','full','verify_review','verify']);args=p.parse_args()
    OUT.mkdir(parents=True,exist_ok=True)
    if args.stage=='assets':prepare();verify_metadata()
    elif args.stage=='verify_review':verify_video('review')
    elif args.stage=='verify':finish()
    else:render(args.stage)


if __name__=='__main__':main()
