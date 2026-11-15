"""Append a BALL-ONLY diagnostic below the existing player comparison video.

python starter/export_tracking_disagreement_ball.py
Existing player metrics/video, alignment and original assets are read-only.
Availability means finite ATD x/y, not verified visibility in the camera image.
"""
import argparse
import csv
import gzip
import json
import os
from pathlib import Path
import shutil
import sys
import time

import cv2
import numpy as np

ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT))
from games import GAME,GAMES,asset_paths,source_frames
from starter.tracking_utils import FLIPS,reflected
from starter.export_tracking_disagreement import load_mapping,sha,distribution,DEFAULT_MAP
from starter.view_all_movement_vectors import timeline_segments

PLAYER_FOLDER=ROOT/'data/derived/tracking_disagreement_20261113'
OUTPUT=ROOT/'data/derived/tracking_disagreement_ball_20261113'
SIZE=(1280,1040)
STATES=('BOTH_OBSERVED','TACTICAL_ONLY','BROADCAST_ONLY','BOTH_MISSING','UNMAPPED')
FIELDS=['frame','broadcast_frame','half','mapped','comparison_run','t_ball_observed','b_ball_observed','ball_state_code',
    'ball_dx_B_minus_T_m','ball_dy_B_minus_T_m','ball_distance_m','ball_presence_B_minus_T',
    'ball_rmse_past2s_m','ball_common_past2s','ball_record_support_past2s','ball_t_observed_past2s','ball_b_observed_past2s']
COL={k:i for i,k in enumerate(FIELDS)}


def read_ball(path,expected,pitch):
    """Read only the final BALL field. Player fields cannot affect these metrics."""
    ball=np.full((expected,2),np.nan,np.float32);count=0
    with Path(path).open() as f:
        for i,line in enumerate(f):
            parts=line.rstrip('\r\n').split(':')
            if i>=expected or len(parts)!=3 or int(parts[0])!=i+1:raise ValueError('ATD frame axis invalid')
            values=parts[2].split(',')
            if len(values)!=3:raise ValueError('BALL field invalid')
            ball[i]=float(values[0]),float(values[1]);count+=1
    if count!=expected:raise ValueError('ATD frame count invalid')
    return ball*np.asarray(pitch,np.float32)


def compute_ball(t_ball,b_ball,mapping,halves,flips,pitch=(105.3,68),window=50):
    """No player observations, pair distances or counts are accepted as inputs.

    Current distance is undefined unless BOTH coordinates are observed.
    Trailing RMSE uses only jointly observed frames inside one original mapping
    run. Missing errors do not enter its numerator OR denominator.
    """
    t_ball=np.asarray(t_ball);b_ball=np.asarray(b_ball);mapping=np.asarray(mapping)
    n=len(mapping)
    if (t_ball.shape!=(n,2) or b_ball.ndim!=2 or b_ball.shape[1]!=2 or len(halves)!=n or len(flips)!=n
            or window<1 or not np.issubdtype(mapping.dtype,np.integer) or (mapping<-1).any() or (mapping>=len(b_ball)).any()):raise ValueError('Ball input axes invalid')
    v=np.full((n,len(FIELDS)),np.nan);mapped=mapping>=0;tf=np.flatnonzero(mapped);v[:,COL['frame']]=np.arange(n)
    v[:,COL['half']]=halves;v[:,COL['mapped']]=mapped;v[:,COL['comparison_run']]=0
    v[:,COL['ball_state_code']]=4;v[:,COL['t_ball_observed']]=np.isfinite(t_ball).all(1)
    for key in ('ball_common_past2s','ball_record_support_past2s','ball_t_observed_past2s','ball_b_observed_past2s'):v[:,COL[key]]=0
    if not len(tf):return v
    if not np.isin(flips[tf],np.arange(len(FLIPS))).all():raise ValueError('Invalid reflection')
    if (np.diff(mapping[tf])<=0).any():raise ValueError('Reversed/reused broadcast frame')
    aligned=np.full((n,2),np.nan)
    for flip in np.unique(flips[tf]):
        take=tf[np.asarray(flips)[tf]==flip];aligned[take]=reflected(b_ball[mapping[take]],FLIPS[int(flip)],pitch)
    ts=np.isfinite(t_ball[tf]).all(1);bs=np.isfinite(aligned[tf]).all(1);both=ts&bs
    v[tf,COL['broadcast_frame']]=mapping[tf];v[tf,COL['b_ball_observed']]=bs
    v[tf,COL['ball_presence_B_minus_T']]=bs.astype(int)-ts.astype(int)
    v[tf,COL['ball_state_code']]=np.select([both,ts&~bs,~ts&bs],[0,1,2],default=3)
    common=tf[both];delta=aligned[common]-t_ball[common]
    v[common,COL['ball_dx_B_minus_T_m']]=delta[:,0];v[common,COL['ball_dy_B_minus_T_m']]=delta[:,1]
    v[common,COL['ball_distance_m']]=np.linalg.norm(delta,axis=1)
    breaks=np.r_[True,(np.diff(tf)!=1)|(np.diff(mapping[tf])!=1)|(np.diff(np.asarray(halves)[tf])!=0)|(np.diff(np.asarray(flips)[tf])!=0)]
    starts=np.flatnonzero(breaks);ends=np.r_[starts[1:],len(tf)]
    for run,(a,b) in enumerate(zip(starts,ends),1):
        rows=tf[a:b];length=b-a;hi=np.arange(1,length+1);lo=np.maximum(0,hi-window);v[rows,COL['comparison_run']]=run
        valid=np.isfinite(v[rows,COL['ball_distance_m']]);sq=np.where(valid,v[rows,COL['ball_distance_m']]**2,0)
        c=np.r_[0,np.cumsum(valid)];s=np.r_[0.,np.cumsum(sq)];count=c[hi]-c[lo]
        mean=np.divide(s[hi]-s[lo],count,out=np.full(length,np.nan),where=count>0)
        v[rows,COL['ball_rmse_past2s_m']]=np.sqrt(np.maximum(mean,0));v[rows,COL['ball_common_past2s']]=count
        v[rows,COL['ball_record_support_past2s']]=hi-lo
        for flag,key in [('t_ball_observed','ball_t_observed_past2s'),('b_ball_observed','ball_b_observed_past2s')]:
            c=np.r_[0,np.cumsum(v[rows,COL[flag]])];v[rows,COL[key]]=c[hi]-c[lo]
    return v


def save_metrics(v,folder):
    np.save(folder/'ball_values.npy',v);(folder/'ball_metric_fields.json').write_text(json.dumps(FIELDS))
    ints=set(FIELDS)-{'ball_dx_B_minus_T_m','ball_dy_B_minus_T_m','ball_distance_m','ball_rmse_past2s_m'}
    with gzip.open(folder/'ball_metrics.csv.gz','wt',newline='') as f:
        w=csv.writer(f);w.writerow(FIELDS+['ball_state'])
        for row in v:
            w.writerow(['' if not np.isfinite(z) else int(z) if k in ints else f'{z:.6f}' for k,z in zip(FIELDS,row)]+[STATES[int(row[COL['ball_state_code']])]])


def ball_summary(v):
    result={};mapped=v[:,COL['mapped']]>0
    for name,choose in [('all_mapped',mapped),('1st Half',mapped&(v[:,COL['half']]==1)),('2nd Half',mapped&(v[:,COL['half']]==2))]:
        part=v[choose];n=len(part)
        result[name]=dict(frames=n,availability={state:dict(frames=int((part[:,COL['ball_state_code']]==i).sum()),fraction=float((part[:,COL['ball_state_code']]==i).mean()) if n else None) for i,state in enumerate(STATES[:4])},
            ball_distance_m=distribution(part[:,COL['ball_distance_m']]),t_observed_frames=int(np.nansum(part[:,COL['t_ball_observed']])),b_observed_frames=int(np.nansum(part[:,COL['b_ball_observed']])))
    return result


def true_runs(mask):
    edges=np.diff(np.r_[False,mask,False].astype(np.int8));return zip(np.flatnonzero(edges==1),np.flatnonzero(edges==-1))


def number(value):return f'{value:.2f}' if np.isfinite(value) else 'N/A'


def append_ball_panel(player_frame,display_values,index):
    """Only drawing changes above760px: replace old ball line with a separation label."""
    if player_frame.shape!=(760,1280,3):raise ValueError('Unexpected player video size')
    image=np.full((SIZE[1],SIZE[0],3),(24,22,18),np.uint8);image[:760]=player_frame
    def text(s,x,y,color=(235,235,235),scale=.5):cv2.putText(image,s,(x,y),cv2.FONT_HERSHEY_SIMPLEX,scale,color,1,cv2.LINE_8)
    cv2.rectangle(image,(882,459),(1278,483),(30,25,20),-1);text('Players above / ball separately below',884,474,scale=.43)
    cv2.line(image,(0,760),(1279,760),(140,175,190),1)
    text('BALL ONLY | availability = finite ATD coordinates | NO contribution to player RMSE or counts',20,787,(220,220,120),.55)
    row=display_values[index];t_seen=row[COL['t_ball_observed']]==1;b_seen=row[COL['b_ball_observed']]==1;state=int(row[COL['ball_state_code']])
    green=(100,220,100);cyan=(245,210,90);dim=(150,150,150)
    text('Tactical: '+('OBSERVED' if t_seen else 'MISSING'),20,813,green if t_seen else dim,.57)
    text('Broadcast: '+('OBSERVED' if b_seen else 'MISSING'),360,813,cyan if b_seen else dim,.57)
    text('State: '+STATES[state].replace('_',' '),770,813,(225,225,225),.51)
    text(f'Current distance {number(row[COL["ball_distance_m"]])} m | dx {number(row[COL["ball_dx_B_minus_T_m"]])} | dy {number(row[COL["ball_dy_B_minus_T_m"]])}',20,839,scale=.48)
    text(f'Past2s BALL RMSE {number(row[COL["ball_rmse_past2s_m"]])} m | common {int(row[COL["ball_common_past2s"]])}/{int(row[COL["ball_record_support_past2s"]])} frames',675,839,scale=.48)
    text('dx/dy = B - T. Current distance is N/A when either ball is missing. Past2s uses common observed frames only.',20,862,scale=.44)
    first=max(0,index-250);last=min(len(display_values),index+251);local=display_values[first:last]
    same_run=local[:,COL['comparison_run']]==row[COL['comparison_run']];indices=np.arange(first,last)
    x=np.rint(140+(indices-index+250)*(1100/500)).astype(np.int32);lo,hi=879,962
    cv2.rectangle(image,(140,lo),(1240,hi),(55,51,42),1)
    text('20m',92,lo+6,scale=.36);text('0m',105,hi,scale=.36)
    distance=local[:,COL['ball_distance_m']];good=same_run&np.isfinite(distance)
    yy=np.rint(hi-np.clip(np.nan_to_num(distance,nan=0),0,20)/20*(hi-lo)).astype(np.int32)
    for a,b in true_runs(good):
        points=np.c_[x[a:b],yy[a:b]]
        if len(points)>1:cv2.polylines(image,[points],False,(90,200,245),1,cv2.LINE_8)
        else:cv2.circle(image,tuple(points[0]),1,(90,200,245),-1)
    for key,y,color,label in [('t_ball_observed',979,green,'T ball'),('b_ball_observed',1000,cyan,'B ball')]:
        text(label,35,y+9,color,.42);cv2.rectangle(image,(140,y),(1240,y+10),(15,15,15),-1)
        for a,b in true_runs(same_run):cv2.rectangle(image,(int(x[a]),y),(min(1240,int(x[b-1])+2),y+10),(60,60,60),-1)
        observed=(local[:,COL[key]]==1)&same_run
        for a,b in true_runs(observed):cv2.rectangle(image,(int(x[a]),y),(min(1240,int(x[b-1])+2),y+10),color,-1)
    cv2.line(image,(690,lo),(690,1013),(235,235,235),1)
    text('Past10s <--- now ---> Future10s | plot0-20m: display clip only | strips: color=observed / gray=missing / black=no support',20,1031,scale=.42)
    return image


def export_video(player_folder,folder,v,tf,bf,halves):
    source=player_folder/'tracking_disagreement.mp4';cap=cv2.VideoCapture(str(source));temporary=folder/'tracking_disagreement_with_ball.partial.mp4'
    if (not cap.isOpened() or round(cap.get(cv2.CAP_PROP_FRAME_COUNT))!=len(tf) or abs(cap.get(cv2.CAP_PROP_FPS)-25)>.01
            or (round(cap.get(cv2.CAP_PROP_FRAME_WIDTH)),round(cap.get(cv2.CAP_PROP_FRAME_HEIGHT)))!=(1280,760)):raise ValueError('Player movie differs from mapping')
    writer=cv2.VideoWriter(str(temporary),cv2.VideoWriter_fourcc(*'avc1'),25,SIZE)
    if not writer.isOpened():cap.release();raise RuntimeError('H.264 encoder unavailable')
    display=v[tf];start=time.monotonic()
    try:
        for i in range(len(tf)):
            ok,image=cap.read()
            if not ok:raise ValueError('Player source movie ended early')
            writer.write(append_ball_panel(image,display,i))
            if (i+1)%10000==0:print('Appended ball panel',i+1,'/',len(tf),'elapsed_s',round(time.monotonic()-start,1),flush=True)
    finally:cap.release();writer.release()
    check=cv2.VideoCapture(str(temporary))
    try:
        if not check.isOpened() or round(check.get(cv2.CAP_PROP_FRAME_COUNT))!=len(tf) or abs(check.get(cv2.CAP_PROP_FPS)-25)>.01:raise ValueError('Output video axis invalid')
        for i in sorted(set([0,len(tf)-1,*list(np.flatnonzero(np.diff(halves)!=0)+1)])):
            check.set(cv2.CAP_PROP_POS_FRAMES,int(i));ok,image=check.read()
            if not ok or image.shape!=(SIZE[1],SIZE[0],3):raise ValueError('Output video sample invalid')
    finally:check.release()
    os.replace(temporary,folder/'tracking_disagreement_with_ball.mp4')
    html=(player_folder/'tracking_disagreement_player.html').read_text().replace('tracking_disagreement.mp4','tracking_disagreement_with_ball.mp4')
    html=html.replace('Broadcast–Tactical 위치·인원 비교','Broadcast–Tactical 선수(위)·공(아래) 분리 비교').replace('max-height:72vh','max-height:none')
    relative=os.path.relpath(player_folder,folder).replace(os.sep,'/')
    html=html.replace('href="frame_metrics.csv.gz"',f'href="{relative}/frame_metrics.csv.gz"').replace('href="minute_summary.csv"',f'href="{relative}/minute_summary.csv"')
    explanation='''<p class="muted"><strong>아래쪽은 공 전용입니다.</strong> Tactical/Broadcast 공 좌표가 모두 유한하면 관측, 하나라도 없으면 해당 소스 결측입니다. 실제 영상의 육안 보임/안보임 판정이 아닙니다.<br>
양쪽 관측 때만 현재 공 거리·dx/dy를 계산하며, 한쪽/양쪽 결측은 N/A입니다. 결측을0으로 채우거나 위치를 보간하지 않습니다.<br>
공 RMSE는 최근2초 중 양쪽 공이 관측된 프레임만 사용합니다. 현재 공이 없어도 과거 관측으로 계산될 수 있으므로 common 지원 수를 함께 확인하세요. 전후반·정렬 공백을 넘지 않습니다.<br>
공은 선수 RMSE·선수 수·B−T count에 합산하지 않습니다. 아래 색 띠는 소스별 공 관측/결측, 곡선은 공 거리만 표시합니다.<br>
<a href="ball_metrics.csv.gz">공 전용 프레임별 CSV</a> · <a href="ball_summary.json">공 관측·거리 요약</a></p>'''
    html=html.replace('<script>\n\'use strict\';',explanation+'\n<script>\n\'use strict\';')
    (folder/'tracking_disagreement_ball_player.html').write_text(html);shutil.copyfile(player_folder/'frame_timeline.csv',folder/'frame_timeline.csv')
    return dict(frames=len(tf),fps=25,size=list(SIZE),duration_seconds=len(tf)/25,render_seconds=round(time.monotonic()-start,2),decoded_samples_verified=True)


def write_report(summary,folder):
    lines=['# 선수와 공을 분리한 트래킹 비교','',
        '- 위쪽: 기존 선수 비교 영상의 수치/그래프. 아래쪽: 독립된 공 전용 패널. 공은 선수 RMS/인원/매칭에 들어가지 않는다.',
        '- 관측은 ATD 공 x/y가 모두 유한하다는 뜻이다. 실제 영상에 공이 보이는지, 유효한 공 검출인지 확인한 정답이 아니다.',
        '- 네 상태: BOTH_OBSERVED / TACTICAL_ONLY / BROADCAST_ONLY / BOTH_MISSING. 정렬표가 없는 원본 프레임은 UNMAPPED이며 Broadcast를 결측으로 단정하지 않는다.',
        '- 현재 공 거리=sqrt((Bx−Tx)²+(By−Ty)²), dx/dy=Broadcast−Tactical, 단위m. 한쪽이라도 결측이면 모두N/A. 거리 제한이나 결측 보간을 사용하지 않는다.',
        '- 최근2초 RMSE=sqrt(sum(공 거리²)/공통 관측 수). 현재 포함 과거50프레임, 같은 전후반/연속 정렬 run만 참조한다. run 시작은 존재하는 프레임만 지원 수에 포함한다.',
        '- 현재 결측이어도 과거2초 안에 공통 관측이 있으면 과거 RMSE는 계산된다. 현재 거리 N/A와 과거 RMSE 및 common/record 지원 수를 구분한다.',
        '- 하단 곡선: 현재±10초 공 거리(표시만0~20m로 제한), 결측은 선을 잇지 않는다. T/B 띠의 색=관측, 회색=결측이다. 원본 source 위치/선수 지표를 보간하거나 수정하지 않는다.',
        '- 기존 선수 영상/수치/정렬/원본 데이터와 모델은 보존했다. 새 출력은 별도 로컬 폴더에만 저장했다. 서버/업로드/GT/Smart Tagging/수작업 라벨/Git Commit/Push를 사용하지 않았다.', '',
        '| 구간 | 프레임 | 양쪽 관측 | T만 관측 | B만 관측 | 양쪽 결측 | 공 RMSE m | 공 거리 P95 m |',
        '| --- | --- | --- | --- | --- | --- | --- | --- |']
    for name,g in summary['groups'].items():
        a=g['availability'];d=g['ball_distance_m'];fmt=lambda x:'N/A' if x is None else f'{x:.3f}'
        lines.append(f'| {name} | {g["frames"]} | '+ ' | '.join(str(a[s]['frames']) for s in STATES[:4])+f' | {fmt(d["rmse"])} | {fmt(d["p95"])} |')
    lines+=['','공 거리 분포는 양쪽 공통 관측 프레임에 한정한다. 결측을 거리0으로 세지 않는다. 기존 시간 정렬은 전체 구간에서 검증되지 않았으며 거리 차이에 시간 어긋남과 추적 노이즈가 함께 반영될 수 있다.', '',
        '전체 Tactical199,319행은 ball_metrics.csv.gz에 보존하고, 기존 비교 영상과 같은155,923프레임을25fps로 재생한다.', '',
        '```bash','conda activate mtah2027','python starter/export_tracking_disagreement_ball.py',
        'python -m pytest -q tests/test_tracking_disagreement_ball.py','```']
    (folder/'REPORT.md').write_text('\n'.join(lines)+'\n')


def main():
    p=argparse.ArgumentParser(description=__doc__.splitlines()[0]);p.add_argument('--player-folder',type=Path,default=PLAYER_FOLDER);p.add_argument('--output',type=Path,default=OUTPUT);p.add_argument('--data-dir');p.add_argument('--no-video',action='store_true');args=p.parse_args()
    args.output.mkdir(parents=True,exist_ok=True)
    if args.output.resolve()==args.player_folder.resolve():raise ValueError('Use separate output folder to preserve player results')
    old=json.loads((args.player_folder/'summary.json').read_text());mapping_path=DEFAULT_MAP
    paths={s:asset_paths(source=s,data_dir=args.data_dir) for s in ('tactical','broadcast')}
    protected=[mapping_path,*[Path(paths[s][k]) for s in paths for k in ('atd_raw','atd_meta')],
        *[args.player_folder/k for k in ('frame_values.npy','frame_metrics.csv.gz','tracking_disagreement.mp4','frame_timeline.csv','computed.json','summary.json','tracking_disagreement_player.html')]]
    before={str(path):sha(path) for path in protected}
    if sha(mapping_path)!=old['provenance']['mapping_sha256']:raise ValueError('Mapping changed since player comparison')
    for s in paths:
        for k in ('atd_raw','atd_meta'):
            if sha(paths[s][k])!=old['provenance']['source'][s][k]:raise ValueError('ATD source changed since player comparison')
    config=dict(window_frames=50,fps=25,protected_sha256=before)
    marker=args.output/'input_manifest.json'
    if marker.exists() and json.loads(marker.read_text())!=config:raise ValueError('Different input/config in output; choose a new folder')
    marker.write_text(json.dumps(config,indent=2))
    mapping,halves,flips=load_mapping(mapping_path,source_frames('tactical'),source_frames('broadcast'))
    tb=read_ball(paths['tactical']['atd_raw'],source_frames('tactical'),GAMES[GAME]['pitch']);bb=read_ball(paths['broadcast']['atd_raw'],source_frames('broadcast'),GAMES[GAME]['pitch'])
    v=compute_ball(tb,bb,mapping,halves,flips);save_metrics(v,args.output)
    tf=np.flatnonzero(mapping>=0);bf=mapping[tf];hs=halves[tf]
    if timeline_segments(tf,bf,hs)!=old['video']['segments']:raise ValueError('Player video segments disagree with ball frame axis')
    player=np.load(args.player_folder/'frame_values.npy',mmap_mode='r');fields=json.loads((args.player_folder/'metric_fields.json').read_text())
    expected=player[:,fields.index('ball_distance_m')];distance=v[:,COL['ball_distance_m']]
    # Original per-frame norm was float32; the independent ball path accumulates
    # in float64. Compare with a float32 roundoff bound, not a quality threshold.
    roundoff=8*np.finfo(np.float32).eps
    if not np.array_equal(np.isfinite(expected),np.isfinite(distance)) or not np.allclose(expected,distance,equal_nan=True,atol=roundoff,rtol=roundoff):raise ValueError('Ball distance disagrees with source-axis check')
    common=np.isfinite(distance);numeric_difference=float(np.max(abs(expected[common]-distance[common]))) if common.any() else None
    summary=dict(groups=ball_summary(v),tactical_rows=len(mapping),mapped_rows=len(tf),unmapped_rows=int((mapping<0).sum()),
        ball_only=True,ball_window_frames=50,player_metrics_recomputed=False,frame_distances_missing_not_zero=True,
        observed_means_finite_ATD_coordinates=True,GT_used=False,SmartTag_read=False,source_match_video_read=False,
        reference_distance_max_numeric_difference_m=numeric_difference,reference_distance_finite_mask_equal=True)
    if not args.no_video:summary['video']=export_video(args.player_folder,args.output,v,tf,bf,hs)
    elif (args.output/'ball_summary.json').exists():
        prior=json.loads((args.output/'ball_summary.json').read_text())
        if 'video' in prior:summary['video']=prior['video']
    after={str(path):sha(path) for path in protected};summary['protected_inputs_unchanged']=before==after
    if before!=after:raise ValueError('Protected inputs changed')
    (args.output/'ball_summary.json').write_text(json.dumps(summary,indent=2));write_report(summary,args.output)
    print(json.dumps(dict(tactical_rows=len(mapping),paired_video_frames=len(tf),groups=summary['groups'],protected_inputs_unchanged=True)),flush=True)


if __name__=='__main__':main()
