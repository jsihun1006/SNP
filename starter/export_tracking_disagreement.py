"""Local-only ATD coordinate disagreement, counts, full-frame video and seek player.

python starter/export_tracking_disagreement.py
No source video, GT, Smart Tagging, model inference, tracking correction or server.
Existing per-half mapping is reused as an UNVERIFIED constant-offset assumption.
"""
import argparse
import csv
import gzip
import hashlib
import json
import math
import os
from pathlib import Path
import sys
import time

import cv2
import numpy as np
try:
    from scipy.optimize import linear_sum_assignment as FAST_ASSIGNMENT
except ImportError:
    FAST_ASSIGNMENT=None

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from games import GAME, GAMES, asset_paths, source_frames
from starter.tracking_utils import read_tracking, track_teams, FLIPS, reflected
from starter.view_all_movement_vectors import HTML as PLAYER_HTML, timeline_segments

DEFAULT_OUTPUT = ROOT/'data/derived/tracking_disagreement_20261113'
DEFAULT_MAP = ROOT/'data/derived/sync/final_tracking_alignment.csv'
SIZE = (1280, 760)
PITCH = np.asarray(GAMES[GAME]['pitch'], float)
SIDE_FIELDS = ['t_count','b_count','count_delta_B_minus_T','forced_pairs','forced_rmse_m','forced_median_m',
    'forced_p95_m','gated_pairs','gated_rmse_m','gated_median_m','gated_p95_m','gated_dx_bias_m','gated_dy_bias_m',
    'unmatched_t','unmatched_b','nearest_T_to_B_m','nearest_B_to_T_m','ambiguous_gated_pairs','t_duplicates','b_duplicates']
ALL_FIELDS = ['t_count','b_count','count_delta_B_minus_T','forced_pairs','forced_rmse_m','forced_p95_m',
    'gated_pairs','gated_rmse_m','gated_p95_m','gated_t_coverage','gated_b_coverage','gated_dx_rmse_m','gated_dy_rmse_m',
    'ball_distance_m','t_outside_pitch','b_outside_pitch']
FIELDS = ['frame','broadcast_frame','half','mapped']+[side+'_'+k for side in ('H','A') for k in SIDE_FIELDS]+ALL_FIELDS
COL = {k:i for i,k in enumerate(FIELDS)}


def assignment(cost):
    """Indices for min-cost rectangular assignment. NumPy Hungarian, no new dependency.

    Adapted from starter.find_layout_match.assignment_costs to retain pair indices.
    Every row of the smaller side is assigned; callers handle empty sets first.
    """
    cost=np.asarray(cost,float)
    if cost.ndim!=2 or not np.isfinite(cost).all():raise ValueError('Assignment needs finite2D costs')
    if not cost.size:return np.empty(0,int),np.empty(0,int)
    if FAST_ASSIGNMENT is not None:return FAST_ASSIGNMENT(cost)
    transpose=cost.shape[0]>cost.shape[1]
    if transpose:cost=cost.T
    n,m=cost.shape;u=np.zeros(n+1);v=np.zeros(m+1);p=np.zeros(m+1,int);way=np.zeros(m+1,int)
    for row in range(1,n+1):
        p[0]=row;j0=0;distance=np.full(m+1,np.inf);used=np.zeros(m+1,bool)
        while True:
            used[j0]=True;row0=p[j0];reduced=cost[row0-1]-u[row0]-v[1:]
            improve=(~used[1:])&(reduced<distance[1:]);distance[1:][improve]=reduced[improve];way[1:][improve]=j0
            j1=int(np.argmin(np.where(~used[1:],distance[1:],np.inf)))+1;delta=distance[j1]
            u[p[used]]+=delta;v[used]-=delta;distance[~used]-=delta;j0=j1
            if p[j0]==0:break
        while j0:
            j1=way[j0];p[j0]=p[j1];j0=j1
            if j0==0:break
    cc=np.flatnonzero(p[1:])+1;rr=p[cc]-1;cc=cc-1
    return (cc,rr) if transpose else (rr,cc)


def gated_assignment(squared,gate):
    """Maximize number of <=gate pairs FIRST, then minimize squared displacement.

    n dummy columns support all-unmatched. B>n*gate² prevents exchanging a valid
    pair for lower costs on other pairs. This is not verified player identity.
    """
    squared=np.asarray(squared,float)
    if not np.isfinite(gate) or gate<=0:raise ValueError('Gate must be positive')
    if not squared.size:return np.empty(0,int),np.empty(0,int)
    n,m=squared.shape;big=(n+m+1)*max(1.,gate*gate)
    augmented=np.c_[np.where(squared<=gate*gate,squared,2*big),np.full((n,n),big)]
    rr,cc=assignment(augmented);keep=cc<m;return rr[keep],cc[keep]


def finite_points(points):return np.flatnonzero(np.isfinite(points).all(1))


def duplicate_count(points):return int(len(points)-len(np.unique(points,axis=0))) if len(points) else 0


def rmse(values):return float(np.sqrt(np.mean(np.square(values)))) if len(values) else np.nan
def quantile(values,q):return float(np.percentile(values,q)) if len(values) else np.nan


def compare_frame(tactical,broadcast,t_teams,b_teams,gate=5.):
    """Metre coordinates including last BALL slot; per-frame anonymous team sets."""
    result={};forced=[];gated=[];deltas=[];pairs=[];t_teams=np.asarray(t_teams);b_teams=np.asarray(b_teams)
    for side in ('H','A'):
        ti=np.flatnonzero((t_teams==side)&np.isfinite(tactical[:-1]).all(1));bi=np.flatnonzero((b_teams==side)&np.isfinite(broadcast[:-1]).all(1))
        left=tactical[ti];right=broadcast[bi];nt,nb=len(ti),len(bi)
        entry={k:np.nan for k in SIDE_FIELDS};entry.update(t_count=nt,b_count=nb,count_delta_B_minus_T=nb-nt,
            forced_pairs=0,gated_pairs=0,unmatched_t=nt,unmatched_b=nb,ambiguous_gated_pairs=0,
            t_duplicates=duplicate_count(left),b_duplicates=duplicate_count(right))
        if nt and nb:
            delta=right[None]-left[:,None];sq=np.sum(delta**2,axis=-1);rr,cc=assignment(sq);dist=np.sqrt(sq[rr,cc]);forced.extend(zip(ti[rr],dist))
            gr,gc=gated_assignment(sq,gate);gd=np.sqrt(sq[gr,gc]);dv=delta[gr,gc];gated.extend(zip(ti[gr],gd));deltas.extend(dv)
            ambiguous=0
            for r,c,d in zip(gr,gc,gd):
                alternatives_left=np.delete(np.sqrt(sq[r]),c);alternatives_right=np.delete(np.sqrt(sq[:,c]),r)
                ambiguous+=int((alternatives_left<=d+.5).any() or (alternatives_right<=d+.5).any())
            pairs.extend((int(ti[r]),int(bi[c])) for r,c in zip(gr,gc))
            entry.update(forced_pairs=len(dist),forced_rmse_m=rmse(dist),forced_median_m=quantile(dist,50),forced_p95_m=quantile(dist,95),
                gated_pairs=len(gd),gated_rmse_m=rmse(gd),gated_median_m=quantile(gd,50),gated_p95_m=quantile(gd,95),
                gated_dx_bias_m=float(dv[:,0].mean()) if len(dv) else np.nan,gated_dy_bias_m=float(dv[:,1].mean()) if len(dv) else np.nan,
                unmatched_t=nt-len(gd),unmatched_b=nb-len(gd),nearest_T_to_B_m=float(np.sqrt(sq.min(1)).mean()),
                nearest_B_to_T_m=float(np.sqrt(sq.min(0)).mean()),ambiguous_gated_pairs=ambiguous)
        result.update({side+'_'+k:v for k,v in entry.items()})
    nt=sum(result[s+'_t_count'] for s in ('H','A'));nb=sum(result[s+'_b_count'] for s in ('H','A'))
    fd=np.array([d for _,d in forced]);gd=np.array([d for _,d in gated]);dv=np.asarray(deltas).reshape(-1,2)
    def outside(points):
        good=np.isfinite(points[:-1]).all(1);p=points[:-1][good];return int(((p<0)|(p>PITCH)).any(1).sum())
    result.update(t_count=nt,b_count=nb,count_delta_B_minus_T=nb-nt,forced_pairs=len(fd),forced_rmse_m=rmse(fd),forced_p95_m=quantile(fd,95),
        gated_pairs=len(gd),gated_rmse_m=rmse(gd),gated_p95_m=quantile(gd,95),gated_t_coverage=len(gd)/nt if nt else np.nan,
        gated_b_coverage=len(gd)/nb if nb else np.nan,gated_dx_rmse_m=rmse(dv[:,0]),gated_dy_rmse_m=rmse(dv[:,1]),
        ball_distance_m=float(np.linalg.norm(tactical[-1]-broadcast[-1])) if np.isfinite([tactical[-1],broadcast[-1]]).all() else np.nan,
        t_outside_pitch=outside(tactical),b_outside_pitch=outside(broadcast))
    return result,pairs,forced,gated


def load_mapping(path,nt,nb):
    mapping=np.full(nt,-1,np.int32);half=np.zeros(nt,np.int8);flips=np.zeros(nt,np.int8);n=0
    with Path(path).open(newline='') as f:
        for i,row in enumerate(csv.DictReader(f)):
            if i>=nt or int(row['tactical_frame'])!=i:raise ValueError('Mapping source axis invalid')
            tag=row['tactical_half_tag'];half[i]={'':0,'1st Half':1,'2nd Half':2}[tag]
            if row['broadcast_atd_frame0']:
                b=int(row['broadcast_atd_frame0']);flip=int(row['xy_reflection_index'])
                if not 0<=b<nb or not 0<=flip<len(FLIPS) or not half[i] or row['broadcast_half_tag']!=tag:raise ValueError('Mapping row invalid')
                mapping[i]=b;flips[i]=flip
            n+=1
    if n!=nt:raise ValueError('Mapping is incomplete')
    selected=np.flatnonzero(mapping>=0)
    if len(selected)>1 and (np.diff(mapping[selected])<=0).any():raise ValueError('Broadcast frames reused/reversed')
    return mapping,half,flips


def sha(path):
    h=hashlib.sha256()
    with Path(path).open('rb') as f:
        for b in iter(lambda:f.read(2**20),b''):h.update(b)
    return h.hexdigest()


def load_sources(data_dir=None):
    arrays={};teams={};paths={}
    for source in ('tactical','broadcast'):
        paths[source]=asset_paths(source=source,data_dir=data_dir);teams[source]=track_teams(paths[source]['atd_meta'])
        n=source_frames(source);arrays[source]=read_tracking(paths[source]['atd_raw'],teams[source],n,0,n)*PITCH.astype(np.float32)
    return arrays,teams,paths


def compute(arrays,teams,mapping,halves,flips,folder,gate):
    n=len(mapping);values=np.full((n,len(FIELDS)),np.nan);values[:,COL['frame']]=np.arange(n);values[:,COL['half']]=halves;values[:,COL['mapped']]=mapping>=0
    indices=np.full((n,len(teams['tactical'])),-1,np.int16);forced=np.full(indices.shape,np.nan,np.float32);gated=forced.copy();started=time.monotonic()
    for source,xy in arrays.items():
        if source=='tactical':
            for s in ('H','A'):values[:,COL[s+'_t_count']]=np.isfinite(xy[:,:-1][:,np.asarray(teams[source])==s]).all(-1).sum(1)
            values[:,COL['t_count']]=values[:,COL['H_t_count']]+values[:,COL['A_t_count']]
    for k,t in enumerate(np.flatnonzero(mapping>=0)):
        b=int(mapping[t]);bp=reflected(arrays['broadcast'][b],FLIPS[flips[t]],PITCH)
        row,pairs,fd,gd=compare_frame(arrays['tactical'][t],bp,teams['tactical'],teams['broadcast'],gate)
        values[t,COL['broadcast_frame']]=b
        for key,v in row.items():values[t,COL[key]]=v
        for a,z in pairs:indices[t,a]=z
        for a,d in fd:forced[t,a]=d
        for a,d in gd:gated[t,a]=d
        if k%10000==0:print('Compared paired frames',k,'elapsed_s',round(time.monotonic()-started,1),flush=True)
    np.save(folder/'frame_values.npy',values);np.save(folder/'gated_indices.npy',indices)
    np.save(folder/'forced_distances.npy',forced);np.save(folder/'gated_distances.npy',gated)
    (folder/'metric_fields.json').write_text(json.dumps(FIELDS))
    integer={k for k in FIELDS if k in ('frame','broadcast_frame','half','mapped') or any(k.endswith(x) for x in ('count','pairs','duplicates','_t','_b','pitch','minus_T'))}
    with gzip.open(folder/'frame_metrics.csv.gz','wt',newline='') as f:
        w=csv.writer(f);w.writerow(FIELDS)
        for row in values:w.writerow('' if not np.isfinite(v) else int(v) if key in integer else f'{v:.6f}' for key,v in zip(FIELDS,row))
    return values,indices,forced,gated


def distribution(x):
    x=np.asarray(x);x=x[np.isfinite(x)]
    if not len(x):return dict(n=0,mean=None,rmse=None,median=None,p95=None,max=None)
    return dict(n=len(x),mean=float(x.mean()),rmse=rmse(x),median=quantile(x,50),p95=quantile(x,95),max=float(x.max()))


def summarize(values,forced,gated,teams,folder):
    result={};mapped=values[:,COL['mapped']]>0
    for tag,choose in [('all',mapped),('1st Half',mapped&(values[:,COL['half']]==1)),('2nd Half',mapped&(values[:,COL['half']]==2))]:
        group={}
        for side in ('ALL','H','A'):
            slots=np.ones(len(teams['tactical']),bool) if side=='ALL' else np.asarray(teams['tactical'])==side
            prefix='' if side=='ALL' else side+'_'
            delta=values[choose,COL[prefix+'count_delta_B_minus_T']]
            group[side]=dict(forced_distance_m=distribution(forced[np.ix_(choose,slots)]),gated_distance_m=distribution(gated[np.ix_(choose,slots)]),
                mean_t_count=float(np.nanmean(values[choose,COL[prefix+'t_count']])),mean_b_count=float(np.nanmean(values[choose,COL[prefix+'b_count']])),
                mean_B_minus_T=float(np.nanmean(delta)),mean_absolute_count_difference=float(np.nanmean(abs(delta))),
                equal_count_fraction=float((delta==0).mean()),broadcast_fewer_fraction=float((delta<0).mean()))
        result[tag]=dict(frames=int(choose.sum()),teams=group,ball_distance_m=distribution(values[choose,COL['ball_distance_m']]),
            zero_gated_pairs_frames=int((values[choose,COL['gated_pairs']]==0).sum()),
            frames_with_forced_pairs=int((values[choose,COL['forced_pairs']]>0).sum()),
            frames_with_gated_pairs=int((values[choose,COL['gated_pairs']]>0).sum()),
            broadcast_zero_observation_frames=int((values[choose,COL['b_count']]==0).sum()),
            both_teams_observed_in_both_sources_frames=int(((values[choose,COL['H_t_count']]>0)&(values[choose,COL['A_t_count']]>0)&(values[choose,COL['H_b_count']]>0)&(values[choose,COL['A_b_count']]>0)).sum()),
            frame_mean_t_coverage=float(np.nanmean(values[choose,COL['gated_t_coverage']])),frame_mean_b_coverage=float(np.nanmean(values[choose,COL['gated_b_coverage']])))
    with (folder/'minute_summary.csv').open('w',newline='') as f:
        w=csv.writer(f);w.writerow(['half','tactical_minute','frames','forced_pair_rmse_m','gated_pair_rmse_m','mean_t_count','mean_b_count','mean_B_minus_T','ball_rmse_m'])
        for half in (1,2):
            for minute in np.unique(np.floor(values[mapped&(values[:,COL['half']]==half),COL['frame']]/1500).astype(int)):
                choose=mapped&(values[:,COL['half']]==half)&(values[:,COL['frame']]//1500==minute)
                w.writerow([half,int(minute),int(choose.sum()),distribution(forced[choose])['rmse'],distribution(gated[choose])['rmse'],
                    values[choose,COL['t_count']].mean(),values[choose,COL['b_count']].mean(),values[choose,COL['count_delta_B_minus_T']].mean(),distribution(values[choose,COL['ball_distance_m']])['rmse']])
    return result


def pixel(point):return (round(38+float(point[0])*7.7),round(118+float(point[1])*7.7))


def template():
    image=np.full((SIZE[1],SIZE[0],3),(30,25,20),np.uint8);white=(210,225,210)
    cv2.rectangle(image,pixel((0,0)),pixel(PITCH),(42,94,42),-1);cv2.rectangle(image,pixel((0,0)),pixel(PITCH),white,1)
    cv2.line(image,pixel((PITCH[0]/2,0)),pixel((PITCH[0]/2,PITCH[1])),white,1)
    cv2.circle(image,pixel(PITCH/2),round(9.15*7.7),white,1)
    for side in (0,1):
        for depth,width in ((16.5,40.3),(5.5,18.3)):
            a=0 if not side else PITCH[0]-depth;cv2.rectangle(image,pixel((a,(PITCH[1]-width)/2)),pixel((a+depth,(PITCH[1]+width)/2)),white,1)
    return image


def fmt(x):return f'{x:.2f}' if np.isfinite(x) else 'N/A'


def render(base,tp,bp,tt,bt,indices,row,t,b,half,gate,trace,index):
    image=base.copy()
    def text(value,x,y,color=(235,235,235),scale=.52):cv2.putText(image,value,(x,y),cv2.FONT_HERSHEY_SIMPLEX,scale,color,1,cv2.LINE_8)
    text(f'{"1st" if half==1 else "2nd"} Half | Tactical {t} ({t/25/60:.2f} min) | Broadcast {b} ({b/25/60:.2f} min)',25,28,scale=.63)
    text('Filled circle = TACTICAL   X = BROADCAST   Blue = Home / Red = Away',25,56)
    text(f'Yellow links = anonymous same-team pairs <= {gate:g}m; NOT verified player IDs',25,80,(65,210,245))
    for a,z in enumerate(indices):
        if z>=0:
            pa,pb=pixel(tp[a]),pixel(bp[z]);ok,p1,p2=cv2.clipLine((30,107,830,540),pa,pb)
            if ok:cv2.line(image,p1,p2,(55,205,245),1,cv2.LINE_8)
    for source,points,teams in [('T',tp,tt),('B',bp,bt)]:
        for k,p in enumerate(points):
            if not np.isfinite(p).all():continue
            x,y=pixel(p)
            if not 30<x<865 or not 107<y<650:continue
            color=(245,245,245) if k==len(teams) else (245,150,60) if teams[k]=='H' else (75,75,245)
            if source=='T':cv2.circle(image,(x,y),5,color,-1,cv2.LINE_8)
            else:cv2.drawMarker(image,(x,y),color,cv2.MARKER_TILTED_CROSS,13,2,cv2.LINE_8)
    def v(key):return row[COL[key]]
    y=124
    for line in ['OBSERVATION COUNTS  (B - T)',f'Tactical  H {int(v("H_t_count")):2d} / A {int(v("A_t_count")):2d} / All {int(v("t_count")):2d}',
        f'Broadcast H {int(v("H_b_count")):2d} / A {int(v("A_b_count")):2d} / All {int(v("b_count")):2d}',
        f'Difference H {int(v("H_count_delta_B_minus_T")):+d} / A {int(v("A_count_delta_B_minus_T")):+d} / All {int(v("count_delta_B_minus_T")):+d}',
        '', 'POSITION DISAGREEMENT (metres)',f'All-pair RMSE    {fmt(v("forced_rmse_m"))}',f'All-pair P95     {fmt(v("forced_p95_m"))}',
        f'Gate {gate:g}m pairs     {int(v("gated_pairs"))}',f'Gate RMSE / P95  {fmt(v("gated_rmse_m"))} / {fmt(v("gated_p95_m"))}',
        f'Home gate RMSE   {fmt(v("H_gated_rmse_m"))}',f'Away gate RMSE   {fmt(v("A_gated_rmse_m"))}',
        f'Coverage T / B   {fmt(100*v("gated_t_coverage"))}% / {fmt(100*v("gated_b_coverage"))}%',
        f'Unpaired T / B   {int(v("t_count")-v("gated_pairs"))} / {int(v("b_count")-v("gated_pairs"))}',
        f'Ball distance    {fmt(v("ball_distance_m"))}', '', 'Same-team assignment each frame.', 'No identity or ground-truth guarantee.', 'Large error may include time mismatch.', 'Constant-offset half map is unverified.']:
        text(line,884,y,scale=.43);y+=25
    # Native-frame trace, only inside this half; thin blue=count, yellow=forced RMSE.
    start=max(0,index-250);stop=min(len(trace),index+251);positions=np.arange(start,stop);positions=positions[trace[positions,2]==half]
    for dimension,origin,height,maximum,color,label in [(0,663,60,20,(55,205,245),'All-pair RMSE (plot clipped at20m)'),(1,663,60,22,(245,150,60),'B-T count (plot +/-22)')]:
        points=[]
        for j in positions[::5]:
            z=trace[j,dimension]
            if np.isfinite(z):points.append((round(38+(j-index+250)*1.62),round(origin+height-(np.clip(z,0,maximum)/maximum*height if dimension==0 else (np.clip(z,-22,22)+22)/44*height))))
        if len(points)>1:cv2.polylines(image,[np.asarray(points,np.int32)],False,color,1,cv2.LINE_8)
    cv2.line(image,(443,660),(443,729),(200,200,200),1);text('Past10s               now              Future10s  |  yellow=RMSE / blue=B-T count',38,746,scale=.43)
    return image


def export_video(arrays,teams,mapping,halves,flips,values,indices,folder,gate,limit=None):
    tf=np.flatnonzero(mapping>=0)
    if limit is not None:tf=tf[:limit]
    bf=mapping[tf];hs=halves[tf];path=folder/'tracking_disagreement.mp4';temporary=folder/'tracking_disagreement.partial.mp4'
    writer=cv2.VideoWriter(str(temporary),cv2.VideoWriter_fourcc(*'avc1'),25,SIZE)
    if not writer.isOpened():raise RuntimeError('Local H.264 encoder unavailable')
    base=template();trace=np.c_[values[tf,COL['forced_rmse_m']],values[tf,COL['count_delta_B_minus_T']],hs];started=time.monotonic()
    try:
        for i,(t,b,h) in enumerate(zip(tf,bf,hs)):
            bp=reflected(arrays['broadcast'][b],FLIPS[flips[t]],PITCH)
            writer.write(render(base,arrays['tactical'][t],bp,teams['tactical'],teams['broadcast'],indices[t],values[t],t,b,h,gate,trace,i))
            if (i+1)%10000==0:print('Rendered frames',i+1,'/',len(tf),'elapsed_s',round(time.monotonic()-started,1),flush=True)
    finally:writer.release()
    cap=cv2.VideoCapture(str(temporary))
    try:
        assert cap.isOpened() and round(cap.get(cv2.CAP_PROP_FRAME_COUNT))==len(tf) and abs(cap.get(cv2.CAP_PROP_FPS)-25)<.01
        for i in set([0,len(tf)-1,*list(np.flatnonzero(np.diff(hs)!=0)+1)]):
            cap.set(cv2.CAP_PROP_POS_FRAMES,int(i));ok,image=cap.read();assert ok and image.shape[:2]==(SIZE[1],SIZE[0])
    finally:cap.release()
    os.replace(temporary,path);segments=timeline_segments(tf,bf,hs)
    html=PLAYER_HTML.replace('전체 non-X 선수 이동 벡터','두 트래킹 위치 오차 및 선수 수 비교').replace('전체 프레임 선수 이동 벡터 · X 제외','Broadcast–Tactical 위치·인원 비교')
    html=html.replace('full_movement_vectors.mp4','tracking_disagreement.mp4').replace(' · 두 영상의 트래킹을 나란히 표시',' · 두 트래킹 좌표와 오차·관측 인원 비교')
    start=html.index('<p class="muted">파랑 Home');end=html.index('</p>',start)+4
    html=html[:start]+f'''<p class="muted">파랑 Home · 빨강 Away · 채운 원 Tactical · X표 Broadcast · 노란 선은 같은 팀 내 거리 제한을 만족한 기하학적 대응입니다.<br>
슬롯 번호와 대응선은 실제 선수 동일성을 보장하지 않습니다. 전체 대응 오차와 {gate:g}m 제한 오차·대응 비율을 함께 확인하세요.<br>
선수 수 차이는 Broadcast − Tactical입니다. 카메라 시야, 결측, 중복 관측, 팀 분류 및 시간 정렬 오차가 섞일 수 있습니다.<br>
이 영상은 원본 경기 영상이 아니라 제공된 두 ATD 좌표를 겹쳐 그린 비교 영상입니다. 기존 Half 일정 시차를 적용했으며 전체 동기화는 검증되지 않았습니다.<br>
←/→: 한 프레임 · J/L: 1초 · Space: 재생/정지. HTML과 MP4를 같은 폴더에 보관하세요. 서버·외부 통신 없이 로컬 파일만 사용합니다.<br>
<a href="tracking_disagreement.mp4" download>비교 MP4</a> · <a href="frame_metrics.csv.gz">전체199,319프레임 수치 CSV.gz</a> · <a href="minute_summary.csv">시간대별 비교</a> · <a href="REPORT.md">계산식과 요약</a></p>'''+html[end:]
    meta=dict(count=len(tf),fps=25,segments=segments);html=html.replace('__META__',json.dumps(meta,separators=(',',':')))
    (folder/'tracking_disagreement_player.html').write_text(html)
    with (folder/'frame_timeline.csv').open('w',newline='') as f:
        w=csv.writer(f);w.writerow(['video_frame','video_time_seconds','tactical_frame','broadcast_frame','half']);w.writerows((i,i/25,int(t),int(b),int(h)) for i,(t,b,h) in enumerate(zip(tf,bf,hs)))
    return dict(frames=len(tf),fps=25,duration_seconds=len(tf)/25,size=list(SIZE),segments=segments,render_seconds=round(time.monotonic()-started,2),decoded_samples_verified=True)


def write_report(summary,folder,gate):
    lines=['# Broadcast–Tactical 트래킹 불일치 비교','',
        '제공된 두 ATD 좌표를 기존 프레임 정렬표로 비교했다. 두 측정치의 불일치이며 어느 소스가 실제 위치에 더 정확한지는 알 수 없다.',
        '정렬표의 일정 시차도 전체 경기에서 검증되지 않았다. 시간 어긋남·리플레이·시야·좌표/팀 분류·중복 관측이 오차에 함께 반영될 수 있다.', '',
        '| 구간 | 팀 | 전체 대응 RMSE m | 전체 대응 중앙값 m | 전체 대응 P95 m | 제한 대응 RMSE m | T 관측 수 | B 관측 수 | B−T |',
        '| --- | --- | --- | --- | --- | --- | --- | --- | --- |']
    def f(x):return 'N/A' if x is None else f'{x:.3f}'
    for tag,g in summary['groups'].items():
        for side,v in g['teams'].items():
            a=v['forced_distance_m'];b=v['gated_distance_m'];lines.append(f'| {tag} | {side} | {f(a["rmse"])} | {f(a["median"])} | {f(a["p95"])} | {f(b["rmse"])} | {f(v["mean_t_count"])} | {f(v["mean_b_count"])} | {f(v["mean_B_minus_T"])} |')
    lines+=['','| 구간 | 매핑 프레임 | 거리 계산 가능 프레임 | B 관측0 프레임 | T 대응률 평균% | B 대응률 평균% | 공 공통관측 프레임 | 공 RMSE m |',
        '| --- | --- | --- | --- | --- | --- | --- | --- |']
    for tag,g in summary['groups'].items():
        lines.append(f'| {tag} | {g["frames"]} | {g["frames_with_forced_pairs"]} | {g["broadcast_zero_observation_frames"]} | {100*g["frame_mean_t_coverage"]:.2f} | {100*g["frame_mean_b_coverage"]:.2f} | {g["ball_distance_m"]["n"]} | {f(g["ball_distance_m"]["rmse"])} |')
    lines+=['','## 수치의 의미','',
        '- 좌표는 정규화 x/y에105.3m/68m를 곱했다. 기존 정렬표의 reflection만 적용하고 새 좌표 보정·보간·슬롯 재분류는 하지 않았다.',
        '- 전체 대응: 같은 팀의 작은 관측 집합을 큰 집합에 일대일 최소 제곱 거리로 대응한다. 서로 다른 선수를 강제로 대응할 수 있어 큰 차이도 포함한다.',
        f'- 제한 대응: 같은 팀 안에서≤{gate:g}m 대응 수를 최대화한 뒤 거리 제곱 합을 최소화한다. 범위 밖은 미대응으로 남긴다. 제한 RMSE는 잘 맞는 부분집합이므로 전체 추적 정확도가 아니다.',
        '- RMSE=sqrt(mean(dx²+dy²)), 중앙값/P95는 대응 쌍 거리의 경험적 분위수다. P95는95% 신뢰구간이나 보증된 허용 오차가 아니다. 구간 요약은 프레임 RMSE 평균이 아닌 대응 쌍을 합쳐 계산한다.',
        '- 대응 쌍이 하나도 없는 프레임은 거리 분포에서 제외되며 관측 수·커버리지 통계에는 남긴다. 대응 비율은 분모의 소스 관측 수가0이면 정의되지 않는다. summary.json에 거리 계산 가능 프레임과 Broadcast 관측0 프레임 수를 함께 기록한다.',
        '- signed dx/dy는 Broadcast−Tactical이다. 같은 팀 내0.5m 이내 경쟁 대응이 있는 쌍은 ambiguous로 집계한다. 대응은 프레임별 기하학적 관계이며 시간적으로 유지되는 선수 ID가 아니다.',
        '- 선수 수는 양 좌표가 유한한 슬롯 수다. 실제 출전 인원 수와 같다고 가정하지 않는다. 정확히 중복된 좌표와 경기장 밖 유한 관측도 별도 집계하고 삭제하지 않았다.',
        '- B−T가 음수면 Broadcast 관측이 더 적다. 시야 밖, 누락, 중복, 오분류를 숫자만으로 분리할 수 없다. 인원 수가 같아도 같은 선수를 관측했다고 가정하지 않는다.',
        '- 양 팀을 합쳐 매칭하지 않는다. 상대 팀으로 잘못 분류된 키퍼 등을 수작업으로 수정하지 않는다. 공 오차는 양쪽에서 관측된 같은 대응 프레임의 별도 거리다.',
        '- 매핑이 없는 프레임의 Broadcast 수와 오차는 결측으로 남긴다. 양쪽 레코드는 있지만 관측이 없으면 관측 수0, 거리 지표는 결측이다. 미대응을 거리0으로 대체하지 않는다.',
        f'- CSV.gz는 tactical 전체{summary["tactical_rows"]:,}행을 보존한다. 매핑된 프레임은{summary["mapped_rows"]:,}개이며 영상은'+(f'{summary["video"]["frames"]:,}프레임을25fps로 담았다.' if 'video' in summary else '생성하지 않았다.')+' 하프타임 등은 제외되며 frame_timeline.csv로 원본 프레임을 찾을 수 있다.',
        '- 실제 경기 영상/GT/Smart Tagging을 새로 읽지 않았다. 기존 정렬표는 과거 Smart Tagging 기반 구간 설정을 포함한다. 이번 작업에서는 해당 표를 변경하거나 새 태그를 사용하지 않았다.',
        '- 모든 계산·출력은 로컬이다. 서버·클라우드·외부 업로드·Git Commit/Push를 사용하지 않았다. 기존 데이터/정렬표/모델은 보존했다.', '',
        '## 보기와 재현','',
        '`tracking_disagreement_player.html`을 로컬 브라우저에서 연다. 슬라이더, ±1프레임, 전·후반 이동, Tactical 원본 프레임 검색을 사용할 수 있다. MP4와 HTML을 같은 폴더에 둔다.', '',
        '```bash','conda activate mtah2027','python starter/export_tracking_disagreement.py',
        'python -m pytest -q tests/test_tracking_disagreement.py','```','',
        'frame_metrics.csv.gz: 프레임·팀별 인원/차이, 전체/거리 제한 오차, dx/dy, 방향별 최근접 거리, 미대응·중복·경기장 밖 관측, 공 거리.',
        'minute_summary.csv: 원본 Tactical 영상 시간의1분 단위 요약. summary.json: 전체·전후반·팀별 분포, 관측 지원, 영상 검증과 provenance.',
        '영상 하단 그래프는 현재±10초의 시계열이다. 노란 선은 전체 대응 RMSE(표시 범위0~20m), 파란 선은 B−T 관측 수 차이(표시 범위−22~+22)이며 그래프 밖 값은 표시만 잘랐다. 우측 숫자와 CSV의 실제 값은 자르지 않았다.',
        'NumPy Hungarian만으로 실행 가능하다. 이번 실행은 기존 로컬 격리 SciPy를 PYTHONPATH로 재사용해 가속했고 Conda 환경을 변경하거나 새 패키지를 설치하지 않았다. assignment_backend.txt에 실제 계산 backend를 기록했다. 동률 대응은 backend/순서에 따라 달라질 수 있고 검증된 선수 ID가 아니다.',
        '보조 NPY는 같은 로컬 결과의 캐시이며 좌표 추출물과 함께 외부에 게시하면 안 된다.']
    (folder/'REPORT.md').write_text('\n'.join(lines)+'\n')


def main():
    p=argparse.ArgumentParser(description=__doc__.splitlines()[0]);p.add_argument('--mapping',type=Path,default=DEFAULT_MAP);p.add_argument('--output',type=Path,default=DEFAULT_OUTPUT)
    p.add_argument('--data-dir');p.add_argument('--gate-m',type=float,default=5);p.add_argument('--no-video',action='store_true');p.add_argument('--video-limit',type=int)
    args=p.parse_args()
    if not np.isfinite(args.gate_m) or args.gate_m<=0 or (args.video_limit is not None and args.video_limit<1):p.error('Invalid gate/video limit')
    args.output.mkdir(parents=True,exist_ok=True)
    arrays,teams,paths=load_sources(args.data_dir);mapping,halves,flips=load_mapping(args.mapping,len(arrays['tactical']),len(arrays['broadcast']))
    provenance=dict(mapping_sha256=sha(args.mapping),gate_m=args.gate_m,source={s:{k:sha(paths[s][k]) for k in ('atd_raw','atd_meta')} for s in paths},fields=FIELDS)
    marker=args.output/'computed.json'
    if marker.exists():
        if json.loads(marker.read_text())!=provenance:raise ValueError('Existing output has different inputs/config; use a new folder')
        values=np.load(args.output/'frame_values.npy',mmap_mode='r');indices=np.load(args.output/'gated_indices.npy',mmap_mode='r')
        forced=np.load(args.output/'forced_distances.npy',mmap_mode='r');gated=np.load(args.output/'gated_distances.npy',mmap_mode='r')
    else:
        values,indices,forced,gated=compute(arrays,teams,mapping,halves,flips,args.output,args.gate_m);marker.write_text(json.dumps(provenance,indent=2))
        (args.output/'assignment_backend.txt').write_text('SciPy linear_sum_assignment (existing local installation)' if FAST_ASSIGNMENT is not None else 'NumPy Hungarian fallback')
    summary=dict(groups=summarize(values,forced,gated,teams,args.output),tactical_rows=len(mapping),mapped_rows=int((mapping>=0).sum()),
        unmatched_tactical_rows=int((mapping<0).sum()),gate_m=args.gate_m,source_slot_ids_not_person_ids=True,full_half_sync_verified=False,
        GT_used=False,smart_tagging_read=False,source_video_read=False,provenance=provenance)
    if not args.no_video:summary['video']=export_video(arrays,teams,mapping,halves,flips,values,indices,args.output,args.gate_m,args.video_limit)
    elif (args.output/'summary.json').exists():
        prior=json.loads((args.output/'summary.json').read_text())
        if 'video' in prior:summary['video']=prior['video']
    summary['source_hashes_unchanged']=all(sha(paths[s][k])==provenance['source'][s][k] for s in paths for k in ('atd_raw','atd_meta')) and sha(args.mapping)==provenance['mapping_sha256']
    assert summary['source_hashes_unchanged'];(args.output/'summary.json').write_text(json.dumps(summary,indent=2));write_report(summary,args.output,args.gate_m)
    print(json.dumps(dict(tactical_rows=summary['tactical_rows'],mapped_rows=summary['mapped_rows'],overall=summary['groups']['all']['teams']['ALL'],source_hashes_unchanged=True)),flush=True)


if __name__=='__main__':main()
