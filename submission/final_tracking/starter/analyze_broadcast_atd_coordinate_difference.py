"""Compare actual original/corrected ATD XY in common meters, without GT claims."""
from pathlib import Path
import hashlib
import json
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

ROOT=Path(__file__).resolve().parents[1]
BASE=ROOT/'outputs/broadcast_atd_from_csv_v1'
OUT=BASE/'coordinate_difference_analysis'
RAW=ROOT/'data/broadcast/BAR-ATH_broadcast_atd.txt'
CORRECTED=BASE/RAW.name
INPUT=ROOT/'data/derived/ball_temporal_ai_20261117/ball_homography_all_frames.csv'
PITCH=np.array([105.,68.]);FPS=25


def sha(path):
    h=hashlib.sha256()
    with Path(path).open('rb') as stream:
        for block in iter(lambda:stream.read(8*1024*1024),b''):h.update(block)
    return h.hexdigest()


def clean(value):
    if isinstance(value,dict):return {str(k):clean(v) for k,v in value.items()}
    if isinstance(value,(tuple,list)):return [clean(v) for v in value]
    if isinstance(value,np.ndarray):return clean(value.tolist())
    if isinstance(value,np.integer):return int(value)
    if isinstance(value,np.bool_):return bool(value)
    if isinstance(value,(float,np.floating)):return float(value) if np.isfinite(value) else None
    return value


def save(name,value):(OUT/name).write_text(json.dumps(clean(value),ensure_ascii=False,indent=2,allow_nan=False)+'\n',encoding='utf-8')


def read_ball(path):
    values=[]
    with path.open('rb') as stream:
        for f,line in enumerate(stream):
            body=line.rstrip(b'\r\n');prefix,ball=body.rsplit(b':',1)
            assert int(prefix.split(b':',1)[0])==f+1
            row=np.fromstring(ball.decode('ascii'),sep=',');assert len(row)==3
            values.append(row)
    result=np.asarray(values,float);assert result.shape==(164488,3);return result


def clock(frame):
    seconds=int(frame)//25
    return f'{seconds//3600:02d}:{seconds//60%60:02d}:{seconds%60:02d}.{int(frame)%25*4:02d}'


def stats(error,delta):
    n=len(error)
    if not n:return dict(frames=0)
    p=np.quantile(error,[.025,.25,.5,.75,.9,.95,.975,.99])
    return dict(frames=n,minimum_m=error.min(),mean_m=error.mean(),median_m=p[2],
        rms_distance_m=np.sqrt(np.mean(error*error)),p025_m=p[0],p25_m=p[1],p75_m=p[3],p90_m=p[4],p95_m=p[5],
        p975_m=p[6],p99_m=p[7],maximum_m=error.max(),
        central_95_percent_empirical_interval_m=[p[0],p[6]],interval_is_confidence_interval=False,
        signed_mean_dx_m=delta[:,0].mean(),signed_mean_dy_m=delta[:,1].mean(),
        mean_abs_dx_m=np.mean(abs(delta[:,0])),mean_abs_dy_m=np.mean(abs(delta[:,1])),
        rms_dx_m=np.sqrt(np.mean(delta[:,0]**2)),rms_dy_m=np.sqrt(np.mean(delta[:,1]**2)),
        absolute_dx_p95_m=np.quantile(abs(delta[:,0]),.95),absolute_dy_p95_m=np.quantile(abs(delta[:,1]),.95),
        threshold_exceedances={str(t):dict(frames=int((error>t).sum()),fraction=float((error>t).mean())) for t in [.5,1,2,3,5,10,20,50]})


def main():
    OUT.mkdir(parents=True,exist_ok=True)
    paths=[RAW,CORRECTED,INPUT,BASE/'corrected_ball_coordinates.csv']
    hashes={str(p.relative_to(ROOT)):sha(p) for p in paths}
    original=read_ball(RAW);corrected=read_ball(CORRECTED)
    audit=pd.read_csv(BASE/'corrected_ball_coordinates.csv',float_precision='round_trip',low_memory=False)
    assert np.array_equal(original[:,:2],audit[['original_ball_x','original_ball_y']],equal_nan=True)
    assert np.array_equal(corrected[:,:2],audit[['corrected_ball_x','corrected_ball_y']],equal_nan=True)
    assert np.array_equal(original[:,2],corrected[:,2],equal_nan=True)
    a=original[:,:2]*PITCH;b=corrected[:,:2]*PITCH
    valid_a=np.isfinite(a).all(axis=1);valid_b=np.isfinite(b).all(axis=1);comparable=valid_a&valid_b
    delta=b-a;distance=np.linalg.norm(delta,axis=1);distance[~comparable]=np.nan
    changed=comparable&audit.coordinate_changed.to_numpy(bool)
    filled=~valid_a&valid_b;lost=valid_a&~valid_b;both_missing=~valid_a&~valid_b
    assert not lost.any() and changed.sum()+filled.sum()==audit.coordinate_changed.sum()
    primary=pd.read_csv(INPUT,usecols=['source_frame','half'],low_memory=False)
    half=np.full(len(a),-1,int);half[:len(primary)]=primary.half.to_numpy(int)
    result=pd.DataFrame(dict(frame=np.arange(len(a)),atd_frame_id=np.arange(1,len(a)+1),
        video_time=np.array([clock(f) for f in range(len(a))]),time_sec=np.arange(len(a))/25,
        half=half,original_x_m=a[:,0],original_y_m=a[:,1],corrected_x_m=b[:,0],corrected_y_m=b[:,1],
        dx_m=delta[:,0],dy_m=delta[:,1],distance_m=distance,comparable=comparable,
        changed_existing_xy=changed,newly_filled_original_missing=filled,
        both_missing=both_missing,source_kind=audit.source_kind,source_is_prediction=audit.source_is_prediction,
        comparison_status=np.where(changed,'CHANGED_AND_COMPARABLE',np.where(comparable,'ORIGINAL_RETAINED_ZERO_DIFFERENCE',
            np.where(filled,'NEWLY_FILLED_NO_ORIGINAL_REFERENCE','BOTH_MISSING')))))
    result.to_csv(OUT/'per_frame_coordinate_difference.csv',index=False)
    cohorts=[]
    definitions=[('ALL_COMPARABLE',comparable),('CHANGED_EXISTING_ONLY',changed),('RETAINED_EXISTING_ONLY',comparable&~changed),
        ('CHANGED_EXISTING_FIRST_HALF',changed&(half==1)),('CHANGED_EXISTING_SECOND_HALF',changed&(half==2))]
    for label in audit.source_kind.unique():definitions.append(('CHANGED_'+label,changed&audit.source_kind.eq(label).to_numpy()))
    detailed={}
    for name,mask in definitions:
        record=stats(distance[mask],delta[mask]);detailed[name]=record
        cohorts.append(dict(cohort=name,**{k:v for k,v in record.items() if not isinstance(v,(dict,list))}))
    pd.DataFrame(cohorts).to_csv(OUT/'summary_by_cohort.csv',index=False)
    result.loc[changed].nlargest(100,'distance_m').to_csv(OUT/'largest_100_differences.csv',index=False)
    large=changed&(distance>10)
    starts=np.flatnonzero(large&~np.r_[False,large[:-1]]);ends=np.flatnonzero(large&~np.r_[large[1:],False])
    segments=[]
    for first,last in zip(starts,ends):
        g=result.iloc[first:last+1];peak=int(g.distance_m.idxmax())
        segments.append(dict(first_frame=int(first),last_frame=int(last),frames=int(last-first+1),
            duration_sec=(last-first+1)/25,first_video_time=clock(first),last_video_time=clock(last),
            mean_difference_m=float(g.distance_m.mean()),peak_difference_m=float(g.distance_m.max()),
            peak_frame=peak,peak_video_time=clock(peak),source_kinds=';'.join(sorted(g.source_kind.unique()))))
    pd.DataFrame(segments).sort_values('peak_difference_m',ascending=False).to_csv(OUT/'over_10m_segments.csv',index=False)
    maximum_frame=int(np.nanargmax(distance));summary=dict(
        method='Euclidean difference between original and corrected ATD XY in common105x68 meters',
        equation='sqrt(((corrected_x_norm-original_x_norm)*105)^2 + ((corrected_y_norm-original_y_norm)*68)^2)',
        total_ATD_rows=len(a),video_frames=164487,all_comparable_frames=int(comparable.sum()),
        changed_existing_comparable_frames=int(changed.sum()),retained_existing_comparable_frames=int((comparable&~changed).sum()),
        newly_filled_frames_excluded_from_distance=int(filled.sum()),both_missing_frames_excluded=int(both_missing.sum()),
        finite_original_coordinates_lost=int(lost.sum()),cohorts=detailed,
        maximum_difference=dict(frame=maximum_frame,atd_frame_id=maximum_frame+1,video_time=clock(maximum_frame),
            distance_m=distance[maximum_frame],source_kind=audit.source_kind.iloc[maximum_frame],
            original_xy_m=a[maximum_frame],corrected_xy_m=b[maximum_frame]),
        over_10m_contiguous_segments=len(segments),original_speed_channel_unchanged=True,
        ground_truth_available=False,true_location_accuracy_tested=False,original_ATD_assumed_ground_truth=False,
        quantiles_are_empirical_distribution_not_calibrated_confidence_intervals=True,
        source_hashes=hashes)
    save('coordinate_difference_summary.json',summary)
    fig,axes=plt.subplots(2,1,figsize=(13,8),constrained_layout=True)
    ix=np.flatnonzero(changed);x=distance[changed];statistics=detailed['CHANGED_EXISTING_ONLY']
    axes[0].scatter(ix/25/60,x,s=1.2,alpha=.35,color='#2878b5',rasterized=True)
    axes[0].axhline(statistics['p95_m'],color='#dd7d16',lw=1.2,ls='--',label=f"Changed-frame P95: {statistics['p95_m']:.2f} m")
    axes[0].scatter([maximum_frame/25/60],[distance[maximum_frame]],s=28,color='#d43c3c',zorder=5)
    axes[0].annotate(f"max {distance[maximum_frame]:.2f} m / f{maximum_frame}",
        (maximum_frame/25/60,distance[maximum_frame]),xytext=(-110,-22),textcoords='offset points',fontsize=9)
    axes[0].set(xlabel='Original video time (minutes)',ylabel='Coordinate difference (m)',title=f'Changed existing coordinates: {len(x):,} frames')
    axes[0].grid(alpha=.2);axes[0].legend(loc='upper left')
    for mask,label,color in [(comparable,'All comparable (including retained zero differences)','#828282'),(changed,'Changed existing only','#2878b5')]:
        values=np.sort(distance[mask]);axes[1].plot(values,np.arange(1,len(values)+1)/len(values)*100,label=label,color=color,lw=1.7)
    axes[1].axhline(95,color='#dd7d16',lw=1,ls='--');axes[1].set_ylim(0,101)
    axes[1].set(xlabel='Coordinate difference (m)',ylabel='Cumulative share of frames (%)',
        title=f"Changed frames: median {statistics['median_m']:.2f} m | mean {statistics['mean_m']:.2f} m | P95 {statistics['p95_m']:.2f} m")
    axes[1].grid(alpha=.2);axes[1].legend(loc='lower right')
    fig.suptitle('Original vs corrected Broadcast ATD — coordinate differences',fontsize=14)
    fig.savefig(OUT/'coordinate_difference.png',dpi=180);fig.savefig(OUT/'coordinate_difference.pdf');plt.close(fig)
    assert all(sha(ROOT/p)==signature for p,signature in hashes.items())
    save('verification.json',dict(passed=True,actual_original_and_corrected_TXT_read=True,all_164488_ATD_rows_compared=True,
        original_coordinate_CSV_and_corrected_coordinate_CSV_match_TXT=True,original_speed_channel_preserved=True,
        sources_unchanged=True,units_meters_105x68=True,missing_original_excluded_from_distance=True,
        true_accuracy_metrics_not_computed=True))
    (OUT/'README.md').write_text(f'''# 원본 ATD와 보정 ATD의 좌표 차이

정답 좌표가 없으므로 **두 데이터 사이의 거리 차이**를 계산했습니다. 기존 ATD를 실제 공 위치 정답으로 가정하지 않았습니다.
105×68m로 변환한 x/y의 유클리드 거리이며, 단위는 m입니다.

- 양쪽 좌표가 있는 전체 {int(comparable.sum()):,}프레임과 실제 변경된 {int(changed.sum()):,}프레임을 분리했습니다.
- 원본 ATD가 결측이어서 새로 채운 {int(filled.sum()):,}프레임은 이전 위치와의 차이를 계산할 수 없어 거리 통계에서 제외했습니다.
- 변경 프레임 중앙값 {statistics['median_m']:.3f}m, 평균 {statistics['mean_m']:.3f}m, P95 {statistics['p95_m']:.3f}m, 최대 {statistics['maximum_m']:.3f}m입니다.
- 최대 차이는 f{maximum_frame}, 원본 영상 {clock(maximum_frame)}입니다.

[분포·시간 그래프](coordinate_difference.png) · [PDF](coordinate_difference.pdf)

`per_frame_coordinate_difference.csv`: 모든 프레임 원본/보정 좌표, dx/dy/거리, 결측/신규 채움 구분.
`summary_by_cohort.csv`: 전체/변경만/전·후반/SNP·YOLO·기존 예측별 통계.
`largest_100_differences.csv`: 가장 큰 차이 100프레임.
`over_10m_segments.csv`: 차이 10m 초과의 연속 구간과 원본 영상 시간.
`coordinate_difference_summary.json`: 백분위와 임계값 초과 비율을 포함한 전체 지표.

백분위는 경험적 분포이며 보정된 신뢰구간이나 실제 위치 정확도 지표가 아닙니다. 원본과 보정 파일은 변경하지 않았습니다.
''',encoding='utf-8')
    print(json.dumps(clean(dict(changed=detailed['CHANGED_EXISTING_ONLY'],all_comparable=detailed['ALL_COMPARABLE'],maximum=summary['maximum_difference'])),ensure_ascii=False),flush=True)


if __name__=='__main__':main()
