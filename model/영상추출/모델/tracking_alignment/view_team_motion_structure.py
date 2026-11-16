"""Compute team motion structure on ALL non-X frames, with video and diagnostics.

    python model/영상추출/모델/tracking_alignment/view_team_motion_structure.py

Outputs private model/영상추출/추출csv/tracking_alignment/team_motion_structure/: one H.264 MP4, offline frame
player, frame timeline, per-frame/source/team gzip CSV, source overview PNGs and
parameter/coverage report. No event catalog or possession GT. Filtering affects
this diagnostic only; original positions, labels and previous videos are intact.
"""

from collections import Counter
from dataclasses import asdict
import csv
import gzip
import io
import json
import math
from pathlib import Path
import sys
import xml.etree.ElementTree as ET

import cv2
import numpy as np

ROOT = next(parent for parent in Path(__file__).resolve().parents
            if (parent / "games.py").is_file())
OUTPUT_ROOT = ROOT / "model/영상추출/추출csv/tracking_alignment"
sys.path.insert(0, str(ROOT))
from games import GAME, GAMES, asset_paths, source_frames  # noqa: E402
from model.영상추출.모델.tracking_alignment.tracking_utils import track_teams, read_mapping, source_state, trailing_mean  # noqa: E402
from model.영상추출.모델.tracking_alignment.compare_tracking_player import read_tracking, COLORS, pitch_point, panel_geometry  # noqa: E402
from model.영상추출.모델.tracking_alignment.view_all_movement_vectors import select_timeline, timeline_segments, export_movie, HTML as PLAYER_HTML  # noqa: E402
from model.영상추출.모델.tracking_alignment.team_motion_metrics import MotionConfig, filtered_velocities, measure_team  # noqa: E402

FOLDER = OUTPUT_ROOT / 'team_motion_structure'
PLOT_KEYS = ('convergence_score', 'focus_confidence', 'direction_coherence', 'pair_distance_rate_mps', 'hull_area_rate_m2ps')
OUTPUT_SIZE = (1280, 850)


def display(value, digits=2):
    return f'{value:.{digits}f}' if np.isfinite(value) else '-'


def overlay_structure(canvas, metrics, local_index):
    result = np.full((OUTPUT_SIZE[1], OUTPUT_SIZE[0], 3), 28, dtype=np.uint8)
    result[:canvas.shape[0]] = canvas
    cv2.rectangle(result, (0, 535), (1279, 580), (28, 28, 28), -1)
    cv2.putText(result, 'Thin: trailing 0.4 s slot velocity x 1 s | Thick: team mean velocity x 3 s | Star: accepted focus',
                (18, 551), cv2.FONT_HERSHEY_SIMPLEX, .45, (235, 235, 235), 1, cv2.LINE_AA)
    cv2.putText(result, 'C: convergence strength | Q: diagnostic confidence, NOT a probability | Negative pair/area rate: compression',
                (18, 573), cv2.FONT_HERSHEY_SIMPLEX, .43, (70, 200, 245), 1, cv2.LINE_AA)
    for panel, source in enumerate(('tactical', 'broadcast')):
        origin = panel*640+18
        for row, team in enumerate(('H', 'A')):
            data = metrics[source][team]
            value = lambda key: data[key][local_index]
            color = COLORS[team]
            cx, cy = value('centroid_x_m'), value('centroid_y_m')
            vx, vy = value('mean_vx_mps'), value('mean_vy_mps')
            if np.isfinite([cx, cy, vx, vy]).all():
                start = pitch_point(cx, GAMES[GAME]['pitch'][1]-cy, panel)
                scale = panel_geometry(panel)[2]
                end = (round(start[0]+vx*3*scale), round(start[1]-vy*3*scale))
                ok, first, last = cv2.clipLine((panel*640+4,104,632,428), start, end)
                if ok and first != last:
                    cv2.arrowedLine(result, first, last, (245,245,245), 5, cv2.LINE_AA, tipLength=.3)
                    cv2.arrowedLine(result, first, last, color, 3, cv2.LINE_AA, tipLength=.3)
                cv2.circle(result, start, 6, (245,245,245), 1, cv2.LINE_AA)
                cv2.putText(result, team+' AVG', (start[0]+8,start[1]+14), cv2.FONT_HERSHEY_SIMPLEX,.38,color,1,cv2.LINE_AA)
            qx, qy = value('focus_x_m'), value('focus_y_m')
            if np.isfinite([qx, qy]).all():
                point = pitch_point(qx, GAMES[GAME]['pitch'][1]-qy, panel)
                size = 24 if team == 'H' else 15
                cv2.drawMarker(result, point, (245,245,245), cv2.MARKER_STAR, size, 3, cv2.LINE_AA)
                cv2.drawMarker(result, point, color, cv2.MARKER_STAR, size, 1, cv2.LINE_AA)
                label = f'{team} FOCUS C={value("convergence_score"):.2f}'
                at = (min(max(point[0]+8, panel*640+5), (panel+1)*640-130),
                      min(max(point[1]+(-14 if team=='H' else 20),116),525))
                cv2.putText(result, label, at, cv2.FONT_HERSHEY_SIMPLEX,.34,(10,20,10),3,cv2.LINE_AA)
                cv2.putText(result, label, at, cv2.FONT_HERSHEY_SIMPLEX,.34,color,1,cv2.LINE_AA)
            y = 609+row*112
            lines = [f'{team} | v=({display(vx)},{display(vy)}) m/s | R={display(value("direction_coherence"))} | obs/valid/move={value("n_observed")}/{value("n_valid_vectors")}/{value("n_moving")}',
                     (f'Focus=({qx:.1f},{qy:.1f}) | C={value("convergence_score"):.2f} Q={value("focus_confidence"):.2f} | support {value("focus_support_count")}/{value("n_moving")}'
                      if np.isfinite(qx) else 'Focus: not estimated | '+str(value('focus_status'))),
                     f'Pair rate={display(value("pair_distance_rate_mps"))} m/s | Hull rate={display(value("hull_area_rate_m2ps"),1)} m2/s | A={display(value("valid_vector_hull_area_m2"),0)} m2',
                     f'Fit={display(value("focus_fit_rmse_m"),1)} m | geometry={display(value("focus_geometry_ratio"))} | ball d={display(value("focus_to_ball_distance_m"),1)} m']
            for offset, line in enumerate(lines):
                cv2.putText(result,line,(origin,y+offset*23),cv2.FONT_HERSHEY_SIMPLEX,.4,color if offset==0 else (230,230,230),1,cv2.LINE_AA)
    cv2.putText(result,'Observed slot subsets; identity/team errors remain. C/Q are engineering diagnostics, not possession labels.',
                (18,841),cv2.FONT_HERSHEY_SIMPLEX,.41,(180,180,180),1,cv2.LINE_AA)
    return result


def plot_diagnostics(folder, series, halves, fps):
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    x = np.arange(len(halves))/fps/60
    join = np.flatnonzero(np.diff(halves) != 0)+1
    for source in ('tactical','broadcast'):
        fig, axes = plt.subplots(4,2,figsize=(18,12),sharex=True,sharey='row',constrained_layout=True)
        for col,team in enumerate(('H','A')):
            color = '#246bd1' if team=='H' else '#dc493d'
            data = series[source][team]
            axes[0,col].set_title(source.upper()+' | '+('HOME Barcelona' if team=='H' else 'AWAY Athletic Club'))
            for half in (1,2):
                indices = np.flatnonzero(halves==half)
                for key,row,label in [('convergence_score',0,'C: convergence'),('focus_confidence',0,'Q: confidence'),
                                      ('direction_coherence',1,'R: direction coherence'),('pair_distance_rate_mps',2,'Pair distance rate'),('hull_area_rate_m2ps',3,'Hull area rate')]:
                    values = data[key][indices]
                    style = '--' if key=='focus_confidence' else '-'
                    axes[row,col].plot(x[indices],values,color=color,alpha=.16,lw=.4,rasterized=True)
                    axes[row,col].plot(x[indices],trailing_mean(values,int(fps)),style,color='#555' if key=='focus_confidence' else color,
                                       lw=.8,label=label if half==1 else None,rasterized=True)
            for row in range(4):
                axes[row,col].grid(alpha=.2)
                for index in join:
                    axes[row,col].axvline(x[index],color='#555',ls=':',lw=.8)
                if row<2:
                    axes[row,col].set_ylim(-.03,1.03)
                else:
                    axes[row,col].axhline(0,color='#444',lw=.6)
            axes[0,col].legend(fontsize=9)
            axes[3,col].set_xlabel('Playback minutes (X removed; dotted line = half boundary)')
        for row,label in enumerate(['Focus C / Q (not probabilities)','Direction coherence R','Pair distance rate (m/s)','Hull area rate (m²/s)']):
            axes[row,0].set_ylabel(label)
        fig.suptitle('Team motion structure | All non-X frames | Faint: frame values; solid: trailing 1 s display mean\n'
                     'Unavailable focus stays missing; negative distance/area rate = compression; observed subsets only',fontsize=13)
        fig.savefig(folder/f'{source}_motion_structure.png',dpi=160)
        plt.close(fig)


def main():
    config=MotionConfig()
    try:
        fps,pitch=GAMES[GAME]['fps'],GAMES[GAME]['pitch']
        states={s:source_state(OUTPUT_ROOT / f'sync/final_tracking_{s}_labels.csv',source_frames(s)) for s in ('tactical','broadcast')}
        mapping=read_mapping(OUTPUT_ROOT / 'sync/final_tracking_alignment.csv',source_frames('tactical'),source_frames('broadcast'))
        tf,bf,halves=select_timeline(mapping,states)
        positions,teams,velocities,rejected={},{},{},{}
        for source in ('tactical','broadcast'):
            paths=asset_paths(source=source)
            teams[source]=track_teams(paths['atd_meta'])
            positions[source]=read_tracking(paths['atd_raw'],teams[source],source_frames(source),0,source_frames(source))
            velocities[source],rejected[source]=filtered_velocities(positions[source],states[source],fps,pitch,config)
            print(f'Prepared local {source} motion estimates; no private coordinate rows logged.',flush=True)
        FOLDER.mkdir(parents=True,exist_ok=True)
        series={s:{t:{k:np.full(len(tf),np.nan) for k in PLOT_KEYS} for t in ('H','A')} for s in ('tactical','broadcast')}
        counters={s:{t:Counter() for t in ('H','A')} for s in ('tactical','broadcast')}
        chunk={}
        csv_path=FOLDER/'team_motion_structure.csv.gz'
        temporary=FOLDER/'.team_motion_structure.partial.csv.gz'
        base_columns=['video_frame','video_time_seconds','tactical_frame','broadcast_atd_frame0','half_tag','source','source_frame','team']
        with temporary.open('wb') as binary, \
             gzip.GzipFile(filename='',mode='wb',fileobj=binary,compresslevel=1,mtime=0) as compressed, \
             io.TextIOWrapper(compressed,encoding='utf-8',newline='') as text:
            writer=None
            def prepare(packed,first,end):
                nonlocal writer
                chunk['first']=first
                metrics={}
                for source,frames in [('tactical',tf[first:end]),('broadcast',bf[first:end])]:
                    valid=frames>=0
                    packed[source][:,:-1,2:]=np.nan
                    packed[source][valid,:-1,2:]=velocities[source][frames[valid]]
                    metrics[source]={}
                    for team in ('H','A'):
                        mask=np.asarray(teams[source])==team
                        bad=np.zeros((end-first,int(mask.sum())),dtype=bool)
                        bad[valid]=rejected[source][frames[valid]][:,mask]
                        result=measure_team(packed[source][:,:-1,:2][:,mask]*np.asarray(pitch),
                                            packed[source][:,:-1,2:][:,mask],packed[source][:,-1,:2]*np.asarray(pitch),pitch,bad,config)
                        metrics[source][team]=result
                        if writer is None:
                            writer=csv.writer(text)
                            writer.writerow(base_columns+list(result))
                        for key in PLOT_KEYS:
                            series[source][team][key][first:end]=result[key]
                        counters[source][team].update(str(s) for s in result['focus_status'])
                        for i,index in enumerate(range(first,end)):
                            values=[]
                            for value in result.values():
                                item=value[i]
                                values.append(str(item) if isinstance(item,(str,np.str_)) else
                                              (format(float(item),'.9g') if np.isfinite(item) else ''))
                            writer.writerow([index,f'{index/fps:.6f}',int(tf[index]),int(bf[index]) if bf[index]>=0 else '',
                                             '1st Half' if halves[index]==1 else '2nd Half',source,int(frames[i]) if frames[i]>=0 else '',team]+values)
                chunk['metrics']=metrics
                return packed
            movie=export_movie(positions,teams,states,tf,bf,halves,FOLDER/'team_motion_structure.mp4',fps,
                               packed_transform=prepare,
                               frame_overlay=lambda canvas,index:overlay_structure(canvas,chunk['metrics'],index-chunk['first']),
                               output_size=OUTPUT_SIZE)
        temporary.replace(csv_path)
        meta={'count':len(tf),'fps':fps,'segments':timeline_segments(tf,bf,halves)}
        html=PLAYER_HTML.replace('full_movement_vectors.mp4','team_motion_structure.mp4')
        html=html.replace('전체 non-X 선수 이동 벡터','팀별 수렴·압축 구조').replace('전체 프레임 선수 이동 벡터 · X 제외','팀별 수렴·압축 구조 · 전체 non-X')
        html=html.replace('<p class="muted">','<p class="muted">별 마커: 유효한 추정 수렴점 · 굵은 화살표: 팀 평균 이동 · C: 수렴 강도 · Q: 진단용 신뢰도(확률 아님)<br>개별 화살표는 0.4초 연속 관측 평균입니다. 평행 이동·관측 부족 등은 수렴점을 표시하지 않습니다.<br>R은 같은 방향 이동 정도이며, Pair/Hull rate는 음수면 압축·양수면 확산입니다. 방송은 보이는 선수 부분집합의 지표입니다.<br><a href="tactical_motion_structure.png">Tactical 지표 그래프</a> · <a href="broadcast_motion_structure.png">Broadcast 지표 그래프</a> · <a href="team_motion_structure.csv.gz">전체 수치 압축 CSV</a><br>')
        (FOLDER/'team_motion_structure_player.html').write_text(html.replace('__META__',json.dumps(meta)),encoding='utf-8')
        with (FOLDER/'frame_timeline.csv').open('w',newline='',encoding='utf-8') as handle:
            writer=csv.writer(handle);writer.writerow(['video_frame','video_time_seconds','tactical_frame','broadcast_atd_frame0','half_tag'])
            writer.writerows((i,f'{i/fps:.6f}',int(t),int(b) if b>=0 else '', '1st Half' if half==1 else '2nd Half') for i,(t,b,half) in enumerate(zip(tf,bf,halves)))
        plot_diagnostics(FOLDER,series,halves,fps)
        report={**movie,'parameters':asdict(config),'segments':meta['segments'],'focus_status_counts':counters,
                'rows':len(tf)*4,'event_catalog_used':False,'possession_gt_used':False,
                'definition':'IRLS line fit + forward ray support; coherence separate; analytic pair/hull rates on current valid subset',
                'confidence_note':'Engineering quality index, not a probability; thresholds not fitted to public GT.',
                'tracking_note':'Supplied source-specific slots/team labels; all identity switches/duplicates are not resolved.',
                'broadcast_scope':'Observed camera subset, not necessarily the full team.','full_half_synchronization_verified':False}
        (FOLDER/'report.json').write_text(json.dumps(report,indent=2)+'\n',encoding='utf-8')
    except (OSError,ValueError,KeyError,IndexError,TypeError,csv.Error,cv2.error,ET.ParseError):
        print('Local team motion export failed; no private data rows logged.',file=sys.stderr)
        return 1
    print(json.dumps({'frames':len(tf),'rows':len(tf)*4,'focus_status_counts':counters}),flush=True)
    print('Saved local full-range team motion video, frame player, plots and metrics.',flush=True)
    return 0


if __name__=='__main__':
    raise SystemExit(main())
