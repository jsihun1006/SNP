"""Compare provided broadcast ATD ball with the user's corrected SNP ball exports.

The broadcast homography is read only to put the ATD ball into the source image.
No calibration is estimated, repaired, or overlaid as pitch lines on the footage.
SNP image observations and initially converted pitch coordinates stay unchanged.
"""
from pathlib import Path
import argparse
import csv
import json
import sys
import time

import cv2
import numpy as np
import pandas as pd

ROOT=Path(__file__).resolve().parents[1]
BASE=ROOT/'data/derived/ball_tracking_homography_20261116'
OUT=ROOT/'data/derived/ball_tracking_comparison_20261116'
VIDEO=ROOT/'data/broadcast/BAR-ATH_broadcast-001.mp4'
ATD=ROOT/'data/broadcast/BAR-ATH_broadcast_atd.txt'
HFILE=ROOT/'data/broadcast/BAR-ATH_broadcast_homography.csv'
N,FPS=164487,25
PITCH=np.array([105.3,68.])
FIELDS=['atd_x_m','atd_y_m','snp_x_m','snp_y_m','atd_pixel_x','atd_pixel_y','snp_pixel_x','snp_pixel_y','distance_m']
STATE_NAMES=['outside_export','missing','review_candidate','observed','interpolated']
OUTPUT_SIZE=(1280,520)
ATD_COLOR=(255,205,0)
SNP_COLOR=(0,170,255)


def distribution(values):
    v=np.asarray(values);v=v[np.isfinite(v)]
    return dict(n=len(v),median=float(np.median(v)) if len(v) else None,
        p95=float(np.percentile(v,95)) if len(v) else None,max=float(v.max()) if len(v) else None,
        gt5m=int((v>5).sum()))


def prepare():
    OUT.mkdir(parents=True,exist_ok=True)
    normal=[]
    with ATD.open() as stream:
        for i,line in enumerate(stream):
            frame,rest=line.split(':',1);assert int(frame)==i+1
            ball=rest.rsplit(':',1)[1].split(',')
            normal.append([float(ball[0]),float(ball[1])])
    assert len(normal)==164488 and not np.isfinite(normal[-1]).any()
    normal=np.asarray(normal[:N],float)
    source=pd.read_csv(BASE/'combined_corrected_ball_pitch.csv',low_memory=False)
    frame=source.source_frame.to_numpy(int)
    assert not source.source_frame.duplicated().any() and len(source)==155923
    values=np.full((N,len(FIELDS)),np.nan)
    values[:,:2]=normal*PITCH
    values[frame,2:4]=source[['pitch_x_m','pitch_y_m']].to_numpy()
    values[frame,6:8]=source[['pixel_x','pixel_y']].to_numpy()
    state=np.zeros(N,np.uint8);half=np.zeros(N,np.uint8)
    code=dict(missing=1,review_candidate=2,observed=3,interpolated=4)
    state[frame]=source.observation_kind.map(code).to_numpy(np.uint8)
    half[frame]=source.half.to_numpy(np.uint8)
    table=pd.read_csv(HFILE)
    assert np.array_equal(table.frame,np.arange(len(table)))
    matrices=table[[f'h{i}{j}' for i in range(3) for j in range(3)]].to_numpy().reshape(-1,3,3)[:N]
    hvalid=(table.valid.to_numpy()[:N]==1)&np.isfinite(matrices).all(axis=(1,2))
    ray=np.einsum('nij,nj->ni',matrices,np.column_stack([normal,np.ones(N)]))
    projectable=hvalid&np.isfinite(ray).all(axis=1)&(np.abs(ray[:,2])>1e-12)
    values[projectable,4:6]=ray[projectable,:2]/ray[projectable,2,None]
    both=np.isfinite(values[:,:4]).all(axis=1)
    values[both,8]=np.linalg.norm(values[both,:2]-values[both,2:4],axis=1)
    assert both.sum()==74646
    np.savez_compressed(OUT/'ball_comparison_arrays.npz',values=values,fields=FIELDS,
        source_frame=np.arange(N,dtype=np.int32),snp_state=state,half=half)
    metrics=pd.DataFrame(values,columns=FIELDS)
    metrics.insert(0,'source_frame',np.arange(N));metrics.insert(1,'video_seconds',np.arange(N)/FPS)
    metrics.insert(2,'half',half);metrics.insert(3,'snp_state',np.array(STATE_NAMES)[state])
    metrics['atd_pitch_available']=np.isfinite(values[:,:2]).all(axis=1)
    metrics['snp_pitch_available']=np.isfinite(values[:,2:4]).all(axis=1)
    metrics['atd_image_projection_available']=np.isfinite(values[:,4:6]).all(axis=1)
    metrics['snp_image_observation_available']=np.isfinite(values[:,6:8]).all(axis=1)
    metrics['distance_available']=both
    metrics.to_csv(OUT/'ball_comparison_metrics.csv',index=False)
    summary={}
    for label,choose in [('all_video',np.ones(N,bool)),('all_exports',half>0),('first_half',half==1),('second_half',half==2)]:
        v=values[choose];s=state[choose]
        summary[label]=dict(frames=int(choose.sum()),atd_pitch_points=int(np.isfinite(v[:,:2]).all(axis=1).sum()),
            snp_pitch_points=int(np.isfinite(v[:,2:4]).all(axis=1).sum()),
            snp_image_points=int(np.isfinite(v[:,6:8]).all(axis=1).sum()),
            distance_m=distribution(v[:,8]),observed_only_distance_m=distribution(v[s==3,8]),
            interpolated_distance_m=distribution(v[s==4,8]))
    summary.update(pitch_size_m=PITCH.tolist(),fps=25,source_frames=N,duration_seconds=N/25,
        source_frame_alignment='broadcast ATD frame_id - 1 = MP4 source_frame; no temporal shift',
        data_compared='Provided broadcast ATD ball vs original user-provided corrected SNP exports',
        new_calibration_used=False,ground_truth_error=False,
        snp_source_commits=dict(first_half='843c33078559fe095b96e5fd7ecd5c5a15c2a29e',second_half='39f6e1ddc8f7a2081f058a48f3adc384666bdeea'))
    (OUT/'summary.json').write_text(json.dumps(summary,indent=2)+'\n')
    print(json.dumps(summary,indent=2),flush=True)


def load():
    with np.load(OUT/'ball_comparison_arrays.npz') as a:return a['values'],a['snp_state'],a['half']


def stamp(frame):
    seconds,centis=divmod(frame*4,100);minutes,seconds=divmod(seconds,60);hours,minutes=divmod(minutes,60)
    return f'{hours:02d}:{minutes:02d}:{seconds:02d}.{centis:02d}'


class Renderer:
    def __init__(self):
        self.base=np.full((520,1280,3),22,np.uint8)
        self.scale=432/(PITCH[0]+16)
        self.origin=np.array([824+8*self.scale,76+8*self.scale])
        field=self.base
        cv2.rectangle(field,(824,76),(1256,376),(30,70,35),-1)
        for p in self.field_paths():
            xy=np.rint(self.origin+np.asarray(p)*self.scale).astype(np.int32)
            cv2.polylines(field,[xy],False,(210,215,210),1,cv2.LINE_AA)
        self.text(field,'PITCH COORDINATES (105.3 x 68 m)',824,68,.48)
        self.text(field,'ATD: cyan circle | SNP: orange cross',824,392,.43)

    def field_paths(self):
        length,width=PITCH
        paths=[[[0,0],[length,0],[length,width],[0,width],[0,0]],[[length/2,0],[length/2,width]]]
        for depth,area_width in [(16.5,40.32),(5.5,18.32)]:
            for x0,x1 in [(0,depth),(length,length-depth)]:
                paths.append([[x0,(width-area_width)/2],[x1,(width-area_width)/2],
                    [x1,(width+area_width)/2],[x0,(width+area_width)/2]])
        t=np.linspace(0,2*np.pi,120)
        paths.append(np.column_stack([length/2+9.15*np.cos(t),width/2+9.15*np.sin(t)]))
        return paths

    @staticmethod
    def text(image,text,x,y,size=.48,color=(235,235,235)):
        cv2.putText(image,text,(x,y),cv2.FONT_HERSHEY_SIMPLEX,size,color,1,cv2.LINE_AA)

    @staticmethod
    def number(value):return f'{value:,.2f}' if np.isfinite(value) else 'N/A'

    def marker(self,image,point,color,kind,label=None):
        xy=tuple(np.rint(point).astype(int))
        if kind=='circle':cv2.circle(image,xy,7,color,2,cv2.LINE_AA)
        else:cv2.drawMarker(image,xy,color,cv2.MARKER_TILTED_CROSS,13,2,cv2.LINE_AA)
        if label:
            x=max(3,min(image.shape[1]-52,xy[0]+9));y=max(13,min(image.shape[0]-8,xy[1]+(-10 if kind=='circle' else 16)))
            cv2.putText(image,label,(x,y),cv2.FONT_HERSHEY_SIMPLEX,.4,(10,10,10),3,cv2.LINE_AA)
            self.text(image,label,x,y,.4,color)

    def frame(self,image,frame,row,state,half):
        canvas=self.base.copy()
        photo=canvas[54:504,:800]
        photo[:]=cv2.resize(image,(800,450),interpolation=cv2.INTER_AREA)
        self.text(canvas,'BALL ONLY | Broadcast ATD vs SNP corrected ball',12,20,.57)
        self.text(canvas,f'frame {frame} | {stamp(frame)} | half {half or "-"} | SNP {STATE_NAMES[state]}',12,43,.49)
        image_points=row[4:8].reshape(2,2)
        visible=np.isfinite(image_points).all(axis=1)&(image_points[:,0]>=0)&(image_points[:,0]<1920)&(image_points[:,1]>=0)&(image_points[:,1]<1080)
        if visible.all():
            a,b=np.rint(image_points*(800/1920)).astype(int)
            cv2.line(photo,tuple(a),tuple(b),(230,230,230),1,cv2.LINE_AA)
        for j,(color,kind,name) in enumerate([(ATD_COLOR,'circle','ATD'),(SNP_COLOR,'cross','SNP-I' if state==4 else 'SNP')]):
            if visible[j]:self.marker(photo,image_points[j]*(800/1920),color,kind,name)
        self.text(canvas,'VIDEO: cyan circle = existing ATD | orange cross = SNP | I = interpolated',12,518,.43)
        pitch_points=row[:4].reshape(2,2)
        on_view=np.isfinite(pitch_points).all(axis=1)&(pitch_points[:,0]>=-8)&(pitch_points[:,0]<=PITCH[0]+8)&(pitch_points[:,1]>=-8)&(pitch_points[:,1]<=PITCH[1]+8)
        if on_view.all():
            a,b=np.rint(self.origin+pitch_points*self.scale).astype(int)
            cv2.line(canvas,tuple(a),tuple(b),(245,245,245),1,cv2.LINE_AA)
        for j,(color,kind,name) in enumerate([(ATD_COLOR,'circle','ATD'),(SNP_COLOR,'cross','SNP')]):
            if on_view[j]:self.marker(canvas,self.origin+pitch_points[j]*self.scale,color,kind)
            elif np.isfinite(pitch_points[j]).all():self.text(canvas,name+' outside diagram',830,88+j*18,.43,color)
        self.text(canvas,'ATD  x='+self.number(row[0])+'  y='+self.number(row[1])+' m',824,417,.46,ATD_COLOR)
        self.text(canvas,'SNP  x='+self.number(row[2])+'  y='+self.number(row[3])+' m',824,440,.46,SNP_COLOR)
        self.text(canvas,'DISTANCE: '+self.number(row[8])+' m',824,471,.70)
        self.text(canvas,'SNP: '+STATE_NAMES[state],824,494,.45)
        self.text(canvas,'Distance = disagreement, not GT error',824,516,.42)
        return canvas


def export(first=0,last=N,preview=False):
    cv2.setNumThreads(4)
    values,states,half=load();renderer=Renderer()
    cap=cv2.VideoCapture(str(VIDEO));assert cap.isOpened() and int(cap.get(7))==N and cap.get(5)==FPS
    folder=OUT/'snapshots';folder.mkdir(exist_ok=True)
    if preview:
        intervals=[(19066,19116),(98263,98313),(125007,125057),(152546,152596)]
        stem='ball_comparison_preview'
    else:intervals=[(first,last)];stem='ball_tracking_full_comparison'
    total=sum(b-a for a,b in intervals);target=OUT/f'{stem}.mp4';partial=OUT/f'{stem}.partial.mp4'
    writer=cv2.VideoWriter(str(partial),cv2.CAP_FFMPEG,cv2.VideoWriter_fourcc(*'avc1'),FPS,OUTPUT_SIZE)
    assert writer.isOpened()
    snapshots={19091,47145,98288,110104,110200,125032,151810,152571};written=0;start=time.monotonic()
    try:
        for a,b in intervals:
            cap.set(cv2.CAP_PROP_POS_FRAMES,a)
            for frame in range(a,b):
                ok,image=cap.read();assert ok,frame
                assert int(cap.get(cv2.CAP_PROP_POS_FRAMES))==frame+1
                result=renderer.frame(image,frame,values[frame],int(states[frame]),int(half[frame]))
                writer.write(result);written+=1
                if frame in snapshots:cv2.imwrite(str(folder/f'frame_{frame}.jpg'),result)
                if written%1000==0 or written==total:
                    elapsed=time.monotonic()-start;rate=written/elapsed
                    progress=dict(stage='rendering',written=written,total=total,source_frame=frame,fps=round(rate,2),
                        elapsed_seconds=round(elapsed,2),remaining_seconds=round((total-written)/rate,1))
                    (OUT/f'{stem}_progress.json').write_text(json.dumps(progress,indent=2)+'\n')
                    print(f'{written}/{total} ({written/total:.1%}); {rate:.1f} fps; remaining {(total-written)/rate:.0f}s',flush=True)
    finally:cap.release();writer.release()
    assert written==total
    check=cv2.VideoCapture(str(partial));assert int(check.get(7))==total and check.get(5)==FPS
    assert (int(check.get(3)),int(check.get(4)))==OUTPUT_SIZE;check.release();partial.replace(target)
    (OUT/f'{stem}_manifest.json').write_text(json.dumps(dict(video=str(target.resolve()),output_frames=written,
        source_intervals=intervals,fps=25,width=1280,height=520,duration_seconds=written/FPS,
        source_frame_equals_output_frame=(intervals==[(0,N)]),pitch_size_m=PITCH.tolist(),
        video_markers='ATD reprojected with original H; SNP raw observed image point',
        pitch_markers='ATD and original SNP pitch coordinates in metres; never clipped',
        distance='Euclidean pitch disagreement when both present; missing = NaN, not zero',
        calibration_changed=False,frame_sampling=False,bytes=target.stat().st_size,
        render_seconds=round(time.monotonic()-start,2)),indent=2)+'\n')
    print('Completed '+str(target),flush=True)


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('stage',choices=['prepare','preview','video'])
    a=p.parse_args()
    if a.stage=='prepare':prepare()
    elif a.stage=='preview':export(preview=True)
    else:export()


if __name__=='__main__':main()
