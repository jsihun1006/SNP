"""Local tactical-video debugger. No human labels, model tuning or network server.

python -m competition_judgement.view_candidate_video
python -m competition_judgement.view_candidate_video --check
python -m competition_judgement.view_candidate_video --export-preview --seconds 12
"""
import argparse
import json
from pathlib import Path
import time

import cv2
import numpy as np
import pandas as pd
from games import GAME, GAMES, asset_paths
from starter.view_team_extents import default_video, load_homographies, read_frame
from .candidate_decision_v23 import KEY, STATES, CANDIDATES

ROOT = Path(__file__).resolve().parents[3]
RUN = ROOT/'data/derived/competition_judgement/candidate_v23_final_20261110'
WINDOW = 'MTAH candidate video - local debugger'
FILTERS = {'0': CANDIDATES, '1': ('PATTERN_CANDIDATE',), '2': ('LOCAL_CANDIDATE',),
           '3': ('REVIEW',), '4': ('NO_DETECTED_SIGNAL',), '5': ('INSUFFICIENT',), '6': STATES}
COLORS = {'PATTERN_CANDIDATE': (30,180,255), 'LOCAL_CANDIDATE': (230,100,225),
          'REVIEW': (230,190,70), 'NO_DETECTED_SIGNAL': (170,170,170), 'INSUFFICIENT': (90,90,190)}


def navigation_starts(frames, allowed):
    """Contiguous display ranges only, NOT tracked clusters or contest episodes."""
    active = frames.decision_state.isin(allowed).to_numpy(copy=True)
    record = frames.current_record_valid.to_numpy(bool)
    active &= record
    ids = frames.frame.to_numpy(int)
    gap = np.r_[True, np.diff(ids) != 1]
    # Source half/split boundaries cannot be visually stitched into a range.
    for key in ('period','split'):
        if key in frames: gap |= np.r_[False, frames[key].to_numpy()[1:] != frames[key].to_numpy()[:-1]]
    return ids[active & (gap | np.r_[True, ~active[:-1]])]


def jump_to(starts, frame, direction, context, prior_target=None):
    """Do not repeatedly land on the same target when showing its leading context."""
    reference = prior_target if prior_target is not None and prior_target-context <= frame <= prior_target else frame
    position = np.searchsorted(starts, reference, side='right') if direction > 0 else np.searchsorted(starts, reference, side='left')-1
    if not 0 <= position < len(starts): return None
    target = int(starts[position])
    return max(0, target-context), target


def project_anchor(matrix, center_m, radius_m, pitch):
    """Normalized pitch -> source pixels. Circle is analysis region, not a ball ROI."""
    if not np.isfinite(matrix).all() or np.linalg.matrix_rank(matrix) < 3: return None
    if not np.isfinite(center_m).all() or not np.isfinite(radius_m) or radius_m <= 0: return None
    theta = np.linspace(0, 2*np.pi, 65)
    xy = np.asarray(center_m)[None,:] + radius_m*np.column_stack([np.cos(theta),np.sin(theta)])
    xy = np.vstack([xy, center_m]) / np.asarray(pitch)
    homogeneous = np.column_stack([xy,np.ones(len(xy))]) @ matrix.T
    denominator = homogeneous[:,2]
    if not ((denominator > 1e-10).all() or (denominator < -1e-10).all()): return None
    pixels = homogeneous[:,:2]/denominator[:,None]
    if not np.isfinite(pixels).all() or (np.abs(pixels)>2**28).any(): return None
    return np.rint(pixels).astype(np.int32)


def basis_text(mask):
    return '+'.join(name for bit,name in [(1,'A'),(2,'B'),(4,'RF-short'),(8,'RF-near')] if int(mask)&bit) or 'none'


def load_tables(run=RUN):
    columns=['frame','decision_state','decision_reason','current_strict','current_record_valid',
             'representative_radius_m','representative_cluster_id','representative_basis_mask','decision_future_frames']
    frames = pd.read_csv(run/'frame_candidate_decisions.csv.gz',usecols=columns)
    n=GAMES[GAME]['n_video_frames']
    if len(frames)!=n or not np.array_equal(frames.frame,np.arange(n)) or not frames.decision_state.isin(STATES).all():
        raise ValueError('Expected every tactical frame in order and valid v2-3 states')
    # Read-only period/split metadata; no ball, tagging, labels or extra alignment table.
    context=pd.read_csv(ROOT/'data/derived/competition_judgement/local_v2_1_20261109/local_frame_features.csv',usecols=['frame','period'])
    if not np.array_equal(context.frame,frames.frame):raise ValueError('Period frame axis mismatch')
    frames['period']=context.period
    manifest=json.loads((ROOT/'data/derived/relational_temporal_v1_20261109/split_manifest.json').read_text())
    frames['split']='outside'
    for block in manifest['blocks']:
        frames.loc[frames.frame.between(block['first'],block['last']),'split']=block['split']
    decisions=pd.read_csv(run/'cluster_candidate_decisions.csv.gz',usecols=KEY+['decision_state','spatial_gate_valid','decision_basis_mask'])
    if decisions.duplicated(KEY).any():raise ValueError('Duplicate anchor keys')
    reps=frames[['frame','representative_radius_m','representative_cluster_id']].dropna().rename(columns={'representative_radius_m':'radius_m','representative_cluster_id':'cluster_id'})
    reps['representative']=True
    # Retain current representative plus actual candidate views; no 3m/5m event dedup.
    decisions=decisions.merge(reps,on=KEY,how='left',validate='one_to_one')
    decisions['representative']=decisions.representative.eq(True)
    decisions=decisions[decisions.representative | decisions.decision_state.isin(CANDIDATES)]
    geometry=pd.read_csv(ROOT/'data/derived/competition_judgement/local_v2_1_20261109/local_clusters.csv',
                         usecols=KEY+['cluster_centroid_x','cluster_centroid_y','cluster_max_pairwise_distance_m'])
    views=decisions.merge(geometry,on=KEY,how='left',validate='one_to_one',indicator=True)
    if (views['_merge']!='both').any():raise ValueError('Unresolved anchor geometry')
    views=views.drop(columns='_merge').sort_values(KEY).reset_index(drop=True)
    groups=views.groupby('frame').indices
    return frames, views, groups


def draw(image, row, views, matrix, *, width=1280, all_candidates=False, radius_filter=0, navigation_filter='0'):
    canvas=image.copy();height,original_width=canvas.shape[:2];pitch=GAMES[GAME]['pitch']
    shown=0
    for anchor in views.itertuples(index=False):
        if not anchor.spatial_gate_valid:continue
        if radius_filter and anchor.radius_m!=radius_filter:continue
        if not anchor.representative and not all_candidates:continue
        pixels=project_anchor(matrix,[anchor.cluster_centroid_x,anchor.cluster_centroid_y],
                              max(6.,anchor.cluster_max_pairwise_distance_m+3.),pitch) if matrix is not None else None
        if pixels is None:continue
        color=COLORS[anchor.decision_state];points=pixels[:-1];center=tuple(map(int,pixels[-1]))
        for a,b in zip(points[:-1],points[1:]):
            visible,a,b=cv2.clipLine((0,0,original_width,height),tuple(map(int,a)),tuple(map(int,b)))
            if visible:cv2.line(canvas,a,b,color,3 if anchor.representative else 1,cv2.LINE_AA)
        if 0<=center[0]<original_width and 0<=center[1]<height:
            cv2.circle(canvas,center,6,color,-1)
            label=f'r{int(anchor.radius_m)} c{int(anchor.cluster_id)} {basis_text(anchor.decision_basis_mask)}'
            cv2.putText(canvas,label,(max(0,min(center[0]+8,original_width-330)),max(25,min(center[1]-12,height-5))),cv2.FONT_HERSHEY_SIMPLEX,.6,color,2,cv2.LINE_AA)
        shown+=1
    resized=cv2.resize(canvas,(width,round(height*width/original_width)))
    panel=np.full((180,width,3),24,np.uint8);frame=int(row['frame']);state=row['decision_state']
    lines=[(f'v2-3: {state} | frame {frame} (0-based) | video {frame/25/60:.2f} min',COLORS[state]),
           (f"Reason: {row['decision_reason']} | QC {'strict' if row['current_strict'] else 'insufficient'} | {row['split']}",(225,225,225)),
           (f"Representative r={row['representative_radius_m']:g} id={row['representative_cluster_id']:g} | basis {basis_text(row['representative_basis_mask'])} | future dependency {int(row['decision_future_frames'])}f",(225,225,225)),
           (f'Analysis regions: {shown} | view {"all candidates + representative" if all_candidates else "representative"} | radius {radius_filter or "both"}',(190,190,190)),
           ('OFFLINE GEOMETRIC CANDIDATES - NOT VERIFIED CONTEST / NON-CONTEST LABELS',(110,205,250)),
           (f'Space play | A/D frame | J/L 1s | N/P filter {navigation_filter} | 0..6 filter | V all | R radius | Q quit',(190,190,190))]
    for j,(text,color) in enumerate(lines):cv2.putText(panel,text,(12,25+j*28),cv2.FONT_HERSHEY_SIMPLEX,.53,color,1,cv2.LINE_AA)
    return np.vstack([panel,resized])


def validate_video(cap,n):
    if not cap.isOpened():raise ValueError('Cannot open local tactical video')
    if int(cap.get(cv2.CAP_PROP_FRAME_COUNT))!=n or not np.isclose(cap.get(cv2.CAP_PROP_FPS),25.,atol=.01):
        raise ValueError('Tactical video and decisions must have identical frame count and 25fps')


def render_frame(cap,frame,frames,views,groups,matrices,valid,**kwargs):
    image=read_frame(cap,frame)
    local=views.iloc[groups.get(frame,np.array([],int))]
    return draw(image,frames.iloc[frame],local,matrices[frame] if valid[frame] else None,**kwargs)


def export_preview(cap,start,count,output,frames,views,groups,matrices,valid):
    output=output.resolve();allowed=ROOT/'data/derived/competition_judgement'
    if not output.is_relative_to(allowed) or output.suffix.lower()!='.mp4':raise ValueError('Preview must stay in local data/derived/competition_judgement/*.mp4')
    if output.exists() or output.with_suffix('.json').exists():raise FileExistsError('Refusing to overwrite preview')
    end=min(len(frames),start+count);first=render_frame(cap,start,frames,views,groups,matrices,valid)
    output.parent.mkdir(parents=True,exist_ok=True)
    writer=cv2.VideoWriter(str(output),cv2.VideoWriter_fourcc(*'avc1'),25.,(first.shape[1],first.shape[0]))
    if not writer.isOpened():raise RuntimeError('H.264 encoder unavailable in this OpenCV build')
    try:
        writer.write(first)
        for frame in range(start+1,end):writer.write(render_frame(cap,frame,frames,views,groups,matrices,valid))
    finally:writer.release()
    probe=cv2.VideoCapture(str(output))
    try:
        if not probe.isOpened() or int(probe.get(cv2.CAP_PROP_FRAME_COUNT))!=end-start:raise RuntimeError('Preview frame-count verification failed')
    finally:probe.release()
    report=dict(start_frame=start,end_frame_inclusive=end-1,frames=end-start,fps=25,source='tactical',
                local_only=True,automatic_geometric_overlay=True,human_labels=False,actual_contest_validation=False)
    output.with_suffix('.json').write_text(json.dumps(report,indent=2))
    return report


def play(cap,start,frames,views,groups,matrices,valid):
    cv2.namedWindow(WINDOW,cv2.WINDOW_NORMAL);cv2.resizeWindow(WINDOW,1280,940)
    cv2.createTrackbar('Frame',WINDOW,start,len(frames)-1,lambda value:None)
    frame=start;playing=False;dirty=True;filter_key='0';all_candidates=False;radius=0;prior=None;tick=time.monotonic()
    try:
        while True:
            if dirty:
                cv2.imshow(WINDOW,render_frame(cap,frame,frames,views,groups,matrices,valid,all_candidates=all_candidates,radius_filter=radius,navigation_filter=filter_key))
                cv2.setTrackbarPos('Frame',WINDOW,frame);dirty=False
            key=cv2.waitKeyEx(10)
            if cv2.getWindowProperty(WINDOW,cv2.WND_PROP_VISIBLE)<1 or key in (27,ord('q')):break
            position=cv2.getTrackbarPos('Frame',WINDOW)
            if position!=frame:frame=position;playing=False;prior=None;dirty=True
            if key==ord(' '):playing=not playing;tick=time.monotonic()
            if key in [ord(x) for x in 'adjl']:
                delta={ord('a'):-1,ord('d'):1,ord('j'):-25,ord('l'):25}[key]
                frame=max(0,min(len(frames)-1,frame+delta));playing=False;prior=None;dirty=True
            if key in [ord(x) for x in FILTERS]:filter_key=chr(key);prior=None;dirty=True
            if key in (ord('n'),ord('p')):
                jump=jump_to(navigation_starts(frames,FILTERS[filter_key]),frame,1 if key==ord('n') else -1,50,prior)
                if jump is not None:frame,prior=jump;playing=False;dirty=True
            if key==ord('v'):all_candidates=not all_candidates;dirty=True
            if key==ord('r'):radius={0:3,3:5,5:0}[radius];dirty=True
            if playing and time.monotonic()>=tick:
                if frame<len(frames)-1:frame+=1;dirty=True;tick=max(tick+1/25,time.monotonic())
                else:playing=False
    finally:cv2.destroyWindow(WINDOW)


def main():
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--video',type=Path,default=default_video())
    p.add_argument('--homography',type=Path,default=Path(asset_paths(source='tactical')['homography']))
    p.add_argument('--run-dir',type=Path,default=RUN);p.add_argument('--start-frame',type=int)
    p.add_argument('--check',action='store_true');p.add_argument('--export-preview',action='store_true');p.add_argument('--seconds',type=float,default=12.)
    p.add_argument('--output',type=Path,default=ROOT/'data/derived/competition_judgement/candidate_video_debug/candidate_preview.mp4')
    args=p.parse_args();frames,views,groups=load_tables(args.run_dir)
    candidate=frames.decision_state.isin(CANDIDATES)&frames.split.eq('validation')
    start=args.start_frame if args.start_frame is not None else max(0,int(frames.loc[candidate,'frame'].iloc[0])-50)
    if not 0<=start<len(frames):raise ValueError('Start frame outside tactical video')
    matrices,valid=load_homographies(args.homography,len(frames))
    cap=cv2.VideoCapture(str(args.video))
    try:
        validate_video(cap,len(frames))
        if args.check:
            # Decode selected points locally without displaying/exporting any match image.
            probes=sorted(set([0,start,len(frames)-1]))
            for frame in probes:render_frame(cap,frame,frames,views,groups,matrices,valid)
            print(json.dumps(dict(frame_rows=len(frames),video_frames=int(cap.get(cv2.CAP_PROP_FRAME_COUNT)),fps=25,decoded_checks=len(probes),homography_rows=len(valid),gui_tested=False,labels_created=False)))
        elif args.export_preview:
            if not np.isfinite(args.seconds) or args.seconds<=0:raise ValueError('Preview seconds must be positive')
            print(json.dumps(export_preview(cap,start,max(1,round(args.seconds*25)),args.output,frames,views,groups,matrices,valid)))
        else:play(cap,start,frames,views,groups,matrices,valid)
    finally:cap.release()


if __name__=='__main__':main()
