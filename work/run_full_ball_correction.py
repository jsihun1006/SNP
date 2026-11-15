"""Resumable first-half high-recall ball correction, with explicit uncertainty.

Original observations and assets are read only. No estimated coordinate is ever
written as an observed ball, and automatic selection is not ground truth.
"""
import argparse,copy,csv,gzip,hashlib,json,math,os,sys,time,traceback
from collections import Counter
from pathlib import Path
import cv2
import numpy as np

ROOT=Path(__file__).resolve().parent
PROJECT=ROOT/'source/SNP-ball-tracking'
sys.path.insert(0,str(PROJECT))
from ball_only.run import atomic,output_lock,sha,read,environment
from ball_only.runtime_control import RuntimeController
from full_ball_tracking import solve
from full_ball_recovery import recover_chunk

VERSION='first_half_high_recall_observed_only_v1'
ACCEPTED={'selected_observation','visually_reviewed_observation'}
environment()

def hash_file(path):
    return sha(path)

def candidate_features(candidate,image,h,prior):
    p=np.asarray(candidate['point'],dtype=float)
    candidate['atd_distance_px']=float(np.linalg.norm(p-prior)) if np.isfinite(prior).all() else None
    if np.isfinite(h).all():
        try:
            q=np.linalg.solve(h,np.r_[p,1.])
            if np.isfinite(q).all() and abs(q[2])>1e-12:
                xy=q[:2]/q[2];candidate['pitch']=xy.tolist()
                candidate['penalty_spot_distance_m']=float(min(np.linalg.norm((xy-[11/105.3,.5])*[105.3,68]),np.linalg.norm((xy-[1-11/105.3,.5])*[105.3,68])))
        except np.linalg.LinAlgError:pass
    x,y=map(int,np.round(p));r=14
    patch=image[max(0,y-r):min(image.shape[0],y+r+1),max(0,x-r):min(image.shape[1],x+r+1)]
    if patch.size:
        hsv=cv2.cvtColor(patch,cv2.COLOR_BGR2HSV)
        candidate['green_fraction']=float(np.mean((hsv[:,:,0]>=30)&(hsv[:,:,0]<=95)&(hsv[:,:,1]>45)&(hsv[:,:,2]>35)))
    else:candidate['green_fraction']=0.

def camera_context(frames,first,h,errors,prior,cap):
    n=len(frames);warps=np.tile(np.eye(2,3),(n,1,1));cuts=np.zeros(n,bool);motion_valid=np.zeros(n,bool)
    previous=previous_hist=None;last_cut=-10
    if not cap.set(cv2.CAP_PROP_POS_FRAMES,first):raise RuntimeError('Cannot seek source context')
    for i,record in enumerate(frames):
        ok,image=cap.read()
        if not ok:raise RuntimeError('Video decode failed at '+str(record['frame']))
        small=cv2.resize(image,(480,270));gray=cv2.cvtColor(small,cv2.COLOR_BGR2GRAY)
        hsv=cv2.cvtColor(small,cv2.COLOR_BGR2HSV)
        hist=cv2.calcHist([hsv],[0,1],None,[24,8],[0,180,0,256]);cv2.normalize(hist,hist)
        if previous is not None:
            unaligned=float(np.mean(cv2.absdiff(previous,gray)))
            hist_diff=cv2.compareHist(previous_hist,hist,cv2.HISTCMP_BHATTACHARYYA)
            ratio=0.;residual=unaligned
            points=cv2.goodFeaturesToTrack(previous,100,.02,12)
            if points is not None:
                nxt,status,_=cv2.calcOpticalFlowPyrLK(previous,gray,points,None,winSize=(21,21),maxLevel=2)
                good=status[:,0].astype(bool)
                if good.sum()>=10:
                    m,inliers=cv2.estimateAffinePartial2D(points[good],nxt[good],method=cv2.RANSAC,ransacReprojThreshold=2)
                    if m is not None and inliers is not None:
                        ratio=float(np.mean(inliers));aligned=cv2.warpAffine(previous,m,(480,270))
                        residual=float(np.mean(cv2.absdiff(aligned[20:-20,20:-20],gray[20:-20,20:-20])))
                        if inliers.sum()>=10 and ratio>=.35 and residual<28 and .85<np.linalg.det(m[:,:2])<1.18:
                            m[:,2]*=4;warps[i]=m;motion_valid[i]=True
            # Failed optical flow is not a camera cut. Require global image evidence.
            hard_cut=(hist_diff>.48 and unaligned>25) or (unaligned>55 and residual>45 and ratio<.18)
            if hard_cut and i-last_cut>4:cuts[i]=True;motion_valid[i]=False;last_cut=i
        previous=gray;previous_hist=hist
        for c in record['candidates']:candidate_features(c,image,h[i],prior[i])
    return dict(H=h,h_error=errors,atd_pixel=prior,warps=warps,cuts=cuts,motion_valid=motion_valid)

class SequentialFrames:
    def __init__(self,video,start):
        self.cap=cv2.VideoCapture(str(video));self.start=start;self.index=-1;self.last=None
        if not self.cap.set(cv2.CAP_PROP_POS_FRAMES,start):raise RuntimeError('Cannot seek recovery decoder')
    def __call__(self,i):
        if i==self.index:return self.last
        if i<self.index:
            self.cap.set(cv2.CAP_PROP_POS_FRAMES,self.start+i);self.index=i-1
        while self.index+1<i:
            if not self.cap.grab():raise RuntimeError('Recovery video grab failed')
            self.index+=1
        ok,image=self.cap.read()
        if not ok:raise RuntimeError('Recovery video decode failed')
        self.index=i;self.last=image;return image
    def close(self):self.cap.release();self.last=None

def load_metadata(hfile,atdfile,total):
    h=np.full((total,3,3),np.nan);err=np.full(total,np.nan);ball=np.full((total,2),np.nan)
    with hfile.open(encoding='utf-8-sig',newline='') as f:
        for expected,row in enumerate(csv.DictReader(f)):
            frame=int(row['frame']);assert frame==expected
            if frame>=total:continue
            if int(float(row['valid']))==1:
                matrix=np.array([float(row[f'h{a}{b}']) for a in range(3) for b in range(3)]).reshape(3,3)
                if np.isfinite(matrix).all() and abs(np.linalg.det(matrix))>1e-12:h[frame]=matrix;err[frame]=float(row['reprojection_error'])
    with atdfile.open(encoding='utf-8') as f:
        for frame,line in enumerate(f):
            if frame>=total:break
            parts=line.strip().split(':');assert int(parts[0])==frame+1
            ball[frame]=np.fromstring(parts[2],sep=',')[:2]
    prior=np.full((total,2),np.nan)
    usable=np.isfinite(h).all(axis=(1,2))&np.isfinite(ball).all(axis=1)
    q=np.einsum('nij,nj->ni',h[usable],np.column_stack([ball[usable],np.ones(int(usable.sum()))]))
    indices=np.flatnonzero(usable);good=np.isfinite(q).all(axis=1)&(np.abs(q[:,2])>1e-12)
    prior[indices[good]]=q[good,:2]/q[good,2:]
    return h,err,prior

def load_original(path,start,end):
    result=[]
    with path.open(encoding='utf-8') as f:
        for line in f:
            row=json.loads(line);frame=row['source_frame']
            if frame>=end:break
            if frame<start:continue
            assert frame==start+len(result)
            candidates=[]
            for i,c in enumerate(row['candidates']):
                candidates.append(dict(point=c['point'],bbox_xyxy=c['bbox_xyxy'],confidence=c['confidence'],source=c['source'],origin='original_detector',original_candidate_index=i))
            result.append(dict(frame=frame,candidates=candidates,original_candidate_count=len(candidates)))
    assert len(result)==end-start
    return result

def consolidate(folder,config):
    csv_path=folder/'corrected_ball_coordinates.csv';decisions=folder/'frame_decisions.jsonl.gz'
    csv_tmp=csv_path.with_suffix('.csv.tmp');json_tmp=decisions.with_suffix('.gz.tmp')
    fields=['source_frame','part_frame','video_seconds','status','pixel_x','pixel_y','pitch_x_normalized','pitch_y_normalized','pitch_x_m','pitch_y_m','confidence','origin','candidate_index','needs_review','provisional_pixel_x','provisional_pixel_y','original_candidate_count','recovered_candidate_count']
    total=0;counts=Counter();new_count=0;original_count=0
    with csv_tmp.open('w',encoding='utf-8-sig',newline='') as f,gzip.open(json_tmp,'wt',encoding='utf-8',compresslevel=5) as dst:
        w=csv.DictWriter(f,fieldnames=fields);w.writeheader()
        for first in range(config['start'],config['end'],config['chunk_frames']):
            chunk=folder/'chunks'/str(first);manifest=read(chunk/'manifest.json');payload=chunk/'decisions.jsonl.gz'
            if hash_file(payload)!=manifest['sha256']:raise RuntimeError('Checkpoint checksum mismatch')
            with gzip.open(payload,'rt',encoding='utf-8') as src:
                for line in src:
                    row=json.loads(line);assert row['source_frame']==config['start']+total
                    state=row['selection'];item=dict(source_frame=row['source_frame'],part_frame=total,video_seconds=row['source_frame']/25,status=state['status'],pixel_x=state.get('pixel_x'),pixel_y=state.get('pixel_y'),confidence=state.get('candidate_confidence'),origin=state.get('origin'),candidate_index=state['candidate_index'],needs_review=state.get('needs_review',True),provisional_pixel_x=state.get('provisional_pixel_x'),provisional_pixel_y=state.get('provisional_pixel_y'),original_candidate_count=row['original_candidate_count'],recovered_candidate_count=sum(c.get('origin')!='original_detector' for c in row['candidates']))
                    if state['status'] in ACCEPTED:
                        c=row['candidates'][state['candidate_index']]
                        assert np.allclose([item['pixel_x'],item['pixel_y']],c['point'],atol=1e-9,rtol=0)
                        if c.get('pitch') is not None:
                            x,y=c['pitch'];item.update(pitch_x_normalized=x,pitch_y_normalized=y,pitch_x_m=x*105.3,pitch_y_m=y*68)
                    else:assert item['pixel_x'] is None and item['pixel_y'] is None
                    w.writerow(item);dst.write(line)
                    counts[state['status']]+=1;new_count+=item['recovered_candidate_count'];original_count+=row['original_candidate_count'];total+=1
        assert total==config['end']-config['start']
    os.replace(csv_tmp,csv_path);os.replace(json_tmp,decisions)
    return dict(frames=total,statuses=dict(counts),original_candidate_observations=original_count,new_candidate_observations=new_count,csv_sha256=hash_file(csv_path),decisions_sha256=hash_file(decisions),observed_coordinates_verified=True,accuracy_validated=False)

def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--out',type=Path,default=Path(r'H:\data\ball-tracking-results\first_half\correction_recall_v1'))
    parser.add_argument('--start',type=int,default=7171);parser.add_argument('--end',type=int,default=86638)
    parser.add_argument('--chunk-frames',type=int,default=1000);parser.add_argument('--halo',type=int,default=50)
    parser.add_argument('--no-full-fallback',action='store_true')
    args=parser.parse_args()
    if not 7171<=args.start<args.end<=86638:raise ValueError('This run is restricted to the previously detected first half')
    folder=args.out.resolve();folder.mkdir(parents=True,exist_ok=True);(folder/'chunks').mkdir(exist_ok=True)
    video=Path(r'H:\data\broadcast\BAR-ATH_broadcast.mp4');source=Path(r'H:\data\ball-tracking-results\first_half\candidates.jsonl');hfile=Path(r'H:\data\broadcast\BAR-ATH_broadcast_homography.csv');atdfile=Path(r'H:\data\broadcast\BAR-ATH_broadcast_atd.txt');weights=PROJECT/'ball_only/weights/yolo26x.pt'
    started=time.perf_counter();finished=0;resumed_frames=0;states=Counter();phase='checking_inputs'
    def publish(extra=None):
        elapsed=time.perf_counter()-started;rate=(finished-resumed_frames)/elapsed if finished>resumed_frames else None
        data=dict(status='running',phase=phase,start_frame=args.start,end_frame_exclusive=args.end,completed_frames=finished,total_frames=args.end-args.start,elapsed_seconds=elapsed,processing_fps=rate,eta_seconds=(args.end-args.start-finished)/rate if rate else None,status_counts=dict(states),updated_unix=time.time())
        if extra:data.update(extra)
        atomic(folder/'progress.json',data);print(json.dumps(data),flush=True)
    with output_lock(folder):
        publish()
        config=dict(version=VERSION,start=args.start,end=args.end,chunk_frames=args.chunk_frames,halo=args.halo,
                    recovery=dict(confidence=.08,crop_size=512,imgsz=1280,batch_size=4,max_jobs_per_frame=2,full_frame_fallback=not args.no_full_fallback),
                    input_sha256={str(p):hash_file(p) for p in (video,source,hfile,atdfile,weights)},
                    code_sha256={p.name:hash_file(p) for p in (Path(__file__),ROOT/'full_ball_tracking.py',ROOT/'full_ball_recovery.py')},
                    no_atd_coordinate_copy=True,ground_truth_accuracy_validated=False)
        original_config=read(source.parent/'run_config.json');original_complete=read(source.parent/'job_complete.json')
        if config['input_sha256'][str(video)]!=original_config['video']['sha256']:raise RuntimeError('Source video changed since original detection')
        if config['input_sha256'][str(weights)]!=original_config['weights']['sha256']:raise RuntimeError('Model weights changed since original detection')
        if config['input_sha256'][str(source)]!=original_complete['file_sha256']:raise RuntimeError('Original candidate file checksum changed')
        fingerprint=hashlib.sha256(json.dumps(config,sort_keys=True).encode()).hexdigest();config['fingerprint']=fingerprint
        if (folder/'run_config.json').exists():
            if read(folder/'run_config.json')['fingerprint']!=fingerprint:raise ValueError('Inputs or settings changed; use a new output directory')
        else:atomic(folder/'run_config.json',config)
        phase='loading_context';publish()
        h,err,prior=load_metadata(hfile,atdfile,164487);original=load_original(source,args.start,args.end)
        for first in range(args.start,args.end,args.chunk_frames):
            manifest=folder/'chunks'/str(first)/'manifest.json'
            if manifest.exists():
                m=read(manifest)
                if m['fingerprint']!=fingerprint or hash_file(manifest.parent/'decisions.jsonl.gz')!=m['sha256']:raise RuntimeError('Invalid checkpoint')
                finished+=m['frames'];states.update(m['status_counts'])
        resumed_frames=finished
        if finished==args.end-args.start:
            phase='consolidating';publish();result=consolidate(folder,config);atomic(folder/'job_complete.json',dict(result,finished_unix=time.time()));publish(dict(status='complete',eta_seconds=0));return
        import torch
        from ultralytics import YOLO,settings
        settings.update({'sync':False});model=YOLO(str(weights));controller=RuntimeController(target_low=98,target_high=99.5)
        cap=cv2.VideoCapture(str(video))
        try:
            for first in range(args.start,args.end,args.chunk_frames):
                last=min(args.end,first+args.chunk_frames);chunk=folder/'chunks'/str(first)
                if (chunk/'manifest.json').exists():continue
                if (folder/'STOP').exists():publish(dict(status='stopped',next_source_frame=first));return
                a=max(args.start,first-args.halo);b=min(args.end,last+args.halo)
                records=copy.deepcopy(original[a-args.start:b-args.start])
                phase='camera_and_candidate_context';publish(dict(current_core=[first,last]))
                context=camera_context(records,a,h[a:b],err[a:b],prior[a:b],cap)
                phase='initial_association';initial=solve(records,context)
                phase='expanded_redetection';publish(dict(current_core=[first,last]))
                decoder=SequentialFrames(video,a)
                def progress_cb(event):
                    atomic(folder/'recovery_progress.json',dict(source_context_start=a,core_start=first,updated_unix=time.time(),**event))
                try:
                    stats=recover_chunk(records,context,decoder,initial,model,controller,batch_size=4,crop_size=512,imgsz=1280,confidence=.08,max_jobs_per_frame=2,temporal_gap_frames=12,full_frame_fallback=not args.no_full_fallback,progress_callback=progress_cb)
                finally:decoder.close()
                phase='final_association';final=solve(records,context)
                chunk.mkdir(exist_ok=True);tmp=chunk/'decisions.jsonl.gz.tmp';counts=Counter()
                with gzip.open(tmp,'wt',encoding='utf-8',compresslevel=5) as f:
                    for i in range(first-a,last-a):
                        record=records[i];state=final[i];state['part_frame']=record['frame']-args.start;state['local_frame']=state['part_frame']
                        counts[state['status']]+=1
                        f.write(json.dumps(dict(source_frame=record['frame'],original_candidate_count=record['original_candidate_count'],selection=state,candidates=record['candidates']),allow_nan=False,separators=(',',':'))+'\n')
                payload=chunk/'decisions.jsonl.gz';os.replace(tmp,payload)
                atomic(chunk/'manifest.json',dict(fingerprint=fingerprint,start_frame=first,end_frame_exclusive=last,frames=last-first,sha256=hash_file(payload),status_counts=dict(counts),recovery=stats,created_unix=time.time(),camera_cut_events=int(context['cuts'].sum())))
                finished+=last-first;states.update(counts);phase='chunk_complete';publish(dict(gpu=controller.snapshot(refresh=True)))
            phase='consolidating';publish();result=consolidate(folder,config)
            atomic(folder/'job_complete.json',dict(result,finished_unix=time.time(),elapsed_seconds=time.perf_counter()-started))
            publish(dict(status='complete',eta_seconds=0))
        finally:cap.release()

if __name__=='__main__':
    try:main()
    except BaseException as exc:
        detail=traceback.format_exc()
        try:
            folder=Path(sys.argv[sys.argv.index('--out')+1]) if '--out' in sys.argv else Path(r'H:\data\ball-tracking-results\first_half\correction_recall_v1')
            if folder.exists():atomic(folder/'job_error.json',dict(status='failed',error=repr(exc),traceback=detail,updated_unix=time.time()))
        except Exception:pass
        print(detail,file=sys.stderr,flush=True);raise
