"""Use a pretrained human pose model to verify jersey-attached ball false positives.

Every observed crop is appearance-scored upstream. This cascade runs the expensive
pose model on appearance score >= .5 (plus evaluation samples), then requires a
confident torso, logo appearance and stable body-relative position across time.
It never infers a ball coordinate, changes homography or fills missing points.
"""
from pathlib import Path
import argparse
import hashlib
import json
import sys
import time

import cv2
import numpy as np
import pandas as pd

ROOT=Path(__file__).resolve().parents[1]
BASE=ROOT/'data/derived/ball_fp_reduction_20261116'
OUT=ROOT/'data/derived/ball_uniform_context_20261116'
VIDEO=ROOT/'data/broadcast/BAR-ATH_broadcast-001.mp4'
MODEL_SHA='93fc5e1d6b7690f33b4e1d60d6e9aec1cea14bdbc361bfae11778969be662078'
JOINTS=[5,6,11,12]


def arrays(path):
    with np.load(path) as z:return {k:z[k] for k in z.files}


def model_session():
    sys.path.insert(0,str(OUT/'runtime'))
    import onnxruntime as ort
    path=OUT/'models/yolo26n-pose.onnx'
    assert hashlib.sha256(path.read_bytes()).hexdigest()==MODEL_SHA
    options=ort.SessionOptions();options.intra_op_num_threads=4;options.inter_op_num_threads=1
    options.enable_cpu_mem_arena=True
    session=ort.InferenceSession(str(path),options,providers=['CPUExecutionProvider'])
    session.disable_fallback()
    return session


def infer_image(session,image,point,bbox):
    """Original image context around candidate; output remains in original pixels."""
    h,w=image.shape[:2]
    side=float(np.clip(np.max(bbox[2:]-bbox[:2])*12,480,1920))
    x1=max(0,int(point[0]-side/2));y1=max(0,int(point[1]-side*.45))
    x2=min(w,int(point[0]+side/2));y2=min(h,int(point[1]+side*.55))
    crop=image[y1:y2,x1:x2];scale=640/max(crop.shape[:2])
    nh=round(crop.shape[0]*scale);nw=round(crop.shape[1]*scale)
    canvas=np.full((640,640,3),114,np.uint8)
    canvas[:nh,:nw]=cv2.resize(crop,(nw,nh),interpolation=cv2.INTER_LINEAR)
    tensor=np.ascontiguousarray(canvas[:,:,::-1].transpose(2,0,1)[None],dtype=np.float32)/255
    result=session.run(None,{'images':tensor})[0][0]
    # End-to-end YOLO26: xyxy, person confidence, class ID, 17 x (x,y,conf).
    result=result[(result[:,4]>=.25)&(result[:,5]==0)].copy()
    result[:,:4]=result[:,:4]/scale+[x1,y1,x1,y1]
    k=result[:,6:].reshape(-1,17,3)
    k[:,:,:2]=k[:,:,:2]/scale+[x1,y1]
    return result,np.array([x1,y1,x2,y2],np.int32)


def associate(people,point,bbox):
    """Find the candidate's torso, never treat a whole person box as non-ball."""
    best=None;rank=-np.inf
    for i,row in enumerate(people):
        kp=row[6:].reshape(17,3)
        quality=float(kp[JOINTS,2].min())
        if row[4]<.4:continue
        shoulder=kp[[5,6],:2].mean(0);hip=kp[[11,12],:2].mean(0)
        across=kp[6,:2]-kp[5,:2];down=hip-shoulder
        mode=1
        if quality<.35:
            # Close-ups often crop off the pelvis. Do not trust invisible hips:
            # use confident shoulders and at least one visible downward upper arm
            # to define a conservative upper-shirt region instead.
            if row[4]<.5 or np.min(kp[[5,6],2])<.6:continue
            width=float(np.linalg.norm(across))
            if width<18:continue
            normal=np.array([across[1],-across[0]])/width
            if normal[1]<0:normal=-normal
            arms=kp[[7,8],:2]-kp[[5,6],:2]
            visible=(kp[[7,8],2]>=.5)&((arms@normal)>.10*width)
            if not visible.any():continue
            quality=float(min(kp[[5,6],2].min(),kp[[7,8],2][visible].max()))
            down=normal*(1.4*width);mode=2
        width=float(np.linalg.norm(across));height=float(np.linalg.norm(down))
        matrix=np.column_stack([across,down]);det=float(np.linalg.det(matrix))
        if min(width,height)<12 or abs(det)<.15*width*height:continue
        uv=np.linalg.solve(matrix,point-shoulder)
        # Shirt-front and back only; no hands, face, feet, or expanded person mask.
        inside=abs(uv[0])<=.58 and -.02<=uv[1]<=(.80 if mode==2 else .92)
        if not inside:continue
        wrist=kp[[9,10]];valid=wrist[:,2]>=.35
        wrist_distance=float(np.min(np.linalg.norm(wrist[valid,:2]-point,axis=1))/height) if valid.any() else np.inf
        size_fraction=float(np.max(bbox[2:]-bbox[:2])/max(width,height))
        score=quality+row[4]-.15*abs(float(uv[0]))-(.1 if mode==2 else 0)
        if score>rank:
            polygon=np.array([shoulder-across/2,shoulder+across/2,
                shoulder+down+across/2,shoulder+down-across/2])
            best=dict(person_index=i,torso_uv=uv,torso_scale=np.array([width,height]),
                      torso_quality=quality,person_confidence=float(row[4]),
                      wrist_distance=wrist_distance,bbox_torso_ratio=size_fraction,
                      torso_mode=mode,torso_polygon=polygon)
            rank=score
    return best


def evaluation_frames():
    frames=[]
    ref=arrays(BASE/'reference_evidence.npz');frames.extend(ref['source_frame'].tolist())
    for path in [BASE/'independent_review/holdout_notes.json',OUT/'evaluation_reference.json']:
        if not path.exists():continue
        data=json.loads(path.read_text())
        rows=data.get('labels',data.get('records',[])) if isinstance(data,dict) else data
        frames.extend([int(r['source_frame']) for r in rows])
    return np.unique(frames).astype(int)


def infer():
    OUT.mkdir(parents=True,exist_ok=True);cache=OUT/'pose_batches';cache.mkdir(exist_ok=True)
    d=arrays(BASE/'fp_inputs.npz');score=pd.read_csv(BASE/'appearance_scores.csv')
    n=len(d['source_frame']);queue=np.zeros(n,bool)
    queue[score.loc[score.nonball_score>=.5,'source_frame'].to_numpy(int)]=True
    queue[evaluation_frames()]=True
    queue&=(d['status_codes']==3)&np.isfinite(d['bbox']).all(1)&np.isfinite(d['pixel_xy']).all(1)
    already=np.zeros(n,bool)
    for path in cache.glob('batch_*.npz'):
        with np.load(path) as z:already[z['frames']]=True
    pending=np.flatnonzero(queue&~already)
    (OUT/'inference_queue.json').write_text(json.dumps(dict(observations=106689,appearance_gate=.5,
        eligible_observations=int(queue.sum()),reused=int((queue&already).sum()),pending=len(pending),
        note='Every observation appearance-scored; low-score candidates retained without pose rejection'),indent=2))
    if not len(pending):print('All queued poses cached',flush=True);return
    session=model_session();cap=cv2.VideoCapture(str(VIDEO));assert cap.isOpened()
    cap.set(cv2.CAP_PROP_POS_FRAMES,int(pending[0]));current=int(pending[0])-1
    records=[];people=[];offsets=[0];crops=[];start=time.monotonic();processed=0

    def flush():
        if not records:return
        fs=np.array(records,np.int32)
        target=cache/f'batch_{int(fs[0]):06d}_{int(fs[-1]):06d}.npz'
        np.savez_compressed(target,frames=fs,people=np.concatenate(people),
                            offsets=np.array(offsets,np.int32),crop_rect=np.array(crops))
        records.clear();people.clear();crops.clear();offsets[:]=[0]

    try:
        for frame in pending:
            # Long gaps are seeked; short gaps decoded without RGB retrieval.
            if frame-current>75:
                cap.set(cv2.CAP_PROP_POS_FRAMES,int(frame));current=int(frame)-1
            while current<frame:
                assert cap.grab(),int(frame);current+=1
            ok,image=cap.retrieve();assert ok
            pred,crop=infer_image(session,image,d['pixel_xy'][frame],d['bbox'][frame])
            records.append(frame);people.append(pred);offsets.append(offsets[-1]+len(pred));crops.append(crop)
            processed+=1
            if processed%250==0:
                flush();elapsed=time.monotonic()-start
                progress=dict(processed=processed,remaining=len(pending)-processed,source_frame=int(frame),
                    elapsed_seconds=round(elapsed,1),remaining_seconds=round((len(pending)-processed)*elapsed/processed,1))
                (OUT/'inference_progress.json').write_text(json.dumps(progress,indent=2))
                print(json.dumps(progress),flush=True)
        flush()
    finally:cap.release()


def score_context():
    d=arrays(BASE/'fp_inputs.npz');n=len(d['source_frame'])
    scored=pd.read_csv(BASE/'appearance_scores.csv').set_index('source_frame')
    appearance=np.full(n,np.nan);similarity=np.full(n,np.nan)
    appearance[scored.index]=scored.nonball_score;similarity[scored.index]=scored.closest_nonball_hog_similarity
    uv=np.full((n,2),np.nan);scale=np.full((n,2),np.nan);quality=np.full(n,np.nan)
    confidence=np.full(n,np.nan);wrist=np.full(n,np.nan);ratio=np.full(n,np.nan)
    torso_kp=np.full((n,17,3),np.nan,np.float32);inferred=np.zeros(n,bool)
    torso_mode=np.zeros(n,np.uint8);polygon=np.full((n,4,2),np.nan,np.float32)
    for path in sorted((OUT/'pose_batches').glob('batch_*.npz')):
        z=arrays(path)
        for i,frame in enumerate(z['frames']):
            assert not inferred[frame],f'duplicate {frame}'
            inferred[frame]=True;persons=z['people'][z['offsets'][i]:z['offsets'][i+1]]
            a=associate(persons,d['pixel_xy'][frame],d['bbox'][frame])
            if a is None:continue
            uv[frame]=a['torso_uv'];scale[frame]=a['torso_scale'];quality[frame]=a['torso_quality']
            confidence[frame]=a['person_confidence'];wrist[frame]=a['wrist_distance'];ratio[frame]=a['bbox_torso_ratio']
            torso_kp[frame]=persons[a['person_index'],6:].reshape(17,3)
            torso_mode[frame]=a['torso_mode'];polygon[frame]=a['torso_polygon']
    # A single high appearance score or torso overlap cannot trigger a rejection.
    # Real balls near hands remain unresolved; no geometric ball projection is used.
    appearance_ok=(appearance>=.85)&(similarity>=.67)
    body_ok=np.isfinite(uv).all(1)&(ratio>=.05)&(ratio<=.9)&(wrist>=.35)
    seeds=appearance_ok&body_ok
    support=np.zeros(n,np.int16);dispersion=np.full(n,np.nan);span=np.zeros(n,np.int16)
    for f in np.flatnonzero(seeds):
        neighbors=np.arange(max(0,f-12),min(n,f+13))
        good=seeds[neighbors]& (d['half'][neighbors]==d['half'][f])
        near=neighbors[good]
        if not len(near):continue
        # A coherent torso scale and original-image position prevent shot/person jumps.
        within=(np.linalg.norm(uv[near]-uv[f],axis=1)<=.12)
        within&=(np.abs(np.log(scale[near,1]/scale[f,1]))<.22)
        within&=(np.linalg.norm(d['pixel_xy'][near]-d['pixel_xy'][f],axis=1)<scale[f,1]*.6)
        near=near[within];support[f]=len(near)
        if len(near):
            span[f]=int(near[-1]-near[0]);dispersion[f]=float(np.max(np.linalg.norm(uv[near]-np.median(uv[near],axis=0),axis=1)))
    rejected=seeds&(support>=5)&(span>=5)&(dispersion<=.10)
    reason=np.full(n,'source observation unchanged',object)
    reason[inferred]='pose evaluated; insufficient joint evidence for rejection'
    reason[seeds]='non-ball-like torso candidate; insufficient temporal evidence'
    reason[rejected]='non-ball appearance + confident body context + stable body-relative attachment'
    result=pd.DataFrame(dict(source_frame=d['source_frame'],pose_inferred=inferred,
        nonball_score=appearance,closest_nonball_hog_similarity=similarity,
        torso_u=uv[:,0],torso_v=uv[:,1],torso_width=scale[:,0],torso_height=scale[:,1],
        torso_joint_confidence=quality,person_confidence=confidence,wrist_distance_torso_units=wrist,
        bbox_torso_ratio=ratio,appearance_gate=appearance_ok,body_gate=body_ok,
        temporal_support=support,temporal_span_frames=span,temporal_uv_dispersion=dispersion,
        torso_mode=torso_mode,model_rejected=rejected,reason=reason))
    result.to_csv(OUT/'context_decisions.csv',index=False)
    np.savez_compressed(OUT/'context_evidence.npz',source_frame=d['source_frame'],pose_inferred=inferred,
        torso_uv=uv,torso_scale=scale,torso_keypoints=torso_kp,model_rejected=rejected,
        torso_mode=torso_mode,torso_polygon=polygon,support=support,span=span,uv_dispersion=dispersion)
    summary=dict(pose_frames=int(inferred.sum()),confident_torso_candidates=int(np.isfinite(uv).all(1).sum()),
        appearance_and_body_seeds=int(seeds.sum()),automatic_uniform_rejections=int(rejected.sum()),
        source_coordinates_changed=False,manual_frame_whitelist_used=False,
        atd_or_homography_used=False,body_model='official YOLO26n-pose COCO pretrained',
        appearance_model='existing local L2 logistic crop ranker; scores not calibrated',
        torso_modes={'1':'visible shoulders and hips','2':'truncated close-up: confident shoulders and visible downward upper arm; proxy upper shirt'},
        rule=dict(appearance=.85,nonball_hog_similarity=.67,min_joint_confidence=.35,min_person_confidence=.4,
            torso_u=[-.58,.58],torso_v=[-.02,.92],min_wrist_distance=.35,min_support=5,
            window_frames=25,min_span=5,max_uv_distance=.12,max_uv_dispersion=.10),
        limitation='Conservative cascade; low-score or ambiguous body/hand cases remain, not a full-match accuracy benchmark')
    (OUT/'model_summary.json').write_text(json.dumps(summary,indent=2)+'\n')
    print(json.dumps(summary,indent=2),flush=True)


def main():
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('stage',choices=['infer','score'])
    a=p.parse_args();{'infer':infer,'score':score_context}[a.stage]()


if __name__=='__main__':main()
