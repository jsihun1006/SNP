"""Score candidates with frozen YOLO26s two-view embeddings and a small binary head.

Scores are uncalibrated evidence, never an automatic ball/nonball decision.
Official model weights, input candidates and tracking coordinates stay unchanged.
"""
from pathlib import Path
import argparse
import csv
import hashlib
import json
import os
import sys
import time
import uuid

import numpy as np

ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT))
from starter.classify_ball_yolo26_candidates import square_crop
from starter.benchmark_ball_yolo26 import sha

BASE=ROOT/'data/derived/ball_yolo26_model_audit_20261116'
SCALES=[1.25,2.0]


def score_features(head,features):
    z=np.clip((features-head['mean'])/head['std'],-6,6)
    first=z[:,:1280]@head['coef'][:1280]
    second=z[:,1280:]@head['coef'][1280:-1]
    logit=first+second+head['coef'][-1]
    score=1/(1+np.exp(-np.clip(logit,-40,40)))
    return score,logit,first,second


def write_json(path,value):
    temp=path.with_name(path.name+'.tmp.'+uuid.uuid4().hex)
    temp.write_text(json.dumps(value,indent=2)+'\n');temp.replace(path)


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--source-candidates',type=Path,default=ROOT/'data/derived/ball_targeted_redetection_20261116/yolo26/raw_ball_candidates.csv')
    parser.add_argument('--output',type=Path,default=BASE/'binary_candidates')
    parser.add_argument('--model',type=Path,default=BASE/'models/yolo26s-cls.pt')
    parser.add_argument('--head',type=Path,default=BASE/'binary_probe/binary_head.npz')
    parser.add_argument('--video',type=Path,default=ROOT/'data/broadcast/BAR-ATH_broadcast-001.mp4')
    parser.add_argument('--device',choices=['cpu','mps'],default='cpu')
    parser.add_argument('--batch-size',type=int,default=32,help='Number of 224-pixel patches; two per candidate')
    parser.add_argument('--threads',type=int,default=4)
    parser.add_argument('--limit',type=int)
    parser.add_argument('--save-features',action='store_true',help='Cache exact object embeddings for later verifier updates')
    parser.add_argument('--parity',action='store_true',help='Only compare four original CPU reference embeddings/scores with the frozen probe')
    args=parser.parse_args()
    if args.batch_size<2 or args.batch_size%2 or args.threads<1 or (args.limit is not None and args.limit<1):parser.error('Positive even patch batch-size, positive threads and limit required')
    output=args.output.resolve();output.mkdir(parents=True,exist_ok=True)
    os.environ['YOLO_CONFIG_DIR']=str(output/'runtime/ultralytics');os.environ['MPLCONFIGDIR']=str(output/'runtime/matplotlib');os.environ['YOLO_AUTOINSTALL']='false'
    import cv2
    import pandas as pd
    import torch
    import ultralytics
    from ultralytics import YOLO
    torch.set_num_threads(args.threads)
    if args.device=='mps' and not torch.backends.mps.is_available():raise RuntimeError('MPS unavailable; use GPU-enabled execution or --device cpu')
    with np.load(args.head) as archive:head={k:archive[k] for k in ['mean','std','coef']}
    assert head['mean'].shape==head['std'].shape==(2560,) and head['coef'].shape==(2561,) and (head['std']>0).all()
    model=YOLO(str(args.model));assert model.task=='classify' and model.names[805].replace('_',' ').lower()=='soccer ball'
    net=model.model.to(device=args.device,dtype=torch.float32).eval();net.requires_grad_(False)
    linear=net.model[-1].linear;assert linear.in_features==1280
    captured=[]
    hook=linear.register_forward_pre_hook(lambda module,items:captured.append(items[0].detach().cpu().numpy().copy()))
    def embed(patches):
        captured.clear()
        rgb=np.stack(patches)[:,:,:,::-1].transpose(0,3,1,2)
        tensor=torch.from_numpy(np.ascontiguousarray(rgb)).to(device=args.device,dtype=torch.float32)/255.
        with torch.inference_mode():net(tensor)
        if args.device=='mps':torch.mps.synchronize()
        assert len(captured)==1 and captured[0].shape==(len(patches),1280)
        return captured[0].reshape(len(patches)//2,2560)
    if args.parity:
        reference=json.loads((BASE/'reference.json').read_text())['records']
        with np.load(BASE/'binary_probe/frozen_features.npz') as archive:
            frozen=archive['features'];signature=json.loads(str(archive['signature']))
        assert signature['model_sha256']==sha(args.model)
        indices=[0,10,45,len(reference)-1];patches=[]
        cap=cv2.VideoCapture(str(args.video));assert cap.isOpened()
        for i in indices:
            r=reference[i];assert cap.set(cv2.CAP_PROP_POS_FRAMES,int(r['source_frame']));ok,image=cap.read();assert ok
            patches.extend(square_crop(image,r['bbox_xyxy'],s) for s in SCALES)
        cap.release();features=embed(patches);expected=frozen[indices]
        before=score_features(head,expected)[0];after=score_features(head,features)[0]
        feature_error=float(np.max(np.abs(features-expected)));score_error=float(np.max(np.abs(before-after)))
        assert feature_error<2e-4 and score_error<2e-5,(feature_error,score_error)
        report=dict(passed=True,device=args.device,reference_indices=indices,source_frames=[reference[i]['source_frame'] for i in indices],
            maximum_embedding_absolute_error=feature_error,maximum_score_absolute_error=score_error,
            expected_scores=before.tolist(),actual_scores=after.tolist(),model_sha256=sha(args.model),head_sha256=sha(args.head))
        write_json(output/'parity_verification.json',report);print(json.dumps(report,indent=2));hook.remove();return
    table=pd.read_csv(args.source_candidates,float_precision='round_trip').sort_values(['source_frame','candidate_index']).reset_index(drop=True)
    assert not table.duplicated(['source_frame','candidate_index']).any()
    all_keys=[(int(r.source_frame),int(r.candidate_index)) for r in table.itertuples()]
    rows=table.iloc[:args.limit] if args.limit is not None else table
    keys=[(int(r.source_frame),int(r.candidate_index)) for r in rows.itertuples()]
    stat=args.video.stat()
    config=dict(model=str(args.model.resolve()),model_sha256=sha(args.model),head=str(args.head.resolve()),head_sha256=sha(args.head),
        source_candidates_sha256=sha(args.source_candidates),video=str(args.video.resolve()),video_bytes=stat.st_size,video_mtime_ns=stat.st_mtime_ns,
        device=args.device,precision='float32',head_training_device='cpu',threads=args.threads,
        batch_size=args.batch_size,scales=SCALES,imgsz=224,padding=114,
        embedding='1280 inputs before the frozen classification linear layer, concatenated by candidate/view',
        normalization='BGR to RGB, CHW, float32 / 255; then head mean/std, clip[-6,6], L2 logistic coefficients')
    if args.save_features:
        config['save_features']=True
        (output/'features').mkdir(exist_ok=True)
    signature=hashlib.sha256(json.dumps(config,sort_keys=True).encode()).hexdigest()
    cache=output/'batches';cache.mkdir(exist_ok=True);prior={}
    for path in sorted(cache.glob('batch_*.jsonl')):
        with path.open() as stream:
            assert json.loads(next(stream))['request_signature']==signature
            for line in stream:
                r=json.loads(line);key=(r['source_frame'],r['candidate_index']);assert key not in prior;prior[key]=r
    assert set(prior)<=set(all_keys)
    pending=rows.loc[[key not in prior for key in keys]]
    metadata=dict(status='running',request_signature=signature,request=config,total_source_candidates=len(table),requested_candidates=len(rows),
        limit=args.limit,cached_before_run=len(prior),original_tracking_changed=False,official_weights_changed=False,
        scores_are_calibrated_probabilities=False,automatic_binary_decisions=False,
        runtime=dict(torch=torch.__version__,ultralytics=ultralytics.__version__,opencv=cv2.__version__))
    write_json(output/'run_metadata.json',metadata)
    buffer=[];patches=[];items=[];processed=0;start=time.monotonic();last_report=start;times=[]
    def flush():
        if not buffer:return
        a,b=buffer[0],buffer[-1]
        target=cache/f"batch_{a['source_frame']:06d}_{a['candidate_index']:03d}_{b['source_frame']:06d}_{b['candidate_index']:03d}.jsonl"
        temp=target.with_name(target.name+'.tmp.'+uuid.uuid4().hex)
        with temp.open('w') as stream:
            stream.write(json.dumps(dict(request_signature=signature))+'\n')
            for r in buffer:stream.write(json.dumps(r,separators=(',',':'))+'\n')
        temp.replace(target);prior.update({(r['source_frame'],r['candidate_index']):r for r in buffer});buffer.clear()
    def batch():
        nonlocal processed,last_report
        if not items:return
        started=time.perf_counter();features=embed(patches);scores,logits,first,second=score_features(head,features);times.append((time.perf_counter()-started)*1000)
        if args.save_features:
            name=f"batch_{items[0]['source_frame']:06d}_{items[0]['candidate_index']}_{items[-1]['source_frame']:06d}_{items[-1]['candidate_index']}.npz"
            np.savez_compressed(output/'features'/name,features=features,
                source_frame=np.array([r['source_frame'] for r in items],np.int32),candidate_index=np.array([r['candidate_index'] for r in items],np.int32))
        for i,item in enumerate(items):
            buffer.append(dict(source_frame=int(item['source_frame']),candidate_index=int(item['candidate_index']),
                bbox_x1=float(item['bbox_x1']),bbox_y1=float(item['bbox_y1']),bbox_x2=float(item['bbox_x2']),bbox_y2=float(item['bbox_y2']),
                source_confidence=float(item['confidence']),binary_ball_score=float(scores[i]),scorelogit=float(logits[i]),
                view1_logit_contribution=float(first[i]),view2_logit_contribution=float(second[i])))
            processed+=1
        items.clear();patches.clear()
        if len(buffer)>=1000:flush()
        now=time.monotonic()
        if now-last_report>=20:
            elapsed=now-start
            progress=dict(processed_this_run=processed,cached_before_run=metadata['cached_before_run'],pending_candidates=len(pending),
                requested_candidates=len(rows),elapsed_seconds=round(elapsed,1),remaining_seconds=round((len(pending)-processed)*elapsed/max(processed,1),1))
            write_json(output/'progress.json',progress);print(json.dumps(progress),flush=True);last_report=now
    cap=cv2.VideoCapture(str(args.video));assert cap.isOpened();current=-1
    try:
        for frame,group in pending.groupby('source_frame',sort=True):
            frame=int(frame)
            if current<0 or frame-current>75:assert cap.set(cv2.CAP_PROP_POS_FRAMES,frame);current=frame-1
            while current<frame:assert cap.grab();current+=1
            ok,image=cap.retrieve();assert ok and image.shape==(1080,1920,3)
            for item in group.to_dict('records'):
                bbox=[item[k] for k in ['bbox_x1','bbox_y1','bbox_x2','bbox_y2']]
                items.append(item);patches.extend(square_crop(image,bbox,s) for s in SCALES)
                if len(patches)>=args.batch_size:batch()
        batch();flush()
    except BaseException as exc:
        flush();metadata.update(status='interrupted' if isinstance(exc,KeyboardInterrupt) else 'failed',error=str(exc),completed_candidates=len(prior))
        write_json(output/'run_metadata.json',metadata);raise
    finally:cap.release();hook.remove()
    assert all(key in prior for key in keys)
    with (output/'scores.csv').open('w',newline='') as stream:
        writer=csv.DictWriter(stream,list(prior[keys[0]]));writer.writeheader();writer.writerows(prior[key] for key in keys)
    metadata.update(status='complete_limited' if len(rows)<len(table) else 'complete',completed_candidates=len(rows),
        newly_processed=processed,elapsed_seconds=round(time.monotonic()-start,2),timed_batches=len(times),
        median_batch_ms=float(np.median(times)) if times else None,p95_batch_ms=float(np.percentile(times,95)) if times else None)
    write_json(output/'run_metadata.json',metadata)
    print(json.dumps({k:metadata[k] for k in ['status','completed_candidates','newly_processed','elapsed_seconds']},indent=2),flush=True)


if __name__=='__main__':main()
