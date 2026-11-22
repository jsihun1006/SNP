"""Benchmark/use YOLO26 ImageNet classifiers as candidate verification evidence.

Soccer-ball probability is a closed-set image-classification score. A low score
does not prove an object is nonball and never changes the original tracking.
"""
from pathlib import Path
import argparse
import csv
import gc
import hashlib
import json
import os
import sys
import time
import uuid

import numpy as np

ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT))
from starter.benchmark_ball_yolo26 import read_reference,measures,ranking_stats,sha

BASE=ROOT/'data/derived/ball_yolo26_model_audit_20261116'
SCALES=[1.25,2.0]
THRESHOLDS=sorted(set([factor*10.**power for power in range(-9,-1) for factor in [1.,3.]]+
    [.05,.10,.20,.30,.50,.70,.80,.90]))


def square_crop(image,bbox,scale):
    """Exact bbox-centred square with 114 padding, then resize to 224 pixels."""
    import cv2
    box=np.asarray(bbox,float)
    if box.shape!=(4,) or not np.isfinite(box).all() or (box[2:]<=box[:2]).any():raise ValueError('Invalid candidate box')
    side=int(np.ceil(max(12.,float((box[2:]-box[:2]).max())*scale)))
    center=(box[:2]+box[2:])/2;left,top=np.floor(center-side/2).astype(int)
    right,bottom=left+side,top+side;h,w=image.shape[:2]
    patch=np.full((side,side,3),114,np.uint8)
    x1,y1=max(0,left),max(0,top);x2,y2=min(w,right),min(h,bottom)
    if x2>x1 and y2>y1:patch[y1-top:y2-top,x1-left:x2-left]=image[y1:y2,x1:x2]
    return cv2.resize(patch,(224,224),interpolation=cv2.INTER_LINEAR)


def combine(first,second,strategy):
    if strategy=='min':return float(min(first,second))
    if strategy=='geometric_mean':return float(np.sqrt(first*second))
    raise ValueError(strategy)


def setup(output,device):
    output.mkdir(parents=True,exist_ok=True)
    os.environ['YOLO_CONFIG_DIR']=str(output/'runtime/ultralytics');os.environ['MPLCONFIGDIR']=str(output/'runtime/matplotlib');os.environ['YOLO_AUTOINSTALL']='false'
    import torch
    import cv2
    import ultralytics
    from ultralytics import YOLO
    torch.set_num_threads(4)
    if device=='mps' and not torch.backends.mps.is_available():raise RuntimeError('MPS unavailable; run with GPU access or choose --device cpu')
    return torch,cv2,ultralytics,YOLO


def load_classifier(path,YOLO):
    model=YOLO(str(path))
    assert model.task=='classify'
    name=model.names[805].lower().replace('_',' ').replace('-',' ')
    assert name=='soccer ball',(805,model.names[805])
    return model


def predict_patches(model,patches,device,batch_size,torch):
    started=time.perf_counter()
    results=model.predict(patches,imgsz=224,device=device,batch=batch_size,verbose=False,save=False)
    output=[]
    for result in results:
        probabilities=result.probs.data.detach().cpu().numpy()
        assert probabilities.shape==(1000,) and np.isfinite(probabilities).all()
        indices=np.argsort(probabilities)[-5:][::-1]
        output.append(dict(prob805=float(probabilities[805]),top1_class=str(model.names[int(indices[0])]),
            top1_score=float(probabilities[indices[0]]),
            top5=[dict(class_id=int(i),label=str(model.names[int(i)]),score=float(probabilities[i])) for i in indices]))
    if device=='mps':torch.mps.synchronize()
    assert len(output)==len(patches)
    return output,(time.perf_counter()-started)*1000


def joined(first,second,bbox,strategy):
    return dict(cls_prob805_view1=first['prob805'],cls_prob805_view2=second['prob805'],
        combined_soccer_score=combine(first['prob805'],second['prob805'],strategy),combined_strategy=strategy,
        top1_class=first['top1_class'],top1_score=first['top1_score'],
        top1_class_view1=first['top1_class'],top1_score_view1=first['top1_score'],
        top1_class_view2=second['top1_class'],top1_score_view2=second['top1_score'],
        top5_view1=json.dumps(first['top5'],separators=(',',':')),
        top5_view2=json.dumps(second['top5'],separators=(',',':')),
        object_native_maxside=float(max(bbox[2]-bbox[0],bbox[3]-bbox[1])))


def write_csv(path,rows):
    if not rows:raise ValueError('No rows to export')
    with path.open('w',newline='') as stream:
        writer=csv.DictWriter(stream,list(rows[0]));writer.writeheader();writer.writerows(rows)


def benchmark(args):
    output=args.output.resolve();torch,cv2,ultralytics,YOLO=setup(output,args.device)
    records=read_reference(args.reference)
    assert all(r['bbox_xyxy'] is not None for r in records),'Candidate classification requires reviewed boxes'
    cap=cv2.VideoCapture(str(args.video));assert cap.isOpened();images={}
    for f in sorted({r['source_frame'] for r in records}):
        assert cap.set(cv2.CAP_PROP_POS_FRAMES,f);ok,img=cap.read();assert ok
        images[f]=img
    cap.release()
    patches=[square_crop(images[r['source_frame']],r['bbox_xyxy'],s) for r in records for s in SCALES]
    rows=[];performance={};cache=output/'benchmark_cache';cache.mkdir(exist_ok=True)
    for path in args.model:
        path=path.resolve();name=path.stem
        signature=hashlib.sha256(json.dumps(dict(model_sha256=sha(path),reference_sha256=sha(args.reference),
            scales=SCALES,imgsz=224,device=args.device,batch_size=args.batch_size),sort_keys=True).encode()).hexdigest()
        target=cache/f'{name}.json'
        if target.exists():
            stored=json.loads(target.read_text());assert stored['signature']==signature
            values=stored['predictions'];timing=stored['performance']
        else:
            model=load_classifier(path,YOLO)
            warm=[]
            for _ in range(2):_,ms=predict_patches(model,patches[:min(args.batch_size,len(patches))],args.device,args.batch_size,torch);warm.append(ms)
            values=[];times=[]
            for first in range(0,len(patches),args.batch_size):
                predicted,ms=predict_patches(model,patches[first:first+args.batch_size],args.device,args.batch_size,torch)
                values.extend(predicted);times.append(ms)
            timing=dict(model_sha256=sha(path),model_path=str(path),device=args.device,
                warmup_batch_ms=warm,batch_size=args.batch_size,patches=len(patches),
                timed_batch_ms=times,median_batch_ms=float(np.median(times)),p95_batch_ms=float(np.percentile(times,95)),
                patches_per_second=len(patches)/sum(times)*1000,
                class805=model.names[805],probability_interpretation='closed-set ImageNet score, not calibrated ball probability')
            target.write_text(json.dumps(dict(signature=signature,predictions=values,performance=timing),separators=(',',':')))
            del model;gc.collect()
            if args.device=='mps':torch.mps.empty_cache()
        performance[name]=timing
        for i,r in enumerate(records):
            for strategy in ['geometric_mean','min']:
                rows.append(dict(model=name,reference_id=r['reference_id'],source_frame=r['source_frame'],
                    label=r['label'],kind=r['kind'],split=r['split'],
                    **joined(values[i*2],values[i*2+1],r['bbox_xyxy'],strategy)))
        print(name,'classification benchmark complete',flush=True)
    write_csv(output/'classifier_reference_predictions.csv',rows)
    comparisons=[];groups=[];candidates=[]
    for name in performance:
        for strategy in ['geometric_mean','min']:
            own=[r for r in rows if r['model']==name and r['combined_strategy']==strategy]
            splits=sorted({r['split'] for r in own})
            summaries={}
            for split in [*splits,'all']:
                group=[r for r in own if r['split']==split] if split!='all' else own
                metrics=[measures([r['label'] for r in group],[r['combined_soccer_score'] for r in group],t) for t in THRESHOLDS]
                comparisons.extend(dict(model=name,combined_strategy=strategy,split=split,**m) for m in metrics)
                summaries[split]=dict(samples=len(group),**ranking_stats([r['label'] for r in group],[r['combined_soccer_score'] for r in group]),thresholds=metrics)
            calibration=next((s for s in ['calibration','selection','development'] if s in summaries),'all')
            values=summaries[calibration]['thresholds'];eligible=[m for m in values if m['recall']>=.9]
            chosen=max(eligible or values,key=lambda m:(m['specificity'],m['recall'],m['threshold']) if eligible else (m['balanced_accuracy'],m['specificity'],m['recall']))
            summary=dict(model=name,combined_strategy=strategy,selected_threshold=chosen['threshold'],
                threshold_selected_from=calibration,calibration=chosen,calibration_ball_recall_target_met=bool(eligible),groups=summaries)
            for split in ['heldout','holdout','test']:
                if split in summaries:
                    group=[r for r in own if r['split']==split]
                    summary['heldout']=measures([r['label'] for r in group],[r['combined_soccer_score'] for r in group],chosen['threshold'])
            groups.append(summary);candidates.append(summary)
    write_csv(output/'classifier_threshold_comparison.csv',comparisons)
    ranked=sorted(candidates,key=lambda r:(not r['calibration_ball_recall_target_met'],
        -r['calibration']['balanced_accuracy'],-r['calibration']['specificity'],
        performance[r['model']]['median_batch_ms']))
    summary=dict(reference_sha256=sha(args.reference),reference_objects=len(records),scales=SCALES,imgsz=224,
        recommended_model_from_calibration=ranked[0]['model'],recommended_strategy=ranked[0]['combined_strategy'],
        recommended_threshold=ranked[0]['selected_threshold'],models=groups,performance=performance,
        scope='Selected exact-object challenge, not whole-match accuracy. Heldout labels never select threshold/strategy.',
        low_score_interpretation='No soccer-ball support, not a definitive nonball label',
        runtime=dict(torch=torch.__version__,ultralytics=ultralytics.__version__,opencv=cv2.__version__,device=args.device))
    (output/'classifier_benchmark_summary.json').write_text(json.dumps(summary,indent=2)+'\n')
    print(json.dumps({k:summary[k] for k in ['recommended_model_from_calibration','recommended_strategy','recommended_threshold']},indent=2),flush=True)


def infer(args):
    if len(args.model)!=1:raise ValueError('infer requires one explicitly selected classifier')
    import pandas as pd
    output=args.output.resolve();torch,cv2,ultralytics,YOLO=setup(output,args.device)
    path=args.model[0].resolve();model=load_classifier(path,YOLO)
    table=pd.read_csv(args.source_candidates,float_precision='round_trip')
    assert not table.duplicated(['source_frame','candidate_index']).any()
    table=table.sort_values(['source_frame','candidate_index']).reset_index(drop=True)
    config=dict(model=str(path),model_sha256=sha(path),source_candidates_sha256=sha(args.source_candidates),
        reference_sha256=sha(args.reference),scales=SCALES,imgsz=224,padding=114,device=args.device,
        batch_size=args.batch_size,combined_strategy=args.combined,source_video=str(args.video.resolve()),
        classifier_class_id=805,classifier_class_name=model.names[805],
        probability_interpretation='Closed-set ImageNet score, not calibrated binary ball probability')
    signature=hashlib.sha256(json.dumps(config,sort_keys=True).encode()).hexdigest()
    cache=output/'candidate_batches';cache.mkdir(exist_ok=True);prior={}
    for p in sorted(cache.glob('batch_*.jsonl')):
        with p.open() as stream:
            header=json.loads(next(stream));assert header['request_signature']==signature
            for line in stream:
                r=json.loads(line);key=(r['source_frame'],r['candidate_index']);assert key not in prior;prior[key]=r
    keys=[(int(r.source_frame),int(r.candidate_index)) for r in table.itertuples()]
    assert set(prior)<=set(keys)
    pending=table.loc[[key not in prior for key in keys]]
    metadata=dict(status='running',request_signature=signature,request=config,total_candidates=len(table),
        cached_before_run=len(prior),original_tracking_changed=False,low_score_is_not_nonball_proof=True,
        runtime=dict(torch=torch.__version__,ultralytics=ultralytics.__version__,opencv=cv2.__version__))
    (output/'classifier_run_metadata.json').write_text(json.dumps(metadata,indent=2)+'\n')
    cap=cv2.VideoCapture(str(args.video));assert cap.isOpened();current=-1;buffer=[];patches=[];items=[]
    start=time.monotonic();last_report=start;processed=0;batch_times=[]
    def flush():
        if not buffer:return
        a,b=buffer[0],buffer[-1]
        target=cache/f"batch_{a['source_frame']:06d}_{a['candidate_index']:03d}_{b['source_frame']:06d}_{b['candidate_index']:03d}.jsonl"
        temporary=target.with_name(target.name+'.tmp.'+uuid.uuid4().hex)
        with temporary.open('w') as stream:
            stream.write(json.dumps(dict(request_signature=signature))+'\n')
            for r in buffer:stream.write(json.dumps(r,separators=(',',':'))+'\n')
        temporary.replace(target)
        prior.update({(r['source_frame'],r['candidate_index']):r for r in buffer});buffer.clear()
    def batch():
        nonlocal processed,last_report
        if not items:return
        predictions,ms=predict_patches(model,patches,args.device,args.batch_size,torch);batch_times.append(ms)
        for i,item in enumerate(items):
            bbox=[float(item[k]) for k in ['bbox_x1','bbox_y1','bbox_x2','bbox_y2']]
            result=dict(source_frame=int(item['source_frame']),candidate_index=int(item['candidate_index']),
                bbox_x1=bbox[0],bbox_y1=bbox[1],bbox_x2=bbox[2],bbox_y2=bbox[3],source_confidence=float(item['confidence']),
                **joined(predictions[2*i],predictions[2*i+1],bbox,args.combined))
            buffer.append(result);processed+=1
        items.clear();patches.clear()
        if len(buffer)>=1000:flush()
        now=time.monotonic()
        if now-last_report>=20:
            progress=dict(processed_this_run=processed,cached_before_run=metadata['cached_before_run'],
                total_candidates=len(table),pending_candidates=len(pending),elapsed_seconds=round(now-start,1),
                remaining_seconds=round((len(pending)-processed)*(now-start)/max(processed,1),1))
            (output/'classifier_progress.json').write_text(json.dumps(progress,indent=2));print(json.dumps(progress),flush=True);last_report=now
    try:
        for f,group in pending.groupby('source_frame',sort=True):
            f=int(f)
            if current<0 or f-current>75:assert cap.set(cv2.CAP_PROP_POS_FRAMES,f);current=f-1
            while current<f:assert cap.grab();current+=1
            ok,image=cap.retrieve();assert ok and image.shape==(1080,1920,3)
            for item in group.to_dict('records'):
                bbox=[item[k] for k in ['bbox_x1','bbox_y1','bbox_x2','bbox_y2']]
                items.append(item);patches.extend(square_crop(image,bbox,s) for s in SCALES)
                if len(patches)>=args.batch_size:batch()
        batch();flush()
    except BaseException as exc:
        flush();metadata.update(status='interrupted' if isinstance(exc,KeyboardInterrupt) else 'failed',error=str(exc),completed_candidates=len(prior))
        (output/'classifier_run_metadata.json').write_text(json.dumps(metadata,indent=2));raise
    finally:cap.release()
    assert set(prior)==set(keys)
    write_csv(output/'classified_candidates.csv',[prior[key] for key in keys])
    metadata.update(status='complete',completed_candidates=len(prior),newly_processed=processed,
        elapsed_seconds=round(time.monotonic()-start,2),timed_batches=len(batch_times),
        median_batch_ms=float(np.median(batch_times)) if batch_times else None,
        p95_batch_ms=float(np.percentile(batch_times,95)) if batch_times else None)
    (output/'classifier_run_metadata.json').write_text(json.dumps(metadata,indent=2)+'\n')
    print(json.dumps({k:metadata[k] for k in ['status','completed_candidates','newly_processed','elapsed_seconds']},indent=2),flush=True)


def main():
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('stage',choices=['benchmark','infer'])
    p.add_argument('--model',type=Path,action='append',required=True)
    p.add_argument('--reference',type=Path,default=BASE/'reference.json')
    p.add_argument('--source-candidates',type=Path,default=ROOT/'data/derived/ball_targeted_redetection_20261116/yolo26/raw_ball_candidates.csv')
    p.add_argument('--output',type=Path,default=BASE/'classifier')
    p.add_argument('--video',type=Path,default=ROOT/'data/broadcast/BAR-ATH_broadcast-001.mp4')
    p.add_argument('--device',choices=['cpu','mps'],default='mps');p.add_argument('--batch-size',type=int,default=32)
    p.add_argument('--combined',choices=['geometric_mean','min'],default='geometric_mean')
    args=p.parse_args()
    if args.batch_size<2 or args.batch_size%2:p.error('--batch-size must be a positive even number of patches')
    {'benchmark':benchmark,'infer':infer}[args.stage](args)


if __name__=='__main__':main()
