"""Compare YOLO26 models on exact reviewed ball/non-ball objects in full frames.

This measures a purposive reference-object challenge set, not whole-match AP.
All models receive the same original frame and inference settings. A detector
not finding a candidate is a miss, not proof that an unlabelled object is not a ball.
"""
from pathlib import Path
import argparse
import csv
import gc
import hashlib
import json
import math
import os
import time

import numpy as np

ROOT=Path(__file__).resolve().parents[1]
DEFAULT_OUT=ROOT/'data/derived/ball_yolo26_model_audit_20261116'
THRESHOLDS=[.05,.10,.15,.20,.25,.30,.40,.50,.60,.70,.80,.90]
BALL_LABELS={'ball','match_ball','match_ball_observed','real_match_ball'}
NONBALL_LABELS={'nonball','not_ball','jersey_logo','painted_penalty_mark_not_ball','penalty_mark_false_positive'}


def sha(path):
    h=hashlib.sha256()
    with Path(path).open('rb') as f:
        for chunk in iter(lambda:f.read(4*1024*1024),b''):h.update(chunk)
    return h.hexdigest()


def read_reference(path):
    data=json.loads(path.read_text());records=data['records'] if isinstance(data,dict) else data
    result=[]
    for index,row in enumerate(records):
        label=row['label']
        if label not in BALL_LABELS|NONBALL_LABELS:continue
        point=row.get('point',row.get('pixel_xy',row.get('source_pixel_xy')))
        bbox=row.get('bbox_xyxy',row.get('bbox',row.get('source_bbox_xyxy')))
        if point is None:raise ValueError(f'Reference {index} has no image point')
        if len(point)!=2 or not np.isfinite(point).all():raise ValueError('Invalid reference point')
        if bbox is not None and (len(bbox)!=4 or not np.isfinite(bbox).all() or bbox[2]<=bbox[0] or bbox[3]<=bbox[1]):
            raise ValueError('Invalid reference bounding box')
        result.append(dict(reference_id=str(row.get('reference_id',row.get('id',index))),
            source_frame=int(row['source_frame']),label='ball' if label in BALL_LABELS else 'nonball',
            point=list(map(float,point)),bbox_xyxy=None if bbox is None else list(map(float,bbox)),
            kind=row.get('kind',row.get('object_kind',row.get('case_type',label))),
            split=row.get('split','challenge')))
    if not result or len({r['label'] for r in result})!=2:raise ValueError('Both reviewed ball and nonball references required')
    keys=[(r['source_frame'],*np.round(r['point'],3)) for r in result]
    if len(keys)!=len(set(keys)):raise ValueError('Duplicate exact reference objects')
    return result


def object_scores(boxes,reference):
    """Match the reviewed object location, never the highest ball in the frame."""
    boxes=np.asarray(boxes,float).reshape(-1,6)
    if not len(boxes):return dict(local_score=0.,point_score=0.,iou_score=0.,matched_index=None)
    xy=np.asarray(reference['point']);gt=reference['bbox_xyxy']
    centers=(boxes[:,:2]+boxes[:,2:4])/2;sides=boxes[:,2:4]-boxes[:,:2]
    distance=np.linalg.norm(centers-xy,axis=1)
    if gt is None:
        tolerance=4.;size_cap=300.;ious=np.zeros(len(boxes))
    else:
        gt=np.asarray(gt);wh=gt[2:]-gt[:2];tolerance=max(3.,.25*float(np.linalg.norm(wh)))
        size_cap=max(16.,4*float(wh.max()))
        intersection=np.maximum(0,np.minimum(boxes[:,2:4],gt[2:])-np.maximum(boxes[:,:2],gt[:2])).prod(1)
        ious=intersection/np.maximum(1e-12,sides.prod(1)+wh.prod()-intersection)
    contains=((boxes[:,:2]-2<=xy)&(boxes[:,2:4]+2>=xy)).all(1)
    near=(contains|(distance<=tolerance))&(sides.max(1)<=size_cap)
    overlap=ious>=.30
    matched=near|overlap
    indices=np.flatnonzero(matched)
    selected=int(indices[np.argmax(boxes[indices,4])]) if len(indices) else None
    return dict(local_score=float(boxes[matched,4].max()) if matched.any() else 0.,
        point_score=float(boxes[near,4].max()) if near.any() else 0.,
        iou_score=float(boxes[overlap,4].max()) if overlap.any() else 0.,matched_index=selected)


def measures(labels,scores,threshold):
    truth=np.asarray(labels)=='ball';pred=np.asarray(scores)>=threshold
    tp=int((pred&truth).sum());fp=int((pred&~truth).sum());fn=int((~pred&truth).sum());tn=int((~pred&~truth).sum())
    recall=tp/max(tp+fn,1);specificity=tn/max(tn+fp,1);precision=tp/max(tp+fp,1)
    return dict(threshold=float(threshold),ball_samples=int(truth.sum()),nonball_samples=int((~truth).sum()),
        true_ball_supported=tp,nonball_supported_as_ball=fp,ball_missed=fn,nonball_not_detected=tn,
        recall=recall,specificity=specificity,false_positive_rate=1-specificity,precision_on_challenge=precision,
        balanced_accuracy=(recall+specificity)/2,f1_on_challenge=2*tp/max(2*tp+fp+fn,1))


def ranking_stats(labels,scores):
    y=np.asarray(labels)=='ball';s=np.asarray(scores,float)
    if y.all() or (~y).all():return dict(roc_auc=None,reference_object_average_precision=None)
    positive=s[y];negative=s[~y]
    auc=float(((positive[:,None]>negative[None,:])+.5*(positive[:,None]==negative[None,:])).mean())
    ap=0.;last_recall=0.
    for threshold in np.unique(s)[::-1]:
        pred=s>=threshold;recall=float((pred&y).sum()/y.sum());precision=float((pred&y).sum()/pred.sum())
        ap+=(recall-last_recall)*precision;last_recall=recall
    return dict(roc_auc=auc,reference_object_average_precision=float(ap))


def evaluate(output,records,detections,performances):
    rows=[];models=list(performances)
    for model in models:
        for record in records:
            boxes=detections[model][record['source_frame']]['boxes']
            rows.append(dict(model=model,**record,**object_scores(boxes,record)))
    with (output/'reference_predictions.csv').open('w',newline='') as stream:
        fields=['model','reference_id','source_frame','label','kind','split','point','bbox_xyxy','local_score','point_score','iou_score','matched_index']
        writer=csv.DictWriter(stream,fields);writer.writeheader();writer.writerows(rows)
    reports=[];threshold_rows=[]
    for model in models:
        own=[r for r in rows if r['model']==model]
        groups={split:[r for r in own if r['split']==split] for split in sorted({r['split'] for r in own})}
        groups['all']=own
        group_report={}
        for split,group in groups.items():
            labels=[r['label'] for r in group];scores=[r['local_score'] for r in group]
            values=[measures(labels,scores,t) for t in THRESHOLDS]
            threshold_rows.extend(dict(model=model,split=split,**v) for v in values)
            group_report[split]=dict(samples=len(group),**ranking_stats(labels,scores),thresholds=values)
        calibration_name=next((s for s in ['calibration','selection','development','train'] if s in groups), 'all')
        calibration=groups[calibration_name]
        values=group_report[calibration_name]['thresholds']
        eligible=[r for r in values if r['recall']>=.9]
        pool=eligible or values
        chosen=max(pool,key=lambda r:(r['balanced_accuracy'],r['specificity'],r['recall'],r['threshold']))
        report=dict(model=model,performance=performances[model],groups=group_report,
            threshold_selected_from=calibration_name,selected_threshold=chosen['threshold'],
            calibration_at_selected_threshold=chosen,calibration_recall_target=.9,
            calibration_recall_target_met=bool(eligible))
        for split,group in groups.items():
            if split in ['holdout','heldout','test']:
                report['heldout_at_selected_threshold']=measures([r['label'] for r in group],[r['local_score'] for r in group],chosen['threshold'])
        reports.append(report)
    with (output/'threshold_comparison.csv').open('w',newline='') as stream:
        fields=['model','split',*threshold_rows[0].keys()];fields=list(dict.fromkeys(fields))
        writer=csv.DictWriter(stream,fields);writer.writeheader();writer.writerows(threshold_rows)
    ranked=sorted(reports,key=lambda r:(-r['calibration_at_selected_threshold']['balanced_accuracy'],
        -r['calibration_at_selected_threshold']['specificity'],r['performance']['median_ms']))
    summary=dict(scope='Purposive exact-object ball/nonball challenge; not whole-match detection mAP or field accuracy',
        matching='Local confidence is maximum detection containing the reviewed point (+2px) or near its centre, with size cap, or IoU>=.30. Whole-frame maximum confidence is never substituted.',
        missed_detection_interpretation='A miss on a reviewed ball is a false negative. A miss on an unlabelled object is not proof it is nonball.',
        models=reports,calibration_ranking=[r['model'] for r in ranked],
        recommended_model_from_calibration=ranked[0]['model'],recommended_threshold=ranked[0]['selected_threshold'],
        selection_note='Calibration balanced accuracy, preferring >=90% ball recall if attainable; ties use specificity and measured latency. Heldout results are reported, not used to tune the threshold.')
    (output/'benchmark_summary.json').write_text(json.dumps(summary,indent=2)+'\n')
    print(json.dumps(dict(recommended_model=summary['recommended_model_from_calibration'],
        threshold=summary['recommended_threshold'],ranking=summary['calibration_ranking']),indent=2),flush=True)


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--model',type=Path,action='append',required=True)
    parser.add_argument('--reference',type=Path,default=DEFAULT_OUT/'reference.json')
    parser.add_argument('--output',type=Path,default=DEFAULT_OUT/'benchmark')
    parser.add_argument('--video',type=Path,default=ROOT/'data/broadcast/BAR-ATH_broadcast-001.mp4')
    parser.add_argument('--device',choices=['cpu','mps'],default='mps')
    parser.add_argument('--quantize',type=int,choices=[16,32],default=32)
    args=parser.parse_args();output=args.output.resolve();output.mkdir(parents=True,exist_ok=True)
    os.environ['YOLO_CONFIG_DIR']=str(output/'runtime/ultralytics');os.environ['MPLCONFIGDIR']=str(output/'runtime/matplotlib');os.environ['YOLO_AUTOINSTALL']='false'
    import cv2
    import torch
    import ultralytics
    from ultralytics import YOLO
    torch.set_num_threads(4)
    if args.device=='mps' and not torch.backends.mps.is_available():raise RuntimeError('MPS unavailable; run with GPU access or explicitly use CPU')
    records=read_reference(args.reference);frames=sorted({r['source_frame'] for r in records})
    cap=cv2.VideoCapture(str(args.video));assert cap.isOpened()
    images={}
    for f in frames:
        assert cap.set(cv2.CAP_PROP_POS_FRAMES,f);ok,image=cap.read();assert ok and image.shape==(1080,1920,3)
        images[f]=image
    cap.release()
    detections={};performances={};cache=output/'model_cache';cache.mkdir(exist_ok=True)
    config=dict(imgsz=1280,rect=True,classes=[32],conf=.001,iou=.7,max_det=300,device=args.device,verbose=False,augment=False)
    if args.quantize==16:config['quantize']=16
    def sync():
        if args.device=='mps':torch.mps.synchronize()
    for path in args.model:
        path=path.resolve();name=path.stem
        if name in performances:raise ValueError('Model file stems must be distinct')
        signature=hashlib.sha256(json.dumps(dict(model_sha256=sha(path),reference_sha256=sha(args.reference),config=config,
            inference_path='one-to-many + NMS; end2end=False'),sort_keys=True).encode()).hexdigest()
        saved=cache/f'{name}.json'
        if saved.exists():
            data=json.loads(saved.read_text());assert data['signature']==signature,f'Cache settings differ for {name}'
            detections[name]={int(r['source_frame']):r for r in data['detections']};performances[name]=data['performance']
            print(f'{name}: reused {len(frames)} frames',flush=True);continue
        model=YOLO(str(path));assert model.task=='detect' and model.names.get(32)=='sports ball'
        model.model.end2end=False
        assert model.model.end2end is False and model.model.model[-1].end2end is False
        warm=[]
        for _ in range(3):
            started=time.perf_counter();result=model.predict(images[frames[0]],**config)[0];result.boxes.data.cpu();sync();warm.append((time.perf_counter()-started)*1000)
        results=[];elapsed=[]
        for i,f in enumerate(frames):
            started=time.perf_counter();prediction=model.predict(images[f],**config)[0]
            boxes=np.empty((0,6)) if prediction.boxes is None else prediction.boxes.data.detach().cpu().numpy()
            sync();milliseconds=(time.perf_counter()-started)*1000
            assert tuple(prediction.orig_shape)==(1080,1920) and np.isfinite(boxes).all()
            results.append(dict(model=name,source_frame=f,boxes=boxes.tolist(),wall_ms=milliseconds,inference_ms=float(prediction.speed.get('inference',0.))))
            elapsed.append(milliseconds)
            if (i+1)%25==0 or i+1==len(frames):print(f'{name}: {i+1}/{len(frames)} frames',flush=True)
        timing=dict(model_path=str(path),model_sha256=sha(path),warmup_ms=warm,
            actual_backend_parameter_dtype=str(next(model.predictor.model.backend.model.parameters()).dtype),
            model_end2end=bool(model.model.end2end),head_end2end=bool(model.model.model[-1].end2end),
            inference_path='one-to-many + NMS',
            timed_frames=len(frames),median_ms=float(np.median(elapsed)),p95_ms=float(np.percentile(elapsed,95)),
            mean_ms=float(np.mean(elapsed)),fps_from_mean=1000/float(np.mean(elapsed)),
            note='wall time includes pre/postprocessing and MPS synchronization; image decoding excluded')
        saved.write_text(json.dumps(dict(signature=signature,performance=timing,detections=results),separators=(',',':')))
        detections[name]={r['source_frame']:r for r in results};performances[name]=timing
        del model,result,prediction;gc.collect()
        if args.device=='mps':torch.mps.empty_cache()
    with (output/'benchmark_detections.jsonl').open('w') as stream:
        for name in detections:
            for f in frames:stream.write(json.dumps(detections[name][f],separators=(',',':'))+'\n')
    (output/'performance.json').write_text(json.dumps(dict(models=performances,config=config,
        reference_sha256=sha(args.reference),reference_objects=len(records),source_frames=len(frames),
        runtime=dict(torch=torch.__version__,ultralytics=ultralytics.__version__,opencv=cv2.__version__,device=args.device)),indent=2))
    evaluate(output,records,detections,performances)


if __name__=='__main__':main()
