"""Run full-frame YOLO26 only on an explicit source-frame manifest.

Raw candidates are preserved even when they overlap a requested negative region.
No candidate is promoted into the original tracking. Cached JSONL batches permit
resuming exactly the same request after interruption or a limited smoke run.
"""
import argparse
import csv
import hashlib
import json
import math
import os
from pathlib import Path
import time
import uuid


SOURCE_LABEL = 'YOLO26s_FULL_REDETECT'
CANDIDATE_FIELDS = ['source_frame', 'half', 'video_seconds', 'interval_id', 'reason', 'is_seed',
    'candidate_index', 'bbox_x1', 'bbox_y1', 'bbox_x2', 'bbox_y2', 'image_x', 'image_y',
    'confidence', 'class_id', 'source', 'candidate_group', 'overlaps_negative_region',
    'negative_region_max_iou', 'negative_region_center_overlap', 'negative_region_annotations']
FRAME_FIELDS = ['source_frame', 'half', 'video_seconds', 'interval_id', 'reason', 'is_seed',
    'source_width', 'source_height', 'ball_candidate_count', 'overlapping_candidate_count',
    'outside_region_candidate_count', 'negative_region_count', 'inference_ms', 'result']


def digest(path):
    h = hashlib.sha256()
    with Path(path).open('rb') as stream:
        for chunk in iter(lambda: stream.read(4 * 1024 * 1024), b''):
            h.update(chunk)
    return h.hexdigest()


def atomic_json(path, data):
    temporary = path.with_name(path.name + '.tmp.' + uuid.uuid4().hex)
    temporary.write_text(json.dumps(data, ensure_ascii=False, indent=2) + '\n')
    temporary.replace(path)


def truth(value):
    return str(value).lower() in ('true', '1', 'yes')


def load_manifest(path):
    with path.open(newline='', encoding='utf-8-sig') as stream:
        rows = list(csv.DictReader(stream))
    if not rows:
        raise ValueError('Manifest has no selected frames')
    for row in rows:
        row['source_frame'] = int(row['source_frame'])
        row['half'] = int(row.get('half', 0))
        row['is_seed'] = truth(row.get('is_seed', False))
        row['interval_id'] = int(row.get('interval_id', -1))
        if 'selected_for_yolo' in row and not truth(row['selected_for_yolo']):
            raise ValueError('Manifest must contain only selected_for_yolo frames')
    frames = [r['source_frame'] for r in rows]
    if frames != sorted(set(frames)) or frames[0] < 0:
        raise ValueError('source_frame must be unique, increasing, nonnegative and zero based')
    return rows


def load_regions(path):
    result = {}
    with path.open() as stream:
        for line in stream:
            if not line.strip():
                continue
            record = json.loads(line)
            frame = int(record['source_frame'])
            if frame in result:
                raise ValueError(f'Duplicate negative-region frame {frame}')
            for region in record['regions']:
                box = region['bbox_xyxy']
                if len(box) != 4 or not all(math.isfinite(float(v)) for v in box) or box[2] <= box[0] or box[3] <= box[1]:
                    raise ValueError(f'Invalid negative-region box at frame {frame}')
            result[frame] = record['regions']
    return result


def annotate(box, regions):
    x1, y1, x2, y2 = map(float, box[:4])
    cx, cy = (x1+x2)/2, (y1+y2)/2
    area = (x2-x1)*(y2-y1)
    annotations = []
    for region in regions:
        a, b, c, d = map(float, region['bbox_xyxy'])
        intersection = max(0., min(x2,c)-max(x1,a))*max(0., min(y2,d)-max(y1,b))
        center_inside = a <= cx <= c and b <= cy <= d
        if not intersection and not center_inside:
            continue
        iou = intersection / max(area+(c-a)*(d-b)-intersection, 1e-12)
        annotations.append(dict(kind=region.get('kind', 'unspecified'),
            certainty=region.get('certainty', 'suspected_only'), bbox_xyxy=region['bbox_xyxy'],
            iou=iou, center_inside=center_inside, candidate_overlap_fraction=intersection/max(area,1e-12)))
    return dict(candidate_group=('overlaps_requested_region' if annotations else
        ('outside_requested_region' if regions else 'no_requested_region')),
        overlaps_negative_region=bool(annotations),
        negative_region_max_iou=max([a['iou'] for a in annotations], default=0.),
        negative_region_center_overlap=any(a['center_inside'] for a in annotations),
        negative_region_annotations=json.dumps(annotations, ensure_ascii=False, separators=(',',':')))


def cached_records(folder, signature):
    result = {}
    for path in sorted(folder.glob('batch_*.jsonl')):
        with path.open() as stream:
            header = json.loads(next(stream))
            if header.get('request_signature') != signature:
                raise ValueError(f'Cache request signature differs: {path}')
            for line in stream:
                record = json.loads(line)
                frame = int(record['source_frame'])
                if frame in result:
                    raise ValueError(f'Duplicate cached source frame {frame}')
                result[frame] = record
    return result


def save_batch(folder, signature, records):
    if not records:
        return
    first, last = records[0]['source_frame'], records[-1]['source_frame']
    path = folder / f'batch_{first:06d}_{last:06d}.jsonl'
    temporary = path.with_name(path.name + '.tmp.' + uuid.uuid4().hex)
    with temporary.open('w') as stream:
        stream.write(json.dumps(dict(request_signature=signature))+'\n')
        for record in records:
            stream.write(json.dumps(record, ensure_ascii=False, separators=(',',':'))+'\n')
    temporary.replace(path)


def export_csv(output, records, frames):
    with (output/'frame_results.csv').open('w', newline='') as fs, \
         (output/'raw_ball_candidates.csv').open('w', newline='') as cs, \
         (output/'overlapping_region_candidates.csv').open('w', newline='') as ns, \
         (output/'alternate_candidates.csv').open('w', newline='') as alt:
        fw = csv.DictWriter(fs, FRAME_FIELDS);fw.writeheader()
        writers = [csv.DictWriter(stream, CANDIDATE_FIELDS) for stream in [cs,ns,alt]]
        for writer in writers:writer.writeheader()
        for frame in frames:
            if frame not in records:
                continue
            record = records[frame];fw.writerow({k:record[k] for k in FRAME_FIELDS})
            for candidate in record['candidates']:
                writers[0].writerow(candidate)
                writers[1 if candidate['overlaps_negative_region'] else 2].writerow(candidate)
    with (output/'invalid_model_boxes.jsonl').open('w') as stream:
        for frame in frames:
            for row in records.get(frame,{}).get('invalid_candidates',[]):
                stream.write(json.dumps(row,separators=(',',':'))+'\n')


def run(args):
    output = args.output.resolve();output.mkdir(parents=True, exist_ok=True)
    # Importing Ultralytics otherwise tries to write outside the project.
    os.environ['YOLO_CONFIG_DIR'] = str(output/'runtime/ultralytics')
    os.environ['MPLCONFIGDIR'] = str(output/'runtime/matplotlib')
    os.environ['YOLO_AUTOINSTALL'] = 'false'
    import cv2
    import numpy as np
    import torch
    import ultralytics
    from ultralytics import YOLO
    torch.set_num_threads(args.threads)
    if args.device == 'mps':
        if not torch.backends.mps.is_available():
            raise RuntimeError('MPS requested but unavailable in this execution environment')
        (torch.ones(1, device='mps') + 1).cpu()
    full_rows = load_manifest(args.manifest)
    rows = full_rows[:args.limit] if args.limit is not None else full_rows
    regions = load_regions(args.negative_regions)
    video = args.video.resolve();model_path = args.model.resolve()
    source_label='YOLO26'+model_path.stem.removeprefix('yolo26')+'_FULL_REDETECT'
    stat = video.stat()
    spec = dict(manifest_sha256=digest(args.manifest), negative_regions_sha256=digest(args.negative_regions),
        model_sha256=digest(model_path), model=str(model_path), video=str(video),
        video_bytes=stat.st_size, video_mtime_ns=stat.st_mtime_ns,
        imgsz=args.imgsz, rect=True, conf=args.conf, iou=args.iou, classes=[32],
        max_det=300, device=args.device, threads=args.threads, batch_size=args.batch_size,
        full_frame=True, roi=False, tiles=False, augmentation=False,
        source=source_label, schema_version=1)
    if args.quantize is not None:spec['quantize']=args.quantize
    if args.nms_head:spec['nms_head']=True
    signature = hashlib.sha256(json.dumps(spec,sort_keys=True).encode()).hexdigest()
    cache = output/'batches';cache.mkdir(exist_ok=True)
    prior = cached_records(cache, signature)
    all_frames = {r['source_frame'] for r in full_rows}
    if not set(prior) <= all_frames:
        raise ValueError('Cache contains frames not in this manifest')
    cap = cv2.VideoCapture(str(video));assert cap.isOpened()
    fps = float(cap.get(cv2.CAP_PROP_FPS));width=int(cap.get(3));height=int(cap.get(4));count=int(cap.get(7))
    if not math.isfinite(fps) or fps <= 0 or full_rows[-1]['source_frame'] >= count:
        raise ValueError('Video metadata and source-frame manifest are incompatible')
    for row in full_rows:
        if 'video_seconds' in row and abs(float(row['video_seconds'])-row['source_frame']/fps)>1e-6:
            raise ValueError('Manifest video_seconds differs from source_frame / source FPS')
    metadata = dict(status='running',request_signature=signature,request=spec,
        total_manifest_frames=len(full_rows),requested_frames=len(rows),limit=args.limit,
        previously_cached_frames=len(prior),source_fps=fps,source_width=width,source_height=height,
        source_video_frames=count,frame_convention='zero-based source_frame / source FPS',
        runtime=dict(python=os.sys.version,torch=torch.__version__,ultralytics=ultralytics.__version__,
                     opencv=cv2.__version__,torch_threads=torch.get_num_threads(),
                     mps_available=torch.backends.mps.is_available(),selected_device=args.device),
        raw_candidates_preserved=True,negative_region_overlap_is_annotation_only=True,
        original_tracking_changed=False,highest_confidence_automatically_promoted=False)
    atomic_json(output/'run_metadata.json',metadata)
    pending = [r for r in rows if r['source_frame'] not in prior]
    model = YOLO(str(model_path)) if pending else None
    if model is not None and (model.task!='detect' or model.names.get(32)!='sports ball'):
        raise ValueError('Expected YOLO26 COCO detection model with sports ball class 32')
    if model is not None and args.nms_head:model.model.end2end=False
    started=time.monotonic();last_report=started;current=-1;processed=0;buffer=[]

    def flush():
        if buffer:
            save_batch(cache,signature,buffer)
            prior.update({r['source_frame']:r for r in buffer});buffer.clear()

    try:
        for begin in range(0,len(pending),args.batch_size):
            chunk=pending[begin:begin+args.batch_size];images=[]
            for row in chunk:
                frame=row['source_frame']
                if current<0 or frame-current>75:
                    if not cap.set(cv2.CAP_PROP_POS_FRAMES,frame) or abs(cap.get(cv2.CAP_PROP_POS_FRAMES)-frame)>.5:
                        raise RuntimeError(f'Cannot seek to exact source frame {frame}')
                    current=frame-1
                while current<frame:
                    if not cap.grab():raise RuntimeError(f'Cannot decode source frame {frame}')
                    current+=1
                ok,image=cap.retrieve()
                if not ok or image.shape[:2]!=(height,width):raise RuntimeError(f'Invalid frame {frame}')
                images.append(image)
            precision_options={} if args.quantize is None else dict(quantize=args.quantize)
            if args.nms_head:precision_options['nms']=True
            predictions=model.predict(images if len(images)>1 else images[0],classes=[32],device=args.device,
                imgsz=args.imgsz,rect=True,conf=args.conf,iou=args.iou,max_det=300,save=False,verbose=False,augment=False,
                **precision_options)
            metadata['runtime']['actual_parameter_dtype']=str(next(model.predictor.model.backend.model.parameters()).dtype)
            metadata['runtime']['actual_end2end']=bool(model.predictor.model.end2end)
            if len(predictions)!=len(chunk):raise RuntimeError('Model result count differs from input frame count')
            for row,result in zip(chunk,predictions):
                if tuple(result.orig_shape)!=(height,width):raise RuntimeError('Model coordinates differ from source image dimensions')
                frame=row['source_frame'];requested_regions=regions.get(frame,[])
                boxes=np.empty((0,6)) if result.boxes is None else result.boxes.data.detach().cpu().numpy()
                if not np.isfinite(boxes).all():raise RuntimeError(f'Nonfinite model output at frame {frame}')
                common=dict(source_frame=frame,half=row['half'],video_seconds=frame/fps,
                    interval_id=row['interval_id'],reason=row.get('reason',''),is_seed=row['is_seed'])
                candidates=[];invalid_candidates=[]
                for index,box in enumerate(boxes):
                    x1,y1,x2,y2,confidence,class_id=map(float,box[:6])
                    if int(class_id)!=32:raise RuntimeError(f'Unexpected class {class_id} at frame {frame}')
                    if x2<=x1 or y2<=y1:
                        invalid_candidates.append(dict(source_frame=frame,raw_model_index=index,
                            raw_box=list(map(float,box[:6])),reason='nonpositive-area box after scaling/clipping; not a usable object location'))
                        continue
                    candidates.append(dict(**common,candidate_index=len(candidates),bbox_x1=x1,bbox_y1=y1,
                        bbox_x2=x2,bbox_y2=y2,image_x=(x1+x2)/2,image_y=(y1+y2)/2,
                        confidence=confidence,class_id=32,source=source_label,**annotate(box,requested_regions)))
                overlapping=sum(c['overlaps_negative_region'] for c in candidates)
                buffer.append(dict(**common,source_width=width,source_height=height,
                    ball_candidate_count=len(candidates),overlapping_candidate_count=overlapping,
                    outside_region_candidate_count=len(candidates)-overlapping,negative_region_count=len(requested_regions),
                    inference_ms=float(result.speed.get('inference',0.)),result='candidates' if len(candidates) else 'no_ball_candidate',
                    candidates=candidates,invalid_candidates=invalid_candidates))
                processed+=1
                if len(buffer)>=500:flush()
            now=time.monotonic()
            if now-last_report>=20 or processed==len(pending):
                elapsed=now-started
                progress=dict(processed_this_run=processed,total_pending=len(pending),
                    cached_before_run=metadata['previously_cached_frames'],last_source_frame=chunk[-1]['source_frame'],
                    elapsed_seconds=round(elapsed,2),frames_per_second=round(processed/max(elapsed,1e-9),3),
                    remaining_seconds=round((len(pending)-processed)*elapsed/max(processed,1),1))
                atomic_json(output/'progress.json',progress);print(json.dumps(progress),flush=True);last_report=now
        flush()
        requested_frames=[r['source_frame'] for r in rows]
        if not all(f in prior for f in requested_frames):raise RuntimeError('Not all requested frames have a result')
        export_csv(output,prior,requested_frames)
        metadata.update(status='complete_limited' if args.limit is not None and len(rows)<len(full_rows) else 'complete',
            completed_frames=len(requested_frames),ball_candidates=sum(prior[f]['ball_candidate_count'] for f in requested_frames),
            frames_without_ball_candidate=sum(prior[f]['ball_candidate_count']==0 for f in requested_frames),
            elapsed_seconds=round(time.monotonic()-started,2))
        metadata['invalid_boxes_preserved_separately']=sum(len(prior[f].get('invalid_candidates',[])) for f in requested_frames)
    except BaseException as exc:
        flush();export_csv(output,prior,[r['source_frame'] for r in rows])
        metadata.update(status='interrupted' if isinstance(exc,KeyboardInterrupt) else 'failed',error=str(exc),
            completed_frames=sum(r['source_frame'] in prior for r in rows))
        atomic_json(output/'run_metadata.json',metadata)
        raise
    finally:cap.release()
    atomic_json(output/'run_metadata.json',metadata)
    print(json.dumps({k:metadata[k] for k in ['status','completed_frames','ball_candidates','frames_without_ball_candidate','elapsed_seconds']},indent=2))


def main(argv=None):
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--manifest',type=Path,required=True)
    parser.add_argument('--negative-regions',type=Path,required=True)
    parser.add_argument('--video',type=Path,required=True)
    parser.add_argument('--output',type=Path,required=True)
    parser.add_argument('--model',type=Path,required=True)
    parser.add_argument('--device',choices=['cpu','mps'],default='cpu')
    parser.add_argument('--quantize',type=int,choices=[16,32])
    parser.add_argument('--nms-head',action='store_true',help='Use the one-to-many head with NMS explicitly')
    parser.add_argument('--limit',type=int)
    parser.add_argument('--threads',type=int,default=4)
    parser.add_argument('--batch-size',type=int,default=1)
    parser.add_argument('--imgsz',type=int,default=1280)
    parser.add_argument('--conf',type=float,default=.05)
    parser.add_argument('--iou',type=float,default=.7)
    args=parser.parse_args(argv)
    if args.limit is not None and args.limit<=0:parser.error('--limit must be positive')
    if min(args.threads,args.batch_size,args.imgsz)<1 or not 0<args.conf<1 or not 0<args.iou<1:
        parser.error('Positive threads, batch-size and imgsz, and confidence/IoU in (0,1), required')
    run(args)


if __name__=='__main__':main()
