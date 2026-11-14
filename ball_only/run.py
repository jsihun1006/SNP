"""Portable, resumable ball candidate generation for explicit source-video ranges."""
import argparse
from contextlib import contextmanager
from decimal import Decimal, InvalidOperation
import hashlib
import json
import os
from pathlib import Path
import time
import traceback
import uuid

VERSION = 'ball_only_original_frame_ranges_v1'


def atomic(path, value):
    path=Path(path);path.parent.mkdir(parents=True,exist_ok=True)
    tmp=path.with_name(path.name+'.tmp-'+uuid.uuid4().hex)
    with tmp.open('w',encoding='utf-8') as h:
        json.dump(value,h,ensure_ascii=False,indent=2,allow_nan=False);h.flush();os.fsync(h.fileno())
    os.replace(tmp,path)


def read(path):
    return json.loads(Path(path).read_text(encoding='utf-8-sig'))


def sha(path):
    result=hashlib.sha256()
    with Path(path).open('rb') as h:
        for block in iter(lambda:h.read(4*1024*1024),b''):result.update(block)
    return result.hexdigest()


def environment():
    base=Path(__file__).resolve().parent
    for name in ('.cache','.tmp','.cache/pip','.cache/torch','.cache/ultralytics'):(base/name).mkdir(parents=True,exist_ok=True)
    os.environ.update(TEMP=str(base/'.tmp'),TMP=str(base/'.tmp'),YOLO_CONFIG_DIR=str(base/'.cache/ultralytics'),TORCH_HOME=str(base/'.cache/torch'))
    os.environ['PYTHONDONTWRITEBYTECODE']='1'
    return base


def asset_identity(path,cache):
    path=Path(path).resolve(strict=True);st=path.stat()
    key=hashlib.sha256(str(path).encode()).hexdigest();stamp=dict(path=str(path),bytes=st.st_size,mtime_ns=st.st_mtime_ns)
    marker=Path(cache)/(key+'.json')
    if marker.exists():
        old=read(marker)
        if old.get('stamp')==stamp:return dict(sha256=old['sha256'],bytes=st.st_size)
    digest=sha(path)
    if path.stat().st_size!=st.st_size or path.stat().st_mtime_ns!=st.st_mtime_ns:raise ValueError('An input changed while hashing')
    atomic(marker,dict(stamp=stamp,sha256=digest))
    return dict(sha256=digest,bytes=st.st_size)


def time_frame(value,fps,total,*,end=False):
    text=str(value).strip()
    if text.lower() in ('end','eof'):
        if not end:raise ValueError('EOF is allowed only as an end boundary')
        return total
    parts=text.split(':')
    try:
        if len(parts)==1:seconds=Decimal(parts[0])
        elif len(parts) in (2,3):
            if any(not p.isdigit() for p in parts[:-1]):raise ValueError('Hours/minutes must be whole numbers')
            second=Decimal(parts[-1]);minute=int(parts[-2]);hour=int(parts[0]) if len(parts)==3 else 0
            if not 0<=second<60 or (len(parts)==3 and not 0<=minute<60):raise ValueError('Invalid timestamp')
            seconds=Decimal(hour*3600+minute*60)+second
        else:raise ValueError('Use seconds or HH:MM:SS[.fraction]')
        if not seconds.is_finite() or seconds<0:raise ValueError('Time must be finite and nonnegative')
        frame=seconds*Decimal(str(fps));nearest=frame.to_integral_value()
        if abs(frame-nearest)>Decimal('0.00001'):raise ValueError('Timestamp must lie on a source frame boundary')
        return int(nearest)
    except InvalidOperation as exc:raise ValueError('Invalid timestamp: '+text) from exc


@contextmanager
def output_lock(folder):
    with (Path(folder)/'run.lock').open('a+b',buffering=0) as h:
        if not os.fstat(h.fileno()).st_size:h.write(b'\0')
        if os.name=='nt':
            import msvcrt
            def acquire():h.seek(0);msvcrt.locking(h.fileno(),msvcrt.LK_NBLCK,1)
            def release():h.seek(0);msvcrt.locking(h.fileno(),msvcrt.LK_UNLCK,1)
        else:
            import fcntl
            def acquire():fcntl.flock(h.fileno(),fcntl.LOCK_EX|fcntl.LOCK_NB)
            def release():fcntl.flock(h.fileno(),fcntl.LOCK_UN)
        try:acquire()
        except OSError as exc:raise RuntimeError('Another process owns this output folder') from exc
        try:yield
        finally:release()


def resume_at(out,config):
    position=config['start_frame'];end=config['end_frame'];chunk=config['chunk_frames']
    while position<end:
        folder=Path(out)/'chunks'/f'{position:09d}';marker=folder/'manifest.json'
        if not marker.exists():break
        m=read(marker);last=min(end,position+chunk);payload=folder/'detections.jsonl'
        if (m.get('status')!='complete' or m.get('fingerprint')!=config['fingerprint'] or m.get('start_frame')!=position or m.get('end_frame')!=last or m.get('frames')!=last-position):raise ValueError('Checkpoint settings/range mismatch')
        digest=hashlib.sha256();count=0
        with payload.open('rb') as h:
            for line in h:
                digest.update(line);row=json.loads(line)
                if row.get('frame_idx')!=position+count or row.get('source_frame')!=position+count or row.get('part_frame')!=position+count-config['start_frame'] or row.get('part_id')!=config['part_id']:raise ValueError('Checkpoint frame identity mismatch')
                count+=1
        if count!=last-position or digest.hexdigest()!=m['sha256']:raise ValueError('Checkpoint content changed')
        position=last
    return position


def consolidate(out,config):
    target=Path(out)/'candidates.jsonl';tmp=target.with_name(target.name+'.tmp-'+uuid.uuid4().hex);digest=hashlib.sha256();count=0
    with tmp.open('wb') as dst:
        for first in range(config['start_frame'],config['end_frame'],config['chunk_frames']):
            folder=Path(out)/'chunks'/f'{first:09d}';m=read(folder/'manifest.json');chunk_digest=hashlib.sha256();n=0
            with (folder/'detections.jsonl').open('rb') as src:
                for line in src:
                    row=json.loads(line);expected=config['start_frame']+count
                    if row['source_frame']!=expected or row['frame_idx']!=expected:raise ValueError('Final frame coverage mismatch')
                    dst.write(line);digest.update(line);chunk_digest.update(line);count+=1;n+=1
            if n!=m['frames'] or chunk_digest.hexdigest()!=m['sha256']:raise ValueError('Chunk changed during final export')
        dst.flush();os.fsync(dst.fileno())
    if count!=config['end_frame']-config['start_frame']:raise ValueError('Final frame count mismatch')
    os.replace(tmp,target);return target,digest.hexdigest(),count


def run(video,weights,out,*,start,end,part_id,chunk_frames=32,full_batch=1,roi_batch=4,tile_batch=8,max_rois=2,
        gpu_target_low=98.,gpu_target_high=99.5,device=0,detector_factory=None,controller_factory=None,expected_video_sha256=None):
    base=environment()
    import cv2
    from .detector import BallDetector,VERSION as DETECTOR_VERSION
    from .runtime_control import RuntimeController
    out=Path(out).resolve();out.mkdir(parents=True,exist_ok=True);video=Path(video).resolve(strict=True);weights=Path(weights).resolve(strict=True)
    if min(chunk_frames,full_batch,roi_batch,tile_batch)<1 or max_rois<0:raise ValueError('Chunk/batch sizes must be positive; max_rois >= 0')
    if not part_id.strip():raise ValueError('part_id is required')
    cap=cv2.VideoCapture(str(video))
    if not cap.isOpened():raise ValueError('Cannot open source video')
    try:total=int(cap.get(cv2.CAP_PROP_FRAME_COUNT));fps=float(cap.get(cv2.CAP_PROP_FPS));width=int(cap.get(cv2.CAP_PROP_FRAME_WIDTH));height=int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    finally:cap.release()
    if min(total,width,height)<1 or fps<=0:raise ValueError('Invalid source video metadata')
    first,last=time_frame(start,fps,total),time_frame(end,fps,total,end=True)
    if not 0<=first<last<=total:raise ValueError('Require 0 <= start < end <= source video duration')
    cache=base/'.cache/fingerprints';v=asset_identity(video,cache);v.update(frames=total,fps=fps,width=width,height=height)
    if expected_video_sha256 is not None:
        expected=str(expected_video_sha256).strip().lower()
        if len(expected)!=64 or any(c not in '0123456789abcdef' for c in expected):raise ValueError('Expected video SHA256 must contain 64 hexadecimal characters')
        if v['sha256']!=expected:raise ValueError('This video does not match the source video used to compute the half ranges')
    w=asset_identity(weights,cache)
    detector_settings=dict(version=DETECTOR_VERSION,model='YOLO26x',precision='FP32',full_imgsz=1536,roi_imgsz=960,roi_crop_size=640,tile_size=640,tile_overlap=.20,confidence=.10,classes=[32],rect=True,max_rois=max_rois,
        full_and_tile_on_every_frame=True,roi_policy='full-image candidate centers; image-center fallback',output_kind='ball_candidates_not_confirmed_match_ball')
    identity=dict(version=VERSION,video=v,weights=w,detector=detector_settings,batches=dict(full=full_batch,roi=roi_batch,tile=tile_batch,max_rois=max_rois),start_frame=first,end_frame=last,part_id=part_id,chunk_frames=chunk_frames,
        algorithm_sha256={name:sha(base/name) for name in ('run.py','detector.py','runtime_control.py')})
    fingerprint=hashlib.sha256(json.dumps(identity,sort_keys=True).encode()).hexdigest()
    config=dict(identity,fingerprint=fingerprint,local_paths=dict(video=str(video),weights=str(weights)),created_unix=time.time())
    with output_lock(out):
        prior=out/'run_config.json'
        if prior.exists() and read(prior)['fingerprint']!=fingerprint:raise ValueError('This output folder belongs to different settings or inputs; use another output folder')
        if not prior.exists():atomic(prior,config)
        else:config=read(prior)
        position=resume_at(out,config);resumed=position;started=time.perf_counter();last_report=time.monotonic();cap=None
        protected={p:(p.stat().st_size,p.stat().st_mtime_ns) for p in (video,weights)}
        controller=(controller_factory or RuntimeController)(gpu_index=device,target_low=gpu_target_low,target_high=gpu_target_high)
        def publish(status='running',force=False):
            nonlocal last_report
            elapsed=max(.001,time.perf_counter()-started);rate=(position-resumed)/elapsed if position>resumed else None
            value=dict(status=status,phase='ball_only_detection',part_id=part_id,start_frame=first,end_frame=last,next_source_frame=position,completed_frames=position-first,total_frames=last-first,source_total_frames=total,fps=fps,
                processing_fps=rate,eta_seconds=(last-position)/rate if rate else None,gpu=controller.snapshot(refresh=True),updated_unix=time.time(),source_frames_are_global=True)
            atomic(out/'progress.json',value)
            if force or time.monotonic()-last_report>=300:
                with (out/'progress_5min.jsonl').open('a',encoding='utf-8') as h:h.write(json.dumps(value,ensure_ascii=False)+'\n')
                print(json.dumps(value,ensure_ascii=False),flush=True);last_report=time.monotonic()
        try:
            publish(force=True)
            if position<last:
                if (out/'STOP').exists():publish('stopped',True);return False
                model=(detector_factory or BallDetector)(weights,controller,device=device)
                runtime=dict(detector=model.settings(),created_unix=time.time(),note='local hardware/path metadata is not used to match video parts')
                atomic(out/'runtime_info.json',runtime)
                cap=cv2.VideoCapture(str(video))
                if not cap.set(cv2.CAP_PROP_POS_FRAMES,position):raise RuntimeError('Cannot seek exact source checkpoint')
                started=time.perf_counter()
                while position<last:
                    if (out/'STOP').exists():publish('stopped',True);atomic(out/'job_stopped.json',dict(status='stopped',next_source_frame=position));return False
                    end_chunk=min(last,position+chunk_frames);images=[]
                    for source_frame in range(position,end_chunk):
                        ok,image=cap.read()
                        if not ok or image.shape[:2]!=(height,width):raise RuntimeError(f'Source decode failed at frame {source_frame}')
                        images.append(image)
                    results=model.run_chunk(images,full_batch=full_batch,roi_batch=roi_batch,tile_batch=tile_batch,max_rois=max_rois)
                    if len(results)!=len(images):raise ValueError('Detector output frame count mismatch')
                    folder=out/'chunks'/f'{position:09d}';folder.mkdir(parents=True,exist_ok=True);target=folder/'detections.jsonl';tmp=target.with_name(target.name+'.tmp-'+uuid.uuid4().hex);digest=hashlib.sha256()
                    with tmp.open('wb') as h:
                        for offset,result in enumerate(results):
                            frame=position+offset;row=dict(result,frame_idx=frame,source_frame=frame,part_frame=frame-first,video_seconds=frame/fps,part_id=part_id,candidates_are_ground_truth=False)
                            encoded=(json.dumps(row,ensure_ascii=False,allow_nan=False,separators=(',',':'))+'\n').encode('utf-8');h.write(encoded);digest.update(encoded)
                        h.flush();os.fsync(h.fileno())
                    os.replace(tmp,target)
                    atomic(folder/'manifest.json',dict(status='complete',fingerprint=fingerprint,start_frame=position,end_frame=end_chunk,frames=end_chunk-position,sha256=digest.hexdigest(),completed_unix=time.time()))
                    position=end_chunk;publish()
                cap.release();cap=None
            target,digest,count=consolidate(out,config)
            if any((p.stat().st_size,p.stat().st_mtime_ns)!=stamp for p,stamp in protected.items()):raise ValueError('A protected input changed while running')
            complete=dict(status='complete',start_frame=first,end_frame=last,frames=count,part_id=part_id,output_path=str(target),file_sha256=digest,source_video_sha256=v['sha256'],weights_sha256=w['sha256'],elapsed_seconds=time.perf_counter()-started,source_files_modified=False,player_detection_performed=False,ball_candidates_only=True,ground_truth_accuracy_validated=False,finished_unix=time.time())
            atomic(out/'job_complete.json',complete)
            for name in ('job_error.json','job_stopped.json'):
                previous=out/name
                if previous.exists():
                    archive=out/'history';archive.mkdir(exist_ok=True);previous.replace(archive/(str(time.time_ns())+'_'+name))
            publish('complete',True);return complete
        except BaseException as exc:
            atomic(out/'job_error.json',dict(status='failed',error=repr(exc),traceback=traceback.format_exc(),next_source_frame=position,updated_unix=time.time()));raise
        finally:
            if cap is not None:cap.release()


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--video',required=True);p.add_argument('--weights',required=True);p.add_argument('--out',required=True)
    p.add_argument('--start',required=True,help='Original video timestamp, inclusive');p.add_argument('--end',required=True,help='Original video timestamp, exclusive; EOF allowed')
    p.add_argument('--part',required=True,dest='part_id');p.add_argument('--chunk-frames',type=int,default=32)
    p.add_argument('--expected-video-sha256',help='Reject a different broadcast file before inference when using precomputed ranges')
    p.add_argument('--full-batch',type=int,default=1);p.add_argument('--roi-batch',type=int,default=4);p.add_argument('--tile-batch',type=int,default=8);p.add_argument('--max-rois',type=int,default=2)
    p.add_argument('--gpu-target-low',type=float,default=98.);p.add_argument('--gpu-target-high',type=float,default=99.5);p.add_argument('--device',type=int,default=0)
    result=run(**vars(p.parse_args()))
    if result is False:print('Stopped at a committed checkpoint. Remove STOP before resuming.',flush=True)


if __name__=='__main__':main()
