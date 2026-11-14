import csv
import json
import platform
import time
from pathlib import Path
import cv2
import numpy as np
from .geometry import Calibration
from .io import new_run, json_write, signature
from .upstream import Tracker, repository_state
from . import VERSION

FIELDS = ['match_id','source','frame_idx','timestamp_sec','camera_segment_id','scene_type',
          'track_id','track_id_raw','role','jersey','color','team_raw','team_ha',
          'bbox_x','bbox_y','bbox_w','bbox_h','image_x','image_y','pitch_x_norm','pitch_y_norm',
          'pitch_x_m','pitch_y_m','detection_score','role_score','color_similarity',
          'track_observed','homography_valid','calibration_quality','team_confidence','id_confidence',
          'feature_index','quality_flags']
FRAME_FIELDS = ['frame_idx','timestamp_sec','camera_segment_id','scene_type','camera_cut',
                'cut_histogram_distance','detections','observed_tracks','homography_valid','quality_flags']


class Replay:
    def __init__(self,path):
        self.frames = {}
        self.dim = None
        with Path(path).open(encoding='utf-8-sig') as handle:
            for line in handle:
                r = json.loads(line)
                key = int(r['frame_idx'])
                if key<0 or key in self.frames:
                    raise ValueError('Replay frame indices must be unique and nonnegative')
                ds = np.asarray(r['detections'],dtype=float).reshape(-1,5)
                fs = np.asarray(r['features'],dtype=float)
                if len(ds):
                    if fs.ndim!=2 or len(fs)!=len(ds):
                        raise ValueError('Replay needs one feature per detection')
                    self.dim = self.dim or fs.shape[1]
                    if fs.shape[1]!=self.dim or not np.isfinite(fs).all() or np.any(np.linalg.norm(fs,axis=1)<1e-12):
                        raise ValueError('Replay features must be finite, nonzero, and have fixed width')
                    fs /= np.linalg.norm(fs,axis=1,keepdims=True)
                if not np.isfinite(ds).all() or np.any((ds[:,4]<0)|(ds[:,4]>1)) or np.any(ds[:,2:4]<=ds[:,:2]):
                    raise ValueError('Replay detections require valid xyxy boxes and scores in [0,1]')
                attrs = r.get('attributes',[dict(role='person',jersey=None,color=None) for _ in ds])
                if len(attrs)!=len(ds):
                    raise ValueError('Replay attributes and detections differ in length')
                self.frames[key]=(ds,fs,attrs)
        self.dim = self.dim or 512

    def infer(self,frame,index):
        if index not in self.frames or not len(self.frames[index][0]):
            return np.empty((0,5)),np.empty((0,self.dim)),[]
        return self.frames[index]


def histogram(frame):
    small = cv2.resize(frame,(96,54))
    hsv = cv2.cvtColor(small,cv2.COLOR_BGR2HSV)
    h = cv2.calcHist([hsv],[0,1],None,[32,32],[0,180,0,256])
    return cv2.normalize(h,None).flatten()


def overlay(frame,rows,index,segment):
    h,w = frame.shape[:2]
    ratio = min(960/w,540/h)
    left = cv2.resize(frame,(round(w*ratio),round(h*ratio)))
    panel = np.full((left.shape[0],360,3),(40,90,40),dtype=np.uint8)
    for r in rows:
        x,y,bw,bh = [r[k] for k in ('bbox_x','bbox_y','bbox_w','bbox_h')]
        color = (0,210,255) if r['team_ha']!='A' else (230,160,70)
        cv2.rectangle(left,(round(x*ratio),round(y*ratio)),(round((x+bw)*ratio),round((y+bh)*ratio)),color,1)
        cv2.putText(left,r['track_id'],(round(x*ratio),max(12,round(y*ratio))),cv2.FONT_HERSHEY_SIMPLEX,.35,color,1)
        if r['homography_valid'] and r['pitch_x_norm']!='' and 0<=r['pitch_x_norm']<=1 and 0<=r['pitch_y_norm']<=1:
            px = int(10+340*r['pitch_x_norm']);py=int(35+(len(panel)-50)*r['pitch_y_norm'])
            cv2.circle(panel,(px,py),4,color,-1)
            cv2.putText(panel,r['track_id'],(px+4,py),cv2.FONT_HERSHEY_SIMPLEX,.3,color,1)
    cv2.rectangle(panel,(10,35),(350,len(panel)-15),(230,230,230),1)
    cv2.putText(panel,f'frame={index} segment={segment}',(8,20),cv2.FONT_HERSHEY_SIMPLEX,.4,(255,255,255),1)
    return np.concatenate([left,panel],axis=1)


def run(cfg,output=None,backend='official'):
    if output is not None and Path(output).exists():
        raise FileExistsError('Output already exists; choose a new directory: '+str(output))
    for key in ('video','calibration_csv'):
        if not Path(cfg[key]).is_file():
            raise FileNotFoundError(cfg[key])
    if cfg.get('refine') and cfg['max_frames']>3000:
        raise ValueError('Offline pilot refinement is limited to 3000 frames per run')
    upstream = repository_state(cfg['upstream'])
    cap = cv2.VideoCapture(cfg['video'])
    if not cap.isOpened():
        raise ValueError('Cannot open video: '+cfg['video'])
    fps = cap.get(cv2.CAP_PROP_FPS)
    if not np.isfinite(fps) or fps<=0:
        cap.release();raise ValueError('Video FPS is invalid')
    total = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    start = int(cfg.get('start_frame',0))
    if start>=total:
        cap.release();raise ValueError('start_frame is outside the video metadata range')
    try:
        calibration = Calibration(cfg['calibration_csv'],cfg.get('reprojection_error_max'))
        if backend=='replay':
            models = Replay(cfg['replay'])
        else:
            from .models import VisionModels
            models = VisionModels(cfg)
        tracker = Tracker(cfg['upstream'],fps)
    except Exception:
        cap.release();raise
    out = new_run(cfg,output)
    manifest = dict(version=VERSION,status='running',backend=backend,source='broadcast_gsr',
                    platform=platform.platform(),windows_process_executed=platform.system()=='Windows',
                    performance_validated=False,upstream=upstream,video=signature(cfg['video']),
                    calibration=signature(cfg['calibration_csv'],True),fps=fps,
                    video_metadata_frames=total,start_frame=start,model_inference_executed=False,
                    variant='upstream Deep_EIoU + NumPy inclusive IoU + SciPy LAP; last-bbox association',
                    temporal_alignment_to_atd_verified=False,team_ha_mapping='explicit color map only',
                    video_calibration_alignment_verified=bool(cfg.get('video_calibration_alignment_verified',False)),
                    calibration_independence=cfg.get('calibration_origin','provided; independence unverified'))
    if backend=='official':
        manifest['weights'] = {k:signature(cfg[k],True) for k in ('detector_weights','reid_weights')}
        if cfg.get('jersey')=='clip':
            manifest['weights']['clip_weights']=signature(cfg['clip_weights'],True)
    json_write(out/'manifest.json',manifest)
    track_handle = frame_handle = feat_handle = writer = None
    saved_rows = []
    count = observations = valid_frames = feature_count = 0
    feature_dim = None
    segment = 0
    previous_hist = None
    seen = set();previous_positions = {}
    started = time.perf_counter()
    try:
        if start and not cap.set(cv2.CAP_PROP_POS_FRAMES,start):
            raise ValueError('Decoder refused the requested seek')
        track_handle = (out/'tracks.csv').open('w',newline='',encoding='utf-8')
        frame_handle = (out/'frames.csv').open('w',newline='',encoding='utf-8')
        tw = csv.DictWriter(track_handle,fieldnames=FIELDS);tw.writeheader()
        fw = csv.DictWriter(frame_handle,fieldnames=FRAME_FIELDS);fw.writeheader()
        if cfg.get('refine'):
            feat_handle = (out/'features.float32').open('wb')
        for index in range(start,min(total,start+int(cfg['max_frames']))):
            ok,frame = cap.read()
            if not ok:
                raise ValueError(f'Decode ended before expected frame {index}; run is incomplete')
            reported = cap.get(cv2.CAP_PROP_POS_FRAMES)
            if reported and abs(reported-(index+1))>.5:
                raise ValueError('Decoder frame counter differs from requested timeline')
            current_hist = histogram(frame)
            distance = float(cv2.compareHist(previous_hist,current_hist,cv2.HISTCMP_BHATTACHARYYA)) if previous_hist is not None else 0.
            cut = previous_hist is not None and (index in cfg.get('cut_frames',[]) or distance>cfg.get('cut_threshold',.65))
            if cut:
                segment+=1;tracker.reset();previous_positions={}
            previous_hist=current_hist
            detections,features,attrs = models.infer(frame,index) if backend=='replay' else models.infer(frame)
            if backend=='official':
                manifest['model_inference_executed']=True
            targets = tracker.update(detections,features)
            matrix,cal_error,cal_flag = calibration.get(index)
            valid_frames+=int(matrix is not None)
            frame_flags = [cal_flag] if cal_flag else []
            if cfg.get('calibration_origin')!='learned_sfr' and not cfg.get('video_calibration_alignment_verified',False):
                frame_flags.append('video_calibration_alignment_unverified')
            if cfg.get('scene_type','unverified')!='live':
                frame_flags.append('scene_not_verified_live')
            if cut:
                frame_flags.append('camera_cut_heuristic_or_explicit')
            rows=[]
            for tid,det_index,box in targets:
                x,y,w,h = map(float,box)
                if w*h<=10:
                    continue
                attr=attrs[det_index]
                norm,err,flag = calibration.feet(index,box)
                flags=frame_flags.copy()
                if flag and flag not in flags:flags.append(flag)
                xy = norm[2]
                projected = not flag and np.isfinite(xy).all()
                if projected and np.any((xy<0)|(xy>1)):
                    flags.append('outside_pitch')
                color=attr.get('color')
                color=color.lower().strip() if color else ''
                team=cfg.get('team_colors',{}).get(color,'')
                if not team:flags.append('team_ha_unknown')
                key=f's{segment}:t{tid}'
                point=xy*np.array([cfg['pitch_length_m'],cfg['pitch_width_m']]) if projected else None
                if point is not None and key in previous_positions:
                    old_idx,old_point=previous_positions[key]
                    speed=float(np.linalg.norm(point-old_point)*fps/(index-old_idx))
                    if speed>12.:flags.append('position_jump_diagnostic')
                if point is not None:previous_positions[key]=(index,point)
                row=dict.fromkeys(FIELDS,'')
                row.update(match_id=cfg.get('match_id','unknown'),source='broadcast_gsr',frame_idx=index,
                    timestamp_sec=index/fps,camera_segment_id=segment,scene_type=cfg.get('scene_type','unverified'),
                    track_id=key,track_id_raw=key,role=attr.get('role','person'),jersey=attr.get('jersey') if attr.get('jersey') is not None else '',
                    color=color,team_raw=('color:'+color) if color else '',team_ha=team,
                    bbox_x=x,bbox_y=y,bbox_w=w,bbox_h=h,image_x=x+w/2,image_y=y+h,
                    pitch_x_norm=float(xy[0]) if projected else '',pitch_y_norm=float(xy[1]) if projected else '',
                    pitch_x_m=float(point[0]) if projected else '',pitch_y_m=float(point[1]) if projected else '',
                    detection_score=float(detections[det_index,4]),role_score=attr.get('role_score',''),
                    color_similarity=attr.get('color_similarity',''),track_observed=1,homography_valid=int(projected),
                    calibration_quality=err if err is not None else '',quality_flags=';'.join(flags))
                if feat_handle:
                    f=np.asarray(features[det_index],dtype='<f4')
                    feature_dim=feature_dim or len(f)
                    if len(f)!=feature_dim:raise ValueError('Embedding dimension changed during the run')
                    feat_handle.write(f.tobytes());row['feature_index']=feature_count;feature_count+=1
                    saved_rows.append(row.copy())
                rows.append(row);seen.add(key)
            tw.writerows(rows)
            fw.writerow(dict(frame_idx=index,timestamp_sec=index/fps,camera_segment_id=segment,
                scene_type=cfg.get('scene_type','unverified'),camera_cut=int(cut),cut_histogram_distance=distance,
                detections=len(detections),observed_tracks=len(rows),homography_valid=int(matrix is not None),quality_flags=';'.join(frame_flags)))
            if cfg.get('write_overlay',True):
                vis=overlay(frame,rows,index,segment)
                if writer is None:
                    writer=cv2.VideoWriter(str(out/'overlay.mp4'),cv2.VideoWriter_fourcc(*'mp4v'),fps,(vis.shape[1],vis.shape[0]))
                    if not writer.isOpened():raise ValueError('mp4v encoder unavailable; rerun with --no-overlay in a new output directory')
                writer.write(vis)
            count+=1;observations+=len(rows)
            if count%50==0:print(f'Processed {count} frames; observed tracks={len(rows)}',flush=True)
        if feat_handle:
            feat_handle.close();feat_handle=None
            json_write(out/'features.json',dict(dtype='<f4',count=feature_count,dimension=feature_dim or 0))
            from .refinement import refine
            refine(saved_rows,out,cfg,fps,FIELDS)
        elapsed=time.perf_counter()-started
        manifest.update(status='complete',processed_frames=count,observations=observations,unique_online_tracks=len(seen),
            homography_valid_frames=valid_frames,camera_segments=segment+1,elapsed_seconds=elapsed,
            processing_fps=count/elapsed if elapsed else None,
            note='Measured processing speed for this run only; no accuracy benchmark or possession prediction')
    except BaseException as exc:
        manifest.update(status='failed',processed_frames=count,error_type=type(exc).__name__,error=str(exc))
        raise
    finally:
        cap.release()
        for handle in (track_handle,frame_handle,feat_handle):
            if handle is not None:handle.close()
        if writer is not None:writer.release()
        json_write(out/'manifest.json',manifest)
    print('Saved:',out)
    return out
