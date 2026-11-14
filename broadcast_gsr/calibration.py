"""Optional learned calibration. Source model weights are required."""
import importlib.util
import sys
from pathlib import Path
from types import SimpleNamespace
import cv2
import numpy as np
from scipy.optimize import least_squares
from .geometry import template_to_pitch_h
from .io import new_run,json_write,csv_write,signature
from .upstream import load_file,repository_state


def estimate_homography(source,line_pairs,line_confidences,conic,circle_points,point_pairs):
    """Fix empty-point stacking and LM's 8-variable/4-scalar-residual failure.

    Matrix maps normalized image -> normalized template, matching the actual
    upstream usage. Original line/conic residuals are retained. Point residuals
    have two components per correspondence, rather than one scalar norm.
    """
    image=np.array([p for p,_,_ in point_pairs],dtype=float).reshape(-1,3)
    target=np.array([t for _,t,_ in point_pairs],dtype=float).reshape(-1,3)
    conf=np.array([c for _,_,c in point_pairs],dtype=float)
    initial=np.eye(3)
    if len(image)>=4:
        candidate,_=cv2.findHomography(image[:,:2],target[:,:2],cv2.RANSAC,10)
        if candidate is not None:initial=candidate/candidate[2,2]
    def residual(x):
        if len(line_pairs)>=5:
            return source.get_inv_residual_line_circle_points_middle_pair(x,line_pairs,line_confidences,
                conic,circle_points,image,target,conf)
        base=source.get_inv_residual_line_circle_points_middle_pair(x,[],[],conic,circle_points,
            np.empty((0,3)),np.empty((0,3)),np.empty(0))
        if len(image):
            h=np.append(x,1.).reshape(3,3)
            p=(h@image.T).T
            if np.any(abs(p[:,2])<1e-12):return np.full(len(base)+2*len(image),1e6)
            point=((p[:,:2]/p[:,2:]-target[:,:2]/target[:,2:])*conf[:,None]).ravel()
            return np.r_[base,point]
        return base
    x0=initial.ravel()[:8]
    if len(residual(x0))<8:
        raise ValueError('Insufficient independent homography residuals')
    fitted=least_squares(residual,x0,method='lm',max_nfev=400)
    matrix=np.append(fitted.x,1.).reshape(3,3)
    if not fitted.success or not np.isfinite(matrix).all() or abs(np.linalg.det(matrix))<1e-12:
        raise ValueError('Homography fit failed or is singular')
    return matrix


def calibrate(cfg,output=None):
    from .models import device_of
    root=Path(cfg['upstream']);repository_state(root)
    if not Path(cfg['sfr_weights']).is_file():raise FileNotFoundError(cfg['sfr_weights'])
    cap=cv2.VideoCapture(cfg['video'])
    if not cap.isOpened():raise ValueError('Cannot open video')
    fps=cap.get(cv2.CAP_PROP_FPS);total=int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    start=cfg.get('start_frame',0)
    if start>=total:cap.release();raise ValueError('start_frame outside video')
    out=new_run(cfg,output,prefix='calibration')
    status=dict(status='running',model_inference_executed=False,performance_validated=False,
                method='upstream SFR with empty-point/LM adapter',video=signature(cfg['video']),
                weights=signature(cfg['sfr_weights'],True),source_frame_indices='zero_based_original_video')
    json_write(out/'manifest.json',status)
    frames=out/'img1';frames.mkdir()
    mapping=[]
    try:
        if start and not cap.set(cv2.CAP_PROP_POS_FRAMES,start):raise ValueError('Seek failed')
        for local,index in enumerate(range(start,min(total,start+cfg['max_frames'])),1):
            ok,frame=cap.read()
            if not ok:raise ValueError('Video decode ended early')
            if not cv2.imwrite(str(frames/f'{local:06d}.jpg'),frame):raise ValueError('JPEG write failed')
            mapping.append(dict(local_frame=local,frame_idx=index,timestamp_sec=index/fps))
        csv_write(out/'frame_map.csv',mapping,['local_frame','frame_idx','timestamp_sec'])
        sys.dont_write_bytecode=True
        source=load_file('_b2p_sfr',root/'kpts.py')
        # The complete SFR checkpoint already includes the encoder parameters.
        source.EfficientNet_V2_S_Weights=SimpleNamespace(DEFAULT=None)
        original_config=source.Configuration
        device=str(device_of(cfg.get('device','auto')))
        source.Configuration=lambda:original_config(num_workers=0,device=device)
        source.estimate_inv_homography_line_circle_points_middle_pair=lambda *a:estimate_homography(source,*a)
        np.random.seed(0);source.torch.manual_seed(0)
        source.predict(str(frames),str(out/'visualization'),checkpoints=cfg['sfr_weights'],
            template_image=str(root/'template/Radar_Dimen.png'),template_npy=str(root/'template/soccernet_template_97.npy'),
            save_viz=False,verbose=True)
        rows=[]
        cols=['h%d%d'%(i,j) for i in range(3) for j in range(3)]
        for item in mapping:
            path=frames/f'{item["local_frame"]:06d}.npy'
            h=np.load(path) if path.exists() else np.eye(3)
            valid=h.shape==(3,3) and np.isfinite(h).all() and not np.allclose(h,np.eye(3))
            try:
                converted=template_to_pitch_h(h) if valid else np.full((3,3),np.nan)
                valid=bool(valid and np.isfinite(converted).all() and np.linalg.cond(converted)<1e12)
            except np.linalg.LinAlgError:
                valid=False;converted=np.full((3,3),np.nan)
            row=dict(frame=item['frame_idx'],valid=int(valid),reprojection_error='',
                     status='estimated_unvalidated' if valid else 'failed_or_identity')
            row.update({name:float(v) if valid else '' for name,v in zip(cols,converted.ravel())})
            rows.append(row)
        csv_write(out/'calibration.csv',rows,['frame']+cols+['reprojection_error','valid','status'])
        status.update(status='complete',model_inference_executed=True,frames=len(rows),valid_frames=sum(r['valid'] for r in rows),
                      output_direction='normalized_pitch_to_original_image_pixels',calibration_accuracy_validated=False)
    except BaseException as exc:
        status.update(status='failed',error_type=type(exc).__name__,error=str(exc));raise
    finally:
        cap.release();json_write(out/'manifest.json',status)
    print('Saved calibration:',out/'calibration.csv')
    return out
