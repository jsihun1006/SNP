"""Read-only image evidence for conservative match-ball false-positive reduction.

This extracts appearance and image motion, not decisions. No ATD, pitch coordinate,
or homography enters a feature. A failed motion fit remains missing evidence.
"""
from pathlib import Path
import argparse
import hashlib
import json
import time

import cv2
import numpy as np

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / 'data/derived/ball_fp_reduction_20261116'
VIDEO = ROOT / 'data/broadcast/BAR-ATH_broadcast-001.mp4'
N = 164487
MOTION_STEP = 5
MOTION_SIZE = (640,360)
CELL_IDS=(np.indices((32,32))[0]//8*4+np.indices((32,32))[1]//8).ravel()
SCALAR_NAMES = [
    'object_b_mean','object_g_mean','object_r_mean',
    'object_b_std','object_g_std','object_r_std',
    'object_h_mean','object_s_mean','object_v_mean',
    'object_h_std','object_s_std','object_v_std',
    'bbox_width_fraction','bbox_height_fraction','bbox_aspect',
    'bbox_log_area','object_green','object_white','object_dark','object_yellow',
    'context_green','context_white','context_dark','context_yellow',
    'object_edge_fraction','context_edge_fraction','object_log_laplacian_variance',
    'gray_center_ring_contrast',
]
FEATURE_NAMES = ([f'hog_{i:03d}' for i in range(324)] +
                 [f'hist_h_{i:02d}' for i in range(12)] +
                 [f'hist_s_{i:02d}' for i in range(8)] +
                 [f'hist_v_{i:02d}' for i in range(8)] + SCALAR_NAMES)


def sha(path):
    digest=hashlib.sha256()
    with Path(path).open('rb') as stream:
        for chunk in iter(lambda:stream.read(4*1024*1024),b''):
            digest.update(chunk)
    return digest.hexdigest()


def hsv_fractions(hsv):
    h,s,v=cv2.split(hsv)
    return np.array([
        np.mean((h>=30)&(h<=95)&(s>45)&(v>35)),
        np.mean((s<65)&(v>130)),
        np.mean(v<60),
        np.mean((h>=15)&(h<=40)&(s>80)&(v>90)),
    ],np.float32)


def hog32(gray):
    """Unsigned 9-bin 8x8-cell HOG with nine L2-Hys 2x2-cell blocks (324D)."""
    gray=gray.astype(np.float32)
    gx=cv2.Sobel(gray,cv2.CV_32F,1,0,ksize=1)
    gy=cv2.Sobel(gray,cv2.CV_32F,0,1,ksize=1)
    magnitude=np.hypot(gx,gy).ravel()
    orientation=((np.arctan2(gy,gx)*180/np.pi)%180/20).ravel()
    low=np.floor(orientation).astype(np.int32);fraction=orientation-low
    first=np.bincount(CELL_IDS*9+low%9,weights=magnitude*(1-fraction),minlength=144)
    second=np.bincount(CELL_IDS*9+(low+1)%9,weights=magnitude*fraction,minlength=144)
    cells=(first+second).reshape(4,4,9)
    blocks=np.array([cells[y:y+2,x:x+2].ravel() for y in range(3) for x in range(3)])
    blocks/=np.sqrt((blocks**2).sum(axis=1,keepdims=True)+1e-6)
    blocks=np.minimum(blocks,.2)
    blocks/=np.sqrt((blocks**2).sum(axis=1,keepdims=True)+1e-6)
    return blocks.ravel().astype(np.float32)


def image_crop(image,bbox,point):
    """Exact bbox patch plus a 1.8x context patch; never moves the detection."""
    bbox=np.asarray(bbox,float);point=np.asarray(point,float)
    height,width=image.shape[:2]
    if not (np.isfinite(bbox).all() and np.isfinite(point).all()):
        raise ValueError('Nonfinite observation bbox/point')
    x1,y1,x2,y2=bbox
    if x2<=x1 or y2<=y1:raise ValueError('Empty bbox')
    left=max(0,min(width-1,int(np.floor(x1))))
    top=max(0,min(height-1,int(np.floor(y1))))
    right=max(left+1,min(width,int(np.ceil(x2))))
    bottom=max(top+1,min(height,int(np.ceil(y2))))
    target=image[top:bottom,left:right]
    context_width=max(8.,(x2-x1)*1.8)
    context_height=max(8.,(y2-y1)*1.8)
    cx,cy=(x1+x2)/2,(y1+y2)/2
    c_left=max(0,min(width-1,int(np.floor(cx-context_width/2))))
    c_top=max(0,min(height-1,int(np.floor(cy-context_height/2))))
    c_right=max(c_left+1,min(width,int(np.ceil(cx+context_width/2))))
    c_bottom=max(c_top+1,min(height,int(np.ceil(cy+context_height/2))))
    context=image[c_top:c_bottom,c_left:c_right]
    target32=cv2.resize(target,(32,32),interpolation=cv2.INTER_AREA)
    context96=cv2.resize(context,(96,96),interpolation=cv2.INTER_AREA)
    return target32,context96


def appearance(image,bbox,point):
    target,context=image_crop(image,bbox,point)
    target_hsv=cv2.cvtColor(target,cv2.COLOR_BGR2HSV)
    context_hsv=cv2.cvtColor(context,cv2.COLOR_BGR2HSV)
    gray=cv2.cvtColor(target,cv2.COLOR_BGR2GRAY)
    context_gray=cv2.cvtColor(context,cv2.COLOR_BGR2GRAY)
    hog=hog32(gray)
    hist=[]
    for channel,bins,upper in [(0,12,180),(1,8,256),(2,8,256)]:
        counts=cv2.calcHist([target_hsv],[channel],None,[bins],[0,upper]).ravel()
        hist.append(counts/max(float(counts.sum()),1.))
    color=target.reshape(-1,3).astype(float)/255
    hsv=target_hsv.reshape(-1,3).astype(float)/[180,255,255]
    width,height=np.asarray(bbox,float)[2:]-np.asarray(bbox,float)[:2]
    obj_frac=hsv_fractions(target_hsv);ctx_frac=hsv_fractions(context_hsv)
    center=float(gray[8:24,8:24].mean())
    ring=np.ones(gray.shape,bool);ring[8:24,8:24]=False
    scalars=np.r_[color.mean(0),color.std(0),hsv.mean(0),hsv.std(0),
        width/1920,height/1080,np.clip(width/height,0,10),np.log1p(width*height)/15,
        obj_frac,ctx_frac,
        np.mean(cv2.Canny(gray,50,130)>0),np.mean(cv2.Canny(context_gray,50,130)>0),
        np.log1p(cv2.Laplacian(gray,cv2.CV_32F).var())/12,
        (center-float(gray[ring].mean()))/255]
    features=np.r_[hog,*hist,scalars].astype(np.float32)
    assert len(features)==len(FEATURE_NAMES) and np.isfinite(features).all()
    return features,context,float(ctx_frac[0])


def motion(old,new):
    mask=np.full(old.shape,255,np.uint8);mask[:40]=0;mask[-12:]=0
    points=cv2.goodFeaturesToTrack(old,maxCorners=600,qualityLevel=.012,minDistance=8,mask=mask,blockSize=5)
    empty=dict(valid=False,affine=np.full((2,3),np.nan),inliers=0,ratio=0.,residual=np.nan,spread=np.zeros(2))
    if points is None or len(points)<25:return empty
    current,status,_=cv2.calcOpticalFlowPyrLK(old,new,points,None,winSize=(25,25),maxLevel=3)
    if current is None:return empty
    back,back_status,_=cv2.calcOpticalFlowPyrLK(new,old,current,None,winSize=(25,25),maxLevel=3)
    if back is None:return empty
    good=(status.ravel()>0)&(back_status.ravel()>0)&(np.linalg.norm(back-points,axis=2).ravel()<1.2)
    a,b=points[good,0],current[good,0]
    if len(a)<25:return empty
    inside=(b[:,0]>=0)&(b[:,0]<640)&(b[:,1]>=0)&(b[:,1]<360)
    a,b=a[inside],b[inside]
    if len(a)<25:return empty
    affine,inlier=cv2.estimateAffinePartial2D(a,b,method=cv2.RANSAC,ransacReprojThreshold=1.8,
        maxIters=1500,confidence=.995,refineIters=10)
    if affine is None or inlier is None:return empty
    keep=inlier.ravel().astype(bool);n=int(keep.sum());ratio=float(keep.mean())
    spread=np.ptp(a[keep],axis=0)/[640,360] if n else np.zeros(2)
    residual=float(np.median(np.linalg.norm(np.column_stack([a[keep],np.ones(n)])@affine.T-b[keep],axis=1))) if n else np.nan
    valid=n>=25 and ratio>=.55 and spread[0]>=.20 and spread[1]>=.12 and residual<=1.3
    # Conversion from the 640x360 motion fit to original 1920x1080 image pixels.
    full=affine.copy();full[:,2]*=3
    return dict(valid=bool(valid),affine=full if valid else np.full((2,3),np.nan),
        inliers=n,ratio=ratio,residual=residual,spread=spread)


def references(inputs):
    rows=json.loads((OUT/'sources/reference_samples.json').read_text())
    independent=OUT/'independent_review/notes.json'
    if independent.exists():
        extra=json.loads(independent.read_text())
        rows+=extra.get('labels',[]) if isinstance(extra,dict) else extra
    result=[]
    for i,row in enumerate(rows):
        row=dict(row);f=int(row['source_frame'])
        point=row.get('pixel_xy',row.get('point'))
        if point is None:point=inputs['pixel_xy'][f].tolist()
        bbox=row.get('bbox')
        if bbox is None or not np.isfinite(bbox).all():bbox=inputs['bbox'][f].tolist()
        if not np.isfinite(bbox).all():
            # No fabricated box is made for a sample lacking its source detection bbox.
            row.update(reference_id=i,excluded_reason='reference_bbox_unavailable')
            result.append(row);continue
        row.update(reference_id=i,pixel_xy=list(map(float,point)),bbox=list(map(float,bbox)),
            training_allowed=bool(row.get('training_allowed',row['label'] in ['match_ball_observed','not_ball'])))
        result.append(row)
    return result


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--out',type=Path,default=OUT)
    args=parser.parse_args()
    if args.out.resolve()!=OUT.resolve():raise ValueError('Use task-owned output folder')
    cv2.setNumThreads(4)
    with np.load(OUT/'fp_inputs.npz') as archive:
        inputs={k:archive[k] for k in ['source_frame','pixel_xy','bbox','confidence','half','status_codes']}
    assert np.array_equal(inputs['source_frame'],np.arange(N))
    observed=(inputs['status_codes']==3)&np.isfinite(inputs['pixel_xy']).all(axis=1)
    frames=np.flatnonzero(observed);assert len(frames)==106689
    assert np.isfinite(inputs['bbox'][frames]).all()
    index=np.full(N,-1,int);index[frames]=np.arange(len(frames))
    features=np.empty((len(frames),len(FEATURE_NAMES)),np.float32)
    greens=np.empty(len(frames),np.float32)
    offsets=np.zeros(len(frames),np.int64);sizes=np.zeros(len(frames),np.int32)
    refs=references(inputs);ref_by_frame={}
    for r in refs:
        if 'excluded_reason' not in r:ref_by_frame.setdefault(int(r['source_frame']),[]).append(r)
    ref_features={};ref_greens={}
    sample=np.unique(np.r_[np.arange(0,N,MOTION_STEP),N-1])
    motion_index=np.full(N,-1,int);motion_index[sample]=np.arange(len(sample))
    affines=np.full((len(sample),2,3),np.nan);valid=np.zeros(len(sample),bool)
    ratios=np.zeros(len(sample),np.float32);inliers=np.zeros(len(sample),np.int32)
    residuals=np.full(len(sample),np.nan,np.float32);spreads=np.zeros((len(sample),2),np.float32)
    cut_scores=np.full(len(sample),np.nan,np.float32)
    crop_folder=OUT/'reference_crops';crop_folder.mkdir(exist_ok=True)
    cap=cv2.VideoCapture(str(VIDEO));assert cap.isOpened()
    assert int(cap.get(cv2.CAP_PROP_FRAME_COUNT))==N
    old_gray=old_hist=None;start=time.monotonic();processed=0
    with (OUT/'observation_crops.mjpg').open('wb') as cache:
        for f in range(N):
            assert cap.grab(),f
            j=int(index[f]);m=int(motion_index[f]);need_refs=f in ref_by_frame
            if j<0 and m<0 and not need_refs:continue
            ok,image=cap.retrieve();assert ok and image.shape==(1080,1920,3)
            if j>=0:
                vector,crop,green=appearance(image,inputs['bbox'][f],inputs['pixel_xy'][f])
                features[j]=vector;greens[j]=green
                ok,jpg=cv2.imencode('.jpg',crop,[cv2.IMWRITE_JPEG_QUALITY,80]);assert ok
                offsets[j],sizes[j]=cache.tell(),len(jpg);cache.write(jpg.tobytes());processed+=1
            if need_refs:
                for row in ref_by_frame[f]:
                    vector,crop,green=appearance(image,row['bbox'],row['pixel_xy'])
                    rid=int(row['reference_id']);ref_features[rid]=vector;ref_greens[rid]=green
                    name=f"reference_{rid:03d}_f{f}_{row['label']}.jpg"
                    cv2.imwrite(str(crop_folder/name),crop,[cv2.IMWRITE_JPEG_QUALITY,95])
                    row['crop_path']='reference_crops/'+name
            if m>=0:
                small=cv2.resize(image,MOTION_SIZE,interpolation=cv2.INTER_AREA)
                gray=cv2.cvtColor(small,cv2.COLOR_BGR2GRAY)
                hsv=cv2.cvtColor(small,cv2.COLOR_BGR2HSV)
                hist=cv2.calcHist([hsv],[0,1],None,[16,8],[0,180,0,256]);cv2.normalize(hist,hist,norm_type=cv2.NORM_L1)
                if old_gray is not None:
                    cut_scores[m]=float(cv2.compareHist(old_hist,hist,cv2.HISTCMP_BHATTACHARYYA))
                    measurement=motion(old_gray,gray)
                    affines[m]=measurement['affine'];valid[m]=measurement['valid']
                    ratios[m]=measurement['ratio'];inliers[m]=measurement['inliers']
                    residuals[m]=measurement['residual'];spreads[m]=measurement['spread']
                old_gray,old_hist=gray,hist
            if f%5000==0 or f==N-1:
                elapsed=time.monotonic()-start
                progress=dict(source_frame=f,total_video_frames=N,observed_features=processed,
                    total_observed_features=len(frames),elapsed_seconds=round(elapsed,2),
                    video_frames_per_second=round((f+1)/elapsed,2),
                    remaining_seconds=round((N-f-1)/max((f+1)/elapsed,1),1))
                (OUT/'evidence_progress.json').write_text(json.dumps(progress,indent=2)+'\n')
                print(json.dumps(progress),flush=True)
    cap.release();assert processed==len(frames)
    assert np.isfinite(features).all()
    boxes=inputs['bbox'][frames];box_sizes=boxes[:,2:]-boxes[:,:2]
    np.savez_compressed(OUT/'evidence.npz',features=features,feature_frames=frames.astype(np.int32),
        context_green=greens,boxes_aspect=(box_sizes[:,0]/box_sizes[:,1]).astype(np.float32),
        bbox_size=box_sizes.astype(np.float32),crop_offsets=offsets,crop_sizes=sizes,
        motion_frames=sample.astype(np.int32),motion_affines=affines,motion_valid=valid,
        motion_inlier_ratio=ratios,motion_inliers=inliers,motion_median_residual_small_px=residuals,
        motion_inlier_spread=spreads,cut_color_distance=cut_scores)
    ids=sorted(ref_features)
    included=[refs[i] for i in ids]
    np.savez_compressed(OUT/'reference_evidence.npz',features=np.stack([ref_features[i] for i in ids]),
        reference_id=np.asarray(ids,np.int32),source_frame=np.array([r['source_frame'] for r in included],np.int32),
        labels=np.array([r['label'] for r in included]),object_kind=np.array([r.get('object_kind','') for r in included]),
        training_allowed=np.array([r['training_allowed'] for r in included],bool),
        pixel_xy=np.array([r['pixel_xy'] for r in included]),bbox=np.array([r['bbox'] for r in included]),
        context_green=np.array([ref_greens[i] for i in ids],np.float32))
    (OUT/'reference_evidence.json').write_text(json.dumps(refs,ensure_ascii=False,indent=2,allow_nan=False)+'\n')
    summary=dict(video_frames=N,observed_feature_rows=len(frames),feature_dimension=len(FEATURE_NAMES),
        feature_names=FEATURE_NAMES,reference_rows=len(ids),excluded_reference_rows=len(refs)-len(ids),
        appearance_definition='Tight 32x32 unsigned 9-bin 8x8-cell HOG, nine L2-Hys 2x2 blocks (324); HSV marginal histograms (28); image color/context/shape scalars (28)',
        context_crop_definition='1.8 times the unchanged source bbox, minimum 8 source pixels per dimension; clipped only to source image bounds, then resized to 96x96',
        observations_crop_file='observation_crops.mjpg',observation_crop_shape=[96,96,3],crop_codec='JPEG quality 80',
        motion_sample_step_frames=MOTION_STEP,motion_samples=len(sample),motion_valid_pairs=int(valid.sum()),
        motion_affine_units='original 1920x1080 pixels; motion_affines[i] maps previous sampled image to motion_frames[i]',
        motion_method='KLT forward-backward check and RANSAC partial affine; failure is NaN, never identity',
        cut_color_distance='Bhattacharyya distance between sampled HSV histograms; a cue, not a verified cut label',
        features_exclude=['ATD','homography','pitch coordinates','inside_pitch','distance to ATD','confidence'],
        input_sha256=sha(OUT/'fp_inputs.npz'),elapsed_seconds=round(time.monotonic()-start,2))
    (OUT/'evidence_schema.json').write_text(json.dumps(summary,indent=2)+'\n')
    print(json.dumps({k:v for k,v in summary.items() if k!='feature_names'}),flush=True)


if __name__=='__main__':main()
