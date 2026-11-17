"""Freeze point-labelled ball/non-ball challenge cases before YOLO26 comparison.

These labels describe exact reviewed objects, not exhaustive image annotations.
No detector output from the new comparison may alter the labels or split.
"""
from pathlib import Path
import hashlib
import json

import cv2
import numpy as np
from PIL import Image,ImageDraw,ImageFont

ROOT=Path(__file__).resolve().parents[1]
OUT=ROOT/'data/derived/ball_yolo26_model_audit_20261116'
BASE=ROOT/'data/derived/ball_fp_reduction_20261116'
UNIFORM=ROOT/'data/derived/ball_uniform_context_20261116'
TARGETED=ROOT/'data/derived/ball_targeted_redetection_20261116'
VIDEO=ROOT/'data/broadcast/BAR-ATH_broadcast-001.mp4'


def records(path):
    data=json.loads(path.read_text())
    return data if isinstance(data,list) else data.get('records',data.get('labels',[]))


def digest(path):return hashlib.sha256(path.read_bytes()).hexdigest()


def main():
    OUT.mkdir(exist_ok=True,parents=True);images=OUT/'reference_images';images.mkdir(exist_ok=True)
    with np.load(BASE/'fp_inputs.npz') as z:source={k:z[k] for k in z.files}
    with np.load(BASE/'reference_evidence.npz') as z:reference={k:z[k] for k in z.files}
    original_ranker=np.load(BASE/'appearance_ranker.npz')
    training_frames=set(map(int,original_ranker['train_frames']))
    inputs=[UNIFORM/'evaluation_reference.json',BASE/'reference_evidence.json',
        BASE/'independent_review/notes.json',BASE/'independent_review/holdout_notes.json',
        TARGETED/'frame_audit/penalty_reference.json']
    lookup={}
    for i,f in enumerate(reference['source_frame']):
        lookup[(int(f),*np.round(reference['pixel_xy'][i],3))]=reference['bbox'][i]
    seen={}
    for path in inputs:
        for item in records(path):
            raw_label=item['label']
            label='ball' if raw_label in ['match_ball','match_ball_observed','real_match_ball'] else (
                'nonball' if raw_label in ['jersey_logo','not_ball','painted_penalty_mark_not_ball'] else None)
            # Uncertain, localization-uncertain, identity-uncertain and secondary
            # ball identity labels never become binary non-ball examples.
            if label is None:continue
            frame=int(item['source_frame'])
            point=np.array(item.get('pixel_xy',item.get('point',item.get('source_pixel_xy'))),float)
            key=(frame,*np.round(point,3))
            prior=seen.get(key)
            if prior:
                assert prior['label']==label,(key,prior['label'],label)
                prior['source_annotations'].append(str(path.relative_to(ROOT)))
                continue
            box=item.get('bbox_xyxy',item.get('bbox'))
            provenance='annotation bounding box'
            if box is None and key in lookup:box=lookup[key];provenance='exact reference_evidence bounding box'
            if box is None and np.allclose(source['pixel_xy'][frame],point,rtol=0,atol=1e-5):
                box=source['bbox'][frame];provenance='exact original selected candidate bounding box'
            assert box is not None,(frame,label,'missing box')
            box=np.asarray(box,float);assert np.isfinite(point).all() and np.isfinite(box).all()
            assert box.shape==(4,) and box[2]>box[0] and box[3]>box[1]
            kind=item.get('object_kind') or item.get('case_type') or ('penalty_mark' if raw_label=='painted_penalty_mark_not_ball' else (
                'jersey_logo' if raw_label=='jersey_logo' else 'real_match_football' if label=='ball' else 'other_nonball'))
            seen[key]=dict(source_frame=frame,pixel_xy=point.tolist(),bbox_xyxy=box.tolist(),label=label,
                object_kind=kind,case_type=item.get('case_type',kind),reason=item['reason'],half=int(source['half'][frame]),
                group_id=frame//500,active_ball_identity='active_match_ball' if label=='ball' else 'not_a_ball',
                scope='Exact previously reviewed object only; other objects in the image are not exhaustively labelled',
                bbox_provenance=provenance,source_annotations=[str(path.relative_to(ROOT))],
                upstream_appearance_training_overlap=frame in training_frames)
    # Fixed entire temporal blocks, chosen before any new YOLO26 inference.
    # Adjacent observations and positive/negative objects in the same block
    # necessarily share the split. These are held-out challenge labels, not a
    # statistically representative or previously unseen-video test set.
    heldout_anchor_frames=[19091,33329,45711,47422,53505,58745,98102,119500,125032,145898,150694,69850,
        21487,36592,41901,81365,99998,128172,139284,151532,163214,107348,126300,128900,23808]
    heldout_groups={f//500 for f in heldout_anchor_frames}
    data=sorted(seen.values(),key=lambda r:(r['source_frame'],r['pixel_xy']))
    for i,r in enumerate(data):
        r['reference_id']=i;r['split']='heldout' if r['group_id'] in heldout_groups else 'calibration'
        r['evidence']=f"reference_images/object_{i:03d}_f{r['source_frame']}.jpg"
    for group in {r['group_id'] for r in data}:
        assert len({r['split'] for r in data if r['group_id']==group})==1
    count=lambda group:{label:sum(r['label']==label for r in group) for label in ['ball','nonball']}
    report=dict(schema_version=1,purpose='Fixed exact-object challenge set for YOLO26 family detection and candidate classification comparison',
        source_video=str(VIDEO.relative_to(ROOT)),frame_convention='0-based original source frame, 25fps',
        labels_frozen_before_new_model_predictions=True,records=data,
        counts=dict(total=len(data),**count(data),by_split={split:count([r for r in data if r['split']==split]) for split in ['calibration','heldout']}),
        temporal_grouping='source_frame // 500; all objects and neighbouring observations within a block share split',
        limitations=['Purposive reviewed cases; not whole-match accuracy, full-image precision, recall or mAP.',
            'Images do not exhaustively label every actual or secondary football; evaluate only the specified target object.',
            'The existing upstream crop ranker used some of these labels; the official YOLO26 comparison is not trained on this local set.',
            'Heldout temporal blocks are fixed before model comparison; these video scenes have previously been viewed during earlier project QA.',
            'Source candidate boxes aid association and are not independently hand-drawn localization ground truth.',
            'Repeated YOLO detections are model support, not semantic truth or proof of active in-play ball identity.'],
        excluded_labels=['uncertain','localization_uncertain','identity_uncertain','not_match_ball'],
        recommended_evaluation='Compare sports-ball confidence for outputs containing the labelled target point; report positive target detection and non-ball false-detection separately. Do not treat unlabelled detections as false positives.',
        source_sha256={str(path.relative_to(ROOT)):digest(path) for path in inputs})
    # Freeze the numeric reference immediately; evidence images are generated
    # afterwards and may be audited independently of the model comparison.
    target=OUT/'reference.json';target.write_text(json.dumps(report,ensure_ascii=False,indent=2)+'\n')
    (OUT/'reference_sha256.txt').write_text(digest(target)+'  reference.json\n')
    print(json.dumps(report['counts'],indent=2),flush=True)
    cap=cv2.VideoCapture(str(VIDEO));assert cap.isOpened()
    pages=[]
    for r in data:
        f=r['source_frame'];cap.set(cv2.CAP_PROP_POS_FRAMES,f);ok,image=cap.read();assert ok
        px,py=r['pixel_xy'];color=(60,230,90) if r['label']=='ball' else (50,165,255)
        scene=cv2.resize(image,(960,540));cv2.circle(scene,tuple(np.rint([px/2,py/2]).astype(int)),8,color,2,cv2.LINE_AA)
        x1,y1,x2,y2=r['bbox_xyxy'];side=max(48.,(x2-x1)*2.,(y2-y1)*2.)
        left=max(0,int(px-side/2));top=max(0,int(py-side/2));right=min(1920,int(px+side/2));bottom=min(1080,int(py+side/2))
        crop=image[top:bottom,left:right];crop=cv2.resize(crop,(240,240),interpolation=cv2.INTER_CUBIC)
        canvas=np.full((600,1200,3),22,np.uint8);canvas[60:600,:960]=scene;canvas[60:300,960:]=crop
        cv2.putText(canvas,f"{r['reference_id']:03d} f{f} {r['label']} | {r['split']} | group {r['group_id']}",(15,26),0,.62,(235,235,235),1,cv2.LINE_AA)
        cv2.putText(canvas,r['object_kind'][:100],(15,49),0,.5,(210,210,210),1,cv2.LINE_AA)
        cv2.imwrite(str(OUT/r['evidence']),canvas)
        tile=np.full((200,360,3),22,np.uint8);tile[25:200,:]=cv2.resize(canvas[60:600,:960],(360,175))
        # The small crop is overlaid at right so the exact target remains legible.
        tile[80:200,240:360]=cv2.resize(crop,(120,120))
        cv2.putText(tile,f"{r['reference_id']:03d} f{f} {r['label']} {r['split'][:4]}",(3,17),0,.39,color,1,cv2.LINE_AA);pages.append(tile)
    cap.release()
    for start in range(0,len(pages),12):
        chunk=pages[start:start+12];sheet=np.full((int(np.ceil(len(chunk)/3))*200,1080,3),22,np.uint8)
        for i,tile in enumerate(chunk):sheet[i//3*200:i//3*200+200,i%3*360:i%3*360+360]=tile
        cv2.imwrite(str(images/f'sheet_{start//12+1}.jpg'),sheet)
    print('Reference and image evidence complete',flush=True)


if __name__=='__main__':main()
