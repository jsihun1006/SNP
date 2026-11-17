"""CPU-only small challenge experiment: frozen YOLO26 embeddings + binary head.

This never runs on the whole candidate set, changes official weights or edits GT.
Five-fold group OOF calibration selects a threshold before heldout evaluation.
"""
from pathlib import Path
import hashlib
import json
import os
import sys
import time

import numpy as np

ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT))
from starter.classify_ball_yolo26_candidates import square_crop

BASE=ROOT/'data/derived/ball_yolo26_model_audit_20261116'
OUT=BASE/'binary_probe'


def sha(path):return hashlib.sha256(path.read_bytes()).hexdigest()
def sigmoid(a):return 1/(1+np.exp(-np.clip(a,-40,40)))


def sample_weights(y,groups):
    unique,counts=np.unique(groups,return_counts=True)
    sizes=dict(zip(unique,counts));w=np.array([1./sizes[g] for g in groups])
    for label in [0,1]:w[y==label]*=len(y)/(2*w[y==label].sum())
    return w


def fit(features,y,groups):
    w=sample_weights(y,groups);mean=np.average(features,axis=0,weights=w)
    std=np.maximum(np.sqrt(np.average((features-mean)**2,axis=0,weights=w)),.025)
    x=np.column_stack([np.clip((features-mean)/std,-6,6),np.ones(len(y))])
    gram=(x*np.sqrt(w[:,None]))@(x*np.sqrt(w[:,None])).T/len(x)
    rate=1/(.25*float(np.linalg.eigvalsh(gram).max())+.1);coef=np.zeros(x.shape[1])
    for _ in range(1600):
        gradient=x.T@(w*(sigmoid(x@coef)-y))/len(x);gradient[:-1]+=.1*coef[:-1]
        updated=coef-rate*gradient
        if np.linalg.norm(updated-coef)<1e-8:coef=updated;break
        coef=updated
    return dict(mean=mean,std=std,coef=coef)


def predict(model,x):
    z=np.clip((x-model['mean'])/model['std'],-6,6)
    return sigmoid(z@model['coef'][:-1]+model['coef'][-1])


def metrics(y,score,threshold):
    guess=score>=threshold;y=y.astype(bool)
    tp=int((guess&y).sum());tn=int((~guess&~y).sum());fp=int((guess&~y).sum());fn=int((~guess&y).sum())
    recall=tp/max(1,tp+fn);specificity=tn/max(1,tn+fp)
    return dict(threshold=float(threshold),tp=tp,tn=tn,fp=fp,fn=fn,recall=recall,
        specificity=specificity,balanced_accuracy=(recall+specificity)/2,
        precision=tp/max(1,tp+fp))


def auc(y,s):
    positive=s[y==1];negative=s[y==0]
    return float(((positive[:,None]>negative).astype(float)+.5*(positive[:,None]==negative)).mean())


def main():
    started=time.monotonic();OUT.mkdir(exist_ok=True)
    os.environ['YOLO_CONFIG_DIR']=str(OUT/'runtime/ultralytics');os.environ['MPLCONFIGDIR']=str(OUT/'runtime/matplotlib');os.environ['YOLO_AUTOINSTALL']='false'
    import cv2
    import torch
    from ultralytics import YOLO
    torch.set_num_threads(4)
    ref_path=BASE/'reference.json';reference=json.loads(ref_path.read_text());records=reference['records']
    assert len(records)==79
    model_path=BASE/'models/yolo26s-cls.pt';before_sha=sha(model_path)
    target=OUT/'frozen_features.npz'
    signature=dict(model_sha256=before_sha,reference_sha256=sha(ref_path),scales=[1.25,2.],imgsz=224,
        embedding='1280 features immediately before frozen classification linear head',device='cpu')
    if target.exists():
        stored=np.load(target);assert json.loads(str(stored['signature']))==signature
        features=stored['features']
    else:
        model=YOLO(str(model_path));assert model.task=='classify' and model.names[805].replace('_',' ').lower()=='soccer ball'
        net=model.model.cpu().eval();net.requires_grad_(False)
        linear=net.model[-1].linear;assert linear.in_features==1280
        captured=[]
        hook=linear.register_forward_pre_hook(lambda module,args:captured.append(args[0].detach().cpu().numpy().copy()))
        cap=cv2.VideoCapture(str(ROOT/reference['source_video']));assert cap.isOpened()
        images={}
        for f in sorted({r['source_frame'] for r in records}):
            assert cap.set(cv2.CAP_PROP_POS_FRAMES,int(f));ok,image=cap.read();assert ok;images[f]=image
        cap.release()
        patches=[square_crop(images[r['source_frame']],r['bbox_xyxy'],scale) for r in records for scale in [1.25,2.]]
        with torch.inference_mode():
            for first in range(0,len(patches),16):
                batch=np.stack(patches[first:first+16])[:,:,:,::-1].transpose(0,3,1,2)
                net(torch.from_numpy(np.ascontiguousarray(batch)).float()/255.)
        hook.remove();features=np.concatenate(captured).reshape(len(records),2560)
        assert features.shape==(79,2560) and np.isfinite(features).all()
        np.savez_compressed(target,features=features,signature=json.dumps(signature))
        del net,model,images,patches
    assert sha(model_path)==before_sha
    labels=np.array([r['label']=='ball' for r in records],int)
    groups=np.array([r['group_id'] for r in records],int)
    calibration=np.array([r['split']=='calibration' for r in records]);heldout=~calibration
    assert calibration.sum()==46 and heldout.sum()==33
    x=features[calibration];y=labels[calibration];g=groups[calibration]
    unique=np.unique(g);rng=np.random.default_rng(20261116);rng.shuffle(unique)
    assignment={int(v):i%5 for i,v in enumerate(unique)};fold=np.array([assignment[int(v)] for v in g])
    oof=np.full(len(x),np.nan)
    for k in range(5):
        train=fold!=k;test=~train;assert len(np.unique(y[train]))==2
        model=fit(x[train],y[train],g[train]);oof[test]=predict(model,x[test])
    assert np.isfinite(oof).all()
    thresholds=np.unique(np.r_[0.,oof,np.nextafter(oof,np.inf),1.])
    options=[metrics(y,oof,t) for t in thresholds]
    eligible=[r for r in options if r['recall']>=.9]
    chosen=max(eligible,key=lambda r:(r['balanced_accuracy'],r['specificity'],r['threshold']))
    model=fit(x,y,g)
    # The heldout scores and metrics are computed once, after the OOF-selected
    # threshold is fixed. There is no heldout-based threshold/model search.
    heldout_score=predict(model,features[heldout])
    heldout_result=metrics(labels[heldout],heldout_score,chosen['threshold'])
    np.savez_compressed(OUT/'binary_head.npz',**model,calibration_reference_ids=np.array([r['reference_id'] for r in records])[calibration],
        grouped_oof_scores=oof,fold=fold,selected_threshold=chosen['threshold'])
    predictions=[]
    for i,r in enumerate(np.array(records,dtype=object)[calibration]):
        predictions.append(dict(reference_id=int(r['reference_id']),source_frame=r['source_frame'],split='calibration_oof',label=r['label'],score=float(oof[i]),fold=int(fold[i])))
    for i,r in enumerate(np.array(records,dtype=object)[heldout]):
        predictions.append(dict(reference_id=int(r['reference_id']),source_frame=r['source_frame'],split='heldout',label=r['label'],score=float(heldout_score[i])))
    report=dict(method='Frozen official YOLO26s-cls two-context embeddings + group/class-balanced L2 logistic binary head',
        **signature,features=2560,calibration_samples=46,heldout_samples=33,
        training_groups=len(unique),regularization=.1,grouped_folds=5,
        sample_weighting='Inverse source500-group size, followed by balanced total weight per class',
        selected_threshold=chosen['threshold'],threshold_selected_on='5fold grouped OOF calibration only; recall>=0.9 then maximum balanced accuracy',
        calibration_oof=chosen,calibration_oof_auc=auc(y,oof),heldout=heldout_result,
        heldout_auc=auc(labels[heldout],heldout_score),heldout_tuned=False,
        official_weights_changed=False,all_candidate_inference_performed=False,
        scope='79 purposive reviewed challenge objects only. Small trained head is not independently established whole-match accuracy. Scores are not calibrated probabilities.',
        elapsed_seconds=round(time.monotonic()-started,2),predictions=predictions)
    (OUT/'probe_report.json').write_text(json.dumps(report,indent=2)+'\n')
    print(json.dumps({k:v for k,v in report.items() if k!='predictions'},indent=2))


if __name__=='__main__':main()
