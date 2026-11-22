"""Train a small appearance ranker and propose, never automatically remove, observations.

The ranker uses exact visually reviewed ball/non-ball crops. Pitch coordinates,
ATD distances and homographies are not model inputs. All removals require a
separate exact-observation/interval visual review manifest.
"""
from pathlib import Path
import json
import sys

import cv2
import numpy as np
import pandas as pd

ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT))
OUT=ROOT/'data/derived/ball_fp_reduction_20261116'


def arrays(path):
    with np.load(path) as data:return {k:data[k] for k in data.files}


def sigmoid(value):return 1/(1+np.exp(-np.clip(value,-40,40)))


def fit(features,labels,regularization=.1):
    features=np.asarray(features,float);labels=np.asarray(labels,float)
    mean=features.mean(axis=0);std=np.maximum(features.std(axis=0),.025)
    x=np.clip((features-mean)/std,-6,6)
    x=np.column_stack([x,np.ones(len(x))])
    weights=np.where(labels==1,len(labels)/(2*(labels==1).sum()),len(labels)/(2*(labels==0).sum()))
    gram=(x*weights[:,None]**.5)@(x*weights[:,None]**.5).T/len(x)
    rate=1/(.25*float(np.linalg.eigvalsh(gram).max())+regularization)
    coef=np.zeros(x.shape[1])
    for _ in range(1600):
        gradient=x.T@(weights*(sigmoid(x@coef)-labels))/len(x)
        gradient[:-1]+=regularization*coef[:-1]
        updated=coef-rate*gradient
        if np.linalg.norm(updated-coef)<1e-8:coef=updated;break
        coef=updated
    return dict(mean=mean,std=std,coef=coef)


def predict(model,features):
    x=np.clip((features-model['mean'])/model['std'],-6,6)
    return sigmoid(x@model['coef'][:-1]+model['coef'][-1])


def main():
    evidence=arrays(OUT/'evidence.npz');reference=arrays(OUT/'reference_evidence.npz')
    inputs=arrays(OUT/'fp_inputs.npz')
    x=evidence['features'];frames=evidence['feature_frames'].astype(int)
    labels=reference['labels'].astype(str)
    eligible=reference['training_allowed'].astype(bool)&np.isin(labels,['match_ball_observed','not_ball'])
    chosen=[];seen=set()
    for i in np.flatnonzero(eligible):
        key=(int(reference['source_frame'][i]),*np.round(reference['pixel_xy'][i],3).tolist())
        if key not in seen:seen.add(key);chosen.append(i)
    chosen=np.array(chosen,int)
    train_x=reference['features'][chosen].astype(float)
    train_y=(labels[chosen]=='not_ball').astype(int)
    train_frames=reference['source_frame'][chosen].astype(int)
    assert len(np.unique(train_y))==2
    groups=train_frames//500
    unique=np.unique(groups);rng=np.random.default_rng(20261116);rng.shuffle(unique)
    fold_for_group={int(group):k%5 for k,group in enumerate(unique)}
    fold=np.array([fold_for_group[int(g)] for g in groups]);oof=np.full(len(chosen),np.nan)
    for k in range(5):
        take=fold!=k;test=~take
        assert len(np.unique(train_y[take]))==2
        model=fit(train_x[take],train_y[take]);oof[test]=predict(model,train_x[test])
    model=fit(train_x,train_y)
    scores=np.empty(len(frames))
    for first in range(0,len(frames),8000):scores[first:first+8000]=predict(model,x[first:first+8000])
    np.savez_compressed(OUT/'appearance_ranker.npz',**model,train_frames=train_frames,train_labels=train_y,
        reference_indices=chosen,grouped_oof_scores=oof)
    # Prototype similarity makes each proposal inspectable; it is not independent evidence.
    ref_hog=train_x[:,:324];ref_hog/=np.maximum(np.linalg.norm(ref_hog,axis=1,keepdims=True),1e-12)
    nearest=np.zeros(len(frames),int);similarity=np.zeros(len(frames))
    for first in range(0,len(frames),4000):
        z=x[first:first+4000,:324].astype(float)
        z/=np.maximum(np.linalg.norm(z,axis=1,keepdims=True),1e-12)
        sim=z@ref_hog[train_y==1].T;k=np.argmax(sim,axis=1)
        nearest[first:first+len(z)]=np.flatnonzero(train_y==1)[k]
        similarity[first:first+len(z)]=sim[np.arange(len(z)),k]
    records=pd.DataFrame(dict(source_frame=frames,half=inputs['half'][frames],nonball_score=scores,
        context_green=evidence['context_green'],bbox_aspect=evidence['boxes_aspect'],
        closest_nonball_frame=train_frames[nearest],closest_nonball_hog_similarity=similarity))
    records.to_csv(OUT/'appearance_scores.csv',index=False)
    high=scores>=.85
    # Review intervals contain only score-flagged points and never bridge source gaps.
    target=frames[high]
    starts=np.r_[0,np.flatnonzero(np.diff(target)>1)+1] if len(target) else []
    ends=np.r_[starts[1:],len(target)] if len(target) else []
    intervals=[]
    for i,(a,b) in enumerate(zip(starts,ends)):
        segment=target[a:b];ix=np.searchsorted(frames,segment)
        p=scores[ix]
        center=int(segment[np.argmax(p)])
        intervals.append(dict(id=i,start_frame=int(segment[0]),end_frame=int(segment[-1]),frames=len(segment),
            center_frame=center,score_max=float(p.max()),score_median=float(np.median(p)),
            priority=float(np.median(p)*np.sqrt(min(len(segment),100))),
            closest_nonball_frame=int(records.iloc[ix[len(ix)//2]].closest_nonball_frame)))
    intervals=sorted(intervals,key=lambda r:r['priority'],reverse=True)
    (OUT/'proposal_intervals.json').write_text(json.dumps(intervals,indent=2)+'\n')
    holdout=json.loads((OUT/'independent_review/holdout_notes.json').read_text())['labels']
    holdout_scores=[]
    for row in holdout:
        if row['label'] not in ('match_ball_observed','not_ball'):continue
        frame=int(row['source_frame']);ix=int(np.searchsorted(frames,frame))
        assert frames[ix]==frame and frame not in train_frames
        holdout_scores.append(dict(source_frame=frame,label=row['label'],nonball_score=float(scores[ix]),reason=row['reason']))
    report=dict(method='L2-regularized logistic appearance ranker; image crops only',
        training_samples=len(chosen),training_ball=int((train_y==0).sum()),training_nonball=int((train_y==1).sum()),
        features=x.shape[1],grouped_validation='5 folds grouped by 500 source frames',
        grouped_oof=[dict(source_frame=int(f),label=int(y),nonball_score=float(s)) for f,y,s in zip(train_frames,train_y,oof)],
        independent_holdout=holdout_scores,observations_scored=len(frames),proposal_threshold=.85,
        proposed_observations=int(high.sum()),proposed_intervals=len(intervals),
        scores_are_calibrated_probabilities=False,automatic_coordinate_changes=False,
        original_homography_changed=False,atd_used=False)
    (OUT/'ranker_report.json').write_text(json.dumps(report,indent=2)+'\n')
    # Compact crop contact sheets for interval review. Always use actual cached observation pixels.
    cropfile=(OUT/'observation_crops.mjpg').open('rb')
    for page in range(3):
        candidates=intervals[page*24:(page+1)*24]
        if not candidates:break
        sheet=np.full((len(candidates)//4*136+(136 if len(candidates)%4 else 0),4*240,3),22,np.uint8)
        for k,item in enumerate(candidates):
            frames_to_show=np.unique([item['start_frame'],item['center_frame'],item['end_frame']])
            for j,frame in enumerate(frames_to_show):
                idx=int(np.searchsorted(frames,frame))
                cropfile.seek(int(evidence['crop_offsets'][idx]));encoded=cropfile.read(int(evidence['crop_sizes'][idx]))
                crop=cv2.imdecode(np.frombuffer(encoded,np.uint8),cv2.IMREAD_COLOR)
                crop=cv2.resize(crop,(76,90))
                y=(k//4)*136;x0=(k%4)*240+j*78
                sheet[y+37:y+127,x0:x0+76]=crop
            label=f"{item['id']} f{item['start_frame']}-{item['end_frame']}"
            cv2.putText(sheet,label,((k%4)*240+3,(k//4)*136+14),0,.37,(240,240,240),1)
            cv2.putText(sheet,f"s={item['score_median']:.2f} n={item['frames']}",((k%4)*240+3,(k//4)*136+30),0,.36,(200,200,200),1)
        cv2.imwrite(str(OUT/f'proposal_sheet_{page+1}.jpg'),sheet)
    cropfile.close()
    print(json.dumps({k:v for k,v in report.items() if k!='grouped_oof'},indent=2))


if __name__=='__main__':main()
