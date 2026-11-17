"""Combine object-local YOLO26x evidence and a conservative frozen-feature verifier.

These are candidate-level model verdicts. They do not edit the original ball
trajectory, identify the active match ball, or infer missing coordinates.
"""
from pathlib import Path
import argparse
import hashlib
import json
import sys

import numpy as np
import pandas as pd

ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT))
from starter.benchmark_ball_yolo26 import object_scores

OUT=ROOT/'data/derived/ball_yolo26_model_audit_20261116'
OLD=ROOT/'data/derived/ball_targeted_redetection_20261116'
KEY=['source_frame','candidate_index']
BOX=['bbox_x1','bbox_y1','bbox_x2','bbox_y2']


def read(path):return pd.read_csv(path,float_precision='round_trip',low_memory=False)
def sha(path):return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def x_boxes():
    metadata=json.loads((OUT/'yolo26x_recheck/run_metadata.json').read_text())
    assert metadata['status']=='complete' and metadata['completed_frames']==11092
    table=read(OUT/'yolo26x_recheck/raw_ball_candidates.csv')
    groups={int(f):g.reset_index(drop=True) for f,g in table.groupby('source_frame',sort=False)}
    return table,groups


def prepare_new():
    old=read(OLD/'yolo26/raw_ball_candidates.csv');x,groups=x_boxes()
    old_groups={int(f):g[BOX+['confidence','class_id']].to_numpy(float) for f,g in old.groupby('source_frame',sort=False)}
    new=[]
    for row in x.itertuples():
        if row.confidence<.3:continue
        bbox=[getattr(row,k) for k in BOX]
        evidence=object_scores(old_groups.get(int(row.source_frame),np.empty((0,6))),
            dict(point=[row.image_x,row.image_y],bbox_xyxy=bbox))
        if evidence['local_score']>0:continue
        new.append(dict(source_frame=int(row.source_frame),candidate_index=100000+int(row.candidate_index),
            **dict(zip(BOX,bbox)),image_x=row.image_x,image_y=row.image_y,confidence=row.confidence,
            class_id=32,source='YOLO26x_NEW_CANDIDATE',x_candidate_index=int(row.candidate_index)))
    columns=[*KEY,*BOX,'image_x','image_y','confidence','class_id','source','x_candidate_index']
    table=pd.DataFrame(new,columns=columns).sort_values(KEY).reset_index(drop=True)
    assert not table.duplicated(KEY).any()
    table.to_csv(OUT/'new_x_candidates.csv',index=False)
    print('New YOLO26x candidates without an old matching candidate:',len(table))


def merge_scores(source,score_path,origin):
    scores=read(score_path)
    a=source.sort_values(KEY).reset_index(drop=True);b=scores.sort_values(KEY).reset_index(drop=True)
    assert np.array_equal(a[KEY],b[KEY])
    np.testing.assert_allclose(a[BOX],b[BOX],atol=0,rtol=0)
    a['binary_ball_score']=b.binary_ball_score
    a['candidate_origin']=origin
    return a


def decide():
    old=read(OLD/'yolo26/raw_ball_candidates.csv');assert len(old)==33744
    old=merge_scores(old,OUT/'binary_candidates/scores.csv','original_s')
    new=read(OUT/'new_x_candidates.csv')
    if len(new):new=merge_scores(new,OUT/'binary_new_candidates/scores.csv','new_x')
    else:new['binary_ball_score']=np.empty(0);new['candidate_origin']=np.empty(0,object)
    combined=pd.concat([old,new],ignore_index=True).sort_values(KEY).reset_index(drop=True)
    assert not combined.duplicated(KEY).any()
    raw_x,groups=x_boxes();scores=[];indices=[]
    for row in combined.itertuples():
        g=groups.get(int(row.source_frame));bbox=[getattr(row,k) for k in BOX]
        boxes=np.empty((0,6)) if g is None else g[BOX+['confidence','class_id']].to_numpy(float)
        evidence=object_scores(boxes,dict(point=[row.image_x,row.image_y],bbox_xyxy=bbox))
        scores.append(evidence['local_score'])
        indices.append(-1 if evidence['matched_index'] is None else int(g.iloc[evidence['matched_index']].candidate_index))
    combined['x_local_score']=scores;combined['matched_x_candidate_index']=indices
    reference=json.loads((OUT/'reference.json').read_text())['records']
    binary=json.loads((OUT/'binary_probe/probe_report.json').read_text())
    positive_oof=[r['score'] for r in binary['predictions'] if r['split']=='calibration_oof' and r['label']=='ball']
    negative_threshold=min(.1,.5*min(positive_oof));positive_threshold=float(binary['selected_threshold'])
    selected=json.loads((OUT/'benchmark/benchmark_summary.json').read_text())
    x_threshold=float(selected['recommended_threshold']);assert selected['recommended_model_from_calibration']=='yolo26x'
    b=combined.binary_ball_score.to_numpy();x=combined.x_local_score.to_numpy()
    nonball=b<=negative_threshold
    ball=(b>=positive_threshold)&(x>=x_threshold)
    combined['verdict']=np.select([nonball,ball],['model_nonball','model_ball'],default='uncertain')
    combined['verdict_ko']=combined.verdict.map(dict(model_nonball='비공 추정',model_ball='공 추정',uncertain='판단 유보'))
    combined['verdict_reason']=np.select([nonball,ball],
        ['strong negative verifier evidence; not detector absence','positive verifier and object-local YOLO26x agree'],
        default='insufficient or conflicting evidence; retain for review')
    combined['ground_truth_confirmed']=False
    combined.to_csv(OUT/'all_candidate_verdicts.csv',index=False)
    combined[combined.candidate_origin.eq('original_s')].to_csv(OUT/'candidate_verdicts.csv',index=False)
    combined[combined.candidate_origin.eq('new_x')].to_csv(OUT/'new_x_candidate_verdicts.csv',index=False)
    for verdict,name in [('model_ball','ball_supported_candidates.csv'),('model_nonball','nonball_supported_candidates.csv'),('uncertain','uncertain_candidates.csv')]:
        combined[combined.verdict.eq(verdict)].to_csv(OUT/name,index=False)
    queue=read(OLD/'frames_to_redetect.csv')[['source_frame','half','video_seconds','interval_id']]
    counts=combined.groupby(['source_frame','verdict']).size().unstack(fill_value=0).reindex(columns=['model_ball','model_nonball','uncertain'],fill_value=0)
    queue=queue.merge(counts.reset_index(),on='source_frame',how='left').fillna(0)
    queue[['model_ball','model_nonball','uncertain']]=queue[['model_ball','model_nonball','uncertain']].astype(int)
    queue.to_csv(OUT/'frame_verdict_summary.csv',index=False)
    parameters=dict(x_model='YOLO26x',x_threshold=x_threshold,positive_binary_threshold=positive_threshold,
        negative_binary_threshold=negative_threshold,
        negative_threshold_rule='min(0.1, 0.5 * minimum positive grouped-OOF calibration score)',
        positive_threshold_rule='Fixed threshold selected from grouped OOF calibration only',
        ball_rule='binary >= positive threshold AND object-local X score >= X threshold',
        nonball_rule='binary <= negative threshold; detector absence alone never triggers nonball',
        uncertain_rule='all other candidates; do not remove existing coordinates',
        reference_sha256=sha(OUT/'reference.json'),binary_head_sha256=sha(OUT/'binary_probe/binary_head.npz'),
        verifier='Official frozen YOLO26s-cls embeddings + small L2 binary head trained only on 46 calibration objects',
        scores_are_calibrated_probabilities=False,gt_frame_whitelist_used=False,original_tracking_changed=False,
        active_ball_identity_determined=False)
    (OUT/'decision_parameters.json').write_text(json.dumps(parameters,indent=2)+'\n')
    by_origin={origin:g.verdict.value_counts().to_dict() for origin,g in combined.groupby('candidate_origin')}
    summary=dict(original_candidates=33744,new_x_candidates=len(new),total_candidates=len(combined),
        counts=combined.verdict.value_counts().to_dict(),per_origin=by_origin,source_frames=11092,
        no_original_coordinates_changed=True,model_verdicts_not_manual_ground_truth=True,
        uncertain_candidates_remain_available=True)
    (OUT/'candidate_summary.json').write_text(json.dumps(summary,indent=2)+'\n')
    print(json.dumps(summary,indent=2))


def main():
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('stage',choices=['new','decide'])
    args=p.parse_args();{'new':prepare_new,'decide':decide}[args.stage]()


if __name__=='__main__':main()
