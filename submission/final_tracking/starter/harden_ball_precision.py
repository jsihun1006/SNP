"""Full-observation verification with microphone/referee/paint hard negatives.

Raw observations remain immutable. Field bounds are never non-ball evidence.
"""
from pathlib import Path
import argparse,json,os,sys,time,hashlib
import numpy as np
import pandas as pd
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT))
from starter.probe_ball_yolo26_binary_verifier import fit,predict,metrics
OUT=ROOT/'data/derived/ball_precision_review_20261117'
MODEL=ROOT/'data/derived/ball_yolo26_model_audit_20261116'
RESTORED=ROOT/'data/derived/ball_full_match_restored_20261116'
VIDEO=ROOT/'data/broadcast/BAR-ATH_broadcast-001.mp4'
def read(p):return pd.read_csv(p,float_precision='round_trip',low_memory=False)
def save(n,v):(OUT/n).write_text(json.dumps(v,ensure_ascii=False,indent=2)+'\n')
def prepare():
 OUT.mkdir(exist_ok=True)
 current=read(RESTORED/'ball_homography_all_frames.csv');green=read(ROOT/'data/derived/ball_yolo26x_pitch_20261116/reviewed_ball_pitch.csv').set_index(['source_frame','candidate_index'])
 with np.load(ROOT/'data/derived/ball_fp_reduction_20261116/fp_inputs.npz') as z:boxes=z['bbox']
 rows=[]
 for r in current[current.coordinate_kind.isin(['original_snp_observed','yolo26x_observed'])].itertuples():
  f=int(r.source_frame)
  if r.coordinate_kind=='original_snp_observed':box=boxes[f];conf=np.nan;origin='original_snp';cid=-1
  else:
   g=green.loc[(f,int(r.candidate_index))];box=g[['bbox_x1','bbox_y1','bbox_x2','bbox_y2']].to_numpy(float);conf=g.confidence;origin='yolo26x_selected';cid=int(r.candidate_index)
  assert np.isfinite(box).all();np.testing.assert_allclose((box[:2]+box[2:])/2,[r.image_x,r.image_y],atol=1e-4,rtol=0)
  rows.append(dict(source_frame=f,candidate_index=cid,bbox_x1=box[0],bbox_y1=box[1],bbox_x2=box[2],bbox_y2=box[3],image_x=r.image_x,image_y=r.image_y,confidence=conf,origin=origin))
 full=pd.DataFrame(rows);full.to_csv(OUT/'current_observations.csv',index=False)
 raw=read(ROOT/'data/derived/ball_yolo26x_redetection_20261116/raw_ball_candidates.csv')
 raw['origin']='yolo26x_raw';raw.candidate_index+=100000
 both=pd.concat([full,raw[full.columns]],ignore_index=True).sort_values(['source_frame','candidate_index'])
 assert not both.duplicated(['source_frame','candidate_index']).any()
 both.to_csv(OUT/'verification_candidates.csv',index=False)
 old=json.loads((MODEL/'reference.json').read_text())['records'];refs=[]
 heldout_groups={r['group_id'] for r in old if r['split']!='calibration'}
 for r in old:refs.append(dict(source_frame=r['source_frame'],bbox_xyxy=r['bbox_xyxy'],label=r['label'],kind=r.get('object_kind',''),split='heldout' if r['split']!='calibration' else 'training',group=r['group_id'],provenance='original_79_reference'))
 additions=[]
 for name in ['visual_audit','independent_audit']:
  additions+=json.loads((ROOT/f'data/derived/ball_yolo26x_pitch_20261116/{name}/review.json').read_text())['records']
 post=json.loads((MODEL/'candidate_classification_reference.json').read_text())['records']
 additions += [dict(r,kind=r.get('kind','')) for r in post if r['source_frame']!=47423]
 for r in additions:
  if r['label'] not in ['ball','not_ball','nonball']:continue
  box=r.get('bbox_xyxy');box=box if box is not None else r.get('bbox')
  if box is None:continue
  f=int(r['source_frame']);label='ball' if r['label']=='ball' else 'nonball';center=(np.array(box[:2])+box[2:])/2
  if any(x['source_frame']==f and np.linalg.norm((np.array(x['bbox_xyxy'][:2])+x['bbox_xyxy'][2:])/2-center)<3 for x in refs):continue
  group=f//500;seed=f in [130087,126300,14669,24108,59087]
  split='heldout' if group in heldout_groups or (not seed and group%5==1) else 'training'
  refs.append(dict(source_frame=f,bbox_xyxy=box,label=label,kind=r.get('object_kind',r.get('kind','')),split=split,group=group,provenance='previous_exact_visual_audit'))
 # Freeze group split before any new scores are computed.
 testgroups={r['group'] for r in refs if r['split']=='heldout'}
 for r in refs:
  if r['group'] in testgroups:r['split']='heldout'
 save('training_reference.json',dict(records=refs,split_frozen_before_scoring=True,heldout_groups_never_used_for_training=True))
 save('request.json',dict(original_observations=len(full),raw_x_candidates=len(raw),total_candidates=len(both),reference_objects=len(refs),source_coordinates_modified=False,
  guarantees_100_percent=False,field_bounds_used_to_reject_objects=False,reason='Verify the entire restored observation set rather than only the old selected redetection subset'))
 print(json.dumps(dict(observations=len(full),raw=len(raw),references=len(refs)),indent=2),flush=True)
def train(version=''):
 OUT.mkdir(exist_ok=True);os.environ['YOLO_CONFIG_DIR']=str(OUT/'runtime/ultralytics');os.environ['MPLCONFIGDIR']=str(OUT/'runtime/matplotlib');os.environ['YOLO_AUTOINSTALL']='false'
 import cv2,torch
 from ultralytics import YOLO
 from starter.classify_ball_yolo26_candidates import square_crop
 torch.set_num_threads(4)
 refs=json.loads((OUT/f'training_reference{version}.json').read_text())['records']
 path=OUT/f'reference_features{version}.npz'
 if path.exists():x=np.load(path)['features']
 else:
  model=YOLO(str(MODEL/'models/yolo26s-cls.pt'));net=model.model.cpu().eval();net.requires_grad_(False);captured=[]
  hook=net.model[-1].linear.register_forward_pre_hook(lambda m,items:captured.append(items[0].detach().numpy().copy()))
  cap=cv2.VideoCapture(str(VIDEO));patches=[]
  images={}
  for f in sorted({r['source_frame'] for r in refs}):cap.set(cv2.CAP_PROP_POS_FRAMES,f);ok,im=cap.read();assert ok;images[f]=im
  cap.release()
  for r in refs:patches.extend(square_crop(images[r['source_frame']],r['bbox_xyxy'],s) for s in [1.25,2.])
  with torch.inference_mode():
   for a in range(0,len(patches),32):
    batch=np.stack(patches[a:a+32])[:,:,:,::-1].transpose(0,3,1,2);net(torch.from_numpy(np.ascontiguousarray(batch)).float()/255)
  hook.remove();x=np.concatenate(captured).reshape(len(refs),2560);np.savez_compressed(path,features=x)
 labels=np.array([r['label']=='ball' for r in refs],int);groups=np.array([r['group'] for r in refs]);train=np.array([r['split']=='training' for r in refs]);test=~train
 assert not set(groups[train])&set(groups[test])
 unique=np.unique(groups[train]);rng=np.random.default_rng(20261117);rng.shuffle(unique);folds={int(g):i%5 for i,g in enumerate(unique)}
 fold=np.array([folds[g] for g in groups[train]]);oof=np.full(train.sum(),np.nan);tx=x[train];ty=labels[train];tg=groups[train]
 for k in range(5):
  take=fold!=k;model=fit(tx[take],ty[take],tg[take]);oof[~take]=predict(model,tx[~take])
 values=np.unique(np.r_[0,oof,np.nextafter(oof,np.inf),1]);choices=[metrics(ty,oof,t) for t in values]
 # Maximize specificity subject to retaining >=95% of training OOF balls.
 eligible=[v for v in choices if v['recall']>=.95];chosen=max(eligible,key=lambda v:(v['specificity'],v['precision'],v['recall'],v['threshold']))
 head=fit(tx,ty,tg);np.savez_compressed(OUT/f'binary_head{version}.npz',**head,selected_threshold=chosen['threshold'])
 score=predict(head,x);records=[dict(r,score=float(s)) for r,s in zip(refs,score)]
 save(f'training_report{version}.json',dict(training_objects=int(train.sum()),heldout_objects=int(test.sum()),training_oof=chosen,
  heldout=metrics(labels[test],score[test],chosen['threshold']),selected_threshold=chosen['threshold'],heldout_tuned=False,
  scores_calibrated_probabilities=False,whole_match_accuracy_established=False,previous_development_holdout_reused=bool(version),
  oof_predictions=[dict(source_frame=int(r['source_frame']),label=r['label'],score=float(s)) for r,s in zip(np.array(refs,dtype=object)[train],oof)],predictions=records))
 print(json.dumps(dict(training=int(train.sum()),heldout=int(test.sum()),heldout_metrics=metrics(labels[test],score[test],chosen['threshold'])),indent=2),flush=True)
def upgrade():
 refs=json.loads((OUT/'training_reference.json').read_text())['records'];groups={r['group'] for r in refs if r['split']=='heldout'}
 d=read(OUT/'current_observations.csv').set_index('source_frame')
 manifest=json.loads((OUT/'yellow_review/manifest.json').read_text())
 approved={*range(0,12),14,15,16,17,18,19,20,22,24,25,28,29,30}
 labels=[]
 for r in manifest:
  rid=r['review_id']
  label='nonball' if rid in approved else 'uncertain'
  kind='pitchside_microphone' if rid in {*range(0,12),14,15,16,17,18} else ('tactical_graphic' if rid==19 else 'referee_body_hand_or_flag')
  for f in [r['first'],r['center'],r['last']]:
   row=d.loc[f];labels.append(dict(source_frame=f,bbox_xyxy=row[['bbox_x1','bbox_y1','bbox_x2','bbox_y2']].to_list(),label=label,kind=kind,
    group=f//500,split='heldout' if f//500 in groups or f//500%5==1 else 'training',
    provenance='new_exact_object_visual_review',evidence=f'yellow_review/page_{rid//8+1}.jpg'))
 save('new_visual_labels.json',dict(records=labels,uncertain_objects_not_used_as_negative_training=True,scope='Exact three pictured frames per interval; no frame-range GT assumption'))
 independent=json.loads((GREEN:=(ROOT/'data/derived/ball_yolo26x_pitch_20261116/independent_audit/review.json')).read_text())['records']
 for r in independent:
  if r['label'] not in ['ball','not_ball']:continue
  f=r['source_frame'];labels.append(dict(source_frame=f,bbox_xyxy=r['source_bbox_xyxy'],label='ball' if r['label']=='ball' else 'nonball',kind=r['object_kind'],
   group=f//500,split='heldout' if f//500 in groups or f//500%5==1 else 'training',provenance='independent_existing_visual_audit'))
 for r in labels:
  if r['label']=='uncertain':continue
  center=(np.array(r['bbox_xyxy'][:2])+r['bbox_xyxy'][2:])/2
  if any(x['source_frame']==r['source_frame'] and np.linalg.norm((np.array(x['bbox_xyxy'][:2])+x['bbox_xyxy'][2:])/2-center)<3 for x in refs):continue
  refs.append(r)
 testgroups={r['group'] for r in refs if r['split']=='heldout'}
 for r in refs:
  if r['group'] in testgroups:r['split']='heldout'
 save('training_reference_v2.json',dict(records=refs,split_frozen_before_new_scoring=True,heldout_groups_never_used_for_training=True))
 train('_v2')
def upgrade_white():
 refs=json.loads((OUT/'training_reference_v2.json').read_text())['records'];d=read(OUT/'current_observations.csv').set_index('source_frame')
 manifest=json.loads((OUT/'white_review/manifest.json').read_text());labels=[]
 positives={2,8,9,21};uncertain={15}
 for r in manifest:
  rid=r['review_id'];label='ball' if rid in positives else ('uncertain' if rid in uncertain else 'nonball')
  kind='real_football_or_restart_ball' if label=='ball' else ('pitchside_microphone' if rid==16 else 'advertisement_graphic_net_or_uniform')
  for f in [r['first'],r['center'],r['last']]:
   row=d.loc[f];labels.append(dict(source_frame=f,bbox_xyxy=row[['bbox_x1','bbox_y1','bbox_x2','bbox_y2']].to_list(),label=label,kind=kind,group=f//500,
    split='heldout' if f//500%5==1 else 'training',provenance='new_white_object_visual_review',evidence=f'white_review/page_{rid//8+1}.jpg'))
 save('white_visual_labels.json',dict(records=labels,real_corner_restart_ball_included=True,uncertain_not_rejected=True))
 for r in labels:
  if r['label']=='uncertain':continue
  center=(np.array(r['bbox_xyxy'][:2])+r['bbox_xyxy'][2:])/2
  if any(x['source_frame']==r['source_frame'] and np.linalg.norm((np.array(x['bbox_xyxy'][:2])+x['bbox_xyxy'][2:])/2-center)<3 for x in refs):continue
  refs.append(r)
 # A diagnostic line failure is now an explicit training case, and is removed
 # from evaluation together with its entire temporal group.
 for r in refs:
  if r['group']==252:r['split']='training'
 parent={r['group']:r['group'] for r in refs}
 def find(g):
  while parent[g]!=g:parent[g]=parent[parent[g]];g=parent[g]
  return g
 for name in ['yellow_review','white_review']:
  for r in json.loads((OUT/name/'manifest.json').read_text()):
   gs=[f//500 for f in [r['first'],r['center'],r['last']] if f//500 in parent]
   for g in gs[1:]:parent[find(g)]=find(gs[0])
 tests={find(r['group']) for r in refs if r['split']=='heldout'};explicit=find(252)
 for r in refs:
  g=find(r['group']);r['group']=g;r['split']='heldout' if g in tests and g!=explicit else 'training'
 save('training_reference_v3.json',dict(records=refs,entire_reviewed_episode_kept_in_one_split=True,
  diagnostic_group_252_now_training_and_excluded_from_validation=True,not_pristine_final_evaluation=True))
 train('_v3')
def rescore():
 from starter.infer_ball_yolo26_binary_head import score_features
 heads=[]
 for suffix in ['_v2','_v3']:
  with np.load(OUT/f'binary_head{suffix}.npz') as z:heads.append({k:z[k] for k in ['mean','std','coef']})
 pieces=[]
 for path in sorted((OUT/'full_verifier/features').glob('*.npz')):
  with np.load(path) as z:
   pieces.append(pd.DataFrame(dict(source_frame=z['source_frame'],candidate_index=z['candidate_index'],score_v2=score_features(heads[0],z['features'])[0],score_v3=score_features(heads[1],z['features'])[0])))
 scores=pd.concat(pieces,ignore_index=True);assert not scores.duplicated(['source_frame','candidate_index']).any()
 table=read(OUT/'verification_candidates.csv').merge(scores,on=['source_frame','candidate_index'],validate='one_to_one');assert len(table)==140235
 t2=json.loads((OUT/'training_report_v2.json').read_text())['selected_threshold'];t3=json.loads((OUT/'training_report_v3.json').read_text())['selected_threshold']
 table['strong_nonball_agreement']=table.score_v2.le(.05)&table.score_v3.le(.05)
 table['ball_support_v2']=table.score_v2.ge(t2);table['ball_support_v3']=table.score_v3.ge(t3)
 table.to_csv(OUT/'candidate_review_scores.csv',index=False)
 current=table[table.origin.ne('yolo26x_raw')].copy()
 # Retain the previously validated body-attachment rejections. Match the actual
 # observation coordinates, never a whole frame or all objects near a person.
 prior=read(ROOT/'data/derived/ball_uniform_context_20261116/new_uniform_rejections.csv').set_index('source_frame')
 oldmask=[]
 for r in current.itertuples():
  hit=False
  if r.source_frame in prior.index:
   o=prior.loc[r.source_frame];hit=np.hypot(r.image_x-o.source_pixel_x,r.image_y-o.source_pixel_y)<1e-4
  oldmask.append(hit)
 current['prior_uniform_rejection']=oldmask
 # Propose sequences for human/temporal review; detector absence and out-of-field
 # position never appear in the rejection rule.
 targets=current[current.strong_nonball_agreement|current.prior_uniform_rejection].sort_values('source_frame').copy()
 delta=np.linalg.norm(np.diff(targets[['image_x','image_y']],axis=0),axis=1)
 breaks=np.r_[True,(np.diff(targets.source_frame)>3)|(delta>60)]
 targets['group']=np.cumsum(breaks);intervals=[]
 for _,g in targets.groupby('group'):
  intervals.append(dict(first=int(g.source_frame.min()),last=int(g.source_frame.max()),center=int(g.source_frame.iloc[len(g)//2]),count=len(g),
   score=float(np.maximum(g.score_v2,g.score_v3).median())))
 intervals=sorted(intervals,key=lambda r:r['count'],reverse=True)
 save('negative_review_intervals.json',intervals)
 current.to_csv(OUT/'current_observation_reviews.csv',index=False)
 summary=dict(scored_candidates=len(table),scored_original_and_selected_observations=len(current),scored_raw_x_candidates=int(table.origin.eq('yolo26x_raw').sum()),
  strong_nonball_agreement=int(current.strong_nonball_agreement.sum()),prior_uniform_rejections_restored=int(current.prior_uniform_rejection.sum()),
  detector_absence_used_as_nonball=False,field_bounds_used_as_nonball=False,automatic_guarantee_100_percent=False)
 save('scoring_summary.json',summary);print(json.dumps(summary,indent=2),flush=True)
if __name__=='__main__':
 p=argparse.ArgumentParser();p.add_argument('stage',choices=['prepare','train','upgrade','upgrade_white','rescore']);globals()[p.parse_args().stage]()
