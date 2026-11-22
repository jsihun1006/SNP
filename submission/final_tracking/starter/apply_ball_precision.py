"""Apply object-local precision reviews while conserving the full original data."""
from pathlib import Path
import argparse,json,sys,hashlib
import numpy as np
import pandas as pd
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT))
from starter.fill_ball_bidirectional import predict_gap,homography_project
BASE=ROOT/'data/derived/ball_full_match_restored_20261116'
WORK=ROOT/'data/derived/ball_precision_review_20261117'
OUT=WORK/'final';N=164487
HFILE=ROOT/'data/broadcast/BAR-ATH_broadcast_homography.csv'
def read(p):return pd.read_csv(p,float_precision='round_trip',low_memory=False)
def save(n,v):(OUT/n).write_text(json.dumps(v,ensure_ascii=False,indent=2,allow_nan=False)+'\n')
def pair(x,y):return [float(x),float(y)] if np.isfinite([x,y]).all() else None
def scalar(x):return float(x) if np.isfinite(x) else None
def annotations():
 refs=json.loads((WORK/'training_reference_v3.json').read_text())['records']
 refs+=json.loads((WORK/'negative_audit/review.json').read_text())['records']
 if (WORK/'paint_review/review.json').exists():refs+=json.loads((WORK/'paint_review/review.json').read_text())['records']
 for audit in ['visual_audit','independent_audit']:
  path=ROOT/f'data/derived/ball_yolo26x_pitch_20261116/{audit}/review.json'
  audit_data=json.loads(path.read_text())
  for item in audit_data.get('records',[]):
   if item['label']!='uncertain':continue
   box=item.get('bbox_xyxy') or item.get('source_bbox_xyxy')
   if box is not None:refs.append(dict(source_frame=item['source_frame'],bbox_xyxy=box,label='uncertain',provenance='previous_unresolved_visual_object'))
  for item in audit_data.get('false_negatives',[]):
   if audit=='independent_audit' and int(item['source_frame'])==126534:continue
   box=item.get('bbox_xyxy') or item.get('suggested_bbox_xyxy')
   if box is None:
    x,y=float(item['image_x']),float(item['image_y']);radius=float(item.get('reference_radius_pixels',5));box=[x-radius,y-radius,x+radius,y+radius]
   refs.append(dict(source_frame=item['source_frame'],bbox_xyxy=box,label='ball',provenance='previous_exact_missed_ball_visual_review'))
 for name in ['new_visual_labels.json','white_visual_labels.json']:
  if not (WORK/name).exists():continue
  refs.extend(r for r in json.loads((WORK/name).read_text())['records'] if r['label']=='uncertain')
 if (WORK/'reference_corrections.json').exists():refs+=json.loads((WORK/'reference_corrections.json').read_text())['records']
 byframe={}
 for r in refs:byframe.setdefault(int(r['source_frame']),[]).append(r)
 return byframe
def manual_label(row,refs):
 labels=set()
 for r in refs.get(int(row['source_frame']),[]):
  box=r['bbox_xyxy'];center=(np.array(box[:2])+box[2:])/2
  width,height=np.array(box[2:])-box[:2]
  if not(box[0]-1<=row['image_x']<=box[2]+1 and box[1]-1<=row['image_y']<=box[3]+1):continue
  if np.linalg.norm(center-[row['image_x'],row['image_y']])>max(3.,.45*np.hypot(width,height)):continue
  labels.add(r['label'])
 if 'ball' in labels and ('nonball' in labels or 'not_ball' in labels):return 'annotation_conflict'
 if 'ball' in labels:return 'ball'
 if labels&{'nonball','not_ball'}:return 'nonball'
 return 'uncertain' if 'uncertain' in labels else 'not_reviewed'
def apply():
 OUT.mkdir(exist_ok=True)
 before=read(BASE/'ball_homography_all_frames.csv');d=before.copy();scores=read(WORK/'current_observation_reviews.csv');all_scores=read(WORK/'candidate_review_scores.csv')
 refs=annotations();scores['exact_visual_label']=[manual_label(r,refs) for r in scores.to_dict('records')]
 # Count coherent negative observations; an isolated classifier score cannot
 # erase an observed trajectory. Motion compares images, not field membership.
 s=scores.sort_values('source_frame').copy();delta=np.linalg.norm(np.diff(s[['image_x','image_y']],axis=0),axis=1)
 size=(s.bbox_x2-s.bbox_x1).to_numpy();jump=np.diff(s.source_frame)>3
 jump|=delta>np.maximum(40.,2*np.maximum(size[:-1],size[1:]))
 s['track_group']=np.cumsum(np.r_[True,jump])
 negative=s[s.strong_nonball_agreement].groupby('track_group').size();s['negative_track_support']=s.track_group.map(negative).fillna(0)
 # Safeguard moving white objects when both appearance heads are uncertain in
 # a difficult light/blur condition. Known exact non-balls remain rejectable.
 with np.load(ROOT/'data/derived/ball_fp_reduction_20261116/evidence.npz') as z:
  names=json.loads((ROOT/'data/derived/ball_fp_reduction_20261116/evidence_schema.json').read_text())['feature_names']
  white=np.zeros(N);yellow=np.zeros(N);white[z['feature_frames']]=z['features'][:,names.index('object_white')];yellow[z['feature_frames']]=z['features'][:,names.index('object_yellow')]
  mf=z['motion_frames'];affine=z['motion_affines'];valid=z['motion_valid']
 point=np.full((N,2),np.nan);point[s.source_frame.to_numpy(int)]=s[['image_x','image_y']]
 moving=np.zeros(N,bool)
 for k in range(1,len(mf)):
  a,b=int(mf[k-1]),int(mf[k])
  if not valid[k] or not np.isfinite(point[[a,b]]).all():continue
  residual=np.linalg.norm(affine[k]@np.r_[point[a],1]-point[b])
  if 6<residual<70:moving[a:b+1]=True
 s['moving_white_conflict']=moving[s.source_frame.to_numpy(int)]&(white[s.source_frame.to_numpy(int)]>.35)&(yellow[s.source_frame.to_numpy(int)]<.3)&s.strong_nonball_agreement
 s['reject_nonball']=(s.prior_uniform_rejection|(s.strong_nonball_agreement&s.negative_track_support.ge(3)&~s.moving_white_conflict)|s.exact_visual_label.eq('nonball'))&~s.exact_visual_label.isin(['ball','annotation_conflict','uncertain'])
 s['requires_review']=s.exact_visual_label.isin(['uncertain','annotation_conflict'])|s.moving_white_conflict
 s.loc[s.exact_visual_label.isin(['ball','nonball']),'requires_review']=False
 # A small yellow card/shirt region needs separate treatment from a white ball
 # held by a person. Human overlap alone is never a rejection condition.
 near_person=np.zeros(N,bool)
 for path in (ROOT/'data/derived/ball_uniform_context_20261116/pose_batches').glob('*.npz'):
  with np.load(path) as z:
   for j,f in enumerate(z['frames']):
    f=int(f)
    if yellow[f]<=.6 or white[f]>=.12 or not np.isfinite(point[f]).all():continue
    people=z['people'][z['offsets'][j]:z['offsets'][j+1]];x,y=point[f]
    near_person[f]=bool(((people[:,4]>=.5)&(people[:,0]<=x)&(people[:,2]>=x)&(people[:,1]<=y)&(people[:,3]>=y)).any()) if len(people) else False
 s['yellow_person_or_card_context']=near_person[s.source_frame.to_numpy(int)]&s.origin.eq('original_snp')&s.strong_nonball_agreement
 s.loc[s.yellow_person_or_card_context&~s.exact_visual_label.isin(['ball','uncertain','annotation_conflict']),'reject_nonball']=True
 for name in ['yellow_review','white_review']:
  manifest=json.loads((WORK/name/'manifest.json').read_text())
  approved=({*range(0,12),14,15,16,17,18,19,20,22,24,25,28,29,30} if name=='yellow_review' else set(range(24))-{2,8,9,15,21})
  for r in manifest:
   if r['review_id'] not in approved:continue
   m=s.source_frame.between(r['first'],r['last'])&s.score_v2.le(.3)&s.score_v3.le(.2)&~s.exact_visual_label.isin(['ball','uncertain','annotation_conflict'])
   s.loc[m,'reject_nonball']=True
 s['paint_object_temporal_rejection']=False
 if (WORK/'paint_review/review.json').exists():
  aspect=(s.bbox_x2-s.bbox_x1)/(s.bbox_y2-s.bbox_y1)
  for ref in json.loads((WORK/'paint_review/review.json').read_text())['records']:
   f=int(ref['source_frame']);p=before.loc[f,['pitch_x_m','pitch_y_m']].to_numpy(float)
   if not np.isfinite(p).all():continue
   nearby=s.source_frame.between(f-50,f+50)
   xy=before.loc[s.source_frame.to_numpy(int),['pitch_x_m','pitch_y_m']].to_numpy()
   match=nearby&(np.linalg.norm(xy-p,axis=1)<.25)&((aspect>1.6)|(aspect<.625))&s.score_v3.le(.3)&~s.exact_visual_label.isin(['ball','uncertain','annotation_conflict'])
   s.loc[match,'reject_nonball']=True;s.loc[match,'paint_object_temporal_rejection']=True
 # Keep every pre-correction measurement as an explicit immutable column.
 for key in ['image_x','image_y','pitch_x_m','pitch_y_m']:d['before_precision_'+key]=before[key]
 clear=before.is_prediction.copy()
 d.loc[clear,['image_x','image_y','pitch_x_m','pitch_y_m','candidate_index','left_anchor_frame','right_anchor_frame']]=np.nan
 d.loc[clear,'coordinate_kind']='missing';d.loc[clear,'is_prediction']=False;d.loc[clear,'pitch_usable']=False
 d['precision_decision']=np.where(d.coordinate_kind.eq('missing'),'missing','retained_observation')
 d['score_v2']=np.nan;d['score_v3']=np.nan;d['manual_object_label']='not_reviewed'
 sf=s.source_frame.to_numpy(int)
 d.loc[sf,['score_v2','score_v3']]=s[['score_v2','score_v3']].to_numpy()
 d.loc[sf,'manual_object_label']=s.exact_visual_label.to_numpy()
 d['before_precision_score_v2']=d.score_v2
 d['before_precision_score_v3']=d.score_v3
 bad=s.loc[s.reject_nonball|s.requires_review,'source_frame'].to_numpy(int)
 d.loc[bad,['image_x','image_y','pitch_x_m','pitch_y_m','candidate_index']]=np.nan
 d.loc[bad,'coordinate_kind']='missing';d.loc[bad,'pitch_usable']=False
 d.loc[s.loc[s.reject_nonball,'source_frame'].to_numpy(int),'precision_decision']='rejected_nonball'
 d.loc[s.loc[s.requires_review&~s.reject_nonball,'source_frame'].to_numpy(int),'precision_decision']='uncertain_object_preserved_for_review'
 s.to_csv(OUT/'observation_decisions.csv',index=False)
 s[s.reject_nonball].to_csv(OUT/'removed_nonball_observations.csv',index=False)
 s[s.requires_review].to_csv(OUT/'review_required_observations.csv',index=False)
 # Add newly acquired raw detections after they have been scored by the same
 # two heads. IDs occupy a disjoint range and every raw box remains available.
 newfile=WORK/'new_candidate_review_scores.csv'
 if newfile.exists():all_scores=pd.concat([all_scores,read(newfile)],ignore_index=True)
 raw=all_scores[all_scores.origin.isin(['yolo26x_raw','yolo26x_reacquisition'])].copy()
 raw['exact_visual_label']=[manual_label(r,refs) for r in raw.to_dict('records')]
 h=read(HFILE);matrices=h[[f'h{i}{j}' for i in range(3) for j in range(3)]].to_numpy().reshape(-1,3,3)
 raw['verified_seed_track']=False
 # Recover measured boxes of the visually verified stationary goal-area ball,
 # including low-confidence detections. This associates actual detections with
 # the reviewed object; it never copies or extrapolates its coordinates.
 seed_frame=126534
 seed=h.valid.iloc[seed_frame]==1
 if seed:
  q=np.linalg.solve(matrices[seed_frame],np.array([265.5,476.25,1.]));world=q[:2]/q[2]*[105.3,68.]
  ratio=(raw.bbox_x2-raw.bbox_x1)/(raw.bbox_y2-raw.bbox_y1)
  near=raw.source_frame.between(seed_frame-25,seed_frame+25)&raw.confidence.ge(.08)&raw.score_v2.ge(.5158360209901821)&raw.score_v3.ge(.3116492660650152)&ratio.between(.6,1.6)
  for idx,row in raw[near].iterrows():
   f=int(row.source_frame)
   if h.valid.iloc[f]!=1:continue
   q=np.linalg.solve(matrices[f],np.array([row.image_x,row.image_y,1.]));p=q[:2]/q[2]*[105.3,68.]
   if np.isfinite(p).all() and np.linalg.norm(p-world)<.6:raw.loc[idx,'verified_seed_track']=True
 confidence_ok=raw.confidence.ge(.3)|(raw.confidence.ge(.08)&raw.score_v2.ge(.95)&raw.score_v3.ge(.95))|raw.exact_visual_label.eq('ball')
 eligible=(((raw.score_v2.ge(.7)&raw.score_v3.ge(.08))|(raw.score_v3.ge(.7)&raw.score_v2.ge(.08))|raw.exact_visual_label.eq('ball'))&confidence_ok|raw.verified_seed_track)&~raw.exact_visual_label.isin(['nonball','uncertain','annotation_conflict'])
 raw=raw[eligible].sort_values(['source_frame','confidence'],ascending=[True,False])
 # A real image observation can support reacquisition even if the camera H is
 # missing. Do not turn a failed pitch conversion into a failed ball detection.
 anchors=d.index[d.coordinate_kind.ne('missing')&((d.score_v2.ge(.1)&d.score_v3.ge(.05))|d.manual_object_label.eq('ball'))].to_numpy()
 with np.load(ROOT/'data/derived/ball_fp_reduction_20261116/evidence.npz') as z:mf=z['motion_frames'];cuts=z['cut_color_distance']
 reacquired=[];recovery_reviews=[]
 for f,g in raw.groupby('source_frame'):
  f=int(f)
  if d.coordinate_kind.iloc[f]!='missing':continue
  i=np.searchsorted(anchors,f);l=int(anchors[i-1]) if i else None;r=int(anchors[i]) if i<len(anchors) else None
  candidates=[]
  for obj in g.to_dict('records'):
   if any(np.hypot(obj['image_x']-o['image_x'],obj['image_y']-o['image_y'])<max(5.,.4*(obj['bbox_x2']-obj['bbox_x1'])) for o in candidates):continue
   if obj['exact_visual_label']=='ball' or obj['verified_seed_track']:candidates.append(obj);continue
   if l is None or r is None or f-l>50 or r-f>50 or d.half.iloc[l]!=d.half.iloc[r]:continue
   sel=(mf>l)&(mf<=r)
   if sel.any() and np.nanmax(cuts[sel])>.30:continue
   t=(f-l)/(r-l);projected=None
   if d.pitch_usable.iloc[l] and d.pitch_usable.iloc[r] and h.valid.iloc[f]==1:
    expected=(1-t)*d.loc[l,['pitch_x_m','pitch_y_m']].to_numpy(float)+t*d.loc[r,['pitch_x_m','pitch_y_m']].to_numpy(float)
    projected=homography_project(matrices[[f]],expected[None])
   elif r-l<=12:
    projected=((1-t)*d.loc[l,['image_x','image_y']].to_numpy(float)+t*d.loc[r,['image_x','image_y']].to_numpy(float))[None]
   if projected is None:continue
   tolerance=max(22.,3*min(obj['bbox_x2']-obj['bbox_x1'],obj['bbox_y2']-obj['bbox_y1']))
   if np.linalg.norm(projected[0]-[obj['image_x'],obj['image_y']])>tolerance:continue
   candidates.append(obj)
  confirmed=[obj for obj in candidates if obj['exact_visual_label']=='ball']
  if confirmed:candidates=confirmed
  if len(candidates)!=1:
   recovery_reviews.append(dict(source_frame=f,candidates=len(candidates),reason='ambiguous_or_no_two_sided_identity_support'));continue
  obj=candidates[0];pixel=np.array([obj['image_x'],obj['image_y']]);pitch=np.full(2,np.nan)
  if h.valid.iloc[f]==1:
   try:
    q=np.linalg.solve(matrices[f],np.r_[pixel,1.]);pitch=q[:2]/q[2]*[105.3,68.]
   except np.linalg.LinAlgError:pass
  d.loc[f,['image_x','image_y','pitch_x_m','pitch_y_m','candidate_index']]=[*pixel,*pitch,obj['candidate_index']]
  d.loc[f,'coordinate_kind']='yolo26x_reacquired_observed';d.loc[f,'precision_decision']='reacquired_actual_detection'
  d.loc[f,'manual_object_label']=obj['exact_visual_label']
  d.loc[f,['score_v2','score_v3']]=[obj['score_v2'],obj['score_v3']]
  d.loc[f,'pitch_usable']=bool(np.isfinite(pitch).all() and -20<=pitch[0]<=125.3 and -20<=pitch[1]<=88)
  reacquired.append(dict(obj,identity_left_anchor=l,identity_right_anchor=r))
 pd.DataFrame(reacquired).to_csv(OUT/'reacquired_observations.csv',index=False);pd.DataFrame(recovery_reviews).to_csv(OUT/'recovery_review.csv',index=False)
 # Recompute short gaps from the remaining actual measurements. Stale previous
 # predictions and newly rejected observations never become endpoints.
 rows=d[d.coordinate_kind.ne('missing')&d.pitch_usable&((d.score_v2.ge(.1)&d.score_v3.ge(.05))|d.manual_object_label.eq('ball'))].copy();rows['is_observation']=True;rows=rows.to_dict('records')
 present=set(d.loc[d.coordinate_kind.ne('missing'),'source_frame']);gaps=[];predicted=[]
 for i,(left,right) in enumerate(zip(rows[:-1],rows[1:])):
  a,b=int(left['source_frame']),int(right['source_frame'])
  if b==a+1:continue
  pred,reason=predict_gap(left,right,rows[i-1] if i else None,rows[i+2] if i+2<len(rows) else None)
  if pred is not None:
   fs=np.arange(a,b+1);sel=(mf>a)&(mf<=b)
   if any(f in present for f in fs[1:-1]):reason='intervening_observation_preserved'
   elif d.precision_decision.iloc[fs].eq('uncertain_object_preserved_for_review').any():reason='uncertain_object_not_overwritten'
   elif not h.valid.iloc[fs].eq(1).all():reason='homography_unavailable'
   elif sel.any() and np.nanmax(cuts[sel])>.30:reason='scene_change'
   elif not d.half.iloc[fs].eq(left['half']).all():reason='half_boundary'
   else:
    image=homography_project(matrices[pred['source_frames']],pred['pitch_xy'])
    if image is None or not ((image>=[4,4])&(image<[1916,1076])).all():reason='image_projection_unavailable'
    else:
     reason='filled_two_sided'
     for j,f in enumerate(pred['source_frames']):
      if manual_label(dict(source_frame=int(f),image_x=image[j,0],image_y=image[j,1]),refs) in ['nonball','annotation_conflict']:continue
      d.loc[f,['image_x','image_y','pitch_x_m','pitch_y_m']]=[*image[j],*pred['pitch_xy'][j]]
      d.loc[f,['left_anchor_frame','right_anchor_frame']]=[a,b];d.loc[f,'coordinate_kind']='predicted_two_sided';d.loc[f,'is_prediction']=True;d.loc[f,'pitch_usable']=True;d.loc[f,'precision_decision']='predicted_between_actual_observations';predicted.append(int(f))
  gaps.append(dict(left_anchor_frame=a,right_anchor_frame=b,missing_frames=b-a-1,decision=reason))
 d['pitch_x_normalized']=d.pitch_x_m/105.3;d['pitch_y_normalized']=d.pitch_y_m/68.
 d.loc[d.is_prediction,['score_v2','score_v3']]=np.nan
 d.loc[d.is_prediction,'manual_object_label']='not_observed_prediction'
 # Known PiP/crossfade frames may contain real balls; reject the projection only.
 d.loc[d.source_frame.isin([131890,159623]),'pitch_usable']=False
 d.to_csv(OUT/'ball_homography_all_frames.csv',index=False);pd.DataFrame(gaps).to_csv(OUT/'gap_decisions.csv',index=False)
 overlay=[]
 for row in d[d.coordinate_kind.ne('missing')].itertuples():
  overlay.append(dict(source_frame=int(row.source_frame),candidate_index=-1 if pd.isna(row.candidate_index) else int(row.candidate_index),image_x=float(row.image_x),image_y=float(row.image_y),pitch_x_m=scalar(row.pitch_x_m),pitch_y_m=scalar(row.pitch_y_m),pitch_usable=bool(row.pitch_usable),is_prediction=bool(row.is_prediction)))
 save('overlay_objects.json',overlay)
 metadata=dict(frame_count=N,fps=25,pitch_size_m=[105.3,68],precision_100_percent_guaranteed=False,
  original_measurements_preserved_in_columns_with_prefix='before_precision_',homography_modified=False,
  outside_pitch_used_as_nonball=False,prediction_rule='Both endpoints actual observations; never extrapolate or use predictions as anchors')
 with (OUT/'ball_homography_all_frames.json').open('w') as out:
  out.write('{"metadata":'+json.dumps(metadata)+',"frames":[\n')
  for f,r in enumerate(d.itertuples()):
   matrix=matrices[f];record=dict(source_frame=f,video_seconds=f/25,half=int(r.half),coordinate_kind=r.coordinate_kind,precision_decision=r.precision_decision,
    ball_image_xy=pair(r.image_x,r.image_y),ball_pitch_xy_m=pair(r.pitch_x_m,r.pitch_y_m),pitch_usable=bool(r.pitch_usable),
    is_prediction=bool(r.is_prediction),left_anchor_frame=scalar(r.left_anchor_frame),right_anchor_frame=scalar(r.right_anchor_frame),
    before_precision_image_xy=pair(r.before_precision_image_x,r.before_precision_image_y),original_image_xy=pair(r.original_image_x,r.original_image_y),
    score_v2=scalar(r.score_v2),score_v3=scalar(r.score_v3),manual_object_label=r.manual_object_label,
    before_precision_score_v2=scalar(r.before_precision_score_v2),before_precision_score_v3=scalar(r.before_precision_score_v3),
    homography_valid=bool(r.homography_valid),homography_matrix=matrix.tolist() if np.isfinite(matrix).all() else None)
   out.write((',' if f else '')+json.dumps(record,separators=(',',':'),allow_nan=False)+'\n')
  out.write(']}\n')
 save('summary.json',dict(frames=N,input_observed_frames=len(s),model_candidates_scored=len(all_scores),
  rejected_nonball_observations=int(s.reject_nonball.sum()),uncertain_observations_separated=int(s.requires_review.sum()),
  reacquired_actual_observations=len(reacquired),predicted_frames=len(predicted),final_image_frames=len(overlay),final_pitch_frames=int(d.pitch_usable.sum()),
  observation_coordinates_not_deleted_from_archive=True,homography_modified=False,outside_pitch_used_as_nonball=False,
  guaranteed_precision_100_percent=False,selection_counts=d.coordinate_kind.value_counts().to_dict()))
 print((OUT/'summary.json').read_text(),flush=True)
if __name__=='__main__':
 parser=argparse.ArgumentParser();parser.add_argument('--draft',action='store_true');args=parser.parse_args()
 if args.draft:OUT=WORK/'draft'
 apply()
