"""Retained actual validation script; only report paths and cwd setup were made durable.
Original temporary source SHA256: 39d944c6c63fb619634d05b58765804c17a3d4843028413cb2acce8837b59d9f
Run from the repository root. Tracking inputs are read only.
"""
from pathlib import Path as _Path
_Path('outputs/ball_tracking_validation').mkdir(parents=True, exist_ok=True)
from pathlib import Path
import sys,json,collections
import numpy as np,pandas as pd
sys.path.insert(0,str(Path.cwd()))
from starter.refine_ball_temporal_ai import curves
N=164487;SEED=20261117
cols=['frame','final_x','final_y','is_prediction','final_actual_observation','tactical_used','highlight_inferred','alignment_half','final_source']
d=pd.read_csv('outputs/broadcast_tactical_ball_merge_v2/ball_homography_all_frames.csv',usecols=cols,float_precision='round_trip')
xy=d[['final_x','final_y']].to_numpy(float)*[105.,68.];valid=np.isfinite(xy).all(1);pred=d.is_prediction.to_numpy(bool);tac=d.tactical_used.to_numpy(bool);half=d.alignment_half.to_numpy(int);hl=d.highlight_inferred.to_numpy(bool)
chron=(half>0)&~(hl&~tac);obs=valid&chron&~pred&(d.final_actual_observation.to_numpy(bool)|tac);frames=np.flatnonzero(obs)
cut=np.zeros(N,bool)
with np.load('data/derived/ball_fp_reduction_20261116/evidence.npz') as z:
 mf=z['motion_frames'];cs=z['cut_color_distance']
 for k in np.flatnonzero(cs>.3):cut[max(0,int(mf[max(k-1,0)])+1):int(mf[k])+1]=True
rng=np.random.default_rng(SEED);records=[];segments=[];gates=collections.Counter();eligible_counts={}
for gap in [1,3,5,8,12,20]:
 eligible={'validation':[],'test':[]}
 # Same source observation can recur across masks; report explicitly rather than claiming independent samples.
 for a0 in frames[::3]:
  a=int(a0);b=a+gap+1
  if a<100 or b+8>=N:continue
  block=a//3000;split='validation' if block%10 in [3,8] else 'test' if block%10 in [1,6] else None
  if split is None or b//3000!=block or a%3000<100 or b%3000>2899:continue
  if not obs[a:b+1].all():gates['interior_or_endpoint_not_actual_recorded']+=1;continue
  if not (half[a:b+1]==half[a]).all():gates['half']+=1;continue
  # curves reads only actual supporting observations <=a and >=b; no hidden target can appear in context.
  c=curves(xy,a,b,frames,obs)
  if c is None:gates['side_support']+=1;continue
  context=c['context'];lo,hi=int(context.min()),int(context.max())
  assert np.all((context<=a)|(context>=b)) and obs[context].all()
  if not chron[lo:hi+1].all() or not (half[lo:hi+1]==half[a]).all():gates['context_chronology']+=1;continue
  if c['max_speed']>40 or c['residual']>.5 or c['disagreement']>2:gates['motion']+=1;continue
  all_tac=bool(tac[context].all())
  if not all_tac and cut[lo+1:hi+1].any():gates['mixed_or_broadcast_cut']+=1;continue
  truth=xy[a+1:b]
  if np.linalg.norm(np.diff(xy[a:b+1],axis=0),axis=1).max()*25>40:gates['target_track_not_stable']+=1;continue
  paths=[np.vstack([xy[a],c[name],xy[b]]) for name in ['line','hermite']]
  if any(np.linalg.norm(np.diff(path,axis=0),axis=1).max()*25>40 for path in paths):gates['proposed_path_speed']+=1;continue
  if not ((c['hermite']>=[-20,-20])&(c['hermite']<=[125,88])).all():gates['path_bounds']+=1;continue
  eligible[split].append(dict(a=a,b=b,block=block,gap_frames=gap,split=split,all_Tactical_support=all_tac,context=context.tolist(),hermite_error=np.linalg.norm(c['hermite']-truth,axis=1),linear_error=np.linalg.norm(c['line']-truth,axis=1),FB_disagreement=c['disagreement'],side_residual=c['residual']))
 eligible_counts[str(gap)]={split:len(v) for split,v in eligible.items()}
 for split,items in eligible.items():
  if len(items)>400:items=[items[i] for i in np.sort(rng.choice(len(items),400,replace=False))]
  for obj in items:
   sid=len(segments);segments.append({k:v for k,v in obj.items() if k not in ['hermite_error','linear_error']})
   for j,(h,l) in enumerate(zip(obj['hermite_error'],obj['linear_error'])):
    records.append(dict(split=split,segment=sid,block=obj['block'],a=obj['a'],b=obj['b'],frame=obj['a']+1+j,gap_frames=gap,all_Tactical_support=obj['all_Tactical_support'],hermite_error_m=float(h),linear_error_m=float(l)))
table=pd.DataFrame(records);table.to_csv('outputs/ball_tracking_validation/merged_v3_masked_validation.csv',index=False)
def stats(values):
 v=np.asarray(values,float);return dict(points=len(v),rmse_m=float(np.sqrt(np.mean(v*v))),median_m=float(np.median(v)),p95_m=float(np.quantile(v,.95)),mean_m=float(v.mean()),maximum_m=float(v.max()))
metrics={}
for split,g in table.groupby('split'):
 vals={method:stats(g[method+'_error_m']) for method in ['hermite','linear']}
 for method in ['hermite','linear']:
  segm=g.groupby('segment')[method+'_error_m'].apply(lambda x:float(np.mean(x.to_numpy()**2)))
  vals[method]['equal_weight_per_segment_rmse_m']=float(np.sqrt(segm.mean()))
 metrics[split]=dict(segments=int(g.segment.nunique()),blocks=int(g.block.nunique()),**vals,per_gap={str(int(gap)):{m:stats(z[m+'_error_m']) for m in ['hermite','linear']} for gap,z in g.groupby('gap_frames')},all_Tactical_support_segments=int(g.drop_duplicates('segment').all_Tactical_support.sum()))
val=metrics['validation'];selected='hermite' if val['hermite']['rmse_m']<val['linear']['rmse_m'] and val['hermite']['p95_m']<=val['linear']['p95_m'] else 'linear'
report=dict(passed=True,selected_method=selected,selection='Validation only: Hermite iff RMSE lower and P95 no worse than linear; test excluded from method choice',physical_ground_truth=False,targets='Artificially hidden current actual broadcast observations or recorded Tactical ATD reference positions; surrogate agreement, not independent physical accuracy',current_merged_track='outputs/broadcast_tactical_ball_merge_v2/ball_homography_all_frames.csv',frames=N,strict_actual_or_recorded_anchor_frames=int(obs.sum()),no_predicted_anchors_or_targets=True,interior_targets_not_used_in_context=True,source_corrected_track_not_reset=True,gap_frames=[1,3,5,8,12,20],max_segments_per_gap=800,max_segments_per_gap_per_split=400,deterministic_seed=SEED,anchor_stride=3,temporal_blocks_frames=3000,purge_frames=100,validation_block_mod10=[3,8],test_block_mod10=[1,6],correlated_overlapping_mask_segments=True,independent_samples_or_confidence_intervals_not_claimed=True,guard_thresholds=dict(max_speed_mps=40,max_side_fit_RMS_m=.5,max_forward_backward_disagreement_m=2,half_and_live_chronology_required=True,no_tactical_highlight_mixed_replay_base_anchors=True,Broadcast_cuts_ignored_only_if_all_frozen_support_and_endpoints_Tactical=True),eligible_segments_before_cap=eligible_counts,rejection_counts=dict(gates),metrics=metrics,segments=segments)
Path('outputs/ball_tracking_validation/merged_v3_masked_validation.json').write_text(json.dumps(report,indent=2)+'\n')
print(json.dumps({k:v for k,v in report.items() if k in ['selected_method','strict_actual_or_recorded_anchor_frames','eligible_segments_before_cap']}))
print(json.dumps(metrics))
