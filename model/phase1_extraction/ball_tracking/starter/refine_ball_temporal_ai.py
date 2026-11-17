"""Offline second pass: visual two-ended association and learned gap interpolation.

All synthesis requires two retained measured endpoints. Original observations,
previous exclusions and the supplied homography remain available unchanged.
"""
from pathlib import Path
import argparse,copy,hashlib,json,sys
import numpy as np
import pandas as pd
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT))
from starter.apply_ball_precision import annotations,manual_label
from starter.fill_ball_bidirectional import homography_project
PREV=ROOT/'data/derived/ball_precision_review_20261117'
SOURCE=PREV/'final'
OUT=ROOT/'data/derived/ball_temporal_ai_20261117'
HFILE=ROOT/'data/broadcast/BAR-ATH_broadcast_homography.csv'
N=164487;FPS=25;COORD=['image_x','image_y','pitch_x_m','pitch_y_m']
def read(p):return pd.read_csv(p,float_precision='round_trip',low_memory=False)
def save(name,value):
 OUT.mkdir(exist_ok=True);(OUT/name).write_text(json.dumps(value,ensure_ascii=False,indent=2,allow_nan=False)+'\n')
def sha(p):return hashlib.sha256(Path(p).read_bytes()).hexdigest()
def environment():
 d=read(SOURCE/'ball_homography_high_precision_all_frames.csv')
 h=read(HFILE);m=h[[f'h{i}{j}' for i in range(3) for j in range(3)]].to_numpy().reshape(-1,3,3)
 with np.load(ROOT/'data/derived/ball_fp_reduction_20261116/evidence.npz') as z:mf=z['motion_frames'];cs=z['cut_color_distance']
 cut=np.zeros(N,bool)
 for i in np.flatnonzero(cs>.3):cut[max(0,int(mf[max(i-1,0)])+1):int(mf[i])+1]=True
 return d,h,m,cut
def bank():
 OUT.mkdir(exist_ok=True)
 a=read(PREV/'candidate_review_scores.csv');b=read(PREV/'new_candidate_review_scores.csv')
 table=pd.concat([a,b],ignore_index=True).sort_values(['source_frame','candidate_index']).reset_index(drop=True)
 assert len(table)==184920 and not table.duplicated(['source_frame','candidate_index']).any()
 lookup={(int(r.source_frame),int(r.candidate_index)):i for i,r in enumerate(table.itertuples())}
 matrix=np.lib.format.open_memmap(OUT/'appearance_features.npy',mode='w+',dtype=np.float16,shape=(len(table),2560));seen=np.zeros(len(table),bool)
 for directory in ['full_verifier','new_verifier','native_verifier']:
  for path in sorted((PREV/directory/'features').glob('*.npz')):
   with np.load(path) as z:
    ids=[lookup[(int(f),int(c))] for f,c in zip(z['source_frame'],z['candidate_index'])]
    assert not seen[ids].any();features=z['features'].astype(np.float32);features/=np.maximum(np.linalg.norm(features,axis=1,keepdims=True),1e-9)
    matrix[ids]=features.astype(np.float16);seen[ids]=True
 matrix.flush();assert seen.all()
 table['feature_index']=np.arange(len(table));table.to_csv(OUT/'candidate_bank.csv',index=False)
 save('bank_summary.json',dict(candidates=len(table),features=2560,all_embeddings_present=True,original_detection_or_coordinate_files_modified=False))
 print('Appearance bank complete',flush=True)
def curves(xy,a,b,frames,observed):
 """Fit velocities using up to four actual observations on each side."""
 left=frames[max(0,np.searchsorted(frames,a)-3):np.searchsorted(frames,a)+1]
 right=frames[np.searchsorted(frames,b):np.searchsorted(frames,b)+4]
 left=left[(a-left)<=8];right=right[(right-b)<=8]
 p0,p1=xy[a],xy[b];D=(b-a)/FPS;t=np.arange(1,b-a)/(b-a)
 avg=(p1-p0)/D
 if len(left)<2 or len(right)<2:return None
 tl=(left-a)/FPS;tr=(right-b)/FPS
 v0=np.sum(tl[:,None]*(xy[left]-p0),axis=0)/np.sum(tl**2)
 v1=np.sum(tr[:,None]*(xy[right]-p1),axis=0)/np.sum(tr**2)
 residual0=float(np.sqrt(np.mean((xy[left]-p0-tl[:,None]*v0)**2)))
 residual1=float(np.sqrt(np.mean((xy[right]-p1-tr[:,None]*v1)**2)))
 line=(1-t[:,None])*p0+t[:,None]*p1
 herm=(2*t**3-3*t**2+1)[:,None]*p0+(t**3-2*t**2+t)[:,None]*D*v0+(-2*t**3+3*t**2)[:,None]*p1+(t**3-t**2)[:,None]*D*v1
 forward=p0+t[:,None]*D*v0;backward=p1-(1-t[:,None])*D*v1
 disagreement=float(np.linalg.norm(forward-backward,axis=1).max())
 def norm(v):return float(np.linalg.norm(v))
 def cosine(u,v):return float(np.dot(u,v)/max(norm(u)*norm(v),1e-6))
 static=[D,norm(avg)/40,norm(v0)/40,norm(v1)/40,norm(v0-avg)/40,norm(v1-avg)/40,
  norm(v1-v0)/40,cosine(v0,avg),cosine(v1,avg),cosine(v0,v1),residual0,residual1,len(left)/4,len(right)/4,
  disagreement/2]
 features=np.c_[np.tile(static,(len(t),1)),t,t*(1-t)]
 return dict(line=line,hermite=herm,features=features,disagreement=disagreement,max_speed=max(norm(avg),norm(v0),norm(v1)),
  context=np.r_[left,right].astype(int),residual=max(residual0,residual1))
def model():
 import torch
 return torch.nn.Sequential(torch.nn.Linear(17,32),torch.nn.Tanh(),torch.nn.Linear(32,16),torch.nn.Tanh(),torch.nn.Linear(16,1),torch.nn.Sigmoid())
def train():
 import torch
 torch.set_num_threads(2);torch.manual_seed(20261117);rng=np.random.default_rng(20261117)
 d,h,m,cut=environment();xy=d[['pitch_x_m','pitch_y_m']].to_numpy();obs=d.ball_detected.to_numpy(bool)&d.pitch_usable.to_numpy(bool)
 frames=np.flatnonzero(obs);eligible=frames
 samples=[];segment=0
 # Temporal blocks are split before masking. Purge 100 frames around block
 # edges so overlapping context and target trajectories cannot leak.
 for gap in [1,3,5,8,12,20,30,45]:
  choices=[]
  for a in eligible[::3]:
   b=int(a)+gap+1
   if b+8>=N or a<8 or a//3000!=b//3000:continue
   if a%3000<100 or b%3000>2899:continue
   if not obs[a-3:b+4].all() or cut[a-8:b+9].any() or d.half.iloc[a]!=d.half.iloc[b]:continue
   choices.append(int(a))
  if len(choices)>1200:choices=rng.choice(choices,1200,replace=False).tolist()
  for a in choices:
   b=a+gap+1;visible=frames[(frames<=a)|(frames>=b)]
   c=curves(xy,a,b,visible,obs)
   if c is None or c['max_speed']>40 or c['disagreement']>3 or c['residual']>.5:continue
   truth=xy[a+1:b]
   if np.linalg.norm(np.diff(xy[a:b+1],axis=0),axis=1).max()*25>40:continue
   block=a//3000;split='test' if block%10 in [1,6] else ('validation' if block%10 in [3,8] else 'train')
   # Five interior samples per gap give long gaps no extra training weight.
   ids=np.unique(np.linspace(0,gap-1,min(gap,5)).astype(int))
   for j in ids:samples.append(dict(features=c['features'][j],line=c['line'][j],hermite=c['hermite'][j],target=truth[j],split=split,segment=segment,a=a,b=b,frame=a+1+int(j),gap=gap,block=block))
   segment+=1
 x=np.array([s['features'] for s in samples],np.float32);line=np.array([s['line'] for s in samples],np.float32);herm=np.array([s['hermite'] for s in samples],np.float32);target=np.array([s['target'] for s in samples],np.float32)
 masks={name:np.array([s['split']==name for s in samples]) for name in ['train','validation','test']}
 assert all(v.sum()>100 for v in masks.values())
 mean=x[masks['train']].mean(0);std=np.maximum(x[masks['train']].std(0),.03);z=torch.from_numpy(np.clip((x-mean)/std,-6,6))
 tl=torch.from_numpy(line);th=torch.from_numpy(herm);ty=torch.from_numpy(target)
 net=model();opt=torch.optim.Adam(net.parameters(),lr=.003,weight_decay=.002);best=None;bestloss=float('inf');history=[]
 ti=np.flatnonzero(masks['train']);vi=torch.from_numpy(np.flatnonzero(masks['validation']))
 for epoch in range(100):
  net.train()
  for ids in np.array_split(rng.permutation(ti),max(1,len(ti)//512)):
   idx=torch.from_numpy(ids);alpha=net(z[idx]);pred=alpha*th[idx]+(1-alpha)*tl[idx]
   loss=torch.nn.functional.smooth_l1_loss(pred,ty[idx],beta=.2);opt.zero_grad();loss.backward();opt.step()
  net.eval()
  with torch.no_grad():
   alpha=net(z[vi]);pred=alpha*th[vi]+(1-alpha)*tl[vi];vl=float(torch.nn.functional.mse_loss(pred,ty[vi]))
  if vl<bestloss:bestloss=vl;best=copy.deepcopy(net.state_dict());bestepoch=epoch
  history.append(vl)
 net.load_state_dict(best);net.eval()
 with torch.no_grad():alpha=net(z).numpy();prediction=alpha*herm+(1-alpha)*line
 def stats(values):return dict(median_m=float(np.median(values)),p95_m=float(np.quantile(values,.95)),rmse_m=float(np.sqrt(np.mean(values**2))))
 report={}
 for split,mask in masks.items():
  report[split]=dict(points=int(mask.sum()),segments=len({s['segment'] for s in np.array(samples,dtype=object)[mask]}),
   learned=stats(np.linalg.norm(prediction[mask]-target[mask],axis=1)),hermite=stats(np.linalg.norm(herm[mask]-target[mask],axis=1)),linear=stats(np.linalg.norm(line[mask]-target[mask],axis=1)))
 # Choose on validation only, before consulting test performance for any choice.
 selected=min(['learned','hermite','linear'],key=lambda name:report['validation'][name]['rmse_m'])
 torch.save(dict(state_dict=best,mean=mean.tolist(),std=std.tolist(),selected=selected),OUT/'bidirectional_blend_model.pt')
 rows=[]
 for i,s in enumerate(samples):rows.append({k:s[k] for k in ['split','segment','a','b','frame','gap','block']}|dict(learned_error_m=float(np.linalg.norm(prediction[i]-target[i])),hermite_error_m=float(np.linalg.norm(herm[i]-target[i])),linear_error_m=float(np.linalg.norm(line[i]-target[i])),alpha=float(alpha[i,0])))
 pd.DataFrame(rows).to_csv(OUT/'masked_observation_evaluation.csv',index=False)
 save('model_report.json',dict(method='Learned bidirectional-context blend of endpoint-preserving Hermite and linear paths',selected=selected,best_epoch=bestepoch,
  selection='Validation temporal blocks only; test excluded from model/epoch choice',blocked_split_frames=3000,purged_block_edges_frames=100,
  original_observations_are_proxy_targets_not_independent_physical_GT=True,metrics=report,source_sha256=sha(SOURCE/'ball_homography_high_precision_all_frames.csv')))
 print(json.dumps(dict(selected=selected,metrics=report),indent=2),flush=True)
def associate():
 d,h,m,cut=environment();table=read(OUT/'candidate_bank.csv');features=np.load(OUT/'appearance_features.npy',mmap_mode='r')
 keys={(int(r.source_frame),int(r.candidate_index)):int(r.feature_index) for r in table.itertuples()}
 observed=d.ball_detected.to_numpy(bool);af=np.flatnonzero(observed)
 feature_id=np.full(N,-1,int)
 for r in d[observed].itertuples():feature_id[r.source_frame]=keys[(int(r.source_frame),int(r.candidate_index))]
 assert (feature_id[observed]>=0).all()
 def vec(i):
  x=features[int(i)].astype(np.float32);return x/max(np.linalg.norm(x),1e-9)
 refs=annotations();table['manual_label']=[manual_label(r,refs) for r in table.to_dict('records')]
 # Object-specific prior exclusions, not frame/field-wide masks.
 excluded=read(SOURCE/'observation_decisions.csv');excluded=excluded[excluded.reject_nonball]
 bad={int(r.source_frame):r for r in excluded.itertuples()}
 def blocked(row):
  label=row['manual_label'];f=int(row['source_frame'])
  if label=='ball':return False
  if label in ['nonball','uncertain','annotation_conflict']:return True
  b=bad.get(f)
  return b is not None and b.bbox_x1-2<=row['image_x']<=b.bbox_x2+2 and b.bbox_y1-2<=row['image_y']<=b.bbox_y2+2
 allowed=((table[['score_v2','score_v3']].min(axis=1)>=.03)&(table[['score_v2','score_v3']].max(axis=1)>=.4)&(table.confidence.isna()|table.confidence.ge(.08)))|table.manual_label.eq('ball')
 groups={int(f):g.to_dict('records') for f,g in table[allowed].groupby('source_frame')}
 xy=d[['pitch_x_m','pitch_y_m']].to_numpy();pixels=d[['image_x','image_y']].to_numpy();half=d.half.to_numpy()
 def span(a,b):return half[a] in [1,2] and half[a]==half[b] and not cut[a+1:b+1].any()
 def expected(a,b,f):
  t=(f-a)/(b-a)
  if d.pitch_usable.iloc[a] and d.pitch_usable.iloc[b] and h.valid.iloc[f]==1:
   c=curves(xy,a,b,af,observed)
   p=(1-t)*xy[a]+t*xy[b] if c is None else c['hermite'][f-a-1]
   q=homography_project(m[[f]],p[None]);return q[0] if q is not None else None
  if b-a<=12:return (1-t)*pixels[a]+t*pixels[b]
  return None
 def candidates(a,b,f,minimum=.92):
  p=expected(a,b,f)
  if p is None:return []
  lvec,rvec=vec(feature_id[a]),vec(feature_id[b]);found=[]
  for obj in groups.get(f,[]):
   if blocked(obj):continue
   v=vec(obj['feature_index']);cl,cr=float(v@lvec),float(v@rvec)
   distance=float(np.linalg.norm(p-[obj['image_x'],obj['image_y']]))
   tolerance=max(9.,min(24.,1.3*max(obj['bbox_x2']-obj['bbox_x1'],obj['bbox_y2']-obj['bbox_y1'])))
   if obj['manual_label']!='ball' and (min(cl,cr)<minimum or distance>tolerance):continue
   item=dict(obj,appearance_left=cl,appearance_right=cr,trajectory_error_px=distance,left_anchor_frame=a,right_anchor_frame=b)
   if any(np.hypot(item['image_x']-x['image_x'],item['image_y']-x['image_y'])<max(4.,.35*(item['bbox_x2']-item['bbox_x1'])) for x in found):continue
   found.append(item)
  found.sort(key=lambda x:(x['manual_label']=='ball',min(x['appearance_left'],x['appearance_right'])-.002*x['trajectory_error_px']),reverse=True)
  if len(found)>1:
   top,next_=found[:2];margin=min(top['appearance_left'],top['appearance_right'])-min(next_['appearance_left'],next_['appearance_right'])
   if margin<.035:return []
  return found[:1]
 recoveries=[];review=[]
 for a,b in zip(af[:-1],af[1:]):
  a,b=int(a),int(b)
  if not 1<b-a<=51 or not span(a,b):continue
  for f in range(a+1,b):
   found=candidates(a,b,f)
   if found:recoveries.append(dict(found[0],correction='two_sided_visual_recovery'))
 # Recheck isolated excursions against both neighbours and the learned visual
 # identity. A kinematic excursion alone is insufficient to erase a point.
 ref=json.loads((PREV/'training_reference_v3.json').read_text())['records'];rf=np.load(PREV/'reference_features_v3.npz')['features'].astype(np.float32)
 rf/=np.maximum(np.linalg.norm(rf,axis=1,keepdims=True),1e-9)
 neg=rf[np.array([r['label']=='nonball' and r['split']=='training' for r in ref])]
 pos=rf[np.array([r['label']=='ball' and r['split']=='training' for r in ref])]
 replacements=[]
 for i in range(1,len(af)-1):
  a,f,b=map(int,af[i-1:i+2])
  if f-a>3 or b-f>3 or not span(a,b) or d.manual_object_label.iloc[f]=='ball':continue
  old=table.iloc[feature_id[f]];p=expected(a,b,f)
  if p is None:continue
  error=float(np.linalg.norm(p-pixels[f]));size=max(old.bbox_x2-old.bbox_x1,old.bbox_y2-old.bbox_y1)
  if error<max(40.,4*size):continue
  v=vec(feature_id[f]);similarity=min(float(v@vec(feature_id[a])),float(v@vec(feature_id[b])))
  sn,sp=float((neg@v).max()),float((pos@v).max())
  found=candidates(a,b,f,.94)
  replace=bool(found and min(found[0]['appearance_left'],found[0]['appearance_right'])>similarity+.04 and sn>.92 and sn>sp+.025 and min(d.score_v2.iloc[f],d.score_v3.iloc[f])<.85)
  review.append(dict(source_frame=f,old_image_x=pixels[f,0],old_image_y=pixels[f,1],trajectory_error_px=error,old_two_sided_similarity=similarity,negative_prototype_similarity=sn,positive_prototype_similarity=sp,replaced=replace))
  if replace:replacements.append(dict(found[0],correction='two_sided_false_positive_replacement'))
 all_changes=pd.DataFrame(recoveries+replacements)
 if len(all_changes):assert all_changes.source_frame.is_unique
 all_changes.to_csv(OUT/'observed_corrections.csv',index=False);pd.DataFrame(review).to_csv(OUT/'false_positive_recheck.csv',index=False)
 save('association_summary.json',dict(input_observations=int(observed.sum()),recovered_actual_measurements=len(recoveries),replaced_observations=len(replacements),temporal_outliers_reviewed=len(review),
  both_endpoints_input_actual_observations=True,predictions_used_as_visual_anchors=False,previous_nonball_objects_reintroduced=False))
 print((OUT/'association_summary.json').read_text(),flush=True)
def build():
 import torch
 torch.set_num_threads(2)
 before,h,m,cut=environment();d=before.copy();changes=read(OUT/'observed_corrections.csv');refs=annotations()
 for key in COORD:d['previous_pass_'+key]=before[key]
 for key in ['coordinate_kind','is_prediction','precision_decision','candidate_index','left_anchor_frame','right_anchor_frame']:
  d['previous_pass_'+key]=before[key]
 old_pred=d.is_prediction.to_numpy(bool)
 d.loc[old_pred,COORD+['pitch_x_normalized','pitch_y_normalized','candidate_index','left_anchor_frame','right_anchor_frame','score_v2','score_v3']]=np.nan
 d.loc[old_pred,'coordinate_kind']='missing';d.loc[old_pred,['ball_detected','ball_observed','ball_position_available','pitch_usable','is_prediction']]=False
 d['correction_type']=np.where(d.ball_detected,'retained_measured_observation','missing')
 d['prediction_method']='';d['prediction_space']='';d['prediction_alpha']=np.nan
 d[['left_anchor_frame','right_anchor_frame']]=np.nan
 d['recovery_left_anchor_frame']=np.nan;d['recovery_right_anchor_frame']=np.nan
 d.loc[~d.ball_position_available,'precision_decision']='missing_prior_candidate_archived'
 for r in changes.itertuples():
  f=int(r.source_frame);pixel=np.array([r.image_x,r.image_y]);world=np.full(2,np.nan)
  if h.valid.iloc[f]==1:
   try:q=np.linalg.solve(m[f],np.r_[pixel,1.]);world=q[:2]/q[2]*[105.3,68.]
   except np.linalg.LinAlgError:pass
  safe=np.isfinite(world).all() and (-20<=world[0]<=125.3) and (-20<=world[1]<=88) and f not in [131890,159623]
  d.loc[f,['image_x','image_y','candidate_index','score_v2','score_v3']]=[r.image_x,r.image_y,r.candidate_index,r.score_v2,r.score_v3]
  d.loc[f,['pitch_x_m','pitch_y_m']]=world if safe else np.nan
  d.loc[f,['ball_detected','ball_observed','ball_position_available']]=True;d.loc[f,'pitch_usable']=safe
  d.loc[f,'coordinate_kind']='ai_context_reacquired_observed';d.loc[f,'correction_type']=r.correction
  d.loc[f,'manual_object_label']=r.manual_label
  d.loc[f,'precision_decision']=r.correction
  d.loc[f,'selection_reason']='Measured candidate matched to both actual observed endpoints'
  d.loc[f,['recovery_left_anchor_frame','recovery_right_anchor_frame']]=[r.left_anchor_frame,r.right_anchor_frame]
 # New real detections may support synthesis; estimates never do.
 obs=d.ball_detected.to_numpy(bool);af=np.flatnonzero(obs);good=obs&d.pitch_usable.to_numpy(bool);gf=np.flatnonzero(good)
 xy=d[['pitch_x_m','pitch_y_m']].to_numpy();pixels=d[['image_x','image_y']].to_numpy();half=d.half.to_numpy()
 ckpt=torch.load(OUT/'bidirectional_blend_model.pt',map_location='cpu',weights_only=True);net=model();net.load_state_dict(ckpt['state_dict']);net.eval()
 report=json.loads((OUT/'model_report.json').read_text());val=report['metrics']['validation'];best=min(val[n]['rmse_m'] for n in ['learned','hermite','linear'])
 eligible=[n for n in ['learned','hermite','linear'] if val[n]['rmse_m']<=best*1.01];selected=min(eligible,key=lambda n:val[n]['p95_m'])
 save('operational_model_selection.json',dict(selected=selected,criterion='Minimum validation p95 subject to RMSE within 1% of validation best',development_metrics_not_independent_final_GT=True,validation_metrics=val))
 mean=np.array(ckpt['mean'],np.float32);std=np.array(ckpt['std'],np.float32)
 bank_table=read(OUT/'candidate_bank.csv');ids={(int(r.source_frame),int(r.candidate_index)):int(r.feature_index) for r in bank_table.itertuples()};feat=np.load(OUT/'appearance_features.npy',mmap_mode='r')
 feature_id=np.full(N,-1,int)
 for r in d[obs].itertuples():feature_id[r.source_frame]=ids[(r.source_frame,int(r.candidate_index))]
 def sim(a,b):
  x=feat[feature_id[a]].astype(np.float32);y=feat[feature_id[b]].astype(np.float32);return float(x@y/(np.linalg.norm(x)*np.linalg.norm(y)))
 predictions=[];gaps=[]
 for a,b in zip(af[:-1],af[1:]):
  a,b=int(a),int(b)
  if b==a+1:continue
  rec=dict(left_anchor_frame=a,right_anchor_frame=b,missing_frames=b-a-1,decision='not_filled')
  if b-a>51:rec['decision']='gap_too_long'
  elif half[a] not in [1,2] or half[a]!=half[b]:rec['decision']='half_boundary'
  elif cut[a+1:b+1].any():rec['decision']='camera_transition'
  elif sim(a,b)<.8:rec['decision']='endpoint_visual_identity_disagreement'
  elif good[a] and good[b] and h.valid.iloc[a:b+1].eq(1).all():
   c=curves(xy,a,b,gf,good)
   if c is None:
    if b-a>6:rec['decision']='insufficient_two_sided_motion_context';gaps.append(rec);continue
    t=np.arange(1,b-a)/(b-a);world=(1-t[:,None])*xy[a]+t[:,None]*xy[b];alpha=np.zeros(len(t));method='two_endpoint_short_linear'
   else:
    if not (half[c['context']]==half[a]).all() or cut[c['context'].min()+1:c['context'].max()+1].any() or c['max_speed']>40 or c['disagreement']>2 or c['residual']>.4:
     rec['decision']='two_sided_motion_inconsistent';gaps.append(rec);continue
    if selected=='learned':
     with torch.no_grad():alpha=net(torch.from_numpy(np.clip((c['features']-mean)/std,-6,6).astype(np.float32))).numpy().ravel()
    else:alpha=np.full(len(c['line']),1. if selected=='hermite' else 0.)
    world=alpha[:,None]*c['hermite']+(1-alpha[:,None])*c['line'];method='learned_bidirectional_blend' if selected=='learned' else 'validation_selected_'+selected
    rec['forward_backward_difference_m']=c['disagreement']
   path=np.vstack([xy[a],world,xy[b]])
   if np.linalg.norm(np.diff(path,axis=0),axis=1).max()*25>40 or not ((world>=[-20,-20])&(world<=[125.3,88])).all():rec['decision']='path_outlier';gaps.append(rec);continue
   # Large instantaneous H changes must not manufacture image locations.
   probe=np.array([[.25,.25,1],[.5,.5,1],[.75,.75,1]])
   ray=np.einsum('nij,kj->nki',m[a:b+1],probe);pp=ray[:,:,:2]/ray[:,:,2:]
   if not np.isfinite(pp).all() or np.median(np.linalg.norm(np.diff(pp,axis=0),axis=2),axis=1).max()>160:
    rec['decision']='camera_H_discontinuity';gaps.append(rec);continue
   image=homography_project(m[np.arange(a+1,b)],world)
   if image is None or not ((image>=[4,4])&(image<[1916,1076])).all():rec['decision']='outside_video_projection';gaps.append(rec);continue
   count=0
   for j,f in enumerate(range(a+1,b)):
    if manual_label(dict(source_frame=f,image_x=image[j,0],image_y=image[j,1]),refs) in ['nonball','uncertain','annotation_conflict']:continue
    predictions.append(dict(source_frame=f,image_x=image[j,0],image_y=image[j,1],pitch_x_m=world[j,0],pitch_y_m=world[j,1],left_anchor_frame=a,right_anchor_frame=b,method=method,space='pitch',alpha=float(alpha[j])));count+=1
   rec['decision']='predicted_two_sided';rec['filled_frames']=count
  elif b-a<=6 and sim(a,b)>.92 and np.linalg.norm(pixels[b]-pixels[a])<=10*(b-a):
   for f in range(a+1,b):
    t=(f-a)/(b-a);p=(1-t)*pixels[a]+t*pixels[b]
    if manual_label(dict(source_frame=f,image_x=p[0],image_y=p[1]),refs) in ['nonball','uncertain','annotation_conflict']:continue
    predictions.append(dict(source_frame=f,image_x=p[0],image_y=p[1],pitch_x_m=np.nan,pitch_y_m=np.nan,left_anchor_frame=a,right_anchor_frame=b,method='two_sided_visual_image_interpolation',space='image_only',alpha=0.))
   rec['decision']='predicted_image_only'
  else:rec['decision']='no_safe_projection_or_context'
  gaps.append(rec)
 pred=pd.DataFrame(predictions);assert pred.source_frame.is_unique
 fs=pred.source_frame.to_numpy(int)
 d.loc[fs,COORD]=pred[COORD].to_numpy();d.loc[fs,['left_anchor_frame','right_anchor_frame']]=pred[['left_anchor_frame','right_anchor_frame']].to_numpy()
 d.loc[fs,['ball_position_available','is_prediction']]=True;d.loc[fs,'pitch_usable']=pred.space.eq('pitch').to_numpy()
 d.loc[fs,'coordinate_kind']='ai_two_sided_predicted';d.loc[fs,'correction_type']='two_sided_gap_prediction'
 d.loc[fs,'prediction_method']=pred.method.to_numpy();d.loc[fs,'prediction_space']=pred.space.to_numpy();d.loc[fs,'prediction_alpha']=pred.alpha.to_numpy()
 d.loc[fs,['score_v2','score_v3']]=np.nan;d.loc[fs,'manual_object_label']='not_observed_prediction'
 d.loc[fs,'precision_decision']='predicted_between_actual_observations'
 d.loc[fs,'selection_reason']='Two actual endpoints; no extrapolation or recursive prediction'
 d.loc[fs,'candidate_index']=np.nan
 detection_fields=['confidence','bbox_x1','bbox_y1','bbox_x2','bbox_y2','origin']
 for field in detection_fields:
  d['detection_'+field]='' if field=='origin' else np.nan
 measured=d[d.ball_detected]
 measurements=bank_table.set_index(['source_frame','candidate_index'])
 measurement_keys=pd.MultiIndex.from_arrays([measured.source_frame,measured.candidate_index.astype(int)])
 matched=measurements.reindex(measurement_keys)
 assert matched.feature_index.notna().all()
 for field in detection_fields:d.loc[measured.index,'detection_'+field]=matched[field].to_numpy()
 d['pitch_x_normalized']=d.pitch_x_m/105.3;d['pitch_y_normalized']=d.pitch_y_m/68.
 d.to_csv(OUT/'ball_homography_all_frames.csv',index=False);pred.to_csv(OUT/'predicted_frames.csv',index=False);pd.DataFrame(gaps).to_csv(OUT/'gap_decisions.csv',index=False)
 save('summary.json',dict(frames=N,input_actual_observations=int(before.ball_detected.sum()),retained_input_predictions_not_used_as_anchors=0,
  recovered_actual_detections=int(changes.correction.eq('two_sided_visual_recovery').sum()),false_positive_replacements=int(changes.correction.eq('two_sided_false_positive_replacement').sum()),
  final_actual_observations=int(d.ball_detected.sum()),predicted_frames=len(pred),pitch_predictions=int(pred.space.eq('pitch').sum()),image_only_predictions=int(pred.space.eq('image_only').sum()),
  total_positions=int(d.ball_position_available.sum()),unresolved_frames=int((~d.ball_position_available).sum()),selected_prediction_model=selected,
  input_sha256={str(SOURCE/'ball_homography_high_precision_all_frames.csv'):sha(SOURCE/'ball_homography_high_precision_all_frames.csv'),str(HFILE):sha(HFILE)},homography_modified=False))
 print((OUT/'summary.json').read_text(),flush=True)
if __name__=='__main__':
 parser=argparse.ArgumentParser();parser.add_argument('stage',choices=['bank','train','associate','build']);globals()[parser.parse_args().stage]()
