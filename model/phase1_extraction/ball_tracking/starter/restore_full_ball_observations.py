"""Restore the full SNP observations before applying local YOLO replacements."""
from pathlib import Path
import json,sys,hashlib
import numpy as np
import pandas as pd
ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT))
from starter.fill_ball_bidirectional import predict_gap,homography_project
OUT=ROOT/'data/derived/ball_full_match_restored_20261116'
OLD=ROOT/'data/derived/ball_bidirectional_fill_20261116'
GREEN=ROOT/'data/derived/ball_yolo26x_pitch_20261116'
HFILE=ROOT/'data/broadcast/BAR-ATH_broadcast_homography.csv'
N=164487
def read(p):return pd.read_csv(p,float_precision='round_trip',low_memory=False)
def sha(p):return hashlib.sha256(Path(p).read_bytes()).hexdigest()
def save(name,d):(OUT/name).write_text(json.dumps(d,ensure_ascii=False,indent=2,allow_nan=False)+'\n')
def finite(v):return float(v) if np.isfinite(v) else None
def pair(x,y):return [float(x),float(y)] if np.isfinite([x,y]).all() else None
def build():
 OUT.mkdir(exist_ok=True)
 basepath=ROOT/'data/derived/ball_fp_reduction_20261116/filtered_ball_coordinates.csv'
 base=read(basepath);green=read(GREEN/'reviewed_ball_pitch.csv');fp=read(GREEN/'confirmed_false_positives.csv');h=read(HFILE)
 inputsha={str(p.relative_to(ROOT)):sha(p) for p in [basepath,GREEN/'reviewed_ball_pitch.csv',HFILE]}
 assert len(base)==N and base.decision.eq('kept_observation').sum()==105400
 result=pd.DataFrame(dict(source_frame=np.arange(N),video_seconds=np.arange(N)/25,half=base.half,
  original_source_status=base.source_status,original_decision=base.decision,
  original_image_x=base.pixel_x,original_image_y=base.pixel_y,
  original_pitch_x_m=base.pitch_x_m,original_pitch_y_m=base.pitch_y_m))
 for k in ['image_x','image_y','pitch_x_m','pitch_y_m','candidate_index','left_anchor_frame','right_anchor_frame'] :result[k]=np.nan
 result['coordinate_kind']='missing';result['is_prediction']=False;result['selection_reason']='no_observation'
 observed=base.decision.eq('kept_observation')
 result.loc[observed,['image_x','image_y','pitch_x_m','pitch_y_m']]=base.loc[observed,['pixel_x','pixel_y','pitch_x_m','pitch_y_m']].to_numpy()
 result.loc[observed,'coordinate_kind']='original_snp_observed';result.loc[observed,'candidate_index']=-1
 result.loc[observed,'selection_reason']='original_observation_retained'
 # An exact visually confirmed non-ball can reject that original point; being
 # outside the targeted YOLO queue is never a reason to remove an observation.
 rejected=[]
 for r in fp.itertuples():
  f=int(r.source_frame);x,y=result.loc[f,['image_x','image_y']]
  if np.isfinite([x,y]).all() and r.bbox_x1-2<=x<=r.bbox_x2+2 and r.bbox_y1-2<=y<=r.bbox_y2+2:
   rejected.append(f);result.loc[f,['image_x','image_y','pitch_x_m','pitch_y_m','candidate_index']]=np.nan
   result.loc[f,'coordinate_kind']='missing';result.loc[f,'selection_reason']='exact_visually_confirmed_nonball'
 chosen=[];ambiguous=[]
 for f,g in green.groupby('source_frame'):
  if len(g)!=1:
   ambiguous.append(int(f));continue
  r=g.iloc[0]
  if not r.pitch_usable or r.visual_audit_label in ['uncertain','conflicting_review']:continue
  result.loc[f,['image_x','image_y','pitch_x_m','pitch_y_m','candidate_index']]=r[['image_x','image_y','pitch_x_m','pitch_y_m','candidate_index']].to_numpy()
  result.loc[f,'coordinate_kind']='yolo26x_observed';result.loc[f,'selection_reason']='local_single_approved_yolo26x_replacement';chosen.append(int(f))
 result['homography_valid']=h.valid.iloc[:N].to_numpy()==1
 context=np.isin(np.arange(N),[131890,159623])
 result['pitch_usable']=np.isfinite(result[['pitch_x_m','pitch_y_m']]).all(axis=1)&result.homography_valid&result.pitch_x_m.between(-20,125.3)&result.pitch_y_m.between(-20,88)&~context
 matrices=h[[f'h{i}{j}' for i in range(3) for j in range(3)]].to_numpy().reshape(-1,3,3)
 rows=result[result.pitch_usable].copy();rows['is_observation']=True;rows=rows.to_dict('records')
 present=set(result.loc[np.isfinite(result[['image_x','image_y']]).all(axis=1),'source_frame'])
 with np.load(ROOT/'data/derived/ball_fp_reduction_20261116/evidence.npz') as z:mf=z['motion_frames'];cuts=z['cut_color_distance']
 predictions=[];decisions=[]
 for i,(left,right) in enumerate(zip(rows[:-1],rows[1:])):
  a,b=int(left['source_frame']),int(right['source_frame'])
  if b==a+1:continue
  pred,reason=predict_gap(left,right,rows[i-1] if i else None,rows[i+2] if i+2<len(rows) else None)
  if pred is not None:
   fs=np.arange(a,b+1);sel=(mf>a)&(mf<=b)
   if any(f in present for f in fs[1:-1]):reason='intervening_original_observation_preserved'
   elif np.isin(fs,ambiguous).any():reason='ambiguous_redetection'
   elif context[fs].any():reason='projection_context_review'
   elif not result.half.iloc[fs].eq(left['half']).all():reason='half_boundary'
   elif not h.valid.iloc[fs].eq(1).all():reason='homography_invalid'
   elif sel.any() and np.nanmax(cuts[sel])>.30:reason='scene_change_cue'
   else:
    fullxy=np.vstack([[left['pitch_x_m'],left['pitch_y_m']],pred['pitch_xy'],[right['pitch_x_m'],right['pitch_y_m']]])
    image=homography_project(matrices[fs],fullxy)
    if image is None or not ((image>=[4,4])&(image<[1916,1076])).all():reason='image_projection_unavailable'
    elif np.linalg.norm(np.diff(image,axis=0),axis=1).max()>160:reason='camera_or_ball_discontinuity'
    else:
     reason='filled_two_sided'
     for j,f in enumerate(pred['source_frames']):
      result.loc[f,['image_x','image_y']]=image[j+1]
      result.loc[f,['pitch_x_m','pitch_y_m']]=pred['pitch_xy'][j]
      result.loc[f,['left_anchor_frame','right_anchor_frame']]=[a,b]
      result.loc[f,'coordinate_kind']='predicted_two_sided';result.loc[f,'is_prediction']=True;result.loc[f,'pitch_usable']=True
      result.loc[f,'selection_reason']=pred['method'];predictions.append(int(f))
  decisions.append(dict(left_anchor_frame=a,right_anchor_frame=b,missing_frames=b-a-1,decision=reason))
 result['pitch_x_normalized']=result.pitch_x_m/105.3;result['pitch_y_normalized']=result.pitch_y_m/68.
 result.to_csv(OUT/'ball_homography_all_frames.csv',index=False)
 base[observed].to_csv(OUT/'original_observations_preserved.csv',index=False)
 green.to_csv(OUT/'all_yolo26x_candidates.csv',index=False)
 pd.DataFrame(decisions).to_csv(OUT/'gap_decisions.csv',index=False)
 sourceframes=set(result.loc[result.coordinate_kind.ne('missing'),'source_frame'])
 unchanged=observed&~base.source_frame.isin(chosen+rejected)
 np.testing.assert_array_equal(result.loc[unchanged,['image_x','image_y','pitch_x_m','pitch_y_m']],base.loc[unchanged,['pixel_x','pixel_y','pitch_x_m','pitch_y_m']])
 assert set(base.loc[observed,'source_frame'])<=sourceframes|set(rejected)
 assert not result.loc[observed,'is_prediction'].any()
 # The new renderer includes every restored observation, not the YOLO subset.
 overlay=[]
 for r in result[result.coordinate_kind.ne('missing')].itertuples():
  overlay.append(dict(source_frame=int(r.source_frame),candidate_index=-1 if pd.isna(r.candidate_index) else int(r.candidate_index),
   image_x=float(r.image_x),image_y=float(r.image_y),pitch_x_m=finite(r.pitch_x_m),pitch_y_m=finite(r.pitch_y_m),pitch_usable=bool(r.pitch_usable),is_prediction=bool(r.is_prediction)))
 save('overlay_objects.json',overlay)
 meta=dict(frame_count=N,fps=25,pitch_size_m=[105.3,68],original_observed=105400,original_input_sha256=inputsha,
  original_observations_not_deleted=True,all_original_observations_preserved_in='original_observations_preserved.csv',
  old_partial_output_invalid='ball_bidirectional_fill_20261116',homography_modified=False,
  homography_direction='normalized pitch to original 1920x1080 pixels',
  legacy_interpolated_points_not_used_as_observed_anchors=True,
  ambiguous_yolo_frames='Original observation retained; all alternatives in all_yolo26x_candidates.csv')
 path=OUT/'ball_homography_all_frames.json'
 with path.open('w') as out:
  out.write('{"metadata":'+json.dumps(meta,ensure_ascii=False)+',"frames":[\n')
  for f,r in enumerate(result.itertuples()):
   matrix=matrices[f];record=dict(source_frame=f,video_seconds=f/25,half=int(r.half),coordinate_kind=r.coordinate_kind,
    image_xy=pair(r.image_x,r.image_y),pitch_xy_m=pair(r.pitch_x_m,r.pitch_y_m),pitch_usable=bool(r.pitch_usable),
    is_prediction=bool(r.is_prediction),left_anchor_frame=finite(r.left_anchor_frame),right_anchor_frame=finite(r.right_anchor_frame),
    original_image_xy=pair(r.original_image_x,r.original_image_y),original_pitch_xy_m=pair(r.original_pitch_x_m,r.original_pitch_y_m),
    original_decision=r.original_decision,selection_reason=r.selection_reason,homography_valid=bool(r.homography_valid),
    homography_matrix=matrix.tolist() if np.isfinite(matrix).all() else None)
   record['ball_image_xy']=record['image_xy']
   record['ball_pitch_xy_m']=record['pitch_xy_m']
   record['ball_pitch_xy_normalized']=None if record['pitch_xy_m'] is None else [record['pitch_xy_m'][0]/105.3,record['pitch_xy_m'][1]/68.]
   out.write((',' if f else '')+json.dumps(record,allow_nan=False,separators=(',',':'))+'\n')
  out.write(']}\n')
 for p,expected in inputsha.items():assert sha(ROOT/p)==expected
 summary=dict(passed=True,frames=N,original_raw_observed=106689,previously_rejected_nonball_observations=1289,
  original_observations_preserved=105400,new_exact_nonball_rejections=rejected,
  yolo_replacement_frames=len(chosen),unchanged_original_observation_frames=int(unchanged.sum()),
  all_nonrejected_original_observed_frames_have_final_point=True,original_points_available_even_after_replacement=True,
  final_observed_frames=int((result.coordinate_kind.str.endswith('observed')).sum()),predicted_frames=len(predictions),
  final_image_frames=len(overlay),final_pitch_frames=int(result.pitch_usable.sum()),
  selection_counts=result.coordinate_kind.value_counts().to_dict(),input_sha256=inputsha)
 save('summary.json',summary);print(json.dumps(summary,indent=2),flush=True)
if __name__=='__main__':build()
