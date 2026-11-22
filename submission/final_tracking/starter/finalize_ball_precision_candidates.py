"""Wait for the new full-frame detector, verify its objects, then apply reviews."""
from pathlib import Path
import json,sys,time,subprocess
import numpy as np
import pandas as pd
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT))
from starter.infer_ball_yolo26_binary_head import score_features
WORK=ROOT/'data/derived/ball_precision_review_20261117'
def main():
 metadata=WORK/'full_frame_reacquisition/run_metadata.json'
 while True:
  record=json.loads(metadata.read_text()) if metadata.exists() else {}
  if record.get('status')=='complete':break
  if record.get('status')=='failed':raise RuntimeError(record)
  time.sleep(2)
 raw=pd.read_csv(WORK/'full_frame_reacquisition/raw_ball_candidates.csv',float_precision='round_trip')
 raw['candidate_index']+=200000;raw['origin']='yolo26x_reacquisition'
 fields=['source_frame','candidate_index','bbox_x1','bbox_y1','bbox_x2','bbox_y2','image_x','image_y','confidence','origin']
 raw=raw[fields].sort_values(['source_frame','candidate_index']);target=WORK/'new_verification_candidates.csv';raw.to_csv(target,index=False)
 command=[str(ROOT/'yolo26_football_detection/.venv/bin/python'),str(ROOT/'starter/infer_ball_yolo26_binary_head.py'),
  '--source-candidates',str(target),'--output',str(WORK/'new_verifier'),'--head',str(WORK/'binary_head_v2.npz'),
  '--device','mps','--batch-size','128','--threads','4','--save-features']
 with (WORK/'new_verifier.log').open('w') as log:subprocess.run(command,cwd=ROOT,stdout=log,stderr=subprocess.STDOUT,check=True)
 heads=[]
 for version in ['v2','v3']:
  with np.load(WORK/f'binary_head_{version}.npz') as z:heads.append({k:z[k] for k in ['mean','std','coef']})
 pieces=[]
 for path in sorted((WORK/'new_verifier/features').glob('*.npz')):
  with np.load(path) as z:pieces.append(pd.DataFrame(dict(source_frame=z['source_frame'],candidate_index=z['candidate_index'],score_v2=score_features(heads[0],z['features'])[0],score_v3=score_features(heads[1],z['features'])[0])))
 scores=pd.concat(pieces,ignore_index=True);assert len(scores)==len(raw) and not scores.duplicated(['source_frame','candidate_index']).any()
 data=raw.merge(scores,on=['source_frame','candidate_index'],validate='one_to_one');data.to_csv(WORK/'new_candidate_review_scores.csv',index=False)
 with (WORK/'final_apply.log').open('w') as log:
  subprocess.run([str(ROOT/'/opt/anaconda3/envs/mtah2027/bin/python'),str(ROOT/'starter/apply_ball_precision.py')],cwd=ROOT,stdout=log,stderr=subprocess.STDOUT,check=True)
 print(json.dumps(dict(completed=True,new_full_frame_detections=record['completed_frames'],new_candidates_verified=len(data))),flush=True)
if __name__=='__main__':main()
