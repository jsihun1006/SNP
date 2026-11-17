"""Export a precision-first view without discarding the original observations."""
from pathlib import Path
import json
import numpy as np
import pandas as pd
from PIL import Image,ImageDraw,ImageFont
ROOT=Path(__file__).resolve().parents[1];WORK=ROOT/'data/derived/ball_precision_review_20261117';OUT=WORK/'final'
def pair(x,y):return [float(x),float(y)] if np.isfinite([x,y]).all() else None
def scalar(v):return float(v) if np.isfinite(v) else None
def main():
 d=pd.read_csv(OUT/'ball_homography_all_frames.csv',float_precision='round_trip',low_memory=False)
 t2=json.loads((WORK/'training_report_v2.json').read_text())['selected_threshold'];t3=json.loads((WORK/'training_report_v3.json').read_text())['selected_threshold']
 available=np.isfinite(d[['image_x','image_y']]).all(axis=1)
 observed=available&~d.is_prediction
 accepted=observed&((d.score_v2.ge(t2)&d.score_v3.ge(t3))|d.manual_object_label.eq('ball')|d.precision_decision.eq('reacquired_actual_detection'))
 pred=d.is_prediction&available;keep_pred=np.zeros(len(d),bool)
 for r in d[pred].itertuples():
  keep_pred[r.source_frame]=bool(accepted.iloc[int(r.left_anchor_frame)] and accepted.iloc[int(r.right_anchor_frame)])
 keep=accepted|keep_pred;withhold=available&~keep
 d['candidate_was_prediction']=d.is_prediction
 for field in ['image_x','image_y','pitch_x_m','pitch_y_m','pitch_x_normalized','pitch_y_normalized']:d['review_candidate_'+field]=d[field]
 fields=['image_x','image_y','pitch_x_m','pitch_y_m','pitch_x_normalized','pitch_y_normalized']
 d.loc[withhold,fields]=np.nan;d.loc[withhold,'pitch_usable']=False
 d.loc[withhold&pred,'coordinate_kind']='prediction_withheld_unverified_endpoint'
 d.loc[withhold&observed,'coordinate_kind']='unverified_observation_preserved_for_review'
 d.loc[withhold,'is_prediction']=False;d.loc[withhold,'precision_decision']='withheld_from_precision_view'
 d.loc[~d.pitch_usable,['pitch_x_m','pitch_y_m','pitch_x_normalized','pitch_y_normalized']]=np.nan
 d['ball_observed']=accepted;d['ball_detected']=accepted;d['ball_position_available']=keep
 d.to_csv(OUT/'ball_homography_high_precision_all_frames.csv',index=False)
 d[withhold].to_csv(OUT/'unverified_candidates_preserved.csv',index=False)
 h=pd.read_csv(ROOT/'data/broadcast/BAR-ATH_broadcast_homography.csv',float_precision='round_trip')
 matrices=h[[f'h{i}{j}' for i in range(3) for j in range(3)]].to_numpy().reshape(-1,3,3)
 metadata=dict(frame_count=len(d),fps=25,pitch_size_m=[105.3,68],source_image_size=[1920,1080],view='precision_first',guaranteed_precision_100_percent=False,
  homography_direction='normalized pitch to original image pixels',scores_are_calibrated_probabilities=False,
  image_observation_can_exist_without_pitch_projection=True,
  uncertain_candidates='Preserved in review_candidate_* columns and candidate_image_xy; omitted from ball_image_xy',
  outside_pitch_used_as_nonball=False,homography_modified=False,observed_and_predicted_separated=True,
  prediction_rule='Both endpoints retained actual observations; predictions never used as anchors')
 with (OUT/'ball_homography_high_precision_all_frames.json').open('w') as out:
  out.write('{"metadata":'+json.dumps(metadata)+',"frames":[\n')
  for f,r in enumerate(d.itertuples()):
   matrix=matrices[f];obj=dict(source_frame=f,video_seconds=f/25,half=int(r.half),coordinate_kind=r.coordinate_kind,
    ball_detected=bool(r.ball_detected),ball_position_available=bool(r.ball_position_available),ball_image_xy=pair(r.image_x,r.image_y),
    ball_pitch_xy_m=pair(r.pitch_x_m,r.pitch_y_m),pitch_usable=bool(r.pitch_usable),is_prediction=bool(r.is_prediction),
    left_anchor_frame=scalar(r.left_anchor_frame),right_anchor_frame=scalar(r.right_anchor_frame),
    candidate_image_xy=pair(r.review_candidate_image_x,r.review_candidate_image_y),
    candidate_pitch_xy_m=pair(r.review_candidate_pitch_x_m,r.review_candidate_pitch_y_m),
    before_precision_image_xy=pair(r.before_precision_image_x,r.before_precision_image_y),original_image_xy=pair(r.original_image_x,r.original_image_y),
    score_v2=scalar(r.score_v2),score_v3=scalar(r.score_v3),precision_decision=r.precision_decision,
    homography_valid=bool(r.homography_valid),homography_matrix=matrix.tolist() if np.isfinite(matrix).all() else None)
   out.write((',' if f else '')+json.dumps(obj,separators=(',',':'),allow_nan=False)+'\n')
  out.write(']}\n')
 overlay=[]
 for r in d[keep].itertuples():overlay.append(dict(source_frame=int(r.source_frame),candidate_index=-1 if pd.isna(r.candidate_index) else int(r.candidate_index),image_x=float(r.image_x),image_y=float(r.image_y),pitch_x_m=scalar(r.review_candidate_pitch_x_m),pitch_y_m=scalar(r.review_candidate_pitch_y_m),pitch_usable=bool(r.pitch_usable),is_prediction=bool(r.is_prediction)))
 (OUT/'overlay_objects_high_precision.json').write_text(json.dumps(overlay,allow_nan=False,separators=(',',':'))+'\n')
 im=Image.open(ROOT/'data/derived/ball_bidirectional_fill_20261116/video_background.png').convert('RGBA');draw=ImageDraw.Draw(im)
 font=ImageFont.truetype('/System/Library/Fonts/AppleSDGothicNeo.ttc',18)
 draw.rectangle((0,45,970,89),fill=(21,24,30,255));draw.text((18,51),'초록: 검출·검증 관측 / 주황: 양방향 예측 · 불확실 후보는 별도 보존',font=font,fill=(238,210,130,255))
 draw.rectangle((0,630,1600,660),fill=(21,24,30,255));draw.text((18,634),'마이크·심판·흰색 물체 재검수 · 경기장 밖 실제 공 허용 · 원본 관측 보존 · H 결측은 영상 관측과 구분 · 100% 정확도 보장 아님',font=font,fill=(237,237,237,255))
 im.save(OUT/'video_background.png')
 messages=['신뢰할 공 관측 좌표 없음','영상 공 관측 있음 · 경기장 좌표 변환 확인 필요','공 좌표가 경기장 도면 바깥에 있음','주황: 앞·뒤 관측 사이의 예측 좌표']
 for i,text in enumerate(messages):
  sprite=Image.new('RGBA',(600,32),(0,0,0,0));ImageDraw.Draw(sprite).text((0,2),text,font=font,fill=(237,237,237,255));sprite.save(OUT/f'state_{i}.png')
 summary=dict(frames=len(d),observed_positions_shown=int(accepted.sum()),predicted_positions_shown=int(keep_pred.sum()),
  additional_unverified_positions_preserved=int(withhold.sum()),total_shown=int(keep.sum()),pitch_positions_shown=int((keep&d.pitch_usable).sum()),
  model_positive_threshold_v2=t2,model_positive_threshold_v3=t3,all_original_measurements_preserved=True,guaranteed_precision_100_percent=False)
 (OUT/'high_precision_summary.json').write_text(json.dumps(summary,indent=2)+'\n');print(json.dumps(summary,indent=2),flush=True)
if __name__=='__main__':main()
