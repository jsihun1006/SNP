"""Read-only source-image audit of green-only ball observations and missed objects."""
from pathlib import Path
from collections import Counter
import json
import cv2
import numpy as np
import pandas as pd
ROOT=Path(__file__).resolve().parents[1]
SRC=ROOT/'data/derived/ball_yolo26x_redetection_20261116'
OUT=ROOT/'data/derived/ball_yolo26x_pitch_20261116/visual_audit'
VIDEO=ROOT/'data/broadcast/BAR-ATH_broadcast-001.mp4'
BOX=['bbox_x1','bbox_y1','bbox_x2','bbox_y2']

def rows():
 OUT.mkdir(parents=True,exist_ok=True)
 green=pd.read_csv(SRC/'ball_candidates.csv',float_precision='round_trip')
 raw=pd.read_csv(SRC/'all_candidates.csv',float_precision='round_trip')
 h=pd.read_csv(ROOT/'data/broadcast/BAR-ATH_broadcast_homography.csv')
 hm=h[[f'h{i}{j}' for i in range(3) for j in range(3)]].to_numpy().reshape(-1,3,3)
 selected=[];used=set();groups=set()
 def choose(table,count,why,min_group=False):
  for row in table.to_dict('records'):
   key=(int(row['source_frame']),int(row['candidate_index']));group=int(row['source_frame'])//500
   if key in used or (min_group and group in groups):continue
   selected.append(dict(**row,audit_stratum=why));used.add(key);groups.add(group)
   count-=1
   if count==0:break
 # Even temporal coverage by 24 equal row quantiles.
 ids=np.linspace(0,len(green)-1,24).astype(int)
 choose(green.iloc[ids],24,'temporal_spread')
 pitch=[]
 for r in green.itertuples():
  f=int(r.source_frame);p=np.array([r.image_x,r.image_y,1.]);value=[np.nan,np.nan]
  if h.valid.iloc[f]==1 and np.isfinite(hm[f]).all():
   try:
    ray=np.linalg.solve(hm[f],p)
    if abs(ray[2])>1e-12:value=ray[:2]/ray[2]*[105.3,68]
   except np.linalg.LinAlgError:pass
  pitch.append(value)
 pitch=np.array(pitch);green['audit_pitch_x']=pitch[:,0];green['audit_pitch_y']=pitch[:,1]
 out=(pitch[:,0]<0)|(pitch[:,0]>105.3)|(pitch[:,1]<0)|(pitch[:,1]>68)|~np.isfinite(pitch).all(1)
 choose(green[out].sort_values(['source_frame','candidate_index']),12,'outside_or_missing_projection',True)
 multi=green.groupby('source_frame').size();multi=set(multi[multi>1].index)
 choose(green[green.source_frame.isin(multi)].sort_values(['binary_ball_score','confidence']),8,'multiple_green_objects',True)
 choose(green.sort_values(['binary_ball_score','confidence']),4,'weakest_accepted',True)
 while len(selected)<48:choose(green.sample(frac=1,random_state=20261116),48-len(selected),'extra_temporal',True)
 assert len(selected)==48
 # Candidate miss proposals are hypotheses until source scenes are inspected.
 uncertain=raw[raw.verdict.eq('uncertain') & raw.confidence.ge(.55) & raw.binary_ball_score.ge(.05)].copy()
 widths=(uncertain.bbox_x2-uncertain.bbox_x1);heights=uncertain.bbox_y2-uncertain.bbox_y1
 uncertain=uncertain[(widths>=8)&(heights>=8)&(widths<200)&(heights<200)]
 # Near-threshold and high-detection candidates, prioritising no green at that location.
 green_groups={int(f):g[BOX].to_numpy(float) for f,g in green.groupby('source_frame')}
 valid=[]
 for i,r in uncertain.iterrows():
  b=green_groups.get(int(r.source_frame),np.empty((0,4)));p=np.array([r.image_x,r.image_y])
  overlap=((b[:,:2]-2<=p)&(b[:,2:]+2>=p)).all(1) if len(b) else []
  if not np.any(overlap):valid.append(i)
 uncertain=uncertain.loc[valid].sort_values(['binary_ball_score','confidence'],ascending=False)
 misses=[];usedgroups=set()
 for row in uncertain.to_dict('records'):
  group=int(row['source_frame'])//500
  if group in usedgroups:continue
  misses.append(dict(**row,audit_stratum='miss_hypothesis_high_detector_support'));usedgroups.add(group)
  if len(misses)==24:break
 (OUT/'green_sample_requests.json').write_text(json.dumps(dict(records=selected,count=len(selected),scope='Purposive stratified exact-green samples; not a random accuracy estimate'),indent=2))
 (OUT/'miss_proposals.json').write_text(json.dumps(dict(records=misses,count=len(misses),labels_are_unverified_hypotheses=True),indent=2))
 return selected,misses

def render(records,prefix):
 cap=cv2.VideoCapture(str(VIDEO));assert cap.isOpened()
 cells=[]
 for k,r in enumerate(records):
  f=int(r['source_frame']);idx=int(r['candidate_index']);box=np.array([r[q] for q in BOX]);p=np.array([r['image_x'],r['image_y']])
  cap.set(1,f);ok,image=cap.read();assert ok
  scene=cv2.resize(image,(480,270));cv2.rectangle(scene,tuple(np.rint(box[:2]/4).astype(int)),tuple(np.rint(box[2:]/4).astype(int)),(50,240,80),2)
  side=max(16,int(np.max(box[2:]-box[:2])*2));left=max(0,int(p[0]-side/2));top=max(0,int(p[1]-side/2));right=min(1920,left+side);bottom=min(1080,top+side)
  crop=cv2.resize(image[top:bottom,left:right],(160,160),interpolation=cv2.INTER_NEAREST)
  cell=np.full((310,640,3),22,np.uint8);cell[35:305,:480]=scene;cell[35:195,480:]=crop
  label=f"{k+1} f{f} c{idx} B{r['binary_ball_score']:.3f} X{r['confidence']:.3f}"
  cv2.putText(cell,label,(5,21),0,.51,(235,235,235),1,cv2.LINE_AA)
  cv2.imwrite(str(OUT/f'{prefix}_case_{f}_{idx}.jpg'),cell);cells.append(cell)
  # Higher resolution scene and f-5/f/f+5 source-point crop evidence.
  full=np.full((850,1280,3),22,np.uint8)
  original=image.copy();cv2.rectangle(original,tuple(np.rint(box[:2]).astype(int)),tuple(np.rint(box[2:]).astype(int)),(50,240,80),3)
  full[35:575,:960]=cv2.resize(original,(960,540));full[35:355,960:]=cv2.resize(crop,(320,320))
  cv2.putText(full,label,(5,24),0,.7,(235,235,235),1,cv2.LINE_AA)
  for j,offset in enumerate([-5,0,5]):
   cap.set(1,f+offset);ok,adj=cap.read();assert ok
   small=cv2.resize(adj,(426,240));cv2.circle(small,tuple(np.rint(p/4.507).astype(int)),5,(0,180,255),1)
   full[605:845,j*426:(j+1)*426]=small
   cv2.putText(full,f'f{f+offset}',(j*426+5,598),0,.5,(235,235,235),1)
  cv2.imwrite(str(OUT/f'{prefix}_evidence_{f}_{idx}.jpg'),full)
 cap.release()
 for first in range(0,len(cells),6):
  subset=cells[first:first+6];sheet=np.full((int(np.ceil(len(subset)/2))*310,1280,3),22,np.uint8)
  for j,cell in enumerate(subset):sheet[j//2*310:(j//2+1)*310,j%2*640:(j%2+1)*640]=cell
  cv2.imwrite(str(OUT/f'{prefix}_sheet_{first//6+1:02d}.jpg'),sheet)
 print(prefix,len(records),flush=True)

def main():
 green,miss=rows();render(green,'green');render(miss,'miss');finalize_review()

def finalize_review():
 """Persist exact, manually inspected object labels without changing detections.

 The source scenes, exact object crops and (for suspect objects / confirmed
 misses) adjacent frames were directly inspected.  These fixed labels are an
 audit manifest, not rules to apply to other frames or nearby candidates.
 """
 samples=json.loads((OUT/'green_sample_requests.json').read_text())['records']
 proposals=json.loads((OUT/'miss_proposals.json').read_text())['records']
 nonballs={
  (162872,0):('goalkeeper_shirt_logo','초록색 골키퍼 유니폼의 Spotify 원형 인쇄와 가로 줄무늬를 검출했다. 원본 확대와 인접 프레임에서 옷에 붙은 인쇄임이 보인다.'),
  (14669,1):('assistant_referee_uniform_hand_or_flag','먼 터치라인 부심의 노란 옷·손·깃발 영역이다. 실제 축구공은 골키퍼 부근에 별도로 보인다.'),
  (19740,1):('assistant_referee_uniform_hand_or_flag','먼 코너 부근 부심의 노란 옷·손·깃발 영역이다. 실제 공은 페널티 지역 선수 발 부근에 따로 보인다.'),
  (20205,1):('assistant_referee_uniform_hand_or_flag','먼 터치라인 부심의 노란 소매·손·깃발 영역이다. 원본과 인접 프레임에서 인물에 붙어 움직인다.'),
  (24108,1):('assistant_referee_uniform_hand_or_flag','먼 터치라인 부심의 노란 옷·손·깃발 영역이다. 실제 축구공은 골키퍼 부근에 별도로 보인다.'),
  (32185,0):('assistant_referee_uniform_hand_or_flag','먼 터치라인 부심의 노란 옷과 손·깃발 영역이다. 실제 공은 경기장 안 볼 보유 선수 발 부근에 별도로 있다.'),
  (32575,1):('assistant_referee_uniform_hand_or_flag','먼 터치라인 부심의 노란 유니폼·손·깃발 영역이다. 드리블 선수의 실제 공과 떨어져 있다.'),
  (34141,0):('assistant_referee_uniform_hand_or_flag','먼 터치라인 부심의 노란 옷·손·깃발 영역이다. 실제 공은 경기장 안에서 따로 움직인다.'),
  (123731,1):('assistant_referee_flag','먼 코너 부심이 든 노란색·빨간색 깃발이다. 실제 공은 페널티 지역 공중에 별도로 보인다.'),
  (59087,1):('referee_uniform_hand','중앙 심판의 노란 유니폼·손과 검은 반바지에 걸친 검출이다. 실제 공은 왼쪽 윙 선수 쪽에 따로 보인다.'),
  (36785,1):('player_boot','달리는 바르셀로나 선수 다리에 연결된 분홍·흰 축구화와 양말이다. 확대 및 인접 프레임에서 발에 붙어 움직이며 실제 공은 골키퍼 쪽에 따로 있다.'),
  (105429,1):('player_boot','공격수 다리에 붙은 흰색·분홍색 축구화이다. 실제 공은 골키퍼 방향으로 따로 이동하며 인접 프레임에서 구분된다.'),
  (130087,1):('pitchside_microphone','코너 근처 회색 마이크 윈드실드이다. 직전 f130082 원본 클로즈업에서 받침·마이크 형태가 보이고, 실제 공은 코너 선수 쪽에 따로 있다.'),
 }
 unsure={
  (42579,1):'코너 깃발 옆 둥근 회색·흰 물체이다. 실제 경기 공은 경기장 안에 따로 보이지만, 이 물체가 여분 축구공인지 마이크인지 원본 해상도에서 확정할 수 없다.',
  (49043,1):'골대 바깥 그물·광고판 부근 작은 흰 물체이다. 여분 공인지 그물 부속인지 확정할 수 없어 비공으로 제거하지 않는다.',
  (64684,0):'먼 광고판 아래 둥근 흰 물체이다. 실제 경기 공은 선수 앞 공중에 따로 보이지만, 여분 공인지 다른 물체인지 확정할 수 없다.',
  (69799,1):'코너 깃발 옆 회색·분홍색 둥근 물체이다. 패널티 지점의 실제 정지 공과는 별개이며, 이 물체가 여분 공인지 마이크인지 확정할 수 없다.',
  (110572,2):'왼쪽 골대 바깥 그물 바닥의 흰 물체이다. 여분 공과 그물 부속을 구분할 수 없어 불확실로 둔다.',
  (90922,1):'먼 광고판 밑 둥근 회색·분홍색 물체이다. 공일 가능성을 배제할 수 없어 경기장 밖이라는 이유로 오탐 처리하지 않는다.',
  (120200,0):'골키퍼 옆 골망과 겹치는 둥근 흰색·분홍색 물체이다. 공 표면과 그물 겹침을 확실히 분리할 수 없어 불확실로 둔다.',
 }
 special={
  (30285,0):('ball','코너 준비 장면에서 먼 엔드라인 선수 발 옆에 흰색·분홍색 패턴의 실제 축구공이 보인다. 경기장 밖 위치만으로 오탐·여분 공으로 판정하지 않는다.',''),
  (129388,0):('ball','골키퍼가 손에 잡은 흰색·분홍색·검은 패턴 축구공이다. 선수·골키퍼와 일부 겹치지만 표면과 구형이 원본 확대에서 확인된다.',''),
  (131890,0):('ball','실제 패턴 축구공이 보이지만 두 방송 장면이 겹치는 전환 프레임이다. 공 분류와 평면 투영 적합성은 별도이다.','crossfade'),
  (159623,0):('ball','큰 빨간 리플레이 그래픽 안의 작은 재생 화면에 있는 실제 축구공이다. 메인 카메라 homography를 이 삽입 화면 좌표에 적용하면 맞지 않을 수 있다.','pip'),
  (99988,1):('ball','광고판 앞 공중에 떠 있는 실제 축구공이다. 인접 f99993 클로즈업에서 선수 옆으로 들어오는 공의 이동이 보이므로 광고판 무늬로 오인하여 제거하지 않는다.',''),
 }
 records=[]
 for s in samples:
  f=int(s['source_frame']);idx=int(s['candidate_index']);key=(f,idx)
  kind='football';label='ball';context=''
  reason='원본 전체 장면과 후보 확대에서 흰색·분홍색 패턴 또는 구형 및 플레이와 연결된 위치의 실제 축구공을 확인했다.'
  if key in nonballs:kind,reason=nonballs[key];label='not_ball'
  elif key in unsure:kind='unresolved_round_object';reason=unsure[key];label='uncertain'
  elif key in special:label,reason,context=special[key]
  records.append(dict(source_frame=f,candidate_index=idx,label=label,object_kind=kind,reason=reason,
   image_x=s['image_x'],image_y=s['image_y'],bbox_xyxy=[s[q] for q in BOX],
   evidence=f'visual_audit/green_evidence_{f}_{idx}.jpg',
   contact_sheet=f"visual_audit/green_sheet_{(samples.index(s)//6)+1:02d}.jpg",
   audit_stratum=s['audit_stratum'],projection_context=context,
   active_inplay_state_labelled=False,source_detection_changed=False))
 fn_reasons={
  147139:'골키퍼 쪽 공중 패스의 실제 흰 축구공이다. 인접 프레임에서 궤적이 이어지지만 이 프레임에 초록 승인 공이 없다.',
  159877:'헤더 이후 먼 터치라인 위 공중으로 이동하는 실제 축구공이다. 인접 프레임의 이동으로 확인되지만 초록 승인 공이 없다.',
  120058:'클로즈업에서 선수 다리 뒤에 부분 가려진 패턴 축구공이다. 물리적인 공이 선명하지만 초록 승인 공이 없다.',
  89834:'클로즈업에서 바르셀로나 선수 발 옆 실제 패턴 축구공이다. 초록 승인 공이 없다.',
  160858:'골키퍼 쪽 잔디 위로 패스되며 이동하는 실제 흰 축구공이다. 인접 프레임 궤적은 보이지만 초록 승인 공이 없다.',
  105449:'골키퍼 지역에서 날아가는 모션 블러가 있는 실제 축구공이다. 전후 프레임 이동으로 구분되지만 초록 승인 공이 없다.',
  18642:'낮은 카메라 경합 장면에서 아틀레틱 선수 발 옆 실제 축구공이다. 인접 프레임에서 움직이며 초록 승인 공이 없다.',
  125631:'다른 카메라에서 골망 뒤로 보이는 공중의 실제 패턴 축구공이다. 초록 승인 공이 없다. 라이브 인플레이 여부는 별도 라벨링하지 않았다.',
  117442:'골망이 보이는 클로즈업에서 이동 중인 실제 패턴 축구공이다. 초록 승인 공이 없다. 라이브 인플레이 여부는 별도 라벨링하지 않았다.',
  145236:'공격수·수비수 경합 발 앞 공중에 보이는 실제 축구공이다. 전후 프레임 이동을 확인했으나 초록 승인 공이 없다.',
  129434:'골키퍼 패스 후 페널티 아크 방향으로 이동하는 실제 축구공이다. 인접 프레임 궤적이 이어지지만 초록 승인 공이 없다.',
  152679:'페널티 지역 경합 중 공중으로 이동하는 실제 축구공이다. 인접 프레임 이동이 보이지만 초록 승인 공이 없다.',
 }
 all_green=pd.read_csv(SRC/'ball_candidates.csv')
 fns=[]
 for s in proposals:
  f=int(s['source_frame']);idx=int(s['candidate_index'])
  if f not in fn_reasons:continue
  assert len(all_green[all_green.source_frame.eq(f)])==0
  fns.append(dict(source_frame=f,candidate_index=idx,label='ball',image_x=s['image_x'],image_y=s['image_y'],
   bbox_xyxy=[s[q] for q in BOX],reason=fn_reasons[f],evidence=f'visual_audit/miss_evidence_{f}_{idx}.jpg',
   raw_yolo26x_confidence=s['confidence'],binary_ball_score=s['binary_ball_score'],source_verdict=s['verdict'],
   miss_stage='raw_X_detected_but_green_gate_rejected_to_uncertain',
   all_green_count_in_exact_frame=0,added_to_green_dataset=False,active_inplay_state_labelled=False))
 assert len(records)==48 and len(fns)==12
 reference=json.loads((ROOT/'data/derived/ball_yolo26_model_audit_20261116/reference.json').read_text())['records']
 retested=set(pd.read_csv(SRC/'frames_to_redetect.csv').source_frame.astype(int))
 reference_coverage=[]
 for ref in reference:
  if ref['label']!='ball':continue
  f=int(ref['source_frame']);p=np.array(ref['pixel_xy']);frame=all_green[all_green.source_frame.eq(f)]
  b=frame[BOX].to_numpy(float)
  hit=((b[:,:2]-2<=p)&(b[:,2:]+2>=p)).all(1)
  status='not_retested' if f not in retested else ('green_covers_reference_point' if hit.any() else 'needs_source_reinspection')
  reference_coverage.append(dict(source_frame=f,reference_id=ref['reference_id'],reference_pixel_xy=ref['pixel_xy'],
   status=status,matching_green_keys=[[f,int(i)] for i in frame.loc[hit,'candidate_index']]))
 manifest=dict(schema_version=1,source_video=str(VIDEO.relative_to(ROOT)),
  frame_convention='0-based original source frame; 25 fps; source 1920x1080',
  records=records,false_negatives=fns,counts=dict(green_samples=len(records),green_labels=dict(Counter(r['label'] for r in records)),confirmed_green_selection_misses=len(fns)),
  sampling=dict(method='Purposive stratified temporal / outside-or-missing-projection / multiple-green / weakest-accepted',
   green_population=4478,retested_source_frames=11092,whole_source_frames=164487),
  review_scope='All 48 source-scene plus exact crops directly inspected; suspect objects and confirmed misses additionally checked using adjacent original frames. Non-exhaustive physical-object audit.',
  fixed_35_ball_reference_coverage=dict(records=reference_coverage,counts=dict(Counter(r['status'] for r in reference_coverage))),
  limitations=[
   'This purposive sample cannot estimate full-match false-positive or false-negative rates.',
   'The 35-ball reference and 12 candidate posthoc records are sparse exact-object annotations, not exhaustive ball labels for all frames.',
   'Source frames outside the 11092-frame redetection queue are not_retested; absence of green there is not counted as a missed detection.',
   'Stationary footballs, penalty-kick balls, airborne footballs and outside-pitch objects are not classified as nonballs using those properties alone.',
   'Real airborne balls can have distorted pitch coordinates under a planar homography; this is separate from object misclassification.',
   'Live in-play state and active-match-ball identity are not exhaustively annotated; physical spare footballs are not binary nonball false positives.',
   'Confirmed misses are not promoted into the user-requested green-only output. No source detections or homography matrices were modified.',
  ],
  reference_caveats=[dict(source='data/derived/ball_yolo26_model_audit_20261116/candidate_classification_reference.json',reference_id=4,
   source_frame=47423,reference_pixel_xy=[1853.9384765625,235.87413024902344],
   issue='Reference claims same football beside goalkeeper, but its coordinates point to a distant touchline object. The actual goalkeeper ball is separately visible and correctly green-detected at [285,447] (current X candidate_index 0). Reference point excluded from confirmed FN count.',
   evidence='visual_audit/reference_evidence_47423_0.jpg',source_reference_changed=False)])
 (OUT/'review.json').write_text(json.dumps(manifest,ensure_ascii=False,indent=2))
 pd.DataFrame([r for r in records if r['label']=='not_ball']).to_csv(OUT/'confirmed_green_false_positives.csv',index=False)
 pd.DataFrame(fns).to_csv(OUT/'confirmed_green_selection_misses.csv',index=False)
 print(json.dumps(manifest['counts'],ensure_ascii=False),flush=True)

if __name__=='__main__':main()
