"""Read-only bounded H-present/penalty-mark/secondary-ball QA material."""
import csv
import gzip
import hashlib
import json
from pathlib import Path

import cv2
import numpy as np

WORK=Path(__file__).resolve().parent
ROOT=Path(r'H:\data\ball-tracking-results\first_half\correction_recall_v1\chunks')
frames=[45711,46249,47422,51596,52155,52706,52798,53270,53505,58745]
wanted=set(frames);rows={};sources={};errors={}
with open(r'H:\data\broadcast\BAR-ATH_broadcast_homography.csv',newline='',encoding='utf-8-sig') as stream:
    for row in csv.DictReader(stream):
        if int(row['frame']) in wanted:errors[int(row['frame'])]=float(row['reprojection_error'])
for start in sorted({7171+((frame-7171)//1000)*1000 for frame in frames}):
    path=ROOT/str(start)/'decisions.jsonl.gz';digest=hashlib.sha256(path.read_bytes()).hexdigest()
    with gzip.open(path,'rt',encoding='utf-8') as stream:
        for row in map(json.loads,stream):
            if row['source_frame'] in wanted:rows[row['source_frame']]=row;sources[row['source_frame']]=(str(path),digest)
cap=cv2.VideoCapture(r'H:\data\broadcast\BAR-ATH_broadcast.mp4');panels=[];samples=[]
for frame in frames:
    row=rows[frame];s=row['selection'];c=row['candidates'][s['candidate_index']]
    assert s['status']=='selected_observation' and 'pitch' in c
    assert cap.set(cv2.CAP_PROP_POS_FRAMES,frame);ok,image=cap.read();assert ok
    panel=np.zeros((470,640,3),np.uint8);panel[30:390]=cv2.resize(image,(640,360))
    point=tuple(np.int32(np.array(c['point'])/3)+[0,30]);cv2.circle(panel,point,9,(0,255,0),2,cv2.LINE_AA)
    x,y=np.int32(np.round(c['point']));patch=image[max(0,y-32):min(1080,y+33),max(0,x-32):min(1920,x+33)]
    panel[391:469,561:639]=cv2.resize(patch,(78,78),interpolation=cv2.INTER_NEAREST)
    cv2.putText(panel,f'f{frame} c={c["confidence"]:.3f} Herror={errors[frame]:.3f}',(6,21),0,.46,(255,255,255),1,cv2.LINE_AA)
    text=f'spot={c.get("penalty_spot_distance_m",0):.2f}m off={s["evidence"]["off_field_ground_projection"]} review={s["needs_review"]}'
    cv2.putText(panel,text,(6,410),0,.4,(230,230,230),1,cv2.LINE_AA)
    cv2.putText(panel,f'origin={c["origin"]}',(6,434),0,.4,(220,220,220),1,cv2.LINE_AA)
    panels.append(panel);samples.append(dict(source_frame=frame,selection=s,candidate=c,h_error=errors[frame],source_chunk_decisions=sources[frame][0],source_chunk_sha256=sources[frame][1],visual_assessment='pending'))
cap.release()
for page in range(3):
    canvas=np.zeros((940,1280,3),np.uint8)
    for j,panel in enumerate(panels[page*4:(page+1)*4]):
        y,x=(j//2)*470,(j%2)*640;canvas[y:y+470,x:x+640]=panel
    assert cv2.imwrite(str(WORK/f'full-run-pitch-review-{page+1}.jpg'),canvas,[cv2.IMWRITE_JPEG_QUALITY,98])
report=dict(samples=samples,sampling='Ten risk-biased H-present selected observations after45000: two near penalty spots, high ATD conflict, off-pitch projection and recovered candidates. At least50source frames apart.',limitations='Sparse risk-biased visual QA only, not full-match ground truth or a representative error-rate estimate.')
(WORK/'full-run-pitch-review.json').write_text(json.dumps(report,indent=2),encoding='utf-8')
print(json.dumps(dict(frames=frames)))
