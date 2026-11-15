"""Read-only QA material for six no-H selected observations >=.60 confidence."""
import gzip
import hashlib
import json
from pathlib import Path

import cv2
import numpy as np

WORK=Path(__file__).resolve().parent
requested=json.loads((WORK/'full-run-late-highconf-selected.json').read_text())
frames=sorted(r['frame'] for r in requested);wanted=set(frames)
ROOT=Path(r'H:\data\ball-tracking-results\first_half\correction_recall_v1\chunks')
rows={};sources={}
for start in sorted({7171+((frame-7171)//1000)*1000 for frame in frames}):
    path=ROOT/str(start)/'decisions.jsonl.gz';digest=hashlib.sha256(path.read_bytes()).hexdigest()
    with gzip.open(path,'rt',encoding='utf-8') as stream:
        for row in map(json.loads,stream):
            if row['source_frame'] in wanted:rows[row['source_frame']]=row;sources[row['source_frame']]=(str(path),digest)
cap=cv2.VideoCapture(r'H:\data\broadcast\BAR-ATH_broadcast.mp4')
canvas=np.zeros((1410,1280,3),dtype=np.uint8);samples=[]
for i,frame in enumerate(frames):
    row=rows[frame];s=row['selection'];c=row['candidates'][s['candidate_index']]
    assert s['status']=='selected_observation' and c['confidence']>=.60 and 'pitch' not in c
    assert cap.set(cv2.CAP_PROP_POS_FRAMES,frame);ok,image=cap.read();assert ok
    panel=np.zeros((470,640,3),np.uint8);panel[30:390]=cv2.resize(image,(640,360))
    point=tuple(np.int32(np.array(c['point'])/3)+[0,30]);cv2.circle(panel,point,9,(0,255,0),2,cv2.LINE_AA)
    x,y=np.int32(np.round(c['point']));patch=image[max(0,y-32):min(1080,y+33),max(0,x-32):min(1920,x+33)]
    panel[391:469,561:639]=cv2.resize(patch,(78,78),interpolation=cv2.INTER_NEAREST)
    cv2.putText(panel,f'f{frame} conf={c["confidence"]:.3f} noH=True',(6,21),0,.47,(255,255,255),1,cv2.LINE_AA)
    cv2.putText(panel,f'green={c.get("green_fraction",0):.2f} needs_review={s["needs_review"]}',(6,411),0,.43,(230,230,230),1,cv2.LINE_AA)
    cv2.putText(panel,'Green=automated selected observation; crop at right.',(6,435),0,.38,(200,200,200),1,cv2.LINE_AA)
    yy,xx=(i//2)*470,(i%2)*640;canvas[yy:yy+470,xx:xx+640]=panel
    samples.append(dict(source_frame=frame,selection=s,candidate=c,source_chunk_decisions=sources[frame][0],source_chunk_sha256=sources[frame][1],visual_assessment='pending'))
cap.release();assert cv2.imwrite(str(WORK/'full-run-late-highconf-review.jpg'),canvas,[cv2.IMWRITE_JPEG_QUALITY,98])
report=dict(samples=samples,sampling='Six risk-biased no-H nongreen selections: three confidence.60-.80 and three>=.80, separate1000-frame chunks and not within100frames of first12late samples.',limitations='Risk-biased visual inspection only; no representative global accuracy claim.')
(WORK/'full-run-late-highconf-review.json').write_text(json.dumps(report,indent=2),encoding='utf-8')
print(json.dumps(dict(frames=frames)))
