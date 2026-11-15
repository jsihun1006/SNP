"""Render exact-frame QA examples without changing observation coordinates."""
import gzip,json
from pathlib import Path
import cv2,numpy as np

BASE=Path(__file__).resolve().parent.parent
OUT=BASE/'outputs/full-correction'
source=Path(r'H:\data\ball-tracking-results\first_half\correction_recall_v1\qa_adjusted\frame_decisions.jsonl.gz')
targets={9042,33172,45711};records={}
with gzip.open(source,'rt',encoding='utf-8') as f:
    for line in f:
        row=json.loads(line)
        if row['source_frame'] in targets:records[row['source_frame']]=row
        if len(records)==len(targets):break
assert len(records)==3
cap=cv2.VideoCapture(r'H:\data\broadcast\BAR-ATH_broadcast.mp4')
tiles=[]
for frame,row in sorted(records.items()):
    cap.set(cv2.CAP_PROP_POS_FRAMES,frame);ok,image=cap.read();assert ok
    before=row.get('pre_quality_gate_selection') or row.get('pre_qa_selection') or row['selection']
    after=row['selection'];panels=[]
    for index,decision in enumerate((before,after)):
        panel=cv2.resize(image,(800,450));accepted=decision['status'] in ('selected_observation','visually_reviewed_observation')
        point=[decision.get('pixel_x'),decision.get('pixel_y')]
        if point[0] is None:point=[decision.get('provisional_pixel_x'),decision.get('provisional_pixel_y')]
        color=(40,230,80) if index and accepted else (0,180,255)
        if point[0] is not None:
            p=tuple(np.int32(np.asarray(point)*800/1920));cv2.circle(panel,p,9,color,2,cv2.LINE_AA)
            x,y=map(int,point);d=40;patch=image[max(0,y-d):min(1080,y+d),max(0,x-d):min(1920,x+d)]
            if patch.size:
                patch=cv2.resize(patch,(120,120));cv2.rectangle(patch,(0,0),(119,119),color,2);panel[36:156,676:796]=patch
        cv2.rectangle(panel,(0,0),(800,28),(20,20,20),-1)
        title=('AUTOMATIC BEFORE QA' if index==0 else 'FINAL QA')+f' | f{frame} | '+decision['status']
        cv2.putText(panel,title,(9,19),cv2.FONT_HERSHEY_SIMPLEX,.47,(240,240,240),1,cv2.LINE_AA)
        panels.append(panel)
    tiles.append(np.hstack(panels))
cap.release()
canvas=np.vstack(tiles)
cv2.imwrite(str(OUT/'qa-correction-examples.jpg'),canvas,[cv2.IMWRITE_JPEG_QUALITY,94])
print(json.dumps(dict(examples=sorted(records),output=str(OUT/'qa-correction-examples.jpg'))))
