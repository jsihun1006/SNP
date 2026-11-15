"""Prepare exact-frame visual review material without altering correction output."""
import gzip
import json
from pathlib import Path

import cv2
import numpy as np

WORK = Path(__file__).resolve().parent
SOURCE = Path(r'H:\data\ball-tracking-results\first_half\correction_recall_v1\chunks\8171\decisions.jsonl.gz')
with gzip.open(SOURCE,'rt',encoding='utf-8') as stream:
    records={r['source_frame']:r for r in map(json.loads,stream)}

cap=cv2.VideoCapture(r'H:\data\broadcast\BAR-ATH_broadcast.mp4')
# Fixed image regions are for viewing only, never replacement coordinates.
canvas=np.zeros((3*385,5*340,3),dtype=np.uint8)
for i,frame in enumerate(range(9039,9054)):
    row=records[frame]
    assert cap.set(cv2.CAP_PROP_POS_FRAMES,frame)
    ok,image=cap.read();assert ok
    x0,y0,x1,y1=1060,260,1240,465
    patch=image[y0:y1,x0:x1].copy()
    labels=[]
    for index,c in enumerate(row['candidates']):
        x,y=c['point']
        if x0<=x<x1 and y0<=y<y1:
            p=tuple(np.int32(np.round([x-x0,y-y0])))
            cv2.circle(patch,p,6,(0,200,255),1,cv2.LINE_AA)
            labels.append(f'{index}:{c["confidence"]:.3f}@({x:.1f},{y:.1f})')
    panel=np.zeros((385,340,3),dtype=np.uint8)
    panel[26:354,:288]=cv2.resize(patch,(288,328),interpolation=cv2.INTER_NEAREST)
    cv2.putText(panel,f'f{frame}: candidates, NOT decisions',(4,18),0,.38,(255,255,255),1,cv2.LINE_AA)
    text=' / '.join(labels) if labels else 'NO candidate in active-ball region'
    cv2.putText(panel,text,(3,371),0,.3,(230,230,230),1,cv2.LINE_AA)
    yy,xx=(i//5)*385,(i%5)*340
    canvas[yy:yy+385,xx:xx+340]=panel
assert cv2.imwrite(str(WORK/'secondary-ball-active-crops-15.jpg'),canvas,[cv2.IMWRITE_JPEG_QUALITY,98])

# Context anchors and center of the rejected sideline selection.
context=np.zeros((430*2,640*3,3),dtype=np.uint8)
for i,frame in enumerate((9031,9039,9040,9041,9053,9054)):
    assert cap.set(cv2.CAP_PROP_POS_FRAMES,frame)
    ok,image=cap.read();assert ok
    panel=np.zeros((430,640,3),dtype=np.uint8)
    panel[26:386]=cv2.resize(image,(640,360))
    for j,c in enumerate(records[frame]['candidates']):
        x,y=c['point']
        if (1060<=x<1240 and 260<=y<465) or (1500<x<1700 and y>1020):
            p=tuple(np.int32(np.array([x,y])/3)+[0,26])
            cv2.circle(panel,p,7,(0,200,255),1,cv2.LINE_AA)
            cv2.putText(panel,str(j),(p[0]+7,p[1]),0,.4,(0,200,255),1,cv2.LINE_AA)
    cv2.putText(panel,f'f{frame}: exact source frame',(6,18),0,.45,(255,255,255),1,cv2.LINE_AA)
    cv2.putText(panel,'Orange labels identify detector candidates only.',(6,409),0,.4,(230,230,230),1,cv2.LINE_AA)
    yy,xx=(i//3)*430,(i%3)*640
    context[yy:yy+430,xx:xx+640]=panel
cap.release()
assert cv2.imwrite(str(WORK/'secondary-ball-full-context.jpg'),context,[cv2.IMWRITE_JPEG_QUALITY,96])
print('Prepared all15 exact-frame active-ball crops and six full-context frames; no outputs modified.')
