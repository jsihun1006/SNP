import csv
import json
from pathlib import Path
import cv2
import numpy as np

root=Path(__file__).resolve().parent
report=json.loads((root/'ranked_windows.json').read_text())
start=report['selected']['start_frame']
targets=[start+i*375 for i in range(12)]
points={f:[] for f in targets}
with Path(report['source']).open(newline='',encoding='utf-8-sig') as s:
    for row in csv.DictReader(s):
        frame=int(row['source_frame'])
        if frame in points:points[frame].append(row)
cap=cv2.VideoCapture(r'H:\data\broadcast\BAR-ATH_broadcast.mp4')
tiles=[]
for frame in targets:
    cap.set(cv2.CAP_PROP_POS_FRAMES,frame)
    ok,image=cap.read();assert ok
    for row in points[frame]:
        x,y=int(float(row['pixel_x'])),int(float(row['pixel_y']))
        score=float(row['confidence'])
        mark=False
        if row['pitch_x_m']:
            mx,my=float(row['pitch_x_m']),float(row['pitch_y_m'])
            mark=min(np.hypot(mx-11,my-34),np.hypot(mx-94.3,my-34))<=1.5
        color=(0,80,255) if mark else(0,255,255)
        cv2.circle(image,(x,y),12,color,3,cv2.LINE_AA)
        cv2.putText(image,f'{score:.2f}',(min(x+15,1800),max(20,y-12)),cv2.FONT_HERSHEY_SIMPLEX,.75,color,2,cv2.LINE_AA)
    image=cv2.resize(image,(640,360))
    cv2.rectangle(image,(0,0),(640,29),(0,0,0),-1)
    cv2.putText(image,f"Source f{frame} | +{(frame-start)/25:.0f}s | {len(points[frame])} candidates",(9,21),cv2.FONT_HERSHEY_SIMPLEX,.52,(255,255,255),1,cv2.LINE_AA)
    tiles.append(image)
cap.release()
grid=np.vstack([np.hstack(tiles[i:i+3]) for i in range(0,len(tiles),3)])
cv2.imwrite(str(root/'ranked-window-contact-sheet.jpg'),grid,[cv2.IMWRITE_JPEG_QUALITY,93])
print(root/'ranked-window-contact-sheet.jpg')
