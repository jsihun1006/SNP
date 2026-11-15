import json,sys,time,math
from pathlib import Path
import cv2
import numpy as np
ROOT=Path(__file__).resolve().parent
sys.path.insert(0,str(ROOT/'source/SNP-ball-tracking'))
from ball_only.run import environment
environment()
from ball_only.runtime_control import RuntimeController
import torch
from ultralytics import YOLO,settings
settings.update({'sync':False})
WORK=ROOT/'ball_pilot'
doc=json.loads((WORK/'candidates.json').read_text());ctx=np.load(WORK/'context.npz')
initial=json.loads((WORK/'corrected.json').read_text())['records']
targets=[]
for i,r in enumerate(doc['records']):
    p=ctx['atd_pixel'][i];err=ctx['h_error'][i]
    if np.isfinite(p).all() and 0<=p[0]<1920 and 0<=p[1]<1080 and 0<=err<=2:
        ds=[c['atd_distance_px'] for c in r['candidates'] if c['atd_distance_px'] is not None]
        if initial[i]['status']=='unresolved' or not ds or min(ds)>40:targets.append(i)
targetset=set(targets)
model=YOLO(str(ROOT/'source/SNP-ball-tracking/ball_only/weights/yolo26x.pt'))
controller=RuntimeController(target_low=98,target_high=99.5)
cap=cv2.VideoCapture(r'H:\data\broadcast\BAR-ATH_broadcast.mp4');assert cap.set(cv2.CAP_PROP_POS_FRAMES,doc['start'])
jobs=[];completed=0;added=0
def process():
    global completed,added
    if not jobs:return
    with controller.gpu_work():
        predictions=model.predict([x[1] for x in jobs],device=0,imgsz=1280,quantize=32,conf=.1,classes=[32],tile=False,rect=True,save=False,verbose=False,batch=len(jobs))
        boxes=[p.boxes.data.float().cpu().numpy() for p in predictions]
    assert not model.predictor.model.fp16
    for (i,crop,offset),rows in zip(jobs,boxes):
        for box in rows:
            x1,y1,x2,y2,confidence,cls=map(float,box)
            if int(cls)!=32:continue
            xyxy=np.asarray([x1,y1,x2,y2])+[offset[0],offset[1],offset[0],offset[1]]
            point=[float((xyxy[0]+xyxy[2])/2),float((xyxy[1]+xyxy[3])/2)]
            if min(x2-x1,y2-y1)<1:continue
            if any(np.linalg.norm(np.asarray(c['point'])-point)<2 and c['confidence']>=confidence for c in doc['records'][i]['candidates']):continue
            c=dict(point=point,bbox_xyxy=xyxy.tolist(),confidence=confidence,source='YOLO26x_TARGETED_512_1280',sources={'YOLO26x_TARGETED_512_1280':confidence},source_observations=[],origin='targeted_redetection',atd_distance_px=float(np.linalg.norm(np.asarray(point)-ctx['atd_pixel'][i])))
            inv=np.linalg.inv(ctx['H'][i]);q=inv@np.r_[point,1.];p=q[:2]/q[2]
            c['pitch']=p.tolist();c['penalty_spot_distance_m']=float(min(np.linalg.norm((p-[11/105.3,.5])*[105.3,68]),np.linalg.norm((p-[1-11/105.3,.5])*[105.3,68])))
            px,py=point[0]-offset[0],point[1]-offset[1];x,y=map(int,[px,py]);r=14
            patch=crop[max(0,y-r):min(512,y+r+1),max(0,x-r):min(512,x+r+1)]
            if patch.size:
                hsv=cv2.cvtColor(patch,cv2.COLOR_BGR2HSV);c['green_fraction']=float(np.mean((hsv[:,:,0]>=30)&(hsv[:,:,0]<=95)&(hsv[:,:,1]>45)&(hsv[:,:,2]>35)))
            else:c['green_fraction']=0.
            doc['records'][i]['candidates'].append(c);added+=1
        completed+=1
    jobs.clear()
for i in range(doc['end']-doc['start']):
    ok,image=cap.read();assert ok
    if i in targetset:
        px,py=ctx['atd_pixel'][i];x=max(0,min(1920-512,int(round(px-256))));y=max(0,min(1080-512,int(round(py-256))))
        jobs.append((i,np.ascontiguousarray(image[y:y+512,x:x+512]),(x,y)))
        if len(jobs)==4:process()
    if i%500==0:print(json.dumps(dict(frame=i,targets=len(targets),processed=completed,new_candidates=added)),flush=True)
process();cap.release()
(WORK/'candidates_enriched.json').write_text(json.dumps(doc),encoding='utf-8')
(WORK/'recovery_report.json').write_text(json.dumps(dict(target_frames=len(targets),new_candidates=added,model='same official YOLO26x FP32',crop=512,input_size=1280,batch=4,prior='provided broadcast ATD; search-region hint only, not truth'),indent=2))
print(json.dumps(dict(complete=True,targets=len(targets),new_candidates=added)),flush=True)
