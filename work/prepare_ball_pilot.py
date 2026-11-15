import csv,json,sys,time
from pathlib import Path
import cv2
import numpy as np

BASE=Path(__file__).resolve().parent.parent
WORK=BASE/'work/ball_pilot';WORK.mkdir(exist_ok=True)
chosen=json.loads((BASE/'work/ranked_windows.json').read_text())['selected']
START,END=chosen['start_frame'],chosen['end_frame_exclusive'];N=END-START
records=[]
with Path(r'H:\data\ball-tracking-results\first_half\candidates.jsonl').open(encoding='utf-8') as f:
    for line in f:
        row=json.loads(line)
        if row['source_frame']>=END:break
        if row['source_frame']>=START:
            records.append(dict(frame=row['source_frame'],candidates=[{k:c[k] for k in ('point','bbox_xyxy','confidence','source','sources','source_observations')} for c in row['candidates']]))
assert len(records)==N
H=np.full((N,3,3),np.nan);errors=np.full(N,np.nan)
with Path(r'H:\data\broadcast\BAR-ATH_broadcast_homography.csv').open(encoding='utf-8') as f:
    for row in csv.DictReader(f):
        frame=int(row['frame'])
        if frame>=END:break
        if frame>=START and row['valid']=='1':
            H[frame-START]=np.array([float(row[f'h{i}{j}']) for i in range(3) for j in range(3)]).reshape(3,3)
            errors[frame-START]=float(row['reprojection_error'])
ball=np.full((N,2),np.nan);players=[]
with Path(r'H:\data\broadcast\BAR-ATH_broadcast_atd.txt').open(encoding='utf-8') as f:
    for frame,line in enumerate(f):
        if frame>=END:break
        if frame>=START:
            parts=line.strip().split(':');assert int(parts[0])==frame+1
            ball[frame-START]=np.fromstring(parts[2],sep=',')[:2]
            players.append(np.fromstring(parts[1].replace(';',','),sep=',').reshape(-1,3)[:,:2])
prior=np.full((N,2),np.nan)
for i in range(N):
    if np.isfinite(H[i]).all() and np.isfinite(ball[i]).all():
        q=H[i]@np.r_[ball[i],1]
        if abs(q[2])>1e-10:prior[i]=q[:2]/q[2]
    if np.isfinite(H[i]).all():
        inv=np.linalg.inv(H[i])
        for c in records[i]['candidates']:
            q=inv@np.r_[c['point'],1];p=q[:2]/q[2]
            c['pitch']=p.tolist()
            c['penalty_spot_distance_m']=min(np.linalg.norm((p-[11/105.3,.5])*[105.3,68]),np.linalg.norm((p-[1-11/105.3,.5])*[105.3,68]))
    for c in records[i]['candidates']:
        c['atd_distance_px']=float(np.linalg.norm(np.asarray(c['point'])-prior[i])) if np.isfinite(prior[i]).all() else None

cap=cv2.VideoCapture(r'H:\data\broadcast\BAR-ATH_broadcast.mp4');assert cap.set(cv2.CAP_PROP_POS_FRAMES,START)
warps=np.tile(np.eye(2,3),(N,1,1));cuts=np.zeros(N,dtype=bool);previous=None
sheet=[];quick=[]
for i in range(N):
    ok,img=cap.read();assert ok
    small=cv2.resize(img,(480,270));gray=cv2.cvtColor(small,cv2.COLOR_BGR2GRAY)
    if previous is not None:
        points=cv2.goodFeaturesToTrack(previous,80,.02,12)
        if points is not None:
            nxt,status,_=cv2.calcOpticalFlowPyrLK(previous,gray,points,None,winSize=(21,21),maxLevel=2)
            valid=status[:,0].astype(bool)
            if valid.sum()>=10:
                m,inliers=cv2.estimateAffinePartial2D(points[valid],nxt[valid],method=cv2.RANSAC,ransacReprojThreshold=2)
                if m is not None and inliers.sum()>=10:
                    aligned=cv2.warpAffine(previous,m,(480,270))
                    diff=np.mean(cv2.absdiff(aligned[20:-20,20:-20],gray[20:-20,20:-20]))
                    if diff<25 and .90<np.linalg.det(m[:,:2])<1.12:
                        m[:,2]*=4;warps[i]=m
                    else:cuts[i]=True
                else:cuts[i]=True
            else:cuts[i]=True
        else:cuts[i]=True
    previous=gray
    for c in records[i]['candidates']:
        x,y=map(int,np.round(c['point']));r=14
        crop=img[max(0,y-r):min(1080,y+r+1),max(0,x-r):min(1920,x+r+1)]
        if crop.size:
            hsv=cv2.cvtColor(crop,cv2.COLOR_BGR2HSV)
            c['green_fraction']=float(np.mean((hsv[:,:,0]>=30)&(hsv[:,:,0]<=95)&(hsv[:,:,1]>45)&(hsv[:,:,2]>35)))
        else:c['green_fraction']=0.
    if i%125==0:
        shown=cv2.resize(img,(640,360))
        for ci,c in enumerate(records[i]['candidates']):
            p=tuple(np.int32(np.asarray(c['point'])/3));cv2.circle(shown,p,5,(0,180,255),1,cv2.LINE_AA)
            cv2.putText(shown,f'{ci}:{c["confidence"]:.2f}',(min(565,p[0]+5),max(25,p[1]-5)),cv2.FONT_HERSHEY_SIMPLEX,.38,(0,180,255),1,cv2.LINE_AA)
        if np.isfinite(prior[i]).all():cv2.drawMarker(shown,tuple(np.int32(prior[i]/3)),(255,180,30),cv2.MARKER_CROSS,14,1)
        cv2.putText(shown,f'frame {START+i} | {i/25:.0f}s',(10,20),cv2.FONT_HERSHEY_SIMPLEX,.5,(255,255,255),1,cv2.LINE_AA)
        sheet.append(shown)
    if i%500==0:print(json.dumps(dict(prepared=i,total=N)),flush=True)
cap.release()
np.savez_compressed(WORK/'context.npz',H=H,h_error=errors,atd_ball=ball,atd_pixel=prior,warps=warps,cuts=cuts)
(WORK/'candidates.json').write_text(json.dumps(dict(start=START,end=END,records=records)),encoding='utf-8')
for page in range((len(sheet)+11)//12):
    images=sheet[page*12:(page+1)*12]
    canvas=np.zeros((360*4,640*3,3),np.uint8)
    for j,image in enumerate(images):canvas[(j//3)*360:(j//3+1)*360,(j%3)*640:(j%3+1)*640]=image
    cv2.imwrite(str(WORK/f'overview-{page+1}.jpg'),canvas,[cv2.IMWRITE_JPEG_QUALITY,92])
print(json.dumps(dict(prepared=N,camera_cut_candidates=int(cuts.sum()),h_frames=int(np.isfinite(H).all(axis=(1,2)).sum()),prior_frames=int(np.isfinite(prior).all(axis=1).sum()))),flush=True)
