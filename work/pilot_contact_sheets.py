import json
from pathlib import Path
import cv2,numpy as np
ROOT=Path(__file__).resolve().parent
WORK=ROOT/'ball_pilot'
doc=json.loads((WORK/'candidates_enriched.json' if (WORK/'candidates_enriched.json').exists() else WORK/'candidates.json').read_text())
track=json.loads((WORK/'corrected.json').read_text())['records']
sample=list(range(0,4500,150))
for frame in (19294,19419,19544,21169,21544,22669):
    if frame-doc['start'] not in sample:sample.append(frame-doc['start'])
sample.sort();tiles=[];metadata=[]
cap=cv2.VideoCapture(r'H:\data\broadcast\BAR-ATH_broadcast.mp4')
for i in sample:
    assert cap.set(cv2.CAP_PROP_POS_FRAMES,doc['start']+i)
    ok,image=cap.read();assert ok
    t=track[i];r=doc['records'][i]
    tile=cv2.resize(image,(640,360))
    for ci,c in enumerate(r['candidates']):
        p=tuple(np.int32(np.asarray(c['point'])/3));cv2.circle(tile,p,4,(0,140,255),1,cv2.LINE_AA)
    if t['pixel_x'] is not None:
        px,py=t['pixel_x'],t['pixel_y'];p=tuple(np.int32(np.asarray([px,py])/3))
        cv2.circle(tile,p,9,(30,255,70),2,cv2.LINE_AA)
        cv2.line(tile,(p[0],max(24,p[1]-30)),(p[0],p[1]-10),(30,255,70),2)
        x,y=map(int,[px,py]);d=48
        crop=image[max(0,y-d):min(1080,y+d),max(0,x-d):min(1920,x+d)]
        if crop.size:
            crop=cv2.resize(crop,(96,96));cv2.rectangle(crop,(0,0),(95,95),(30,255,70),2);tile[25:121,540:636]=crop
    elif t['status']=='review_candidate':
        px,py=t['provisional_pixel_x'],t['provisional_pixel_y'];p=tuple(np.int32(np.asarray([px,py])/3));cv2.circle(tile,p,8,(40,170,255),2)
    cv2.rectangle(tile,(0,0),(640,24),(0,0,0),-1)
    label=f"f{r['frame']} +{i/25:.0f}s {t['status']}"
    if t['candidate_confidence'] is not None:label+=f" {t['candidate_confidence']:.2f}"
    cv2.putText(tile,label,(7,17),cv2.FONT_HERSHEY_SIMPLEX,.44,(255,255,255),1,cv2.LINE_AA)
    tiles.append(tile);metadata.append(dict(frame=r['frame'],status=t['status'],pixel=[t['pixel_x'],t['pixel_y']],confidence=t['candidate_confidence']))
cap.release()
for page in range((len(tiles)+11)//12):
    canvas=np.zeros((1440,1920,3),np.uint8)
    for j,img in enumerate(tiles[page*12:(page+1)*12]):canvas[(j//3)*360:(j//3+1)*360,(j%3)*640:(j%3+1)*640]=img
    cv2.imwrite(str(WORK/f'corrected-review-{page+1}.jpg'),canvas,[cv2.IMWRITE_JPEG_QUALITY,94])
(WORK/'review-samples.json').write_text(json.dumps(metadata,indent=2))
print(json.dumps(dict(samples=len(sample),pages=(len(tiles)+11)//12)))
