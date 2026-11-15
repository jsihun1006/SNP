"""Read-only exact-frame review material for risky later selected observations."""
import collections
import gzip
import hashlib
import json
from pathlib import Path

import cv2
import numpy as np

WORK=Path(__file__).resolve().parent
ROOT=Path(r'H:\data\ball-tracking-results\first_half\correction_recall_v1\chunks')
# Chosen from a frozen read-only risk ranking: missing H, little green context,
# low confidence and >=100source-frame spacing. Includes new recovery29793.
frames=[15352,23808,25153,27431,29366,29793,32805,32963,33329,38232,38342,38573]
wanted=set(frames);rows={};sources={};stats=collections.Counter()
for folder in sorted(ROOT.iterdir(),key=lambda p:int(p.name)):
    if int(folder.name)<15171 or not(folder/'manifest.json').exists():continue
    source=folder/'decisions.jsonl.gz'
    manifest=json.loads((folder/'manifest.json').read_text(encoding='utf-8'))
    digest=hashlib.sha256(source.read_bytes()).hexdigest();assert digest==manifest['sha256']
    with gzip.open(source,'rt',encoding='utf-8') as stream:
        for row in map(json.loads,stream):
            selection=row['selection'];stats[selection['status']]+=1
            if selection['status']=='selected_observation':
                candidate=row['candidates'][selection['candidate_index']]
                assert [selection['pixel_x'],selection['pixel_y']]==candidate['point']
                stats['selected_missing_pitch']+=int('pitch' not in candidate)
                stats['selected_recovery']+=int(candidate.get('origin')!='original_detector')
                stats['selected_full_frame_redetection']+=int(candidate.get('origin')=='full_frame_redetection')
            if row['source_frame'] in wanted:
                rows[row['source_frame']]=row;sources[row['source_frame']]=(str(source),digest)
assert wanted==rows.keys()
cap=cv2.VideoCapture(r'H:\data\broadcast\BAR-ATH_broadcast.mp4')
panels=[];samples=[]
for frame in frames:
    row=rows[frame];selection=row['selection'];index=selection['candidate_index']
    assert selection['status']=='selected_observation'
    candidate=row['candidates'][index]
    raw=max(row['candidates'][:row['original_candidate_count']],key=lambda c:c['confidence'],default=None)
    assert cap.set(cv2.CAP_PROP_POS_FRAMES,frame);ok,image=cap.read();assert ok
    panel=np.zeros((470,640,3),np.uint8);panel[30:390]=cv2.resize(image,(640,360))
    point=tuple(np.int32(np.array(candidate['point'])/3)+[0,30]);cv2.circle(panel,point,9,(0,255,0),2,cv2.LINE_AA)
    if raw:
        p=tuple(np.int32(np.array(raw['point'])/3)+[0,30]);cv2.drawMarker(panel,p,(0,120,255),cv2.MARKER_TILTED_CROSS,15,2)
    def crop(point):
        x,y=np.int32(np.round(point));patch=image[max(0,y-24):min(1080,y+25),max(0,x-24):min(1920,x+25)]
        return cv2.resize(patch,(78,78),interpolation=cv2.INTER_NEAREST)
    panel[391:469,480:558]=crop(candidate['point'])
    if raw:panel[391:469,561:639]=crop(raw['point'])
    title=f'f{frame} {candidate["origin"]} c={candidate["confidence"]:.3f}'
    cv2.putText(panel,title,(6,21),0,.43,(255,255,255),1,cv2.LINE_AA)
    meta=f'noH={"pitch" not in candidate} green={candidate.get("green_fraction",0):.2f} review={selection["needs_review"]}'
    cv2.putText(panel,meta,(6,410),0,.4,(230,230,230),1,cv2.LINE_AA)
    cv2.putText(panel,'Green=selected; orange X=raw highest; crops: chosen/raw',(6,435),0,.35,(200,200,200),1,cv2.LINE_AA)
    panels.append(panel)
    samples.append(dict(source_frame=frame,selection=selection,candidate=candidate,
                        source_chunk_decisions=sources[frame][0],source_chunk_sha256=sources[frame][1],
                        visual_assessment='pending',manual_localization_ground_truth=False))
cap.release()
for page in range(3):
    canvas=np.zeros((940,1280,3),np.uint8)
    for j,panel in enumerate(panels[page*4:(page+1)*4]):
        y,x=(j//2)*470,(j%2)*640;canvas[y:y+470,x:x+640]=panel
    assert cv2.imwrite(str(WORK/f'full-run-late-review-{page+1}.jpg'),canvas,[cv2.IMWRITE_JPEG_QUALITY,97])
report=dict(status_counts=dict(stats),samples=samples,sampling='Risk-biased twelve selected observations after15171, prioritizing missing H, nongreen context and low confidence; includes recovered29793.',limitations='Sparse risk-biased visual QA, not a representative accuracy estimate or full-run ground truth.')
(WORK/'full-run-late-review.json').write_text(json.dumps(report,indent=2),encoding='utf-8')
print(json.dumps(dict(frames=frames,status_counts=dict(stats))))
