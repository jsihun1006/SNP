"""Synthetic end-to-end I/O check. Contains no learned detections."""
import json
from pathlib import Path
import cv2
import numpy as np
from .io import csv_write,json_write
from .pipeline import run


def smoke(upstream,output):
    output=Path(output).resolve();output.mkdir(parents=True,exist_ok=False)
    video=output/'synthetic.avi'
    writer=cv2.VideoWriter(str(video),cv2.VideoWriter_fourcc(*'MJPG'),25,(320,180))
    if not writer.isOpened():raise ValueError('MJPG encoder unavailable')
    calibration=[];replay=[]
    try:
        for i in range(24):
            frame=np.full((180,320,3),(30,90,30),dtype=np.uint8)
            box=[40+i,40,60+i,100,.95]
            cv2.rectangle(frame,(40+i,40),(60+i,100),(230,230,230),-1)
            writer.write(frame)
            h=np.eye(3) if i==5 else np.array([[300,0,10],[0,140,20],[0,0,1.]])
            row={'frame':i,'valid':1,'reprojection_error':''}
            row.update({'h%d%d'%(r,c):h[r,c] for r in range(3) for c in range(3)})
            calibration.append(row)
            replay.append(dict(frame_idx=i,detections=[] if 7<=i<=9 else [box],features=[] if 7<=i<=9 else [[1.,0.]],
                attributes=[] if 7<=i<=9 else [dict(role='Player',jersey=8,color='white')]))
    finally:writer.release()
    cal=output/'calibration.csv'
    csv_write(cal,calibration,['frame']+['h%d%d'%(r,c) for r in range(3) for c in range(3)]+['reprojection_error','valid'])
    predictions=output/'synthetic_detections.jsonl'
    predictions.write_text(''.join(json.dumps(r)+'\n' for r in replay),encoding='utf-8')
    cfg=dict(video=str(video),calibration_csv=str(cal),upstream=str(Path(upstream).resolve()),
        replay=str(predictions),output_root=str(output/'runs'),start_frame=0,max_frames=24,
        pitch_length_m=105.3,pitch_width_m=68.,scene_type='unverified',write_overlay=True,
        cut_threshold=1.1,cut_frames=[12],team_colors={},refine=True)
    path=run(cfg,backend='replay')
    json_write(output/'smoke.json',dict(status='complete',run=str(path),synthetic=True,model_inference_executed=False,
        windows_validated=False,performance_validated=False))
    return path
