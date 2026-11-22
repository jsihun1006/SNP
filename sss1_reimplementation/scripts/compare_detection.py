import argparse
import json
import time
from pathlib import Path
import cv2
from ..detection import DetectionEngine, YOLODetector
from ..detection.yolo import official_sahi
from ..pipeline import Pipeline
from ..visualization import draw_predictions, draw_tiles
from ..image_io import read_image, write_image


def main():
    p=argparse.ArgumentParser(description="Actual detector comparison; no GT metrics are fabricated")
    p.add_argument("--image",type=Path,required=True)
    p.add_argument("--checkpoint",default="yolo26l.pt")
    p.add_argument("--device",default="cpu")
    p.add_argument("--output",type=Path,required=True)
    p.add_argument("--official-sahi",action="store_true")
    a=p.parse_args()
    image=read_image(a.image)
    if image is None: raise ValueError("Cannot read image")
    a.output.mkdir(parents=True,exist_ok=True)
    detector=YOLODetector(a.checkpoint,a.device)
    report={"input":str(a.image.resolve()),"image_shape":list(image.shape),"ground_truth_available":False,
            "purpose":"pretrained detector smoke test, not SynLoc performance reproduction","modes":{}}
    panels=[]
    for mode in ("full","sahi","baat"):
        engine=DetectionEngine(detector,mode=mode)
        rows,timing=Pipeline(engine).infer(image,0)
        report["modes"][mode]=dict(detections=len(rows),**timing)
        rendered=draw_predictions(image,rows)
        write_image(a.output/f"{mode}.jpg",rendered)
        write_image(a.output/f"{mode}_tiles.jpg",draw_tiles(image,engine.last_tiles,[d.bbox for d in engine.last_coarse]))
        (a.output/f"{mode}.json").write_text(json.dumps(rows,allow_nan=False,indent=2),encoding="utf-8")
        panel=cv2.resize(rendered,(640,round(640*image.shape[0]/image.shape[1])))
        cv2.putText(panel,mode.upper(),(12,30),cv2.FONT_HERSHEY_SIMPLEX,.8,(0,0,255),2)
        panels.append(panel)
    if a.official_sahi:
        start=time.perf_counter()
        detections=official_sahi(image,a.checkpoint,a.device)
        report["modes"]["official_sahi"]=dict(detections=len(detections),seconds=time.perf_counter()-start,
            boxes=[dict(bbox=d.bbox,confidence=d.confidence) for d in detections])
    import numpy as np
    write_image(a.output/"comparison.jpg",np.concatenate(panels,axis=1))
    (a.output/"comparison.json").write_text(json.dumps(report,allow_nan=False,indent=2),encoding="utf-8")
    print(json.dumps(report,indent=2))


if __name__=="__main__": main()
