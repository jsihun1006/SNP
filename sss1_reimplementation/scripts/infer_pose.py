import argparse
import json
from pathlib import Path
import cv2
from ..types import Detection
from ..pose.inference import MMPoseEstimator
from ..image_io import read_image


def main():
    p=argparse.ArgumentParser(description="Two-point pose inference on provided original-image xyxy boxes")
    p.add_argument("--image",type=Path,required=True)
    p.add_argument("--boxes",type=Path,required=True,help='JSON list: {"bbox":[x1,y1,x2,y2],"confidence":0.9}')
    p.add_argument("--config",type=Path,default=Path(__file__).resolve().parents[1]/"configs/rtmpose_x_two_points.py")
    p.add_argument("--checkpoint",type=Path,required=True)
    p.add_argument("--device",default="cpu")
    p.add_argument("--output",type=Path,required=True)
    a=p.parse_args()
    image=read_image(a.image)
    if image is None: raise ValueError("Unreadable image")
    boxes=[Detection(tuple(d["bbox"]),float(d["confidence"])) for d in json.loads(a.boxes.read_text(encoding="utf-8"))]
    points=MMPoseEstimator(a.config,a.checkpoint,a.device).predict(image,boxes)
    a.output.parent.mkdir(parents=True,exist_ok=True)
    a.output.write_text(json.dumps(dict(keypoint_order=["pelvis","ground"],coordinate_system="original-image pixels",
                                       keypoints=points.tolist()),allow_nan=False,indent=2),encoding="utf-8")


if __name__=="__main__": main()
