"""Synthetic diagnostic figure only; not athlete localization or benchmark data."""
from pathlib import Path
import argparse
import json
import cv2
import numpy as np
from ..tiling import adaptive_tiles, regular_grid
from ..visualization import draw_tiles
from ..image_io import write_image


def main():
    p=argparse.ArgumentParser()
    p.add_argument("--output",type=Path,required=True)
    a=p.parse_args();a.output.mkdir(parents=True,exist_ok=True)
    image=np.full((700,1500,3),245,np.uint8)
    boxes=[(490,150,540,350),(535,180,590,370),(585,230,680,390),(990,440,1060,590)]
    panels=[]
    for mode,tiles in [("SAHI regular grid",regular_grid(1500,700,500)),
                       ("BAAT coarse-box expansion",adaptive_tiles(1500,700,boxes,500))]:
        panel=draw_tiles(image,tiles,boxes)
        cv2.putText(panel,"SYNTHETIC - "+mode,(15,35),cv2.FONT_HERSHEY_SIMPLEX,.8,(0,0,0),2)
        panels.append(panel)
    write_image(a.output/"synthetic_tiling_comparison.png",np.concatenate(panels,axis=0))
    (a.output/"synthetic_camera.json").write_text(json.dumps(dict(model="opencv",width=2000,height=1200,
        K=[[500,0,1000],[0,500,600],[0,0,1]],R=[[1,0,0],[0,-1,0],[0,0,-1]],t=[0,0,10],
        distortion=[.08,-.01,.001,-.002,.001],pitch_bounds=[-30,-20,30,20]),indent=2),encoding="utf-8")


if __name__=="__main__": main()
