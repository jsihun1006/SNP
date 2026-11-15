"""Additional unlabelled diagnostics for a completed small V4 run."""
import argparse
from collections import Counter,defaultdict
import csv
import json
from pathlib import Path

import cv2
import numpy as np

from run_tryolabs_classification_v3 import FrameReader,VIDEO
from tryolabs_uniform_v2 import crop_region
from tryolabs_uniform_v3 import box_mask
from tryolabs_v4_data import ROOT,ASSETS,OUTPUT,digest,write_json


def analyze(run, frozen):
    out=run/"analysis";out.mkdir(exist_ok=False)
    objects=[json.loads(x) for x in (run/"objects.jsonl").read_text(encoding="utf-8").splitlines()]
    inputs=[json.loads(x) for x in (run/"input_frames.jsonl").read_text(encoding="utf-8").splitlines()]
    v4=json.loads((ROOT/"starter/configs/tryolabs_classification_v4_hsv.json").read_text(encoding="utf-8"))
    config=json.loads((ROOT/"starter/configs/tryolabs_classification_v3_hsv.json").read_text(encoding="utf-8"))
    frozen_labels={}
    if frozen.is_file():
        for line in frozen.read_text(encoding="utf-8").splitlines():
            r=json.loads(line);frozen_labels[r["sequence"],r["source_frame0"],r["track_id"]]=r["final_label"]
    compared=[o for o in objects if (o["sequence"],o["source_frame0"],o["track_id"]) in frozen_labels]
    sequences={}
    for seq in sorted({r["sequence"] for r in inputs}):
        oo=[o for o in objects if o["sequence"]==seq];ii=[r["frame0"] for r in inputs if r["sequence"]==seq]
        sequences[seq]=dict(first_frame0=min(ii),last_frame0=max(ii),frames=len(ii),objects=len(oo),
             v3_counts=dict(Counter(o["v3_label"] for o in oo)),v4_counts=dict(Counter(o["final_label"] for o in oo)))
    grouped=defaultdict(list)
    for o in objects:
        if o["goalkeeper_candidates"] and o["age"]==0:grouped[o["sequence"],o["track_id"]].append(o)
    reader=FrameReader(VIDEO);stats=[];measurements=[]
    try:
        for (seq,tid),rr in grouped.items():
            supports=[];shorts=[];hues=[];widths=[];steps=[];previous=None;samplepixels=[]
            for o in rr:
                im=reader.read(o["source_frame0"]);roi,_=crop_region(im,o["points"])
                hsv=cv2.cvtColor(roi,cv2.COLOR_BGR2HSV);name=o["goalkeeper_candidates"][0]
                pp=hsv[box_mask(hsv,v4["goalkeeper_kits"][name]["bounds"])]
                hue=float(np.median(pp[:,0])) if len(pp) else None
                if len(pp):samplepixels.append(pp[::max(1,len(pp)//64)])
                supports.append(o["goalkeeper_support"][name]);shorts.append(o["goalkeeper_shorts"][name]);hues.append(hue)
                p=np.asarray(o["points"]);center=p.mean(axis=0);widths.append(float(p[1,0]-p[0,0]))
                if previous is not None and previous[0]+1==o["source_frame0"]:steps.append(float(np.linalg.norm(center-previous[1])))
                previous=o["source_frame0"],center
                measurements.append(dict(sequence=seq,track_id=tid,frame0=o["source_frame0"],kit=name,
                    torso_support=supports[-1],shorts_support=shorts[-1],masked_hue_median=hue,
                    weak_goal_hint=o["goal_geometry_hint"],final_label=o["final_label"]))
            pp=np.concatenate(samplepixels) if samplepixels else np.zeros((0,3),np.uint8)
            overlaps={k:int(box_mask(pp.reshape(-1,1,3),v).sum()) for k,v in config["hsv_components"].items()} if len(pp) else {}
            stats.append(dict(sequence=seq,track_id=tid,candidate_observations=len(rr),
                final_counts=dict(Counter(o["final_label"] for o in rr)),
                first_frame0=rr[0]["source_frame0"],last_frame0=rr[-1]["source_frame0"],
                torso_support_q10_q50_q90=np.quantile(supports,[.1,.5,.9]).tolist(),
                shorts_support_q10_q50_q90=np.quantile(shorts,[.1,.5,.9]).tolist(),
                masked_hue_median_q10_q50_q90=np.quantile([h for h in hues if h is not None],[.1,.5,.9]).tolist(),
                bbox_width_median=float(np.median(widths)),
                raw_center_step_px_q10_q50_q90=np.quantile(steps,[.1,.5,.9]).tolist() if steps else [],
                motion_camera_compensated=False,weak_goal_hint_fraction=float(np.mean([o["goal_geometry_hint"] for o in rr])),
                sampled_teal_pixels=len(pp),sampled_teal_overlap_v3_components=overlaps,
                glove_evidence="NOT_RELIABLY_RESOLVED",person_identity_verified=False))
    finally:reader.close()
    if measurements:
        with (out/"goalkeeper_color_series.csv").open("x",encoding="utf-8",newline="") as f:
            w=csv.DictWriter(f,fieldnames=list(measurements[0]));w.writeheader();w.writerows(measurements)
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
        fig,axes=plt.subplots(3,1,figsize=(12,9))
        for seq,tid in grouped:
            mm=[o for o in measurements if o["sequence"]==seq and o["track_id"]==tid]
            x=[o["frame0"] for o in mm];name=f'{seq} / ID {tid}'
            axes[0].plot(x,[o["torso_support"] for o in mm],label=name)
            axes[1].plot(x,[o["shorts_support"] for o in mm],label=name)
            axes[2].plot(x,[o["masked_hue_median"] for o in mm],label=name)
        for ax,title in zip(axes,["Torso teal support", "Shorts teal support", "Masked hue median (appearance, not truth)"]):
            ax.set_title(title);ax.grid(alpha=.2);ax.legend(fontsize=8)
        fig.tight_layout();fig.savefig(out/"goalkeeper_color_stability.png",dpi=140);plt.close(fig)
    write_json(out/"diagnostics.json",dict(sequences=sequences,goalkeeper_tracks=stats,
        frozen_v3_compared=len(compared),frozen_v3_agreement=sum(o["v3_label"]==frozen_labels[o["sequence"],o["source_frame0"],o["track_id"]] for o in compared),
        no_human_ground_truth=True,analysis_code_sha256=digest(Path(__file__)),
        input_objects_sha256=digest(run/"objects.jsonl")))
    print(json.dumps(stats,indent=2))


if __name__=="__main__":
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument("--run",type=Path,default=OUTPUT/"validated")
    p.add_argument("--frozen-v3",type=Path,default=ASSETS/"classification_v3/validated/objects.jsonl")
    a=p.parse_args()
    if not a.run.resolve().is_relative_to(OUTPUT.resolve()):p.error("Use V4 run directory")
    analyze(a.run,a.frozen_v3)
