"""Measure V3 color boxes from unlabelled cached broadcast torso pixels.

No object labels, model inference or tracking updates. Official color descriptions
name appearance hypotheses; automatically selected anchors are NOT ground truth.
"""
import argparse
import json
from pathlib import Path

import numpy as np

from tryolabs_v3_data import ASSETS,ROOT,OUTPUT,digest,write_json


def derive(output,source=ASSETS/"classification_v2"):
    source=Path(source);output=Path(output);output.mkdir(parents=True,exist_ok=False)
    pixels=np.load(source/"torso_pixel_samples.npz")["hsv"]
    records=[json.loads(x) for x in (source/"pixel_sample_sources.jsonl").read_text(encoding="utf-8").splitlines()]
    old=json.loads((ROOT/"starter/configs/tryolabs_classification_v2_hsv.json").read_text())
    from tryolabs_uniform_v2 import mask_range
    masks={k:mask_range(pixels.reshape(-1,1,3),v).ravel() for k,v in old["hsv_components"].items()}
    anchors={"TEAM_A":[],"TEAM_B":[],"REFEREE":[]}
    for r in records:
        a,b=r["sample_start"],r["sample_start"]+r["sample_count"]
        s={k:float(v[a:b].mean()) for k,v in masks.items()}
        # Reuse V2 support/secondary-color gates to find unlabelled appearance
        # cohorts; no final V2 label or human team assignment is consumed.
        if min(s["navy"],s["burgundy"])>=old["quality"]["minimum_secondary_color_fraction"] and s["navy"]+s["burgundy"]>=old["decision"]["TEAM_A"]["minimum_support"] and s["lilac"]<.1:
            anchors["TEAM_A"].append(np.arange(a,b))
        if s["lilac"]>=old["decision"]["TEAM_B"]["minimum_support"] and s["navy"]<.025 and s["burgundy"]<.1:
            anchors["TEAM_B"].append(np.arange(a,b))
        if s["yellow"]>=old["decision"]["REFEREE"]["minimum_support"] and s["navy"]+s["lilac"]<.1:
            anchors["REFEREE"].append(np.arange(a,b))
    cohort={k:pixels[np.concatenate(v)] for k,v in anchors.items()}
    components={};measurements={}
    def bounds(name,pp,q=(.01,.99)):
        if len(pp)<100:raise ValueError(f"Too few measured pixels for {name}: {len(pp)}")
        lo,hi=np.quantile(pp,q,axis=0)
        item=dict(lower=np.floor(lo).astype(int).tolist(),upper=np.ceil(hi).astype(int).tolist())
        measurements[name]=dict(pixels=len(pp),quantiles=list(q),median=np.median(pp,axis=0).tolist(),**item)
        return item
    # Broad hue selection is a color hypothesis. Numerical final box endpoints
    # are measured in broadcast pixels; S/V are not official-kit color codes.
    a=cohort["TEAM_A"];b=cohort["TEAM_B"];y=cohort["REFEREE"]
    components["navy"]=[bounds("navy",a[(a[:,0]>=100)&(a[:,0]<=135)&(a[:,1]>=80)&(a[:,2]<=190)])]
    components["burgundy"]=[bounds("burgundy_lit",a[(a[:,0]>=155)&(a[:,0]<=179)&(a[:,1]>=130)&(a[:,2]>=80)]),
                              bounds("burgundy_shadow",a[(a[:,0]>=135)&(a[:,0]<=160)&(a[:,1]>=90)&(a[:,2]<170)],(.1,.9))]
    components["lavender"]=[bounds("lavender",b[(b[:,0]>=120)&(b[:,0]<=160)&(b[:,1]>=30)&(b[:,1]<=130)&(b[:,2]>=120)])]
    pale=b[(b[:,0]>=125)&(b[:,0]<=160)&(b[:,1]>=8)&(b[:,1]<60)&(b[:,2]>=200)]
    # Lower S uses the measured 25th percentile: low-chroma whites are not all
    # included simply to increase coverage. Hue coherence remains necessary.
    pale_box=bounds("pale_lavender",pale,(.05,.99));pale_box["lower"][1]=int(np.floor(np.quantile(pale[:,1],.25)))
    measurements["pale_lavender"]["lower"]=pale_box["lower"]
    measurements["pale_lavender"]["saturation_lower_quantile"]=.25
    components["lavender"].append(pale_box)
    components["pink"]=[bounds("pink_auxiliary",b[(b[:,0]>=152)&(b[:,0]<=179)&(b[:,1]>=30)&(b[:,1]<=150)&(b[:,2]>=100)],(.05,.95))]
    components["yellow"]=[bounds("fluorescent_yellow",y[(y[:,0]>=25)&(y[:,0]<=45)&(y[:,1]>=170)&(y[:,2]>=170)])]
    config=dict(schema_version=3,hsv_components=components,lower_dark_value_max=old["lower_dark_value_max"],
                provenance=dict(method="Measured unlabelled appearance-cohort pixel quantiles, not manual team labels",
                                source_pixel_sha256=digest(source/"torso_pixel_samples.npz"),source_records_sha256=digest(source/"pixel_sample_sources.jsonl"),
                                source_frames=750,independent_evaluation=False,anchor_observations={k:len(v) for k,v in anchors.items()},measurements=measurements))
    write_json(output/"hsv_config.json",config)
    shared=masks["burgundy"]&masks["lilac"]
    diagnostics=dict(sampled_pixels=len(pixels),v2_burgundy_lilac_shared_pixels=int(shared.sum()),
                     v2_lilac_value_gap_199_202_pixels=int(((pixels[:,2]>=199)&(pixels[:,2]<=202)&(pixels[:,0]>=125)&(pixels[:,0]<=156)).sum()),
                     v2_palette_missing_pale_pixels=int(((pixels[:,0]>=125)&(pixels[:,0]<=160)&(pixels[:,1]>=8)&(pixels[:,1]<51)&(pixels[:,2]>=203)&~masks["lilac"]).sum()),
                     anchors={k:len(v) for k,v in anchors.items()},measurements=measurements,note="Pixel/appearance diagnostics only, not class-conditional ground truth")
    write_json(output/"calibration_diagnostics.json",diagnostics)
    print(json.dumps(config,indent=2));return config


if __name__=="__main__":
    p=argparse.ArgumentParser(description=__doc__);p.add_argument("--output",type=Path,default=OUTPUT/"calibration");p.add_argument("--source",type=Path,default=ASSETS/"classification_v2")
    args=p.parse_args()
    if not args.output.resolve().is_relative_to(OUTPUT.resolve()):p.error("Output must be under classification_v3")
    derive(args.output,args.source)
