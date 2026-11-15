"""Measure teal appearance, without match labels or track/frame allowlists."""
import argparse
from collections import defaultdict
import json
from pathlib import Path

import cv2
import numpy as np

from run_tryolabs_classification_v3 import FrameReader, VIDEO
from tryolabs_uniform_v2 import crop_region
from tryolabs_v4_data import ROOT, OUTPUT, digest, write_json
from tryolabs_v4_context import frame_context


def derive(source, out):
    out.mkdir(parents=True, exist_ok=False)
    records = [json.loads(x) for x in (source/"candidates.jsonl").read_text(encoding="utf-8").splitlines()]
    grouped = defaultdict(list)
    for r in records:
        if r["search_color"] == "teal" and r["torso_support"]["teal"] >= .5 and r["shorts_support"] >= .30:
            grouped[r["sequence"], r["track_id"]].append(r)
    reader = FrameReader(VIDEO); pixels = []; accepted = []; rng = np.random.default_rng(40)
    try:
        for key, rows in grouped.items():
            if len(rows) < 10:
                continue
            for r in rows:
                im = reader.read(r["frame0"])
                # Green coverage only selects an unsupervised appearance cohort;
                # it does not establish pitch position, team or goalkeeper truth.
                ctx = frame_context(im, [])
                if ctx["green_coverage"] < .35:
                    continue
                roi, _ = crop_region(im, r["points"])
                hsv = cv2.cvtColor(roi, cv2.COLOR_BGR2HSV)
                pp = hsv[cv2.inRange(hsv,(75,55,45),(110,255,255)).astype(bool)]
                pixels.append(pp[rng.choice(len(pp), min(256,len(pp)), replace=False)])
                accepted.append({k:r[k] for k in ["sequence","track_id","frame0"]})
    finally:
        reader.close()
    if not pixels:
        raise ValueError("No sufficiently persistent teal cohort; goalkeeper config remains unconfirmed")
    pp = np.concatenate(pixels); lo, hi = np.quantile(pp,[.01,.99],axis=0)
    bounds = dict(lower=np.floor(lo).astype(int).tolist(),upper=np.ceil(hi).astype(int).tolist())
    v3path = ROOT/"starter/configs/tryolabs_classification_v3_hsv.json"
    old = json.loads(v3path.read_text(encoding="utf-8"))
    config = dict(schema_version=4,base_v3_hsv_components=old["hsv_components"],
        goalkeeper_kits={"GK_A":dict(enabled=False,bounds=[],reason="Match kit not independently established; orange/green are search hypotheses only"),
                         "GK_B":dict(enabled=True,bounds=[bounds],color="teal_appearance_hypothesis",
                                     team_assignment_source="User-provided Athletic teal hypothesis; person identity not verified")},
        unresolved_search={"orange":dict(lower=[0,100,90],upper=[25,255,255]),"green":dict(lower=[35,70,45],upper=[74,255,255])},
        provenance=dict(method="Automatic persistent teal torso+shorts appearance cohort; no manual labels",observations=len(accepted),
            sampled_pixels=len(pp),quantiles=[.01,.99],hsv_median=np.median(pp,axis=0).tolist(),
            candidate_records_sha256=digest(source/"candidates.jsonl"),base_v3_sha256=digest(v3path),
            role_or_identity_ground_truth=False,independent_evaluation=False))
    write_json(out/"hsv_config.json",config);write_json(out/"cohort_sources.json",accepted)
    np.savez_compressed(out/"teal_pixels.npz",hsv=pp)
    print(json.dumps(config["provenance"],indent=2));print(bounds)
    return config


if __name__ == "__main__":
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument("--source",type=Path,default=OUTPUT/"discovery")
    p.add_argument("--output",type=Path,default=OUTPUT/"calibration")
    a=p.parse_args()
    if not a.output.resolve().is_relative_to(OUTPUT.resolve()):p.error("Use classification_v4 output")
    derive(a.source,a.output)
