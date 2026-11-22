"""Supplementary equal-support score comparison; no fitting or feature changes."""
import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd

from mtah_relational.frame_store import open_frame_cache
from mtah_relational.splits import SplitManifest
from .diagnostics import validation_supports,perturb
from .quality import observation_row
from .detector import compute_evidence
from .local_features import extract_local_frame
from .local_diagnostics import distribution
from .artifacts import json_write


def equal_support_change(clean,changed,local_name,subset=None):
    if not np.array_equal(clean.frame,changed.frame):
        raise ValueError("Equal-support comparison requires the same frame axis")
    selected=np.ones(len(clean),bool) if subset is None else np.asarray(subset,dtype=bool)
    joint=selected.copy()
    for name in ("v1_score",local_name):
        joint &= np.isfinite(clean[name].to_numpy()) & np.isfinite(changed[name].to_numpy())
    return dict(n=int(joint.sum()),selected_frames=int(selected.sum()),
                v1_absolute_change=distribution(np.abs(clean.v1_score.to_numpy()[joint]-changed.v1_score.to_numpy()[joint])),
                local_absolute_change=distribution(np.abs(clean[local_name].to_numpy()[joint]-changed[local_name].to_numpy()[joint])))


def materialize_condition(store,manifest,config,kind,strength):
    parts=[]
    for first,last in validation_supports(manifest):
        local,geometry=[],[]
        for offset,fid in enumerate(range(first,last+1)):
            frame=perturb(store.frame(store.position(fid)),kind,strength,seed=config["seed"],active=65<=offset<=85)
            summary,_,_=extract_local_frame(frame,store.pitch,config)
            summary["perturbed_span"]=65<=offset<=85
            local.append(summary);geometry.append(observation_row(frame,store.pitch))
        t=pd.DataFrame(local)
        t["v1_score"]=compute_evidence(pd.DataFrame(geometry),manifest).candidate_score.to_numpy()
        parts.append(t)
    return pd.concat(parts,ignore_index=True)


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument("--run-dir",required=True)
    p.add_argument("--cache",default="data/derived/relational_temporal_v1_20261109/raw_cache")
    p.add_argument("--split-manifest",default="data/derived/relational_temporal_v1_20261109/split_manifest.json")
    a=p.parse_args();run=Path(a.run_dir).resolve()
    if not run.is_relative_to(Path.cwd()/"data/derived/competition_judgement"):
        raise ValueError("Local-only output path required")
    output=run/"equal_support_robustness.json"
    if output.exists():raise FileExistsError("Supplement already exists")
    config=json.loads((run/"preregistered_config.json").read_text())
    store=open_frame_cache(a.cache);manifest=SplitManifest.load(a.split_manifest)
    clean=materialize_condition(store,manifest,config,"clean",0.)
    reports={}
    for kind,strength in (("jitter",.1),("jitter",.5),("jitter",1.),("dropout",.1),("dropout",.2),("dropout",.3),("composition",.2)):
        changed=materialize_condition(store,manifest,config,kind,strength)
        result={str(r):equal_support_change(clean,changed,f"max_local_contact_strength_{r}m") for r in (3,5)}
        if kind=="composition":
            result["active_span_only"]={str(r):equal_support_change(clean,changed,f"max_local_contact_strength_{r}m",clean.perturbed_span.to_numpy()) for r in (3,5)}
        reports[f"{kind}_{strength:g}"]=result
    json_write(output,dict(role="validation",conditions=reports,config_changed=False,threshold_refit=False,
                          interpretation="Score ranges both [0,1], but meanings differ. Changes measure stability, not accuracy."))
    print("Equal-support Validation comparison complete")


if __name__=="__main__":main()
