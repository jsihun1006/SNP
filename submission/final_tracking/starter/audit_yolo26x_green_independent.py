"""Read-only time-spread visual audit of single, in-pitch green candidates."""
from pathlib import Path
import json
import sys
import numpy as np
import pandas as pd

ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT))
from starter import audit_yolo26x_green_balls as evidence
OUT=ROOT/'data/derived/ball_yolo26x_pitch_20261116/independent_audit'


def main():
    OUT.mkdir(parents=True,exist_ok=True)
    green=pd.read_csv(ROOT/'data/derived/ball_yolo26x_redetection_20261116/ball_candidates.csv',float_precision='round_trip')
    h=pd.read_csv(ROOT/'data/broadcast/BAR-ATH_broadcast_homography.csv')
    hm=h[[f'h{i}{j}' for i in range(3) for j in range(3)]].to_numpy().reshape(-1,3,3)
    counts=green.groupby('source_frame').size()
    eligible=[]
    for row in green.to_dict('records'):
        f=int(row['source_frame'])
        if counts.loc[f]!=1 or h.valid.iloc[f]!=1 or not np.isfinite(hm[f]).all():continue
        try:ray=np.linalg.solve(hm[f],[row['image_x'],row['image_y'],1.])
        except np.linalg.LinAlgError:continue
        if not np.isfinite(ray).all() or abs(ray[2])<=1e-12:continue
        xy=ray[:2]/ray[2]
        if not (0<=xy[0]<=1 and 0<=xy[1]<=1):continue
        row['audit_pitch_x']=float(xy[0]*105.3);row['audit_pitch_y']=float(xy[1]*68)
        eligible.append(row)
    prior=json.loads((OUT.parent/'visual_audit/green_sample_requests.json').read_text())['records']
    excluded={(int(r['source_frame']),int(r['candidate_index'])) for r in prior}
    used=set(excluded);used_groups=set();selected=[]
    centers=np.linspace(green.source_frame.min(),green.source_frame.max(),48)
    for center in centers:
        ordered=sorted(eligible,key=lambda r:abs(r['source_frame']-center))
        for row in ordered:
            key=(int(row['source_frame']),int(row['candidate_index']));group=key[0]//500
            if key in used or group in used_groups:continue
            row=dict(row,sample_number=len(selected)+1,audit_stratum='single_green_in_pitch_time_spread')
            selected.append(row);used.add(key);used_groups.add(group);break
    assert len(selected)==48
    (OUT/'selection.json').write_text(json.dumps(dict(count=48,selection='Nearest unused single-green in-pitch observation to 48 equally spaced source times; distinct source500 groups; excludes primary audit sample',scope='Purposive time-spread sample; not a random whole-match FP estimate',records=selected),indent=2))
    evidence.OUT=OUT
    evidence.render(selected,'independent')


if __name__=='__main__':main()
