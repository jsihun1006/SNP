"""Anonymous spatial disagreement against ATD; ATD is not ground truth."""
import csv
from pathlib import Path
import numpy as np
from .matching import linear_assignment
from .io import json_write,csv_write


def compare(tracks,atd,output,length=105.3,width=68.,offset=0,gate=5.,verified=False):
    groups={}
    with Path(tracks).open(newline='',encoding='utf-8-sig') as handle:
        for row in csv.DictReader(handle):
            if row['homography_valid']!='1' or not row['pitch_x_m'] or row.get('role','').lower() in ('referee','ball','other'):
                continue
            if row.get('scene_type') in ('replay','closeup'):
                continue
            frame=int(row['frame_idx'])+offset
            if frame<0:raise ValueError('Negative ATD index after applying offset')
            groups.setdefault(frame,[]).append(row)
    output=Path(output);output.mkdir(parents=True,exist_ok=False)
    rows=[];distances=[];n_pred=n_ref=n_match=0;seen=set()
    with Path(atd).open(encoding='utf-8') as handle:
        for frame,line in enumerate(handle):
            parts=line.rstrip().split(':')
            if len(parts)!=3 or int(parts[0])!=frame+1:raise ValueError('ATD is not contiguous and one-based')
            if frame not in groups:continue
            seen.add(frame)
            raw=np.fromstring(parts[1].replace(';',','),sep=',')
            if len(raw)%3:raise ValueError('Malformed ATD player slots')
            ref=raw.reshape(-1,3)[:,:2]*[length,width]
            ref=ref[np.isfinite(ref).all(axis=1)]
            pred=np.array([[float(r['pitch_x_m']),float(r['pitch_y_m'])] for r in groups[frame]])
            cost=np.linalg.norm(pred[:,None]-ref[None,:],axis=2)
            pairs,_,_=linear_assignment(cost,gate)
            ds=[float(cost[i,j]) for i,j in pairs]
            distances+=ds;n_pred+=len(pred);n_ref+=len(ref);n_match+=len(pairs)
            rows.append(dict(video_frame=frame-offset,atd_frame_id=frame+1,predicted=len(pred),atd_observed=len(ref),
                             matched=len(pairs),mean_disagreement_m=float(np.mean(ds)) if ds else ''))
    missing=sorted(set(groups)-seen)
    report=dict(metric='anonymous gated spatial disagreement; not tracking accuracy',atd_is_ground_truth=False,
        video_to_atd_alignment_verified=verified,atd_frame0_minus_video_frame0=offset,gate_m=gate,
        frames_compared=len(rows),prediction_frames_outside_atd=len(missing),predicted=n_pred,atd_observed=n_ref,matched=n_match,
        prediction_rows_outside_atd=sum(len(groups[f]) for f in missing),
        unverified_scene_rows=sum(r.get('scene_type','unverified')!='live' for group in groups.values() for r in group),
        predicted_unmatched=n_pred-n_match,atd_unmatched=n_ref-n_match,
        median_matched_disagreement_m=float(np.median(distances)) if distances else None,
        p95_matched_disagreement_m=float(np.percentile(distances,95)) if distances else None,
        caution='Gating excludes large mismatches. Always inspect unmatched counts and overlays. No player ID score is computed.')
    csv_write(output/'per_frame.csv',rows,['video_frame','atd_frame_id','predicted','atd_observed','matched','mean_disagreement_m'])
    json_write(output/'comparison.json',report)
    print('Saved comparison:',output)
    return report
