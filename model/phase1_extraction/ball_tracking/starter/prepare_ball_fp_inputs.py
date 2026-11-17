"""Prepare unchanged SNP selections and exact original bounding boxes for FP review."""
from pathlib import Path
import gzip
import json

import numpy as np
import pandas as pd

ROOT=Path(__file__).resolve().parents[1]
OUT=ROOT/'data/derived/ball_fp_reduction_20261116'
BASE=ROOT/'data/derived/ball_tracking_homography_20261116'
N=164487


def main():
    source=pd.read_csv(BASE/'combined_corrected_ball_pitch.csv',low_memory=False)
    first=pd.read_csv(BASE/'sources/first_half_corrected_ball_coordinates.csv',low_memory=False)
    second=pd.read_csv(BASE/'sources/second_half_ball_object_audited.csv',low_memory=False)
    xy=np.full((N,2),np.nan);bbox=np.full((N,4),np.nan);pitch=np.full((N,2),np.nan)
    confidence=np.full(N,np.nan);half=np.zeros(N,np.uint8);state=np.zeros(N,np.uint8)
    f=source.source_frame.to_numpy(int)
    xy[f]=source[['pixel_x','pixel_y']];pitch[f]=source[['pitch_x_m','pitch_y_m']]
    half[f]=source.half.to_numpy(np.uint8)
    state[f]=source.observation_kind.map(dict(missing=1,review_candidate=2,observed=3,interpolated=4)).to_numpy(np.uint8)
    confidence[f]=source.confidence
    review=json.loads((OUT/'sources/reference_review_labels.json').read_text())['records']
    refs={int(x['source_frame']):x for x in review if x['label'] in ('match_ball_observed','not_ball')}
    reference=[];selected=0
    with gzip.open(OUT/'sources/first_half_decisions.jsonl.gz','rt',encoding='utf-8-sig') as stream:
        for line in stream:
            record=json.loads(line);frame=int(record['source_frame']);candidates=record['candidates']
            if np.isfinite(xy[frame]).all():
                distances=np.array([np.linalg.norm(np.array(c['point'])-xy[frame]) for c in candidates])
                k=int(np.argmin(distances));assert distances[k]<1e-5,(frame,distances[k])
                bbox[frame]=candidates[k]['bbox_xyxy'];selected+=1
            if frame in refs:
                r=refs[frame];point=np.array([r['pixel_x'],r['pixel_y']])
                distances=np.array([np.linalg.norm(np.array(c['point'])-point) for c in candidates])
                k=int(np.argmin(distances));assert distances[k]<1e-5
                reference.append(dict(source_frame=frame,pixel_xy=point.tolist(),bbox=candidates[k]['bbox_xyxy'],
                    label=r['label'],object_kind=r.get('object_kind',r.get('protected_situation','match_ball')),
                    reason=r['reason'],provenance='pinned_source_exact_visual_review'))
    assert selected==43366
    sf=second.source_frame.to_numpy(int);bbox[sf]=second[['bbox_x1','bbox_y1','bbox_x2','bbox_y2']]
    observed=state==3
    assert observed.sum()==106689 and np.isfinite(bbox[observed]).all()
    assert np.allclose((bbox[observed,:2]+bbox[observed,2:])/2,xy[observed],atol=1e-4)
    np.savez_compressed(OUT/'fp_inputs.npz',source_frame=np.arange(N,dtype=np.int32),pixel_xy=xy,
        bbox=bbox,pitch_xy=pitch,confidence=confidence,half=half,status_codes=state)
    (OUT/'sources/reference_samples.json').write_text(json.dumps(reference,ensure_ascii=False,indent=2)+'\n')
    print(json.dumps(dict(frames=N,observed=int(observed.sum()),bbox_exact=True,reference_samples=len(reference))))


if __name__=='__main__':main()
