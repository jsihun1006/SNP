"""Select whole-match uniform/penalty-mark review frames for full-frame YOLO26.

Selection is an inference request, never a false-ball label or coordinate edit.
The source is always the original 1920x1080 frame, without ROI/tile zoom.
"""
from pathlib import Path
import hashlib
import json

import numpy as np
import pandas as pd

ROOT=Path(__file__).resolve().parents[1]
BASE=ROOT/'data/derived/ball_fp_reduction_20261116'
PREVIOUS=ROOT/'data/derived/ball_uniform_context_20261116'
OUT=ROOT/'data/derived/ball_targeted_redetection_20261116'
VIDEO=ROOT/'data/broadcast/BAR-ATH_broadcast-001.mp4'
N=164487
PADDING=5


def sha(path):
    h=hashlib.sha256()
    with Path(path).open('rb') as f:
        for block in iter(lambda:f.read(4*1024*1024),b''):h.update(block)
    return h.hexdigest()


def main():
    OUT.mkdir(parents=True,exist_ok=True)
    with np.load(BASE/'fp_inputs.npz') as z:d={k:z[k] for k in z.files}
    context=pd.read_csv(PREVIOUS/'context_decisions.csv',float_precision='round_trip')
    previous=pd.read_csv(PREVIOUS/'filtered_ball_coordinates.csv',float_precision='round_trip')
    assert len(context)==len(previous)==N
    for table in [context,previous]:assert np.array_equal(table.source_frame,np.arange(N))
    with np.load(BASE/'evidence.npz') as z:
        green=np.full(N,np.nan);green[z['feature_frames']]=z['context_green']
    observed=d['status_codes']==3
    available=np.isfinite(d['pixel_xy']).all(1)
    scores=context.nonball_score.to_numpy();similarity=context.closest_nonball_hog_similarity.to_numpy()
    # Broad queue includes unresolved logos and hands near logos. All remain
    # hypotheses until candidate/scene review; queueing does not reject a point.
    known=previous.rejected.to_numpy(bool)
    body=observed&context.torso_u.notna().to_numpy()&(scores>=.85)
    max_size=np.max(d['bbox'][:,2:]-d['bbox'][:,:2],axis=1)
    appearance=observed&(scores>=.97)&(similarity>=.8)&(green<.3)&(max_size>=18)
    x,y=d['pitch_xy'].T
    distance=np.minimum(np.hypot(x-11.,y-34.),np.hypot(x-94.3,y-34.))
    mark=observed&(distance<=2.)
    # Geometry only identifies frames to revisit. It never certifies a field
    # marking or removes a real ball near or on the penalty spot.
    boundary_distance=np.minimum.reduce([np.abs(x-16.5),np.abs(x-88.8),np.abs(y-13.84),np.abs(y-54.16)])
    box_extent=(x>=-.5)&(x<=105.8)&(y>=13.04)&(y<=54.96)&((x<=17.3)|(x>=88.0))
    line=observed&box_extent&(boundary_distance<.8)
    reference=json.loads((OUT/'frame_audit/penalty_reference.json').read_text())
    records=reference['records']
    audited=np.zeros(N,bool)
    for r in records:audited[int(r['source_frame'])]=True
    flags=dict(previous_nonball_rejection=known,uniform_body_suspicion=body,
        uniform_appearance_suspicion=appearance,penalty_mark_neighborhood=mark,
        penalty_box_boundary_neighborhood=line,penalty_visual_reference=audited)
    for mask in flags.values():assert mask.shape==(N,)
    seed=np.logical_or.reduce(list(flags.values()))
    selected=seed.copy()
    for shift in range(1,PADDING+1):
        same=d['half'][shift:]==d['half'][:-shift]
        selected[shift:]|=seed[:-shift]&same
        selected[:-shift]|=seed[shift:]&same
    selected&=d['half']>0
    edges=np.diff(np.r_[False,selected,False].astype(np.int8))
    starts=np.flatnonzero(edges==1);ends=np.flatnonzero(edges==-1)
    interval_id=np.full(N,-1,np.int32)
    reasons=np.full(N,'',object)
    for name,mask in flags.items():
        for f in np.flatnonzero(mask):reasons[f]+=(';' if reasons[f] else '')+name
    reasons[selected&~seed]='temporal_context'
    intervals=[]
    for k,(a,b) in enumerate(zip(starts,ends)):
        interval_id[a:b]=k
        intervals.append(dict(interval_id=k,first_frame=int(a),last_frame_exclusive=int(b),
            frames=int(b-a),half=int(d['half'][a]),start_seconds=a/25,end_seconds_exclusive=b/25,
            seed_frames=int(seed[a:b].sum()),reasons=';'.join(name for name,m in flags.items() if m[a:b].any())))
    table=pd.DataFrame(dict(source_frame=np.arange(N),atd_frame_id=np.arange(N)+1,
        video_seconds=np.arange(N)/25,half=d['half'],selected_for_yolo=selected,is_seed=seed,
        context_only=selected&~seed,interval_id=interval_id,reason=reasons,**flags))
    table.to_csv(OUT/'frame_classification.csv',index=False)
    table[selected].to_csv(OUT/'frames_to_redetect.csv',index=False)
    pd.DataFrame(intervals).to_csv(OUT/'frame_intervals.csv',index=False)
    def grow(mask):
        expanded=mask.copy()
        for shift in range(1,PADDING+1):
            same=d['half'][shift:]==d['half'][:-shift]
            expanded[shift:]|=mask[:-shift]&same
            expanded[:-shift]|=mask[shift:]&same
        return expanded&(d['half']>0)
    uniform_request=grow(known|body|appearance)
    penalty_request=grow(mark|line|audited)
    assert np.array_equal(uniform_request|penalty_request,selected)
    grouped=table.loc[selected].copy();frames=grouped.source_frame.to_numpy(int)
    grouped['uniform_request']=uniform_request[frames]
    grouped['penalty_mark_line_request']=penalty_request[frames]
    grouped['request_group']=np.select([uniform_request[frames]&penalty_request[frames],
        uniform_request[frames],penalty_request[frames]],['both','uniform','penalty_mark_line'],default='error')
    grouped[['source_frame','video_seconds','half','interval_id','uniform_request',
        'penalty_mark_line_request','request_group']].to_csv(OUT/'frame_groups.csv',index=False)
    grouped.loc[grouped.uniform_request].to_csv(OUT/'uniform_frames.csv',index=False)
    grouped.loc[grouped.penalty_mark_line_request].to_csv(OUT/'penalty_mark_line_frames.csv',index=False)
    (OUT/'frame_groups_summary.json').write_text(json.dumps(dict(selected_frames=int(selected.sum()),
        uniform_frames=int(uniform_request.sum()),penalty_mark_line_frames=int(penalty_request.sum()),
        both=int((uniform_request&penalty_request).sum()),overlap_frames_deduplicated_in_main_manifest=True),indent=2))
    # Provide source candidate boxes as context to distinguish same-object
    # detections from alternatives. Suspected boxes are not exclusion truth.
    false_refs={int(r['source_frame']) for r in records if r.get('label') in ['not_ball','penalty_mark','false_positive','penalty_mark_false_positive','painted_penalty_mark_not_ball']}
    # The exact review schema may name the object separately from the label.
    false_refs|={int(r['source_frame']) for r in records if r.get('label')=='not_ball_observed'}
    region_count=0
    with (OUT/'negative_regions.jsonl').open('w') as stream:
        for f in np.flatnonzero(selected):
            regions=[]
            if seed[f] and available[f] and np.isfinite(d['bbox'][f]).all():
                kinds=[name for name,m in flags.items() if m[f] and name!='penalty_visual_reference']
                # No region is supplied solely because a protected positive is
                # in the evaluation set. Its near-mark/line hypothesis remains
                # explicitly uncertain and cannot erase the positive label.
                if kinds:
                    regions.append(dict(bbox_xyxy=d['bbox'][f].tolist(),kind=';'.join(kinds),
                        certainty='confirmed_nonball' if known[f] or int(f) in false_refs else 'suspected_only',
                        source_pixel_xy=d['pixel_xy'][f].tolist(),
                        instruction='annotate overlap; preserve all raw YOLO detections'))
            stream.write(json.dumps(dict(source_frame=int(f),regions=regions),separators=(',',':'))+'\n')
            region_count+=len(regions)
    inputs=[BASE/'fp_inputs.npz',BASE/'evidence.npz',PREVIOUS/'context_decisions.csv',
        PREVIOUS/'filtered_ball_coordinates.csv',OUT/'frame_audit/penalty_reference.json']
    hashes={str(p.relative_to(ROOT)):sha(p) for p in inputs}
    hashes.update(json.loads((BASE/'source_integrity.json').read_text()))
    summary=dict(video=str(VIDEO.relative_to(ROOT)),video_frames=N,fps=25,width=1920,height=1080,
        frame_convention='zero-based source video; ATD frame ID = source_frame + 1',
        source_frames_scanned=N,selected_frames=int(selected.sum()),selected_fraction=float(selected.mean()),
        seed_frames=int(seed.sum()),context_frames=int((selected&~seed).sum()),context_padding_each_side_frames=PADDING,
        intervals=len(intervals),source_candidate_regions=region_count,
        reasons={name:int(m.sum()) for name,m in flags.items()},
        per_half={str(h):dict(seed_frames=int((seed&(d['half']==h)).sum()),
            selected_frames=int((selected&(d['half']==h)).sum())) for h in [1,2]},
        penalty_interpretation='Penalty spot and penalty-box boundary neighborhoods are review requests, not confirmed false positives',
        actual_ball_reference_frames_included=[int(r['source_frame']) for r in records if r.get('label') in ['match_ball','match_ball_observed','real_match_ball']],
        uniform_closeup_method='Existing false-observation decisions, body context, and high appearance suspicion; includes unresolved logo candidates',
        model_input='original complete 1920x1080 frame; aspect-preserving resize/padding only; no ROI, tile, or candidate crop',
        original_tracking_changed=False,homography_changed=False,queued_frames_are_not_false_positive_labels=True,
        source_sha256=hashes)
    (OUT/'selection_summary.json').write_text(json.dumps(summary,indent=2)+'\n')
    config=dict(video=str(VIDEO.resolve()),manifest=str((OUT/'frames_to_redetect.csv').resolve()),
        negative_regions=str((OUT/'negative_regions.jsonl').resolve()),
        model=str((ROOT/'yolo26_football_detection/yolo26s.pt').resolve()),
        classes=[32],conf=.05,imgsz=1280,rect=True,iou=.7,device='mps',threads=4,
        cpu_fallback='Use the same runner with --device cpu if MPS is unavailable',
        full_frame=True,roi=False,tiles=False,augmentation=False,
        output=str((OUT/'yolo26').resolve()),frame_convention=summary['frame_convention'],
        manifest_sha256=sha(OUT/'frames_to_redetect.csv'))
    (OUT/'yolo26_request.json').write_text(json.dumps(config,indent=2)+'\n')
    print(json.dumps({k:v for k,v in summary.items() if k!='source_sha256'},indent=2))


if __name__=='__main__':main()
