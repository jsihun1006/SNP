"""Export only accepted green balls through the unchanged broadcast homography."""
from pathlib import Path
import argparse
import hashlib
import json

import numpy as np
import pandas as pd

ROOT=Path(__file__).resolve().parents[1]
SOURCE=ROOT/'data/derived/ball_yolo26x_redetection_20261116'
OUT=ROOT/'data/derived/ball_yolo26x_pitch_20261116'
HFILE=ROOT/'data/broadcast/BAR-ATH_broadcast_homography.csv'
PITCH=np.array([105.3,68.])
N=164487
KEY=['source_frame','candidate_index']
COORD=['image_x','image_y','pitch_x_normalized','pitch_y_normalized','pitch_x_m','pitch_y_m']


def read(path):return pd.read_csv(path,float_precision='round_trip',low_memory=False)
def sha(path):return hashlib.sha256(Path(path).read_bytes()).hexdigest()
def write(name,data):(OUT/name).write_text(json.dumps(data,ensure_ascii=False,indent=2)+'\n')


def export():
    OUT.mkdir(exist_ok=True)
    source=read(SOURCE/'all_candidates.csv');green=source[source.verdict.eq('model_ball')].sort_values(KEY).reset_index(drop=True)
    assert len(green)==4478 and not green.duplicated(KEY).any()
    h=read(HFILE);assert np.array_equal(h.frame,np.arange(len(h))) and len(h)>=N
    matrices=h[[f'h{i}{j}' for i in range(3) for j in range(3)]].to_numpy().reshape(-1,3,3)
    frames=green.source_frame.to_numpy(int);normal=np.full((len(green),2),np.nan)
    status=np.full(len(green),'not_projected',object);condition=np.full(len(green),np.nan);error=np.full(len(green),np.nan)
    for i,f in enumerate(frames):
        if h.valid.iloc[f]!=1:status[i]='homography_invalid_flag';continue
        matrix=matrices[f]
        if not np.isfinite(matrix).all():status[i]='nonfinite_homography';continue
        try:
            ray=np.linalg.solve(matrix,np.r_[green.loc[i,['image_x','image_y']].to_numpy(float),1.])
            condition[i]=np.linalg.cond(matrix)
        except np.linalg.LinAlgError:status[i]='singular_homography';continue
        if not np.isfinite(ray).all() or abs(ray[2])<=1e-12:status[i]='point_at_infinity';continue
        normal[i]=ray[:2]/ray[2];status[i]='projected'
        back=matrix@np.r_[normal[i],1.];back=back[:2]/back[2]
        error[i]=np.linalg.norm(back-green.loc[i,['image_x','image_y']].to_numpy(float))
    green['homography_frame']=frames;green['homography_valid']=h.valid.to_numpy()[frames]==1
    green['projection_status']=status;green['pitch_x_normalized']=normal[:,0];green['pitch_y_normalized']=normal[:,1]
    green['pitch_x_m']=normal[:,0]*PITCH[0];green['pitch_y_m']=normal[:,1]*PITCH[1]
    valid=np.isfinite(normal).all(1)
    inside=np.logical_and(normal>=0,normal<=1).all(1)
    green['inside_pitch']=pd.array([bool(inside[i]) if valid[i] else None for i in range(len(green))],dtype='boolean')
    green['homography_condition']=condition;green['roundtrip_error_px']=error
    green.to_csv(OUT/'green_ball_pitch.csv',index=False)
    queue=read(SOURCE/'frames_to_redetect.csv');tested=np.zeros(N,bool);tested[queue.source_frame.to_numpy(int)]=True
    with np.load(ROOT/'data/derived/ball_fp_reduction_20261116/fp_inputs.npz') as z:half=z['half']
    timeline=pd.DataFrame(dict(source_frame=np.arange(N),video_seconds=np.arange(N)/25,half=half,
        is_retested=tested,accepted_green_count=np.bincount(frames,minlength=N),
        projected_green_count=np.bincount(frames[valid],minlength=N),
        green_on_pitch_count=np.bincount(frames[valid&inside],minlength=N)))
    timeline['frame_status']=np.where(tested,'no_green','not_retested')
    timeline['candidate_indices']='[]'
    for name in COORD:timeline[name]=np.nan
    for f,g in green.groupby('source_frame',sort=False):
        timeline.loc[f,'candidate_indices']=json.dumps(g.candidate_index.to_list(),separators=(',',':'))
        if len(g)>1:timeline.loc[f,'frame_status']='multiple_green_candidates';continue
        timeline.loc[f,'frame_status']='single_green_projected' if g.projection_status.iloc[0]=='projected' else 'single_green_h_unavailable'
        timeline.loc[f,COORD]=g[COORD].iloc[0].to_numpy(float)
    timeline.to_csv(OUT/'all_frames.csv',index=False)
    timeline[tested].to_csv(OUT/'retested_frames.csv',index=False)
    summary=dict(accepted_green_candidates=len(green),accepted_green_frames=int((timeline.accepted_green_count>0).sum()),
        projected_candidates=int(valid.sum()),homography_unavailable_candidates=int((~valid).sum()),
        projected_inside_pitch=int((valid&inside).sum()),projected_outside_pitch=int((valid&~inside).sum()),
        multiple_green_frames=int((timeline.accepted_green_count>1).sum()),
        retested_frames=int(tested.sum()),retested_frames_without_green=int((tested&(timeline.accepted_green_count==0)).sum()),
        untested_frames=int((~tested).sum()),pitch_size_m=PITCH.tolist(),
        matrix_direction='provided H maps normalized pitch to source pixels; solve H * pitch_homogeneous = [image_x,image_y,1]',
        frame_alignment='same zero-based source_frame; no shift',homography_modified=False,
        only_model_ball_exported=True,uncertain_or_nonball_promoted=False,
        coordinates_clipped=False,interpolation=False,highest_score_selection=False,
        aerial_ball_interpretation='2D planar homography projection; not independently estimated vertical ground location',
        maximum_roundtrip_error_px=float(np.nanmax(error)),
        source_sha256={str(p.relative_to(ROOT)):sha(p) for p in [HFILE,SOURCE/'all_candidates.csv',SOURCE/'ball_candidates.csv']})
    write('projection_summary.json',summary);print(json.dumps(summary,indent=2),flush=True)


def audit():
    """Attach exact visual reviews; never promote a yellow/red object into a ball."""
    green=read(OUT/'green_ball_pitch.csv');records=[];misses=[];provenance=[];caveats=[]
    # Each review producer publishes one explicit manifest. Do not discover
    # incidental intermediate JSON files and accidentally double-count labels.
    for name in ['visual_audit/review.json','independent_audit/review.json']:
        path=OUT/name
        if not path.exists():continue
        d=json.loads(path.read_text());records.extend(d.get('records',[]));misses.extend(d.get('false_negatives',[]));provenance.append(name);caveats.extend(d.get('reference_caveats',[]))
    assert records,'Visual review manifests are required'
    by_key={}
    for r in records:
        key=(int(r['source_frame']),int(r['candidate_index']));by_key.setdefault(key,[]).append(r)
    labels=[];reasons=[];evidence=[];conflicts=[];context=[]
    for row in green.itertuples():
        matches=by_key.get((row.source_frame,row.candidate_index),[])
        kinds={r['label'] for r in matches};conflict='not_ball' in kinds and bool(kinds&{'ball','secondary_ball'})
        label='conflicting_review' if conflict else ('not_ball' if 'not_ball' in kinds else ('secondary_ball' if 'secondary_ball' in kinds else ('ball' if 'ball' in kinds else ('uncertain' if matches else 'not_reviewed'))))
        labels.append(label);conflicts.append(conflict)
        reasons.append('; '.join(dict.fromkeys(r.get('reason','') for r in matches)))
        evidence.append('; '.join(dict.fromkeys(str(r.get('evidence','')) for r in matches)))
        context.append('; '.join(dict.fromkeys(str(r.get('projection_context','')) for r in matches if r.get('projection_context'))))
    green['visual_audit_label']=labels;green['visual_audit_reason']=reasons;green['visual_audit_evidence']=evidence
    green['conflicting_visual_review']=conflicts;green['projection_context_review']=context
    green['confirmed_false_positive']=green.visual_audit_label.eq('not_ball')
    frame_context={}
    for r in records:
        value=str(r.get('projection_context',''))
        if any(word in value.lower() for word in ['pip','crossfade','picture_in_picture']):
            frame_context.setdefault(int(r['source_frame']),set()).add(value)
    green['frame_projection_context_review']=['; '.join(sorted(frame_context.get(int(f),set()))) for f in green.source_frame]
    green['pitch_context_requires_review']=green.frame_projection_context_review.str.len().gt(0)
    green['pitch_projection_outlier']=green.projection_status.eq('projected')&(~green.pitch_x_m.between(-20,125.3)|~green.pitch_y_m.between(-20,88))
    green['pitch_usable']=green.projection_status.eq('projected')&~green.pitch_context_requires_review&~green.pitch_projection_outlier&~green.confirmed_false_positive&~green.visual_audit_label.eq('secondary_ball')
    counts=green.groupby('source_frame').size();green['multiple_green_candidates']=green.source_frame.map(counts).gt(1)
    green.to_csv(OUT/'green_ball_pitch.csv',index=False)
    green[green.confirmed_false_positive].to_csv(OUT/'confirmed_false_positives.csv',index=False)
    reviewed=green[~green.visual_audit_label.isin(['not_ball','secondary_ball'])].copy()
    # Non-ball source observations are excluded only in this reviewed subset.
    # Mapping/context doubts are retained as explicit QA flags rather than an H fix.
    reviewed.to_csv(OUT/'reviewed_ball_pitch.csv',index=False)
    reviewed[reviewed.pitch_usable].to_csv(OUT/'ball_pitch_coordinates.csv',index=False)
    reviewed[reviewed.pitch_projection_outlier].to_csv(OUT/'extreme_projection_review.csv',index=False)
    risk=reviewed.projection_status.ne('projected')|reviewed.inside_pitch.eq(False)|reviewed.multiple_green_candidates|reviewed.pitch_context_requires_review|reviewed.visual_audit_label.isin(['uncertain','conflicting_review'])
    reviewed[risk].to_csv(OUT/'review_required_candidates.csv',index=False)
    miss_table=pd.DataFrame(misses)
    if len(miss_table):
        miss_table=miss_table.drop_duplicates(subset=[c for c in ['source_frame','image_x','image_y'] if c in miss_table.columns])
    miss_table.to_csv(OUT/'confirmed_false_negatives.csv',index=False)
    timeline=read(OUT/'all_frames.csv')
    timeline['reviewed_green_count']=timeline.source_frame.map(reviewed.groupby('source_frame').size()).fillna(0).astype(int)
    timeline['usable_pitch_count']=timeline.source_frame.map(reviewed[reviewed.pitch_usable].groupby('source_frame').size()).fillna(0).astype(int)
    timeline['raw_green_frame_status']=timeline.frame_status
    timeline['raw_green_candidate_indices']=timeline.candidate_indices
    timeline['candidate_indices']='[]'
    timeline['frame_status']=np.where(timeline.is_retested,'no_reviewed_green','not_retested')
    rejected_all=timeline.is_retested&(timeline.accepted_green_count>0)&timeline.reviewed_green_count.eq(0)
    timeline.loc[rejected_all,'frame_status']='all_greens_visually_rejected'
    for name in COORD:timeline[name]=np.nan
    for f,g in reviewed.groupby('source_frame',sort=False):
        timeline.loc[f,'candidate_indices']=json.dumps(g.candidate_index.to_list(),separators=(',',':'))
        if len(g)>1:timeline.loc[f,'frame_status']='multiple_reviewed_greens'
        if len(g)==1:
            timeline.loc[f,['image_x','image_y']]=g[['image_x','image_y']].iloc[0].to_numpy(float)
            if g.pitch_usable.iloc[0]:
                timeline.loc[f,COORD[2:]]=g[COORD[2:]].iloc[0].to_numpy(float)
                timeline.loc[f,'frame_status']='single_reviewed_green_projected'
            else:timeline.loc[f,'frame_status']='single_reviewed_green_projection_unavailable_or_review'
    timeline.to_csv(OUT/'reviewed_all_frames.csv',index=False)
    write('reference_caveats.json',dict(records=caveats,previous_reference_files_modified=False))
    summary=dict(review_manifests=provenance,unique_green_objects_reviewed=len(by_key),
        label_counts=green.loc[green.visual_audit_label.ne('not_reviewed'),'visual_audit_label'].value_counts().to_dict(),
        confirmed_false_positive_objects=int(green.confirmed_false_positive.sum()),confirmed_false_negative_cases=len(miss_table),
        reviewed_ball_candidates=len(reviewed),conflicting_reviews=int(sum(conflicts)),
        usable_pitch_candidates=int(reviewed.pitch_usable.sum()),projection_context_review_frames=sorted(frame_context),
        extreme_projection_candidates=int(reviewed.pitch_projection_outlier.sum()),
        extreme_projection_rule='Review planar projections outside pitch rectangle plus 20m margin; original values and image ball observations are preserved, not labelled nonball or clipped',
        confirmed_false_negative_frames=int(miss_table.source_frame.nunique()) if len(miss_table) else 0,
        missing_detections_not_automatically_added=True,reviewed_subset_contains_only_original_greens=True,
        not_a_full_match_error_rate=True,projection_context_flags=int(green.projection_context_review.fillna('').str.len().gt(0).sum()))
    write('audit_summary.json',summary);print(json.dumps(summary,ensure_ascii=False,indent=2),flush=True)


def main():
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('stage',choices=['export','audit']);a=p.parse_args();globals()[a.stage]()


if __name__=='__main__':main()
