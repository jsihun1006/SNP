"""v2-3 immutable-sidecar extraction followed by fixed Validation diagnostics."""
import argparse
import gzip
import hashlib
import json
from pathlib import Path
import time
import numpy as np
import pandas as pd
from .candidate_decision_v23 import KEY, STATES, BASIS, SPATIAL_FAILURE, REVIEW_REASON, decide_clusters
from .candidate_frames_v23 import CandidateFrames
from .local_temporal_features import SIGNALS


def sha(path):
    h=hashlib.sha256()
    with Path(path).open('rb') as f:
        for block in iter(lambda:f.read(8*1024*1024),b''):h.update(block)
    return h.hexdigest()


def profile_columns():
    columns=KEY+['period','strict_quality_valid','n_players','n_home','n_away','n_cross_edges','cluster_centroid_x','cluster_centroid_y',
        'cluster_radius_m','cluster_max_pairwise_distance_m','cross_pair_density','contact_strength','split','current_strict',
        'current_anchor_home','current_anchor_away','current_soft','current_edges','legacy_101_full']
    for key in ('P1','P2','P3','P4','N1','N2','N3'):columns += [key,key+'_state',key+'_reason']
    for key in 'ABCD':columns += [key+'_obs',key+'_seq',key+'_outcome',key+'_reason']
    return columns


def local_columns():
    columns=KEY+list(SIGNALS)+['local_sequence_short','local_sequence_near']
    for key in ('past_short','past_near','future_short','future_near'):
        columns += [key+suffix for suffix in ('_reliable_n','_max_missing_run','_local_changed_n','_direction_reason')]
    return columns


def data_dictionary(config,columns,frame_columns):
    return dict(version=config['version'],key=KEY,states=config['cluster_rules'],frame_priority=config['frame_priority'],
        basis_bits=BASIS,spatial_failure_bits=SPATIAL_FAILURE,review_reason_bits=REVIEW_REASON,
        dynamic_support_bits={'A_complete_known':1,'B_complete_known':2,'rise_short_complete_known':4,'rise_near_complete_known':8,'fall_short_complete_known':16,'fall_near_complete_known':32},
        evidence_mask_bits={'P1':1,'P2':2,'P3':4,'P4':8},
        meanings={'observed_basis_mask':'all observed same-anchor A/B/Short RF/Near RF, before current Spatial Gate; not a score',
          'decision_basis_mask':'candidate supporting bits AFTER Spatial Gate; preserves all coexisting bases, zero for noncandidate states',
          'dynamic_support_mask':'availability only, not signal magnitude; all63 required for dynamic known absence',
          'required_complete_mask':'P1..P4 COMPLETE availability, not values', 'required_partial_mask':'P1..P4 EXPERIMENTAL_PARTIAL availability; Partial cannot create candidate alone',
          'upstream_unknown_reasons':'original A/B/P1..P4 and local direction unknown reasons, no copied raw masks',
          'review_reason_mask':'nonexclusive reasons only for REVIEW rows',
          'frame_state':'existence of an independently decided anchor; never cross-anchor proof fusion',
          'anchor_views':'repeated frame/radius observations, not independent events',
          'radius_source':'best completed anchor at that radius, by fixed priority then canonical id; may be noncandidate',
          'frame_empty':'strict official observed empty mixed graph -> NO_DETECTED_SIGNAL, temporal NOT_APPLICABLE; not a temporal negative example'},
        future=config['future_metadata'],future_frames_by_basis=config['future_frames_by_basis'],offline=True,
        reasons={'PATTERN_A_B':'both frozen A/B support','PATTERN_A':'frozen A support','PATTERN_B':'frozen B support',
          'LOCAL_RF_SHORT_NEAR':'both independently complete same-anchor scopes','LOCAL_RF_SHORT':'short complete same-anchor RF','LOCAL_RF_NEAR':'near complete same-anchor RF',
          'DYNAMIC_CONDITIONS_NOT_OBSERVED':'all A/B and four Local changes adequately observed but conditions unmet',
          'RISE_FALL_UNCONFIRMED':'rise observed, no verified matching-scope fall','FALL_RISE_UNCONFIRMED':'fall observed, no verified matching-scope rise',
          'UNPAIRED_SCOPE_SIGNALS':'rise/fall facts exist but no certified same-scope RF','PARTIAL_DYNAMIC_EVIDENCE':'frozen partial required Evidence available',
          'WHOLE_PATTERN_UNCONFIRMED':'A or B unknown, no qualifying same-scope RF','SPATIAL_ONLY_TEMPORAL_UNKNOWN':'all frozen individual and Local outcomes unavailable',
          'OTHER_TEMPORAL_OBSERVATIONS':'only incomplete/alternative temporal facts','SPATIAL_NO_MIXED_CLUSTER':'valid current observed empty graph; temporal N/A',
          'CURRENT_ANCHOR_UNAVAILABLE':'no usable current anchor records','CURRENT_TEAMS_UNOBSERVED':'frame lacks an opposing observed set',
          **{name:'current spatial gate failure; see spatial_failure_mask' for name in SPATIAL_FAILURE}},
        joins='Frozen v2-1/B.1/B.2 by key. Individual scores, pattern/local outcomes, time arrays and QC masks are not duplicated.',
        robustness='Transition rows are paired fixed CLEAN probes, grouped all/3m/5m. Overlapping windows/anchors are correlated. Same dynamic support != same player identity.',
        scope='No aggregate score/probability, veto, fit, labels, episode stitching, possession, upload or git mutation.',
        columns=columns,frame_columns=frame_columns)


def main():
    parser=argparse.ArgumentParser(description=__doc__);parser.add_argument('--output',required=True)
    parser.add_argument('--config',default='competition_judgement/configs/candidate_v23.json')
    args=parser.parse_args();root=Path.cwd();out=Path(args.output).resolve()
    if not out.is_relative_to(root/'data/derived/competition_judgement'):raise ValueError('Local derived output only')
    if out.exists():raise FileExistsError('Refusing to overwrite existing results')
    config_path=Path(args.config);config=json.loads(config_path.read_text())
    if config['aggregate_score'] or config['probability'] or config['fit_required'] or config['alternative_evidence_veto']:raise ValueError('Traceable fixed states only')
    start=time.perf_counter();base=out.parent
    paths=[base/'evidence_pattern_b1_final_20261110/cluster_pattern_profiles.csv.gz',base/'local_temporal_b2_final_20261110/cluster_local_temporal.csv.gz',
           base/'local_v2_1_20261109/local_frame_features.csv',base/'local_v2_1_20261109/local_clusters.csv',base/'stage1_v1_final_20261109/candidate_frames.csv']
    hashes={str(p.relative_to(root)):sha(p) for p in paths}
    frames=pd.read_csv(paths[2],usecols=['frame','period','strict_quality_valid','n_home','n_away','n_local_clusters_3m','n_local_clusters_5m'])
    if not np.array_equal(frames.frame,np.arange(199319)):raise ValueError('Original frame axis required')
    out.mkdir(parents=True);(out/'preregistered_config.json').write_text(json.dumps(config,indent=2))
    (out/'LOCAL_ONLY.txt').write_text('Local-only match derivatives. Offline geometric candidates, not football ground truth. No upload/commit/push.\n')
    reducer=CandidateFrames(frames,config);chunk=config['chunk_rows'];rows=0;validation=[];columns=None
    old_iter=pd.read_csv(paths[0],usecols=profile_columns(),chunksize=chunk)
    local_iter=pd.read_csv(paths[1],usecols=local_columns(),chunksize=chunk)
    with gzip.open(out/'cluster_candidate_decisions.csv.gz','wt',compresslevel=3) as dest:
        for old in old_iter:
            local=next(local_iter);old=old.reset_index(drop=True);local=local.reset_index(drop=True)
            if not np.array_equal(old[KEY].to_numpy(),local[KEY].to_numpy()):raise ValueError('Frozen B.1/B.2 key mismatch')
            result=decide_clusters(old,old,local,config);combined=pd.concat([old[KEY],result],axis=1);columns=list(combined)
            combined.to_csv(dest,index=False,header=rows==0,na_rep='NaN');reducer.add(old,result);rows+=len(old)
            sel=old.split=='validation'
            if sel.any():
                diag=combined.loc[sel].reset_index(drop=True)
                for col in ['split','current_strict','legacy_101_full']+[k+'_outcome' for k in 'ABCD']+[x for k in ('P1','P2','P3','P4','N1','N2','N3') for x in (k,k+'_state')]:
                    diag[col]=old.loc[sel,col].to_numpy()
                validation.append(diag)
            if rows%100000==0:print(f'v2-3 anchors {rows:,}; {time.perf_counter()-start:.1f}s',flush=True)
    try:next(local_iter);raise ValueError('Extra B.2 rows')
    except StopIteration:pass
    if rows!=789005:raise ValueError('Original cluster axis required')
    frame=reducer.table();frame.to_csv(out/'frame_candidate_decisions.csv.gz',index=False,compression=dict(method='gzip',compresslevel=3))
    from .candidate_diagnostics_v23 import validation_analysis,masking_analysis,noise_analysis
    val=pd.concat(validation,ignore_index=True);diagnostics=validation_analysis(val,frame,root,out);del val,validation
    from .local_temporal_source import load_local_source
    b1=json.loads((root/config['pattern_config']).read_text());lc=json.loads((root/config['local_config']).read_text());legacy=json.loads((root/config['evidence_config']).read_text())
    _,source=load_local_source(root,legacy)
    masking=masking_analysis(source,config,b1,lc,legacy,out);noise=noise_analysis(root,config,b1,lc,legacy,out)
    for name,original in hashes.items():
        if sha(root/name)!=original:raise ValueError('Changed frozen input: '+name)
    (out/'legacy_input_preservation.json').write_text(json.dumps(dict(sha256=hashes,all_unchanged=True,cluster_keys_exact=True,no_old_writes=True),indent=2))
    (out/'data_dictionary.json').write_text(json.dumps(data_dictionary(config,columns,list(frame)),indent=2))
    summary=dict(version=config['version'],frame_rows=len(frame),cluster_rows=rows,cluster_columns=len(columns),frame_columns=len(frame.columns),
                 diagnostics=diagnostics,masking=masking,noise=noise,offline=True,no_event_stitching=True,no_labels=True,no_fit=True,
                 no_test_selection=True,no_alternative_veto=True,config_sha256=sha(config_path),runtime_s=time.perf_counter()-start)
    (out/'summary.json').write_text(json.dumps(summary,indent=2));print(json.dumps(summary),flush=True)


if __name__=='__main__':main()
