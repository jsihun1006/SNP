"""Standalone Stage B extraction, followed by Validation-only diagnostics."""
import argparse
import gzip
import hashlib
import json
from pathlib import Path
import platform
import time
import numpy as np
import pandas as pd
from .evidence_source import EvidenceSource,CLUSTER_COLUMNS,MATCH_COLUMNS
from .evidence_features import compute_evidence,NAMES
from .evidence_frames import FrameEvidence


def main():
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--output',required=True)
    p.add_argument('--config',default='competition_judgement/configs/evidence_stage_b.json')
    a=p.parse_args();root=Path.cwd();out=Path(a.output).resolve()
    if not out.is_relative_to(root/'data/derived/competition_judgement'):raise ValueError('Local-only output required')
    if out.exists():raise FileExistsError('Refusing to overwrite any previous output')
    config_path=Path(a.config);config=json.loads(config_path.read_text())
    if config['fit_required'] or config['candidate_decision'] or config['probability']:raise ValueError('Stage B only')
    if config['radius_frames']!=50:raise ValueError('Frozen 101-frame interface required')
    out.mkdir(parents=True)
    (out/'preregistered_config.json').write_text(json.dumps(config,indent=2))
    (out/'LOCAL_ONLY.txt').write_text('Licensed match derivatives: LOCAL ONLY. No uploads, server, commit or push.\nEvidence descriptors, not contest/possession labels or probabilities.\n')
    start=time.perf_counter();base=root/'data/derived/competition_judgement'
    fcols=['frame','period','n_home','n_away','strict_quality_valid','quality_tier','v1_candidate_score','max_local_contact_strength_3m','max_local_contact_strength_5m']
    frames=pd.read_csv(base/'local_v2_1_20261109/local_frame_features.csv',usecols=fcols)
    if len(frames)!=199319 or not np.array_equal(frames.frame,np.arange(199319)):raise ValueError('Full original axis required')
    clusters=pd.read_csv(base/'local_v2_1_20261109/local_clusters.csv',usecols=CLUSTER_COLUMNS)
    matching=pd.read_csv(base/'temporal_v2_2_final_20261109/cluster_temporal_features.csv',usecols=MATCH_COLUMNS)
    manifest=json.loads((root/'data/derived/relational_temporal_v1_20261109/split_manifest.json').read_text())
    source=EvidenceSource(root,frames,manifest,clusters,matching,config);del matching
    reducer=FrameEvidence(frames);validation=[];batch=config['batch_anchors'];full_count=0
    with gzip.open(out/'cluster_evidence.csv.gz','wt',compresslevel=3) as dest:
        for start_index in range(0,len(clusters),batch):
            selected=np.arange(start_index,min(start_index+batch,len(clusters)))
            ch=clusters.iloc[selected].reset_index(drop=True);s=source.query(selected)
            result=compute_evidence(s,ch.radius_m.to_numpy(),config)
            result.insert(0,'split',source.roles[source.cluster_frame[selected]])
            full_count+=int(result.full_reliable_window.sum())
            result.insert(0,'anchor_radius_m',np.maximum(6,ch.cluster_max_pairwise_distance_m.to_numpy()+3))
            combined=pd.concat([ch,result],axis=1)
            combined.to_csv(dest,index=False,header=start_index==0,na_rep='NaN')
            reducer.add(ch,result,selected)
            v=combined.split=='validation'
            if v.any():validation.append(combined.loc[v])
            if start_index%(batch*500)==0:print(f'Stage B anchors {selected[-1]+1:,}/{len(clusters):,}; {time.perf_counter()-start:.1f}s',flush=True)
    frame=reducer.table();frame.to_csv(out/'frame_evidence.csv.gz',index=False,compression=dict(method='gzip',compresslevel=3))
    val=pd.concat(validation,ignore_index=True);val.to_csv(out/'validation_evidence.csv.gz',index=False,compression=dict(method='gzip',compresslevel=3))
    if full_count!=191868:raise ValueError('Full-reliable audit mismatch')
    from .evidence_diagnostics import validation_analysis,masking_analysis
    validation_analysis(val,frame,frames,config,out)
    print('Stage B full extraction done; fixed Stage A cohort masking diagnostics starting',flush=True)
    masking=masking_analysis(source,config,out)
    code={str(path.relative_to(root)):hashlib.sha256(path.read_bytes()).hexdigest() for path in (root/'competition_judgement/src/competition_judgement').glob('*evidence*.py')}
    summary=dict(version=config['version'],config_sha256=hashlib.sha256(config_path.read_bytes()).hexdigest(),
        frame_rows=len(frames),cluster_rows=len(clusters),official_cluster_rows=int((clusters.period>0).sum()),
        full_reliable_cluster_rows=full_count,validation_rows=len(val),masking=masking,
        runtime_s=time.perf_counter()-start,source_code_sha256=code,platform=platform.platform(),
        numpy=np.__version__,pandas=pd.__version__,test_used_for_selection=False,fit_performed=False,
        candidate_decision=False,probability=False)
    (out/'summary.json').write_text(json.dumps(summary,indent=2))
    print(json.dumps({k:summary[k] for k in ('frame_rows','cluster_rows','validation_rows','runtime_s')}),flush=True)


if __name__=='__main__':main()
