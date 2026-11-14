"""Resume absent diagnostics after completed extraction, never overwrite data."""
import argparse
import hashlib
import json
import platform
from pathlib import Path
import time
import pandas as pd
import numpy as np
from .evidence_source import EvidenceSource,CLUSTER_COLUMNS,MATCH_COLUMNS
from .evidence_diagnostics import validation_analysis,masking_analysis


def main():
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--run-dir',required=True)
    a=p.parse_args();root=Path.cwd();out=Path(a.run_dir).resolve()
    if not out.is_relative_to(root/'data/derived/competition_judgement'):raise ValueError('Local-only output')
    for name in ('summary.json','validation_distributions.csv','masking_distortion.csv'):
        if (out/name).exists():raise FileExistsError('Refusing to overwrite previous diagnostics')
    start=time.perf_counter();config=json.loads((out/'preregistered_config.json').read_text())
    cp=root/'competition_judgement/configs/evidence_stage_b.json'
    if config!=json.loads(cp.read_text()):raise ValueError('Preregistered configuration changed')
    base=root/'data/derived/competition_judgement'
    frames=pd.read_csv(base/'local_v2_1_20261109/local_frame_features.csv',usecols=['frame','period','n_home','n_away','strict_quality_valid','quality_tier',
        'v1_candidate_score','max_local_contact_strength_3m','max_local_contact_strength_5m'])
    frame=pd.read_csv(out/'frame_evidence.csv.gz');val=pd.read_csv(out/'validation_evidence.csv.gz',low_memory=False,
        dtype={'global_count_transition_offsets':str,'global_count_different_offsets':str})
    if len(frame)!=199319 or len(val)!=112286 or not (val.split=='validation').all():raise ValueError('Extraction incomplete')
    validation_analysis(val,frame,frames,config,out)
    print('Stage B Validation distributions/correlations complete (no scipy dependency)',flush=True)
    del frame,val
    clusters=pd.read_csv(base/'local_v2_1_20261109/local_clusters.csv',usecols=CLUSTER_COLUMNS)
    matching=pd.read_csv(base/'temporal_v2_2_final_20261109/cluster_temporal_features.csv',usecols=MATCH_COLUMNS)
    manifest=json.loads((root/'data/derived/relational_temporal_v1_20261109/split_manifest.json').read_text())
    source=EvidenceSource(root,frames,manifest,clusters,matching,config)
    masking=masking_analysis(source,config,out)
    code={str(path.relative_to(root)):hashlib.sha256(path.read_bytes()).hexdigest() for path in (root/'competition_judgement/src/competition_judgement').glob('*evidence*.py')}
    summary=dict(version=config['version'],config_sha256=hashlib.sha256(cp.read_bytes()).hexdigest(),frame_rows=len(frames),
        cluster_rows=len(clusters),official_cluster_rows=int((clusters.period>0).sum()),validation_rows=112286,
        full_reliable_cluster_rows=191868,masking=masking,resume_runtime_s=time.perf_counter()-start,
        extraction_elapsed_to_validation_file_s=(out/'validation_evidence.csv.gz').stat().st_mtime-(out/'preregistered_config.json').stat().st_mtime,
        diagnostics_resumed_after_missing_optional_scipy=True,source_code_sha256=code,platform=platform.platform(),
        numpy=np.__version__,pandas=pd.__version__,test_used_for_selection=False,fit_performed=False,candidate_decision=False,probability=False)
    (out/'summary.json').write_text(json.dumps(summary,indent=2));print(json.dumps(summary['masking']),flush=True)


if __name__=='__main__':main()
