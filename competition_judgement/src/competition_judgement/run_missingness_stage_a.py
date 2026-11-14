"""Stage A ONLY: read existing caches, diagnose masks, run paired deletions."""
import argparse
import hashlib
import json
from pathlib import Path
import platform
import sys
import time
import numpy as np
import pandas as pd
from .missingness_diagnostics import MaskContext,analyze_masks
from .missingness_cache import ReadOnlyAnchorCache
from .missingness_experiments import run_experiments


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--root',default='.')
    p.add_argument('--output',required=True)
    p.add_argument('--config',default='competition_judgement/configs/missingness_stage_a.json')
    a=p.parse_args();root=Path(a.root).resolve();out=Path(a.output).resolve()
    if not out.is_relative_to(root/'data/derived/competition_judgement'):raise ValueError('Local separate derived folder required')
    out.mkdir(parents=True,exist_ok=False)
    config_path=Path(a.config);config=json.loads(config_path.read_text())
    if config['stage_b'] or config['policy_selection'] or config['radius_frames']!=50:raise ValueError('Stage A only')
    (out/'preregistered_diagnostic_config.json').write_text(json.dumps(config,indent=2))
    start=time.time();base=root/'data/derived/competition_judgement'
    frame_path=base/'local_v2_1_20261109/local_frame_features.csv'
    temporal_path=base/'temporal_v2_2_final_20261109/cluster_temporal_features.csv'
    frames=pd.read_csv(frame_path,usecols=['frame','period','strict_quality_valid','n_home','n_away'])
    manifest=json.loads((root/'data/derived/relational_temporal_v1_20261109/split_manifest.json').read_text())
    context=MaskContext(frames,manifest)
    print('Stage A: full original mask audit starting (no source reparsing)',flush=True)
    audit,donors,pool=analyze_masks(temporal_path,context,config,out)
    print(f'Stage A: full mask audit complete; eligible Validation pool={len(pool)}',flush=True)
    # Sampling is fixed before feature-error analysis, by block and radius, seed only.
    rng=np.random.default_rng(config['seed']);parts=[]
    for _,ch in pool.groupby(['support_block','radius_m'],sort=True):
        ix=rng.choice(len(ch),min(len(ch),config['reference_per_block_radius']),replace=False)
        parts.append(ch.iloc[np.sort(ix)])
    reference=pd.concat(parts,ignore_index=True)
    if not reference.strict_quality_valid.all():raise ValueError('Expected strict reference')
    reference[['frame','radius_m','cluster_id','period','n_players','support_block']].to_csv(out/'reference_keys.csv',index=False)
    source=ReadOnlyAnchorCache(root,frames,manifest)
    print(f'Stage A: read-only fixed-anchor reconstruction for {len(reference)} references',flush=True)
    series=source.reference_series(reference)
    # Diagnostic-only local arrays retain exact repeatability, never model inputs or labels.
    np.savez_compressed(out/'reference_temporal_arrays.npz',**series)
    result=run_experiments(reference,series,donors,config,out)
    protected={}
    for path in [frame_path,temporal_path,config_path,root/'data/derived/relational_temporal_v1_20261109/split_manifest.json']:
        h=hashlib.sha256()
        with path.open('rb') as handle:
            for block in iter(lambda:handle.read(8*1024*1024),b''):h.update(block)
        protected[str(path.relative_to(root)) if path.is_absolute() else str(path)]=h.hexdigest()
    summary=dict(stage='A only',elapsed_s=time.time()-start,python=sys.version,platform=platform.platform(),
        numpy=np.__version__,pandas=pd.__version__,configuration=config,
        source_sha256=protected,total_frames=len(frames),official_cluster_rows=audit['official_rows'],
        full_reliable_cluster_rows=audit['aggregates']['all']['reliable_full'],
        eligible_validation_pool=len(pool),**result)
    (out/'summary.json').write_text(json.dumps(summary,indent=2))
    print(f'Stage A complete: {len(reference)} reference windows, {result["computed_conditions"]} computed conditions',flush=True)


if __name__=='__main__':main()
