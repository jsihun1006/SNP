"""Fixed-anchor temporal sensitivity on v1 Validation supports, no tuning."""
import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd

from mtah_relational.frame_store import FrameStore,open_frame_cache
from mtah_relational.splits import SplitManifest
from .diagnostics import validation_supports,perturb
from .local_features import extract_local_frame
from .local_diagnostics import distribution
from .temporal_source import TemporalSource
from .temporal_features import analyze_anchor_batch
from .temporal_matching import adjacent_matches
from .artifacts import json_write


def condition_source(store,manifest,config,local_config,kind,strength):
    frames=[];offsets=[]
    for first,last in validation_supports(manifest):
        for offset,frame in enumerate(range(first,last+1)):
            changed=perturb(store.frame(store.position(frame)),kind,strength,seed=config['seed'],active=65<=offset<=85)
            # Reorder ONLY within this observation to provide static team columns.
            # There is no temporal player correspondence or slot difference.
            order=np.argsort(changed.team_codes,kind='stable')
            frames.append((changed,order));offsets.append(offset)
    ids=np.array([f.frame for f,_ in frames]);periods=np.array([f.period for f,_ in frames])
    xy=np.stack([f.xy_m[o] for f,o in frames]);teams=frames[0][0].team_codes[frames[0][1]]
    flags={k:np.stack([getattr(f,k)[o] for f,o in frames]) for k in ('parse_missing','parse_partial','parse_invalid')}
    subset=FrameStore(ids,periods,xy,teams,store.pitch,**flags)
    table=[];clusters=[]
    for i in range(len(subset)):
        f,c,_=extract_local_frame(subset.frame(i),subset.pitch,local_config)
        table.append(f);clusters.extend(c)
    table=pd.DataFrame(table);clusters=pd.DataFrame(clusters)
    return TemporalSource(subset,table,manifest,config,clusters=clusters),clusters,np.array(offsets)


def compare_temporal(clean,changed):
    if len(clean)!=len(changed):raise ValueError('Need the same fixed spatial probes')
    same_math=clean.observed_mask_hex.eq(changed.observed_mask_hex).to_numpy()
    reliable=clean.pattern_score_valid.to_numpy()&changed.pattern_score_valid.to_numpy()
    names=['present_anchor_soft','observed_approach','observed_future_change','observed_window_soft_std',
           'observed_occupancy_3m','observed_occupancy_5m']
    result={}
    for name in names+['approach_focus_separation_score','stable_close_score','close_duration_lower_bound_s']:
        a,b=clean[name].to_numpy(dtype=float),changed[name].to_numpy(dtype=float)
        mask=reliable if name in ('approach_focus_separation_score','stable_close_score','close_duration_lower_bound_s') else same_math
        valid=mask&np.isfinite(a)&np.isfinite(b)
        result[name]=dict(n=int(valid.sum()),abs_change=distribution(np.abs(a[valid]-b[valid])))
    return dict(probes=len(clean),identical_observed_support=int(same_math.sum()),same_full_reliable=int(reliable.sum()),
                clean_pattern_valid=int(clean.pattern_score_valid.sum()),changed_pattern_valid=int(changed.pattern_score_valid.sum()),
                lost_pattern_valid=int((clean.pattern_score_valid&~changed.pattern_score_valid).sum()),metrics=result)


def probe_features(source,probes):
    size=source.config['batch_anchors']
    return pd.concat([analyze_anchor_batch(source,probes.iloc[i:i+size].reset_index(drop=True))
                      for i in range(0,len(probes),size)],ignore_index=True)


def main():
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--run-dir',required=True)
    p.add_argument('--cache',default='data/derived/relational_temporal_v1_20261109/raw_cache')
    p.add_argument('--split-manifest',default='data/derived/relational_temporal_v1_20261109/split_manifest.json')
    p.add_argument('--local-config',default='competition_judgement/configs/local_clusters_v2_1.json')
    a=p.parse_args();run=Path(a.run_dir).resolve()
    if not run.is_relative_to(Path.cwd()/'data/derived/competition_judgement'):raise ValueError('Local output required')
    dest=run/'temporal_robustness.json'
    if dest.exists():raise FileExistsError('Robustness result already exists')
    config=json.loads((run/'preregistered_config.json').read_text());local=json.loads(Path(a.local_config).read_text())
    store=open_frame_cache(a.cache);manifest=SplitManifest.load(a.split_manifest)
    clean_source,clean_clusters,offsets=condition_source(store,manifest,config,local,'clean',0.)
    centers=clean_source.ids[(offsets>=config['window_radius_frames'])&(offsets<=150-config['window_radius_frames'])]
    probes=clean_clusters.loc[clean_clusters.frame.isin(centers)].reset_index(drop=True)
    if not len(probes):raise ValueError('No complete probe windows in fixed support')
    clean=probe_features(clean_source,probes)
    reports={}
    conditions=[('jitter',.1),('jitter',.5),('jitter',1.),('dropout',.1),('dropout',.2),('dropout',.3),
                ('composition',.2),('permutation',0.),('team_swap',0.)]
    for kind,strength in conditions:
        source,clusters,_=condition_source(store,manifest,config,local,kind,strength)
        changed=probe_features(source,probes)
        summary=compare_temporal(clean,changed)
        match=adjacent_matches(source,clusters,config)
        summary['redetected_adjacent_matching']={str(k):int(v) for k,v in match.match_next_status.value_counts().items()}
        summary['redetected_cluster_rows']=len(clusters)
        reports[f'{kind}_{strength:g}']=summary
        print(f'Temporal Validation sensitivity complete: {kind} {strength:g}',flush=True)
    json_write(dest,dict(seed=config['seed'],supports=[list(x) for x in validation_supports(manifest)],
             role='validation',center_frames=len(centers),probes=len(probes),conditions=reports,
             notes='Fixed CLEAN anchor positions for paired sensitivity; no perturbed-cluster identity matching. Re-detected match rates are descriptive, not paired accuracy.'))


if __name__=='__main__':main()
