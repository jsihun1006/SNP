"""Stage B.1 standalone full extraction, immutable legacy sidecar and validation."""
import argparse
import gzip
import hashlib
import json
from pathlib import Path
import time
import numpy as np
import pandas as pd
from .evidence_source import CLUSTER_COLUMNS,MATCH_COLUMNS
from .pattern_source import PatternSource
from .pattern_profiles import profiles
from .pattern_frames import PatternFrames
from .pattern_values import FEATURES
from .evidence_features import NAMES


def load_source(root,legacy):
    base=root/'data/derived/competition_judgement'
    frames=pd.read_csv(base/'local_v2_1_20261109/local_frame_features.csv',usecols=['frame','period','n_home','n_away','strict_quality_valid','quality_tier'])
    clusters=pd.read_csv(base/'local_v2_1_20261109/local_clusters.csv',usecols=CLUSTER_COLUMNS)
    matching=pd.read_csv(base/'temporal_v2_2_final_20261109/cluster_temporal_features.csv',usecols=MATCH_COLUMNS)
    manifest=json.loads((root/'data/derived/relational_temporal_v1_20261109/split_manifest.json').read_text())
    return frames,PatternSource(root,frames,manifest,clusters,matching,legacy)


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--output',required=True);p.add_argument('--config',default='competition_judgement/configs/pattern_b1.json')
    a=p.parse_args();root=Path.cwd();out=Path(a.output).resolve()
    if not out.is_relative_to(root/'data/derived/competition_judgement'):raise ValueError('Local output only')
    if out.exists():raise FileExistsError('Refusing to overwrite')
    config=json.loads(Path(a.config).read_text());legacy_path=root/config['legacy_config'];legacy=json.loads(legacy_path.read_text())
    if config['aggregate_evidence'] or config['score_bounds'] or config['fit_required'] or config['candidate_decision']:raise ValueError('Profiles only')
    out.mkdir(parents=True);start=time.perf_counter()
    (out/'preregistered_config.json').write_text(json.dumps(config,indent=2))
    (out/'legacy_config_sha256.txt').write_text(hashlib.sha256(legacy_path.read_bytes()).hexdigest())
    frames,source=load_source(root,legacy);clusters=source.clusters;batch=config['batch_anchors'];reducer=PatternFrames(frames)
    # Keep legacy composites separately, never feed these fields to profiles().
    key=['frame','radius_m','cluster_id']
    legacy_cols=key+[f'{mode}_{s}' for mode in ('full','partial') for s in ('positive_evidence_score','negative_evidence_score',
        'positive_used_mask','negative_persistence_source','negative_uses_N2')]
    for mode in ('full','partial'):legacy_cols += [mode+'_'+name for name in NAMES.values()]
    old_iter=pd.read_csv(out.parent/'evidence_stage_b_final_20261109/cluster_evidence.csv.gz',usecols=legacy_cols,chunksize=batch)
    validation=[];comparisons={};full=0
    with gzip.open(out/'cluster_pattern_profiles.csv.gz','wt',compresslevel=3) as dest,gzip.open(out/'legacy_diagnostics.csv.gz','wt',compresslevel=3) as side:
        for begin in range(0,len(clusters),batch):
            selected=np.arange(begin,min(begin+batch,len(clusters)));ch=clusters.iloc[selected].reset_index(drop=True)
            old=next(old_iter).reset_index(drop=True)
            if not np.array_equal(ch[key].to_numpy(),old[key].to_numpy()):raise ValueError('Frozen key mismatch')
            s=source.query(selected);result=profiles(s,ch.radius_m.to_numpy(),config,legacy)
            result.insert(0,'split',source.roles[source.cluster_frame[selected]])
            full+=int(result.legacy_101_full.sum())
            for mode in ('full','partial'):
                for feature,name in NAMES.items():
                    original=old[mode+'_'+name].to_numpy(float);good=np.isfinite(original)
                    if good.any() and not np.allclose(result[feature].to_numpy()[good],original[good],rtol=0,atol=1e-12,equal_nan=False):raise ValueError('Changed individual formula or validity')
                    k=mode+':'+feature;entry=comparisons.setdefault(k,dict(common_n=0,max_abs_difference=0.))
                    entry['common_n']+=int(good.sum())
                    if good.any():entry['max_abs_difference']=max(entry['max_abs_difference'],float(np.max(np.abs(result[feature].to_numpy()[good]-original[good]))))
            combined=pd.concat([ch,result],axis=1);combined['anchor_radius_m']=np.maximum(6,ch.cluster_max_pairwise_distance_m.to_numpy()+3)
            combined.to_csv(dest,index=False,header=begin==0,na_rep='NaN')
            sidecols=[c for c in legacy_cols if c in key or c not in [m+'_'+v for m in ('full','partial') for v in NAMES.values()]]
            old[sidecols].rename(columns={c:'legacy_'+c for c in sidecols if c not in key}).to_csv(side,index=False,header=begin==0,na_rep='NaN')
            reducer.add(ch,result)
            val=combined.split=='validation'
            if val.any():validation.append(combined.loc[val])
            if begin%(batch*400)==0:print(f'B.1 anchors {selected[-1]+1:,}/{len(clusters):,}; {time.perf_counter()-start:.1f}s',flush=True)
    if full!=191868:raise ValueError('Changed full-window axis')
    frame=reducer.table();frame.to_csv(out/'frame_pattern_profiles.csv.gz',index=False,compression=dict(method='gzip',compresslevel=3))
    val=pd.concat(validation,ignore_index=True)
    from .pattern_diagnostics import validation_analysis,masking_analysis,noise_analysis
    validation_analysis(val,frame,config,out)
    pd.DataFrame([dict(comparison=k,**v) for k,v in comparisons.items()]).to_csv(out/'evidence_value_equality.csv',index=False)
    masks=masking_analysis(source,config,legacy,out)
    noise=noise_analysis(root,config,legacy,out)
    summary=dict(stage='B.1',frame_rows=len(frame),cluster_rows=len(clusters),official_cluster_rows=int((clusters.period>0).sum()),
        validation_rows=len(val),legacy_full_reliable=full,masking=masks,noise=noise,runtime_s=time.perf_counter()-start,
        no_new_aggregate=True,no_fit=True,no_test_selection=True,no_labels=True,
        config_sha256=hashlib.sha256(Path(a.config).read_bytes()).hexdigest())
    (out/'summary.json').write_text(json.dumps(summary,indent=2))
    (out/'LOCAL_ONLY.txt').write_text('Local-only licensed match derivatives. No server/upload/commit/push. Profiles describe geometry, not contest/possession truth.\n')
    print(json.dumps(summary),flush=True)


if __name__=='__main__':main()
