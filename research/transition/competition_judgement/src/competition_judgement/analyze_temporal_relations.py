"""Immutable v2-1 inputs -> offline, boundary-aware v2-2 local descriptors."""
import argparse
import json
import os
from pathlib import Path
import tempfile
import time

import numpy as np
import pandas as pd

from games import GAME,GAMES,half_video_blocks,asset_paths
from mtah_relational.frame_store import open_frame_cache
from mtah_relational.splits import SplitManifest,make_split_manifest
from .cli import sha,preservation_snapshot,verify_preserved
from .artifacts import json_write
from .local_diagnostics import distribution
from .temporal_source import TemporalSource
from .temporal_features import analyze_anchor_batch
from .temporal_matching import adjacent_matches

FRAME_COLUMNS=['frame','period','n_home','n_away','strict_quality_valid','quality_tier','local_metric_available',
               'max_local_contact_strength_3m','max_local_contact_strength_5m','v1_candidate_state','v1_candidate_score']
PATTERN_COLUMNS=['present_anchor_soft','observed_approach','observed_future_change','observed_window_soft_mean',
    'observed_window_soft_std','observed_window_soft_range','observed_nearest_approach_m',
    'approach_focus_separation_score','stable_close_score','transient_excess_over_mean',
    'observed_occupancy_3m','observed_occupancy_5m','close_duration_lower_bound_s']


def validate_temporal_config(c):
    if c['fps']!=25 or c['fit_required'] is not False:raise ValueError('25fps, fixed descriptors only')
    if not isinstance(c['window_radius_frames'],int) or c['window_radius_frames']<2:raise ValueError('Invalid temporal radius')
    for key in ('anchor_min_radius_m','anchor_diameter_margin_m','kernel_decay_m','matching_max_center_step_m','matching_min_disk_radius_m'):
        if not np.isfinite(c[key]) or c[key]<=0:raise ValueError('Invalid physical scale')
    for key in ('partial_side_fraction','matching_min_size_ratio','matching_min_disk_iou'):
        if not 0<c[key]<=1:raise ValueError('Invalid fraction')
    if not isinstance(c['batch_anchors'],int) or c['batch_anchors']<1:raise ValueError('Invalid batch size')


def extract_temporal(source,clusters,output_path,progress=False):
    if clusters.duplicated(['frame','radius_m','cluster_id']).any():raise ValueError('Duplicate v2-1 keys')
    if not len(clusters):raise ValueError('Need explicit current clusters for extraction')
    matching=adjacent_matches(source,clusters,source.config)
    summaries=[];size=source.config['batch_anchors'];start_time=time.perf_counter()
    with Path(output_path).open('w') as target:
        for start in range(0,len(clusters),size):
            end=min(len(clusters),start+size);base=clusters.iloc[start:end].reset_index(drop=True)
            features=analyze_anchor_batch(source,base)
            combined=pd.concat([base,features,matching.iloc[start:end].reset_index(drop=True)],axis=1)
            combined.to_csv(target,index=False,header=start==0,na_rep='NaN')
            names=['frame','radius_m','cluster_id','pattern_score_valid','full_record_window','partial_bidirectional_available',
                   'record_past_frames','record_future_frames','reliable_past_frames','reliable_future_frames',
                   'global_count_change','local_count_change',*PATTERN_COLUMNS,
                   'match_prev_status','match_next_status','future_end_reason','onset_censored','end_censored']
            summaries.append(combined[names])
            if progress and ((start//size)%400==0):
                print(f'Temporal anchors {end:,}/{len(clusters):,}; {time.perf_counter()-start_time:.1f}s',flush=True)
    return pd.concat(summaries,ignore_index=True)


def frame_temporal_features(source,frame_table,clusters,temporal):
    result=frame_table.copy()
    r=source.config['window_radius_frames'];positions=np.arange(len(source.ids))
    result['record_past_frames']=np.minimum(r,positions-source.first)
    result['record_future_frames']=np.minimum(r,source.last-positions)
    for radius in (3,5):
        present=clusters.loc[clusters.radius_m.eq(radius)]
        t=temporal.loc[temporal.radius_m.eq(radius)]
        grouped=t.groupby('frame')
        count=grouped.size()
        result[f'n_current_clusters_{radius}m']=result.frame.map(count).fillna(0).astype(int)
        result[f'n_current_clusters_{radius}m']=np.where(result.local_metric_available,result[f'n_current_clusters_{radius}m'],np.nan)
        result[f'n_pattern_valid_{radius}m']=result.frame.map(grouped.pattern_score_valid.sum()).fillna(0).astype(int)
        strong=present.loc[present.compact_mixed_cluster & present.contact_strength.ge(.5)].groupby('frame').size()
        result[f'n_simultaneous_strong_relations_{radius}m']=result.frame.map(strong).fillna(0).astype(int)
        result[f'n_simultaneous_strong_relations_{radius}m']=np.where(result.local_metric_available,result[f'n_simultaneous_strong_relations_{radius}m'],np.nan)
        for name in ('approach_focus_separation_score','stable_close_score','observed_approach','observed_future_change','observed_window_soft_mean'):
            result[f'max_{name}_{radius}m']=result.frame.map(grouped[name].max())
        for direction in ('prev','next'):
            ambiguous=t.assign(ambiguous=t[f'match_{direction}_status'].isin(['geometric_ambiguous','observation_ambiguous'])).groupby('frame').ambiguous.mean()
            result[f'match_{direction}_ambiguous_fraction_{radius}m']=result.frame.map(ambiguous)
        values=source.v2_strength[:,0 if radius==3 else 1]
        means=np.full(len(result),np.nan);supports=np.zeros(len(result),int)
        for first in np.unique(source.first[source.periods>0]):
            last=source.last[first];sl=slice(first,last+1)
            a=pd.Series(np.where(source.strict[sl],values[sl],np.nan))
            means[sl]=a.rolling(2*r+1,center=True,min_periods=1).mean()
            supports[sl]=a.notna().astype(int).rolling(2*r+1,center=True,min_periods=1).sum()
        result[f'global_v2_moving_mean_{radius}m']=means
        result[f'global_v2_moving_support_{radius}m']=supports
    return result


def summarize_temporal(temporal,source):
    ids=np.searchsorted(source.ids,temporal.frame.to_numpy())
    official=temporal.loc[source.periods[ids]>0]
    def summary(t):
        full=t.loc[t.pattern_score_valid]
        return dict(cluster_rows=len(t),full_record_windows=int(t.full_record_window.sum()),
                    partial_bidirectional=int(t.partial_bidirectional_available.sum()),pattern_valid=int(t.pattern_score_valid.sum()),
                    global_count_changed=int(t.global_count_change.sum()),
                    observed_distributions={n:distribution(t[n]) for n in PATTERN_COLUMNS},
                    same_full_quality_distributions={n:distribution(full[n]) for n in PATTERN_COLUMNS},
                    future_end_reason={str(k):int(v) for k,v in t.future_end_reason.value_counts().items()},
                    next_matching={str(k):int(v) for k,v in t.match_next_status.value_counts().items()},
                    onset_censored=int(t.onset_censored.sum()),end_censored=int(t.end_censored.sum()))
    return dict(all=summary(temporal),official=summary(official),
                official_by_radius={str(int(r)):summary(t) for r,t in official.groupby('radius_m')})


def execute(*,root,output,cache,v2_dir,split_path,config_path):
    start=time.perf_counter();root=Path(root).resolve();output=Path(output).resolve();v2_dir=Path(v2_dir)
    if not output.is_relative_to(root/'data/derived/competition_judgement'):raise ValueError('Local-only output path required')
    if output.exists():raise FileExistsError('Refusing to overwrite prior output')
    config=json.loads(Path(config_path).read_text());validate_temporal_config(config)
    snapshot=preservation_snapshot(root)
    snapshot['files'].update({str(p):sha(p) for p in (root/'competition_judgement').rglob('*') if p.is_file() and '__pycache__' not in p.parts and p.name!='.DS_Store'})
    prior=json.loads((v2_dir/'verification.json').read_text())
    for name in ('local_frame_features.csv','local_clusters.csv'):
        if sha(v2_dir/name)!=prior['output_sha256'][name]:raise ValueError('v2-1 immutable output mismatch')
    frames=pd.read_csv(v2_dir/'local_frame_features.csv',usecols=FRAME_COLUMNS)
    clusters=pd.read_csv(v2_dir/'local_clusters.csv')
    store=open_frame_cache(cache);manifest=SplitManifest.load(split_path)
    if tuple(store.pitch)!=GAMES[GAME]['pitch']:raise ValueError('Official metre pitch mismatch')
    if len(store)!=GAMES[GAME]['n_video_frames'] or not np.array_equal(store.frame_ids,np.arange(len(store))):raise ValueError('Full frame axis required')
    if manifest.fingerprint!=make_split_manifest(half_video_blocks(GAME)).fingerprint:raise ValueError('Unexpected split')
    expected=np.zeros(len(store),int)
    for period,first,last in half_video_blocks(GAME):expected[first:last+1]=period
    if not np.array_equal(store.periods,expected):raise ValueError('Official half mismatch')
    source_info=json.loads((Path(cache)/'cache.json').read_text());paths=asset_paths()
    if source_info['source_sha256']!={'metadata':sha(paths['atd_meta']),'raw':sha(paths['atd_raw'])}:raise ValueError('Raw cache provenance mismatch')
    output.parent.mkdir(parents=True,exist_ok=True)
    with tempfile.TemporaryDirectory(prefix='.temporal_v2_2_',dir=output.parent) as temp:
        dest=Path(temp);json_write(dest/'preregistered_config.json',config)
        source=TemporalSource(store,frames,manifest,config,dest/'pair_cache',clusters)
        temporal=extract_temporal(source,clusters,dest/'cluster_temporal_features.csv',progress=True)
        frame=frame_temporal_features(source,frames,clusters,temporal)
        frame.to_csv(dest/'frame_temporal_features.csv',index=False,na_rep='NaN')
        report=dict(version=config['version'],config=config,summary=summarize_temporal(temporal,source),
                    frame_rows=len(frame),cluster_rows=len(temporal),runtime_s=time.perf_counter()-start,
                    existing_v1_states_unchanged=np.array_equal(frame.v1_candidate_state,frames.v1_candidate_state),
                    source_fingerprint=store.fingerprint,split_fingerprint=manifest.fingerprint,
                    offline_future_context=True,fit_performed=False,contest_labels_created=False)
        report['existing_v1_states_unchanged']=bool(report['existing_v1_states_unchanged'])
        json_write(dest/'temporal_report.json',report)
        (dest/'LOCAL_ONLY.txt').write_text('Offline licensed match derivatives. Local only, no uploads or push.\nNo ball, tags, GT, contest labels, neural training or candidate threshold fitting.\n')
        json_write(dest/'verification.json',dict(**verify_preserved(snapshot),v2_1_output_unchanged=True,
                   source_code_sha256={str(p.relative_to(root)):sha(p) for p in (root/'competition_judgement/src').rglob('*.py')},
                   output_sha256={p.name:sha(p) for p in (dest/'cluster_temporal_features.csv',dest/'frame_temporal_features.csv')}))
        if output.exists():raise FileExistsError('Output appeared during run')
        os.rename(dest,output)
    print(json.dumps(dict(output=str(output),frame_rows=len(frame),cluster_rows=len(temporal),pattern_valid=report['summary']['official']['pattern_valid'],runtime_s=report['runtime_s'])))
    return report


def main():
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--out-dir',required=True)
    p.add_argument('--cache',default='data/derived/relational_temporal_v1_20261109/raw_cache')
    p.add_argument('--v2-dir',default='data/derived/competition_judgement/local_v2_1_20261109')
    p.add_argument('--split-manifest',default='data/derived/relational_temporal_v1_20261109/split_manifest.json')
    p.add_argument('--config',default='competition_judgement/configs/temporal_relations_v2_2.json')
    a=p.parse_args();execute(root=Path.cwd(),output=a.out_dir,cache=a.cache,v2_dir=a.v2_dir,split_path=a.split_manifest,config_path=a.config)


if __name__=='__main__':main()
