"""Local Stage B output verification and current-spatial fallback enrichment."""
import argparse
import hashlib
import json
from pathlib import Path
import numpy as np
import pandas as pd
from .evidence_features import NAMES
from .evidence_frames import FrameEvidence,SCORES


def digest(path):
    h=hashlib.sha256()
    with Path(path).open('rb') as f:
        for chunk in iter(lambda:f.read(8*1024*1024),b''):h.update(chunk)
    return h.hexdigest()


def spatial_representatives(clusters,n_frames):
    """Largest observed N, then smaller radius/ID; not a contest decision."""
    ordered=clusters.sort_values(['frame','n_players','radius_m','cluster_id'],ascending=[True,False,True,True])
    first=ordered.drop_duplicates('frame').set_index('frame')
    columns=['radius_m','cluster_id','n_players','n_home','n_away','cluster_centroid_x','cluster_centroid_y',
        'cluster_radius_m','cluster_max_pairwise_distance_m','cross_pair_density','contact_strength']
    result=first[columns].reindex(np.arange(n_frames)).add_prefix('spatial_representative_').reset_index(names='frame')
    return result


def main():
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--run-dir',required=True)
    p.add_argument('--preservation-snapshot');p.add_argument('--root',default='.')
    a=p.parse_args();root=Path(a.root).resolve();run=Path(a.run_dir).resolve()
    if not run.is_relative_to(root/'data/derived/competition_judgement'):raise ValueError('Local output required')
    if (run/'verification.json').exists():raise FileExistsError('Verification already exists')
    config=json.loads((run/'preregistered_config.json').read_text());summary=json.loads((run/'summary.json').read_text())
    assert summary['config_sha256']==digest(root/'competition_judgement/configs/evidence_stage_b.json')
    frames=pd.read_csv(root/'data/derived/competition_judgement/local_v2_1_20261109/local_frame_features.csv',
        usecols=['frame','period','quality_tier','strict_quality_valid','max_local_contact_strength_3m','max_local_contact_strength_5m'])
    clusters=pd.read_csv(root/'data/derived/competition_judgement/local_v2_1_20261109/local_clusters.csv')
    old=pd.read_csv(root/'data/derived/competition_judgement/temporal_v2_2_final_20261109/cluster_temporal_features.csv',
        usecols=['frame','radius_m','cluster_id','pattern_score_valid'])
    reducer=FrameEvidence(frames);keys=['frame','radius_m','cluster_id'];start=full=0
    for ch in pd.read_csv(run/'cluster_evidence.csv.gz',chunksize=20000,low_memory=False,dtype={k+'_mask_hex':str for k in ('record','observed','reliable')}):
        expected=old.iloc[start:start+len(ch)]
        assert np.array_equal(ch[keys].to_numpy(),expected[keys].to_numpy())
        assert np.array_equal(ch.full_reliable_window,expected.pattern_score_valid)
        assert not np.isinf(ch.select_dtypes(include='number').to_numpy(float)).any()
        for mode in ('full','partial'):
            for name in SCORES:
                x=ch[mode+'_'+name].to_numpy(float);good=np.isfinite(x)
                assert ((x[good]>=-1e-12)&(x[good]<=1+1e-12)).all()
                assert not good[~ch.current_strict.to_numpy(bool)].any()
                if mode=='full':assert not good[~ch.full_reliable_window.to_numpy(bool)].any()
            pvalues=ch[[mode+'_'+NAMES[k] for k in ('P1','P2','P3','P4')]].to_numpy(float)
            pm=np.isfinite(pvalues);pn=pm.sum(1)
            target=np.divide(np.where(pm,pvalues,0).sum(1),pn,out=np.full(len(ch),np.nan),where=pn>=2)
            assert np.allclose(ch[mode+'_positive_evidence_score'],target,equal_nan=True)
            n1=ch[mode+'_'+NAMES['N1']].to_numpy(float);n3=ch[mode+'_'+NAMES['N3']].to_numpy(float)
            n2=ch[mode+'_'+NAMES['N2']].to_numpy(float);persistence=np.where(np.isfinite(n1),n1,n3)
            stack=np.stack([persistence,n2],1);v=np.isfinite(stack);count=v.sum(1)
            target=np.divide(np.where(v,stack,0).sum(1),count,out=np.full(len(ch),np.nan),where=count>0)
            assert np.allclose(ch[mode+'_negative_evidence_score'],target,equal_nan=True)
        full+=int(ch.full_reliable_window.sum())
        reducer.add(ch,ch,np.arange(start,start+len(ch)));start+=len(ch)
    assert start==789005 and full==191868
    frame=pd.read_csv(run/'frame_evidence.csv.gz');recomputed=reducer.table()
    assert len(frame)==199319 and np.array_equal(frame.frame,np.arange(199319))
    for metric in reducer.metrics:
        for suffix in ('_max','_source_radius_m','_source_cluster_id','_valid_n'):
            assert np.allclose(frame[metric+suffix],recomputed[metric+suffix],equal_nan=True)
    for mode in ('full','partial'):
        assert np.allclose(frame[mode+'_negative_at_positive_representative'],recomputed[mode+'_negative_at_positive_representative'],equal_nan=True)
    spatial=spatial_representatives(clusters,len(frames))
    enriched=frame.merge(spatial,on='frame',validate='one_to_one').merge(
        frames[['frame','max_local_contact_strength_3m','max_local_contact_strength_5m']],on='frame',validate='one_to_one')
    assert not np.isinf(enriched.select_dtypes(include='number').to_numpy(float)).any()
    enriched.to_csv(run/'frame_evidence_with_spatial.csv.gz',index=False,compression=dict(method='gzip',compresslevel=3))
    validation_ids=pd.read_csv(run/'validation_evidence.csv.gz',usecols=['frame']).frame.unique()
    validation=enriched.loc[enriched.frame.isin(validation_ids)]
    joint=[]
    for mode in ('full','partial'):
        pscore=validation[mode+'_positive_evidence_score_max'];nscore=validation[mode+'_negative_evidence_score_max']
        h=config['diagnostics']['high_band'];both=(pscore>=h)&(nscore>=h)
        same=(validation[mode+'_positive_evidence_score_source_radius_m']==validation[mode+'_negative_evidence_score_source_radius_m'])&(
            validation[mode+'_positive_evidence_score_source_cluster_id']==validation[mode+'_negative_evidence_score_source_cluster_id'])
        coherent=(pscore>=h)&(validation[mode+'_negative_at_positive_representative']>=h)
        joint.append(dict(mode=mode,validation_frames=len(validation),independent_maxima_both_high_n=int(both.sum()),
            independent_maxima_different_source_n=int((both&~same).sum()),positive_representative_both_high_n=int(coherent.sum()),
            warning='Independent maxima may refer to different spatial relations; not one event.'))
    (run/'validation_frame_joint_summary.json').write_text(json.dumps(joint,indent=2))
    protected_files=protected_artifacts=0
    if a.preservation_snapshot:
        snapshot_path=Path(a.preservation_snapshot);snapshot=json.loads(snapshot_path.read_text())
        for path,hash_ in snapshot['files'].items():
            if digest(path)!=hash_:raise ValueError('Protected file changed: '+path)
        for path,stats in snapshot['artifacts'].items():
            s=Path(path).stat()
            if [s.st_size,s.st_mtime_ns]!=stats:raise ValueError('Protected artifact changed: '+path)
        protected_files=len(snapshot['files']);protected_artifacts=len(snapshot['artifacts'])
        (run/'preservation_start.json').write_text(snapshot_path.read_text())
    code={str(path.relative_to(root)):digest(path) for path in (root/'competition_judgement/src/competition_judgement').glob('*evidence*.py')}
    result=dict(frame_rows=len(frame),cluster_rows=start,keys_exact=True,original_v2_2_full_mask_exact=True,
        independent_anchor_maxima_reproduced=True,coherent_negative_at_positive_anchor=True,
        all_scores_bounded=True,inf_count=0,negative_family_dedup_exact=True,current_qc_failure_all_temporal_nan=True,
        protected_files_unchanged=protected_files,protected_artifacts_unchanged=protected_artifacts,
        protection_checked=bool(a.preservation_snapshot),source_sha256=code)
    (run/'verification.json').write_text(json.dumps(result,indent=2))
    print(json.dumps({k:v for k,v in result.items() if k!='source_sha256'}))


if __name__=='__main__':main()
