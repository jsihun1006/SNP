"""Recompute joins/decisions/sources and verify immutable pre-v2-3 artifacts."""
import argparse
import json
from pathlib import Path
import re
import numpy as np
import pandas as pd
from .candidate_decision_v23 import KEY, STATES, CANDIDATES, decide_clusters
from .candidate_frames_v23 import CandidateFrames
from .run_candidate_v23 import profile_columns,local_columns,sha


def assert_table_equal(a,b):
    if list(a)!=list(b) or len(a)!=len(b):raise AssertionError('Output schema/row mismatch')
    for key in a:
        if pd.api.types.is_numeric_dtype(a[key]):np.testing.assert_allclose(a[key],b[key],equal_nan=True,atol=1e-12,rtol=0)
        else:np.testing.assert_array_equal(a[key].fillna('<NA>'),b[key].fillna('<NA>'))


def main():
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--run-dir',required=True)
    p.add_argument('--preservation-snapshot',required=True);p.add_argument('--tests-log',required=True)
    a=p.parse_args();root=Path.cwd();out=Path(a.run_dir).resolve()
    if not out.is_relative_to(root/'data/derived/competition_judgement'):raise ValueError('Local output required')
    if (out/'verification.json').exists():raise FileExistsError('Verification already exists')
    config=json.loads((out/'preregistered_config.json').read_text());summary=json.loads((out/'summary.json').read_text())
    assert summary['config_sha256']==sha(root/'competition_judgement/configs/candidate_v23.json')
    frames=pd.read_csv(out.parent/'local_v2_1_20261109/local_frame_features.csv',usecols=['frame','period','strict_quality_valid','n_home','n_away','n_local_clusters_3m','n_local_clusters_5m'])
    reducer=CandidateFrames(frames,config);chunk=config['chunk_rows'];rows=0
    profiles=pd.read_csv(out.parent/'evidence_pattern_b1_final_20261110/cluster_pattern_profiles.csv.gz',usecols=profile_columns(),chunksize=chunk)
    locals_=pd.read_csv(out.parent/'local_temporal_b2_final_20261110/cluster_local_temporal.csv.gz',usecols=local_columns(),chunksize=chunk)
    for c in pd.read_csv(out/'cluster_candidate_decisions.csv.gz',chunksize=chunk):
        old=next(profiles).reset_index(drop=True);local=next(locals_).reset_index(drop=True)
        assert np.array_equal(c[KEY].to_numpy(),old[KEY].to_numpy())
        expected=pd.concat([old[KEY],decide_clusters(old,old,local,config)],axis=1)
        assert_table_equal(c,expected)
        assert c.decision_state.isin(STATES).all() and not c.duplicated(KEY).any()
        candidate=c.decision_state.isin(CANDIDATES)
        assert c.loc[candidate,'spatial_gate_valid'].all() and (c.loc[candidate,'decision_basis_mask']>0).all()
        assert (c.loc[~candidate,'decision_basis_mask']==0).all()
        known_absence=c.decision_state=='NO_DETECTED_SIGNAL'
        assert (c.loc[known_absence,'dynamic_support_mask']==63).all()
        assert not np.isinf(c.select_dtypes(include='number').to_numpy(float)).any()
        assert not any('score' in k or 'probability' in k for k in c)
        reducer.add(old,c);rows+=len(c)
    for iterator in (profiles,locals_):
        try:next(iterator);raise AssertionError('Frozen row count mismatch')
        except StopIteration:pass
    actual=pd.read_csv(out/'frame_candidate_decisions.csv.gz');assert_table_equal(actual,reducer.table())
    assert rows==789005 and len(actual)==199319 and np.array_equal(actual.frame,np.arange(199319))
    assert not actual.loc[~actual.current_strict,'decision_state'].isin(CANDIDATES).any()
    dictionary=json.loads((out/'data_dictionary.json').read_text());assert list(c)==dictionary['columns'] and list(actual)==dictionary['frame_columns']
    frozen=json.loads((out/'legacy_input_preservation.json').read_text())
    for name,hash_ in frozen['sha256'].items():assert sha(root/name)==hash_,name
    snap=json.loads(Path(a.preservation_snapshot).read_text())
    for name,hash_ in snap['files'].items():assert sha(root/name)==hash_,name
    for name,stat in snap['artifacts'].items():
        now=(root/name).stat();assert [now.st_size,now.st_mtime_ns]==stat,name
    log=Path(a.tests_log).read_text();match=re.search(r'(\d+) passed in ([\d.]+)s',log)
    assert match and int(match[1])>477 and 'FAILED' not in log and 'ERROR ' not in log
    tests=dict(passed=int(match[1]),previous=477,new=int(match[1])-477,failed=0,runtime_s=float(match[2]))
    result=dict(cluster_rows=rows,frame_rows=len(actual),original_keys_exact=True,all_cluster_decisions_recomputed=True,
                frame_representatives_and_radius_sources_recomputed=True,no_cross_anchor_fusion=True,
                no_unknown_as_dynamic_negative=True,no_alternative_veto=True,all_candidates_current_spatial_valid=True,
                frozen_input_sha256_exact=True,protected_files_unchanged=len(snap['files']),protected_artifacts_unchanged=len(snap['artifacts']),tests=tests,
                source_sha256={str(p.relative_to(root)):sha(p) for p in (root/'competition_judgement/src/competition_judgement').glob('*v23.py')})
    (out/'preservation_start.json').write_text(Path(a.preservation_snapshot).read_text())
    (out/'verification.json').write_text(json.dumps(result,indent=2))
    print(json.dumps({k:v for k,v in result.items() if k!='source_sha256'}))


if __name__=='__main__':main()
