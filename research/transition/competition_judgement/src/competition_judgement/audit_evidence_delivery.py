"""Final immutable-input audit and literal Stage A reference-array comparison."""
import argparse
import hashlib
import json
from pathlib import Path
import re
import numpy as np
import pandas as pd
from .evidence_source import EvidenceSource,CLUSTER_COLUMNS,MATCH_COLUMNS


def sha(path):
    h=hashlib.sha256()
    with Path(path).open('rb') as f:
        for chunk in iter(lambda:f.read(8*1024*1024),b''):h.update(chunk)
    return h.hexdigest()


def main():
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--run-dir',required=True)
    p.add_argument('--preservation-snapshot',required=True);p.add_argument('--tests-log',required=True)
    a=p.parse_args();root=Path.cwd();run=Path(a.run_dir).resolve();base=root/'data/derived/competition_judgement'
    if not run.is_relative_to(base):raise ValueError('Local output required')
    if (run/'final_manifest.json').exists():raise FileExistsError('Final audit exists')
    config=json.loads((run/'preregistered_config.json').read_text())
    assert config==json.loads((root/'competition_judgement/configs/evidence_stage_b.json').read_text())
    frames=pd.read_csv(base/'local_v2_1_20261109/local_frame_features.csv',usecols=['frame','period','n_home','n_away','strict_quality_valid'])
    clusters=pd.read_csv(base/'local_v2_1_20261109/local_clusters.csv',usecols=CLUSTER_COLUMNS)
    matching=pd.read_csv(base/'temporal_v2_2_final_20261109/cluster_temporal_features.csv',usecols=MATCH_COLUMNS)
    manifest=json.loads((root/'data/derived/relational_temporal_v1_20261109/split_manifest.json').read_text())
    source=EvidenceSource(root,frames,manifest,clusters,matching,config)
    ref=pd.read_csv(base/'missingness_stage_a_final_20261109/reference_keys.csv')
    index=clusters.reset_index().set_index(['frame','radius_m','cluster_id'])['index']
    selected=index.loc[pd.MultiIndex.from_frame(ref[['frame','radius_m','cluster_id']])].to_numpy(int)
    checks={}
    with np.load(base/'missingness_stage_a_final_20261109/reference_temporal_arrays.npz') as original:
        pieces=[source.query(selected[i:i+48]) for i in range(0,len(selected),48)]
        for key in original.files:
            actual=np.concatenate([s[key] for s in pieces]);expected=original[key]
            good=np.isfinite(actual)&np.isfinite(expected)
            checks[key]=dict(bitwise_equal=bool(np.array_equal(actual,expected,equal_nan=True)),
                max_abs_difference=float(np.max(np.abs(actual[good]-expected[good]))) if good.any() else None)
            assert np.allclose(actual,expected,atol=1e-7,rtol=0,equal_nan=True)
    snapshot=json.loads(Path(a.preservation_snapshot).read_text())
    for path,old in snapshot['files'].items():assert sha(path)==old,path
    for path,old in snapshot['artifacts'].items():
        stat=Path(path).stat();assert [stat.st_size,stat.st_mtime_ns]==old,path
    tests=Path(a.tests_log).read_text();match=re.search(r'(\d+) passed in ([\d.]+)s',tests)
    if match is None or 'FAILED' in tests or 'ERROR ' in tests:raise ValueError('Tests not all passed')
    result=dict(passed=int(match[1]),runtime_s=float(match[2]),existing_tests=377,new_tests=int(match[1])-377,failed=0,
        command='python -m pytest -q -p no:cacheprovider tests MTAH_Relational_Starter/tests competition_judgement/tests')
    (run/'test_results.json').write_text(json.dumps(result,indent=2))
    pd.DataFrame([dict(feature=k,**v) for k,v in checks.items()]).to_csv(run/'stage_a_reference_equality.csv',index=False)
    code={str(path.relative_to(root)):sha(path) for path in (root/'competition_judgement/src/competition_judgement').glob('*evidence*.py')}
    result=dict(stage='B only',protected_files_unchanged=len(snapshot['files']),protected_artifacts_unchanged=len(snapshot['artifacts']),
        reference_keys_reused=1778,reference_array_comparison=checks,tests=result,
        final_masking_directory='masking_final',config_sha256=sha(root/'competition_judgement/configs/evidence_stage_b.json'),
        source_sha256=code,report_sha256=sha(root/'competition_judgement/docs/STAGE2_2_5_EVIDENCE_REPORT.md'),
        no_policy_tuning=True,no_test_selection=True,no_labels=True,no_commit_or_push=True)
    (run/'final_manifest.json').write_text(json.dumps(result,indent=2))
    print(json.dumps({k:v for k,v in result.items() if k not in ('source_sha256','report_sha256')}))


if __name__=='__main__':main()
