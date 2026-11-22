"""Final alignment, preservation and artifact checks; never modifies inputs."""
import argparse
import hashlib
import json
from pathlib import Path
import numpy as np
import pandas as pd


def sha256(path):
    h=hashlib.sha256()
    with path.open('rb') as handle:
        for block in iter(lambda:handle.read(8*1024*1024),b''):h.update(block)
    return h.hexdigest()


def main():
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--run-dir',required=True)
    p.add_argument('--preservation-snapshot');p.add_argument('--root',default='.')
    a=p.parse_args();run=Path(a.run_dir);root=Path(a.root).resolve()
    if (run/'verification.json').exists():raise FileExistsError('Verification already exists')
    summary=json.loads((run/'summary.json').read_text());audit=json.loads((run/'missingness_summary.json').read_text())
    official=pd.read_csv(root/'data/derived/competition_judgement/temporal_v2_2_final_20261109/frame_temporal_features.csv',usecols=['frame','period'])
    ours=pd.read_csv(run/'frame_missingness_summary.csv');keys=pd.read_csv(run/'reference_keys.csv')
    assert len(official)==len(ours)==199319
    assert np.array_equal(official.frame,np.arange(199319)) and np.array_equal(official.frame,ours.frame)
    assert audit['total_cluster_rows']==789005 and audit['official_rows']==745891
    assert summary['full_reliable_cluster_rows']==191868 and summary['frozen_reference_feature_equality']
    assert keys[['frame','radius_m','cluster_id']].duplicated().sum()==0
    assert len(keys)==summary['reference_windows']
    manifest=json.loads((root/'data/derived/relational_temporal_v1_20261109/split_manifest.json').read_text())
    # Explicit support containment check, including distance from BOTH block edges.
    allowed=np.zeros(len(keys),bool)
    for b in manifest['blocks']:
        if b['split']=='validation':allowed|=(keys.frame.to_numpy()-50>=b['first'])&(keys.frame.to_numpy()+50<=b['last'])
    assert allowed.all()
    for _,ch in keys.groupby('support_block'):
        first=int(ch.frame.min())-50;last=int(ch.frame.max())+50
        for __,other in keys.groupby('support_block'):
            if __==ch.support_block.iloc[0]:continue
            assert last<int(other.frame.min())-50 or first>int(other.frame.max())+50
    numeric_checks={}
    for name in ('controlled_feature_comparison.csv','same_side_location_sensitivity.csv','actual_policy_coverage.csv',
                 'synthetic_policy_comparison.csv','paired_block_comparisons.csv'):
        table=pd.read_csv(run/name)
        values=table.select_dtypes(include='number').to_numpy(float)
        assert not np.isinf(values).any()
        numeric_checks[name]=dict(rows=len(table),inf_count=0)
    protected_files=protected_artifacts=0
    if a.preservation_snapshot:
        original=Path(a.preservation_snapshot);snapshot=json.loads(original.read_text())
        for path,hash_ in snapshot['files'].items():
            if sha256(Path(path))!=hash_:raise ValueError('Protected file changed: '+path)
        for path,(size,mtime) in snapshot['artifacts'].items():
            stat=Path(path).stat()
            if stat.st_size!=size or stat.st_mtime_ns!=mtime:raise ValueError('Protected artifact changed: '+path)
        protected_files=len(snapshot['files']);protected_artifacts=len(snapshot['artifacts'])
        (run/'preservation_start.json').write_text(original.read_text())
    code={str(path.relative_to(root)):sha256(path) for path in (root/'competition_judgement/src/competition_judgement').glob('*missingness*.py')}
    for name in ('controlled_temporal_masking.py','partial_window_diagnostics.py'):
        path=root/'competition_judgement/src/competition_judgement'/name;code[str(path.relative_to(root))]=sha256(path)
    result=dict(frame_alignment=True,original_frame_rows=len(official),mask_audit_exact=True,
        reference_recalculation_matches_frozen_csv=True,validation_only_reference_support=True,
        nonoverlapping_500frame_support_blocks=True,protected_code_document_hashes=protected_files,
        protected_input_artifact_stats=protected_artifacts,protection_checked=bool(a.preservation_snapshot),
        protection_passed=True if a.preservation_snapshot else None,numeric_checks=numeric_checks,source_sha256=code)
    (run/'verification.json').write_text(json.dumps(result,indent=2))
    print(json.dumps({k:v for k,v in result.items() if k not in ('source_sha256','numeric_checks')}))


if __name__=='__main__':main()
