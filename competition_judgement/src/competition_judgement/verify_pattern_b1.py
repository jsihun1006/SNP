"""Re-read all profiles, verify immutable inputs and publish a shared dictionary."""
import argparse
import hashlib
import json
from pathlib import Path
import re
import numpy as np
import pandas as pd
from .pattern_values import FEATURES
from .pattern_profiles import PATTERNS
from .pattern_frames import PatternFrames


def sha(path):
    h=hashlib.sha256()
    with Path(path).open('rb') as f:
        for chunk in iter(lambda:f.read(8*1024*1024),b''):h.update(chunk)
    return h.hexdigest()


def main():
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--run-dir',required=True)
    p.add_argument('--preservation-snapshot',required=True);p.add_argument('--tests-log',required=True)
    a=p.parse_args();root=Path.cwd();out=Path(a.run_dir).resolve()
    if not out.is_relative_to(root/'data/derived/competition_judgement'):raise ValueError('Local output required')
    if (out/'verification.json').exists():raise FileExistsError('Final verification exists')
    summary=json.loads((out/'summary.json').read_text());config=json.loads((out/'preregistered_config.json').read_text())
    assert summary['config_sha256']==sha(root/'competition_judgement/configs/pattern_b1.json')
    frames=pd.read_csv(root/'data/derived/competition_judgement/local_v2_1_20261109/local_frame_features.csv',usecols=['frame','period','n_home','n_away','strict_quality_valid'])
    old=pd.read_csv(root/'data/derived/competition_judgement/local_v2_1_20261109/local_clusters.csv',usecols=['frame','radius_m','cluster_id'])
    reducer=PatternFrames(frames);rows=0;columns=None;full=0
    for c in pd.read_csv(out/'cluster_pattern_profiles.csv.gz',chunksize=20000,low_memory=False,dtype={k+'_mask':str for k in ('record','observed','reliable','strict','global_changed','local_changed','path')}):
        columns=list(c);n=len(c)
        assert np.array_equal(c[['frame','radius_m','cluster_id']].to_numpy(),old.iloc[rows:rows+n].to_numpy())
        assert not any('evidence_score' in k or 'candidate_score' in k or 'score_bound' in k for k in columns)
        assert not np.isinf(c.select_dtypes(include='number').to_numpy(float)).any()
        for f in FEATURES:
            state=c[f+'_state'];known=state!='UNKNOWN';values=c[f].to_numpy(float)
            assert state.isin(['COMPLETE','EXPERIMENTAL_PARTIAL','UNKNOWN']).all()
            assert np.isfinite(values[known]).all() and np.isnan(values[~known]).all()
            assert ((values[known]>=-1e-12)&(values[known]<=1+1e-12)).all()
            assert not known[~c.current_strict.to_numpy(bool)].any()
        available=sum((c[f+'_state']!='UNKNOWN').to_numpy(int)<<i for i,f in enumerate(FEATURES))
        assert np.array_equal(c.available_mask,available)
        for pattern in PATTERNS:
            hit=c[pattern+'_outcome']=='SUPPORTED';absent=c[pattern+'_outcome']=='NOT_OBSERVED'
            assert (c.loc[hit,pattern+'_seq']=='VERIFIED').all() and (c.loc[hit,pattern+'_obs']=='COMPLETE').all()
            assert (c.loc[absent,pattern+'_obs']=='COMPLETE').all()
            req=sum(1<<FEATURES.index(f) for f in config['required_evidence'][pattern])
            assert np.array_equal(c[pattern+'_missing_mask'],req&~available)
        b=c.B_outcome=='SUPPORTED'
        assert (c.loc[b,'B_structure_changed']==1).all()
        assert (c.loc[b,'B_peak_offset'].abs()<=config['sequence']['peak_center_radius_frames']).all()
        assert (c.loc[b,'B_rise_end_offset']<=c.loc[b,'B_peak_offset']).all() and (c.loc[b,'B_peak_offset']<=c.loc[b,'B_fall_start_offset']).all()
        d=c.D_outcome=='SUPPORTED';assert c.loc[d,'path_inside_anchor'].all()
        reducer.add(c,c);rows+=n;full+=int(c.legacy_101_full.sum())
    assert rows==789005 and full==191868
    actual=pd.read_csv(out/'frame_pattern_profiles.csv.gz');expected=reducer.table()
    assert len(actual)==199319 and np.array_equal(actual.frame,np.arange(199319))
    for name in expected:
        if pd.api.types.is_numeric_dtype(expected[name]):assert np.allclose(actual[name],expected[name],equal_nan=True)
        else:assert np.array_equal(actual[name].to_numpy(),expected[name].to_numpy())
    snap=json.loads(Path(a.preservation_snapshot).read_text())
    for path,hash_ in snap['files'].items():assert sha(path)==hash_,path
    for path,stat_ in snap['artifacts'].items():
        stat=Path(path).stat();assert [stat.st_size,stat.st_mtime_ns]==stat_,path
    testlog=Path(a.tests_log).read_text();match=re.search(r'(\d+) passed in ([\d.]+)s',testlog)
    assert match and 'FAILED' not in testlog and 'ERROR ' not in testlog
    tests=dict(passed=int(match[1]),runtime_s=float(match[2]),existing=404,new=int(match[1])-404,failed=0)
    dictionary=dict(version=config['version'],feature_bits={f:1<<i for i,f in enumerate(FEATURES)},dependencies=config['dependencies'],
        masks={'record_mask':'101-bit same half/split/record support','observed_mask':'101-bit Anchor contains both teams',
        'reliable_mask':'101-bit frozen v2-2 QC and count support','strict_mask':'101-bit Strict QC AND record',
        'global_changed_mask':'101-bit known global count differs from current','local_changed_mask':'101-bit observed local team count differs',
        'path_mask':'21-bit separate moving geometric-corridor support'},
        pack='little-bit-order hex; preserve strings and leading zeros; no per-feature mask duplication',
        feature_scope='All feature counts/gaps use dependencies. N2 uses path_mask; other features use reliable_mask.',
        required_evidence=config['required_evidence'],observation=['COMPLETE','EXPERIMENTAL_PARTIAL','UNKNOWN'],
        sequence=['VERIFIED','UNVERIFIED','NOT_APPLICABLE'],outcome=['SUPPORTED','NOT_OBSERVED','UNKNOWN'],
        meanings={'COMPLETE':'Required observations and composition/matching available; not identity/football correctness',
        'EXPERIMENTAL_PARTIAL':'Frozen Stage B partial guard passed; no operating-reliability claim',
        'UNKNOWN':'Insufficient data, quality, scope or interpretability. Value is NaN',
        'SUPPORTED':'Predeclared geometric pattern conditions verified at one anchor',
        'NOT_OBSERVED':'Adequate data but defined geometric pattern not met',
        'VERIFIED':'Geometric sequence only; not a contest event'},
        reasons={'OK':'condition/support satisfied','NO_RECORD':'outside official half/split or record unavailable',
        'CURRENT_UNOBSERVED':'current Anchor lacks both teams','CURRENT_QC':'current Strict QC failed',
        'GLOBAL_COMPOSITION':'known global H/A count change','LOCAL_COMPOSITION':'observed local H/A count change',
        'BAND_SUPPORT':'frozen Partial band minimum failed','CENTER_GAP':'frozen Partial central support failed',
        'LONG_GAP':'frozen Partial run maximum failed','GEOMETRY':'insufficient component geometry',
        'MATCHING':'adjacent matching observation/boundary unknown','PATH':'N2 geometric corridor invalid',
        'UNDEFINED':'mathematically undefined feature','EVIDENCE_UNKNOWN':'Evidence-only import has no qualified value',
        'END_AMBIGUOUS':'future end not interpretable',
        'MISSING_EVIDENCE':'required Evidence unknown','SEQUENCE_PARTIAL':'Partial estimates do not verify full temporal order',
        'SEQUENCE_UNVERIFIED':'sequence support unavailable','NO_TIME_ARRAY':'Evidence-only import lacks raw sequence',
        'CONDITION_NOT_MET':'fully observed geometric criterion false','ANCHOR_SCOPE_OR_PATH_UNKNOWN':'path invalid or leaves seed region',
        'ANCHOR_SUPPORTED':'frame has one supported source','NO_OBSERVED_MIXED_ANCHOR':'valid observed empty current graph',
        'ALL_ANCHORS_NOT_MET':'all current anchors adequate but condition false','ANCHORS_UNKNOWN':'at least one unevaluable anchor'},
        long_columns=columns,frame_semantics='pattern-specific existence/source and separate radius view counts, no event sum or score fusion',
        legacy_sidecar='legacy_diagnostics.csv.gz; read-only copy of Stage B composites, never pattern inputs')
    (out/'data_dictionary.json').write_text(json.dumps(dictionary,indent=2))
    (out/'test_results.json').write_text(json.dumps(tests,indent=2))
    (out/'preservation_start.json').write_text(Path(a.preservation_snapshot).read_text())
    result=dict(frame_rows=len(actual),cluster_rows=rows,original_keys_exact=True,full101_mask_count=full,
        independent_equation_equality=json.loads(pd.read_csv(out/'evidence_value_equality.csv').to_json(orient='records')),
        frame_pattern_sources_reproduced=True,no_main_composite=True,no_unknown_zero_fill=True,no_partial_sequence_upgrade=True,
        protected_files_unchanged=len(snap['files']),protected_artifacts_unchanged=len(snap['artifacts']),tests=tests,
        source_sha256={str(p.relative_to(root)):sha(p) for p in (root/'competition_judgement/src/competition_judgement').glob('*pattern*.py')})
    (out/'verification.json').write_text(json.dumps(result,indent=2));print(json.dumps({k:v for k,v in result.items() if k not in ('source_sha256','independent_equation_equality')}))


if __name__=='__main__':main()
