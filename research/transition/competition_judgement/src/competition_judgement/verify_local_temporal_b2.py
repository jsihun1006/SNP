"""Re-read sidecars, check joins/sources and preservation without changing legacy."""
import argparse
import hashlib
import json
from pathlib import Path
import re
import numpy as np
import pandas as pd
from .local_temporal_features import SIGNALS, OUTCOMES, SEQUENCES
from .local_temporal_frames import LocalTemporalFrames


def sha(path):
    digest = hashlib.sha256()
    with Path(path).open('rb') as stream:
        for chunk in iter(lambda: stream.read(8 * 1024 * 1024), b''): digest.update(chunk)
    return digest.hexdigest()


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--run-dir', required=True); parser.add_argument('--preservation-snapshot', required=True)
    parser.add_argument('--tests-log', required=True)
    args = parser.parse_args(); root = Path.cwd(); out = Path(args.run_dir).resolve()
    if not out.is_relative_to(root / 'data/derived/competition_judgement'): raise ValueError('Local results only')
    if (out / 'verification.json').exists(): raise FileExistsError('Verification already exists')
    summary = json.loads((out / 'summary.json').read_text())
    config = json.loads((out / 'preregistered_config.json').read_text())
    assert summary['config_sha256'] == sha(root / 'competition_judgement/configs/local_temporal_b2.json')
    frames = pd.read_csv(out.parent / 'local_v2_1_20261109/local_frame_features.csv', usecols=['frame'])
    keys = pd.read_csv(out.parent / 'local_v2_1_20261109/local_clusters.csv', usecols=['frame', 'radius_m', 'cluster_id'])
    reducer = LocalTemporalFrames(frames); rows = 0; numeric_max = 0.
    for c in pd.read_csv(out / 'cluster_local_temporal.csv.gz', chunksize=20000):
        n = len(c)
        assert np.array_equal(c[['frame', 'radius_m', 'cluster_id']].to_numpy(), keys.iloc[rows:rows+n].to_numpy())
        assert not c.duplicated(['frame', 'radius_m', 'cluster_id']).any()
        assert not np.isinf(c.select_dtypes(include='number').to_numpy(float)).any()
        assert not any('evidence_score' in name or name in ('P1', 'P2', 'P3', 'P4', 'N1', 'N2', 'N3') or name.endswith('_mask') for name in c)
        for signal in SIGNALS: assert c[signal].isin(OUTCOMES).all()
        for name, (lo, hi) in config['scopes'].items():
            scale = name.split('_')[1]; direction = 'rise' if name.startswith('past') else 'fall'
            complete = c[name + '_reliable_n'] == hi - lo + 1
            known = c['local_' + direction + '_' + scale] != 'UNKNOWN'
            assert complete[known].all() and (c.loc[known, name + '_local_changed_n'] == 0).all()
            assert (c.loc[known, name + '_max_missing_run'] == 0).all()
            assert c.loc[~known, name + '_delta_soft'].isna().all()
            proximity_known = c['local_proximity_' + name] != 'UNKNOWN'
            assert complete[proximity_known].all()
            assert (c[name + '_run_n'].dropna() <= hi - lo + 1).all()
        for scale in ('short', 'near'):
            rise = c['local_rise_' + scale]; fall = c['local_fall_' + scale]
            codes = np.array([OUTCOMES.index(a) * 3 + OUTCOMES.index(b) for a, b in zip(rise, fall)])
            assert np.array_equal(c['local_sequence_' + scale], np.asarray(SEQUENCES)[codes])
        motion = c.local_motion_status != 'UNKNOWN'
        assert (c.loc[motion, 'motion_run_n'] >= 2).all()
        assert np.array_equal(c.loc[motion, 'motion_run_n'], c.loc[motion, 'motion_last_offset'] - c.loc[motion, 'motion_first_offset'] + 1)
        assert np.allclose(c.loc[motion, 'motion_span_s'], (c.loc[motion, 'motion_run_n'] - 1) / 25)
        assert c.loc[~motion, 'motion_net_m'].isna().all()
        reducer.add(c, c); rows += n
    actual = pd.read_csv(out / 'frame_local_temporal.csv.gz'); expected = reducer.table()
    assert rows == 789005 and len(actual) == 199319 and np.array_equal(actual.frame, np.arange(199319))
    for name in expected:
        if pd.api.types.is_numeric_dtype(expected[name]): np.testing.assert_allclose(actual[name], expected[name], equal_nan=True, atol=1e-12, rtol=0)
        else: np.testing.assert_array_equal(actual[name], expected[name])
    preservation = json.loads((out / 'legacy_pattern_preservation.json').read_text())
    assert preservation['exact_A_B_C_D_outcomes'] and preservation['exact_cluster_keys']
    numeric_max = max(preservation['numeric_max_abs_difference'].values())
    assert numeric_max < 1e-12
    snapshot = json.loads(Path(args.preservation_snapshot).read_text())
    for path, expected_hash in snapshot['files'].items(): assert sha(root / path) == expected_hash, path
    for path, stat in snapshot['artifacts'].items():
        actual_stat = (root / path).stat()
        assert [actual_stat.st_size, actual_stat.st_mtime_ns] == stat, path
    log = Path(args.tests_log).read_text(); match = re.search(r'(\d+) passed in ([\d.]+)s', log)
    assert match and int(match[1]) > 440 and 'FAILED' not in log and 'ERROR ' not in log
    dictionary = json.loads((out / 'data_dictionary.json').read_text())
    assert list(c) == dictionary['columns'] and list(actual) == dictionary['frame_columns']
    results = dict(frame_rows=len(actual), cluster_rows=rows, original_keys_exact=True,
        frame_sources_and_sequences_reproduced=True, no_cross_anchor_sequence_fusion=True,
        all_scope_complete_guards=True, all_motion_runs_contiguous=True, legacy_recalculated_max_abs_difference=numeric_max,
        protected_files_unchanged=len(snapshot['files']), protected_artifacts_unchanged=len(snapshot['artifacts']),
        tests=dict(passed=int(match[1]), previous=440, new=int(match[1])-440, failed=0, runtime_s=float(match[2])),
        source_sha256={str(p.relative_to(root)): sha(p) for p in (root / 'competition_judgement/src/competition_judgement').glob('*local_temporal*.py')})
    (out / 'preservation_start.json').write_text(Path(args.preservation_snapshot).read_text())
    (out / 'verification.json').write_text(json.dumps(results, indent=2))
    print(json.dumps({k: v for k, v in results.items() if k != 'source_sha256'}))


if __name__ == '__main__': main()
