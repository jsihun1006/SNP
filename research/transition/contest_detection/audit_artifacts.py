"""Read-only checks of preserved BAR-ATH artifacts; no inference or training.

Usage: python audit_artifacts.py SOURCE_ROOT NEW_REPORT_DIRECTORY
The report contains hashes, schemas and aggregate counts, never frame-level data.
"""
from pathlib import Path
import argparse
from datetime import date
import hashlib
import json
import sys

import numpy as np
import pandas as pd


def sha(path):
    digest = hashlib.sha256()
    with path.open('rb') as source:
        for block in iter(lambda: source.read(8 * 1024 * 1024), b''):
            digest.update(block)
    return digest.hexdigest()


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('source_root', type=Path)
    parser.add_argument('report_directory', type=Path)
    args = parser.parse_args()
    root, out = args.source_root.resolve(), args.report_directory.resolve()
    base = root / 'data/derived/competition_judgement'
    if out.exists() or out.is_relative_to(base):
        raise ValueError('Use a new report directory outside existing match artifacts')
    sys.path.insert(0, str(root))
    from games import GAME, half_video_blocks
    paths = {p for p in base.rglob('*') if p.is_file() and p.suffix in {'.json', '.csv', '.gz', '.npy', '.npz'}}
    for folder in ('data/derived/relational_phase1_v1_20261109',
                   'data/derived/relational_temporal_v1_20261109/raw_cache'):
        paths.update(p for p in (root / folder).glob('*') if p.suffix in {'.json', '.csv', '.npy'})
    paths.add(root / 'data/derived/relational_temporal_v1_20261109/split_manifest.json')
    paths.update(p for p in (root / 'data/tactical').glob('*_atd.*') if p.suffix in {'.xml', '.txt'})
    before = {p: (p.stat().st_size, p.stat().st_mtime_ns) for p in paths}
    records = [{'path': p.relative_to(root).as_posix(), 'bytes': before[p][0], 'sha256': sha(p)} for p in sorted(paths)]
    frozen = base / 'candidate_v23_final_20261110'
    frames = pd.read_csv(frozen / 'frame_candidate_decisions.csv.gz')
    clusters = pd.read_csv(frozen / 'cluster_candidate_decisions.csv.gz')
    assert np.array_equal(frames.frame.to_numpy(), np.arange(199319))
    assert len(clusters) == 789005 and not clusters.duplicated(['frame', 'radius_m', 'cluster_id']).any()
    assert clusters.set_index(['frame', 'radius_m', 'cluster_id']).index.is_monotonic_increasing
    candidate = clusters.decision_state.isin(['PATTERN_CANDIDATE', 'LOCAL_CANDIDATE'])
    assert clusters.loc[candidate, 'spatial_gate_valid'].all()
    assert clusters.loc[candidate, 'decision_basis_mask'].gt(0).all()
    assert clusters.loc[~candidate, 'decision_basis_mask'].eq(0).all()
    assert clusters.loc[clusters.decision_state.eq('NO_DETECTED_SIGNAL'), 'dynamic_support_mask'].eq(63).all()
    split = json.loads((root / 'data/derived/relational_temporal_v1_20261109/split_manifest.json').read_text(encoding='utf-8'))
    validation = np.zeros(len(frames), bool)
    for block in split['blocks']:
        if block['split'] == 'validation':
            validation[block['first']:block['last'] + 1] = True
    validation_ids = set(frames.loc[validation, 'frame'])
    validation_clusters = clusters.frame.isin(validation_ids)
    counts = clusters.loc[validation_clusters, 'decision_state'].value_counts().to_dict()
    assert counts == {'PATTERN_CANDIDATE': 8850, 'LOCAL_CANDIDATE': 3503, 'REVIEW': 74313, 'NO_DETECTED_SIGNAL': 1926, 'INSUFFICIENT': 23694}
    assert frames.loc[validation, 'decision_state'].isin(['PATTERN_CANDIDATE', 'LOCAL_CANDIDATE']).sum() == 6139
    v1 = pd.read_csv(base / 'stage1_v1_final_20261109/candidate_frames.csv')
    segments = pd.read_csv(base / 'stage1_v1_final_20261109/candidate_segments.csv')
    official = np.zeros(len(v1), bool)
    for half, first, last in half_video_blocks(GAME):
        official[first:last + 1] = True
    v1counts = v1.loc[official, 'candidate_state'].value_counts().to_dict()
    assert v1counts == {'candidate': 9936, 'not_candidate': 101797, 'insufficient_observation': 47124}
    assert segments.candidate_state.eq('candidate').sum() == 446
    hashes = {r['path']: r['sha256'] for r in records}
    preserved = json.loads((frozen / 'legacy_input_preservation.json').read_text(encoding='utf-8'))['sha256']
    assert all(hashes[name] == value for name, value in preserved.items())
    noise = pd.read_csv(frozen / 'tracking_robustness.csv')
    jitter = noise.loc[(noise.condition == 'jitter_0.1') & (noise.group == 'all')]
    assert int(jitter.new_candidate_n.sum()) == 213 and int(jitter.lost_candidate_n.sum()) == 48
    assert all((p.stat().st_size, p.stat().st_mtime_ns) == stat for p, stat in before.items())
    summary = dict(date=date.today().isoformat(), mode='Read-only existing results; no inference or training',
                   frame_rows=len(frames), cluster_rows=len(clusters), frame_columns=list(frames), cluster_columns=list(clusters),
                   frame_axis_exact=True, cluster_key_unique_and_sorted=True, candidate_gate_and_basis_valid=True,
                   known_absence_support_valid=True, frozen_inputs_sha256_exact=True, artifact_size_mtime_preserved=True,
                   validation_frames=int(validation.sum()), validation_clusters=int(validation_clusters.sum()),
                   validation_cluster_states=counts, validation_candidate_frames=6139,
                   validation_candidate_fraction=6139 / int(validation.sum()),
                   v1_official_frames=int(official.sum()), v1_official_states=v1counts, v1_candidate_segments=446,
                   jitter_0_1m_probe_count=3191, jitter_0_1m_new_candidates=213, jitter_0_1m_lost_candidates=48)
    out.mkdir(parents=True)
    (out / 'artifact_manifest.json').write_text(json.dumps(dict(date=summary['date'], path_base='original repository root',
        uploaded_artifact_contents=False, files=records), indent=2) + '\n', encoding='utf-8')
    (out / 'artifact_validation.json').write_text(json.dumps(summary, indent=2) + '\n', encoding='utf-8')
    print(json.dumps(dict(files_hashed=len(records), bytes_hashed=sum(r['bytes'] for r in records), **summary)))


if __name__ == '__main__':
    main()
