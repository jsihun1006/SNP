"""Independent B.2 extraction and frozen-support local diagnostics (no fitting)."""
import argparse
import gzip
import hashlib
import json
from pathlib import Path
import time
import numpy as np
import pandas as pd
from .local_temporal_source import load_local_source
from .local_temporal_features import local_features, OUTCOMES, SEQUENCES
from .local_temporal_frames import LocalTemporalFrames
from .pattern_profiles import profiles

KEY = ['frame', 'radius_m', 'cluster_id']


def equal_legacy(fresh, frozen, stats):
    """Recompute every B.1 profile column, including all seven values and outcomes."""
    for name in fresh:
        a, b = fresh[name], frozen[name]
        if pd.api.types.is_numeric_dtype(a):
            x, y = a.to_numpy(float), b.to_numpy(float)
            if not np.allclose(x, y, rtol=0, atol=1e-12, equal_nan=True): raise ValueError('Changed B.1: ' + name)
            common = np.isfinite(x) & np.isfinite(y)
            delta = float(np.max(np.abs(x[common] - y[common]))) if common.any() else 0.
            stats[name] = max(stats.get(name, 0.), delta)
        else:
            if not np.array_equal(a.fillna('<NA>').to_numpy(), b.fillna('<NA>').to_numpy()): raise ValueError('Changed B.1: ' + name)


def dictionary(config, columns, frame_columns):
    return dict(version=config['version'], key=KEY, local_outcomes=list(OUTCOMES),
        scopes=config['scopes'], direction=config['direction'], proximity=config['proximity'], motion=config['motion'],
        formulas={'S': 'max exp(-d_HA/3m), frozen fixed spatial anchor', 'd': '-3 log S (m)',
          'delta': 'last scope-contained median5 minus first median5; no segment search',
          'lag5_mean': 'mean of ALL median[t+5]-median[t] inside the fixed scope',
          'rise': 'complete reliable and local-composition-constant; delta S >= .02, delta d <= -.1m, mean lag5 agrees',
          'fall': 'complete reliable and local-composition-constant; delta S <= -.02, delta d >= .1m, mean lag5 agrees',
          'proximity': 'complete fixed scope; every radius-specific E>0, no shape stability requirement',
          'duration_lower_s': 'n/25 frame-bin lower bound from current-connected reliable close prefix; unknown center -> NaN, known not-close center -> 0',
          'raw_geometry_delta': 'last-minus-first raw K/E/contact/R/D in scope, finite complete support only; local composition changes separately flagged',
          'motion': 'maximal center-connected available frozen reciprocal-unique path inside seed anchor; min2 samples; nonzero length is measured motion only',
          'motion_span_s': '(last-first)/25 sample span, not frame-bin duration'},
        scope_metadata={'reliable_n': 'count with all frozen reliability gates plus finite S/d',
          'contiguous_support': 'all existing valid samples form one run; does NOT imply complete scope',
          'first_valid_offset,last_valid_offset': 'extrema of valid observations, NaN if none',
          'max_missing_run': 'max consecutive unsupported positions inside fixed scope',
          'global_changed_n': 'known global team counts differing from current; unknown counts not zero-filled',
          'local_changed_n': 'observed local H/A counts differing from current, not identities',
          'direction_reason': 'priority support reason or LOCAL_COMPOSITION_AMBIGUOUS; no change is distinct from unknown',
          'run_n,run_first_offset,run_last_offset': 'actual center-connected close prefix only',
          'run_censored': 'ends at unsupported sample or fixed scope edge; known inactive stop is not censored',
          'soft_range,distance_m_range,K_range,E_range': 'range within actually retained close prefix; no implication same pair/structure',
          'delta_radius_m,delta_diameter_m': 'max of components with centroid in seed anchor, not necessarily same component'},
        motion_metadata={'reliable_n': 'available nodes anywhere in fixed21 path; run_n retains ONLY center-connected nodes',
          'first_offset,last_offset': 'retained run endpoints', 'past_censored,future_censored': 'run terminated before fixed21 extent; window-limited even if false',
          'radius_range,diameter_range': 'metres, matched component range in retained run', 'density_range': 'unitless',
          'net_m': 'endpoint centroid distance', 'path_length_m': 'sum of adjacent centroid distances',
          'local_matching_ambiguous': 'current adjacent ambiguity flag; subsequent path truncation also reflected in motion support'},
        support_code={name: 1 << i for i, name in enumerate(config['scopes'])} | {'motion_min2_available': 16},
        sequence_codes={str(i): value for i, value in enumerate(SEQUENCES)},
        reasons=['OK', 'CURRENT_RECORD', 'CURRENT_UNOBSERVED_AMBIGUOUS', 'CURRENT_QC', 'RECORD_HALF_SPLIT_BOUNDARY',
                 'LOCAL_ABSENCE_AMBIGUOUS', 'OFFSET_QC', 'GLOBAL_COUNT_CHANGE', 'SUPPORT_GAP', 'LOCAL_COMPOSITION_AMBIGUOUS',
                 'CENTER_CONNECTED_GEOMETRY_ONLY', 'INSUFFICIENT_CONTIGUOUS_MATCHING'],
        joins='Frozen B.1 by key for current QC, masks, spatial seed and original P1..N3/A..D. No arrays or old evidence duplicated.',
        frame_semantics='existence per signal with source key; RF comes from ONE cluster sequence; sequence_view_mask is OR of same-row sequence codes; radius counts are views, not events; no anchor -> UNKNOWN',
        columns=columns, frame_columns=frame_columns,
        warning='Geometric local observations only, not true contest/identity/possession, no whole-pattern upgrade.')


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--output', required=True)
    p.add_argument('--config', default='competition_judgement/configs/local_temporal_b2.json')
    a = p.parse_args(); root = Path.cwd(); out = Path(a.output).resolve()
    if not out.is_relative_to(root / 'data/derived/competition_judgement'): raise ValueError('Local output only')
    if out.exists(): raise FileExistsError('Refusing to overwrite any run directory')
    config_path = Path(a.config); config = json.loads(config_path.read_text())
    if config['aggregate_evidence'] or config['candidate_decision']: raise ValueError('Local observations only')
    legacy = json.loads((root / config['legacy_config']).read_text())
    b1 = json.loads((root / config['pattern_config']).read_text())
    out.mkdir(parents=True)
    (out / 'preregistered_config.json').write_text(json.dumps(config, indent=2))
    (out / 'LOCAL_ONLY.txt').write_text('Licensed match derivatives: local only. No server, upload, commit or push. No football truth labels.\n')
    start = time.perf_counter(); frames, source = load_local_source(root, legacy)
    clusters = source.clusters; reducer = LocalTemporalFrames(frames)
    batch = config['batch_anchors']; val = []; stats = {}; columns = None
    masks = {k + '_mask': str for k in ('record', 'observed', 'reliable', 'strict', 'global_changed', 'local_changed', 'path')}
    previous = out.parent / 'evidence_pattern_b1_final_20261110/cluster_pattern_profiles.csv.gz'
    frozen_iter = pd.read_csv(previous, chunksize=batch, dtype=masks, low_memory=False)
    with gzip.open(out / 'cluster_local_temporal.csv.gz', 'wt', compresslevel=3) as dest:
        for begin in range(0, len(clusters), batch):
            selected = np.arange(begin, min(begin + batch, len(clusters)))
            ch = clusters.iloc[selected].reset_index(drop=True)
            old = next(frozen_iter).reset_index(drop=True)
            if not np.array_equal(ch[KEY].to_numpy(), old[KEY].to_numpy()): raise ValueError('Changed frozen keys')
            s = source.query(selected)
            result = local_features(s, ch.radius_m.to_numpy(), config)
            equal_legacy(profiles(s, ch.radius_m.to_numpy(), b1, legacy), old, stats)
            combined = pd.concat([ch[KEY], result], axis=1)
            columns = list(combined)
            combined.to_csv(dest, index=False, header=begin == 0, na_rep='NaN')
            reducer.add(ch, result)
            isval = source.roles[source.cluster_frame[selected]] == 'validation'
            if isval.any():
                diagnostics = combined.loc[isval].reset_index(drop=True)
                diagnostics['split'] = 'validation'
                diagnostics['current_strict'] = old.loc[isval, 'current_strict'].to_numpy()
                for pattern in 'ABCD': diagnostics[pattern + '_outcome'] = old.loc[isval, pattern + '_outcome'].to_numpy()
                val.append(diagnostics)
            if begin % (batch * 400) == 0: print(f'B.2 anchors {selected[-1]+1:,}/{len(clusters):,}; {time.perf_counter()-start:.1f}s', flush=True)
    try: next(frozen_iter); raise ValueError('Extra old rows')
    except StopIteration: pass
    frame = reducer.table()
    frame.to_csv(out / 'frame_local_temporal.csv.gz', index=False, compression=dict(method='gzip', compresslevel=3))
    preservation = dict(cluster_rows=len(clusters), exact_cluster_keys=True, recomputed_all_legacy_profile_columns=True,
                        numeric_max_abs_difference=stats, exact_A_B_C_D_outcomes=True, no_legacy_writes=True)
    (out / 'legacy_pattern_preservation.json').write_text(json.dumps(preservation, indent=2))
    from .local_temporal_diagnostics import validation_analysis, masking_analysis, noise_analysis
    validation = pd.concat(val, ignore_index=True); validation_analysis(validation, out)
    del val, validation
    masking = masking_analysis(source, config, out)
    noise = noise_analysis(root, config, legacy, out)
    (out / 'data_dictionary.json').write_text(json.dumps(dictionary(config, columns, list(frame)), indent=2))
    summary = dict(stage='B.2', frame_rows=len(frame), cluster_rows=len(clusters), validation_rows=int((source.roles[source.cluster_frame] == 'validation').sum()),
                   official_cluster_rows=int((clusters.period > 0).sum()), cluster_columns=len(columns), frame_columns=len(frame.columns),
                   masking=masking, noise=noise, runtime_s=time.perf_counter()-start,
                   config_sha256=hashlib.sha256(config_path.read_bytes()).hexdigest(),
                   no_fit=True, no_labels=True, no_test_selection=True, no_pattern_promotion=True, no_composites=True)
    (out / 'summary.json').write_text(json.dumps(summary, indent=2))
    print(json.dumps(summary), flush=True)


if __name__ == '__main__': main()
