"""Frozen Validation supports; changes describe geometry, not classification accuracy."""
from collections import Counter
import numpy as np
import pandas as pd
from .local_temporal_features import SIGNALS, local_features
from .local_temporal_source import LocalTemporalSource, mask_local_series
from .controlled_temporal_masking import position_mask, continuity_mask
from .missingness_experiments import draw_donors


def validation_analysis(val, out):
    if not (val.split == 'validation').all(): raise ValueError('Validation only')
    groups = {'all': np.ones(len(val), bool), 'strict': val.current_strict.to_numpy(bool),
              'low_quality': ~val.current_strict.to_numpy(bool),
              'radius=3': val.radius_m.to_numpy() == 3, 'radius=5': val.radius_m.to_numpy() == 5}
    coverage = []; recovery = []
    for group, sel in groups.items():
        for key in SIGNALS:
            for state in ('OBSERVED', 'NOT_OBSERVED', 'UNKNOWN'):
                count = int((sel & (val[key].to_numpy() == state)).sum())
                coverage.append(dict(group=group, signal=key, outcome=state, n=count, total=int(sel.sum()), fraction=count / int(sel.sum()) if sel.any() else None))
        for pattern in 'ABCD':
            held = sel & (val[pattern + '_outcome'].to_numpy() == 'UNKNOWN')
            for key in SIGNALS:
                recovery.append(dict(group=group, pattern=pattern, signal=key, unknown_pattern_n=int(held.sum()),
                                     observed_local_n=int((held & (val[key].to_numpy() == 'OBSERVED')).sum()),
                                     local_calculable_n=int((held & (val[key].to_numpy() != 'UNKNOWN')).sum())))
            for scale in ('short', 'near'):
                recovery.append(dict(group=group, pattern=pattern, signal='local_sequence_' + scale,
                                     unknown_pattern_n=int(held.sum()), observed_local_n=int((held & (val['local_sequence_' + scale].to_numpy() == 'LOCAL_RISE_THEN_FALL')).sum()),
                                     local_calculable_n=int((held & (val['local_rise_' + scale].to_numpy() != 'UNKNOWN') & (val['local_fall_' + scale].to_numpy() != 'UNKNOWN')).sum())))
    pd.DataFrame(coverage).to_csv(out / 'local_coverage_summary.csv', index=False)
    pd.DataFrame(recovery).to_csv(out / 'pattern_unknown_recovery.csv', index=False)


def compare(clean, changed, label, family, metadata=None):
    rows = []; n = len(clean); meta = metadata or {}
    for key in SIGNALS:
        a, b = clean[key].to_numpy(), changed[key].to_numpy()
        rows.append(dict(condition=label, family=family, kind='outcome', signal=key, n=n,
                         observed_n=int((b == 'OBSERVED').sum()), not_observed_n=int((b == 'NOT_OBSERVED').sum()),
                         unknown_n=int((b == 'UNKNOWN').sum()), changed_n=int((a != b).sum()),
                         lost_observed_n=int(((a == 'OBSERVED') & (b != 'OBSERVED')).sum()),
                         new_observed_n=int(((a != 'OBSERVED') & (b == 'OBSERVED')).sum()), **meta))
    metrics = [name for name in clean if any(term in name for term in
               ('delta_soft', 'delta_distance_m', 'lag5_mean', '_duration_lower_s', 'motion_net_m', 'motion_path_length_m', 'motion_diameter_range', 'motion_radius_range'))]
    for key in metrics:
        a, b = clean[key].to_numpy(float), changed[key].to_numpy(float)
        common = np.isfinite(a) & np.isfinite(b)
        diff = b[common] - a[common]
        # Comparable numeric values can still refer to different duration/path scopes.
        same_scope = common.copy()
        if key.startswith('motion_'):
            for suffix in ('first_offset', 'last_offset'): same_scope &= clean['motion_' + suffix].to_numpy() == changed['motion_' + suffix].to_numpy()
        elif '_duration_lower_s' in key:
            prefix = key.removesuffix('_duration_lower_s')
            for suffix in ('_run_first_offset', '_run_last_offset'): same_scope &= clean[prefix + suffix].to_numpy() == changed[prefix + suffix].to_numpy()
        rows.append(dict(condition=label, family=family, kind='raw_geometry', signal=key, n=n, common_n=int(common.sum()),
                         unknown_n=int((~np.isfinite(b)).sum()), mae=float(np.abs(diff).mean()) if len(diff) else None,
                         p95_abs=float(np.quantile(np.abs(diff), .95)) if len(diff) else None,
                         signed_bias=float(diff.mean()) if len(diff) else None,
                         same_scope_n=int(same_scope.sum()), same_scope_mae=float(np.abs(a[same_scope]-b[same_scope]).mean()) if same_scope.any() else None, **meta))
    for scale in ('short', 'near'):
        key = 'local_sequence_' + scale; a = clean[key].to_numpy(); b = changed[key].to_numpy()
        rows.append(dict(condition=label, family=family, kind='sequence', signal=key, n=n,
                         observed_n=int((b == 'LOCAL_RISE_THEN_FALL').sum()), changed_n=int((a != b).sum()),
                         lost_observed_n=int(((a == 'LOCAL_RISE_THEN_FALL') & (b != 'LOCAL_RISE_THEN_FALL')).sum()),
                         new_observed_n=int(((a != 'LOCAL_RISE_THEN_FALL') & (b == 'LOCAL_RISE_THEN_FALL')).sum()), **meta))
    return rows


def masking_analysis(source, config, out):
    stage_a = out.parent / 'missingness_stage_a_final_20261109'
    keys = pd.read_csv(stage_a / 'reference_keys.csv')
    key = ['frame', 'radius_m', 'cluster_id']
    index = source.clusters.reset_index().set_index(key)['index']
    selected = index.loc[pd.MultiIndex.from_frame(keys[key])].to_numpy(int)
    if not (source.roles[source.cluster_frame[selected]] == 'validation').all(): raise ValueError('Validation cohort required')
    pieces = [source.query(selected[i:i+64]) for i in range(0, len(selected), 64)]
    s = {k: np.concatenate([p[k] for p in pieces]) if k != 'offsets' else pieces[0][k] for k in pieces[0]}
    with np.load(stage_a / 'reference_temporal_arrays.npz') as archive:
        for k in archive.files: np.testing.assert_array_equal(s[k], archive[k])
    radii = keys.radius_m.to_numpy(int); clean = local_features(s, radii, config)
    rows = []; conditions = []; donors = {}; n = len(keys)
    with np.load(stage_a / 'train_pattern_library.npz') as archive:
        for radius in (3, 5):
            for basis in ('record', 'observed', 'reliable'):
                packed = np.packbits(archive[f'{radius}_{basis}_masks'], axis=1, bitorder='little')
                donors[(radius, basis)] = Counter({x.tobytes(): int(w) for x, w in zip(packed, archive[f'{radius}_{basis}_weights'])})
    def run(label, mask, family, **meta):
        if mask is None:
            conditions.append(dict(condition=label, status='INCOMPARABLE', **meta)); return
        corrupted = mask_local_series(s, mask)
        changed = local_features(corrupted, radii, config)
        rows.extend(compare(clean, changed, label, family, meta))
        conditions.append(dict(condition=label, status='COMPUTED', center_lost_n=int(mask[:, 50].sum()), **meta))
    locations = ('far_past', 'near_past', 'center_band', 'near_future', 'far_future', 'uniform', 'random', 'empirical', 'center_past_one_run', 'center_future_one_run')
    for count in config['robustness']['mask_counts']:
        for center in (False, True):
            for j, loc in enumerate(locations):
                empirical = None
                if loc == 'empirical':
                    empirical, available = draw_donors(donors, radii, 'reliable', np.random.default_rng(config['seed'] + count + int(center)), count - int(center), True)
                    if not available.all(): empirical = None
                mask = position_mask(n, count, loc, center, config['seed'] + 100 * count + j, empirical)
                run(f'm{count}_{loc}_center_{"lost" if center else "kept"}', mask, 'mask', missing_count=count, location=loc, center_deleted=center)
    for side in ('past', 'future'):
        for runs in ([10], [5, 5], [2] * 5, [1] * 10):
            run(f'continuity_{side}_{len(runs)}', continuity_mask(n, runs, side), 'continuity', missing_count=10, location=side, center_deleted=False)
    for basis in ('record', 'observed', 'reliable'):
        for rep in range(3):
            mask, available = draw_donors(donors, radii, basis, np.random.default_rng(config['seed'] + 1000 + rep))
            assert available.all()
            run(f'empirical_{basis}_{rep}', mask, 'empirical', center_deleted=None)
    pd.DataFrame(rows).to_csv(out / 'masking_robustness.csv', index=False)
    pd.DataFrame(conditions).to_csv(out / 'masking_conditions.csv', index=False)
    return dict(reference_windows=n, blocks=int(keys.support_block.nunique()), stage_a_arrays_bitwise_equal=True,
                computed_conditions=sum(x['status'] == 'COMPUTED' for x in conditions), incomparable_conditions=sum(x['status'] == 'INCOMPARABLE' for x in conditions))


def noise_analysis(root, config, legacy, out):
    import json
    from mtah_relational.frame_store import open_frame_cache
    from mtah_relational.splits import SplitManifest
    from .temporal_robustness import condition_source
    from .temporal_matching import reciprocal_unique
    temporal = json.loads((root / 'competition_judgement/configs/temporal_relations_v2_2.json').read_text())
    local = json.loads((root / 'competition_judgement/configs/local_clusters_v2_1.json').read_text())
    store = open_frame_cache(root / 'data/derived/relational_temporal_v1_20261109/raw_cache')
    manifest = SplitManifest.load(root / 'data/derived/relational_temporal_v1_20261109/split_manifest.json')
    base, c, offsets = condition_source(store, manifest, temporal, local, 'clean', 0.)
    source = LocalTemporalSource.from_memory(base, c, legacy)
    frames = base.ids[(offsets >= 50) & (offsets <= 100)]
    selected = np.flatnonzero(c.frame.isin(frames).to_numpy()); probes = c.iloc[selected].reset_index(drop=True)
    if not (source.roles[source.cluster_frame[selected]] == 'validation').all(): raise ValueError('Validation only')
    centers = probes[['cluster_centroid_x', 'cluster_centroid_y']].to_numpy()
    radii = np.maximum(6, probes.cluster_max_pairwise_distance_m.to_numpy() + 3); rc = (probes.radius_m.to_numpy() == 5).astype(int)
    clean = local_features(source.query(selected), probes.radius_m.to_numpy(), config)
    groups = list(probes.groupby(['frame', 'radius_m']).indices.items()); rows = []; bridges = []
    for kind, strength in config['robustness']['conditions']:
        pert, c2, _ = condition_source(store, manifest, temporal, local, kind, strength)
        changed_source = LocalTemporalSource.from_memory(pert, c2, legacy); mapped = np.full(len(probes), -1, int)
        c2groups = c2.groupby(['frame', 'radius_m']).indices
        for key, left in groups:
            right = c2groups.get(key, np.array([], int)); a = probes.iloc[left]; b = c2.iloc[right]
            _, pairs = reciprocal_unique(a[['cluster_centroid_x', 'cluster_centroid_y']].to_numpy(), b[['cluster_centroid_x', 'cluster_centroid_y']].to_numpy(),
                a.n_players.to_numpy(), b.n_players.to_numpy(), a.cluster_max_pairwise_distance_m.to_numpy(), b.cluster_max_pairwise_distance_m.to_numpy(), temporal)
            for i, j, _ in pairs: mapped[left[i]] = right[j]
        series = changed_source.query_fixed(probes.frame.to_numpy(), centers, radii, rc, mapped)
        changed = local_features(series, probes.radius_m.to_numpy(), config)
        label = f'{kind}_{strength:g}'
        rows.extend(compare(clean, changed, label, 'tracking_noise'))
        bridges.append(dict(condition=label, total=len(probes), unique_current_bridge_n=int((mapped >= 0).sum())))
        if kind in ('permutation', 'team_swap'):
            for name in clean:
                if pd.api.types.is_numeric_dtype(clean[name]): np.testing.assert_allclose(clean[name], changed[name], equal_nan=True, rtol=0, atol=1e-10)
                else: np.testing.assert_array_equal(clean[name], changed[name])
        print('B.2 fixed-anchor tracking robustness:', label, flush=True)
    pd.DataFrame(rows).to_csv(out / 'tracking_robustness.csv', index=False)
    pd.DataFrame(bridges).to_csv(out / 'tracking_probe_correspondence.csv', index=False)
    return dict(validation_support_frames=len(base.ids), center_frames=len(frames), clean_anchor_probes=len(probes), conditions=len(bridges),
                fixed_spatial_anchors=True, no_forced_cluster_identity=True, permutation_and_team_swap_invariant=True)
