"""Traceable same-anchor routing states; no contest truth, score or probability."""
import numpy as np
import pandas as pd
from .local_temporal_features import SIGNALS

KEY = ['frame', 'radius_m', 'cluster_id']
STATES = ('PATTERN_CANDIDATE', 'LOCAL_CANDIDATE', 'REVIEW', 'NO_DETECTED_SIGNAL', 'INSUFFICIENT')
CANDIDATES = STATES[:2]
BASIS = {'A': 1, 'B': 2, 'short': 4, 'near': 8}
SPATIAL_FAILURE = {'OUTSIDE_RECORD': 1, 'CURRENT_QC': 2, 'CURRENT_CLUSTER_UNMAPPED': 4,
                   'NOT_MIXED': 8, 'NO_RADIUS_CROSS_EDGE': 16, 'INVALID_GEOMETRY': 32,
                   'CURRENT_ANCHOR_UNOBSERVED': 64}
REVIEW_REASON = {'RISE_OBSERVED': 1, 'FALL_OBSERVED': 2, 'REQUIRED_EVIDENCE_PARTIAL': 4,
                 'PATTERN_UNKNOWN': 8, 'LOCAL_UNKNOWN': 16, 'SPATIAL_ONLY': 32,
                 'OTHER_TEMPORAL_OBSERVATIONS': 64}


def check_alignment(base, other):
    if len(base) != len(other): raise ValueError('Same-anchor rows required')
    if all(k in other for k in KEY) and not np.array_equal(base[KEY].to_numpy(), other[KEY].to_numpy()):
        raise ValueError('Same-anchor key/order mismatch')


def local_complete(local, scope, size):
    return ((local[scope + '_reliable_n'].to_numpy() == size) &
            (local[scope + '_max_missing_run'].to_numpy() == 0) &
            (local[scope + '_local_changed_n'].to_numpy() == 0) &
            (local[scope + '_direction_reason'].to_numpy() == 'OK'))


def spatial_gate(base, profile, mapped=None):
    n = len(base); mapped = np.ones(n, bool) if mapped is None else np.asarray(mapped, bool)
    if mapped.shape != (n,): raise ValueError('Invalid probe map shape')
    nh, na = base.n_home.to_numpy(float), base.n_away.to_numpy(float)
    npart, edge = base.n_players.to_numpy(float), base.n_cross_edges.to_numpy(float)
    geometry = base[['cluster_centroid_x', 'cluster_centroid_y', 'cluster_radius_m',
                     'cluster_max_pairwise_distance_m', 'cross_pair_density', 'contact_strength']].to_numpy(float)
    radius, diameter, density, contact = geometry[:, 2], geometry[:, 3], geometry[:, 4], geometry[:, 5]
    tol = 1e-12
    counts = np.column_stack([nh, na, npart, edge])
    consistent = np.isfinite(counts).all(1) & (counts >= 0).all(1) & (counts == np.floor(counts)).all(1)
    consistent &= npart == nh + na
    consistent &= (edge <= nh * na) & np.isin(base.radius_m.to_numpy(), [3, 5])
    consistent &= np.isfinite(geometry).all(1) & (radius >= 0) & (diameter >= radius - tol)
    consistent &= (density >= 0) & (density <= 1 + tol) & (contact >= 0) & (contact <= 1 + tol)
    observed = ((profile.current_anchor_home.to_numpy(float) > 0) & (profile.current_anchor_away.to_numpy(float) > 0) &
                np.isfinite(profile.current_soft.to_numpy(float)) & (profile.current_soft.to_numpy(float) > 0) &
                np.isfinite(profile.current_edges.to_numpy(float)) & (profile.current_edges.to_numpy(float) > 0))
    flags = {
        'OUTSIDE_RECORD': ~base.period.isin([1, 2]).to_numpy(),
        'CURRENT_QC': ~(base.strict_quality_valid.to_numpy(bool) & profile.current_strict.to_numpy(bool)),
        'CURRENT_CLUSTER_UNMAPPED': ~mapped,
        'NOT_MIXED': ~((nh > 0) & (na > 0)),
        'NO_RADIUS_CROSS_EDGE': ~(edge > 0),
        'INVALID_GEOMETRY': ~consistent,
        'CURRENT_ANCHOR_UNOBSERVED': ~observed,
    }
    mask = sum(flags[k].astype(int) * value for k, value in SPATIAL_FAILURE.items())
    primary = np.full(n, 'OK', object)
    for key in SPATIAL_FAILURE: primary[(primary == 'OK') & flags[key]] = key
    return mask == 0, mask, primary


def decide_clusters(base, profile, local, config, mapped=None):
    """base[N,*] + frozen profile/local[N,*] -> compact decision-only table[N,*].

    This adapter consumes outcomes/support, never compares individual Evidence
    magnitudes. Upstream contradictory states are contract errors, not new rules.
    """
    check_alignment(base, profile); check_alignment(base, local)
    if base.duplicated(KEY).any(): raise ValueError('Duplicate current anchor keys')
    n = len(base); gate, failure, spatial_reason = spatial_gate(base, profile, mapped)
    supported = {}; full = {}
    for p in 'ABCD':
        if not profile[p + '_outcome'].isin(['SUPPORTED', 'NOT_OBSERVED', 'UNKNOWN']).all(): raise ValueError('Invalid frozen outcome')
        supported[p] = profile[p + '_outcome'].to_numpy() == 'SUPPORTED'
        if (supported[p] & ((profile[p + '_obs'].to_numpy() != 'COMPLETE') | (profile[p + '_seq'].to_numpy() != 'VERIFIED'))).any():
            raise ValueError('Contradictory frozen SUPPORTED pattern')
        full[p] = (profile[p + '_obs'].to_numpy() == 'COMPLETE') & (profile[p + '_outcome'].to_numpy() != 'UNKNOWN')
    local_known = {}; rf = {}
    for key in SIGNALS:
        if not local[key].isin(['OBSERVED', 'NOT_OBSERVED', 'UNKNOWN']).all(): raise ValueError('Invalid local outcome')
    for scale, size in (('short', 11), ('near', 26)):
        r, f = local['local_rise_' + scale].to_numpy(), local['local_fall_' + scale].to_numpy()
        for side, state in (('past', r), ('future', f)):
            complete = local_complete(local, side + '_' + scale, size)
            if ((state != 'UNKNOWN') & ~complete).any(): raise ValueError('Local outcome lacks frozen complete scope')
            local_known[('rise' if side == 'past' else 'fall') + '_' + scale] = (state != 'UNKNOWN') & complete
        rf[scale] = (r == 'OBSERVED') & (f == 'OBSERVED')
        if not np.array_equal(rf[scale], local['local_sequence_' + scale].to_numpy() == 'LOCAL_RISE_THEN_FALL'):
            raise ValueError('Local sequence not certified by same-anchor Rise/Fall')
    basis = (supported['A'].astype(int) + 2 * supported['B'] + 4 * rf['short'] + 8 * rf['near']).astype(int)
    pattern = supported['A'] | supported['B']; local_candidate = rf['short'] | rf['near']
    all_local_not = np.ones(n, bool)
    for key in ('rise_short', 'rise_near', 'fall_short', 'fall_near'):
        all_local_not &= local_known[key] & (local['local_' + key].to_numpy() == 'NOT_OBSERVED')
    sufficient_absence = full['A'] & full['B'] & (profile.A_outcome.to_numpy() == 'NOT_OBSERVED') & (profile.B_outcome.to_numpy() == 'NOT_OBSERVED') & all_local_not
    state = np.where(~gate, 'INSUFFICIENT', np.where(pattern, 'PATTERN_CANDIDATE', np.where(local_candidate, 'LOCAL_CANDIDATE',
                     np.where(sufficient_absence, 'NO_DETECTED_SIGNAL', 'REVIEW'))))
    rise = (local.local_rise_short.to_numpy() == 'OBSERVED') | (local.local_rise_near.to_numpy() == 'OBSERVED')
    fall = (local.local_fall_short.to_numpy() == 'OBSERVED') | (local.local_fall_near.to_numpy() == 'OBSERVED')
    partial = np.zeros(n, int); complete_evidence = partial.copy(); any_dynamic = np.zeros(n, bool)
    for j, key in enumerate(('P1', 'P2', 'P3', 'P4')):
        a = profile[key + '_state'].to_numpy()
        if not np.isin(a, ['COMPLETE', 'EXPERIMENTAL_PARTIAL', 'UNKNOWN']).all(): raise ValueError('Invalid evidence support')
        partial |= (a == 'EXPERIMENTAL_PARTIAL').astype(int) << j
        complete_evidence |= (a == 'COMPLETE').astype(int) << j
        any_dynamic |= a != 'UNKNOWN'
    pattern_unknown = (profile.A_outcome.to_numpy() == 'UNKNOWN') | (profile.B_outcome.to_numpy() == 'UNKNOWN')
    local_unknown = ~np.stack(list(local_known.values()), 1).all(1)
    spatial_only = np.ones(n, bool)
    for key in ('P1', 'P2', 'P3', 'P4', 'N1', 'N2', 'N3'): spatial_only &= profile[key + '_state'].to_numpy() == 'UNKNOWN'
    for key in SIGNALS: spatial_only &= local[key].to_numpy() == 'UNKNOWN'
    reason_mask = (rise.astype(int) | (fall.astype(int) << 1) | ((partial > 0).astype(int) << 2) |
                   (pattern_unknown.astype(int) << 3) | (local_unknown.astype(int) << 4) | (spatial_only.astype(int) << 5) |
                   ((~spatial_only & ~rise & ~fall).astype(int) << 6))
    reason = np.full(n, 'OTHER_TEMPORAL_OBSERVATIONS', object)
    reason[pattern_unknown] = 'WHOLE_PATTERN_UNCONFIRMED'
    reason[partial > 0] = 'PARTIAL_DYNAMIC_EVIDENCE'
    reason[fall & ~rise] = 'FALL_RISE_UNCONFIRMED'
    reason[rise & ~fall] = 'RISE_FALL_UNCONFIRMED'
    reason[rise & fall] = 'UNPAIRED_SCOPE_SIGNALS'
    reason[spatial_only] = 'SPATIAL_ONLY_TEMPORAL_UNKNOWN'
    reason[state == 'NO_DETECTED_SIGNAL'] = 'DYNAMIC_CONDITIONS_NOT_OBSERVED'
    for hit, code in [(rf['short'] & ~rf['near'], 'LOCAL_RF_SHORT'), (rf['near'] & ~rf['short'], 'LOCAL_RF_NEAR'),
                      (rf['short'] & rf['near'], 'LOCAL_RF_SHORT_NEAR'),
                      (supported['A'] & ~supported['B'], 'PATTERN_A'), (supported['B'] & ~supported['A'], 'PATTERN_B'),
                      (supported['A'] & supported['B'], 'PATTERN_A_B')]: reason[gate & hit] = code
    reason[~gate] = spatial_reason[~gate]
    dynamic_support = full['A'].astype(int) | (full['B'].astype(int) << 1)
    for j, key in enumerate(('rise_short', 'rise_near', 'fall_short', 'fall_near')): dynamic_support |= local_known[key].astype(int) << (j + 2)
    decision_basis = np.where(np.isin(state, CANDIDATES), basis, 0)
    future = np.zeros(n, int)
    for key, frames in config['future_frames_by_basis'].items(): future = np.maximum(future, np.where((decision_basis & BASIS[key]) > 0, frames, 0))
    future[np.isin(state, ['REVIEW', 'NO_DETECTED_SIGNAL'])] = 50
    # Keep upstream unknown explanations, not copied raw arrays/feature values.
    reasons = [[] for _ in range(n)]
    for key in ('A', 'B', 'P1', 'P2', 'P3', 'P4'):
        unknown = profile[key + ('_state' if key.startswith('P') else '_outcome')].to_numpy() == 'UNKNOWN'
        for i in np.flatnonzero(unknown): reasons[i].append(key + ':' + str(profile[key + '_reason'].iloc[i]))
    for key in ('past_short', 'past_near', 'future_short', 'future_near'):
        code = local[key + '_direction_reason'].to_numpy()
        for i in np.flatnonzero(code != 'OK'): reasons[i].append(key + ':' + str(code[i]))
    return pd.DataFrame(dict(decision_state=state, decision_reason=reason, spatial_gate_valid=gate,
        spatial_failure_mask=failure, observed_basis_mask=basis, decision_basis_mask=decision_basis,
        dynamic_support_mask=dynamic_support, required_complete_mask=complete_evidence, required_partial_mask=partial,
        review_reason_mask=np.where(state == 'REVIEW', reason_mask, 0),
        decision_future_frames=future, decision_uses_future=future > 0,
        upstream_unknown_reasons=['|'.join(x) if x else 'NONE' for x in reasons]))
