"""Fixed-scope observations, not contest probabilities or whole-pattern decisions."""
import numpy as np
import pandas as pd
from .pattern_profiles import trailing

OUTCOMES = ('OBSERVED', 'NOT_OBSERVED', 'UNKNOWN')
SIGNALS = ('local_rise_short', 'local_rise_near', 'local_fall_short', 'local_fall_near',
           'local_proximity_past_short', 'local_proximity_past_near',
           'local_proximity_future_short', 'local_proximity_future_near', 'local_motion_status')
SEQUENCES = ('LOCAL_RISE_THEN_FALL', 'RISE_OBSERVED_FALL_NOT_OBSERVED', 'RISE_OBSERVED_FALL_UNKNOWN',
             'RISE_NOT_OBSERVED_FALL_OBSERVED', 'NEITHER_OBSERVED', 'RISE_NOT_OBSERVED_FALL_UNKNOWN',
             'RISE_UNKNOWN_FALL_OBSERVED', 'RISE_UNKNOWN_FALL_NOT_OBSERVED', 'LOCAL_SEQUENCE_UNKNOWN')


def max_gap(valid):
    run = np.zeros(len(valid), int)
    maximum = run.copy()
    for col in valid.T:
        run = np.where(col, 0, run + 1)
        maximum = np.maximum(maximum, run)
    return maximum


def support_summary(valid, offsets):
    offsets = np.asarray(offsets)
    has = valid.any(1)
    first = np.min(np.where(valid, offsets, 999), axis=1)
    last = np.max(np.where(valid, offsets, -999), axis=1)
    count = valid.sum(1)
    return dict(reliable_n=count, first_valid_offset=np.where(has, first, np.nan),
                last_valid_offset=np.where(has, last, np.nan), max_missing_run=max_gap(valid),
                contiguous_support=has & (count == last - first + 1))


def scope_reason(s, indices, valid):
    """Priority code; independent flags/counts preserve simultaneous causes."""
    n = len(valid)
    reason = np.full(n, 'OK', object)
    checks = [((~s['record_support'][:, indices]).any(1), 'RECORD_HALF_SPLIT_BOUNDARY'),
              ((s['record_support'][:, indices] & ~s['observed'][:, indices]).any(1), 'LOCAL_ABSENCE_AMBIGUOUS'),
              ((s['record_support'][:, indices] & ~s['strict'][:, indices]).any(1), 'OFFSET_QC'),
              ((s.get('global_count_known', s['record_support'])[:, indices] & ~s['global_count_same'][:, indices]).any(1), 'GLOBAL_COUNT_CHANGE'),
              (~valid.all(1), 'SUPPORT_GAP')]
    for failed, code in checks:
        reason[(reason == 'OK') & failed] = code
    for failed, code in [(~s['strict'][:, 50], 'CURRENT_QC'),
                         (~s['observed'][:, 50], 'CURRENT_UNOBSERVED_AMBIGUOUS'),
                         (~s['record_support'][:, 50], 'CURRENT_RECORD')]:
        reason[failed] = code
    return reason


def outcome(ready, condition):
    return np.where(ready, np.where(condition, 'OBSERVED', 'NOT_OBSERVED'), 'UNKNOWN')


def _prefix(s, indices, valid, edges, out, prefix, fps):
    """Current-connected close run; absent center differs from known zero close."""
    n = len(valid)
    offset = indices - 50
    order = np.arange(len(indices)) if offset[0] == 0 else np.arange(len(indices) - 1, -1, -1)
    close = valid & (edges > 0)
    alive = np.ones(n, bool)
    retained = np.zeros_like(close)
    for j in order:
        alive &= close[:, j]
        retained[:, j] = alive
    count = retained.sum(1)
    center_known = valid[:, order[0]]
    out[prefix + '_run_n'] = np.where(center_known, count, np.nan)
    out[prefix + '_duration_lower_s'] = np.where(center_known, count / fps, np.nan)
    for label, values in [('soft', s['soft'][:, indices]), ('distance_m', s['nearest_m'][:, indices]),
                          ('K', s['nearby_count'][:, indices]), ('E', edges)]:
        maximum = np.max(np.where(retained, values, -np.inf), axis=1)
        minimum = np.min(np.where(retained, values, np.inf), axis=1)
        out[prefix + '_' + label + '_range'] = np.where(count > 0, maximum - minimum, np.nan)
    first = np.min(np.where(retained, offset, 999), axis=1)
    last = np.max(np.where(retained, offset, -999), axis=1)
    out[prefix + '_run_first_offset'] = np.where(count > 0, first, np.nan)
    out[prefix + '_run_last_offset'] = np.where(count > 0, last, np.nan)
    # Invalid stop or scope edge means the true end is censored. Known non-close is an observed stop.
    next_index = np.clip(count, 0, len(indices) - 1)
    stop_known = valid[np.arange(n), order[next_index]]
    out[prefix + '_run_censored'] = np.where(center_known, (count == len(indices)) | ~stop_known, True)


def local_features(s, radii, config):
    n = len(radii)
    if s['soft'].shape != (n, 101):
        raise ValueError('Expected anchor series [N,101]')
    q = config['direction']; w = q['smoothing_frames']; lag = q['lag_frames']; tol = config['numeric_tolerance']
    edges = np.where(np.asarray(radii)[:, None] == 3, s['edges_3m'], s['edges_5m'])
    contact = np.where(np.asarray(radii)[:, None] == 3, s['contact_size_3m'], s['contact_size_5m'])
    out = {}; complete_bits = np.zeros(n, int)
    for bit, (name, (lo, hi)) in enumerate(config['scopes'].items()):
        indices = np.arange(lo + 50, hi + 51); offsets = indices - 50
        valid = s['reliable'][:, indices] & s['record_support'][:, indices] & s['observed'][:, indices]
        valid &= s['strict'][:, indices] & s['strict'][:, 50, None] & s['global_count_same'][:, indices]
        valid &= np.isfinite(s['soft'][:, indices]) & np.isfinite(s['nearest_m'][:, indices])
        ready = valid.all(1)
        for key, value in support_summary(valid, offsets).items(): out[name + '_' + key] = value
        reason = scope_reason(s, indices, valid)
        out[name + '_reason'] = reason
        known = s.get('global_count_known', s['record_support'])[:, indices]
        global_changed = known & ~s['global_count_same'][:, indices]
        local_changed = s['observed'][:, indices] & ((s['anchor_n_home'][:, indices] != s['anchor_n_home'][:, 50, None]) |
                                                   (s['anchor_n_away'][:, indices] != s['anchor_n_away'][:, 50, None]))
        out[name + '_global_changed_n'] = global_changed.sum(1)
        out[name + '_local_changed_n'] = local_changed.sum(1)
        complete_bits |= ready.astype(int) << bit
        # Unobserved numeric placeholders are masked before smoothing; no missing endpoint enters a value.
        a = trailing(np.where(valid, s['soft'][:, indices], np.nan), w)
        d = trailing(np.where(valid, s['nearest_m'][:, indices], np.nan), w)
        delta_s = a[:, -1] - a[:, 0]; delta_d = d[:, -1] - d[:, 0]
        lag_s = (a[:, lag:] - a[:, :-lag]).mean(1)
        lag_d = (d[:, lag:] - d[:, :-lag]).mean(1)
        direction_ready = ready & ~local_changed.any(1)
        out[name + '_direction_reason'] = np.where(ready & local_changed.any(1), 'LOCAL_COMPOSITION_AMBIGUOUS', reason)
        for label, values in [('delta_soft', delta_s), ('delta_distance_m', delta_d), ('lag5_mean_soft', lag_s), ('lag5_mean_distance_m', lag_d)]:
            out[name + '_' + label] = np.where(direction_ready, values, np.nan)
        # Independent raw geometry contrasts use only full local support and their own finite geometry support.
        for label, vals in [('K', s['nearby_count']), ('E', edges), ('contact', contact),
                            ('radius_m', s['component_radius']), ('diameter_m', s['component_diameter'])]:
            data = vals[:, indices]
            geometry_ready = ready & np.isfinite(data).all(1)
            out[name + '_delta_' + label] = np.where(geometry_ready, data[:, -1] - data[:, 0], np.nan)
        if name.startswith('past'):
            condition = (delta_s >= q['soft_deadband'] - tol) & (delta_d <= -q['distance_deadband_m'] + tol) & (lag_s > tol) & (lag_d < -tol)
            out['local_rise_' + name.split('_')[1]] = outcome(direction_ready, condition)
        else:
            condition = (delta_s <= -q['soft_deadband'] + tol) & (delta_d >= q['distance_deadband_m'] - tol) & (lag_s < -tol) & (lag_d > tol)
            out['local_fall_' + name.split('_')[1]] = outcome(direction_ready, condition)
        edge_valid = valid & np.isfinite(edges[:, indices])
        out['local_proximity_' + name] = outcome(edge_valid.all(1), (edges[:, indices] > 0).all(1))
        _prefix(s, indices, edge_valid, edges[:, indices], out, name, config['fps'])
    for scale in ('short', 'near'):
        rise = out['local_rise_' + scale]; fall = out['local_fall_' + scale]
        mapping = {v: i for i, v in enumerate(OUTCOMES)}
        codes = np.array([mapping[x] * 3 + mapping[y] for x, y in zip(rise, fall)])
        out['local_sequence_' + scale] = np.asarray(SEQUENCES, object)[codes]
    _motion(s, config, out)
    out['local_support_code'] = complete_bits | ((out['local_motion_status'] != 'UNKNOWN').astype(int) << 4)
    out['local_matching_ambiguous'] = (np.isin(s['prev_status'], ['geometric_ambiguous', 'observation_ambiguous', 'probe_mapping_unknown']) |
                                       np.isin(s['next_status'], ['geometric_ambiguous', 'observation_ambiguous', 'probe_mapping_unknown']))
    return pd.DataFrame(out)


def _motion(s, config, out):
    support = s['local_path_support'].copy()
    support &= np.isfinite(s['local_path_xy']).all(-1)
    for key in ('local_path_radius', 'local_path_diameter', 'local_path_density'): support &= np.isfinite(s[key])
    support &= s['local_path_inside']
    n = len(support)
    support &= (s['strict'][:, 50] & s['record_support'][:, 50] & s['observed'][:, 50])[:, None]
    for key, value in support_summary(support, np.arange(-10, 11)).items(): out['motion_' + key] = value
    retained = np.zeros_like(support); retained[:, 10] = support[:, 10]
    for side in (-1, 1):
        alive = support[:, 10].copy()
        for step in range(1, 11):
            j = 10 + side * step; alive &= support[:, j]; retained[:, j] = alive
    count = retained.sum(1); first = np.argmax(retained, 1); last = 20 - np.argmax(retained[:, ::-1], 1)
    eligible = count >= config['motion']['min_contiguous_samples']
    xy = s['local_path_xy']; row = np.arange(n)
    net = np.linalg.norm(xy[row, last] - xy[row, first], axis=1)
    segments = retained[:, 1:] & retained[:, :-1]
    length = np.sum(np.where(segments, np.linalg.norm(np.diff(xy, axis=1), axis=-1), 0), axis=1)
    out['local_motion_status'] = outcome(eligible, length > config['numeric_tolerance'])
    out['motion_reason'] = np.where(eligible, 'CENTER_CONNECTED_GEOMETRY_ONLY', 'INSUFFICIENT_CONTIGUOUS_MATCHING')
    out['motion_run_n'] = count
    out['motion_first_offset'] = np.where(count > 0, first - 10, np.nan)
    out['motion_last_offset'] = np.where(count > 0, last - 10, np.nan)
    out['motion_past_censored'] = (first > 0) | ~support[:, 10]
    out['motion_future_censored'] = (last < 20) | ~support[:, 10]
    out['motion_net_m'] = np.where(eligible, net, np.nan)
    out['motion_path_length_m'] = np.where(eligible, length, np.nan)
    out['motion_span_s'] = np.where(eligible, (last - first) / config['fps'], np.nan)
    for name in ('radius', 'diameter', 'density'):
        a = s['local_path_' + name]
        span = np.max(np.where(retained, a, -np.inf), axis=1) - np.min(np.where(retained, a, np.inf), axis=1)
        out['motion_' + name + '_range'] = np.where(eligible, span, np.nan)
