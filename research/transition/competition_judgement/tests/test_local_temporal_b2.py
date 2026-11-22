"""Predeclared geometric behavior, never labelled football scenes."""
import copy
import json
from pathlib import Path
import numpy as np
import pandas as pd
import pytest
from test_evidence_stage_b import scenario, physical_source, CONFIG as LEGACY
from competition_judgement.local_temporal_source import LocalTemporalSource, mask_local_series
from competition_judgement.local_temporal_features import local_features, SIGNALS
from competition_judgement.local_temporal_frames import LocalTemporalFrames
from competition_judgement.pattern_profiles import profiles

CONFIG = json.loads((Path(__file__).parents[1] / 'configs/local_temporal_b2.json').read_text())
B1 = json.loads((Path(__file__).parents[1] / 'configs/pattern_b1.json').read_text())


def data(kind='static'):
    s = scenario(kind)
    s.update(path_support=np.ones((1, 21), bool), path_inside_anchor=np.ones(1, bool),
             local_path_support=np.ones((1, 21), bool), local_path_inside=np.ones((1, 21), bool),
             local_path_xy=np.zeros((1, 21, 2)), local_path_radius=np.ones((1, 21)),
             local_path_diameter=np.ones((1, 21)) * 2, local_path_density=np.ones((1, 21)))
    return s


def calc(s, radius=3): return local_features(s, np.array([radius]), CONFIG)


def deleted(s, offsets):
    m = np.zeros((1, 101), bool); m[:, np.array(offsets) + 50] = True
    return mask_local_series(s, m)


def test_complete_rise():
    p = calc(data('approach_hold'))
    assert p.local_rise_short[0] == p.local_rise_near[0] == 'OBSERVED'
    assert p.past_short_delta_soft[0] > 0 and p.past_short_delta_distance_m[0] < 0


def test_complete_fall():
    p = calc(data('approach_separate'))
    assert p.local_fall_short[0] == p.local_fall_near[0] == 'OBSERVED'


def test_same_anchor_rise_then_fall():
    p = calc(data('approach_separate'))
    assert p.local_sequence_short[0] == p.local_sequence_near[0] == 'LOCAL_RISE_THEN_FALL'


def test_rise_then_hold_is_not_fall():
    p = calc(data('approach_hold'))
    assert p.local_rise_short[0] == 'OBSERVED' and p.local_fall_short[0] == 'NOT_OBSERVED'
    assert p.local_proximity_future_near[0] == 'OBSERVED'


def test_proximity_then_fall():
    s = data('approach_separate'); s['soft'][:, :51] = np.exp(-.4/3); s['nearest_m'][:, :51] = .4
    p = calc(s); assert p.local_rise_short[0] == 'NOT_OBSERVED' and p.local_fall_short[0] == 'OBSERVED'


def test_static_proximity_zero_directions_known():
    p = calc(data())
    assert p.local_rise_short[0] == p.local_fall_short[0] == 'NOT_OBSERVED'
    assert p.local_proximity_past_near[0] == 'OBSERVED'
    assert p.past_near_duration_lower_s[0] == pytest.approx(26/25)
    assert p.past_near_run_censored[0] and p.local_motion_status[0] == 'NOT_OBSERVED'


def test_far_gap_preserves_all_local_scopes():
    a = data('approach_separate'); b = deleted(a, [-50])
    pd.testing.assert_frame_equal(calc(a), calc(b))


def test_near_complete_full_pattern_unknown():
    s = deleted(data('approach_separate'), [-40, 40]); p = calc(s)
    assert p.local_rise_near[0] == p.local_fall_near[0] == 'OBSERVED'
    assert profiles(s, np.array([3]), B1, LEGACY).B_outcome[0] == 'UNKNOWN'


def test_short_gap_never_bridged_and_prefix_stops():
    p = calc(deleted(data('approach_separate'), [-4]))
    assert p.local_rise_short[0] == 'UNKNOWN' and p.past_short_max_missing_run[0] == 1
    assert np.isnan(p.past_short_delta_soft[0]) and p.past_short_run_n[0] == 4
    assert p.past_short_run_first_offset[0] == -3


def test_missing_current_all_unknown():
    p = calc(deleted(data(), [0]))
    assert all(p[k][0] == 'UNKNOWN' for k in SIGNALS)
    assert np.isnan(p.past_short_duration_lower_s[0]) and np.isnan(p.motion_net_m[0])


def test_current_qc_not_silently_relaxed():
    s = data(); s['strict'][:, 50] = False
    p = calc(s); assert all(p[k][0] == 'UNKNOWN' for k in SIGNALS)
    assert p.past_short_reason[0] == 'CURRENT_QC'


def test_global_composition_gates_even_stale_reliable():
    s = data('approach_hold'); s['global_count_same'][:, 45] = False
    p = calc(s); assert p.local_rise_short[0] == 'UNKNOWN'
    assert p.past_short_global_changed_n[0] == 1 and p.past_short_reason[0] == 'GLOBAL_COUNT_CHANGE'


def test_local_composition_ambiguous_but_region_contact_preserved():
    s = data('approach_hold'); s['anchor_n_home'][:, 45] = 2
    p = calc(s); assert p.local_rise_short[0] == 'UNKNOWN'
    assert p.past_short_direction_reason[0] == 'LOCAL_COMPOSITION_AMBIGUOUS'
    assert p.local_proximity_past_short[0] == 'OBSERVED'


def test_anchor_exit_not_fall():
    s = data('approach_separate'); s['observed'][:, 55:] = False
    s['anchor_n_away'][:, 55:] = 0
    p = calc(s); assert p.local_fall_short[0] == 'UNKNOWN'
    assert p.future_short_reason[0] == 'LOCAL_ABSENCE_AMBIGUOUS'


def test_geometric_ambiguity_withholds_motion_but_not_fixed_region():
    s = data(); s['local_path_support'][:] = False; s['local_path_support'][:, 10] = True
    s['prev_status'][:] = 'geometric_ambiguous'; s['next_status'][:] = 'geometric_ambiguous'
    p = calc(s); assert p.local_motion_status[0] == 'UNKNOWN' and p.local_matching_ambiguous[0]
    assert p.local_proximity_past_short[0] == 'OBSERVED'


def test_partial_motion_no_1m_gate():
    s = data(); s['local_path_support'][:, :9] = False; s['local_path_support'][:, 13:] = False
    s['local_path_xy'][0, :, 0] = np.arange(21) * .01
    p = calc(s); assert p.local_motion_status[0] == 'OBSERVED'
    assert p.motion_run_n[0] == 4 and p.motion_net_m[0] == pytest.approx(.03)
    assert p.motion_span_s[0] == pytest.approx(3/25)


def test_motion_cannot_jump_gap_or_select_largest_segment():
    s = data(); s['local_path_support'][:, 12] = False; s['local_path_xy'][0, 13:, 0] = 99
    p = calc(s); assert p.motion_last_offset[0] == 1 and p.local_motion_status[0] == 'NOT_OBSERVED'


def test_different_anchors_never_form_rise_then_fall():
    a = calc(data('approach_hold')); b = calc(data('approach_separate'))
    b.loc[0, 'local_rise_short'] = 'NOT_OBSERVED'; b.loc[0, 'local_sequence_short'] = 'RISE_NOT_OBSERVED_FALL_OBSERVED'
    base = pd.DataFrame(dict(frame=[0, 0], radius_m=[3, 5], cluster_id=[0, 1]))
    agg = LocalTemporalFrames(pd.DataFrame(dict(frame=[0, 1])))
    agg.add(base, pd.concat([a, b], ignore_index=True)); p = agg.table()
    assert p.local_rise_short[0] == p.local_fall_short[0] == 'OBSERVED'
    assert p.local_rise_then_fall_short[0] == 'NOT_OBSERVED'
    assert p.local_rise_short_source_radius_m[0] == 3 and p.local_fall_short_source_radius_m[0] == 5
    assert p.local_rise_short[1] == 'UNKNOWN'


@pytest.mark.parametrize('mask_kind', ['record', 'half', 'split'])
def test_boundary_masks_block_only_crossing_scopes(mask_kind):
    s = data('approach_separate')
    s['record_support'][:, 51:] = False
    p = calc(s); assert p.local_rise_short[0] == 'OBSERVED' and p.local_fall_short[0] == 'UNKNOWN'
    assert p.future_short_reason[0] == 'RECORD_HALF_SPLIT_BOUNDARY'


@pytest.mark.parametrize('permute,swap', [(False, False), (True, False), (True, True)])
def test_physical_slot_permutation_team_exchange(monkeypatch, permute, swap):
    base = physical_source(monkeypatch, permute=permute, swap=swap)
    source = LocalTemporalSource.from_memory(base, base.clusters, LEGACY)
    selected = np.flatnonzero((source.clusters.frame == 50) & (source.clusters.radius_m == 3))
    p = local_features(source.query(selected), np.array([3]), CONFIG)
    assert p.local_motion_status[0] == 'OBSERVED' and p.local_proximity_past_short[0] == 'OBSERVED'


@pytest.mark.parametrize('split', [False, True])
def test_actual_cache_record_or_split_boundary(monkeypatch, split):
    base = physical_source(monkeypatch, kind='dropout' if not split else 'moving', split=split)
    source = LocalTemporalSource.from_memory(base, base.clusters, LEGACY)
    selected = np.flatnonzero((source.clusters.frame == 50) & (source.clusters.radius_m == 3))
    s = source.query(selected); p = local_features(s, np.array([3]), CONFIG)
    assert p.local_rise_short[0] == 'UNKNOWN' or p.local_fall_short[0] == 'UNKNOWN'
    assert p.motion_run_n[0] < 21


def test_hidden_observations_do_not_leak():
    a = deleted(data('approach_separate'), [-4, 3]); b = copy.deepcopy(a)
    for key in ('soft', 'nearest_m', 'component_radius', 'component_diameter', 'nearby_count', 'edges_3m'):
        b[key][:, [46, 53]] = 99999
    b['local_path_xy'][:, [6, 13]] = 99999
    pd.testing.assert_frame_equal(calc(a), calc(b))


def test_legacy_inputs_values_and_patterns_not_mutated():
    s = deleted(data('approach_separate'), [-45]); saved = copy.deepcopy(s)
    before = profiles(s, np.array([3]), B1, LEGACY)
    calc(s); after = profiles(s, np.array([3]), B1, LEGACY)
    pd.testing.assert_frame_equal(before, after)
    for key in s: np.testing.assert_array_equal(s[key], saved[key])


def test_short_near_are_not_best_of_selection():
    p = calc(deleted(data('approach_separate'), [-20]))
    assert p.local_rise_short[0] == 'OBSERVED' and p.local_rise_near[0] == 'UNKNOWN'


def test_persistence_not_structure_stability():
    s = data(); s['edges_3m'][:, 45:50] = 4; s['nearby_count'][:, 45:50] = 2
    p = calc(s); assert p.local_proximity_past_short[0] == 'OBSERVED'
    assert p.past_short_E_range[0] == 3 and p.past_short_K_range[0] == 1


def test_known_nonclose_stops_duration_without_unknown():
    s = data(); s['edges_3m'][:, 47] = 0
    p = calc(s); assert p.local_proximity_past_short[0] == 'NOT_OBSERVED'
    assert p.past_short_run_n[0] == 3 and not p.past_short_run_censored[0]


def test_no_infinities_or_aggregate_scores():
    p = calc(deleted(data(), [0]))
    assert not np.isinf(p.select_dtypes(include='number').to_numpy()).any()
    assert not any('evidence_score' in c or 'candidate' in c for c in p)


def test_interior_peak_is_not_selected_in_place_of_fixed_endpoints():
    s = data(); s['soft'][:, 31:39] = .99; s['nearest_m'][:, 31:39] = .01
    p = calc(s)
    assert p.local_rise_near[0] == 'NOT_OBSERVED' and p.past_near_delta_soft[0] == 0


def test_hidden_counts_and_structure_never_change_results():
    a = deleted(data(), [-4, 3]); b = copy.deepcopy(a)
    for key in ('anchor_n_home', 'anchor_n_away', 'global_n_home', 'global_n_away', 'contact_size_3m', 'contact_size_5m', 'edges_5m'):
        b[key][:, [46, 53]] = 99999
    for key in ('local_path_radius', 'local_path_diameter', 'local_path_density'):
        b[key][:, [6, 13]] = 99999
    pd.testing.assert_frame_equal(calc(a), calc(b))


def test_motion_scope_exit_censors_without_using_later_nodes():
    s = data(); s['local_path_xy'][0, :, 0] = np.arange(21) * .01
    s['local_path_inside'][:, 12] = False
    p = calc(s)
    assert p.motion_last_offset[0] == 1 and p.motion_future_censored[0]
    assert p.motion_path_length_m[0] == pytest.approx(.11)


def test_frame_source_tie_is_deterministic_across_chunks():
    a = calc(data('approach_separate'))
    agg = LocalTemporalFrames(pd.DataFrame(dict(frame=[12])))
    for radius, cid in [(5, 0), (3, 2), (3, 1)]:
        agg.add(pd.DataFrame(dict(frame=[12], radius_m=[radius], cluster_id=[cid])), a)
    p = agg.table()
    assert p.local_rise_then_fall_short[0] == 'OBSERVED'
    assert p.local_rise_then_fall_short_source_radius_m[0] == 3 and p.local_rise_then_fall_short_source_cluster_id[0] == 1
    assert p.local_rise_then_fall_short_observed_3m_views[0] == 2 and p.local_rise_then_fall_short_observed_5m_views[0] == 1


def test_actual_half_boundary_routes_future_to_center_sentinel(monkeypatch):
    base = physical_source(monkeypatch)
    base.periods = base.periods.copy()
    base.periods[51:] = 2
    base.first[:51] = 0; base.last[:51] = 50
    base.first[51:] = 51; base.last[51:] = 100
    clusters = base.clusters.copy(); clusters.loc[clusters.frame >= 51, 'period'] = 2
    source = LocalTemporalSource.from_memory(base, clusters, LEGACY)
    selected = np.flatnonzero((clusters.frame == 50) & (clusters.radius_m == 3))
    s = source.query(selected); p = local_features(s, np.array([3]), CONFIG)
    assert not s['record_support'][:, 51:].any() and (s['index'][:, 51:] == 50).all()
    assert p.local_rise_short[0] == 'NOT_OBSERVED' and p.local_fall_short[0] == 'UNKNOWN'
    assert p.motion_last_offset[0] == 0
