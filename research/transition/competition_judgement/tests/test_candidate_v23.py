"""Candidate routing semantics on synthetic observations, never football labels."""
import copy
import json
from pathlib import Path
import numpy as np
import pandas as pd
import pytest
from test_local_temporal_b2 import data, deleted, calc, B1, LEGACY
from test_evidence_stage_b import physical_source
from competition_judgement.pattern_profiles import profiles
from competition_judgement.candidate_decision_v23 import decide_clusters, STATES
from competition_judgement.candidate_frames_v23 import CandidateFrames
from competition_judgement.local_temporal_source import LocalTemporalSource
from competition_judgement.local_temporal_features import local_features

CONFIG = json.loads((Path(__file__).parents[1] / 'configs/candidate_v23.json').read_text())
LOCAL = json.loads((Path(__file__).parents[1] / 'configs/local_temporal_b2.json').read_text())


def inputs(s=None):
    s = data() if s is None else s
    p = profiles(s, np.array([3]), B1, LEGACY); l = calc(s)
    b = pd.DataFrame(dict(frame=[50], period=[1], radius_m=[3], cluster_id=[0], strict_quality_valid=[True],
         n_home=[1], n_away=[1], n_players=[2], n_cross_edges=[1], cluster_centroid_x=[50.], cluster_centroid_y=[34.],
         cluster_radius_m=[.5], cluster_max_pairwise_distance_m=[1.], cross_pair_density=[1.], contact_strength=[.7]))
    return b, p, l


def force(p, pattern, state):
    p[pattern + '_outcome'] = state
    p[pattern + '_obs'] = 'COMPLETE' if state != 'UNKNOWN' else 'UNKNOWN'
    p[pattern + '_seq'] = 'VERIFIED' if state == 'SUPPORTED' else 'UNVERIFIED'
    p[pattern + '_reason'] = 'OK' if state != 'UNKNOWN' else 'MISSING_EVIDENCE'


@pytest.mark.parametrize('patterns,bits', [('A', 1), ('B', 2), ('AB', 3)])
def test_supported_patterns_preserved_independently(patterns, bits):
    b, p, l = inputs()
    for name in patterns: force(p, name, 'SUPPORTED')
    d = decide_clusters(b, p, l, CONFIG)
    assert d.decision_state[0] == 'PATTERN_CANDIDATE' and d.decision_basis_mask[0] == bits


def test_unknown_full_pattern_short_local_candidate():
    b, p, l = inputs(deleted(data('approach_separate'), [-20, 40]))
    assert p.B_outcome[0] == 'UNKNOWN'
    d = decide_clusters(b, p, l, CONFIG)
    assert d.decision_state[0] == 'LOCAL_CANDIDATE' and d.decision_basis_mask[0] == 4
    assert d.decision_future_frames[0] == 10


def test_unknown_full_pattern_near_only_local_candidate():
    s = data(); t = np.arange(-50, 51); distance = .4 + np.maximum(np.abs(t) - 10, 0)*.1
    s['nearest_m'][:] = distance; s['soft'][:] = np.exp(-distance/3)
    b, p, l = inputs(deleted(s, [-40, 40]))
    assert l.local_sequence_near[0] == 'LOCAL_RISE_THEN_FALL' and l.local_rise_short[0] == 'NOT_OBSERVED'
    d = decide_clusters(b, p, l, CONFIG)
    assert d.decision_state[0] == 'LOCAL_CANDIDATE' and d.decision_basis_mask[0] == 8


def test_rise_without_known_fall_is_review():
    b, p, l = inputs(deleted(data('approach_hold'), [-40] + list(range(1, 51))))
    d = decide_clusters(b, p, l, CONFIG)
    assert d.decision_state[0] == 'REVIEW' and d.decision_reason[0] == 'RISE_FALL_UNCONFIRMED'


def test_fall_without_rise_is_review():
    b, p, l = inputs(deleted(data('approach_separate'), list(range(-50, 0))))
    d = decide_clusters(b, p, l, CONFIG)
    assert d.decision_state[0] == 'REVIEW' and d.decision_reason[0] == 'FALL_RISE_UNCONFIRMED'


@pytest.mark.parametrize('alternative', ['C', 'D', 'N1', 'N2', 'N3', 'motion'])
def test_alternative_evidence_never_vetoes_positive(alternative):
    b, p, l = inputs(); force(p, 'A', 'SUPPORTED')
    if alternative in 'CD': force(p, alternative, 'SUPPORTED')
    elif alternative.startswith('N'): p[alternative] = 1.; p[alternative + '_state'] = 'COMPLETE'
    else: l['local_motion_status'] = 'OBSERVED'
    assert decide_clusters(b, p, l, CONFIG).decision_state[0] == 'PATTERN_CANDIDATE'


@pytest.mark.parametrize('alternative', ['C', 'D', 'N2', 'motion'])
def test_alternative_alone_is_not_candidate(alternative):
    b, p, l = inputs()
    if alternative in 'CD': force(p, alternative, 'SUPPORTED')
    elif alternative == 'N2': p.N2 = 1.
    else: l.local_motion_status = 'OBSERVED'
    assert decide_clusters(b, p, l, CONFIG).decision_state[0] == 'NO_DETECTED_SIGNAL'


def test_valid_space_all_temporal_unknown_is_review():
    s = deleted(data(), [x for x in range(-50, 51) if x != 0]); s['path_valid'][:] = False; s['path_score'][:] = np.nan
    b, p, l = inputs(s); d = decide_clusters(b, p, l, CONFIG)
    assert d.decision_state[0] == 'REVIEW' and d.decision_reason[0] == 'SPATIAL_ONLY_TEMPORAL_UNKNOWN'


def test_fully_observed_nonmatching_conditions_are_known_absence():
    b, p, l = inputs(); d = decide_clusters(b, p, l, CONFIG)
    assert d.decision_state[0] == 'NO_DETECTED_SIGNAL' and d.dynamic_support_mask[0] == 63


@pytest.mark.parametrize('bad', ['qc', 'geometry', 'unmapped', 'unmixed', 'edge', 'current', 'outside'])
def test_spatial_gate_has_priority_over_high_temporal(bad):
    b, p, l = inputs(); force(p, 'A', 'SUPPORTED'); mapped = np.ones(1, bool)
    if bad == 'qc': b.strict_quality_valid = False
    if bad == 'geometry': b.cluster_radius_m = np.nan
    if bad == 'unmapped': mapped[:] = False
    if bad == 'unmixed': b.n_away = 0
    if bad == 'edge': b.n_cross_edges = 0
    if bad == 'current': p.current_soft = np.nan
    if bad == 'outside': b.period = 0
    d = decide_clusters(b, p, l, CONFIG, mapped)
    assert d.decision_state[0] == 'INSUFFICIENT' and d.decision_basis_mask[0] == 0 and d.observed_basis_mask[0] == 1


def frame_context(n=1, strict=True, clusters=0):
    return pd.DataFrame(dict(frame=np.arange(50, 50+n), period=np.ones(n, int), strict_quality_valid=np.full(n, strict),
                            n_home=np.full(n, 8), n_away=np.full(n, 8), n_local_clusters_3m=np.full(n, clusters), n_local_clusters_5m=np.full(n, clusters)))


@pytest.mark.parametrize('strict,expected', [(True, 'NO_DETECTED_SIGNAL'), (False, 'INSUFFICIENT')])
def test_empty_observed_graph_vs_bad_qc(strict, expected):
    f = CandidateFrames(frame_context(strict=strict), CONFIG).table()
    assert f.decision_state[0] == expected
    if strict: assert f.temporal_assessment[0] == 'NOT_APPLICABLE'


def test_radius_views_preserved_without_event_merge():
    b, p, l = inputs(); force(p, 'A', 'SUPPORTED'); d = decide_clusters(b, p, l, CONFIG)
    agg = CandidateFrames(frame_context(clusters=1), CONFIG)
    agg.add(b, d); b2 = b.copy(); b2.radius_m = 5; b2.cluster_id = 1; agg.add(b2, d)
    f = agg.table(); assert f.candidate_3m_exists[0] and f.candidate_5m_exists[0]
    assert f.candidate_3m_anchor_views[0] == f.candidate_5m_anchor_views[0] == 1
    assert f.representative_radius_m[0] == 3


def test_different_anchor_rise_fall_not_fused_at_frame():
    b1, p1, l1 = inputs(deleted(data('approach_hold'), [-40] + list(range(1, 51))))
    b2, p2, l2 = inputs(deleted(data('approach_separate'), list(range(-50, 0)))); b2.cluster_id = 1
    d1 = decide_clusters(b1, p1, l1, CONFIG); d2 = decide_clusters(b2, p2, l2, CONFIG)
    agg = CandidateFrames(frame_context(clusters=2), CONFIG); agg.add(b1, d1); agg.add(b2, d2)
    f = agg.table(); assert f.decision_state[0] == 'REVIEW' and not f.local_candidate_exists[0]


def test_pattern_and_local_bases_all_retained():
    b, p, l = inputs(data('approach_separate')); d = decide_clusters(b, p, l, CONFIG)
    assert d.decision_state[0] == 'PATTERN_CANDIDATE' and d.decision_basis_mask[0] == 15


def test_unknown_cannot_make_no_signal_at_frame():
    b, p, l = inputs(); d1 = decide_clusters(b, p, l, CONFIG)
    b2 = b.copy(); b2.cluster_id = 1; b2.cluster_radius_m = np.nan; d2 = decide_clusters(b2, p, l, CONFIG)
    agg = CandidateFrames(frame_context(clusters=2), CONFIG); agg.add(b, d1); agg.add(b2, d2)
    assert agg.table().decision_state[0] == 'INSUFFICIENT'


@pytest.mark.parametrize('split', [False, True])
def test_real_synthetic_record_split_boundary(monkeypatch, split):
    base = physical_source(monkeypatch, kind='dropout' if not split else 'moving', split=split)
    source = LocalTemporalSource.from_memory(base, base.clusters, LEGACY)
    selected = np.flatnonzero((source.clusters.frame == 50) & (source.clusters.radius_m == 3))
    b = source.clusters.iloc[selected].reset_index(drop=True); s = source.query(selected)
    d = decide_clusters(b, profiles(s, b.radius_m.to_numpy(), B1, LEGACY), local_features(s, b.radius_m.to_numpy(), LOCAL), CONFIG)
    assert d.decision_state[0] == 'REVIEW'


@pytest.mark.parametrize('permute,swap', [(False, False), (True, False), (True, True)])
def test_actual_geometric_source_permutation_team_swap(monkeypatch, permute, swap):
    base = physical_source(monkeypatch, permute=permute, swap=swap)
    source = LocalTemporalSource.from_memory(base, base.clusters, LEGACY)
    selected = np.flatnonzero((source.clusters.frame == 50) & (source.clusters.radius_m == 3))
    b = source.clusters.iloc[selected].reset_index(drop=True); s = source.query(selected)
    d = decide_clusters(b, profiles(s, b.radius_m.to_numpy(), B1, LEGACY), local_features(s, b.radius_m.to_numpy(), LOCAL), CONFIG)
    assert d.decision_state[0] == 'NO_DETECTED_SIGNAL'


def test_reject_cross_anchor_join_and_inconsistent_sequence():
    b, p, l = inputs(); l = pd.concat([b[['frame','radius_m','cluster_id']], l], axis=1)
    wrong = l.copy(); wrong.cluster_id = 9
    with pytest.raises(ValueError, match='key/order'): decide_clusters(b, p, wrong, CONFIG)
    l.local_sequence_short = 'LOCAL_RISE_THEN_FALL'
    with pytest.raises(ValueError, match='sequence'): decide_clusters(b, p, l, CONFIG)


def test_no_mutation_or_dependency_on_arbitrary_scores():
    b, p, l = inputs(); original = [x.copy(deep=True) for x in (b, p, l)]
    a = decide_clusters(b, p, l, CONFIG)
    for x, y in zip((b,p,l), original): pd.testing.assert_frame_equal(x, y)
    for key in ('P1','P2','P3','P4','N1','N2','N3'): p[key] = 9999.
    p['ball_x'] = 99; p['GT'] = 'D'; p['positive_evidence_score'] = 1.
    pd.testing.assert_frame_equal(a, decide_clusters(b, p, l, CONFIG))


def test_radius_family_join_has_one_unambiguous_key(tmp_path):
    from competition_judgement.candidate_diagnostics_v23 import radius_analysis
    folder=tmp_path/'data/derived/competition_judgement/local_v2_1_20261109';folder.mkdir(parents=True)
    clusters=pd.DataFrame(dict(frame=[50,50,50],radius_m=[3,3,5],cluster_id=[0,1,0],family_id_5m=[0,0,0],
                              same_node_set_across_radii=[False,False,False],cluster_centroid_x=[49.,51.,50.],cluster_centroid_y=[34.,34.,34.]))
    clusters.to_csv(folder/'local_clusters.csv',index=False)
    val=clusters[['frame','radius_m','cluster_id']].copy();val['decision_state']=['PATTERN_CANDIDATE','REVIEW','LOCAL_CANDIDATE']
    result=radius_analysis(val,tmp_path,tmp_path)
    assert result['validation_3m_views']==2 and result['candidate_both_radius_views']==1
    assert (pd.read_csv(tmp_path/'radius_overlap_diagnostic.csv').unique_5m_families==1).all()


@pytest.mark.parametrize('strength',[.1,.5,1.])
def test_actual_synthetic_coordinate_jitter_is_finite_not_accuracy(monkeypatch,strength):
    from mtah_relational.frame_store import FrameStore
    from competition_judgement.local_features import extract_local_frame
    from competition_judgement.temporal_source import TemporalSource
    base=physical_source(monkeypatch)
    xy=np.concatenate([base.h,base.a],axis=1).copy()
    xy+=np.random.default_rng(2027).normal(size=xy.shape)*strength
    teams=np.r_[np.zeros(base.h.shape[1],int),np.ones(base.a.shape[1],int)]
    store=FrameStore(base.ids.copy(),base.periods.copy(),xy,teams,(105.3,68.))
    cfg=json.loads((Path(__file__).parents[1]/'configs/local_clusters_v2_1.json').read_text())
    frames=[];clusters=[]
    for i in range(len(store)):
        frame,rows,_=extract_local_frame(store.frame(i),store.pitch,cfg);frames.append(frame);clusters.extend(rows)
    class Manifest:
        def split_of(self,frame):return 'validation'
    clusters=pd.DataFrame(clusters);temporal=TemporalSource(store,pd.DataFrame(frames),Manifest(),base.config,clusters=clusters)
    source=LocalTemporalSource.from_memory(temporal,clusters,LEGACY)
    # Strong jitter may remove every 3m component. Keep actual radius views;
    # empty graphs are handled by the Frame reducer, not the legacy batch API.
    selected=np.flatnonzero((clusters.frame==50))
    if not len(selected):
        current=pd.DataFrame(frames).loc[lambda x:x.frame==50]
        result=CandidateFrames(current,CONFIG).table()
        assert result.decision_state[0] in ('NO_DETECTED_SIGNAL','INSUFFICIENT')
        return
    b=clusters.iloc[selected].reset_index(drop=True);s=source.query(selected)
    result=decide_clusters(b,profiles(s,b.radius_m.to_numpy(),B1,LEGACY),local_features(s,b.radius_m.to_numpy(),LOCAL),CONFIG)
    assert result.decision_state.isin(STATES).all()
    assert not np.isinf(result.select_dtypes(include='number').to_numpy()).any()


def test_actual_half_boundary_preserves_temporal_unknown(monkeypatch):
    base=physical_source(monkeypatch);base.periods=base.periods.copy();base.periods[51:]=2
    base.first[:51]=0;base.last[:51]=50;base.first[51:]=51;base.last[51:]=100
    clusters=base.clusters.copy();clusters.loc[clusters.frame>=51,'period']=2
    source=LocalTemporalSource.from_memory(base,clusters,LEGACY)
    selected=np.flatnonzero((clusters.frame==50)&(clusters.radius_m==3));b=clusters.iloc[selected].reset_index(drop=True);s=source.query(selected)
    result=decide_clusters(b,profiles(s,b.radius_m.to_numpy(),B1,LEGACY),local_features(s,b.radius_m.to_numpy(),LOCAL),CONFIG)
    assert result.decision_state[0]=='REVIEW' and 'UNKNOWN' not in result.decision_state[0]


def test_transition_diagnostics_partition_same_radius_support():
    from competition_judgement.candidate_diagnostics_v23 import compare_decisions
    b,p,l=inputs();a=decide_clusters(b,p,l,CONFIG);force(p,'A','SUPPORTED');c=decide_clusters(b,p,l,CONFIG)
    rows=compare_decisions(a,c,np.array([3]),'synthetic','test')
    assert len(rows)==2
    assert all(row['new_candidate_n']==1 and row['both_full_dynamic_n']==1 for row in rows)
