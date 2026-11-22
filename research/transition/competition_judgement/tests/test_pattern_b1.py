"""Geometric pattern semantics and observability; no football ground truth."""
import copy
import json
from pathlib import Path
import numpy as np
import pandas as pd
import pytest
from test_evidence_stage_b import scenario,physical_source,CONFIG as LEGACY
from competition_judgement.evidence_features import compute_evidence,corrupt_series,NAMES
from competition_judgement.pattern_values import FEATURES
from competition_judgement.pattern_profiles import profiles
from competition_judgement.pattern_profiles import without_sequence
from competition_judgement.pattern_frames import PatternFrames
from competition_judgement.pattern_source import PatternSource

CONFIG=json.loads((Path(__file__).parents[1]/'configs/pattern_b1.json').read_text())


def data(kind='static',n=1):
    s=scenario(kind,n);s['path_support']=np.ones((n,21),bool);s['path_inside_anchor']=np.ones(n,bool)
    return s


def mask(s,m):
    changed=corrupt_series(s,m);changed['path_support']=s['path_support']&~m[:,40:61]
    changed['path_inside_anchor']=s['path_inside_anchor']&changed['path_support'].all(1)
    return changed


def calculate(s):return profiles(s,np.full(len(s['soft']),3),CONFIG,LEGACY)


def test_static_is_geometric_persistence_not_noncontest():
    p=calculate(data());assert p.C_outcome[0]=='SUPPORTED' and p.C_seq[0]=='VERIFIED'
    assert p.A_outcome[0]=='NOT_OBSERVED' and p.B_outcome[0]=='NOT_OBSERVED'
    assert p.D_outcome[0]=='NOT_OBSERVED'
    assert not any('positive_evidence_score' in x or 'negative_evidence_score' in x for x in p)


def test_move_preserves_multiple_patterns_without_single_label():
    s=data();s['path_motion_m'][:]=1.2;s['path_score'][:]=1
    p=calculate(s);assert p.C_outcome[0]=='SUPPORTED' and p.D_outcome[0]=='SUPPORTED'


@pytest.mark.parametrize('kind',['approach_hold','approach_separate'])
def test_ordered_approach_hold_or_separation(kind):
    p=calculate(data(kind));assert p.A_convergence[0]==1 and p.A_outcome[0]=='SUPPORTED'
    if kind=='approach_separate':assert p.B_outcome[0]=='SUPPORTED' and p.B_seq[0]=='VERIFIED'
    else:assert p.B_outcome[0]=='NOT_OBSERVED'


def test_rapid_approach_without_future_still_complete():
    s=data('approach_hold');m=np.zeros((1,101),bool);m[:,51:]=True
    p=calculate(mask(s,m));assert p.P1_state[0]=='COMPLETE' and p.P2_state[0]=='COMPLETE'
    assert p.A_outcome[0]=='SUPPORTED' and p.B_outcome[0]=='UNKNOWN'


def test_future_separation_does_not_use_unneeded_past():
    s=data('approach_separate');base=calculate(s);m=np.zeros((1,101),bool);m[:,:50]=True
    p=calculate(mask(s,m));assert p.P4_state[0]=='COMPLETE'
    assert p.P4[0]==pytest.approx(base.P4[0]) and p.P1_state[0]=='UNKNOWN'


def test_reapproach_observed_separately():
    s=data('approach_separate');s['edges_3m'][:,80:]=1
    p=calculate(s);assert p.B_reapproach[0]==1


def test_local_concentration_and_composition_ambiguity():
    s=data();t=np.arange(-50,51);s['component_diameter'][:]=1+np.maximum(-t,0)*.15;s['component_radius'][:]=s['component_diameter']/2
    p=calculate(s);assert p.A_concentration[0]==1 and p.A_outcome[0]=='SUPPORTED'
    s['anchor_n_home'][:,30]=2;p=calculate(s)
    assert p.P2_state[0]=='UNKNOWN' and p.A_outcome[0]=='UNKNOWN'


def test_reconfiguration_matching_ambiguity_alone_never_verifies_transient():
    s=data();s['prev_candidate_count'][:]=2;s['next_candidate_count'][:]=2
    s['prev_status'][:]='geometric_ambiguous';s['next_status'][:]='geometric_ambiguous'
    s['path_valid'][:]=False;s['path_support'][:]=False;s['path_score'][:]=np.nan
    p=calculate(s);assert p.P3[0]>0 and p.P3_branch_term[0]>0
    assert p.P3_k_change[0]==0 and p.P3_matching_ambiguity[0]
    assert p.B_outcome[0]=='NOT_OBSERVED' and p.D_outcome[0]=='UNKNOWN'


@pytest.mark.parametrize('offset',[-1,0,1])
def test_center_missing_unknown_not_disappearance(offset):
    s=data('approach_separate');m=np.zeros((1,101),bool);m[:,offset+50]=True
    p=calculate(mask(s,m));assert p.B_outcome[0]=='UNKNOWN' and p.B_seq[0]=='UNVERIFIED'
    assert p.C_outcome[0]=='UNKNOWN'
    if offset==0:assert all(p[k+'_state'][0]=='UNKNOWN' for k in FEATURES)


def test_far_missing_preserves_only_appropriate_dependencies():
    s=data();m=np.zeros((1,101),bool);m[:,0]=True;p=calculate(mask(s,m))
    assert p.N1_state[0]=='EXPERIMENTAL_PARTIAL' and p.N3_state[0]=='EXPERIMENTAL_PARTIAL'
    assert p.N2_state[0]=='COMPLETE' and p.P2_state[0]=='COMPLETE' and p.P4_state[0]=='COMPLETE'
    assert p.C_seq[0]=='UNVERIFIED' and p.C_outcome[0]=='UNKNOWN'


def test_same_value_different_observation_support():
    s=data();x=calculate(s);m=np.zeros((1,101),bool);m[:,0]=True;y=calculate(mask(s,m))
    assert x.N3[0]==pytest.approx(y.N3[0])
    assert x.N3_state[0]=='COMPLETE' and y.N3_state[0]=='EXPERIMENTAL_PARTIAL'


def test_count_change_and_current_qc_are_separate():
    s=data();s['global_n_home'][:,20]=7;s['global_count_same'][:,20]=False;s['reliable'][:,20]=False
    p=calculate(s);assert p.N2_state[0]=='COMPLETE' and p.N1_state[0]=='UNKNOWN'
    assert p.N1_composition_flag[0]&1
    s['strict'][:,50]=False;s['reliable'][:]=False;p=calculate(s)
    assert all(p[k+'_state'][0]=='UNKNOWN' for k in FEATURES)


def test_same_present_different_histories():
    x=calculate(data());y=calculate(data('approach_separate'))
    assert x.current_soft[0]==pytest.approx(np.exp(-1/3))
    s=data('approach_separate');s['soft'][:,50]=np.exp(-1/3);s['nearest_m'][:,50]=1
    z=calculate(s);assert z.current_soft[0]==pytest.approx(x.current_soft[0])
    assert z.C_outcome[0]!=x.C_outcome[0]


def test_high_window_means_do_not_prove_peak_order():
    s=data('approach_separate');s['nearest_m'][:,15:21]=.01;s['soft'][:,15:21]=np.exp(-.01/3)
    s['component_diameter'][:,15:21]=.01;s['component_radius'][:,15:21]=.005
    p=calculate(s);assert p.P1[0]>.5 and p.P4[0]>.5
    assert p.B_outcome[0]=='NOT_OBSERVED' and abs(p.B_peak_offset[0])>5


def test_no_time_array_never_verifies_high_evidence_combination():
    evidence=pd.DataFrame({k:[1.] for k in FEATURES}|{k+'_state':['COMPLETE'] for k in FEATURES})
    p=without_sequence(evidence,CONFIG)
    assert p.B_obs[0]=='COMPLETE' and p.B_seq[0]=='UNVERIFIED' and p.B_outcome[0]=='UNKNOWN'
    assert p.B_reason[0]=='NO_TIME_ARRAY'


def test_no_time_array_rejects_nan_with_stale_complete_flag():
    evidence=pd.DataFrame({k:[1.] for k in FEATURES}|{k+'_state':['COMPLETE'] for k in FEATURES})
    evidence['P4']=np.nan;p=without_sequence(evidence,CONFIG)
    assert p.P4_state[0]=='UNKNOWN' and p.B_obs[0]=='UNKNOWN' and p.B_missing_mask[0]&8


def test_unknown_import_hides_unqualified_numeric_value():
    evidence=pd.DataFrame({k:[1.] for k in FEATURES}|{k+'_state':['COMPLETE'] for k in FEATURES})
    evidence['P4_state']='UNKNOWN';p=without_sequence(evidence,CONFIG)
    assert np.isnan(p.P4[0]) and p.P4_reason[0]=='EVIDENCE_UNKNOWN'


def test_diagnostic_same_support_uses_actual_dependency():
    from competition_judgement.pattern_diagnostics import compare
    a=data();m=np.zeros((1,101),bool);m[:,100]=True;b=mask(a,m)
    rows=compare(calculate(a),calculate(b),a,b,'synthetic','mask',np.array([3]),config=CONFIG)
    p1=next(x for x in rows if x['kind']=='feature' and x['code']=='P1')
    assert p1['same_mask_n']==1


def test_pure_record_half_boundary_only_blocks_needed_side():
    s=data('approach_separate')
    for k in ('record_support','observed','reliable'):s[k][:,51:]=False
    s['path_support'][:,11:]=False;s['path_valid'][:]=False;s['next_status'][:]='boundary_or_outside'
    p=calculate(s)
    assert p.P1_state[0]=='COMPLETE' and p.P2_state[0]=='COMPLETE' and p.P3_state[0]=='UNKNOWN'


def test_new_evidence_can_use_complete_required_scope_when_old_partial_held():
    s=data();m=np.ones((1,101),bool);m[:,40:61]=False;changed=mask(s,m)
    old=compute_evidence(changed,np.array([3]),LEGACY);p=calculate(changed)
    assert np.isnan(old.full_N2_co_movement_proxy[0])
    assert p.N2_state[0]=='COMPLETE' and np.isfinite(p.N2[0])


def test_n2_cannot_fuse_a_corridor_outside_seed_region():
    s=data();s['path_motion_m'][:]=2;s['path_score'][:]=1;s['path_inside_anchor'][:]=False
    p=calculate(s);assert p.N2_state[0]=='COMPLETE' and p.D_outcome[0]=='UNKNOWN'


def test_zero_is_observed_not_unknown():
    p=calculate(data());assert p.P1[0]==pytest.approx(0) and p.P1_state[0]=='COMPLETE'
    s=data();s['record_support'][:]=False;s['observed'][:]=False;s['reliable'][:]=False;s['path_support'][:]=False;s['path_valid'][:]=False
    p=calculate(s);assert np.isnan(p.P1[0]) and all(p[k+'_outcome'][0]=='UNKNOWN' for k in 'ABCD')
    assert not np.isinf(p.select_dtypes(include='number').to_numpy()).any()


@pytest.mark.parametrize('kind',['static','approach_hold','approach_separate'])
def test_frozen_individual_equations_equal_legacy(kind):
    s=data(kind);old=compute_evidence(s,np.array([3]),LEGACY);p=calculate(s)
    for key,name in NAMES.items():np.testing.assert_allclose(p[key],old['full_'+name],equal_nan=True,atol=1e-12,rtol=0)


def test_hidden_unneeded_values_do_not_leak():
    s=data('approach_separate');m=np.zeros((1,101),bool);m[:,90:]=True;a=mask(s,m);b=copy.deepcopy(a)
    for key in ('soft','nearest_m','nearby_count','component_radius','component_diameter'):
        b[key][m]=9999
    x=calculate(a);y=calculate(b)
    for key in ('P1','P2','N2'):np.testing.assert_allclose(x[key],y[key],equal_nan=True)
    assert x.A_outcome[0]==y.A_outcome[0] and x.B_outcome[0]=='UNKNOWN'


def test_different_anchors_never_supply_one_pattern():
    frame=pd.DataFrame(dict(frame=[0],period=[1],strict_quality_valid=[True],n_home=[8],n_away=[8]))
    base=pd.DataFrame(dict(frame=[0,0],radius_m=[3,5],cluster_id=[0,1]))
    p=pd.concat([calculate(data()),calculate(data())],ignore_index=True)
    p.loc[0,['P4_state','B_obs','B_outcome']]=['UNKNOWN','UNKNOWN','UNKNOWN']
    p.loc[1,['P1_state','B_obs','B_outcome']]=['UNKNOWN','UNKNOWN','UNKNOWN']
    agg=PatternFrames(frame);agg.add(base,p);out=agg.table()
    assert out.B_outcome[0]=='UNKNOWN' and out.B_supported_3m_views[0]==0 and out.B_supported_5m_views[0]==0


def test_sufficient_observation_without_pattern_and_empty_graph_distinguished():
    frames=pd.DataFrame(dict(frame=[0,1],period=[1,1],strict_quality_valid=[True,False],n_home=[8,1],n_away=[8,0]))
    out=PatternFrames(frames).table();assert out.A_outcome[0]=='NOT_OBSERVED' and out.A_seq[0]=='NOT_APPLICABLE'
    assert out.A_outcome[1]=='UNKNOWN'


@pytest.mark.parametrize('permute,swap',[(False,False),(True,False),(True,True)])
def test_physical_slot_permutation_team_exchange(monkeypatch,permute,swap):
    base=physical_source(monkeypatch,permute=permute,swap=swap)
    source=PatternSource.from_memory(base,base.clusters,LEGACY)
    selected=np.flatnonzero((source.clusters.frame==50)&(source.clusters.radius_m==3))
    p=profiles(source.query(selected),np.array([3]),CONFIG,LEGACY)
    assert p.N2_state[0]=='COMPLETE' and p.D_outcome[0]=='SUPPORTED' and p.C_outcome[0]=='SUPPORTED'


@pytest.mark.parametrize('split',[False,True])
def test_record_or_split_boundary_blocks_only_required_ranges(monkeypatch,split):
    base=physical_source(monkeypatch,kind='dropout' if not split else 'moving',split=split)
    source=PatternSource.from_memory(base,base.clusters,LEGACY)
    selected=np.flatnonzero((source.clusters.frame==50)&(source.clusters.radius_m==3))
    p=profiles(source.query(selected),np.array([3]),CONFIG,LEGACY)
    assert p.N2_state[0]=='UNKNOWN' and p.D_outcome[0]=='UNKNOWN'
