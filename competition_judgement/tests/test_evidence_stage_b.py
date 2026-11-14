"""Synthetic mathematical scenarios; never actual contest/non-contest labels."""
import copy
import json
from pathlib import Path
import numpy as np
import pandas as pd
import pytest
from competition_judgement.evidence_features import compute_evidence,corrupt_series,NAMES
from competition_judgement.evidence_frames import FrameEvidence

CONFIG=json.loads((Path(__file__).parents[1]/'configs/evidence_stage_b.json').read_text())


def scenario(kind='static',n=1):
    t=np.arange(-50,51)
    distance=np.full(101,1.)
    if kind=='approach_separate':distance=.4+np.abs(t)*.1
    if kind=='approach_hold':distance=.4+np.maximum(-t,0)*.1
    d=np.broadcast_to(distance,(n,101)).copy();active=d<=3
    ones=np.ones((n,101),bool);counts=np.ones((n,101),int)
    s=dict(soft=np.exp(-d/3),nearest_m=d,edges_3m=active.astype(float),edges_5m=(d<=5).astype(float),
        contact_size_3m=2*active.astype(float),contact_size_5m=2*(d<=5).astype(float),
        observed=ones.copy(),reliable=ones.copy(),record_support=ones.copy(),strict=ones.copy(),global_count_same=ones.copy(),
        anchor_n_home=counts.copy(),anchor_n_away=counts.copy(),global_n_home=counts*8,global_n_away=counts*8,
        component_diameter=np.where(active,d,np.nan),component_radius=np.where(active,d/2,np.nan),nearby_count=active.astype(float),
        prev_status=np.full(n,'reciprocal_unique_geometry',object),next_status=np.full(n,'reciprocal_unique_geometry',object),
        prev_candidate_count=np.ones(n),next_candidate_count=np.ones(n),path_valid=np.ones(n,bool),
        path_reason=np.full(n,'ok',object),path_score=np.zeros(n),path_motion_m=np.zeros(n),path_diameter_range_m=np.zeros(n),
        path_radius_range_m=np.zeros(n),path_density_range=np.zeros(n))
    return s


def calculate(s,radius=3):return compute_evidence(s,np.full(len(s['soft']),radius),CONFIG)


def test_static_close_relation_expected_descriptors():
    out=calculate(scenario())
    for p in ('P1','P2','P3','P4'):assert out['full_'+NAMES[p]][0]==pytest.approx(0)
    assert out.full_N1_persistent_marking_proxy[0]==pytest.approx(1)
    assert out.full_N3_stable_proximity[0]==pytest.approx(1)
    assert out.full_N2_co_movement_proxy[0]==0


def test_geometric_motion_proxy_separate_from_static_persistence():
    s=scenario();s['path_motion_m'][:]=2;s['path_score'][:]=1
    out=calculate(s)
    assert out.full_N2_co_movement_proxy[0]==1
    assert out.full_negative_evidence_score[0]==1


@pytest.mark.parametrize('kind',['approach_separate','approach_hold'])
def test_approach_not_requiring_subsequent_separation(kind):
    out=calculate(scenario(kind))
    assert out.full_P1_rapid_convergence[0]>.7
    if kind=='approach_separate':assert out.full_P4_subsequent_separation[0]>.5
    else:assert out.full_P4_subsequent_separation[0]==pytest.approx(0)


def test_multiple_observations_concentrate_without_count_change():
    s=scenario();t=np.arange(-50,51)
    s['anchor_n_home'][:]=2;s['anchor_n_away'][:]=2
    s['component_diameter'][:]=1+np.maximum(-t,0)*.15
    s['component_radius'][:]=s['component_diameter']/2
    s['edges_3m'][:]=np.where(t< -10,1,4);s['contact_size_3m'][:]=np.where(t< -10,2,4)
    out=calculate(s)
    assert out.full_P2_local_concentration[0]>.3
    assert not out.full_P2_local_composition_changed[0]
    s['anchor_n_home'][:,30:40]=1
    out=calculate(s)
    assert np.isnan(out.full_P2_local_concentration[0])
    assert out.full_P2_reason[0]=='local_composition_ambiguous'


def test_geometric_merge_split_ambiguity_increases_reconfiguration_withholds_motion():
    s=scenario();s['nearby_count'][:,:50]=2;s['nearby_count'][:,51:]=2
    s['prev_candidate_count'][:]=2;s['next_candidate_count'][:]=2
    s['prev_status'][:]='geometric_ambiguous';s['next_status'][:]='geometric_ambiguous'
    s['path_valid'][:]=False;s['path_score'][:]=np.nan;s['path_reason'][:]='geometric_ambiguous'
    out=calculate(s)
    assert out.full_P3_relationship_reconfiguration[0]>.4
    assert np.isnan(out.full_N2_co_movement_proxy[0])
    assert out.N2_path_reason[0]=='geometric_ambiguous'


@pytest.mark.parametrize('offset',[-1,0,1])
def test_center_dropout_withholds_temporal_evidence(offset):
    s=scenario();deleted=np.zeros((1,101),bool);deleted[:,50+offset]=True
    out=calculate(corrupt_series(s,deleted))
    for name in NAMES.values():assert np.isnan(out['partial_'+name][0])
    if offset==0:assert not out.current_observed[0]
    else:assert out.close_duration_lower_bound_s[0]<4.04


def test_far_end_gap_support_is_feature_specific():
    s=scenario('approach_separate');deleted=np.zeros((1,101),bool);deleted[:,:5]=True
    out=calculate(corrupt_series(s,deleted))
    assert not out.full_reliable_window[0]
    assert np.isfinite(out.partial_P1_rapid_convergence[0])
    assert np.isfinite(out.partial_P2_local_concentration[0])
    assert np.isnan(out.partial_N1_persistent_marking_proxy[0])
    assert np.isnan(out.partial_N3_stable_proximity[0])


def test_global_count_changes_record_timing_and_do_not_equal_current_qc_failure():
    s=scenario();s['global_n_home'][:,20:30]=7;s['global_count_same'][:,20:30]=False;s['reliable'][:,20:30]=False
    out=calculate(s)
    assert out.current_strict[0] and out.global_count_different_n[0]==10
    assert out.global_count_transition_n[0]==2
    assert out.global_count_transition_offsets[0]=='-30|-20'
    assert out.global_count_first_different_offset[0]==-30
    s['strict'][:,50]=False;s['reliable'][:]=False
    out=calculate(s)
    assert out.partial_P1_reason[0]=='current_strict_qc_failed'


def test_global_counts_remain_known_when_only_the_local_anchor_is_unobserved():
    s=scenario();s['observed'][:,20:30]=False;s['reliable'][:,20:30]=False
    s['global_n_home'][:,20:30]=7;s['global_count_same'][:,20:30]=False
    out=calculate(s)
    assert out.global_count_different_n[0]==10 and out.global_count_transition_n[0]==2
    deleted=np.zeros((1,101),bool);deleted[:,20:30]=True
    changed=calculate(corrupt_series(s,deleted))
    assert changed.global_count_different_n[0]==0


def test_artificial_path_loss_hides_path_descriptors_as_well_as_score():
    s=scenario();s['path_score'][:]=1;s['path_motion_m'][:]=2
    deleted=np.zeros((1,101),bool);deleted[:,45]=True
    changed=corrupt_series(s,deleted)
    assert not changed['path_valid'][0] and np.isnan(changed['path_score'][0]) and np.isnan(changed['path_motion_m'][0])


def test_unobserved_future_is_not_known_separation():
    s=scenario('approach_hold');s['observed'][:,75:]=False;s['reliable'][:,75:]=False
    out=calculate(s)
    assert out.future_end_reason[0]=='observation_or_region_ambiguous'
    assert np.isnan(out.partial_P4_subsequent_separation[0])


def test_negative_family_never_sums_N1_and_N3():
    s=scenario();out=calculate(s)
    assert out.full_negative_persistence_source[0]=='N1'
    assert out.full_negative_family_n[0]==2
    assert out.full_negative_evidence_score[0]==pytest.approx(.5)
    assert out.full_positive_evidence_score[0]==pytest.approx(0,abs=1e-14)
    # Independence, not complements or a unit sum.
    s['path_score'][:]=.2;out=calculate(s)
    assert out.full_positive_evidence_score[0]+out.full_negative_evidence_score[0]==pytest.approx(.6)


def test_all_missing_is_nan_and_numeric_metadata_has_no_inf():
    s=scenario();s['observed'][:]=False;s['reliable'][:]=False;s['path_valid'][:]=False
    out=calculate(s)
    for mode in ('full','partial'):
        for name in NAMES.values():assert np.isnan(out[mode+'_'+name][0])
        assert np.isnan(out[mode+'_positive_evidence_score'][0]) and np.isnan(out[mode+'_negative_evidence_score'][0])
    assert not np.isinf(out.select_dtypes(include='number').to_numpy()).any()


@pytest.mark.parametrize('radius',[3,5])
def test_full_and_experimental_estimators_agree_when_full_supported(radius):
    out=calculate(scenario('approach_separate'),radius)
    for name in NAMES.values():np.testing.assert_allclose(out['full_'+name],out['partial_'+name],equal_nan=True)


def test_home_away_swap_and_observation_order_do_not_affect_scores():
    s=scenario('approach_separate',2);other=copy.deepcopy(s)
    for a,b in [('anchor_n_home','anchor_n_away'),('global_n_home','global_n_away')]:other[a],other[b]=s[b],s[a]
    x=calculate(s);y=calculate(other)
    for mode in ('full','partial'):
        for name in NAMES.values():np.testing.assert_allclose(x[mode+'_'+name],y[mode+'_'+name],equal_nan=True)
    # Batch order has no causal/persistent-ID interpretation.
    reversed_={k:(v[::-1] if isinstance(v,np.ndarray) else v) for k,v in s.items()}
    z=calculate(reversed_)
    for name in NAMES.values():np.testing.assert_allclose(x['full_'+name],z['full_'+name][::-1],equal_nan=True)


def test_hidden_values_do_not_change_partial_scores():
    s=scenario('approach_separate');deleted=np.zeros((1,101),bool);deleted[:,2:4]=True
    a=corrupt_series(s,deleted);b=copy.deepcopy(a)
    for k in ('soft','nearest_m','edges_3m','edges_5m','component_radius','component_diameter','nearby_count',
              'anchor_n_home','anchor_n_away','global_n_home','global_n_away','contact_size_3m','contact_size_5m'):
        b[k]=b[k].astype(float);b[k][deleted]=12345
    x=calculate(a);y=calculate(b)
    for name in NAMES.values():np.testing.assert_allclose(x['partial_'+name],y['partial_'+name],equal_nan=True)


def test_frame_maxima_keep_distinct_provenance_and_coherent_pair():
    frames=pd.DataFrame(dict(frame=[0],period=[1],quality_tier=['strict_valid'],strict_quality_valid=[True]))
    base=pd.DataFrame(dict(frame=[0,0],period=[1,1],radius_m=[3,5],cluster_id=[0,1]))
    out=calculate(scenario(n=2))
    for mode in ('full','partial'):
        out[mode+'_positive_evidence_score']=[.9,.2];out[mode+'_negative_evidence_score']=[.1,.9]
    reducer=FrameEvidence(frames);reducer.add(base,out,np.array([0,1]));result=reducer.table()
    assert result.full_positive_evidence_score_max[0]==.9 and result.full_negative_evidence_score_max[0]==.9
    assert result.full_positive_evidence_score_source_radius_m[0]==3
    assert result.full_negative_evidence_score_source_radius_m[0]==5
    assert result.full_negative_at_positive_representative[0]==.1


def physical_source(monkeypatch,kind='moving',permute=False,swap=False,split=False):
    """Actual legacy geometry + matching functions, entirely synthetic positions."""
    from mtah_relational.frame_store import FrameStore
    from competition_judgement.local_features import extract_local_frame
    from competition_judgement.temporal_source import TemporalSource
    from competition_judgement.temporal_matching import adjacent_matches
    from competition_judgement.evidence_source import EvidenceSource
    from competition_judgement.missingness_cache import ReadOnlyAnchorCache
    cfg=json.loads((Path(__file__).parents[1]/'configs/temporal_relations_v2_2.json').read_text())
    local=json.loads((Path(__file__).parents[1]/'configs/local_clusters_v2_1.json').read_text())
    h=np.array([[50.,34.],[10,5],[20,5],[30,5],[10,15],[20,15],[30,15],[10,25]])
    a=np.array([[51.,34.],[75,5],[85,5],[95,5],[75,15],[85,15],[95,15],[75,25]])
    xy=np.broadcast_to(np.r_[h,a],(101,16,2)).copy();teams=np.r_[np.zeros(8,int),np.ones(8,int)]
    if kind=='moving':xy[:,[0,8],0]+=np.arange(-50,51)[:,None]*.06
    if kind=='dropout':xy[45,0]=np.nan
    if swap:teams=1-teams
    if permute:
        rng=np.random.default_rng(2027)
        for t in range(101):xy[t,:8]=xy[t,rng.permutation(8)];xy[t,8:]=xy[t,8+rng.permutation(8)]
    store=FrameStore(np.arange(101),np.ones(101,int),xy,teams,(105.3,68.))
    frame=[];clusters=[]
    for t in range(101):
        f,c,_=extract_local_frame(store.frame(t),store.pitch,local);frame.append(f);clusters.extend(c)
    frame=pd.DataFrame(frame);clusters=pd.DataFrame(clusters)
    class Manifest:
        def split_of(self,f):return 'validation' if not split or f>=45 else 'train'
    manifest=Manifest();old=TemporalSource(store,frame,manifest,cfg,clusters=clusters)
    matching=pd.concat([clusters[['frame','radius_m','cluster_id']],adjacent_matches(old,clusters,cfg)],axis=1)
    monkeypatch.setattr(ReadOnlyAnchorCache,'__init__',lambda self,*args,**kwargs:self.__dict__.update(old.__dict__))
    return EvidenceSource(None,frame,None,clusters,matching,CONFIG)


def test_actual_synthetic_geometry_motion_and_reference_series(monkeypatch):
    source=physical_source(monkeypatch)
    selected=np.flatnonzero((source.clusters.frame==50)&(source.clusters.radius_m==3))
    s=source.query(selected);out=compute_evidence(s,np.array([3]),CONFIG)
    assert s['path_valid'][0]
    assert s['path_motion_m'][0]==pytest.approx(1.2)
    assert out.full_N2_co_movement_proxy[0]>.99
    assert out.full_N1_persistent_marking_proxy[0]>.99
    assert s['reliable'].all()


def test_actual_synthetic_slot_permutation_and_team_exchange(monkeypatch):
    outputs=[]
    for permute,swap in [(False,False),(True,False),(True,True)]:
        source=physical_source(monkeypatch,permute=permute,swap=swap)
        selected=np.flatnonzero((source.clusters.frame==50)&(source.clusters.radius_m==3))
        outputs.append(compute_evidence(source.query(selected),np.array([3]),CONFIG))
    for other in outputs[1:]:
        for name in NAMES.values():np.testing.assert_allclose(outputs[0]['full_'+name],other['full_'+name],atol=1e-6,equal_nan=True)


@pytest.mark.parametrize('kind,split',[('dropout',False),('moving',True)])
def test_actual_geometric_corridor_cannot_bridge_missing_or_split(monkeypatch,kind,split):
    source=physical_source(monkeypatch,kind=kind,split=split)
    selected=np.flatnonzero((source.clusters.frame==50)&(source.clusters.radius_m==3))
    s=source.query(selected);out=compute_evidence(s,np.array([3]),CONFIG)
    assert not s['path_valid'][0]
    assert np.isnan(out.partial_N2_co_movement_proxy[0])


def test_spearman_uses_average_tie_ranks_without_scipy():
    from competition_judgement.evidence_diagnostics import correlations
    t=pd.DataFrame(dict(a=[1.,2.,2.,3.,np.nan],b=[3.,2.,2.,1.,0.]))
    row=correlations(t,[('a','b')],'synthetic')[0]
    assert row['n']==4 and row['spearman']==pytest.approx(-1)


def test_masking_comparison_rejects_changed_p4_term_weights():
    from competition_judgement.evidence_diagnostics import identical_components
    clean=pd.DataFrame(dict(full_positive_used_mask=[15,7],P4_term_mask=[15,15]))
    changed=pd.DataFrame(dict(partial_positive_used_mask=[15,7],P4_term_mask=[7,7]))
    np.testing.assert_array_equal(identical_components(clean,changed,'P4_subsequent_separation'),[False,False])
    np.testing.assert_array_equal(identical_components(clean,changed,'positive_evidence_score'),[False,True])
