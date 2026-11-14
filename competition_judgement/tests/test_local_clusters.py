"""Synthetic local spatial tests. No match data, labels or model fitting."""
from dataclasses import replace
import json
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from mtah_relational.data_quality import PlayerFrame,assess_observations
from mtah_relational.features import max_bipartite_cluster_size
from mtah_relational.frame_store import FrameStore
from competition_judgement.quality import POLICY,observation_row
from competition_judgement.local_clusters import build_mixed_graph,connected_mixed_sets
from competition_judgement.local_features import extract_local_frame,component_features
from competition_judgement.local_diagnostics import deletion_trials,paired_change
from competition_judgement.analyze_local_clusters import extract_store,descriptive_cases,validate_local_config,execute_local
from competition_judgement.compare_local_robustness import equal_support_change
from mtah_relational.candidate_baseline import CausalContestedIndex

CONFIG=json.loads((Path(__file__).parents[1]/"configs/local_clusters_v2_1.json").read_text())
PITCH=(105.3,68.)


def observation(points,teams,frame=0,period=1):
    xy=np.asarray(points,dtype=float).reshape(-1,2);teams=np.asarray(teams)
    missing=np.isnan(xy).all(axis=1)
    partial=np.isnan(xy).any(axis=1)&~missing
    invalid=np.isinf(xy).any(axis=1)
    return PlayerFrame(frame,period,xy,teams,missing,partial,invalid)


def features(points,teams):
    return extract_local_frame(observation(points,teams),PITCH,CONFIG)


def test_two_observation_cluster_exact_formulas():
    frame,rows,g=features([[10,10],[12,10]],[0,1])
    assert len(rows)==2 and frame['n_local_clusters_3m']==frame['n_local_clusters_5m']==1
    r=rows[0]
    assert r['n_players']==r['n_unique_contact_players']==2
    assert r['n_cross_edges']==1 and r['cluster_radius_m']==1
    assert r['cluster_bbox_length_m']==2 and r['cluster_bbox_width_m']==0
    assert r['cluster_bbox_area_m2']==0 and r['cluster_max_pairwise_distance_m']==2
    assert r['cluster_centroid_x']==11 and r['cluster_centroid_y']==10
    assert r['team_balance']==r['cross_pair_density']==r['edge_density']==1
    assert r['max_degree']==r['mean_degree']==1
    assert r['contact_strength']==pytest.approx(np.exp(-2/3))
    assert all(x['same_node_set_across_radii'] for x in rows)


def test_three_observation_star_and_degree_denominators():
    _,rows,_=features([[10,10],[12,10],[10,12]],[0,1,1])
    r=rows[0]
    assert r['n_cross_edges']==2 and r['component_size']==3
    assert r['cross_pair_density']==1 and r['edge_density']==pytest.approx(2/3)
    assert r['mean_degree']==pytest.approx(4/3) and r['max_degree']==2
    assert r['home_fraction']==pytest.approx(1/3) and r['away_fraction']==pytest.approx(2/3)
    assert r['team_balance']==pytest.approx(2/3)


def test_multiple_clusters_and_scale_families_no_event_sum():
    f,rows,g=features([[10,10],[11,10],[14.5,10],[15.5,10],[40,10],[41,10]],[0,1,0,1,0,1])
    assert f['n_local_clusters_3m']==3 and f['n_local_clusters_5m']==2
    c3=g.components[3.];c5=g.components[5.]
    assert c3[0].family_id_5m==c3[1].family_id_5m
    assert c3[2].family_id_5m!=c3[0].family_id_5m
    assert f['total_unique_players_in_mixed_clusters']==6
    assert sum(c.same_node_set_across_radii for c in c3)==1


def test_long_chain_size_is_not_compactness():
    f,rows,g=features([[10+i*2.5,10] for i in range(6)],[0,1,0,1,0,1])
    r=[r for r in rows if r['radius_m']==3][0]
    assert r['n_players']==6 and r['cluster_max_pairwise_distance_m']==12.5
    assert not r['compact_mixed_cluster'] and f['n_compact_mixed_clusters']==0
    assert r['cross_pair_density']==pytest.approx(5/9)


@pytest.mark.parametrize('distance,expected3,expected5',[(3.,1,1),(3.000001,0,1),(5.,0,1),(5.000001,0,0)])
def test_exact_radius_edges(distance,expected3,expected5):
    f,rows,_=features([[10,10],[10+distance,10]],[0,1])
    assert f['n_local_clusters_3m']==expected3 and f['n_local_clusters_5m']==expected5


def test_same_team_edges_and_zone_boundaries():
    x=105.3/8
    f,_,_=features([[x-.1,10],[x+.1,10],[x+.1,11]],[0,1,0])
    assert f['max_local_cluster_size']==3
    f,rows,_=features([[10,10],[11,10]],[0,0])
    assert not f['local_metric_available'] and np.isnan(f['n_local_clusters_3m']) and not rows


@pytest.mark.parametrize('points,teams',[([],[]),([[10,10]],[0]),([[10,10],[20,20]],[1,1])])
def test_no_opponent_is_undefined(points,teams):
    f,rows,_=features(points,teams)
    assert not f['local_metric_available'] and f['quality_tier']=='insufficient'
    assert np.isnan(f['max_local_cluster_size']) and np.isnan(f['max_local_contact_strength']) and rows==[]


def test_observed_but_no_edges_has_zero_counts_and_undefined_extrema():
    f,rows,_=features([[10,10],[50,50]],[0,1])
    assert f['local_metric_available'] and f['n_local_clusters_3m']==0
    assert f['n_cross_edges_5m']==0 and np.isnan(f['max_local_cross_density'])
    assert np.isnan(f['largest_cluster_radius_m']) and rows==[]


@pytest.mark.parametrize('swap',[False,True])
def test_player_permutation_and_team_swap(swap):
    points=np.array([[10,10],[11,10],[10,11],[50,10],[52,10]],float);team=np.array([0,1,1,0,1])
    a,ar,_=features(points,team)
    order=np.random.default_rng(2027).permutation(5)
    b,br,_=features(points[order],(1-team if swap else team)[order])
    for name in a:
        if name not in ('n_home','n_away'):
            if isinstance(a[name],str): assert a[name]==b[name]
            else: np.testing.assert_allclose(a[name],b[name],equal_nan=True,atol=1e-12)
    for ra,rb in zip(ar,br):
        for name in ra:
            partner={'n_home':'n_away','n_away':'n_home','home_fraction':'away_fraction','away_fraction':'home_fraction'}.get(name,name) if swap else name
            if isinstance(ra[name],str): assert ra[name]==rb[partner]
            else: assert ra[name]==pytest.approx(rb[partner],abs=1e-12)


def test_duplicate_observations_are_retained_and_quality_flagged():
    f,rows,_=features([[10,10],[10,10],[11,10]],[0,0,1])
    assert f['duplicate_suspect_pairs']==1 and f['quality_tier']=='low'
    assert f['local_metric_available'] and f['max_local_cluster_size']==3
    assert rows[0]['n_home']==2


def test_nan_inf_and_bounds_reuse_existing_policy():
    p=[[10,10],[11,10],[np.nan,np.nan],[np.inf,3],[-3,5],[-1,5]]
    f,rows,g=features(p,[0,1,0,1,0,0])
    assert f['n_home']==2 and f['n_away']==1
    assert f['excluded_bounds_slots']==1 and f['invalid_slots']==1
    assert f['outside_pitch_slots']==2 and f['quality_tier']=='low'
    assert len(g.xy_m)==3 and np.isfinite(g.xy_m).all()
    assert all(g.observed)


def test_missing_observation_cannot_create_new_connection():
    f=observation([[10,10],[12.5,10],[15,10],[17.5,10]],[0,1,0,1])
    clean,_,_=extract_local_frame(f,PITCH,CONFIG)
    xy=f.xy_m.copy();xy[1]=np.nan
    after,_,_=extract_local_frame(replace(f,xy_m=xy,parse_missing=np.isnan(xy).all(1)),PITCH,CONFIG)
    assert after['total_unique_players_in_mixed_clusters']==2
    assert clean['max_local_cluster_size']==4 and after['max_local_cluster_size']==2
    assert np.isfinite(f.xy_m).all()


def test_single_member_deletion_splits_chain_not_pair():
    _,_,g=features([[10+2.5*i,10] for i in range(6)],[0,1,0,1,0,1])
    trials=deletion_trials(g,3.)
    assert trials['member_deletions']==6 and trials['split_into_multiple_mixed']==2
    assert trials['split_including_isolated']==4
    _,_,g=features([[10,10],[11,10]],[0,1])
    trials=deletion_trials(g,3.)
    assert trials['split_into_multiple_mixed']==0 and trials['no_surviving_mixed_component']==2


def test_max_size_matches_existing_helper_and_nested_membership():
    rng=np.random.default_rng(7)
    for n in range(2,25):
        points=rng.uniform([0,0],[20,20],(n,2));team=rng.integers(0,2,n)
        g=build_mixed_graph(points,team)
        for r in (3.,5.):
            mask=g.distance_m[np.ix_(g.team==0,g.team==1)]<=r
            assert max((len(c.nodes) for c in g.components[r]),default=0)==max_bipartite_cluster_size(mask)
        for c in g.components[3.]: assert set(c.nodes)<=set(g.components[5.][c.family_id_5m].nodes)


def test_full_frame_axis_and_v1_unmodified(tmp_path):
    n=199319
    # Vectorized full-axis comparison, no licensed observations or graph repetition.
    t=pd.DataFrame(dict(frame=np.arange(n),period=np.ones(n,int),split=['train']*n,
           n_home=np.ones(n,int),n_away=np.ones(n,int),local_metric_available=np.ones(n,bool),
           v1_candidate_state=['not_candidate']*n,v1_candidate_score=np.zeros(n)))
    for r in (3,5):
        t[f'has_strong_compact_relation_{r}m']=True;t[f'max_cluster_diameter_m_{r}m']=1.
        t[f'n_local_clusters_{r}m']=1;t[f'max_local_cluster_size_{r}m']=2;t[f'n_cross_edges_{r}m']=1
    before=t[['frame','v1_candidate_score','v1_candidate_state']].copy(deep=True)
    after=descriptive_cases(t,.3,CONFIG)
    assert len(after)==n and np.array_equal(after.frame,np.arange(n))
    pd.testing.assert_frame_equal(before,after[['frame','v1_candidate_score','v1_candidate_state']])
    assert after.case1_v1_negative_local_strong_3m.all()
    assert not after.case4_count_change_structure_jump_3m.any()


def test_synthetic_cache_to_frame_and_long_csv(tmp_path):
    points=np.array([[10,10],[11,10],[40,40],[41,40]],float)
    xy=np.repeat(points[None],6,axis=0);xy[3:]=np.nan
    store=FrameStore(np.arange(6),np.ones(6,int),xy,np.array([0,1,0,1]),PITCH)
    base=[observation_row(store.frame(i),PITCH) for i in range(6)]
    v1=pd.DataFrame(dict(frame=np.arange(6),period=np.ones(6,int),split=['train']*6,
            n_home=[x['n_home'] for x in base],n_away=[x['n_away'] for x in base],
            strict_quality_valid=[x['quality_valid'] for x in base],metric_valid=[x['cross_valid'] for x in base],
            candidate_score=[np.nan]*6,candidate_state=['insufficient_observation']*6))
    before=v1.copy(deep=True);path=tmp_path/'clusters.csv'
    frames,checks=extract_store(store,v1,CONFIG,path)
    clusters=pd.read_csv(path)
    assert len(frames)==6 and len(clusters)==12 and checks==12
    assert frames.n_local_clusters_3m.iloc[:3].eq(2).all()
    assert frames.n_local_clusters_3m.iloc[3:].isna().all()
    assert not clusters.duplicated(['frame','radius_m','cluster_id']).any()
    pd.testing.assert_frame_equal(v1,before)
    invalid=v1.copy();invalid.loc[1,'frame']=99
    with pytest.raises(ValueError):extract_store(store,invalid,CONFIG,path)


def test_gap_period_and_split_stop_case4_comparisons():
    f=pd.DataFrame(dict(frame=[1,2,4,5,6],period=[1,1,1,2,2],split=['train','train','train','train','validation'],
                       n_home=[1,2,3,4,5],n_away=[1]*5,local_metric_available=[True]*5,
                       v1_candidate_state=['not_candidate']*5,v1_candidate_score=[0.]*5))
    for r in (3,5):
        for name in ('max_cluster_diameter_m','n_local_clusters','max_local_cluster_size','n_cross_edges'):
            f[f'{name}_{r}m']=np.arange(5)+1
        f[f'has_strong_compact_relation_{r}m']=True
    o=descriptive_cases(f,.3,CONFIG)
    assert o.case4_comparison_available.tolist()==[False,True,False,False,False]


def test_config_validation_output_guard_and_shape_errors(tmp_path):
    validate_local_config(CONFIG)
    c=dict(CONFIG,fit_required=True)
    with pytest.raises(ValueError):validate_local_config(c)
    for xy,t in [(np.zeros((2,3)),[0,1]),(np.zeros((2,2)),[0,2])]:
        with pytest.raises(ValueError):build_mixed_graph(xy,t)
    with pytest.raises(ValueError):connected_mixed_sets(np.ones((2,2),bool))
    out=tmp_path/'data/derived/competition_judgement/existing';out.mkdir(parents=True)
    with pytest.raises(FileExistsError):execute_local(root=tmp_path,output=out,cache='missing',v1_dir='missing',config_path='missing',split_path='missing')


def test_far_unconnected_observation_does_not_dilute_local_strength():
    h=[[50,34]]+[[10+10*i,5] for i in range(7)]
    a=[[51,34]]+[[10+10*i,65] for i in range(7)]
    f=observation(h+a,[0]*8+[1]*8)
    after=observation(h+[[90,5]]+a,[0]*9+[1]*8)
    first,_,_=extract_local_frame(f,PITCH,CONFIG);second,_,_=extract_local_frame(after,PITCH,CONFIG)
    assert first['max_local_contact_strength']==second['max_local_contact_strength']
    before_geom,after_geom=observation_row(f,PITCH),observation_row(after,PITCH)
    s1=CausalContestedIndex().step(0,1,before_geom,True)['contested_index']
    s2=CausalContestedIndex().step(0,1,after_geom,True)['contested_index']
    assert s2<s1


def test_equal_support_excludes_missing_local_values_for_both_scores():
    a=pd.DataFrame(dict(frame=[0,1,2],v1_score=[.1,.2,np.nan],local=[.3,np.nan,.5]))
    b=pd.DataFrame(dict(frame=[0,1,2],v1_score=[.2,.4,.6],local=[.6,.7,.8]))
    result=equal_support_change(a,b,'local')
    assert result['n']==result['v1_absolute_change']['n']==result['local_absolute_change']['n']==1
    assert result['v1_absolute_change']['p95']==pytest.approx(.1)
    assert result['local_absolute_change']['p95']==pytest.approx(.3)
