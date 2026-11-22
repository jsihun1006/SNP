"""Synthetic offline temporal geometry, not contest or possession labels."""
from dataclasses import replace
import json
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from mtah_relational.frame_store import FrameStore
from mtah_relational.splits import SplitManifest,TimeBlock
from competition_judgement.local_features import extract_local_frame
from competition_judgement.temporal_source import TemporalSource
from competition_judgement.temporal_features import analyze_anchor_batch
from competition_judgement.temporal_matching import reciprocal_unique,adjacent_matches
from competition_judgement.analyze_temporal_relations import validate_temporal_config,extract_temporal,frame_temporal_features,execute

ROOT=Path(__file__).parents[1]
CONFIG=json.loads((ROOT/'configs/temporal_relations_v2_2.json').read_text())
LOCAL=json.loads((ROOT/'configs/local_clusters_v2_1.json').read_text())
PITCH=(105.3,68.)


def scene(distances,missing=None,ids=None,periods=None):
    n=len(distances);ids=np.arange(n) if ids is None else np.asarray(ids)
    periods=np.ones(n,int) if periods is None else np.asarray(periods)
    xy=np.empty((n,16,2),float)
    for i,d in enumerate(distances):
        xy[i,0]=[50-d/2,34];xy[i,8]=[50+d/2,34]
        xy[i,1:8]=[[10+10*j,5] for j in range(7)]
        xy[i,9:16]=[[10+10*j,65] for j in range(7)]
    if missing is not None:
        for row,slot in missing:xy[row,slot]=np.nan
    store=FrameStore(ids,periods,xy,np.r_[np.zeros(8,int),np.ones(8,int)],PITCH)
    frame=[];clusters=[]
    for row in range(n):
        f,c,_=extract_local_frame(store.frame(row),PITCH,LOCAL)
        f['v1_candidate_state']='not_candidate';f['v1_candidate_score']=.2
        frame.append(f);clusters.extend(c)
    return store,pd.DataFrame(frame),pd.DataFrame(clusters)


def analyze(distances,missing=None,config=CONFIG):
    store,f,c=scene(distances,missing)
    manifest=SplitManifest((TimeBlock(1,'train',0,len(store)-1),),(.7,.15,.15))
    source=TemporalSource(store,f,manifest,config,clusters=c)
    anchor=c.loc[c.frame.eq(len(store)//2)&c.radius_m.eq(3)].reset_index(drop=True)
    return analyze_anchor_batch(source,anchor),source,c,f


def test_static_and_transient_same_present_different_time_descriptors():
    peak=1+8*np.abs(np.arange(101)-50)/50
    transient,_,_,_=analyze(peak)
    stable,_,_,_=analyze(np.ones(101))
    a,b=transient.iloc[0],stable.iloc[0]
    assert a.pattern_score_valid and b.pattern_score_valid
    assert a.present_anchor_soft==pytest.approx(b.present_anchor_soft)
    assert a.approach_focus_separation_score>.1 and b.approach_focus_separation_score==pytest.approx(0)
    assert b.stable_close_score>a.stable_close_score
    assert a.observed_nearest_approach_m>0 and a.observed_nearest_future_change_m>0
    assert not a.onset_censored and b.onset_censored
    assert a.close_duration_lower_bound_s<b.close_duration_lower_bound_s


def test_order_information_beyond_identical_mean():
    a=np.r_[np.full(50,8.),1.,np.full(50,2.)]
    b=np.r_[np.full(50,2.),1.,np.full(50,8.)]
    x,_,_,_=analyze(a);y,_,_,_=analyze(b)
    assert x.observed_window_soft_mean.iloc[0]==pytest.approx(y.observed_window_soft_mean.iloc[0])
    assert x.global_v2_moving_mean_3m.iloc[0]==pytest.approx(y.global_v2_moving_mean_3m.iloc[0])
    assert x.observed_approach.iloc[0]>y.observed_approach.iloc[0]
    assert x.observed_future_change.iloc[0]>y.observed_future_change.iloc[0]


def test_dropout_is_ambiguous_not_observed_separation():
    t,_,_,_=analyze(np.ones(101),missing=[(60,0)])
    a=t.iloc[0]
    assert not a.pattern_score_valid and np.isnan(a.approach_focus_separation_score)
    assert a.future_end_reason=='observation_or_region_ambiguous'
    assert a.end_censored and a.global_count_change
    assert np.isnan(a.reapproach_observed)


def test_actual_spatial_separation_and_reapproach_observed_without_missing():
    d=np.ones(101);d[60:70]=8.
    t,_,_,_=analyze(d)
    assert t.future_end_reason.iloc[0]=='observed_local_relation_end'
    assert not t.end_censored.iloc[0]
    assert t.reapproach_observed.iloc[0]==1


def test_region_exit_is_unknown_even_with_global_good_quality():
    d=np.ones(101);d[60:]=20.
    t,_,_,_=analyze(d)
    assert t.future_end_reason.iloc[0]=='observation_or_region_ambiguous'
    assert t.observed_future_frames.iloc[0]==9
    assert not t.pattern_score_valid.iloc[0]


def test_soft_kernel_smooth_at_hard_boundary():
    d=np.full(101,3.);d[::2]=2.99;d[1::2]=3.01
    t,_,_,_=analyze(d)
    assert .4<t.observed_occupancy_3m.iloc[0]<.6
    assert t.observed_occupancy_5m.iloc[0]==1
    assert t.observed_window_soft_std.iloc[0]<.01


@pytest.mark.parametrize('radius',[25,50,75])
def test_configured_window_shape(radius):
    c=dict(CONFIG,window_radius_frames=radius)
    t,s,_,_=analyze(np.ones(2*radius+1),config=c)
    assert t.record_past_frames.iloc[0]==radius and t.record_future_frames.iloc[0]==radius
    assert t.close_duration_lower_bound_s.iloc[0]==pytest.approx((2*radius+1)/25)


def test_split_boundary_and_outside_sentinel_cannot_leak():
    store,f,c=scene(np.ones(101));manifest=SplitManifest((TimeBlock(1,'train',0,49),TimeBlock(1,'validation',50,100)),(.7,.15,.15))
    anchor=c.loc[c.frame.eq(49)&c.radius_m.eq(3)].reset_index(drop=True)
    s=TemporalSource(store,f,manifest,CONFIG,clusters=c);a=analyze_anchor_batch(s,anchor)
    xy=store.xy_m.copy();xy[50:,:,0]+=2
    changed=FrameStore(store.frame_ids,store.periods,xy,store.team_codes,PITCH)
    s2=TemporalSource(changed,f,manifest,CONFIG,clusters=c);b=analyze_anchor_batch(s2,anchor)
    pd.testing.assert_frame_equal(a,b)
    assert a.record_future_frames.iloc[0]==0 and not a.full_record_window.iloc[0]
    m=adjacent_matches(s,c,CONFIG)
    assert m.loc[c.frame.eq(49),'match_next_status'].eq('boundary_or_outside').all()


@pytest.mark.parametrize('half_change',[False,True])
def test_record_gap_and_half_edges(half_change):
    ids=np.r_[np.arange(50),np.arange(51,102)];periods=np.r_[np.ones(50,int),np.full(51,2 if half_change else 1)]
    store,f,c=scene(np.ones(101),ids=ids,periods=periods)
    blocks=(TimeBlock(1,'train',0,49),TimeBlock(2,'train',51,101)) if half_change else (TimeBlock(1,'train',0,101),)
    manifest=SplitManifest(blocks,(.7,.15,.15))
    source=TemporalSource(store,f,manifest,CONFIG,clusters=c)
    anchor=c.loc[c.frame.eq(49)&c.radius_m.eq(3)].reset_index(drop=True)
    t=analyze_anchor_batch(source,anchor)
    assert t.record_future_frames.iloc[0]==0
    assert t.future_end_reason.iloc[0]=='boundary_censored'


@pytest.mark.parametrize('swap',[False,True])
def test_player_permutation_and_team_swap_temporal_invariance(swap):
    store,f,c=scene(np.ones(101));manifest=SplitManifest((TimeBlock(1,'train',0,100),),(.7,.15,.15))
    anchor=c.loc[c.frame.eq(50)&c.radius_m.eq(3)].reset_index(drop=True)
    a=analyze_anchor_batch(TemporalSource(store,f,manifest,CONFIG,clusters=c),anchor)
    order=np.random.default_rng(5).permutation(16)
    changed=FrameStore(store.frame_ids,store.periods,store.xy_m[:,order],(1-store.team_codes if swap else store.team_codes)[order],PITCH)
    b=analyze_anchor_batch(TemporalSource(changed,f,manifest,CONFIG,clusters=c),anchor)
    pd.testing.assert_frame_equal(a,b,atol=1e-6,rtol=1e-6)


def test_matching_merge_split_is_not_forced():
    left=np.array([[49.,34],[51.,34]]);right=np.array([[50.,34]])
    eligible,pairs=reciprocal_unique(left,right,np.array([2,2]),np.array([4]),np.array([2.,2.]),np.array([4.]),CONFIG)
    assert eligible.all() and pairs==[]
    eligible,pairs=reciprocal_unique(right,left,np.array([4]),np.array([2,2]),np.array([4.]),np.array([2.,2.]),CONFIG)
    assert eligible.all() and pairs==[]
    eligible,pairs=reciprocal_unique(right,right,np.array([2]),np.array([2]),np.array([1.]),np.array([1.]),CONFIG)
    assert len(pairs)==1


def test_matching_drops_unknown_composition_and_ids_are_adjacent_only():
    store,f,c=scene(np.ones(101),missing=[(60,0)])
    manifest=SplitManifest((TimeBlock(1,'train',0,100),),(.7,.15,.15))
    source=TemporalSource(store,f,manifest,CONFIG,clusters=c);m=adjacent_matches(source,c,CONFIG)
    assert m.loc[c.frame.eq(59),'match_next_status'].eq('observation_ambiguous').all()
    good=m.match_next_status.eq('reciprocal_unique_geometry')
    np.testing.assert_equal(m.loc[good,'match_next_frame'].to_numpy(),c.loc[good,'frame'].to_numpy()+1)


def test_synthetic_csv_pipeline_and_all_frame_alignment(tmp_path):
    store,f,c=scene(np.ones(101));manifest=SplitManifest((TimeBlock(1,'train',0,100),),(.7,.15,.15))
    config=dict(CONFIG,batch_anchors=32)
    source=TemporalSource(store,f,manifest,config,cache_dir=tmp_path/'pairs',clusters=c)
    old=c.copy(deep=True);temporal=extract_temporal(source,c,tmp_path/'clusters.csv')
    result=frame_temporal_features(source,f,c,temporal)
    out=pd.read_csv(tmp_path/'clusters.csv',low_memory=False)
    assert len(out)==len(c) and not out.duplicated(['frame','radius_m','cluster_id']).any()
    assert len(result)==101 and np.array_equal(result.frame,np.arange(101))
    assert not np.isinf(out.select_dtypes(include='number')).any().any()
    pd.testing.assert_frame_equal(c,old)
    assert np.array_equal(result.v1_candidate_state,f.v1_candidate_state)


def test_invalid_config_and_existing_output_guard(tmp_path):
    validate_temporal_config(CONFIG)
    with pytest.raises(ValueError):validate_temporal_config(dict(CONFIG,window_radius_frames=0))
    output=tmp_path/'data/derived/competition_judgement/existing';output.mkdir(parents=True)
    with pytest.raises(FileExistsError):execute(root=tmp_path,output=output,cache='none',v2_dir='none',split_path='none',config_path='none')


def test_outside_period_keeps_current_key_but_no_temporal_evidence():
    store,f,c=scene(np.ones(11),periods=np.zeros(11,int))
    source=TemporalSource(store,f,SplitManifest((),(.7,.15,.15)),CONFIG,clusters=c)
    anchor=c.loc[c.frame.eq(5)&c.radius_m.eq(3)].reset_index(drop=True)
    t=analyze_anchor_batch(source,anchor)
    assert t.record_past_frames.iloc[0]==t.record_future_frames.iloc[0]==0
    assert not t.pattern_score_valid.iloc[0] and np.isnan(t.present_anchor_soft.iloc[0])
    assert np.isnan(t.local_count_change.iloc[0])
    assert not np.isinf(t.select_dtypes(include='number')).any().any()


def test_fixed_probe_known_no_hard_contact_is_not_missing_observation():
    store,f,c=scene(np.full(101,3.01))
    manifest=SplitManifest((TimeBlock(1,'train',0,100),),(.7,.15,.15))
    source=TemporalSource(store,f,manifest,CONFIG,clusters=c)
    probe=c.loc[c.frame.eq(50)&c.radius_m.eq(5)].copy().reset_index(drop=True)
    probe['radius_m']=3. # A fixed spatial probe can lose its 3m contact under noise.
    t=analyze_anchor_batch(source,probe)
    assert t.pattern_score_valid.iloc[0]
    assert t.close_duration_lower_bound_s.iloc[0]==0
    assert t.future_end_reason.iloc[0]=='current_relation_not_active'
