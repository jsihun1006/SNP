"""Synthetic-only tests for independent Stage A diagnostics."""
import json
from pathlib import Path
import numpy as np
import pandas as pd
import pytest
from competition_judgement.controlled_temporal_masking import (FEATURES,FULL_ONLY,diagnostic_estimates,
    official_estimates,position_mask,continuity_mask,error_summary)
from competition_judgement.partial_window_diagnostics import run_statistics,policy_predicates
from competition_judgement.missingness_diagnostics import MaskContext,unpack_masks
from competition_judgement.missingness_experiments import draw_donors,paired_block_bootstrap
from competition_judgement.temporal_features import summarize_anchor_series

CONFIG=json.loads((Path(__file__).parents[1]/'configs/missingness_stage_a.json').read_text())


def synthetic(n=4):
    time=np.arange(-50,51)[None,:]
    d=1+np.abs(time)*.1+np.arange(n)[:,None]*.2
    return dict(soft=np.exp(-d/3),nearest_m=d,edges_3m=(d<=3).astype(float),edges_5m=(d<=5).astype(float))


@pytest.mark.parametrize('count',[1,5,10,20,30])
@pytest.mark.parametrize('center_deleted',[False,True])
def test_exact_count_and_center_separation(count,center_deleted):
    for location in ('far_past','near_past','center_band','near_future','far_future','uniform','random'):
        m=position_mask(3,count,location,center_deleted,2027)
        if m is None:
            assert count-int(center_deleted)>25 and location!='center_band'
        else:
            assert np.all(m.sum(1)==count)
            assert np.all(m[:,50]==center_deleted)


@pytest.mark.parametrize('side',['past','future'])
def test_continuity_control(side):
    for runs in ([10],[5,5],[2]*5,[1]*10):
        m=continuity_mask(2,runs,side);lengths,max_,count=run_statistics(m)
        assert np.all(m.sum(1)==10) and not m[:,50].any()
        assert np.all(count==len(runs)) and np.all(max_==max(runs))
        assert np.abs(np.flatnonzero(m[0]).mean()-50)==25.5


@pytest.mark.parametrize('radius',[3,5])
def test_official_adapter_matches_frozen_functions(radius):
    s=synthetic();valid=np.ones((4,101),bool);valid[1,5:20]=False;valid[2,48:50]=False
    original=dict(**s,observed=valid,reliable=valid,record_support=np.ones_like(valid),strict=np.ones_like(valid),
        global_count_same=np.ones_like(valid),anchor_n_home=np.ones_like(valid,int),anchor_n_away=np.ones_like(valid,int),
        soft_pair_mean=s['soft'],contact_size_3m=s['edges_3m']*2,contact_size_5m=s['edges_5m']*2)
    expected=summarize_anchor_series(original,radius,25)
    got=official_estimates(valid,np.full(4,radius),s)
    for f in FEATURES:np.testing.assert_allclose(got[f],expected[f],equal_nan=True)


def test_hidden_values_cannot_leak_to_masked_estimator():
    s=synthetic();valid=~position_mask(4,20,'random',False,8)
    changed={k:np.where(valid,v,123456.) for k,v in s.items()}
    # Edge values hidden by validity are never allowed to break/extend known runs.
    a=diagnostic_estimates(s,valid,np.full(4,3));b=diagnostic_estimates(changed,valid,np.full(4,3))
    for f in a:np.testing.assert_allclose(a[f],b[f],equal_nan=True)


def test_duration_bounds_and_center_loss():
    s=synthetic();s['edges_3m'][:]=1
    v=np.ones((4,101),bool);v[:,49]=False
    e=diagnostic_estimates(s,v,np.full(4,3))
    assert np.all(e['close_duration_lower_bound_s']<4.04)
    np.testing.assert_allclose(e['duration_upper_bound_s'],4.04)
    v[:,50]=False;e=diagnostic_estimates(s,v,np.full(4,3))
    assert all(np.isnan(e[f]).all() for f in FEATURES)
    s['edges_3m'][:]=0;v[:]=True;e=diagnostic_estimates(s,v,np.full(4,3))
    assert np.all(e['duration_upper_bound_s']==0)


def test_policy_side_support_not_whole_average():
    v=np.ones((1,101),bool);v[:,:10]=False
    p=policy_predicates(v,CONFIG)
    assert not p[(90,'none')][0] and p[(80,'none')][0]
    assert not p[(80,'max_gap_5')][0]
    v[:,50]=False;assert not any(x[0] for x in policy_predicates(v,CONFIG).values())


def test_mask_hex_leading_zero_and_empty_rle():
    mask=np.zeros((1,101),bool);mask[:,100]=True
    encoded=[v.tobytes().hex() for v in np.packbits(mask,axis=1,bitorder='little')]
    assert encoded[0].startswith('00')
    np.testing.assert_array_equal(unpack_masks(encoded),mask)
    length,maximum,count=run_statistics(np.zeros((2,101),bool))
    assert not len(length) and not maximum.any() and not count.any()


def test_strict_current_gate_cause_and_split_sentinel():
    frames=pd.DataFrame(dict(frame=np.arange(220),period=np.r_[np.ones(110,int),np.full(110,2)],
        strict_quality_valid=np.ones(220,bool),n_home=np.full(220,11),n_away=np.full(220,11)))
    manifest={'blocks':[dict(first=0,last=59,period=1,split='train'),dict(first=60,last=109,period=1,split='validation'),
                        dict(first=110,last=219,period=2,split='train')]}
    ctx=MaskContext(frames,manifest);idx,rec=ctx.support(np.array([58,110]))
    assert not rec[0,52] and idx[0,52]==58
    assert not rec[1,49] and idx[1,49]==110
    ctx.strict=ctx.strict.copy();ctx.strict[58]=False
    obs=rec.copy();rel=obs&ctx.strict[idx]&ctx.strict[np.array([58,110]),None]
    chunk=pd.DataFrame(dict(frame=[58,110]))
    for k,m in [('record',rec),('observed',obs),('reliable',rel)]:chunk[k+'_mask_hex']=[v.tobytes().hex() for v in np.packbits(m,axis=1,bitorder='little')]
    _,_,_,code,_=ctx.audit(chunk)
    assert ((code[0][rec[0]]&4)>0).all()
    assert (code[0][~rec[0]]==1).all()


def test_sign_deadband_nan_and_bias():
    r=error_summary(np.array([.001,.2,-.2]),np.array([-.001,-.1,np.nan]),'observed_approach',CONFIG)
    assert r['sign_strong_n']==1 and r['sign_flip_rate']==1 and r['nan_fraction']==1/3


def test_train_donors_exact_count_and_no_silent_reduction():
    from collections import Counter
    m=np.zeros(101,bool);m[:5]=True;key=np.packbits(m,bitorder='little').tobytes()
    donors={(r,'reliable'):Counter({key:3}) for r in (3,5)}
    drawn,available=draw_donors(donors,np.array([3,5]),'reliable',np.random.default_rng(1),5,True)
    assert available.all() and (drawn.sum(1)==5).all()
    _,available=draw_donors(donors,np.array([3,5]),'reliable',np.random.default_rng(1),10,True)
    assert not available.any()


def test_block_bootstrap_preserves_clustered_repeats():
    result=paired_block_bootstrap(np.array([1.,1.,3.,3.]),np.array([0,0,1,1]),400,8)
    assert result['blocks']==2 and result['n']==4 and result['mean']==2
    assert result['ci_low']>=1 and result['ci_high']<=3


def test_record_gap_is_not_bridged():
    ids=np.r_[np.arange(100),np.arange(200,300)]
    frames=pd.DataFrame(dict(frame=ids,period=np.ones(200,int),strict_quality_valid=np.ones(200,bool),
                             n_home=np.full(200,11),n_away=np.full(200,11)))
    ctx=MaskContext(frames,{'blocks':[dict(first=0,last=299,period=1,split='train')]})
    idx,valid=ctx.support(np.array([99,100]))
    assert not valid[0,51] and idx[0,51]==99
    assert not valid[1,49] and idx[1,49]==100


def test_same_present_different_history_and_long_gap_not_averaged_as_zero():
    s=synthetic(1);r=np.array([3]);valid=np.ones((1,101),bool)
    clean=diagnostic_estimates(s,valid,r)
    valid[:,:50]=False;result=diagnostic_estimates(s,valid,r)
    assert np.isnan(result['observed_approach'][0])
    assert result['observed_future_change'][0]==clean['observed_future_change'][0]
    for name in FULL_ONLY:assert np.isnan(official_estimates(valid,r,s)[name][0])


def test_sign_flip_needs_deadband_on_both_sides():
    r=error_summary(np.array([.001,.2,-.2]),np.array([-.001,-.001,.3]),'observed_approach',CONFIG)
    assert r['sign_strong_n']==2 and r['sign_flip_rate']==.5 and r['direction_loss_rate']==.5
