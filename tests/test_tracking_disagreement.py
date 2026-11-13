"""Synthetic-only set associations, units, missingness and local video checks."""
import itertools
from pathlib import Path
import tempfile
from unittest.mock import patch

import numpy as np
import pytest

from starter import export_tracking_disagreement as m


@pytest.mark.parametrize('shape',[(2,3),(3,2),(3,3),(1,5),(0,3)])
def test_numpy_assignment_matches_brute_force(shape):
    cost=np.random.default_rng(7).random(shape)
    with patch.object(m,'FAST_ASSIGNMENT',None):rr,cc=m.assignment(cost)
    if not cost.size:assert len(rr)==0;return
    small=cost if shape[0]<=shape[1] else cost.T
    expected=min(sum(small[i,j] for i,j in enumerate(c)) for c in itertools.permutations(range(small.shape[1]),small.shape[0]))
    assert np.isclose(cost[rr,cc].sum(),expected)
    assert len(set(rr))==len(rr) and len(set(cc))==len(cc)


def points():return np.array([[10.,10.],[30,20],[80,40],[50,34]])


def test_known_3_4_5_offset_units_and_signed_bias():
    a=points();b=a+np.array([3,4]);r,p,_,_=m.compare_frame(a,b,['H','H','A'],['H','H','A'],gate=5)
    assert np.isclose(r['forced_rmse_m'],5) and np.isclose(r['gated_rmse_m'],5)
    assert r['gated_pairs']==3 and r['H_gated_dx_bias_m']==3 and r['H_gated_dy_bias_m']==4
    assert r['ball_distance_m']==5 and r['count_delta_B_minus_T']==0


def test_permutation_does_not_change_geometry_metrics():
    a=points();b=a[[1,0,2,3]]
    r,_,_,_=m.compare_frame(a,b,['H','H','A'],['H','H','A'])
    assert r['forced_rmse_m']==0 and r['gated_pairs']==3
    assert r['gated_t_coverage']==1 and r['gated_b_coverage']==1


def test_teams_are_not_cross_matched_and_outliers_stay_unpaired():
    a=points();b=a.copy();b[[0,2]]=b[[2,0]]
    r,_,_,_=m.compare_frame(a,b,['H','H','A'],['H','H','A'])
    assert r['gated_pairs']==1 and r['forced_rmse_m']>30
    assert r['gated_rmse_m']==0 and r['gated_t_coverage']==pytest.approx(1/3)


def test_gated_assignment_maximizes_count_not_just_filters_full_assignment():
    # Minimal squared-cost full assignment uses one forbidden edge (sqrt5>2).
    # Two allowed matches are nevertheless possible and must be preserved.
    sq=np.array([[0,3.],[3,5.]])
    rr,cc=m.gated_assignment(sq,2)
    assert len(rr)==2 and set(zip(rr,cc))=={(0,1),(1,0)}
    for gate in (0,-1,np.nan):
        with pytest.raises(ValueError):m.gated_assignment(sq,gate)


def test_empty_and_partial_coordinates_are_not_zero_error():
    a=points();b=np.full_like(a,np.nan);r,p,_,_=m.compare_frame(a,b,['H','H','A'],['H','H','A'])
    assert r['b_count']==0 and r['t_count']==3 and r['count_delta_B_minus_T']==-3
    assert np.isnan(r['forced_rmse_m']) and np.isnan(r['gated_rmse_m']) and not p
    b[0]=[10,np.nan];r,_,_,_=m.compare_frame(a,b,['H','H','A'],['H','H','A']);assert r['b_count']==0


def test_duplicate_and_outside_observations_are_counted_not_deleted():
    a=points();a[1]=a[0];a[2]=[-2,40];r,_,_,_=m.compare_frame(a,a,['H','H','A'],['H','H','A'])
    assert r['H_t_duplicates']==1 and r['t_outside_pitch']==1 and r['t_count']==3 and r['forced_rmse_m']==0


def test_full_axis_and_unmapped_broadcast_missing_not_zero():
    a=np.tile(points(),(3,1,1));b=a[:2].copy();mapping=np.array([-1,0,1]);halves=np.array([0,1,1]);flips=np.zeros(3,int)
    with tempfile.TemporaryDirectory() as folder:
        v,idx,_,_=m.compute(dict(tactical=a,broadcast=b),dict(tactical=['H','H','A'],broadcast=['H','H','A']),mapping,halves,flips,Path(folder),5)
        assert len(v)==3 and np.isnan(v[0,m.COL['b_count']]) and v[0,m.COL['t_count']]==3
        assert np.isnan(v[0,m.COL['gated_rmse_m']]) and (idx[0]==-1).all()
        np.testing.assert_array_equal(v[:,m.COL['frame']],[0,1,2])


def test_synthetic_video_html_and_halftime_boundary_roundtrip():
    a=np.tile(points(),(4,1,1));teams=dict(tactical=['H','H','A'],broadcast=['H','H','A']);mapping=np.arange(4)
    halves=np.array([1,1,2,2]);flips=np.zeros(4,int)
    with tempfile.TemporaryDirectory() as folder:
        folder=Path(folder);values,indices,_,_=m.compute(dict(tactical=a,broadcast=a),teams,mapping,halves,flips,folder,5)
        video=m.export_video(dict(tactical=a,broadcast=a),teams,mapping,halves,flips,values,indices,folder,5)
        assert video['frames']==4 and len(video['segments'])==2 and video['decoded_samples_verified']
        html=(folder/'tracking_disagreement_player.html').read_text()
        assert '__META__' not in html and 'https://' not in html and 'fetch(' not in html and 'type="range"' in html
        assert 'tracking_disagreement.mp4' in html and 'frame_metrics.csv.gz' in html


def test_summary_keeps_zero_observation_frames_and_distinguishes_distance_support():
    a=np.tile(points(),(4,1,1));b=a.copy();b[-1]=np.nan
    teams=dict(tactical=['H','H','A'],broadcast=['H','H','A']);mapping=np.arange(4);halves=np.array([1,1,2,2]);flips=np.zeros(4,int)
    with tempfile.TemporaryDirectory() as folder:
        folder=Path(folder);values,indices,forced,gated=m.compute(dict(tactical=a,broadcast=b),teams,mapping,halves,flips,folder,5)
        groups=m.summarize(values,forced,gated,teams,folder);r=groups['all']
        assert r['frames']==4 and r['frames_with_forced_pairs']==3 and r['frames_with_gated_pairs']==3
        assert r['broadcast_zero_observation_frames']==1 and r['frame_mean_t_coverage']==.75
        assert r['teams']['ALL']['mean_t_count']==3 and r['teams']['ALL']['mean_b_count']==2.25
        assert r['teams']['ALL']['gated_distance_m']['n']==9 and r['teams']['ALL']['gated_distance_m']['rmse']==0
