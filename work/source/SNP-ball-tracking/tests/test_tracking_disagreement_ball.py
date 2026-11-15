"""Synthetic ball-only observations; prove no contamination of player metrics."""
import json
from pathlib import Path
import tempfile
from unittest.mock import patch

import cv2
import numpy as np
import pytest

from starter import export_tracking_disagreement_ball as b
from starter import export_tracking_disagreement as players
from starter.view_all_movement_vectors import HTML


def compute(t,bc,halves=None,mapping=None,flips=None,window=50):
    t=np.asarray(t,float);bc=np.asarray(bc,float);n=len(t)
    return b.compute_ball(t,bc,np.arange(n) if mapping is None else np.asarray(mapping),
        np.ones(n,int) if halves is None else np.asarray(halves),np.zeros(n,int) if flips is None else np.asarray(flips),window=window)


def col(v,key):return v[:,b.COL[key]]


def test_four_states_and_missing_current_distances_not_zero():
    nan=[np.nan,np.nan];v=compute([[0,0],[1,1],nan,nan],[[3,4],nan,[2,2],nan])
    np.testing.assert_array_equal(col(v,'ball_state_code'),[0,1,2,3])
    np.testing.assert_array_equal(col(v,'t_ball_observed'),[1,1,0,0]);np.testing.assert_array_equal(col(v,'b_ball_observed'),[1,0,1,0])
    np.testing.assert_array_equal(col(v,'ball_presence_B_minus_T'),[0,-1,1,0])
    assert col(v,'ball_distance_m')[0]==5 and np.isnan(col(v,'ball_distance_m')[1:]).all()
    assert col(v,'ball_dx_B_minus_T_m')[0]==3 and col(v,'ball_dy_B_minus_T_m')[0]==4


def test_partial_or_infinite_ball_is_not_observed():
    v=compute([[1,np.nan],[1,np.inf],[0,0]],[[0,0],[0,0],[np.nan,2]])
    np.testing.assert_array_equal(col(v,'ball_state_code'),[2,2,1]);assert np.isnan(col(v,'ball_distance_m')).all()


def test_common_only_rmse_is_not_diluted_by_missing_frames():
    v=compute([[0,0],[np.nan,np.nan],[np.nan,np.nan]],[[3,4],[0,0],[0,0]])
    np.testing.assert_allclose(col(v,'ball_rmse_past2s_m'),[5,5,5])
    np.testing.assert_array_equal(col(v,'ball_common_past2s'),[1,1,1]);np.testing.assert_array_equal(col(v,'ball_record_support_past2s'),[1,2,3])
    # Rolling history may exist although CURRENT ball distance is undefined.
    assert np.isnan(col(v,'ball_distance_m')[2])


def test_sliding_window_drops_old_common_frame():
    t=np.full((52,2),np.nan);t[0]=0;bc=np.zeros((52,2));bc[0]=[3,4];v=compute(t,bc)
    assert col(v,'ball_rmse_past2s_m')[49]==5 and np.isnan(col(v,'ball_rmse_past2s_m')[50:]).all()
    assert col(v,'ball_record_support_past2s')[50]==50 and col(v,'ball_common_past2s')[50]==0


@pytest.mark.parametrize('halves,mapping,flips',[
    ([1,1,2,2],[0,1,2,3],[0,0,0,0]),
    ([1,1,1,1],[0,1,3,4],[0,0,0,0]),
    ([1,1,1,1],[0,1,2,3],[0,0,1,1]),
])
def test_half_broadcast_gap_and_reflection_boundaries_reset_history(halves,mapping,flips):
    t=np.full((4,2),np.nan);t[0]=0;bc=np.zeros((5,2));bc[0]=[3,4]
    v=compute(t,bc,halves,mapping,flips)
    assert col(v,'ball_record_support_past2s')[2]==1 and col(v,'ball_common_past2s')[2]==0
    assert np.isnan(col(v,'ball_rmse_past2s_m')[2])


def test_unmapped_is_unknown_broadcast_not_both_missing_and_breaks_history():
    t=np.zeros((4,2));t[3]=np.nan;bc=np.zeros((3,2));bc[0]=[3,4]
    v=compute(t,bc,mapping=[0,1,-1,2])
    assert col(v,'ball_state_code')[2]==4 and np.isnan(col(v,'b_ball_observed')[2])
    assert col(v,'t_ball_observed')[2]==1 and np.isnan(col(v,'ball_presence_B_minus_T')[2])
    assert col(v,'ball_record_support_past2s')[3]==1 and np.isnan(col(v,'ball_rmse_past2s_m')[3])


def test_ball_errors_have_no_player_5m_gate():
    v=compute([[0,0]],[[100,0]])
    assert col(v,'ball_distance_m')[0]==100 and col(v,'ball_rmse_past2s_m')[0]==100


def test_reflection_is_applied_in_metres():
    v=compute([[20,30]],[[85.3,30]],flips=[1])
    assert col(v,'ball_distance_m')[0]==pytest.approx(0,abs=1e-10)


def test_reader_ignores_player_values_and_preserves_ball_units():
    with tempfile.TemporaryDirectory() as folder:
        a=Path(folder)/'a.txt';z=Path(folder)/'z.txt'
        a.write_text('1:0,0,0:0.1,0.2,9\n2:NaN,NaN,0:NaN,0.3,0\n')
        z.write_text('1:999,999,0;1,2,3:0.1,0.2,9\n2:3,4,0:NaN,0.3,0\n')
        np.testing.assert_array_equal(b.read_ball(a,2,(105.3,68)),b.read_ball(z,2,(105.3,68)))
        np.testing.assert_allclose(b.read_ball(a,2,(105.3,68))[0],[10.53,13.6],rtol=1e-6)


def test_changing_ball_never_changes_any_player_metric_or_association():
    a=np.array([[10,10],[30,30],[80,40],[50,34]],float);z=a.copy();z[:3]+=[1,2]
    before=players.compare_frame(a,z,['H','H','A'],['H','H','A'])
    a[-1]=[np.nan,np.nan];z[-1]=[1000,1000]
    after=players.compare_frame(a,z,['H','H','A'],['H','H','A'])
    for key in before[0]:
        if key!='ball_distance_m':np.testing.assert_equal(before[0][key],after[0][key])
    assert before[1]==after[1] and before[2]==after[2] and before[3]==after[3]


def test_panel_is_separate_and_preserves_player_pixels_except_old_ball_caption():
    v=compute([[0,0],[np.nan,np.nan]],[[3,4],[0,0]])
    original=np.full((760,1280,3),70,np.uint8)
    with patch.object(cv2,'putText',wraps=cv2.putText) as mock:
        frame=b.append_ball_panel(original,v,1)
    mask=np.ones(original.shape[:2],bool);mask[459:484,882:1279]=False
    np.testing.assert_array_equal(original[mask],frame[:760][mask]);assert frame.shape==(1040,1280,3)
    strings=[call.args[1] for call in mock.call_args_list]
    assert any('Tactical: MISSING' in s for s in strings) and any('Broadcast: OBSERVED' in s for s in strings)
    assert any('Current distance N/A' in s for s in strings) and any('BALL RMSE 5.00' in s for s in strings)


def test_synthetic_video_and_offline_player_roundtrip():
    v=compute([[0,0],[np.nan,np.nan],[0,0],[np.nan,np.nan]],[[3,4],[0,0],[np.nan,np.nan],[np.nan,np.nan]],halves=[1,1,2,2])
    with tempfile.TemporaryDirectory() as folder:
        source=Path(folder)/'source';out=Path(folder)/'new';source.mkdir();out.mkdir()
        writer=cv2.VideoWriter(str(source/'tracking_disagreement.mp4'),cv2.VideoWriter_fourcc(*'avc1'),25,(1280,760))
        assert writer.isOpened()
        for _ in range(4):writer.write(np.zeros((760,1280,3),np.uint8))
        writer.release();(source/'tracking_disagreement_player.html').write_text(HTML.replace('__META__','{}').replace('full_movement_vectors.mp4','tracking_disagreement.mp4'))
        (source/'frame_timeline.csv').write_text('synthetic source timeline\n')
        result=b.export_video(source,out,v,np.arange(4),np.arange(4),np.array([1,1,2,2]))
        assert result['frames']==4 and result['size']==[1280,1040] and result['decoded_samples_verified']
        html=(out/'tracking_disagreement_ball_player.html').read_text()
        assert 'tracking_disagreement_with_ball.mp4' in html and '아래쪽은 공 전용' in html
        assert 'https://' not in html and 'fetch(' not in html and 'ball_metrics.csv.gz' in html
        assert (source/'frame_timeline.csv').read_bytes()==(out/'frame_timeline.csv').read_bytes()
