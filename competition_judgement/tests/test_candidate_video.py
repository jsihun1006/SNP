"""Synthetic rendering/navigation checks; no match scenes or human labels."""
import json
import cv2
import numpy as np
import pandas as pd
import pytest
from competition_judgement import view_candidate_video as viewer


def timeline(n=10):
    return pd.DataFrame(dict(frame=np.arange(n),decision_state=['REVIEW']*n,decision_reason=['WHOLE_PATTERN_UNCONFIRMED']*n,
        current_strict=[True]*n,current_record_valid=[True]*n,representative_radius_m=[np.nan]*n,
        representative_cluster_id=[np.nan]*n,representative_basis_mask=[0]*n,decision_future_frames=[50]*n,
        period=[1]*n,split=['validation']*n))


def test_navigation_ranges_are_not_individual_frames():
    f=timeline();f.loc[[1,2,3,7,8],'decision_state']='PATTERN_CANDIDATE'
    np.testing.assert_array_equal(viewer.navigation_starts(f,viewer.CANDIDATES),[1,7])


@pytest.mark.parametrize('column,value',[('period',2),('split','test'),('current_record_valid',False)])
def test_navigation_respects_record_half_split(column,value):
    f=timeline();f.decision_state='PATTERN_CANDIDATE';f.loc[5:,column]=value
    expected=[0] if column=='current_record_valid' else [0,5]
    np.testing.assert_array_equal(viewer.navigation_starts(f,viewer.CANDIDATES),expected)


def test_leading_context_next_does_not_repeat_same_candidate():
    starts=np.array([100,200]);assert viewer.jump_to(starts,0,1,50)==(50,100)
    assert viewer.jump_to(starts,50,1,50,100)==(150,200)
    assert viewer.jump_to(starts,150,-1,50,200)==(50,100)
    assert viewer.jump_to(starts,200,1,50) is None


def test_metre_coordinates_normalized_before_projection():
    h=np.diag([1000.,500.,1.]);p=viewer.project_anchor(h,[52.65,34.],6.,(105.3,68.))
    np.testing.assert_array_equal(p[-1],[500,250])
    assert p[:,0].max()==557 and p[:,1].max()==294


@pytest.mark.parametrize('kind',['singular','nan','horizon'])
def test_unsafe_homography_has_no_overlay(kind):
    h=np.eye(3)
    if kind=='singular':h[2]=0
    if kind=='nan':h[0,0]=np.nan
    if kind=='horizon':h[2]=[1,0,-.5]
    assert viewer.project_anchor(h,[52.65,34.],6.,(105.3,68.)) is None


def test_state_zero_unknown_not_relabelled_by_render():
    image=np.zeros((200,400,3),np.uint8);row=timeline().iloc[0];original=row.copy()
    result=viewer.draw(image,row,pd.DataFrame(),None,width=640)
    assert result.shape==(500,640,3)
    pd.testing.assert_series_equal(row,original)
    assert not image.any()


def test_preview_is_local_and_frame_count_verified(tmp_path,monkeypatch):
    monkeypatch.setattr(viewer,'ROOT',tmp_path)
    source=tmp_path/'synthetic.mp4';writer=cv2.VideoWriter(str(source),cv2.VideoWriter_fourcc(*'avc1'),25.,(400,200))
    assert writer.isOpened()
    for i in range(10):writer.write(np.full((200,400,3),i*10,np.uint8))
    writer.release();cap=cv2.VideoCapture(str(source));f=timeline()
    try:
        viewer.validate_video(cap,10)
        output=tmp_path/'data/derived/competition_judgement/preview.mp4'
        report=viewer.export_preview(cap,3,5,output,f,pd.DataFrame(),{},np.repeat(np.eye(3)[None],10,axis=0),np.ones(10,bool))
        assert report['frames']==5 and report['end_frame_inclusive']==7 and not report['human_labels']
        assert output.exists() and json.loads(output.with_suffix('.json').read_text())==report
        with pytest.raises(FileExistsError):viewer.export_preview(cap,3,5,output,f,pd.DataFrame(),{},None,None)
        with pytest.raises(ValueError):viewer.export_preview(cap,3,5,tmp_path/'external.mp4',f,pd.DataFrame(),{},None,None)
    finally:cap.release()


def test_basis_names_are_not_probabilities():
    assert viewer.basis_text(15)=='A+B+RF-short+RF-near'
    assert viewer.basis_text(0)=='none'
