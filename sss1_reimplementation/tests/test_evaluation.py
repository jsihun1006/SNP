import json
import numpy as np
import pytest
from ..evaluation.metrics import locsim, localization_metrics
from ..evaluation.official import evaluate


def example():
    gt=dict(images=[dict(id=1,width=100,height=100),dict(id=2,width=100,height=100)],
            categories=[dict(id=1,name="person")],
            annotations=[dict(id=i,image_id=i,category_id=1,bbox=[10,10,20,50],area=1000,
                              iscrowd=0,position_on_pitch=[float(i),0.,0.]) for i in [1,2]])
    results=[dict(image_id=i,category_id=1,bbox=[10,10,20,50],score=.9,
                  position_on_pitch=[float(i),0.,0.],world_valid=True) for i in [1,2]]
    return gt,results


def test_locsim_sign_and_official_parity():
    from sskit.coco import locsim as official
    d=np.array([0.,.1,1.,5.])
    assert np.allclose(locsim(d),official(d**2))
    assert np.isclose(locsim(1),.05) and locsim(0)==1 and np.all(np.diff(locsim(d))<0)


def test_world_errors_and_misses_denominator():
    gt,pred=example()
    pred[0]["position_on_pitch"]=[1.3,.4,0.]
    out=localization_metrics(gt,pred[:1])
    assert out["world_mae_m"]==pytest.approx(.5)
    assert out["world_rmse_m"]==pytest.approx(.5)
    assert out["localization_success_rate"]==.5
    pred[0].pop("position_on_pitch");pred[0]["world_valid"]=False
    out=localization_metrics(gt,pred[:1])
    assert out["world_mae_m"] is None and out["localization_success_rate"]==0


@pytest.mark.parametrize("empty",[False,True])
def test_official_evaluation_synthetic_fixture(tmp_path,empty):
    gt,pred=example()
    if empty: pred=[]
    (tmp_path/"gt.json").write_text(json.dumps(gt),encoding="utf-8")
    (tmp_path/"pred.json").write_text(json.dumps(pred),encoding="utf-8")
    out=evaluate(tmp_path/"gt.json",tmp_path/"pred.json")
    target=0. if empty else 1.
    assert out["detection_coco"]["AP_50_95"]==pytest.approx(target)
    assert out["sskit"]["tau1m"]["mAP_LocSim"]==pytest.approx(target)


def test_zero_annotation_id_is_not_lost_by_standard_coco(tmp_path):
    gt,pred=example()
    gt["annotations"][0]["id"]=0
    (tmp_path/"gt.json").write_text(json.dumps(gt),encoding="utf-8")
    (tmp_path/"pred.json").write_text(json.dumps(pred),encoding="utf-8")
    out=evaluate(tmp_path/"gt.json",tmp_path/"pred.json",include_locsim=False)
    assert out["detection_coco"]["AP_50_95"]==pytest.approx(1.)


def test_missing_ground_truth_refuses_scores(tmp_path):
    (tmp_path/"gt.json").write_text('{"images":[],"annotations":[]}',encoding="utf-8")
    (tmp_path/"pred.json").write_text('[]',encoding="utf-8")
    with pytest.raises(ValueError,match="Ground truth"): evaluate(tmp_path/"gt.json",tmp_path/"pred.json")
