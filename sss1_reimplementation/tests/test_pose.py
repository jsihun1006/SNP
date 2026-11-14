from pathlib import Path
import runpy
import numpy as np
import pytest


def test_two_point_config_and_no_halfbody():
    config=runpy.run_path(str(Path(__file__).resolve().parents[1]/"configs/rtmpose_x_two_points.py"))
    assert config["model"]["head"]["out_channels"]==2
    assert not any(t["type"]=="RandomHalfBody" for t in config["train_pipeline"])
    assert config["train_dataloader"]["num_workers"]==0
    assert config["train_cfg"]["max_epochs"]==700
    assert config["optim_wrapper"]["optimizer"]["lr"]==1e-4


@pytest.mark.model
def test_real_rtmcc_head_two_by_two_attention_and_gradients(monkeypatch):
    import torch
    from ..pose.head import TwoPointRTMCCHead
    torch.manual_seed(21)
    head=TwoPointRTMCCHead(in_channels=16,out_channels=2,input_size=(288,384),
                          in_featuremap_size=(9,12),final_layer_kernel_size=7,
                          loss=dict(type="KLDiscretLoss",use_target_weight=True,beta=10.,label_softmax=True),
                          gau_cfg=dict(hidden_dims=32,s=16,expansion_factor=2,dropout_rate=0.,
                                       drop_path=0.,act_fn="SiLU",use_rel_bias=False,pos_enc=False))
    observed=[]
    original=torch.bmm
    def bmm(a,b):
        out=original(a,b)
        observed.append(tuple(out.shape))
        return out
    monkeypatch.setattr(torch,"bmm",bmm)
    features=torch.randn(2,16,12,9,requires_grad=True)
    x,y=head((features,))
    assert x.shape==(2,2,576) and y.shape==(2,2,768)
    assert (2,2,2) in observed
    from mmpose.codecs import SimCCLabel
    encoder=SimCCLabel(input_size=(288,384),sigma=(6.,6.93),simcc_split_ratio=2.,normalize=False)
    target=encoder.encode(np.array([[[100.,150.],[110.,300.]]]*2),np.ones((2,2)))
    loss=head.loss_module((x,y),
                          (torch.tensor(target["keypoint_x_labels"]),torch.tensor(target["keypoint_y_labels"])),
                          torch.tensor(target["keypoint_weights"]))
    assert torch.isfinite(loss) and loss>0
    loss.backward()
    assert torch.isfinite(features.grad).all() and features.grad.abs().sum()>0


@pytest.mark.model
def test_real_simcc_encoding_two_targets():
    from mmpose.codecs import SimCCLabel
    encoder=SimCCLabel(input_size=(288,384),sigma=(6.,6.93),simcc_split_ratio=2.)
    target=encoder.encode(np.array([[[100.,150.],[110.,300.]]]),np.ones((1,2)))
    assert target["keypoint_x_labels"].shape==(1,2,576)
    assert target["keypoint_y_labels"].shape==(1,2,768)


@pytest.mark.model
def test_reject_17_point_checkpoint(tmp_path):
    import torch
    from ..pose.inference import MMPoseEstimator
    path=tmp_path/"seventeen.pth"
    torch.save({"state_dict":{"head.final_layer.weight":torch.zeros(17,1,1,1)}},path)
    with pytest.raises(ValueError,match="2-keypoint"):
        MMPoseEstimator("unused.py",path)


def test_gpu_training_guard_on_cpu():
    import torch
    from ..scripts.probe_environment import require_cuda
    if not torch.cuda.is_available():
        with pytest.raises(RuntimeError,match="blocked"): require_cuda()
