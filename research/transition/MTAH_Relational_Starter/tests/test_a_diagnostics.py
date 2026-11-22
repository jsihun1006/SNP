"""A diagnostic tests use synthetic features only."""
import numpy as np
import torch

from mtah_relational.model_a import SpatialBiLSTM
from mtah_relational.a_diagnostic_models import (TokenSpatialBiLSTM, SmallTemporal,
                                               MixedSpanMasker, covariance_diagnostics)


def test_token_initialization_preserves_original_function_and_parameter_difference():
    torch.manual_seed(4)
    original = SpatialBiLSTM().eval()
    model = TokenSpatialBiLSTM(original).eval()
    x = torch.randn(2, 151, 53); x[..., 52] = 0
    x[:, 50:63] = 0; x[:, 50:63, 52] = 1
    for a,b in zip(original(x), model(x)):
        torch.testing.assert_close(a,b,atol=2e-7,rtol=1e-6)
    assert sum(p.numel() for p in model.parameters()) == 38029
    prediction, _ = model(x)
    prediction[:, 50:63].square().mean().backward()
    assert torch.isfinite(model.mask_token.grad).all() and model.mask_token.grad.norm() > 0


def test_token_never_reads_hidden_spatial_features_or_quality():
    model = TokenSpatialBiLSTM().eval()
    x = torch.randn(1,151,53); x[...,52] = 0
    x[:,50:60] = 0; x[:,50:60,52] = 1
    altered = x.clone(); altered[:,50:60,:52] = 300
    for a,b in zip(model(x),model(altered)):
        torch.testing.assert_close(a,b,rtol=0,atol=0)


def test_forward_and_reverse_states_use_opposite_temporal_context():
    torch.manual_seed(12)
    model = SpatialBiLSTM().eval()
    x = torch.randn(1,151,53,requires_grad=True)
    _,h = model(x)
    forward, = torch.autograd.grad(h[0,75,:32].sum(),x,retain_graph=True)
    reverse, = torch.autograd.grad(h[0,75,32:].sum(),x)
    assert forward[:,:75].norm() > 0 and not forward[:,76:].any()
    assert reverse[:,76:].norm() > 0 and not reverse[:,:75].any()


def test_mixed_span_masker_is_routing_only_and_holdout_fixed():
    masker = MixedSpanMasker()
    sizes=set()
    for frame in range(100):
        mask=masker.sample(151,center_frame=frame,period=1,split='train',epoch=12)
        sizes.add(int(mask.sum()))
        assert mask.sum() in (*range(8,14),25,50)
        assert (np.diff(np.flatnonzero(mask))==1).all()
    assert {25,50}.issubset(sizes) and any(8<=n<=13 for n in sizes)
    a=masker.sample(151,center_frame=15,period=2,split='validation',epoch=0)
    b=masker.sample(151,center_frame=15,period=2,split='validation',epoch=99)
    np.testing.assert_array_equal(a,b)


def test_small_temporal_shapes_and_gradients():
    model=SmallTemporal()
    prediction,h=model(torch.randn(2,151,53))
    assert prediction.shape==(2,151,13) and h.shape==(2,151,64)
    prediction.square().mean().backward()
    assert all(p.grad is not None and torch.isfinite(p.grad).all() for p in model.parameters())


def test_covariance_diagnostics_rank_one_and_constant():
    x=np.arange(100)[:,None]*np.ones((1,64))
    result=covariance_diagnostics(x)
    assert abs(result['effective_rank']-1)<1e-6
    assert result['dimensions_for_90pct']==1
    assert result['explained_variance_ratio'][0]>.99999
    assert covariance_diagnostics(np.zeros((100,64)))['effective_rank']==0


def test_experimental_checkpoint_roundtrip(tmp_path):
    from mtah_relational.diagnose_a_training import save_variant,restore_variant
    x=torch.randn(1,151,53);x[...,52]=0;x[:,40:65]=0;x[:,40:65,52]=1
    for kind,model in (("token_mixed",TokenSpatialBiLSTM()),("small_temporal_mixed",SmallTemporal())):
        model.eval();expected=model(x)
        path=tmp_path/f'{kind}.pt';save_variant(path,model,kind,epoch=1)
        restored,_=restore_variant(path);restored.eval()
        for a,b in zip(expected,restored(x)):
            torch.testing.assert_close(a,b,rtol=0,atol=0)


def test_adoption_rule_uses_only_fixed_validation_objectives():
    from mtah_relational.diagnose_a_training import choose_candidate
    def result(values):
        return {mask:{'metrics':{'bilstm':{'huber':value}}} for mask,value in zip(('8-13','25','50'),values)}
    assert choose_candidate({'original':result([1,2,3]),'candidate':result([1,1,2])})['accepted']
    assert not choose_candidate({'original':result([1,2,3]),'candidate':result([1.1,.5,.5])})['accepted']


def test_representation_statistics_keep_zero_variance_finite():
    from mtah_relational.diagnose_a_representation import summary,correlation,spearman,NoArtificialMask
    assert summary([])['n']==0 and summary([])['mean'] is None
    assert correlation([1,1,1],[2,3,4]) is None
    assert spearman([1,2,3],[10,20,30])==1
    assert not NoArtificialMask().sample(151).any()
