"""Synthetic-only same-frame diagnostic contracts."""
import numpy as np
import pytest
import torch

from mtah_relational.model_b import GraphEncoder
from mtah_relational.dynamic_graph import build_graph
from mtah_relational.same_frame_graph import (SameFrameReconstructor,graph_statistics,pack_graphs,node_pool96,
    same_frame_huber,fit_ridge,predict_ridge,MODES)


def example():
    return build_graph([[10,10],[18,10],[10,12],[18,12]],[0,0,1,1])


def test_same_frame_pool_matches_existing_graph_encoder():
    graph=GraphEncoder().eval();args=pack_graphs([example()])
    raw,teams=node_pool96(graph,*args);scene,ordered,_=graph(*args)
    torch.testing.assert_close(graph.readout(raw),scene,rtol=0,atol=0)
    torch.testing.assert_close(teams,ordered,rtol=0,atol=0)


@pytest.mark.parametrize('mode',MODES)
def test_same_frame_shapes_gradients_original_frozen_and_no_temporal(mode):
    ref=GraphEncoder().eval().requires_grad_(False)
    before={k:v.clone() for k,v in ref.state_dict().items()}
    model=SameFrameReconstructor(ref,mode)
    args=pack_graphs([example(),build_graph([],[])])
    stats=torch.tensor(np.stack([graph_statistics(example()),graph_statistics(build_graph([],[]))]))
    pred=model(*args,statistics=stats)
    assert pred.shape==(2,13) and torch.isfinite(pred).all()
    assert not any(isinstance(m,torch.nn.LSTM) for m in model.modules())
    assert not any('mask_token' in name for name,_ in model.named_parameters())
    loss=same_frame_huber(pred,torch.zeros_like(pred),torch.ones_like(pred,dtype=torch.bool),torch.ones_like(pred))
    loss.loss.backward()
    assert all(p.grad is not None and torch.isfinite(p.grad).all() for p in model.parameters() if p.requires_grad)
    for name,value in ref.state_dict().items():torch.testing.assert_close(value,before[name],rtol=0,atol=0)
    assert all(p.grad is None for p in ref.parameters())


def test_zero_initialized_added_columns_preserve_initial_function():
    graph=GraphEncoder();args=pack_graphs([example()]);stats=torch.tensor(graph_statistics(example()))[None]
    base=SameFrameReconstructor(graph,'same_frame64')
    for mode in ('count_readout','count_distance_readout'):
        model=SameFrameReconstructor(graph,mode)
        torch.testing.assert_close(base(*args),model(*args,statistics=stats),atol=1e-7,rtol=1e-6)


def test_statistics_are_symmetric_and_histogram_overlap_is_explicit():
    g=example();stats=graph_statistics(g)
    assert stats.shape==(15,) and np.isfinite(stats).all()
    np.testing.assert_allclose(stats[:2],[2/11,2/11])
    assert stats[5:10].sum()==pytest.approx(1.)
    # Two of four cross pairs at2m, the other two at sqrt(68).
    np.testing.assert_allclose(stats[5:10],[.5,0,.5,0,0])
    changed=build_graph(g.x*[105.3,68],1-g.team)
    np.testing.assert_allclose(graph_statistics(changed),stats,atol=1e-7)


def test_pooling_count_collision_changes_quantile_length_target():
    from mtah_relational.diagnose_b_same_frame import collision_audit
    result=collision_audit(GraphEncoder().eval(),'cpu')
    assert result['node_counts']==[4,8] and result['directed_edge_counts']==[16,64]
    assert result['max_pool96_difference']<2e-6 and result['max_readout64_difference']<2e-6
    assert result['length_sum_targets'][0]!=result['length_sum_targets'][1]


def test_ridge_fits_train_only_and_checkpoint_roundtrip(tmp_path):
    from mtah_relational.diagnose_b_same_frame import save_frame_model,load_frame_model
    x=np.random.default_rng(4).normal(size=(100,5));y=x[:,:2]*2+3
    fit=fit_ridge(x,y);np.testing.assert_allclose(predict_ridge(fit,x),y,atol=.0001)
    graph=GraphEncoder();model=SameFrameReconstructor(graph,'count_distance_readout').eval()
    args=pack_graphs([example()]);stats=torch.tensor(graph_statistics(example()))[None]
    expected=model(*args,statistics=stats)
    save_frame_model(tmp_path/'d.pt',model,epoch=1)
    restored,_=load_frame_model(tmp_path/'d.pt',graph);restored.eval()
    torch.testing.assert_close(expected,restored(*args,statistics=stats),rtol=0,atol=0)


def test_empty_target_loss_and_empty_stratum_are_safe():
    from mtah_relational.diagnose_b_same_frame import score
    p=torch.ones(3,13,requires_grad=True);y=torch.full_like(p,torch.nan)
    assert same_frame_huber(p,y,torch.zeros_like(p,dtype=torch.bool),torch.zeros_like(p)).loss is None
    r=score(np.zeros((3,13)),np.zeros((3,13)),np.zeros((3,13),bool),np.zeros((3,13)),np.ones(13))
    assert r['valid_cells']==0 and r['feature_macro_mae'] is None


def test_trainable_encoder_rejects_detached_frozen_feature_path():
    model=SameFrameReconstructor(GraphEncoder(),'same_frame64')
    with pytest.raises(ValueError):model(frozen_features=torch.zeros(2,64))


def test_synthetic_full_diagnostic_preserves_original_checkpoint_and_locks_test(synthetic_atd,tmp_path):
    import json
    from pathlib import Path
    from mtah_relational.data_quality import ObservationPolicy
    from mtah_relational.temporal_cli import build_temporal_data,open_temporal_dataset
    from mtah_relational.windows import WindowSpec
    from mtah_relational.model_b import GraphBiLSTM,save_b_checkpoint
    from mtah_relational.diagnose_b_same_frame import run,role_rows
    meta,raw=synthetic_atd(n_frames=2400)
    data=tmp_path/'temporal'
    build_temporal_data(meta_path=meta,raw_path=raw,output_dir=data,pitch=(105.3,68.),n_frames=2400,
        blocks=[(1,0,1199),(2,1200,2399)],expected_slots=4,
        policy=ObservationPolicy(min_team_players=2),spec=WindowSpec(stride=25))
    original=tmp_path/'b20.pt';save_b_checkpoint(original,GraphBiLSTM(),epoch=18)
    before=original.read_bytes()
    config=json.loads((Path(__file__).parents[1]/'configs/b_same_frame_diagnostic.json').read_text())
    config.update(epochs=1,train_raw_stride=50,validation_raw_stride=25,batch_size=16,device='cpu',cpu_threads=2)
    report=run(data,tmp_path/'cache',original,tmp_path/'diagnostic',config)
    assert original.read_bytes()==before
    assert report['original_b20_frozen'] and not report['test_used'] and not report['temporal_training']
    assert set(report['models'])==set(MODES)
    paired=__import__('json').loads((tmp_path/'diagnostic/paired_comparison.json').read_text())
    cells={m['strict_all']['metrics']['valid_cells'] for m in paired['models'].values()}
    assert len(cells)==1
    with pytest.raises(ValueError):role_rows(open_temporal_dataset(data,split='train'),'test')
