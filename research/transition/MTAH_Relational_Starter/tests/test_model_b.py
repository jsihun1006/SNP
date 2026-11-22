"""Synthetic B graph, invariance, masking, gradients and checkpoint contracts."""
from dataclasses import replace

import numpy as np
import pytest
import torch

from mtah_relational.dynamic_graph import build_graph
from mtah_relational.graph_windows import GraphWindow,pack_windows,VisibleGraphCache,GraphWindowDataset
from mtah_relational.model_b import GraphBiLSTM,GraphEncoder,DenseEdgeGAT,dense_edge_attributes,save_b_checkpoint,load_b_checkpoint


def windows(graph,length=151):
    hidden=np.zeros(length,dtype=bool);hidden[65:76]=True
    pad=np.zeros(length,dtype=bool)
    return [GraphWindow(tuple(None if a else graph for a in hidden),hidden,pad,None)]


def graph():
    return build_graph([[10,10],[20,15],[30,20],[12,10],[25,17],[70,40]],[0,0,0,1,1,1])


def test_graph_shapes_reverse_edges_distances_and_explicit_self_loops():
    g=build_graph([[10,10],[13,14],[40,30]],[0,1,1])
    assert g.x.shape==(3,2) and g.edge_index.shape[0]==2 and g.edge_attr.shape[1]==8
    assert np.array_equal(g.adjacency,g.adjacency.T)
    assert np.trace(g.adjacency)==3 and len(set(map(tuple,g.edge_index.T)))==g.edge_index.shape[1]
    src,dst=g.edge_index;a=g.edge_attr[(src==0)&(dst==1)][0]
    np.testing.assert_allclose(a,[3/105.3,4/68,.5,0,1,0,1,1],atol=1e-6)
    b=g.edge_attr[(src==1)&(dst==0)][0]
    np.testing.assert_allclose(a[:2],-b[:2]);np.testing.assert_allclose(a[2:],b[2:])
    np.testing.assert_array_equal(g.edge_attr[src==dst],np.tile([0,0,0,1,0,0,0,0],(3,1)))


def test_knn_ties_include_all_and_distant_opponent_fallback():
    xy=np.array([[30,30],[29,30],[31,30],[30,29],[30,31],[90,60]])
    team=np.array([0,0,0,0,0,1])
    g=build_graph(xy,team,k=1)
    assert g.adjacency[0,1:5].all() and g.adjacency[0,5]
    for seed in range(5):
        p=np.random.default_rng(seed).permutation(len(xy))
        changed=build_graph(xy[p],team[p],k=1)
        np.testing.assert_array_equal(changed.adjacency,g.adjacency[np.ix_(p,p)])


def test_graph_cross_radius_is_not_zone_limited():
    g=build_graph([[13.15,10],[13.17,10],[50,50]],[0,1,1])
    assert g.adjacency[0,1] and g.edge_attr.shape[-1]==8


@pytest.mark.parametrize('n',[0,1,2,9,22])
def test_zero_singleton_variable_and_nonfinite_players(n):
    xy=np.random.default_rng(n).uniform([0,0],[100,60],(n,2))
    team=np.arange(n)%2
    g=build_graph(xy,team)
    model=GraphBiLSTM().eval()
    out=model(pack_windows(windows(g)))
    assert out.prediction.shape==(1,151,13) and out.embedding.shape==(1,151,64)
    assert out.team_embedding.shape==(1,151,2,32)
    assert torch.isfinite(out.prediction).all() and torch.isfinite(out.embedding).all()
    assert not out.team_observed[:,65:76].any() and not out.team_embedding[:,65:76].any()
    if n==0:assert not out.team_observed.any()
    filtered=build_graph([[np.nan,1],[np.inf,0],[0,0]],[0,1,0]);assert len(filtered.x)==1


def test_permutation_and_team_swap_scene_and_team_readouts():
    g=graph();model=GraphBiLSTM().eval();expected=model(pack_windows(windows(g)))
    p=np.array([5,2,0,4,1,3]);changed=build_graph(g.x[p]*[105.3,68],g.team[p])
    swapped=build_graph(g.x*[105.3,68],1-g.team)
    for out in (model(pack_windows(windows(changed))),model(pack_windows(windows(swapped)))):
        torch.testing.assert_close(expected.graph_embedding,out.graph_embedding,atol=3e-6,rtol=2e-5)
        torch.testing.assert_close(expected.prediction,out.prediction,atol=3e-6,rtol=2e-5)
    other=model(pack_windows(windows(swapped)))
    torch.testing.assert_close(expected.team_embedding,other.team_embedding.flip(2),atol=3e-6,rtol=2e-5)


def test_all_gat_and_mask_token_gradients_with_checkpointed_graph():
    model=GraphBiLSTM(graph_chunk=1,checkpoint_graph=True)
    out=model(pack_windows(windows(graph())))
    out.prediction[:,65:76].square().mean().backward()
    assert sum(p.numel() for p in model.parameters())==43853
    for name,p in model.named_parameters():
        assert p.grad is not None and torch.isfinite(p.grad).all(),name
    assert model.mask_token.grad.norm()>0 and model.graph.gat1.lin.weight.grad.norm()>0


def test_checkpoint_restore_and_padding_no_recurrent_crossing(tmp_path):
    model=GraphBiLSTM().eval();w=windows(graph())[0]
    pad=np.zeros(151,dtype=bool);pad[:25]=True
    gs=tuple(None if pad[i] else g for i,g in enumerate(w.graphs))
    w=replace(w,graphs=gs,padding_mask=pad)
    batch=pack_windows([w]);expected=model(batch)
    assert not expected.prediction[:,:25].any()
    save_b_checkpoint(tmp_path/'b.pt',model,seed=2027)
    restored,metadata=load_b_checkpoint(tmp_path/'b.pt');restored.eval()
    assert metadata['seed']==2027
    actual=restored(batch)
    for name in ('prediction','embedding','graph_embedding','team_embedding'):
        torch.testing.assert_close(getattr(expected,name),getattr(actual,name),atol=0,rtol=0)


def test_hidden_graph_rejected_by_collator():
    w=windows(graph())[0]
    with pytest.raises(AssertionError):pack_windows([replace(w,graphs=(graph(),)*151)])


def test_dense_attention_matches_pyg_forward_and_parameter_gradients():
    pyg=pytest.importorskip('torch_geometric.nn')
    torch.manual_seed(18)
    g=graph();layer=DenseEdgeGAT(2)
    reference=pyg.GATConv(2,32,heads=4,concat=False,edge_dim=8,add_self_loops=False,dropout=0,residual=False)
    with torch.no_grad():
        reference.lin.weight.copy_(layer.lin.weight);reference.lin_edge.weight.copy_(layer.lin_edge.weight)
        for name in ('att_src','att_dst','att_edge'):getattr(reference,name).copy_(getattr(layer,name)[None])
        reference.bias.copy_(layer.bias)
    x=torch.tensor(g.x);edge_index=torch.tensor(g.edge_index);attr=torch.tensor(g.edge_attr)
    actual=layer(x[None],torch.tensor(g.adjacency)[None],dense_edge_attributes(x[None],torch.tensor(g.team)[None]),torch.ones(1,len(x),dtype=torch.bool))[0]
    expected=reference(x,edge_index,attr)
    torch.testing.assert_close(actual,expected,atol=3e-6,rtol=2e-5)
    actual.square().mean().backward();expected.square().mean().backward()
    torch.testing.assert_close(layer.lin.weight.grad,reference.lin.weight.grad,atol=1e-6,rtol=5e-5)
    torch.testing.assert_close(layer.lin_edge.weight.grad,reference.lin_edge.weight.grad,atol=1e-6,rtol=5e-5)


def test_repeated_frame_encoding_reuse_preserves_output_and_gradient():
    g=graph();w=windows(g)[0]
    copies=tuple(None if a else replace(g,x=g.x.copy(),team=g.team.copy(),adjacency=g.adjacency.copy()) for a in w.artificial_mask)
    a=GraphBiLSTM(checkpoint_graph=False);b=GraphBiLSTM(checkpoint_graph=False);b.load_state_dict(a.state_dict())
    out_a=a(pack_windows([w]));out_b=b(pack_windows([replace(w,graphs=copies)]))
    torch.testing.assert_close(out_a.prediction,out_b.prediction,atol=2e-6,rtol=2e-5)
    out_a.prediction[:,65:76].square().mean().backward();out_b.prediction[:,65:76].square().mean().backward()
    torch.testing.assert_close(a.graph.gat1.lin.weight.grad,b.graph.gat1.lin.weight.grad,atol=2e-6,rtol=1e-4)


def test_raw_hidden_read_guard_and_targets_match_existing_dataset(tmp_path):
    from mtah_relational.frame_store import FrameStore
    from mtah_relational.data_quality import ObservationPolicy
    from mtah_relational.splits import make_split_manifest
    from mtah_relational.normalization import fit_train_normalizer
    from mtah_relational.windows import WindowDataset,build_window_index
    from mtah_relational.model_a_data import build_feature_cache
    from mtah_relational.temporal_cli import _CounterfactualSource
    ids=np.arange(1000);period=np.where(ids<500,1,2)
    xy=np.broadcast_to([[10.,10.],[25.,20.],[12.,10.],[27.,20.]],(1000,4,2)).copy()
    source=FrameStore(ids,period,xy,[0,0,1,1],(105.3,68.))
    manifest=make_split_manifest([(1,0,499),(2,500,999)])
    policy=ObservationPolicy(min_team_players=2)
    normalizer=fit_train_normalizer(source,manifest,policy)
    index=build_window_index(source,manifest,split='train')
    base=WindowDataset(source,index,normalizer,manifest,policy=policy)
    cache=build_feature_cache(base,tmp_path/'targets')
    dataset=GraphWindowDataset(base,VisibleGraphCache(base),cache)
    w=dataset[10];original=base[10]
    a=dataset.get_targets(10,window=w);b=base.get_targets(10,window=original)
    for name in ('values','observed','quality_weight','loss_mask','artificial_mask'):
        np.testing.assert_array_equal(getattr(a,name),getattr(b,name))
    hidden=w.routing.frame_ids[w.artificial_mask]
    guarded=_CounterfactualSource(source,hidden,forbid_reads=True)
    alt=WindowDataset(guarded,index,normalizer,manifest,policy=policy)
    other=GraphWindowDataset(alt,VisibleGraphCache(alt),cache)[10]
    aa,bb=pack_windows([w]),pack_windows([other])
    for name in aa.__dataclass_fields__:torch.testing.assert_close(getattr(aa,name),getattr(bb,name),atol=0,rtol=0)
    assert all(w.graphs[t] is None for t in np.flatnonzero(w.artificial_mask))
    from mtah_relational.graph_windows import grouped_batches
    positions=np.arange(len(base))
    batches=grouped_batches(index,positions)
    np.testing.assert_array_equal(np.sort(np.concatenate(batches)),positions)


def test_synthetic_training_full_pipeline_preserves_reference_models(synthetic_atd,tmp_path):
    import json
    from pathlib import Path
    from mtah_relational.temporal_cli import build_temporal_data
    from mtah_relational.data_quality import ObservationPolicy
    from mtah_relational.windows import WindowSpec
    from mtah_relational.model_a import SpatialBiLSTM,save_checkpoint
    from mtah_relational.diagnose_a_training import save_variant
    from mtah_relational.train_model_b import run
    meta,raw=synthetic_atd(n_frames=2400)
    temporal=tmp_path/'temporal'
    build_temporal_data(meta_path=meta,raw_path=raw,output_dir=temporal,pitch=(105.3,68.),n_frames=2400,
                        blocks=[(1,0,1199),(2,1200,2399)],expected_slots=4,
                        policy=ObservationPolicy(min_team_players=2),spec=WindowSpec(stride=50))
    config=json.loads((Path(__file__).parents[1]/'configs/model_b_small.json').read_text())
    config.update(epochs=1,budget_checkpoints=[1],device='cpu',train_stride=1,validation_stride=1,cpu_threads=2)
    reference=SpatialBiLSTM();a12=tmp_path/'a12.pt';a20=tmp_path/'a20.pt'
    save_checkpoint(a12,reference,epoch=12);save_variant(a20,reference,'continued_short',epoch=8)
    before=(a12.read_bytes(),a20.read_bytes())
    report=run(temporal,tmp_path/'targets',tmp_path/'run',config,{},a12_path=a12,a20_path=a20)
    assert report['parameters']==43853 and not report['test_evaluated']
    assert report['history'][0]['optimizer_updates']>0
    assert before==(a12.read_bytes(),a20.read_bytes())
    for models in report['comparison'].values():assert len({v['valid_cells'] for v in models.values()})==1


def test_synthetic_tracking_interventions_are_repeatable_and_read_only():
    from mtah_relational.data_quality import PlayerFrame
    from mtah_relational.graph_robustness import perturb_frame,delta_stats
    xy=np.array([[10.,10.],[20.,20.],[30.,30.],[12.,10.],[22.,20.],[32.,30.]])
    teams=np.array([0,0,0,1,1,1]);flags=np.zeros(6,dtype=bool)
    frame=PlayerFrame(123,1,xy,teams,flags,flags,flags)
    before=xy.copy()
    for kind,level in (('noise',.5),('dropout',.3),('permutation',0.),('team_swap',0.),('composition',.2),('physical_shift',1.)):
        a=perturb_frame(frame,kind,level,2027);b=perturb_frame(frame,kind,level,2027)
        np.testing.assert_array_equal(a.xy_m,b.xy_m);np.testing.assert_array_equal(xy,before)
        assert np.isfinite(build_graph(a.xy_m,a.team_codes).x).all()
    a=perturb_frame(frame,'noise',.1,2027);b=perturb_frame(frame,'noise',1.,2027)
    np.testing.assert_allclose((a.xy_m-xy)*10,b.xy_m-xy,atol=1e-12)
    a=perturb_frame(frame,'dropout',.1,2027);b=perturb_frame(frame,'dropout',.3,2027)
    assert (np.isnan(a.xy_m).all(1)<=np.isnan(b.xy_m).all(1)).all()
    assert perturb_frame(frame,'composition',.2,2027,active=False) is frame
    zeros=delta_stats(np.zeros((3,64)),np.zeros((3,64)))
    assert zeros['mean_l2']==0 and np.isfinite(zeros['mean_l2_per_reference_std'])


def test_raw_team_swap_and_permutation_pass_encoder_symmetry():
    from mtah_relational.data_quality import PlayerFrame
    from mtah_relational.graph_robustness import perturb_frame
    xy=np.array([[10.,10.],[20.,20.],[30.,30.],[12.,10.],[22.,20.],[32.,30.]])
    flags=np.zeros(6,dtype=bool);frame=PlayerFrame(123,1,xy,np.array([0,0,0,1,1,1]),flags,flags,flags)
    model=GraphBiLSTM().eval();clean=model(pack_windows(windows(build_graph(xy,frame.team_codes))))
    for kind in ('permutation','team_swap'):
        changed=perturb_frame(frame,kind,0.,9127)
        out=model(pack_windows(windows(build_graph(changed.xy_m,changed.team_codes))))
        torch.testing.assert_close(out.graph_embedding,clean.graph_embedding,atol=3e-6,rtol=1e-5)
        if kind=='team_swap':torch.testing.assert_close(out.team_embedding,clean.team_embedding.flip(2),atol=3e-6,rtol=1e-5)
