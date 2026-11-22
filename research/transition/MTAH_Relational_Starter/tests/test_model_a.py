"""Model/loss/cache tests use synthetic inputs only, never competition labels."""
from dataclasses import replace

import numpy as np
import pytest
import torch

from mtah_relational.model_a import SpatialBiLSTM, save_checkpoint, load_checkpoint
from mtah_relational.reconstruction import masked_huber, ReconstructionMetrics
from mtah_relational.reconstruction_baselines import reconstruct_baselines
from mtah_relational.data_quality import ObservationPolicy
from mtah_relational.frame_store import FrameStore
from mtah_relational.normalization import fit_train_normalizer
from mtah_relational.splits import make_split_manifest, assert_disjoint_support
from mtah_relational.windows import WindowDataset, build_window_index
from mtah_relational.model_a_data import build_feature_cache, FeatureWindowDataset


def test_requested_architecture_forward_backward_and_parameter_count():
    torch.manual_seed(7)
    model = SpatialBiLSTM()
    assert sum(p.numel() for p in model.parameters()) == 37965
    x = torch.randn(2, 151, 53)
    prediction, embedding = model(x)
    assert prediction.shape == (2, 151, 13)
    assert embedding.shape == (2, 151, 64)
    mask = torch.zeros(2, 151, dtype=torch.bool)
    mask[:, 50:60] = True
    target = torch.randn_like(prediction)
    out = masked_huber(prediction, target, torch.ones_like(target, dtype=torch.bool),
                       torch.ones_like(target), mask)
    out.loss.backward()
    assert all(p.grad is not None and torch.isfinite(p.grad).all() for p in model.parameters())
    assert out.valid_cells == 260


def test_huber_ignores_natural_missing_quality_and_nan_targets():
    p = torch.tensor([[[2.]*13]], requires_grad=True)
    target = torch.zeros_like(p)
    target[..., 0] = torch.nan
    observed = torch.ones_like(target, dtype=torch.bool)
    observed[..., 1] = False
    q = torch.ones_like(target)
    q[..., 2] = 0
    q[..., 3] = torch.nan
    result = masked_huber(p, target, observed, q, torch.ones(1, 1, dtype=torch.bool))
    assert result.valid_cells == 9
    assert result.loss.item() == pytest.approx(1.5)  # Huber(delta=1, residual=2)
    result.loss.backward()
    assert (p.grad[..., :4] == 0).all()
    assert torch.isfinite(p.grad).all()


def test_all_invalid_returns_no_update_loss():
    p = torch.full((2, 151, 13), torch.nan, requires_grad=True)
    result = masked_huber(p, p.detach(), torch.zeros_like(p, dtype=torch.bool),
                          torch.zeros_like(p), torch.ones(2, 151, dtype=torch.bool))
    assert result.loss is None and result.weight_sum == 0 and result.valid_cells == 0


def test_invalid_batch_skips_optimizer_without_changing_parameters():
    from mtah_relational.train_model_a import train_update
    from mtah_relational.windows import LossTargets
    model = SpatialBiLSTM()
    optimizer = torch.optim.Adam(model.parameters(), lr=.001)
    before = {k: v.clone() for k,v in model.state_dict().items()}
    shape = (2, 151, 13)
    target = LossTargets(np.full(shape, np.nan, dtype=np.float32), np.zeros(shape, dtype=bool),
                          np.zeros(shape, dtype=np.float32), np.zeros(shape, dtype=bool),
                          np.ones((2, 151), dtype=bool))
    out = train_update(model, optimizer, np.zeros((2,151,53), dtype=np.float32),
                        np.zeros((2,151), dtype=bool), target, torch.device('cpu'))
    assert out['skipped'] and not optimizer.state
    for key,value in model.state_dict().items():
        torch.testing.assert_close(before[key], value, rtol=0, atol=0)


def test_early_stopping_uses_validation_only_and_patience():
    from mtah_relational.train_model_a import EarlyStopping
    stopper = EarlyStopping(2, .01)
    assert stopper.step(1.) == (True, False)
    assert stopper.step(.98) == (True, False)
    assert stopper.step(.979) == (False, False)
    assert stopper.step(1.1) == (False, True)


def test_nonfinite_eligible_prediction_is_rejected():
    p = torch.full((1, 1, 13), torch.inf)
    with pytest.raises(FloatingPointError):
        masked_huber(p, torch.zeros_like(p), torch.ones_like(p, dtype=torch.bool),
                     torch.ones_like(p), torch.ones(1, 1, dtype=torch.bool))


def test_padding_is_not_recurrent_context_and_outputs_are_finite():
    torch.manual_seed(4)
    model = SpatialBiLSTM().eval()
    x = torch.randn(1, 151, 53)
    pad = torch.ones(1, 151, dtype=torch.bool)
    pad[:, 30:130] = False
    p, h = model(x, pad)
    p_valid, h_valid = model(x[:, 30:130])
    torch.testing.assert_close(p[:, 30:130], p_valid)
    torch.testing.assert_close(h[:, 30:130], h_valid)
    changed = x.masked_fill(pad[..., None], torch.nan)
    p2, h2 = model(changed, pad)
    torch.testing.assert_close(p, p2)
    torch.testing.assert_close(h, h2)
    assert not p[pad].any() and not h[pad].any()
    all_pad, all_embedding = model(x, torch.ones_like(pad))
    assert not all_pad.any() and not all_embedding.any()
    with pytest.raises(FloatingPointError):
        model(torch.full_like(x, torch.nan))


def test_checkpoint_reload_identical(tmp_path):
    torch.manual_seed(32)
    model = SpatialBiLSTM().eval()
    x = torch.randn(2, 151, 53)
    expected = model(x)
    save_checkpoint(tmp_path/"a.pt", model, epoch=1, seed=32)
    restored, metadata = load_checkpoint(tmp_path/"a.pt")
    restored.eval()
    actual = restored(x)
    assert metadata["seed"] == 32
    for left, right in zip(expected, actual):
        torch.testing.assert_close(left, right, rtol=0, atol=0)


@pytest.mark.parametrize("span", [8, 13, 25, 50])
def test_baselines_linear_truth_and_persistence(span):
    x = np.zeros((1, 151, 53), dtype=np.float32)
    truth = 2*np.arange(151)[:, None]+np.arange(13)[None]
    x[0, :, :13] = truth
    x[:, :, 24:37] = 1
    x[:, 60:60+span, :24] = 0
    x[:, 60:60+span, 24:48] = 0
    x[:, 60:60+span, 52] = 1
    result, fallback = reconstruct_baselines(x, np.zeros((1, 151), dtype=bool))
    np.testing.assert_allclose(result["interpolation"][0, 60:60+span], truth[60:60+span])
    np.testing.assert_allclose(result["persistence"][0, 60:60+span], np.broadcast_to(truth[59], (span, 13)))
    assert not fallback.any()


def test_no_anchor_baselines_use_frozen_train_median():
    x = np.zeros((1, 151, 53), dtype=np.float32)
    x[:, 50:60, 52] = 1
    result, fallback = reconstruct_baselines(x, np.zeros((1, 151), dtype=bool))
    assert fallback.all()
    assert not result["persistence"].any() and not result["interpolation"].any()


def test_numpy_metrics_match_weighted_torch_loss():
    rng = np.random.default_rng(2)
    p, y = rng.normal(size=(2, 151, 13)), rng.normal(size=(2, 151, 13))
    observed = rng.random(y.shape) > .2
    q = (rng.random(y.shape) > .2).astype(float)
    mask = np.zeros((2, 151), dtype=bool)
    mask[:, 50:63] = True
    loss = masked_huber(torch.tensor(p), torch.tensor(y), torch.tensor(observed), torch.tensor(q), torch.tensor(mask))
    metrics = ReconstructionMetrics()
    metrics.update(p, y, observed, q, mask)
    report = metrics.report([str(i) for i in range(13)], np.ones(13))
    assert report["huber"] == pytest.approx(loss.loss.item())


@pytest.fixture
def a_data(tmp_path):
    ids = np.arange(1000)
    periods = np.where(ids < 500, 1, 2)
    xy = np.broadcast_to([[10., 10.], [25., 25.], [12., 10.], [27., 25.]], (1000, 4, 2)).copy()
    source = FrameStore(ids, periods, xy, np.array([0, 0, 1, 1]), (100, 60))
    manifest = make_split_manifest([(1, 0, 499), (2, 500, 999)])
    policy = ObservationPolicy(min_team_players=2)
    normalizer = fit_train_normalizer(source, manifest, policy)
    base = WindowDataset(source, build_window_index(source, manifest, split="train"), normalizer, manifest, policy=policy)
    cache = build_feature_cache(base, tmp_path/"cache")
    return base, manifest, FeatureWindowDataset(base, cache)


def test_fast_adapter_exactly_matches_original_dataset_and_targets(a_data):
    base, _, fast = a_data
    for i in (0, 17, len(base)-1):
        old, new = base[i], fast[i]
        np.testing.assert_array_equal(old.inputs.dense_features(), new.dense)
        expected = base.get_targets(i, window=old)
        actual = fast.get_targets(i, window=new)
        for name in ("values", "observed", "quality_weight", "loss_mask", "artificial_mask"):
            np.testing.assert_array_equal(getattr(expected, name), getattr(actual, name))


def test_fast_adapter_hidden_sentinel_and_split_audit(a_data, tmp_path):
    base, manifest, fast = a_data
    original = fast[17]
    times = original.routing.frame_ids[original.artificial_mask]
    xy = base.source.xy_m.copy()
    xy[times] = 999
    changed = FrameStore(base.source.frame_ids, base.source.periods, xy, base.source.team_codes, base.source.pitch)
    alternative = WindowDataset(changed, base.index, base.normalizer, manifest, policy=base.policy)
    other = FeatureWindowDataset(alternative, build_feature_cache(alternative, tmp_path/"altered_cache"))
    np.testing.assert_array_equal(original.dense, other[17].dense)
    assert fast.get_targets(17, window=original).loss_mask.any()
    assert not other.get_targets(17, window=other[17]).loss_mask.any()
    indices = [build_window_index(base.source, manifest, split=s) for s in ("train", "validation", "test")]
    assert assert_disjoint_support(indices, manifest)["cross_split_overlap_frames"] == 0


@pytest.mark.parametrize("skip_training", [False, True])
def test_complete_training_evaluation_and_embedding_pipeline(synthetic_atd, tmp_path, monkeypatch, skip_training):
    """Exercise the actual runner, including an entirely skipped training epoch."""
    import json
    from pathlib import Path
    from mtah_relational import train_model_a as runner
    from mtah_relational.temporal_cli import build_temporal_data
    from mtah_relational.windows import WindowSpec
    meta, raw = synthetic_atd(n_frames=2400)
    temporal = tmp_path/"temporal"
    build_temporal_data(meta_path=meta, raw_path=raw, output_dir=temporal, pitch=(100, 60),
                        n_frames=2400, blocks=[(1, 0, 1199), (2, 1200, 2399)], expected_slots=4,
                        policy=ObservationPolicy(min_team_players=2), spec=WindowSpec(stride=50))
    config = json.loads((Path(__file__).parents[1]/"configs/model_a_small.json").read_text())
    config.update(epochs=1, batch_size=8, train_stride=1, evaluation_stride=1,
                  embedding_batch_size=128, device="cpu", cpu_threads=2)
    if skip_training:
        monkeypatch.setattr(runner, "train_update", lambda *a, **kw:
                            dict(skipped=True, loss=None, weight_sum=0., valid_cells=0, gradient_norm=None))
    output = tmp_path/"run"
    result = runner.run_training(temporal, tmp_path/"cache", output, config)
    assert result["epochs_completed"] == result["best_epoch"] == 1
    epoch = result["history"][0]
    assert (epoch["optimizer_updates"] == 0) == skip_training
    assert (epoch["train_huber"] is None) == skip_training
    assert np.isfinite(result["best_validation_huber"])
    assert result["support_audit"]["cross_split_overlap_frames"] == 0
    assert (output/"best.pt").is_file() and (output/"report.json").is_file()
    embedding = np.load(output/"embeddings.npy", allow_pickle=False)
    assert embedding.shape == (2400, 64) and np.isfinite(embedding).all()
    np.testing.assert_array_equal(np.load(output/"embedding_frame_ids.npy"), np.arange(2400))
    for split in ("validation", "test"):
        for mask in ("8-13", "25", "50"):
            metrics = result["evaluation"][split][mask]["metrics"]
            assert len({v["valid_cells"] for v in metrics.values()}) == 1
            assert all(np.isfinite(v["huber"]) and np.isfinite(v["mae_standardized"]) for v in metrics.values())
