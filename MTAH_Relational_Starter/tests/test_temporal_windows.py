"""Synthetic specifications for the temporal stage. No match files/labels."""
from dataclasses import fields

import numpy as np
import pytest

from mtah_relational.frame_store import FrameStore
from mtah_relational.splits import make_split_manifest, assert_disjoint_support
from mtah_relational.windows import WindowSpec, build_window_index, WindowDataset, collate_inputs, collate_targets
from mtah_relational.corruption import SpanMasker
from mtah_relational.normalization import fit_train_normalizer, TrainOnlyNormalizer
from mtah_relational.data_quality import ObservationPolicy


@pytest.fixture
def temporal_source():
    ids = np.r_[np.arange(2000), np.arange(3000, 5000)]
    periods = np.r_[np.ones(2000), np.full(2000, 2)].astype(np.int8)
    base = np.array([[15, 15], [30, 30], [17, 15], [32, 30]], dtype=float)
    xy = np.broadcast_to(base, (4000, 4, 2)).copy()
    xy[:, :, 0] += np.sin(ids[:, None]/30)
    return FrameStore(ids, periods, xy, np.array([0, 0, 1, 1]), (100, 60))


@pytest.fixture
def temporal_setup(temporal_source):
    source = temporal_source
    manifest = make_split_manifest([(1, 0, 1999), (2, 3000, 4999)])
    policy = ObservationPolicy(min_team_players=2)
    normalizer = fit_train_normalizer(source, manifest, policy)
    return source, manifest, policy, normalizer


def test_151_center_and_original_ids(temporal_setup):
    source, manifest, policy, normalizer = temporal_setup
    index = build_window_index(source, manifest, split="train")
    ds = WindowDataset(source, index, normalizer, manifest, policy=policy)
    w = ds[0]
    assert len(ds) == 2500
    assert w.routing.center_frame == 75
    np.testing.assert_array_equal(w.routing.frame_ids, np.arange(151))
    assert w.routing.frame_ids[75] == w.routing.center_frame
    assert w.inputs.values.shape == (151, 24)
    assert w.inputs.team_structure.shape == (151, 2, 10)
    assert w.inputs.quality.shape == (151, 4)
    assert w.inputs.dense_features().shape == (151, 53)
    assert len(w.inputs.raw_xy_m) == 151
    assert not w.inputs.padding_mask.any()
    assert not hasattr(w, "targets")
    assert not hasattr(w.inputs, "targets")
    t = ds.get_targets(0)
    assert t.values.shape == t.observed.shape == t.loss_mask.shape == (151, 13)
    assert t.loss_mask.sum() == w.inputs.artificial_mask.sum()*13


def test_split_support_includes_inputs_and_targets_without_overlap(temporal_setup):
    source, manifest, _, _ = temporal_setup
    indices = [build_window_index(source, manifest, split=s) for s in ("train", "validation", "test")]
    result = assert_disjoint_support(indices, manifest)
    assert result["cross_split_overlap_frames"] == 0
    assert [len(x) for x in indices] == [2500, 300, 300]
    for index in indices:
        for rec in index.records:
            block = manifest.blocks[int(rec["block_index"])]
            assert block.first <= rec["support_first"] <= rec["support_last"] <= block.last
            assert rec["support_last"]-rec["support_first"] == 150
    # Counterfeit an index that has leaked one raw frame into the next split.
    bad = build_window_index(source, manifest, split="train")
    records = bad.records.copy()
    records[-1]["support_last"] += 1
    from dataclasses import replace
    with pytest.raises(ValueError):
        assert_disjoint_support([replace(bad, records=records), *indices[1:]], manifest)


def test_wrong_split_records_cannot_be_mislabeled_as_validation(temporal_setup):
    from dataclasses import replace
    source, manifest, policy, norm = temporal_setup
    train = build_window_index(source, manifest, split="train")
    validation = build_window_index(source, manifest, split="validation")
    with pytest.raises(ValueError):
        WindowDataset(source, replace(validation, records=train.records), norm, manifest, policy=policy)


def test_missing_record_splits_runs_but_missing_players_do_not(temporal_source):
    keep = temporal_source.frame_ids != 700
    xy = temporal_source.xy_m[keep].copy()
    row_missing_players = np.flatnonzero(temporal_source.frame_ids[keep] == 500)[0]
    xy[row_missing_players] = np.nan
    source = FrameStore(temporal_source.frame_ids[keep], temporal_source.periods[keep], xy,
                        temporal_source.team_codes, temporal_source.pitch)
    manifest = make_split_manifest([(1, 0, 1999), (2, 3000, 4999)])
    index = build_window_index(source, manifest, split="train")
    assert not ((index.records["support_first"] < 700) & (index.records["support_last"] > 700)).any()
    assert (index.records["center_frame"] == 500).any()
    assert not (index.records["center_frame"] == 700).any()


def test_inference_padding_never_borrows_another_half_or_run(temporal_setup):
    source, manifest, policy, norm = temporal_setup
    index = build_window_index(source, manifest, mode="inference")
    ds = WindowDataset(source, index, norm, manifest, policy=policy)
    assert len(ds) == 4000
    for i, center, padding in [(0, 0, 75), (1999, 1999, 75), (2000, 3000, 75), (3999, 4999, 75)]:
        w = ds[i]
        assert w.routing.center_frame == center
        assert w.routing.frame_ids[75] == center
        assert w.inputs.padding_mask.sum() == padding
        assert (w.routing.frame_ids[w.inputs.padding_mask] == -1).all()
        assert not w.inputs.artificial_mask.any()
        assert all(len(w.inputs.raw_xy_m[j]) == 0 for j in np.flatnonzero(w.inputs.padding_mask))
    with pytest.raises(ValueError, match="inference"):
        ds.get_targets(0)


def test_holdout_inference_context_is_split_restricted(temporal_setup):
    source, manifest, policy, norm = temporal_setup
    index = build_window_index(source, manifest, split="validation", mode="inference")
    ds = WindowDataset(source, index, norm, manifest, policy=policy)
    w = ds[0]
    assert w.routing.center_frame == 1400
    assert w.inputs.padding_mask[:75].all()
    assert w.routing.frame_ids[75:].min() == 1400


@pytest.mark.parametrize("seed", range(12))
def test_span_lengths_contiguous_reproducible_and_data_independent(seed):
    masker = SpanMasker(seed=seed)
    a = masker.sample(151, center_frame=999, period=1, split="train", epoch=2)
    b = masker.sample(151, center_frame=999, period=1, split="train", epoch=2)
    np.testing.assert_array_equal(a, b)
    locations = np.flatnonzero(a)
    assert 8 <= len(locations) <= 13
    assert (np.diff(locations) == 1).all()
    val = masker.sample(151, center_frame=999, period=1, split="validation", epoch=2)
    np.testing.assert_array_equal(val, masker.sample(151, center_frame=999, period=1, split="validation", epoch=500))


def test_raw_mask_precedes_reads_and_all_derived_inputs(temporal_setup):
    source, manifest, policy, norm = temporal_setup
    idx = build_window_index(source, manifest, split="train")
    ds = WindowDataset(source, idx, norm, manifest, policy=policy)
    original = ds[100]
    hidden_frames = original.routing.frame_ids[original.inputs.artificial_mask]
    changed = source.xy_m.copy()
    for frame in hidden_frames:
        row = source.position(int(frame))
        changed[row] = [[np.nan, np.nan], [999, -30], [999, -30], [np.inf, 0]]
    alternative = FrameStore(source.frame_ids, source.periods, changed, source.team_codes, source.pitch)
    other = WindowDataset(alternative, idx, norm, manifest, policy=policy)[100]
    for f in fields(original.inputs):
        a, b = getattr(original.inputs, f.name), getattr(other.inputs, f.name)
        if isinstance(a, tuple):
            for x, y in zip(a, b):
                np.testing.assert_array_equal(x, y)
        else:
            np.testing.assert_array_equal(a, b)
    target_original = ds.get_targets(100)
    target_other = WindowDataset(alternative, idx, norm, manifest, policy=policy).get_targets(100)
    assert target_original.loss_mask.any()
    assert not target_other.loss_mask.any()
    # A second audit disallows even fetching raw observations at hidden times.
    old_frame = source.frame
    def guarded(row):
        assert source.frame_ids[row] not in hidden_frames
        return old_frame(row)
    from unittest.mock import patch
    with patch.object(source, "frame", side_effect=guarded):
        ds[100]


def test_natural_missing_artificial_and_padding_are_different(temporal_setup):
    source, manifest, policy, norm = temporal_setup
    missing = source.xy_m.copy()
    missing[500] = np.nan
    changed = FrameStore(source.frame_ids, source.periods, missing, source.team_codes, source.pitch)
    idx = build_window_index(changed, manifest, split="train", mode="inference")
    ds = WindowDataset(changed, idx, norm, manifest, policy=policy)
    w = ds[500]
    assert not w.inputs.feature_valid[75].any()
    assert not w.inputs.artificial_mask[75] and not w.inputs.padding_mask[75]
    assert len(w.inputs.raw_xy_m[75]) == 0
    assert np.isfinite(w.inputs.values).all()


def test_batching_keeps_targets_outside_forward_interface(temporal_setup):
    source, manifest, policy, norm = temporal_setup
    ds = WindowDataset(source, build_window_index(source, manifest, split="train"), norm, manifest, policy=policy)
    inputs, routing = collate_inputs([ds[0], ds[1]])
    targets = collate_targets([ds.get_targets(0), ds.get_targets(1)])
    assert inputs.values.shape == (2, 151, 24)
    assert inputs.dense_features().shape == (2, 151, 53)
    assert inputs.team_structure.shape == (2, 151, 2, 10)
    assert targets.values.shape == (2, 151, 13)
    assert len(routing) == 2
    assert not hasattr(inputs, "targets")


def test_train_only_fit_ignores_all_holdout_coordinates(temporal_setup, tmp_path):
    source, manifest, policy, norm = temporal_setup
    xy = source.xy_m.copy()
    for row, frame in enumerate(source.frame_ids):
        if manifest.split_of(int(frame)) != "train":
            xy[row] = 100000
    changed = FrameStore(source.frame_ids, source.periods, xy, source.team_codes, source.pitch)
    other = fit_train_normalizer(changed, manifest, policy)
    np.testing.assert_array_equal(norm.median, other.median)
    np.testing.assert_array_equal(norm.scale, other.scale)
    np.testing.assert_array_equal(norm.counts, other.counts)
    assert norm.fit_frames == 2800
    path = tmp_path/"normalizer.json"
    norm.save(path)
    loaded = TrainOnlyNormalizer.load(path, manifest=manifest, policy=policy)
    np.testing.assert_array_equal(norm.scale, loaded.scale)
    bad_manifest = make_split_manifest([(1, 0, 1999), (2, 3000, 4999)], fractions=(.6, .2, .2))
    with pytest.raises(ValueError):
        TrainOnlyNormalizer.load(path, manifest=bad_manifest, policy=policy)


@pytest.mark.parametrize("length", [150, 0, 152])
def test_bad_centered_window_lengths(length):
    with pytest.raises(ValueError):
        WindowSpec(length=length)
