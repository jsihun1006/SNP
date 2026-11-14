import json

import numpy as np
import pytest

from mtah_relational.data_quality import ObservationPolicy
from mtah_relational.frame_store import FrameStore, build_frame_cache, open_frame_cache
from mtah_relational.temporal_cli import build_temporal_data, open_temporal_dataset
from mtah_relational.windows import WindowSpec


def test_raw_cache_preserves_original_frame_and_flags(synthetic_atd, tmp_path):
    meta, raw = synthetic_atd()
    store = build_frame_cache(meta_path=meta, raw_path=raw, output_dir=tmp_path/"cache",
                             pitch=(100, 60), n_frames=60, blocks=[(1, 0, 29), (2, 40, 59)], expected_slots=4)
    assert store.runs == ((1, 0, 29), (2, 40, 59))
    assert store.frame(25).frame == 25
    assert store.frame(25).period == 1
    np.testing.assert_allclose(store.frame(0).xy_m[0], [20, 12])
    again = open_frame_cache(tmp_path/"cache")
    assert again.fingerprint == store.fingerprint
    with pytest.raises(ValueError):
        again.xy_m[0, 0] = 999
    with pytest.raises(FileExistsError):
        build_frame_cache(meta_path=meta, raw_path=raw, output_dir=tmp_path/"cache",
                          pitch=(100, 60), n_frames=60, blocks=[(1, 0, 59)], expected_slots=4)


def test_corrupted_raw_cache_not_published(synthetic_atd, tmp_path):
    meta, raw = synthetic_atd()
    rows = raw.read_text().splitlines()
    rows[10] = "10:"+rows[10].split(":", 1)[1]
    raw.write_text("\n".join(rows)+"\n")
    with pytest.raises(ValueError):
        build_frame_cache(meta_path=meta, raw_path=raw, output_dir=tmp_path/"bad_cache", pitch=(100, 60),
                          n_frames=60, blocks=[(1, 0, 59)], expected_slots=4)
    assert not (tmp_path/"bad_cache").exists()
    assert not list(tmp_path.glob(".player_cache_*"))


@pytest.mark.parametrize("ids,periods", [([0, 1, 1], [1, 1, 1]), ([0, 2, 1], [1, 1, 1]),
                                       ([0, .5, 1], [1, 1, 1]), ([0, 1, 2], [1, 3, 1])])
def test_store_rejects_invalid_frame_keys(ids, periods):
    with pytest.raises(ValueError):
        FrameStore(ids, periods, np.zeros((3, 2, 2)), [0, 1], (100, 60))


def test_synthetic_temporal_end_to_end_and_reload(synthetic_atd, tmp_path):
    meta, raw = synthetic_atd(n_frames=2400)
    dest = tmp_path/"temporal"
    report = build_temporal_data(meta_path=meta, raw_path=raw, output_dir=dest, pitch=(100, 60),
                                n_frames=2400, blocks=[(1, 0, 1199), (2, 1200, 2399)], expected_slots=4,
                                policy=ObservationPolicy(min_team_players=2), spec=WindowSpec(stride=25))
    assert report["support_audit"]["cross_split_overlap_frames"] == 0
    assert report["deployment_index"]["windows"] == 2400  # deployment ALWAYS covers all frames
    assert report["deployment_index"]["padded_windows"] == 300
    assert not report["training_performed"]
    assert all(r["hidden_sentinel_passed"] for r in report["materialization_audit"].values())
    train = open_temporal_dataset(dest)
    w = train[0]
    assert train.get_targets(0, window=w).loss_mask.any()
    inference = open_temporal_dataset(dest, mode="inference", split=None)
    assert len(inference) == 2400
    assert inference[0].inputs.padding_mask.sum() == 75
    val = open_temporal_dataset(dest, split="validation")
    val.set_epoch(123)
    np.testing.assert_array_equal(val[0].inputs.artificial_mask,
                                  open_temporal_dataset(dest, split="validation")[0].inputs.artificial_mask)
    assert open_temporal_dataset(dest, split="test", mode="inference")[0].inputs.padding_mask[:75].all()
    # A materialized sample's mask epoch remains valid even after set_epoch.
    train.set_epoch(77)
    np.testing.assert_array_equal(train.get_targets(0, window=w).artificial_mask, w.inputs.artificial_mask)
    assert json.loads((dest/"train_normalizer.json").read_text())["fit_role"] == "train_only_unique_frames"
    header = dest/"validation_windows.json"
    original_header = json.loads(header.read_text())
    for wrong_field, wrong_value in (("split", "train"), ("mode", "inference")):
        bad = {**original_header, wrong_field: wrong_value}
        header.write_text(json.dumps(bad))
        with pytest.raises(ValueError):
            open_temporal_dataset(dest, split="validation")
    header.write_text(json.dumps(original_header))
    with pytest.raises(FileExistsError):
        build_temporal_data(meta_path=meta, raw_path=raw, output_dir=dest, pitch=(100, 60),
                            n_frames=2400, blocks=[(1, 0, 1199)], expected_slots=4)
