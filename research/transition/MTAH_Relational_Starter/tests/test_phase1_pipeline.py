import csv
import json

import numpy as np
import pandas as pd
import pytest

from mtah_relational.data_quality import ObservationPolicy
from mtah_relational.phase1 import run_phase1


def run(meta, raw, dest, **kwargs):
    return run_phase1(meta_path=meta, raw_path=raw, output_dir=dest, pitch=(100, 60), fps=25,
                      n_frames=60, blocks=[(1, 0, 29), (2, 40, 59)], expected_slots=4,
                      policy=ObservationPolicy(min_team_players=2), **kwargs)


def test_complete_local_pipeline_with_optional_model_arrays(synthetic_atd, tmp_path):
    meta, raw = synthetic_atd()
    dest = tmp_path/"phase1"
    report = run(meta, raw, dest, save_nodes=True, zone_stride=5)
    f = pd.read_csv(dest/"features.csv")
    s = pd.read_csv(dest/"symmetric_features.csv")
    z = pd.read_csv(dest/"zones_48.csv")
    assert f.frame.tolist() == s.frame.tolist() == list(range(60))
    assert f.loc[25, "time_s"] == 1
    assert f.index_valid.sum() == 50
    assert f.loc[30:39, "contested_index"].isna().all()
    assert f.loc[25, "temporal_available"]
    assert not f.loc[40:, "temporal_available"].any()
    assert report["quality"]["official_halves"]["frames"] == 50
    assert report["alignment"]["strict_contiguous_ids"]
    assert not report["index"]["probability"]
    assert len(z) == 10*48
    for frame, frame_zones in z.groupby("frame"):
        assert frame_zones.n_players.sum() == f.loc[frame, "n_players"]
        assert frame_zones.within_zone_pairs_3m.sum() <= f.loc[frame, "cross_pairs_3m"]
    nodes = np.load(dest/"source_slot_xy_m.npy", mmap_mode="r")
    observed = np.load(dest/"observed.npy", mmap_mode="r")
    assert nodes.shape == (60, 4, 2)
    np.testing.assert_array_equal(np.isfinite(nodes).all(axis=-1), observed)
    assert np.load(dest/"team_codes.npy").tolist() == [0, 0, 1, 1]
    schema = json.loads((dest/"schema.json").read_text())
    assert len(schema["symmetric_features"]) == 24
    assert len(schema["future_reconstruction_targets"]) == 13
    assert not schema["neural_model_implemented"]
    assert "home_cx" in f and "away_cx" in f
    with pytest.raises(FileExistsError):
        run(meta, raw, dest)


def test_bad_file_does_not_publish_partial_outputs(synthetic_atd, tmp_path):
    meta, raw = synthetic_atd()
    rows = raw.read_text().splitlines()
    rows[10] = "10:" + rows[10].split(":", 1)[1]
    raw.write_text("\n".join(rows)+"\n")
    out = tmp_path/"bad_output"
    with pytest.raises(ValueError, match="line 11"):
        run(meta, raw, out, save_nodes=True)
    assert not out.exists()
    assert not list(tmp_path.glob(".relational_phase1_*"))


def test_ball_speed_and_smart_tagging_never_influence_outputs(synthetic_atd, tmp_path):
    meta, raw = synthetic_atd(ball="irrelevant:ball")
    (meta.parent/"smart_tagging.xml").write_text("THIS IS NOT XML; DO NOT READ")
    run(meta, raw, tmp_path/"one", zone_stride=0)
    raw.write_text(raw.read_text().replace("irrelevant:ball", "entirely_different").replace("unused_speed", "999999"))
    run(meta, raw, tmp_path/"two", zone_stride=0)
    for name in ("features.csv", "symmetric_features.csv"):
        assert (tmp_path/"one"/name).read_bytes() == (tmp_path/"two"/name).read_bytes()


@pytest.mark.parametrize("kwargs", [{"zone_stride": -1}, {"zone_stride": .5}, {"zone_stride": "25"}])
def test_invalid_output_sampling_rejected(synthetic_atd, tmp_path, kwargs):
    meta, raw = synthetic_atd()
    with pytest.raises(ValueError):
        run(meta, raw, tmp_path/"bad", **kwargs)
