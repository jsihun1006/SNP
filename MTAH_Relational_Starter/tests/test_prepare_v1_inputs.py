"""Tiny ATD integration tests for the v1 input-generation entry point."""
import hashlib
import json

import numpy as np
import pandas as pd
import pytest

from contest_detection.prepare_v1_inputs import prepare_inputs, main, CACHE_FILES, FEATURE_FILES
from mtah_relational.data_quality import ObservationPolicy
from mtah_relational.frame_store import open_frame_cache
from mtah_relational.splits import SplitManifest


def options(meta, raw, root, **changes):
    result = dict(meta_path=meta, raw_path=raw, cache_dir=root / "temporal/raw_cache",
                  split_path=root / "temporal/split_manifest.json", features_dir=root / "features",
                  pitch=(100, 60), fps=25, n_frames=60, blocks=[(1, 0, 29), (2, 40, 59)],
                  expected_slots=4, policy=ObservationPolicy(min_team_players=2), progress=False)
    result.update(changes)
    return result


def test_prepare_complete_v1_input_contract_without_changing_raw(synthetic_atd, tmp_path):
    meta, raw = synthetic_atd()
    before = {p: (hashlib.sha256(p.read_bytes()).hexdigest(), p.stat().st_mtime_ns) for p in (meta, raw)}
    root = tmp_path / "Windows path with spaces"
    result = prepare_inputs(**options(meta, raw, root))
    assert all((root / "temporal/raw_cache" / name).is_file() for name in CACHE_FILES)
    assert all((root / "features" / name).is_file() for name in FEATURE_FILES)
    store = open_frame_cache(root / "temporal/raw_cache")
    assert store.frame_ids.tolist() == list(range(60))
    manifest = SplitManifest.load(root / "temporal/split_manifest.json")
    assert manifest.split_of(30) is None and manifest.split_of(50) is not None
    frame = pd.read_csv(root / "features/features.csv")
    np.testing.assert_array_equal(frame.frame, store.frame_ids)
    np.testing.assert_array_equal(frame.period, store.periods)
    zones = pd.read_csv(root / "features/zones_48.csv")
    assert set(zones.frame) == {0, 25, 50}
    assert zones.groupby("frame").size().eq(48).all()
    assert json.loads((root / "features/quality_report.json").read_text())["policy"]["min_team_players"] == 2
    assert result["original_inputs_preserved"]
    assert not result["training_performed"] and not result["candidate_inference_performed"]
    assert before == {p: (hashlib.sha256(p.read_bytes()).hexdigest(), p.stat().st_mtime_ns) for p in (meta, raw)}


def test_existing_output_blocks_all_stages_before_any_write(synthetic_atd, tmp_path):
    meta, raw = synthetic_atd()
    root = tmp_path / "keep existing"
    sentinel = root / "features/existing_experiment.txt"
    sentinel.parent.mkdir(parents=True)
    sentinel.write_text("preserve me")
    with pytest.raises(FileExistsError, match="Outputs already exist"):
        prepare_inputs(**options(meta, raw, root))
    assert sentinel.read_text() == "preserve me"
    assert not (root / "temporal").exists()


def test_split_can_be_generated_without_raw_inputs(tmp_path):
    result = prepare_inputs(**options(tmp_path / "missing.xml", tmp_path / "missing.txt", tmp_path / "out", only="split"))
    assert set(result["generated"]) == {"split"}
    assert not (tmp_path / "out/temporal/raw_cache").exists()
    assert not (tmp_path / "out/features").exists()


def test_missing_raw_does_not_create_outputs(tmp_path):
    root = tmp_path / "out"
    with pytest.raises(FileNotFoundError, match="Original ATD inputs required"):
        prepare_inputs(**options(tmp_path / "missing.xml", tmp_path / "missing.txt", root))
    assert not root.exists()


def test_only_features_preserves_an_existing_cache(synthetic_atd, tmp_path):
    meta, raw = synthetic_atd()
    root = tmp_path / "out"
    sentinel = root / "temporal/raw_cache/preserve.npy"
    sentinel.parent.mkdir(parents=True)
    sentinel.write_bytes(b"existing cache")
    result = prepare_inputs(**options(meta, raw, root, only="features"))
    assert set(result["generated"]) == {"features"}
    assert sentinel.read_bytes() == b"existing cache"
    assert not (root / "temporal/split_manifest.json").exists()


def test_nested_output_paths_are_rejected_without_writes(synthetic_atd, tmp_path):
    meta, raw = synthetic_atd()
    root = tmp_path / "out"
    with pytest.raises(ValueError, match="must be distinct and not nested"):
        prepare_inputs(**options(meta, raw, root, features_dir=root / "temporal"))
    assert not root.exists()


def test_plan_lists_missing_inputs_without_creating_files(tmp_path, capsys):
    root = tmp_path / "new checkout"
    assert main(["--repo-root", str(root), "--data-dir", str(tmp_path / "source"), "--plan"]) == 0
    plan = json.loads(capsys.readouterr().out)
    assert not plan["writes_performed"] and not plan["creates_raw_atd"]
    assert not any(row["exists"] for row in plan["inputs"].values())
    assert not root.exists()
