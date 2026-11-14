"""Check preservation and the new Windows launch boundary, not new algorithms."""
from pathlib import Path
import os

from contest_detection import features, run
from competition_judgement import local_features, temporal_features, pattern_profiles
from competition_judgement import local_temporal_features


def test_adapter_exports_original_function_objects():
    assert features.extract_local_frame is local_features.extract_local_frame
    assert features.component_features is local_features.component_features
    assert features.summarize_anchor_series is temporal_features.summarize_anchor_series
    assert features.profiles is pattern_profiles.profiles
    assert features.local_features is local_temporal_features.local_features


def test_original_source_snapshot_is_unchanged():
    run.verify_snapshot()


def test_launcher_preserves_environment_and_uses_platform_path_separator(monkeypatch):
    monkeypatch.setenv("PYTHONPATH", "existing import path")
    env = run.environment()
    assert env["PYTHONPATH"].split(os.pathsep)[-1] == "existing import path"
    assert Path(env["PYTHONPATH"].split(os.pathsep)[0]) == run.ROOT / "competition_judgement/src"
    assert env["PYTHONUTF8"] == "1"


def test_stage_launch_uses_repository_root_and_forwards_arguments(monkeypatch, tmp_path):
    calls = []
    def execute(command, **kwargs):
        calls.append((command, kwargs))
        return type("Result", (), {"returncode": 7})()
    monkeypatch.setattr(run.subprocess, "run", execute)
    monkeypatch.chdir(tmp_path)
    assert run.main(["v2_3", "--output", "data/derived/competition_judgement/new run"]) == 7
    command, options = calls[0]
    assert command[2] == "competition_judgement.run_candidate_v23"
    assert command[-1] == "data/derived/competition_judgement/new run"
    assert options["cwd"] == run.ROOT
    assert options["env"]["PYTHONUTF8"] == "1"
