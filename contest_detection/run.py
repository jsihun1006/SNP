"""Portable launcher; use `python contest_detection/run.py --help` from any cwd."""
from pathlib import Path
import json
import hashlib
import os
import subprocess
import sys

ROOT = Path(__file__).resolve().parents[1]
STAGES = {
    "v1": "competition_judgement.cli",
    "v2_1": "competition_judgement.analyze_local_clusters",
    "v2_2": "competition_judgement.analyze_temporal_relations",
    "stage_a": "competition_judgement.run_missingness_stage_a",
    "stage_b": "competition_judgement.run_evidence_stage_b",
    "stage_b_revised": "competition_judgement.run_pattern_b1",
    "stage_b_2": "competition_judgement.run_local_temporal_b2",
    "v2_3": "competition_judgement.run_candidate_v23",
}


def environment():
    env = os.environ.copy()
    paths = [ROOT / "competition_judgement/src", ROOT / "MTAH_Relational_Starter", ROOT]
    env["PYTHONPATH"] = os.pathsep.join([str(p) for p in paths] + ([env["PYTHONPATH"]] if env.get("PYTHONPATH") else []))
    env["PYTHONUTF8"] = "1"
    env["PYTHONDONTWRITEBYTECODE"] = "1"
    return env


def verify_snapshot():
    manifest = json.loads((ROOT / "contest_detection/results/source_snapshot.json").read_text(encoding="utf-8"))
    for item in manifest["files"]:
        actual = hashlib.sha256((ROOT / item["path"]).read_bytes()).hexdigest()
        if actual != item["sha256"]:
            raise ValueError("Preserved source differs: " + item["path"])
    print(f"Verified {len(manifest['files'])} preserved source files (SHA256).")


def main(argv=None):
    args = list(sys.argv[1:] if argv is None else argv)
    if not args or args[0] in ("-h", "--help"):
        print("Usage: run.py {test [--legacy|--repository]|verify-snapshot|" + "|".join(STAGES) + "} [original stage arguments]\n"
              "test: synthetic contest tests, no video/PyTorch. --legacy: also relational and video tests.\n"
              "--repository: also unrelated root tests; requires their separate dependencies/assets.\n"
              "Stage outputs require separate local licensed inputs. Existing outputs are never overwritten.")
        return 0
    action, *rest = args
    if action == "verify-snapshot":
        if rest:
            raise ValueError("verify-snapshot takes no arguments")
        verify_snapshot()
        return 0
    if action == "test":
        repository = "--repository" in rest
        legacy = "--legacy" in rest or repository
        rest = [a for a in rest if a not in ("--legacy", "--repository")]
        suites = ["MTAH_Relational_Starter/tests", "competition_judgement/tests", "contest_detection/tests"] if legacy else ["competition_judgement/tests", "contest_detection/tests", "--ignore=competition_judgement/tests/test_candidate_video.py"]
        if repository:
            suites.insert(0, "tests")
        command = [sys.executable, "-m", "pytest", "-q", "-p", "no:cacheprovider", *suites, *rest]
    elif action in STAGES:
        command = [sys.executable, "-m", STAGES[action], *rest]
    else:
        raise ValueError("Unknown action: " + action)
    return subprocess.run(command, cwd=ROOT, env=environment(), check=False).returncode


if __name__ == "__main__":
    raise SystemExit(main())
