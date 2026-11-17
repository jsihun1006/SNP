"""Check prediction.csv with the organisers' validator (evaluation/validate_submission.py).

    python model/phase2_possession/model1_dead_ball/validate.py                 # latest run
    python model/phase2_possession/model1_dead_ball/validate.py --run-id submission
"""
import argparse
from pathlib import Path
import subprocess
import sys

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[2]
OUT_ROOT = ROOT / "outputs" / "phase2" / "model1_dead_ball"


def run_dir(run_id):
    if run_id:
        return OUT_ROOT / run_id
    runs = [p for p in OUT_ROOT.iterdir() if (p / "prediction.csv").is_file()] if OUT_ROOT.is_dir() else []
    if not runs:
        raise SystemExit("no prediction; run possession.py first")
    return max(runs, key=lambda p: (p / "prediction.csv").stat().st_mtime)


def main():
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--run-id")
    args = parser.parse_args()
    prediction = run_dir(args.run_id) / "prediction.csv"
    return subprocess.run([sys.executable, str(ROOT / "evaluation" / "validate_submission.py"), str(prediction)]).returncode


if __name__ == "__main__":
    raise SystemExit(main())
