"""Regenerate the submission file from the provided data in one command.

    python run_submission.py                 # full pipeline -> SNP.csv
    python run_submission.py --run-id submission --skip-score
    python run_submission.py --possession model2_<name>   # another possession model on top of the dead ball

Steps
  1. Phase 1  model/phase1_extraction/dead_ball/extract.py     -> outputs/phase1/dead_ball/<run_id>/
  2. Phase 2  model/phase2_possession/<possession>/possession.py -> outputs/phase2/<possession>/<run_id>/
              (dead ball first, possession on top; default model1_dead_ball)
  3. copy the prediction to SNP.csv and check it with evaluation/validate_submission.py
  4. (optional) score the published 5 minutes as a pipeline check
The provided match data must be unpacked under data/ (see data/README.md).
"""
import argparse
import os
from pathlib import Path
import shutil
import subprocess
import sys

ROOT = Path(__file__).resolve().parent
TEAM = "SNP"


def step(args):
    print("\n== {} ==".format(" ".join(str(a) for a in args)), flush=True)
    env = dict(os.environ, PYTHONDONTWRITEBYTECODE="1")
    subprocess.run([sys.executable, *[str(a) for a in args]], cwd=ROOT, env=env, check=True)


def main():
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--run-id", default="submission")
    parser.add_argument("--skip-score", action="store_true", help="do not score the public 5 minutes")
    parser.add_argument("--possession", default="model1_dead_ball",
                        help="Phase 2 model folder under model/phase2_possession (default model1_dead_ball)")
    args = parser.parse_args()
    step(["model/phase1_extraction/dead_ball/extract.py", "--run-id", args.run_id])
    model = ROOT / "model" / "phase2_possession" / args.possession / "possession.py"
    if not model.is_file():
        raise SystemExit("no possession model {}".format(model))
    step([model.relative_to(ROOT), "--dead-run", args.run_id, "--run-id", args.run_id])
    prediction = ROOT / "outputs" / "phase2" / args.possession / args.run_id / "prediction.csv"
    shutil.copyfile(prediction, ROOT / (TEAM + ".csv"))
    step(["evaluation/validate_submission.py", TEAM + ".csv"])
    if not args.skip_score:
        step(["model/phase2_possession/score_public.py", "--model", args.possession, "--run-id", args.run_id])
    print("\nsubmission file: {}".format(ROOT / (TEAM + ".csv")))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
