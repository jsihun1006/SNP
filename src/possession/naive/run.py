"""Run prediction -> format validation -> official public evaluation."""
import argparse
import os
from pathlib import Path
import subprocess
import sys


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input-mode", choices=("auto", "atd", "published-raw"), default="auto")
    parser.add_argument("--data-dir", type=Path)
    parser.add_argument("--atd-raw", type=Path)
    parser.add_argument("--atd-meta", type=Path)
    parser.add_argument("--bootstrap", type=int, default=1000)
    args = parser.parse_args()
    here = Path(__file__).resolve().parent
    env = dict(os.environ, PYTHONDONTWRITEBYTECODE="1")
    baseline_args = ["--input-mode", args.input_mode]
    for flag, value in (("--data-dir", args.data_dir), ("--atd-raw", args.atd_raw), ("--atd-meta", args.atd_meta)):
        if value is not None:
            baseline_args += [flag, str(value)]
    for script, flags in (("baseline.py", baseline_args), ("validate.py", []),
                          ("evaluation.py", ["--bootstrap", str(args.bootstrap)])):
        print(f"\n== {script} ==", flush=True)
        subprocess.run([sys.executable, str(here / script), *flags], env=env, check=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
