"""Rebuild the DFL LightGBM experiments from the repository's raw XML files."""
import argparse
import csv
import math
from pathlib import Path
import subprocess
import sys

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[3]
EXPERIMENTS = (
    ("dfl_half_7fold_comparison", "compare_deadball_half_folds.py", 6, None),
    ("dfl_event_7fold_comparison", "compare_deadball_event_folds.py", 10, "predicted"),
    ("dfl_event_training_only_7fold", "compare_deadball_event_folds.py", 8, "missing"),
)


def compare_csv(reference, actual):
    with reference.open(encoding="utf-8", newline="") as stream:
        expected = list(csv.reader(stream))
    with actual.open(encoding="utf-8", newline="") as stream:
        observed = list(csv.reader(stream))
    if len(expected) != len(observed) or expected[0] != observed[0]:
        raise ValueError(f"CSV schema or row count differs: {actual}")
    exact = True
    for index, (left, right) in enumerate(zip(expected[1:], observed[1:]), start=2):
        if len(left) != len(right):
            raise ValueError(f"CSV field count differs: {actual}:{index}")
        for a, b in zip(left, right):
            if a == b:
                continue
            exact = False
            try:
                equivalent = math.isclose(float(a), float(b), abs_tol=1e-12, rel_tol=0.0)
            except ValueError:
                equivalent = False
            if not equivalent:
                raise ValueError(f"Value differs: {actual}:{index}: {a!r} != {b!r}")
    print(f"VERIFIED {actual.parent.name}/{actual.name} ({'exact' if exact else 'within 1e-12'})", flush=True)


def run_script(script, *args):
    subprocess.run([sys.executable, "-B", str(ROOT / "src" / "deadball" / script), *map(str, args)],
                   cwd=ROOT, check=True)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--cache", type=Path, required=True)
    args = parser.parse_args()
    output, cache = args.output.resolve(), args.cache.resolve()
    splits = HERE / "dfl_half_7fold.json"
    baseline = output / EXPERIMENTS[0][0] / "by_half.csv"
    for name, script, threads, mode in EXPERIMENTS:
        destination = output / name
        options = ["--splits", splits, "--dfl-cache", cache, "--output", destination, "--threads", threads]
        if mode is not None:
            options += ["--baseline", baseline, "--evaluation-events", mode]
        run_script(script, *options)
        for reference in sorted((HERE / "reference" / name).glob("*.csv")):
            compare_csv(reference, destination / reference.name)


if __name__ == "__main__":
    main()
