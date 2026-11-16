"""Generate model1, validate it, then run one official public evaluation."""
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
    args = parser.parse_args()
    here = Path(__file__).resolve().parent
    flags = ["--input-mode", args.input_mode]
    for key in ("data_dir", "atd_raw", "atd_meta"):
        value = getattr(args, key)
        if value is not None:
            flags += ["--" + key.replace("_", "-"), str(value)]
    env = dict(os.environ, PYTHONDONTWRITEBYTECODE="1")
    for name, arguments in (("model.py", flags), ("validate.py", []), ("evaluation.py", [])):
        print(f"\n== {name} ==", flush=True)
        subprocess.run([sys.executable, str(here / name), *arguments], env=env, check=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
