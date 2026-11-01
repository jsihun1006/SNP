"""Run local detection, integrity checks, ATD diagnostics and preview generation."""
import argparse
from pathlib import Path
import subprocess
import sys


def main():
    here = Path(__file__).resolve().parent
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--out", type=Path, default=here / "outputs")
    parser.add_argument("--stride", type=int, default=25)
    args = parser.parse_args()
    for name, flags in (("extract.py", ["--stride", str(args.stride)]),
                        ("validate.py", []), ("assess.py", []), ("preview.py", [])):
        subprocess.run([sys.executable, str(here / name), "--out", str(args.out), *flags], check=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
