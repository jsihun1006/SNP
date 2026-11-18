"""Run local detection, integrity checks, ATD diagnostics and preview generation."""
import argparse
from pathlib import Path

from ..detection.profile import PROFILES
import subprocess
import sys


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--out", type=Path, default=PROFILES["video_naive"].artifacts)
    parser.add_argument("--stride", type=int, default=25)
    args = parser.parse_args()
    for name, flags in (("extract", ["--stride", str(args.stride)]),
                        ("validate", []), ("assess", []), ("preview", [])):
        subprocess.run([sys.executable, "-m", f"src.tracking.detection.{name}",
                        "--model", "video_naive", "--out", str(args.out), *flags], check=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
