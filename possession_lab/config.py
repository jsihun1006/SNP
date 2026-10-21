"""Constants, labels, and paths for the possession lab.

The package is intentionally self-contained and does not import the starter notebooks.  The match
constants mirror ``games.py`` for BAR-ATH because this lab is written for that one fixture; the
gated data directory can still be relocated with ``MTAH_DATA_DIR``.
"""

from __future__ import annotations

import os
from pathlib import Path

# ---------------------------------------------------------------------------
# Labels
# ---------------------------------------------------------------------------
H, A, D, X = 0, 1, 2, 3
LABELS = ("H", "A", "D", "X")
LABEL_TO_CODE = {label: code for code, label in enumerate(LABELS)}
CODE_TO_LABEL = {code: label for label, code in LABEL_TO_CODE.items()}

# ---------------------------------------------------------------------------
# Match
# ---------------------------------------------------------------------------
GAME = "BAR-ATH"
SOURCE = "tactical"
FPS = 25
N_FRAMES = 199_319
PITCH = (105.3, 68.0)  # metres, corrected tracking pitch used by games.py
PUBLIC_START, PUBLIC_STOP = 11_185, 18_685  # half-open
GRADED_HALVES = ((11_185, 90_652), (110_808, 190_198))  # half-open
CONTEXT = 500

ROOT = Path(__file__).resolve().parents[1]
DATA_DIR = Path(os.environ.get("MTAH_DATA_DIR", ROOT / "data"))
OUTPUT_DIR = Path(__file__).resolve().parent / "outputs"
PUBLIC_GT_PATH = ROOT / "evaluation" / "gt" / "BAR-ATH_possession_gt_public.csv"
# Written by the local viewer only.  Never read by the model, features, or training code.
USER_TRUTH_PATH = OUTPUT_DIR / "user_truth.json"

# The exact parameters of the recommended model.  Kept here so the CLI, tests, and report agree.
MODEL_PARAMS = {
    "sigma": 3.0,
    "d_dead": 7.0,
    "dead_scale": 1.5,
    "edge0": 5.0,
    "edge_scale": 1.5,
    "w_edge": 0.0,
    "w_gap": 0.0,
    "gap0": 100.0,
    "gap_scale": 25.0,
    "w_out": 1.0,
    "w_tag": 0.0,
    "w_sp": 0.0,
    "w_lt": 0.0,
    "lt_R": 0.8,
    "p_stay": 0.995,
    "p_ha": 0.0005,
    "p_hd": 0.001,
    "p_dh": 0.001,
    "gate": 0.0,
    "miss_mode": "hybrid",
    "sigma_miss": 6.0,
    "gap_thresh": 120.0,
    "min_players": 6,
}


def asset_paths(data_dir: str | os.PathLike[str] | None = None) -> dict[str, Path]:
    """Paths for the tactical ATD, tags, and homography.

    Missing files are not checked here.  ``pipeline.load_lab`` produces a useful message if the
    gated bundle has not been unpacked.
    """
    root = Path(data_dir) if data_dir is not None else DATA_DIR
    stem = f"{GAME}_{SOURCE}"
    folder = root / SOURCE
    return {
        "atd_raw": folder / f"{stem}_atd.txt",
        "atd_meta": folder / f"{stem}_atd.xml",
        "smart_tagging": folder / f"{stem}_smart_tagging.xml",
        "homography": folder / f"{stem}_homography.csv",
    }


def ensure_output_dir(path: str | os.PathLike[str] | None = None) -> Path:
    out = Path(path) if path is not None else OUTPUT_DIR
    out.mkdir(parents=True, exist_ok=True)
    return out
