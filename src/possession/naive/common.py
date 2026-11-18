"""Model-specific paths and imports for the possession reference."""
from pathlib import Path
import sys

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parent))

from reference_common import (
    ROOT, GAME, GAMES, asset_paths, metrics, official_evaluation, official_baseline,
    N_FRAMES, FPS, checked_labels, sha256, timestamp, write_json,
)

RUN_DIR = ROOT / "artifacts/possession/naive/runs/latest"
PRIMARY = "baseline_smooth50"  # Published reference, fixed before seeing public scores.
