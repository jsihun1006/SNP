"""Model-specific paths and imports for the possession reference."""
from pathlib import Path
import importlib.util
import sys

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parent))

from reference_common import (
    ROOT, GAME, GAMES, asset_paths, metrics, official_evaluation, official_baseline,
    N_FRAMES, FPS, checked_labels, sha256, timestamp, write_json,
)

RUN_DIR = ROOT / "artifacts/possession/model1/runs/latest"
PRIMARY = "model1"


def naive_helper(filename):
    """Reuse existing diagnostic/validation code without modifying naive files."""
    path = HERE.parent / "naive" / filename
    spec = importlib.util.spec_from_file_location("_model1_helper_" + path.stem, path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module
