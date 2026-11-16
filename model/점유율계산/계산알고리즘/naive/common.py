"""Shared paths and the unchanged organiser implementation."""
from pathlib import Path
import hashlib
import json
import sys

HERE = Path(__file__).resolve().parent
ROOT = next((parent for parent in (HERE, *HERE.parents) if (parent / "games.py").is_file()), None)
if ROOT is None:
    raise RuntimeError("Cannot locate repository root containing games.py")
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "evaluation"))

import possession_metrics as metrics
import run_evaluation as official_evaluation
from games import GAME, GAMES, asset_paths
from starter import baseline as official_baseline

N_FRAMES = GAMES[GAME]["n_video_frames"]
FPS = GAMES[GAME]["fps"]
PRIMARY = "baseline_smooth50"  # Published reference, fixed before seeing public scores.


def sha256(path):
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def write_json(path, value):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")


def checked_labels(path):
    table = metrics.read_label_csv(str(path))
    reason = metrics.validate_submission(table, N_FRAMES)
    if reason:
        raise ValueError(f"{path}: {reason}")
    return table, metrics.label_codes(table)


def timestamp(frame):
    """Tactical video elapsed time, not the football match clock."""
    millis = round(int(frame) * 1000 / FPS)
    hours, rem = divmod(millis, 3600000)
    minutes, rem = divmod(rem, 60000)
    seconds, rem = divmod(rem, 1000)
    return f"{hours:02}:{minutes:02}:{seconds:02}.{rem:03}"
