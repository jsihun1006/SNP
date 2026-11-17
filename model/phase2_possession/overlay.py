"""Shared Phase 2 step: put the Phase 1 dead ball first, then a possession model's H/A on top.

A possession model only has to give a team (H or A) for every frame. This module
  1. reads the Phase 1 dead-ball run (outputs/phase1/dead_ball/<run_id>/features.csv),
  2. overlays:  graded & dead  -> D          (the dead ball always wins inside the graded halves)
                graded & alive -> the model's H/A (required on every such frame)
                ungraded       -> the model's H/A, or D where it gives none (not scored)
  3. writes outputs/phase2/<model>/<run_id>/ (prediction.csv, frame_diagnostics.csv, possession_share.csv,
     run_manifest.json) and runs the organisers' validator on prediction.csv.
load_match() gives the possession model the provided match with the same ball the dead-ball step used
(corrected or ATD, CONFIG["competition_ball"] of the dead-ball model).
"""

from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path
import subprocess
import sys

import numpy as np
import pandas as pd

PHASE2 = Path(__file__).resolve().parent
ROOT = PHASE2.parents[1]
DEAD_BALL = ROOT / "model" / "phase1_extraction" / "dead_ball"
DEAD_ROOT = ROOT / "outputs" / "phase1" / "dead_ball"
N_FRAMES, FPS = 199319, 25
TEAMS = ("H", "A")


def sha256(path):
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for chunk in iter(lambda: handle.read(1 << 20), b""):
            digest.update(chunk)
    return digest.hexdigest()


def dead_ball_run(run_id=None):
    """Folder of a dead-ball run: the given id, else the latest one."""
    if run_id:
        path = DEAD_ROOT / run_id
        if not (path / "features.csv").is_file():
            raise SystemExit("no dead-ball run {}; run model/phase1_extraction/dead_ball/extract.py".format(path))
        return path
    runs = [p for p in DEAD_ROOT.iterdir() if (p / "features.csv").is_file()] if DEAD_ROOT.is_dir() else []
    if not runs:
        raise SystemExit("no dead-ball run found; run model/phase1_extraction/dead_ball/extract.py first")
    return max(runs, key=lambda p: (p / "features.csv").stat().st_mtime)


def load_dead_ball(run_id=None):
    """(features DataFrame frame,graded,dead,dead_source for all frames, run folder)."""
    run = dead_ball_run(run_id)
    features = pd.read_csv(run / "features.csv", usecols=["frame", "graded", "dead", "dead_source"])
    if len(features) != N_FRAMES or not (features["frame"].to_numpy() == np.arange(N_FRAMES)).all():
        raise SystemExit("{}: expected frames 0..{}".format(run / "features.csv", N_FRAMES - 1))
    features["dead_source"] = features["dead_source"].fillna("")
    return features, run


def load_match():
    """(features.Match, graded half blocks) of the provided match, read exactly as the dead-ball step reads it:
    players from the ATD, ball per CONFIG["competition_ball"]. Metres, 0-based Tactical frames."""
    sys.path.insert(0, str(DEAD_BALL / "src"))
    import competition_data
    return competition_data.load_competition_match()


def overlay(team, graded, dead):
    """Labels H/A/D per frame. team: N_FRAMES values 'H', 'A' or '' (no call)."""
    team = np.asarray(team, dtype=object)
    graded, dead = np.asarray(graded, dtype=bool), np.asarray(dead, dtype=bool)
    if not len(team) == len(graded) == len(dead) == N_FRAMES:
        raise ValueError("team, graded and dead need {} frames".format(N_FRAMES))
    has_team = np.isin(team, TEAMS)
    if not np.isin(team[~has_team], ["", None]).all():
        raise ValueError("team values must be 'H', 'A' or '' (no call)")
    missing = graded & ~dead & ~has_team
    if missing.any():
        raise ValueError("the possession model gave no team on {} in-play graded frames (first {})".format(
            int(missing.sum()), int(np.flatnonzero(missing)[0])))
    labels = np.where(has_team, team, "D").astype(object)
    labels[graded & dead] = "D"
    return labels.astype(str)


def share_table(labels, graded):
    rows = []
    for scope, mask in (("graded", graded), ("full_video", np.ones(len(labels), dtype=bool))):
        alive = int(np.sum(mask & (labels != "D")))
        for team in TEAMS:
            frames = int(np.sum(mask & (labels == team)))
            rows.append({"scope": scope, "team": team, "frames": frames, "seconds": round(frames / FPS, 2),
                         "share": round(frames / alive, 6) if alive else ""})
    return pd.DataFrame(rows)


def write_run(model, run_id, team, features, dead_run, extra=None, code=None):
    """Overlay the dead ball on `team`, write outputs/phase2/<model>/<run_id>/ and validate prediction.csv."""
    graded = features["graded"].to_numpy() == 1
    dead = features["dead"].to_numpy() == 1
    labels = overlay(team, graded, dead)
    out = ROOT / "outputs" / "phase2" / model / run_id
    out.mkdir(parents=True, exist_ok=True)
    prediction = out / "prediction.csv"
    with prediction.open("w", newline="\n") as handle:
        handle.write("frame,label\n")
        handle.writelines("{},{}\n".format(frame, label) for frame, label in enumerate(labels))
    team = np.asarray(team, dtype=object)
    pd.DataFrame({"frame": np.arange(N_FRAMES), "label": labels, "model_team": np.where(np.isin(team, TEAMS), team, ""),
                  "graded": graded.astype(int), "dead": dead.astype(int), "dead_source": features["dead_source"],
                  "label_source": np.where(graded & dead, "dead_ball", np.where(np.isin(team, TEAMS), "possession_model",
                                                                                "no_call"))}
                 ).to_csv(out / "frame_diagnostics.csv", index=False)
    share_table(labels, graded).to_csv(out / "possession_share.csv", index=False)
    counts = {label: int(np.sum(labels == label)) for label in ("H", "A", "D")}
    manifest = {
        "model": model, "run_id": run_id, "phase": 2, "created_at_utc": datetime.now(timezone.utc).isoformat(),
        "overlay": "graded & dead -> D; graded & alive -> model H/A; ungraded -> model H/A or D",
        "phase1_input": {"model": "dead_ball", "run_id": dead_run.name,
                         "features_sha256": sha256(dead_run / "features.csv")},
        "code_sha256": {Path(p).name: sha256(p) for p in ([code] if code else []) + [__file__]},
        "prediction_sha256": sha256(prediction), "label_counts": counts, **(extra or {})}
    (out / "run_manifest.json").write_text(json.dumps(manifest, indent=2) + "\n", encoding="utf-8")
    check = subprocess.run([sys.executable, str(ROOT / "evaluation" / "validate_submission.py"), str(prediction)],
                           capture_output=True, text=True)
    if check.returncode != 0:
        raise SystemExit("validator failed on {}:\n{}".format(prediction, check.stdout + check.stderr))
    print("{}: wrote {} (validator OK)".format(model, prediction.relative_to(ROOT)))
    print("label counts:", counts)
    return out
