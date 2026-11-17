"""Phase 2 possession on top of the dead ball: 5fps PathCRF + 25fps specialist (README_MODEL.md) on the match.

    python model/phase2_possession/model2_pathcrf/possession.py --dead-run submission --run-id submission
    python run_submission.py --possession model2_pathcrf                 # same, inside the full pipeline
    python model/phase2_possession/model2_pathcrf/possession.py --dead-run submission --prepare-only

Workflow
  1. Phase 1 dead ball: outputs/phase1/dead_ball/<dead_run>/features.csv (frame, graded, dead, dead_source).
  2. The match as the dead-ball step reads it (../overlay.load_match): ATD players (46 track slots, anonymous,
     never joined), corrected ball, 0-based Tactical frames. Written as the model's input NPZ
     (README_MODEL.md "입력 NPZ"): players, ball, team, frame, half, seconds, pitch, player_names, dead
     (dead = the dead-ball call, and every frame outside the graded halves, as predict.py requires).
  3. predict.py (unchanged model code, frozen weights in weights/, no training or search) -> frame,label H/A/D.
  4. ../overlay.write_run: D wherever the dead ball says dead, the model's H/A on the other graded frames,
     X outside the graded halves; prediction.csv is checked with the organisers' validator.
Fallback (--fallback): the 5fps stage can leave in-play frames uncovered (it skips player-set phases with
fewer than 100 in-play frames). predict.py then stops unless an explicit H/A fallback is given. "model1" uses
the H/A of model1_dead_ball's prediction for the same run id (run it first); "none" stops instead.
Output: outputs/phase2/model2_pathcrf/<run_id>/ (input/, model_output.csv, and the overlay files).
"""

import argparse
from pathlib import Path
import subprocess
import sys
from time import perf_counter

import numpy as np
import pandas as pd

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parent))
import overlay  # noqa: E402

NAME = HERE.name
FPS = overlay.FPS


def model_input(match, features):
    """Arrays of the model's input NPZ (README_MODEL.md), from the match and the dead-ball call."""
    n = overlay.N_FRAMES
    period = np.asarray(match.period, dtype=np.int8)
    graded = features["graded"].to_numpy() == 1
    if len(period) != n or not np.array_equal(graded, period > 0):
        raise SystemExit("the dead-ball run and the match disagree on the graded halves")
    team = np.asarray(match.team, dtype=np.int8)
    dead = (features["dead"].to_numpy() == 1) | ~graded
    names = np.array(["{}_{:02d}".format("home" if t == 0 else "away", slot + 1) for slot, t in enumerate(team)])
    frame = np.arange(n, dtype=np.int64)
    return {"players": np.asarray(match.players, dtype=np.float32), "ball": np.asarray(match.ball, dtype=np.float32),
            "team": team, "frame": frame, "half": period, "seconds": frame / FPS,
            "pitch": np.asarray(match.pitch, dtype=np.float64), "player_names": names,
            "dead": dead.astype(np.int8)}


def fallback_from_model1(run_id, dead):
    path = overlay.ROOT / "outputs" / "phase2" / "model1_dead_ball" / run_id / "prediction.csv"
    if not path.is_file():
        raise SystemExit("--fallback model1 needs {} (run model1_dead_ball/possession.py first)".format(path))
    labels = pd.read_csv(path)["label"].to_numpy(str)
    fallback = np.where(labels == "H", 0, np.where(labels == "A", 1, -1)).astype(np.int8)
    if (fallback[~dead] < 0).any():
        raise SystemExit("model1_dead_ball has no H/A on some in-play frames of this dead-ball run")
    return fallback


def main():
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--dead-run", help="dead-ball run id (default: latest)")
    parser.add_argument("--run-id", help="run id of this prediction (default: same as the dead-ball run)")
    parser.add_argument("--fallback", choices=("none", "model1"), default="none",
                        help="explicit H/A where the 5fps stage leaves in-play frames uncovered")
    parser.add_argument("--device", choices=("auto", "cpu", "mps", "cuda"), default="auto")
    parser.add_argument("--batch", type=int, default=16)
    parser.add_argument("--threads", type=int, default=4)
    parser.add_argument("--prepare-only", action="store_true", help="write the model input and stop")
    args = parser.parse_args()
    started = perf_counter()
    features, dead_run = overlay.load_dead_ball(args.dead_run)
    run_id = args.run_id or dead_run.name
    out = overlay.ROOT / "outputs" / "phase2" / NAME / run_id
    (out / "input").mkdir(parents=True, exist_ok=True)
    match, _ = overlay.load_match()
    arrays = model_input(match, features)
    if args.fallback == "model1":
        arrays["fallback_ha"] = fallback_from_model1(run_id, arrays["dead"].astype(bool))
    npz, dead_csv = out / "input" / "match.npz", out / "input" / "dead.csv"
    np.savez_compressed(npz, **arrays)
    pd.DataFrame({"frame": arrays["frame"], "dead": arrays["dead"]}).to_csv(dead_csv, index=False)
    print("{}: model input written to {} ({} frames, {} track slots, fallback {})".format(
        NAME, npz.parent.relative_to(overlay.ROOT), len(arrays["frame"]), len(arrays["team"]), args.fallback))
    if args.prepare_only:
        return 0
    raw = out / "model_output.csv"
    subprocess.run([sys.executable, str(HERE / "predict.py"), "--input", str(npz), "--dead", str(dead_csv),
                    "--output", str(raw), "--device", args.device, "--batch", str(args.batch),
                    "--threads", str(args.threads)], check=True)
    labels = pd.read_csv(raw)
    if len(labels) != overlay.N_FRAMES or not (labels["frame"].to_numpy() == np.arange(overlay.N_FRAMES)).all():
        raise SystemExit("{}: expected frames 0..{}".format(raw, overlay.N_FRAMES - 1))
    label = labels["label"].to_numpy(str)
    team = np.where(np.isin(label, overlay.TEAMS), label, "").astype(object)
    overlay.write_run(NAME, run_id, team, features, dead_run, code=__file__,
                      extra={"possession_model": "5fps PathCRF trial 140 + 25fps specialist epoch 29 (README_MODEL.md)",
                             "fallback": args.fallback, "device": args.device,
                             "seconds": round(perf_counter() - started, 1)})
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
