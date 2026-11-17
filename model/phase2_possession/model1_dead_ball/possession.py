"""Phase 2 possession: D from the Phase 1 dead-ball model, H/A from a nearest-team cost decoder.

    python model/phase2_possession/model1_dead_ball/possession.py                    # latest dead_ball run
    python model/phase2_possession/model1_dead_ball/possession.py --dead-run submission

Inputs
  Phase 1  outputs/phase1/dead_ball/<run_id>/features.csv   (frame, graded, dead, dead_source)
  ATD      data/tactical/  nearest player to the ball per frame (starter/baseline.nearest_player_labels)
Decision
  A three-state (H/A/D) minimum-cost sequence decoder. Observation and switch costs are fixed priors
  (config.json). Inside the graded halves, frames the dead-ball model calls dead may only be D and all
  other frames may only be H or A, so dead-ball boundaries come from Phase 1 and only H<->A changes
  inside an in-play stretch come from this decoder. Outside the graded halves the decoder runs unconstrained
  (those frames are not scored).
Output: outputs/phase2/model1_dead_ball/<run_id>/
  prediction.csv          frame,label for all 199,319 frames
  frame_diagnostics.csv   frame,predicted_label,input_source,observation,dead,dead_source
  possession_share.csv    scope,team,frames,seconds,share
  run_manifest.json       inputs, settings and hashes
No ground truth is read.
"""
import argparse
from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path
import sys

import numpy as np
import pandas as pd

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[2]
sys.path[:0] = [str(ROOT), str(ROOT / "starter")]
from games import GAME, GAMES, require  # noqa: E402
import baseline  # noqa: E402

NAME = "model1_dead_ball"
N_FRAMES = GAMES[GAME]["n_video_frames"]
FPS = GAMES[GAME]["fps"]
H, A, D = 0, 1, 2
MISSING = -1
STATES = np.array(["H", "A", "D"])
BIG = 1e9
DEAD_ROOT = ROOT / "outputs" / "phase1" / "dead_ball"
OUT_ROOT = ROOT / "outputs" / "phase2" / NAME
COST_KEYS = ("long_missing_seconds", "opposite_team_cost_per_second", "dead_when_observed_cost_per_second",
             "dead_in_short_gap_cost_per_second", "alive_in_long_gap_cost_per_second", "team_switch_cost",
             "dead_switch_cost")


def sha256(path):
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for chunk in iter(lambda: handle.read(1 << 20), b""):
            digest.update(chunk)
    return digest.hexdigest()


def write_json(path, value):
    Path(path).write_text(json.dumps(value, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")


def validate_config(config):
    if config["fps"] != FPS:
        raise ValueError("config fps must match the tactical fps")
    for key in COST_KEYS:
        if not np.isfinite(config[key]) or config[key] <= 0:
            raise ValueError(key + " must be finite and positive")


def missing_run_lengths(observations):
    """Length of the containing missing run; zero at observed frames."""
    missing = np.asarray(observations) == MISSING
    edges = np.diff(np.r_[False, missing, False].astype(np.int8))
    lengths = np.zeros(len(observations), dtype=np.int32)
    for start, stop in zip(np.flatnonzero(edges == 1), np.flatnonzero(edges == -1)):
        lengths[start:stop] = stop - start
    return lengths


def emission_costs(observations, config):
    """Per-frame cost of each state given the nearest-team observation (H=0, A=1, missing=-1)."""
    observations = np.asarray(observations, dtype=np.int8)
    long_gap = missing_run_lengths(observations) >= int(np.ceil(config["long_missing_seconds"] * FPS))
    costs = np.zeros((len(observations), 3), dtype=np.float64)
    costs[observations == H, A] = config["opposite_team_cost_per_second"]
    costs[observations == A, H] = config["opposite_team_cost_per_second"]
    costs[observations != MISSING, D] = config["dead_when_observed_cost_per_second"]
    costs[(observations == MISSING) & ~long_gap, D] = config["dead_in_short_gap_cost_per_second"]
    costs[long_gap, :2] = config["alive_in_long_gap_cost_per_second"]
    return costs / FPS


def constrained_costs(base_costs, graded, dead):
    """Inside the graded halves: dead frames may only be D, all other frames only H or A."""
    costs = np.asarray(base_costs, dtype=np.float64).copy()
    g, d = np.asarray(graded, dtype=bool), np.asarray(dead, dtype=bool)
    costs[g & d, H] = BIG
    costs[g & d, A] = BIG
    costs[g & ~d, D] = BIG
    return costs


def viterbi(costs, team_switch, dead_switch):
    """Exact minimum of emission costs plus state-change costs (offline, uses future frames)."""
    transitions = np.array([[0, team_switch, dead_switch], [team_switch, 0, dead_switch],
                            [dead_switch, dead_switch, 0]], dtype=float)
    back = np.zeros((len(costs), 3), dtype=np.int8)
    score = costs[0].copy()
    for frame in range(1, len(costs)):
        candidates = score[:, None] + transitions
        parent = candidates.argmin(axis=0)
        back[frame] = parent
        score = costs[frame] + candidates[parent, np.arange(3)]
    labels = np.empty(len(costs), dtype=np.int8)
    labels[-1] = score.argmin()
    for frame in range(len(costs) - 1, 0, -1):
        labels[frame - 1] = back[frame, labels[frame]]
    return labels


def nearest_team_observations():
    meta, raw = require(["atd_meta", "atd_raw"])
    codes = baseline.nearest_player_labels(raw, baseline.team_of_each_track(meta), N_FRAMES, GAMES[GAME]["pitch"])
    return np.where(codes == H, H, np.where(codes == A, A, MISSING)).astype(np.int8), [Path(raw), Path(meta)]


def latest_dead_run():
    runs = [p for p in DEAD_ROOT.iterdir() if (p / "features.csv").is_file()] if DEAD_ROOT.is_dir() else []
    if not runs:
        raise SystemExit("no dead_ball run found; run model/phase1_extraction/dead_ball/extract.py first")
    return max(runs, key=lambda p: (p / "features.csv").stat().st_mtime)


def share_table(labels, graded):
    rows = []
    for scope, mask in (("graded", graded), ("full_video", np.ones(len(labels), dtype=bool))):
        alive = int(np.sum(mask & (labels != D)))
        for team, code in (("H", H), ("A", A)):
            frames = int(np.sum(mask & (labels == code)))
            rows.append({"scope": scope, "team": team, "frames": frames, "seconds": round(frames / FPS, 2),
                         "share": round(frames / alive, 6) if alive else ""})
    return pd.DataFrame(rows)


def main():
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--dead-run", help="dead_ball run id (default: latest)")
    parser.add_argument("--run-id", help="run id of this prediction (default: same as the dead_ball run)")
    parser.add_argument("--config", type=Path, default=HERE / "config.json")
    args = parser.parse_args()
    config = json.loads(args.config.read_text(encoding="utf-8"))
    validate_config(config)
    dead_dir = DEAD_ROOT / args.dead_run if args.dead_run else latest_dead_run()
    features_path = dead_dir / "features.csv"
    features = pd.read_csv(features_path, usecols=["frame", "graded", "dead", "dead_source"])
    if len(features) != N_FRAMES or not (features["frame"].to_numpy() == np.arange(N_FRAMES)).all():
        raise SystemExit("{}: expected frames 0..{}".format(features_path, N_FRAMES - 1))
    graded = features["graded"].to_numpy() == 1
    dead = features["dead"].to_numpy() == 1
    observations, inputs = nearest_team_observations()
    labels = viterbi(constrained_costs(emission_costs(observations, config), graded, dead),
                     config["team_switch_cost"], config["dead_switch_cost"])
    out = OUT_ROOT / (args.run_id or dead_dir.name)
    out.mkdir(parents=True, exist_ok=True)
    prediction = out / "prediction.csv"
    baseline.write_submission(str(prediction), STATES[labels])
    pd.DataFrame({"frame": np.arange(N_FRAMES), "predicted_label": STATES[labels],
                  "input_source": np.where(graded & dead, "dead_ball", np.where(graded, "decoder", "ungraded")),
                  "observation": np.where(observations == H, "H", np.where(observations == A, "A", "MISSING")),
                  "dead": dead.astype(int), "dead_source": features["dead_source"].fillna("").to_numpy()}
                 ).to_csv(out / "frame_diagnostics.csv", index=False)
    share_table(labels, graded).to_csv(out / "possession_share.csv", index=False)
    write_json(out / "run_manifest.json", {
        "model": NAME, "run_id": out.name, "phase": 2, "created_at_utc": datetime.now(timezone.utc).isoformat(),
        "inference": "fixed-cost H/A/D decoder constrained by Phase 1 dead_ball; no ground truth, fitting or search",
        "phase1_input": {"model": "dead_ball", "run_id": dead_dir.name,
                         "features": str(features_path.relative_to(ROOT)).replace("\\", "/"),
                         "features_sha256": sha256(features_path)},
        "atd_inputs": [{"path": str(p.relative_to(ROOT)).replace("\\", "/"), "sha256": sha256(p)} for p in inputs],
        "config": config, "code_sha256": sha256(Path(__file__)), "prediction_sha256": sha256(prediction),
        "label_counts": {label: int(np.sum(labels == i)) for i, label in enumerate(STATES)}})
    print("{}: wrote {}".format(NAME, str(prediction.relative_to(ROOT))))
    print("label counts:", {label: int(np.sum(labels == i)) for i, label in enumerate(STATES)})
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
