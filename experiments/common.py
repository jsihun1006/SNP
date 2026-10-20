"""Shared helpers for the VARcelona experiments: cached tracking arrays, graded mask, local scoring.

Tracking is parsed once from the raw ATD (numpy only, same format reading as starter/baseline.py)
and cached under data/.cache/ (gitignored with the rest of data/, so derived tracking never enters
the repo). Coordinates are converted to metres on load.
"""
import os
import sys

import numpy as np

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)
sys.path.insert(0, os.path.join(ROOT, "evaluation"))
sys.path.insert(0, os.path.join(ROOT, "starter"))
from games import GAMES, GAME, asset_paths, half_video_blocks  # noqa: E402
import possession_metrics as pm  # noqa: E402
from baseline import team_of_each_track  # noqa: E402

FPS = 25
LENGTH, WIDTH = GAMES[GAME]["pitch"]
N_FRAMES = GAMES[GAME]["n_video_frames"]
CACHE = os.path.join(ROOT, "data", ".cache", "tracking.npz")
GT_PATH = os.path.join(ROOT, "evaluation", "gt", "BAR-ATH_possession_gt_public.csv")
RESULTS = os.path.join(ROOT, "evaluation", "results")
HOME, AWAY = 0, 1


def _parse(raw_path, n_slots):
    pos = np.full((N_FRAMES, n_slots, 2), np.nan, np.float32)
    spd = np.full((N_FRAMES, n_slots), np.nan, np.float32)
    ball = np.full((N_FRAMES, 2), np.nan, np.float32)
    with open(raw_path) as fh:
        for line in fh:
            parts = line.rstrip("\n").split(":")
            if len(parts) != 3:
                continue
            f = int(parts[0]) - 1
            groups = parts[1].split(";")
            for s, g in enumerate(groups[:n_slots]):
                if g[0] == "N":
                    continue
                x, y, v = g.split(",")
                pos[f, s] = (float(x) * LENGTH, float(y) * WIDTH)
                spd[f, s] = float(v) if v[0] != "N" else np.nan
            b = parts[2]
            if b[0] != "N":
                x, y = b.split(",")[:2]
                ball[f] = (float(x) * LENGTH, float(y) * WIDTH)
    return pos, spd, ball


def load_tracking():
    """dict: pos (n,46,2) m, spd (n,46) m/s, ball (n,2) m, team (46,) 0=home 1=away, graded (n,) bool,
    half (n,) 0/1/2."""
    P = asset_paths()
    team = team_of_each_track(P["atd_meta"])
    if not os.path.isfile(CACHE):
        os.makedirs(os.path.dirname(CACHE), exist_ok=True)
        pos, spd, ball = _parse(P["atd_raw"], len(team))
        np.savez_compressed(CACHE, pos=pos, spd=spd, ball=ball)
    z = np.load(CACHE)
    graded = np.zeros(N_FRAMES, bool)
    half = np.zeros(N_FRAMES, np.int8)
    for h, a, b in half_video_blocks(GAME):
        graded[a:b + 1] = True
        half[a:b + 1] = h
    return dict(pos=z["pos"], spd=z["spd"], ball=z["ball"], team=team, graded=graded, half=half)


def load_gt():
    import pandas as pd
    return pd.read_csv(GT_PATH)["label"].to_numpy()


def score(labels, gt=None):
    """Local scorer on the public GT. labels: array of 'H','A','D','X' (len N_FRAMES)."""
    gt = load_gt() if gt is None else gt
    res = pm.evaluate(pm.label_codes(gt), pm.label_codes(np.asarray(labels)), bootstrap_n=0)
    return res


def write_csv(path, labels):
    with open(path, "w") as fh:
        fh.write("frame,label\n")
        fh.writelines("{},{}\n".format(i, l) for i, l in enumerate(labels))
