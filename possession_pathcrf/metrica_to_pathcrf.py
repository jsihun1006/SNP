"""Metrica sample games (public) -> the PathCRF repo's tracking input format, for inference in Colab.

    python possession_pathcrf/metrica_to_pathcrf.py

One parquet per game (frame_id = our row index + 10000, metres on a 105 x 68 pitch like the repo's DFL files):
  frame_id, period_id, timestamp, ball_state ("alive"/"dead" from the Metrica dead-ball label), ball_owning_team_id
  (empty), ball_x, ball_y, home_<k>_x, home_<k>_y, away_<k>_x, away_<k>_y   (k = our player index)
Velocities, accelerations, episodes and phases are computed in Colab with the repo's own functions.
Output: possession_pathcrf/colab/upload/metrica_tracking.zip (metrica1.parquet ... metrica3.parquet)
"""

import os
import sys
import zipfile

import numpy as np
import pandas as pd

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
import metrica_possession as mp  # noqa: E402

OFFSET = 10000
OUT_DIR = os.path.join(HERE, "colab", "upload")


def to_repo_frame(g):
    t = mp.load_game(g)
    match = t["match"]
    players = np.asarray(match.players, dtype=np.float64)
    ball = np.asarray(match.ball, dtype=np.float64)
    team = np.asarray(match.team)
    n = len(t["period"])
    cols = {"frame_id": np.arange(n) + OFFSET, "period_id": t["period"].astype(int),
            "timestamp": np.arange(n) / 25.0, "ball_state": np.where(t["dead"], "dead", "alive"),
            "ball_owning_team_id": pd.Series([None] * n, dtype=object), "ball_x": ball[:, 0], "ball_y": ball[:, 1]}
    for k in range(players.shape[1]):
        side = "home" if team[k] == 0 else "away"
        cols["{}_{}_x".format(side, k)] = players[:, k, 0]
        cols["{}_{}_y".format(side, k)] = players[:, k, 1]
    df = pd.DataFrame(cols)
    return df[df["period_id"] > 0].reset_index(drop=True)


def main():
    os.makedirs(OUT_DIR, exist_ok=True)
    path = os.path.join(OUT_DIR, "metrica_tracking.zip")
    tmp = os.path.join(OUT_DIR, "_metrica_tmp")
    os.makedirs(tmp, exist_ok=True)
    with zipfile.ZipFile(path, "w", zipfile.ZIP_DEFLATED) as z:
        for g in mp.GAMES:
            df = to_repo_frame(g)
            f = os.path.join(tmp, "metrica{}.parquet".format(g))
            df.to_parquet(f, index=False)
            z.write(f, arcname="metrica_tracking/metrica{}.parquet".format(g))
            n_players = sum(1 for c in df.columns if c.endswith("_x") and not c.startswith("ball"))
            print("metrica{}: {} frames, {} players, alive {:.1%}, x range {:.1f}..{:.1f}".format(
                g, len(df), n_players, (df["ball_state"] == "alive").mean(),
                np.nanmin(df.filter(like="_x").to_numpy()), np.nanmax(df.filter(like="_x").to_numpy())))
    print(path, round(os.path.getsize(path) / 1e6, 1), "MB")


if __name__ == "__main__":
    main()
