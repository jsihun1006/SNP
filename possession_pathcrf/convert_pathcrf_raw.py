"""Convert the raw PathCRF outputs (Colab or local) to our frame and player index.

    python possession_pathcrf/convert_pathcrf_raw.py J03WR9 trial140

Input  outputs/pathcrf_repo/<match>_<tag>_raw_{prev,next,micro}.parquet
       (index = DFL frame_id; columns = node names home_<jersey> / away_<jersey> / out_*)
Output outputs/pathcrf_repo/<match>_probs.csv.gz    frame, player, p_sender, p_receiver
       outputs/pathcrf_repo/<match>_decoded.csv.gz  frame, sender, receiver  (-1 = outside node)
"""

import os
import sys

import numpy as np
import pandas as pd

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
from run_pathcrf_repo import our_player_index, OUT  # noqa: E402


def main():
    match_id = sys.argv[1] if len(sys.argv) > 1 else "J03WR9"
    tag = sys.argv[2] if len(sys.argv) > 2 else "trial140"
    name_to_idx, row_of, n_rows, n_players = our_player_index(match_id)
    raw = {k: pd.read_parquet(os.path.join(OUT, "{}_{}_raw_{}.parquet".format(match_id, tag, k)))
           for k in ("prev", "next", "micro")}
    prev, nxt, micro = raw["prev"], raw["next"], raw["micro"]
    if "frame_id" in prev.columns:
        prev = prev.set_index("frame_id")
    rows = np.array([row_of.get(int(f), -1) for f in prev.index])
    keep = rows >= 0
    long = []
    for name in prev.columns:
        j = name_to_idx.get(name)
        if j is None:
            continue
        long.append(pd.DataFrame({"frame": rows[keep], "player": j, "p_sender": prev[name].to_numpy()[keep],
                                  "p_receiver": nxt[name].to_numpy()[keep]}))
    pd.concat(long).round(5).to_csv(os.path.join(OUT, "{}_probs.csv.gz".format(match_id)), index=False)
    drows = np.array([row_of.get(int(f), -1) for f in micro.index])
    ok = drows >= 0
    pd.DataFrame({"frame": drows[ok],
                  "sender": [name_to_idx.get(s, -1) for s in micro["edge_src"].to_numpy()[ok]],
                  "receiver": [name_to_idx.get(s, -1) for s in micro["edge_dst"].to_numpy()[ok]]}).to_csv(
        os.path.join(OUT, "{}_decoded.csv.gz".format(match_id)), index=False)
    unmapped = [c for c in prev.columns if c not in name_to_idx and not str(c).startswith("out")]
    print("frames mapped {}/{} (our rows {}), players {}, unmapped node names {}".format(
        int(keep.sum()), len(keep), n_rows, n_players, unmapped[:6]))


if __name__ == "__main__":
    main()
