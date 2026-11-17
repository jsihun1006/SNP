"""On non-contested frames, is 'team of the player nearest to the ball' as good as the hybrid? (frame level, DFL)

    python possession_pathcrf/check_nearest_rule.py

Non-contested = in play and |P(H) - P(A)| >= 0.3 (hybrid P from outputs/25fps). Compared per frame, no decoding:
  hybrid    P(H) > 0.5
  nearest   team of the nearest player to the ball (ball seen)
Split by ball state: at someone's feet (nearest within 1.5 m) or in flight / loose (nobody within 1.5 m).
Truths: DFL ball_owning_team and touch labels.
"""

import os
import sys

import numpy as np
import pandas as pd

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
import eval_25fps_step as st  # noqa: E402


def main():
    import dfl_load
    pd.set_option("display.width", 200)
    rows = []
    for m in dfl_load.MATCHES:
        d = st.load(m, add_team=False)
        who, dmin = st.nearest_with_dist(d["players"], d["ball"], d["team"])
        seen = np.isfinite(dmin)
        near_team = np.where(seen, d["team"][who], -1)
        hyb_team = np.where(d["p_home"] > 0.5, 0, 1)
        calm = d["alive"] & (np.abs(2 * d["p_home"] - 1) >= 0.3) & seen
        tt = np.asarray(d["tt"])
        for state, mask in (("all", calm), ("at_feet", calm & (dmin <= 1.5)), ("in_flight", calm & (dmin > 1.5))):
            for truth, ref, ok in (("dfl", d["gt"], np.ones(len(tt), bool)), ("touch", np.where(tt == "A", 1, 0),
                                                                              np.isin(tt, ["H", "A"]))):
                mm = mask & ok
                rows.append({"match": m, "ball": state, "truth": truth, "share_of_calm": float(mm.sum() / max(calm.sum(), 1)),
                             "hybrid_acc": float((hyb_team[mm] == ref[mm]).mean()),
                             "nearest_acc": float((near_team[mm] == ref[mm]).mean())})
        print("done", m, flush=True)
    t = pd.DataFrame(rows)
    print("\n== non-contested frames: frame accuracy (mean over 7) ==")
    print(t.groupby(["truth", "ball"])[["share_of_calm", "hybrid_acc", "nearest_acc"]].mean().round(4).to_string())


if __name__ == "__main__":
    main()
