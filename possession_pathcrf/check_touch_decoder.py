"""Last-touch rule + a switch-cost decoder (DFL, 25 fps, ball available, dead = truth).

    python possession_pathcrf/check_touch_decoder.py

Observation per in-play frame from check_touch_rule.touches(R = 1.0, real touches only):
  rule_every   the last-touch team on every frame
  rule_touch   the touching team only on touch frames, missing in between (the decoder bridges)
Decoder: the model1 three-state cost decoder (Viterbi) with dead frames fixed, team-switch cost from SWITCH
(per change; observation cost 2.0 per second for the opposite team). For each test match the switch cost is chosen
on the other 6 matches (DFL truth, spellF1).
"""

import os
import sys

import numpy as np
import pandas as pd

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
import eval_25fps_step as st  # noqa: E402
from check_touch_rule import touches  # noqa: E402
from eval_dfl import load_model1  # noqa: E402

SWITCH = (0.15, 0.3, 0.6, 0.9, 1.2, 3.0)   # 2026-11-17: lower values added, 1.2 was the smallest and was chosen everywhere
CFG = {"fps": 25, "long_missing_seconds": 3.0, "opposite_team_cost_per_second": 2.0,
       "dead_when_observed_cost_per_second": 2.0, "dead_in_short_gap_cost_per_second": 1.0,
       "alive_in_long_gap_cost_per_second": 0.5, "team_switch_cost": 1.2, "dead_switch_cost": 0.8}


def touch_only(d):
    """Team on real-touch frames (R 1.0), -1 elsewhere."""
    who, dmin = st.nearest_with_dist(d["players"], d["ball"], d["team"])
    every = touches(d, 1.0, True)
    near = np.isfinite(dmin) & (dmin <= 1.0)
    changed = np.r_[True, every[1:] != every[:-1]]
    return np.where(near | changed, every, -1)


def main():
    import dfl_load
    pd.set_option("display.width", 200)
    m1 = load_model1()
    rows = []
    for m in dfl_load.MATCHES:
        d = st.load(m, add_team=False)
        graded = d["period"] > 0
        obs_sets = {"rule_every": touches(d, 1.0, True), "rule_touch": touch_only(d)}
        r0 = {"match": m, "method": "rule_raw", "switch": 0.0}
        r0.update(st.score(d, obs_sets["rule_every"]))
        rows.append(r0)
        for name, obs_team in obs_sets.items():
            obs = np.where(d["alive"], obs_team, -1).astype(np.int8)
            costs = m1.emission_costs(obs, CFG)
            for sw in SWITCH:
                lab = m1.viterbi(m1.constrained_costs(costs, graded, d["dead"]), sw, CFG["dead_switch_cost"])
                team = np.where(lab == 2, 0, lab)
                r = {"match": m, "method": name, "switch": sw}
                r.update(st.score(d, team))
                rows.append(r)
        print("done", m, flush=True)
    t = pd.DataFrame(rows)
    os.makedirs(os.path.join(HERE, "outputs", "touch_rule"), exist_ok=True)
    t.to_csv(os.path.join(HERE, "outputs", "touch_rule", "results_per_match.csv"), index=False)
    print("\n== mean over 7 matches ==")
    print(t.groupby(["method", "switch"], sort=False)[["spellF1_dfl", "spellF1_touch", "frame_acc", "pred_spells",
                                                      "true_spells"]].mean().round(4).to_string())
    for name in obs_sets:
        pk = st.choose(t[t["method"] == name], ["switch"])
        print("{} with switch cost chosen on the other 6: spellF1 DFL {:.4f}, frame acc {:.4f}, chosen {}".format(
            name, pk["spellF1_dfl"].mean(), pk["frame_acc"].mean(), sorted(set(pk["switch"]))))


if __name__ == "__main__":
    main()
