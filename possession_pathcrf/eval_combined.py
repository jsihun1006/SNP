"""Contest structure + last-touch rule, joined by one team-level decoder (DFL, 25 fps, ball available, dead = truth).

    python possession_pathcrf/eval_combined.py

Per in-play frame an observed team:
  not contested (|P(H) - P(A)| >= TAU, hybrid P)   the last-touch rule (R 1.0 m, real touches only)
  contested                                          the contest model (q from outputs/25fps_contest), > 0.5 = H
  OUTCOME_S after each contest episode               the contest-outcome model (retrained here, LOO), > 0.5 = H
Then the model1 cost decoder (opposite-team cost 2.0 per second, team-switch cost from SWITCH, dead fixed).
Variants: rule (rule everywhere), rule_contest (+ contest model), rule_contest_outcome (+ outcome model).
The switch cost is chosen per test match on the other 6 matches (DFL truth, spellF1).
Outputs: outputs/combined/ (parameters, per-match results, chosen settings, per-frame observations).
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
from eval_contest25 import contest_features, episodes, episode_rows, OUTCOME, OUTCOME_S  # noqa: E402

TAU = 0.3
SWITCH = (0.3, 0.6, 0.9, 1.2)
SRC = os.path.join(HERE, "outputs", "25fps_contest")
OUT = os.path.join(HERE, "outputs", "combined")
CFG = {"fps": 25, "long_missing_seconds": 3.0, "opposite_team_cost_per_second": 2.0,
       "dead_when_observed_cost_per_second": 2.0, "dead_in_short_gap_cost_per_second": 1.0,
       "alive_in_long_gap_cost_per_second": 0.5, "team_switch_cost": 1.2, "dead_switch_cost": 0.8}


def main():
    import lightgbm as lgb
    import dfl_load
    pd.set_option("display.width", 200)
    os.makedirs(OUT, exist_ok=True)
    m1 = load_model1()
    data = []
    for m in dfl_load.MATCHES:
        d = st.load(m, add_team=False)
        d["tt"] = np.asarray(d["tt"], dtype=object)
        c = pd.read_csv(os.path.join(SRC, "contest_probs_{}.csv.gz".format(m)))
        d["contested"] = (c["contested"].to_numpy() == 1) & d["alive"]
        d["q_contest"] = c["q_home_contest"].to_numpy(np.float64)
        d["rule"] = touches(d, 1.0, True)
        d["X"] = contest_features(d)
        data.append(d)
        print("loaded", m, flush=True)
    rows = []
    for k, d in enumerate(data):
        Xs, ys = [], []
        for j, o in enumerate(data):
            if j != k:
                X, y, _ = episode_rows(o, episodes(o["contested"], o["alive"]), o["rule"], o["X"])
                Xs.append(X[y >= 0])
                ys.append((y[y >= 0] == 0).astype(np.int8))
        booster = lgb.train(OUTCOME, lgb.Dataset(np.concatenate(Xs), np.concatenate(ys)), 200)
        X, y, spans = episode_rows(d, episodes(d["contested"], d["alive"]), d["rule"], d["X"])
        qo = booster.predict(X)
        obs = {"rule": d["rule"].copy()}
        o2 = d["rule"].copy()
        o2[d["contested"]] = np.where(d["q_contest"][d["contested"]] > 0.5, 0, 1)
        obs["rule_contest"] = o2
        o3 = o2.copy()
        for (b, e), qh in zip(spans, qo):
            o3[b:e] = 0 if qh > 0.5 else 1
        obs["rule_contest_outcome"] = o3
        graded = d["period"] > 0
        for name, team_obs in obs.items():
            ob = np.where(d["alive"], team_obs, -1).astype(np.int8)
            costs = m1.emission_costs(ob, CFG)
            for sw in SWITCH:
                lab = m1.viterbi(m1.constrained_costs(costs, graded, d["dead"]), sw, CFG["dead_switch_cost"])
                team = np.where(lab == 2, 0, lab)
                r = {"match": d["m"], "variant": name, "switch": sw}
                r.update(st.score(d, team))
                rows.append(r)
        pd.DataFrame({"frame": np.arange(len(o3)), "alive": d["alive"].astype(np.int8),
                      "contested": d["contested"].astype(np.int8), "obs_rule": d["rule"],
                      "obs_rule_contest": o2, "obs_rule_contest_outcome": o3}).to_csv(
            os.path.join(OUT, "observations_{}.csv.gz".format(d["m"])), index=False)
        print("done", d["m"], flush=True)
    t = pd.DataFrame(rows)
    t.to_csv(os.path.join(OUT, "results_per_match.csv"), index=False)
    pd.DataFrame([("TAU", TAU), ("SWITCH", ",".join(map(str, SWITCH))), ("OUTCOME_S", OUTCOME_S),
                  ("touch_radius_m", 1.0), ("opposite_team_cost_per_second", 2.0)],
                 columns=["parameter", "value"]).to_csv(os.path.join(OUT, "parameters.csv"), index=False)
    print("\n== mean over 7 matches by variant and switch cost ==")
    print(t.groupby(["variant", "switch"], sort=False)[["spellF1_dfl", "spellF1_touch", "frame_acc", "pred_spells",
                                                       "true_spells"]].mean().round(4).to_string())
    print("\n== switch cost chosen on the other 6 matches ==")
    chosen = []
    for name, tv in t.groupby("variant", sort=False):
        pk = st.choose(tv, ["switch"])
        chosen.append(pk.assign(variant=name))
        print("{:22s} spellF1 DFL {:.4f}  frame acc {:.4f}  chosen {}".format(
            name, pk["spellF1_dfl"].mean(), pk["frame_acc"].mean(), sorted(set(pk["switch"]))))
    pd.concat(chosen).to_csv(os.path.join(OUT, "chosen.csv"), index=False)
    print("reference: contest structure + outcome (edge decoder) 0.5956")


if __name__ == "__main__":
    main()
