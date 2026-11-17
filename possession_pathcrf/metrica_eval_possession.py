"""Possession models trained on DFL only, scored on the Metrica sample events (games 1-3, an independent check).

    python possession_pathcrf/metrica_eval_possession.py

Truth: metrica_possession.load_game (event team carried to the next event, dead from the Metrica dead-ball label;
dead frames are fixed to the truth so only possession is judged).
Methods (25 fps on contested frames, 5 fps elsewhere for the edge decoder):
  baseline_model1   nearest player's team + model1 decoder (current submission logic)
  rule_decoder      last-touch rule (R 1.0 m, real touches) + model1 decoder, switch 0.9
  hybrid            PathCRF-lite hybrid (node models = mean of the 7 DFL fold boosters), penalty 40
  contest_v2        hybrid + contest model v2 (weight 2) + contest-outcome model, all trained on the 7 DFL matches
Metrics: spellF1 (official), frame accuracy on in-play frames, frame accuracy without the frames near a truth change
(+-0.4 s, hand-tagged event frames), frame accuracy on CHALLENGE frames, predicted / true spell counts.
Outputs: outputs/metrica/ (results.csv, parameters.csv).
"""

import os
import sys
import time

import numpy as np
import pandas as pd

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
import metrica_possession as mp  # noqa: E402
import pathcrf_lite as pc  # noqa: E402
import possession_metrics as pm  # noqa: E402
import eval_25fps_step as st  # noqa: E402
from eval_25fps import node_probs_25, hybrid_25  # noqa: E402
from eval_contest25 import contest_features, episodes, episode_rows, OUTCOME  # noqa: E402
from eval_contest_tune import rule_features, train_contest, TAU, LAM  # noqa: E402
from check_touch_rule import touches  # noqa: E402
from eval_dfl import codes, load_model1  # noqa: E402

OUT = os.path.join(HERE, "outputs", "metrica")
CACHE = os.path.join(HERE, "cache")
RULE_SWITCH = 0.9
CFG = {"fps": 25, "long_missing_seconds": 3.0, "opposite_team_cost_per_second": 2.0,
       "dead_when_observed_cost_per_second": 2.0, "dead_in_short_gap_cost_per_second": 1.0,
       "alive_in_long_gap_cost_per_second": 0.5, "team_switch_cost": 1.2, "dead_switch_cost": 0.8}


class Boosters:
    """Mean probability of the 7 DFL fold node models (each trained on 6 DFL matches)."""

    def __init__(self, which):
        import lightgbm as lgb
        import dfl_load
        self.models = [lgb.Booster(model_file=os.path.join(CACHE, "booster_{}_{}.txt".format(m, which)))
                       for m in dfl_load.MATCHES]

    def predict(self, X):
        return np.mean([b.predict(X) for b in self.models], axis=0)


def score(d, team):
    pred = codes(team, d["dead"], d["period"])
    gt = codes(d["truth"], d["dead"], d["period"])
    sc = pm.spell_scores(gt, pred, pm.valid_runs(gt))
    alive = d["alive"]
    return {"spellF1": sc["mean_f1"], "pred_spells": sc["n_pred"], "true_spells": sc["n_gt"],
            "frame_acc": float((team[alive] == d["truth"][alive]).mean()),
            "frame_acc_no_change": float((team[alive & ~d["near_change"]] == d["truth"][alive & ~d["near_change"]]).mean()),
            "frame_acc_challenge": float((team[alive & d["challenge"]] == d["truth"][alive & d["challenge"]]).mean())}


def main():
    import lightgbm as lgb
    import dfl_load
    pd.set_option("display.width", 200)
    os.makedirs(OUT, exist_ok=True)
    t0 = time.time()
    # DFL training data for the contest and outcome models
    dfl = []
    for m in dfl_load.MATCHES:
        d = st.load(m, add_team=False)
        d["tt"] = np.asarray(d["tt"], dtype=object)
        d["contested"] = d["alive"] & (np.abs(2 * d["p_home"] - 1) < TAU)
        d["X1"] = contest_features(d)
        d["rule"], extra = rule_features(d)
        d["X2"] = np.concatenate([d["X1"], extra], axis=1)
        dfl.append(d)
    contest = train_contest(dfl, -1, "X2")
    Xs, ys = [], []
    for o in dfl:
        X, y, _ = episode_rows(o, episodes(o["contested"], o["alive"]), o["rule"], o["X2"])
        Xs.append(X[y >= 0])
        ys.append((y[y >= 0] == 0).astype(np.int8))
    outcome = lgb.train(OUTCOME, lgb.Dataset(np.concatenate(Xs), np.concatenate(ys)), 200)
    print("DFL models ready {:.0f}s".format(time.time() - t0), flush=True)
    boosters = {"s": Boosters("s"), "r": Boosters("r")}
    m1 = load_model1()
    rows = []
    for g in mp.GAMES:
        t = mp.load_game(g)
        match = t["match"]
        d = {"m": "metrica{}".format(g), "players": np.asarray(match.players, dtype=np.float64),
             "ball": np.asarray(match.ball, dtype=np.float64), "team": np.asarray(match.team), "period": t["period"],
             "match": match, "dead": t["dead"], "truth": np.where(t["team"] == 1, 1, 0).astype(np.int8),
             "near_change": t["near_change"], "challenge": t["challenge"]}
        d["alive"] = (d["period"] > 0) & ~d["dead"]
        probs, present = node_probs_25(d, boosters)
        d["present"] = present
        ls, lr, d["p_home"] = hybrid_25(d, probs, present)
        d["contested"] = d["alive"] & (np.abs(2 * d["p_home"] - 1) < TAU)
        d["tt"] = np.full(len(d["period"]), "", dtype=object)
        graded = d["period"] > 0
        results = {}
        # baseline: nearest team + model1 decoder
        who, dmin = st.nearest_with_dist(d["players"], d["ball"], d["team"])
        near = np.where(np.isfinite(dmin), d["team"][who], -1)
        obs = np.where(d["alive"], near, -1).astype(np.int8)
        lab = m1.viterbi(m1.constrained_costs(m1.emission_costs(obs, CFG), graded, d["dead"]), 1.2, 0.8)
        results["baseline_model1"] = np.where(lab == 2, 0, lab)
        # last-touch rule + decoder
        rule = touches(d, 1.0, True)
        obs = np.where(d["alive"], rule, -1).astype(np.int8)
        lab = m1.viterbi(m1.constrained_costs(m1.emission_costs(obs, CFG), graded, d["dead"]), RULE_SWITCH, 0.8)
        results["rule_decoder"] = np.where(lab == 2, 0, lab)
        # hybrid
        results["hybrid"], _ = st.decode(d, ls, lr, LAM, fine=d["contested"])
        # contest v2 + outcome
        X1 = contest_features(d)
        d["rule"], extra = rule_features(d)
        X2 = np.concatenate([X1, extra], axis=1)
        q = np.clip(contest.predict(X2), 1e-4, 1 - 1e-4)
        cf = np.flatnonzero(d["contested"])
        home = d["team"] == 0
        lc = ls.copy()
        lc[np.ix_(cf, np.flatnonzero(home))] += 2.0 * np.log(q[cf])[:, None]
        lc[np.ix_(cf, np.flatnonzero(~home))] += 2.0 * np.log(1 - q[cf])[:, None]
        X, _, spans = episode_rows(d, episodes(d["contested"], d["alive"]), d["rule"], X2)
        for (b, e), qh in zip(spans, np.clip(outcome.predict(X), 1e-4, 1 - 1e-4)):
            lc[b:e, home] += np.log(qh)
            lc[b:e, ~home] += np.log(1 - qh)
        results["contest_v2"], _ = st.decode(d, lc, lr, LAM, fine=d["contested"])
        for name, team in results.items():
            r = {"game": g, "method": name}
            r.update(score(d, team))
            rows.append(r)
        print("done game", g, "{:.0f}s".format(time.time() - t0), flush=True)
    res = pd.DataFrame(rows)
    res.to_csv(os.path.join(OUT, "results.csv"), index=False)
    pd.DataFrame([("TAU", TAU), ("LAM", LAM), ("RULE_SWITCH", RULE_SWITCH), ("contest_weight", 2.0),
                  ("node_models", "mean of 7 DFL fold boosters"), ("contest_outcome_training", "all 7 DFL matches"),
                  ("challenge_pad_frames", mp.CHALLENGE_PAD), ("change_tol_frames", mp.CHANGE_TOL)],
                 columns=["parameter", "value"]).to_csv(os.path.join(OUT, "parameters.csv"), index=False)
    print("\n== Metrica 3 games, mean (dead = truth) ==")
    print(res.groupby("method", sort=False)[["spellF1", "frame_acc", "frame_acc_no_change", "frame_acc_challenge",
                                             "pred_spells", "true_spells"]].mean().round(4).to_string())


if __name__ == "__main__":
    main()
