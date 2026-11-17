"""Contest model round 2: larger tau (up to every step) and lower switch penalties, chosen on DFL.

    python possession_pathcrf/eval_contest2.py

Same pipeline as eval_contest.py. Grid: tau in (0.3, 0.4, 0.5, all) x Viterbi penalties in (8, 4, 2).
'all' applies the contest model to every in-play step. For each test match the (tau, penalty) pair with the best
mean spellF1 (DFL truth) on the other 6 matches is used. Hybrid without the contest model is reported per penalty.
"""

import os
import sys

import numpy as np
import pandas as pd

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
import pathcrf_lite as pc  # noqa: E402
from eval_dfl import prepare  # noqa: E402
from eval_dfl2 import node_probs  # noqa: E402
from eval_contest import hybrid_scores, contest_features, touch_team, to25, spell  # noqa: E402

TAUS = (0.3, 0.4, 0.5, 2.0)        # 2.0 = every step (|2p-1| is at most 1)
LAMS = ((8.0, 8.0), (4.0, 4.0), (2.0, 2.0))


def decode(ls, lr, d, alive5, lam):
    sender = np.full(len(ls), -1)
    edges = np.diff(np.r_[False, alive5, False].astype(np.int8))
    for a, b in zip(np.flatnonzero(edges == 1), np.flatnonzero(edges == -1)):
        su, _ = pc.viterbi_edges(ls[a:b], lr[a:b], d["present"][a:b], *lam)
        sender[a:b] = su
    return sender


def main():
    import lightgbm as lgb
    import dfl_load
    pd.set_option("display.width", 200)
    data = [prepare(m) for m in dfl_load.MATCHES]
    prep = []
    for k, d in enumerate(data):
        ls, lr, p_home = hybrid_scores(d, node_probs(data, k))
        alive5 = ((d["period"] > 0) & ~d["dead_true"])[d["idx"]]
        tt = touch_team(d)
        prep.append(dict(ls=ls, lr=lr, p_home=p_home, alive5=alive5, X=contest_features(d, p_home),
                         y5=tt[d["idx"]], touch=np.where(tt >= 0, tt, 0)))
    params = dict(objective="binary", learning_rate=0.05, num_leaves=31, min_data_in_leaf=50, verbose=-1, seed=7)
    rows = []
    for k, d in enumerate(data):
        P = prep[k]
        home = d["team"] == 0
        for lam in LAMS:
            team = to25(d, decode(P["ls"], P["lr"], d, P["alive5"], lam))
            rows.append({"match": d["m"], "tau": "none", "lam": lam[0], "spellF1_dfl": spell(d["gt_team"], team, d),
                         "spellF1_touch": spell(P["touch"], team, d)})
        for tau in TAUS:
            Xs, ys = [], []
            for j, Q in enumerate(prep):
                if j == k:
                    continue
                m = Q["alive5"] & (np.abs(2 * Q["p_home"] - 1) < tau) & (Q["y5"] >= 0)
                Xs.append(Q["X"][m])
                ys.append((Q["y5"][m] == 0).astype(np.int8))
            booster = lgb.train(params, lgb.Dataset(np.concatenate(Xs), np.concatenate(ys)), num_boost_round=200)
            q_home = np.clip(booster.predict(P["X"]), 1e-4, 1 - 1e-4)
            contested5 = P["alive5"] & (np.abs(2 * P["p_home"] - 1) < tau)
            rc = np.flatnonzero(contested5)
            ls = P["ls"].copy()
            ls[np.ix_(rc, np.flatnonzero(home))] += np.log(q_home[rc])[:, None]
            ls[np.ix_(rc, np.flatnonzero(~home))] += np.log(1 - q_home[rc])[:, None]
            for lam in LAMS:
                team = to25(d, decode(ls, P["lr"], d, P["alive5"], lam))
                rows.append({"match": d["m"], "tau": "all" if tau > 1 else tau, "lam": lam[0],
                             "spellF1_dfl": spell(d["gt_team"], team, d),
                             "spellF1_touch": spell(P["touch"], team, d)})
        print("done", d["m"], flush=True)
    t = pd.DataFrame(rows)
    t["key"] = t["tau"].astype(str) + "/" + t["lam"].astype(str)
    t.to_csv(os.path.join(HERE, "cache", "eval_contest2.csv"), index=False)
    print("\n== spellF1 (mean over 7 test matches) by tau / penalty ==")
    print(t.groupby(["tau", "lam"], sort=False)[["spellF1_dfl", "spellF1_touch"]].mean().round(4).to_string())
    sc = t.pivot(index="match", columns="key", values="spellF1_dfl")
    pick = []
    for m in sc.index:
        best = sc.drop(index=m).mean().idxmax()
        pick.append({"match": m, "chosen": best, "spellF1_dfl": sc.loc[m, best], "hybrid_8": sc.loc[m, "none/8.0"]})
    pk = pd.DataFrame(pick)
    print("\n== (tau, penalty) chosen on the other 6 matches ==")
    print(pk.round(4).to_string(index=False))
    print("mean: hybrid (penalty 8) {:.4f} -> chosen {:.4f}; better {} / worse {}".format(
        pk["hybrid_8"].mean(), pk["spellF1_dfl"].mean(), int((pk["spellF1_dfl"] > pk["hybrid_8"]).sum()),
        int((pk["spellF1_dfl"] < pk["hybrid_8"]).sum())))


if __name__ == "__main__":
    main()
