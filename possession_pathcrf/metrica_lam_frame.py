"""Frame-dependent switch penalty on Metrica (models trained on DFL only; same grid as eval_lam_frame.py).

    python possession_pathcrf/metrica_lam_frame.py

Model: hybrid + contest model v2 (weight 2) + contest-outcome model, all trained on the 7 DFL matches;
25 fps on contested frames, 5 fps elsewhere; penalty 40 x M_CONTEST on contested frames, 40 x M_CALM elsewhere.
The 25-fps hybrid arrays of each Metrica game are cached (cache/metrica<g>_hybrid.npz) for later experiments.
Outputs: outputs/metrica_lam_frame/results.csv
"""

import os
import sys
import time

import numpy as np
import pandas as pd

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
import metrica_possession as mp  # noqa: E402
import eval_25fps_step as st  # noqa: E402
from eval_25fps import node_probs_25, hybrid_25  # noqa: E402
from eval_contest25 import contest_features, episodes, episode_rows, OUTCOME  # noqa: E402
from eval_contest_tune import rule_features, train_contest, TAU, LAM  # noqa: E402
from eval_lam_frame import M_CONTEST, M_CALM  # noqa: E402
from metrica_eval_possession import Boosters, score, CACHE  # noqa: E402

OUT = os.path.join(HERE, "outputs", "metrica_lam_frame")


def metrica_match(g, boosters):
    t = mp.load_game(g)
    match = t["match"]
    d = {"m": "metrica{}".format(g), "players": np.asarray(match.players, dtype=np.float64),
         "ball": np.asarray(match.ball, dtype=np.float64), "team": np.asarray(match.team), "period": t["period"],
         "match": match, "dead": t["dead"], "truth": np.where(t["team"] == 1, 1, 0).astype(np.int8),
         "near_change": t["near_change"], "challenge": t["challenge"]}
    d["alive"] = (d["period"] > 0) & ~d["dead"]
    path = os.path.join(CACHE, "metrica{}_hybrid.npz".format(g))
    if os.path.exists(path):
        z = np.load(path)
        d["present"], ls, lr, d["p_home"] = z["present"], z["ls"], z["lr"], z["p_home"]
    else:
        probs, present = node_probs_25(d, boosters)
        d["present"] = present
        ls, lr, d["p_home"] = hybrid_25(d, probs, present)
        np.savez_compressed(path, present=present, ls=ls.astype(np.float32), lr=lr.astype(np.float32), p_home=d["p_home"])
    d["contested"] = d["alive"] & (np.abs(2 * d["p_home"] - 1) < TAU)
    d["tt"] = np.full(len(d["period"]), "", dtype=object)
    return d, np.asarray(ls, dtype=np.float64), np.asarray(lr, dtype=np.float64)


def main():
    import lightgbm as lgb
    import dfl_load
    pd.set_option("display.width", 200)
    os.makedirs(OUT, exist_ok=True)
    t0 = time.time()
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
    del dfl
    print("DFL models ready {:.0f}s".format(time.time() - t0), flush=True)
    boosters = {"s": Boosters("s"), "r": Boosters("r")}
    rows = []
    for g in mp.GAMES:
        d, ls, lr = metrica_match(g, boosters)
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
        for mc in M_CONTEST:
            for mn in M_CALM:
                team, _ = st.decode(d, lc, lr, LAM, fine=d["contested"], lam_frame=np.where(d["contested"], mc, mn))
                r = {"game": g, "m_contest": mc, "m_calm": mn}
                r.update(score(d, team))
                rows.append(r)
        print("done game", g, "{:.0f}s".format(time.time() - t0), flush=True)
    t = pd.DataFrame(rows)
    t.to_csv(os.path.join(OUT, "results.csv"), index=False)
    print("\n== Metrica 3 games, mean (dead = truth) ==")
    print(t.groupby(["m_contest", "m_calm"])[["spellF1", "frame_acc", "frame_acc_challenge", "pred_spells",
                                              "true_spells"]].mean().round(4).to_string())


if __name__ == "__main__":
    main()
