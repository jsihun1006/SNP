"""Contested-possession model on top of hybrid PathCRF-lite (DFL, leave-one-match-out, ball available).

    python possession_pathcrf/eval_contest.py

1. Per 5-fps step, team probability from the hybrid node scores: P(team) = sum over that team's players of the
   normalised sender score. Contested = |P(H) - P(A)| < tau, tau in TAUS (chosen on DFL, see 5).
2. Diagnosis: share of in-play steps that are contested, and share of team errors that fall in them.
3. Contest model (LightGBM, trained on contested steps of the other 6 matches): target = team of the sender in
   our touch labels (the competition definition: the last team to touch the ball keeps it until the other touches).
   Features: for each team nearest and 2nd-nearest distance to the ball, closing speed of the nearest player
   towards the ball, players within 3 m, ball speed, P(H) of the hybrid.
4. Combination: on contested steps add log q(team of u) to every player's sender score, Viterbi again.
5. tau selection: for each test match, the tau with the best mean spellF1 on the other 6 matches.
Ground truths for spellF1: DFL ball_owning_team and our touch labels. Dead frames = DFL truth.
"""

import os
import sys

import numpy as np
import pandas as pd

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
import pathcrf_lite as pc  # noqa: E402
import possession_metrics as pm  # noqa: E402
from eval_dfl import prepare, codes  # noqa: E402
from eval_dfl2 import node_probs  # noqa: E402
from eval_dfl3 import CONTROL_OBS_M, HYBRID_BONUS, LAM, nearest_with_dist  # noqa: E402

TAUS = (0.1, 0.2, 0.3)
FPS5 = 25 / pc.STEP


def hybrid_scores(d, probs):
    """Hybrid node log-scores at 5 fps and the step-wise team probability P(H)."""
    eps = 1e-6
    who, dmin = nearest_with_dist(d["players"], d["ball"], d["team"])
    who5, near5 = who[d["idx"]], (dmin <= CONTROL_OBS_M)[d["idx"]]
    ls = np.log(np.clip(probs["s"], eps, 1))
    lr = np.log(np.clip(probs["r"], eps, 1))
    rows = np.flatnonzero(near5)
    ls[rows, who5[rows]] += HYBRID_BONUS
    lr[rows, who5[rows]] += HYBRID_BONUS
    w = np.where(d["present"], np.exp(ls - np.nanmax(np.where(d["present"], ls, -np.inf), axis=1, keepdims=True)), 0)
    w = w / np.clip(w.sum(axis=1, keepdims=True), eps, None)
    p_home = (w * (d["team"] == 0)[None, :]).sum(axis=1)
    return ls, lr, p_home


def contest_features(d, p_home):
    """Per 5-fps step features of the two teams around the ball (ball available)."""
    p = d["players"][d["idx"]]
    ball = d["ball"][d["idx"]]
    vel = np.full_like(p, np.nan)
    vel[1:] = (p[1:] - p[:-1]) * FPS5
    bvel = np.full_like(ball, np.nan)
    bvel[1:] = (ball[1:] - ball[:-1]) * FPS5
    to_ball = ball[:, None, :] - p
    dist = np.linalg.norm(to_ball, axis=2)
    with np.errstate(invalid="ignore", divide="ignore"):
        closing = np.nansum(vel * to_ball, axis=2) / dist          # speed towards the ball
    cols = {}
    for k, name in ((0, "home"), (1, "away")):
        dk = np.where((d["team"] == k)[None, :] & np.isfinite(dist), dist, np.inf)
        srt = np.sort(dk, axis=1)
        idx = dk.argmin(axis=1)
        cols[name + "_d1"] = np.where(np.isfinite(srt[:, 0]), srt[:, 0], 60)
        cols[name + "_d2"] = np.where(np.isfinite(srt[:, 1]), srt[:, 1], 60)
        cols[name + "_closing"] = closing[np.arange(len(idx)), idx]
        cols[name + "_n3"] = (dk < 3).sum(axis=1)
    cols["d_diff"] = cols["home_d1"] - cols["away_d1"]
    cols["ball_speed"] = np.linalg.norm(bvel, axis=1)
    cols["p_home"] = p_home
    return pd.DataFrame(cols).to_numpy(np.float32)


def touch_team(d):
    s = d["s25"]
    return np.where(s >= 0, d["team"][np.maximum(s, 0)], -1)


def decode(ls, lr, d, alive5):
    sender = np.full(len(ls), -1)
    edges = np.diff(np.r_[False, alive5, False].astype(np.int8))
    for a, b in zip(np.flatnonzero(edges == 1), np.flatnonzero(edges == -1)):
        su, _ = pc.viterbi_edges(ls[a:b], lr[a:b], d["present"][a:b], *LAM)
        sender[a:b] = su
    return sender


def to25(d, sender5):
    return np.repeat(np.where(sender5 >= 0, d["team"][np.maximum(sender5, 0)], 0), pc.STEP)[:len(d["period"])]


def spell(gt_team, team, d):
    gt = codes(gt_team, d["dead_true"], d["period"])
    return pm.spell_scores(gt, codes(team, d["dead_true"], d["period"]), pm.valid_runs(gt))["mean_f1"]


def main():
    import lightgbm as lgb
    import dfl_load
    pd.set_option("display.width", 200)
    data = [prepare(m) for m in dfl_load.MATCHES]
    prep = []
    for k, d in enumerate(data):
        probs = node_probs(data, k)
        ls, lr, p_home = hybrid_scores(d, probs)
        alive5 = ((d["period"] > 0) & ~d["dead_true"])[d["idx"]]
        tt = touch_team(d)
        prep.append(dict(ls=ls, lr=lr, p_home=p_home, alive5=alive5, X=contest_features(d, p_home),
                         y5=tt[d["idx"]], touch=np.where(tt >= 0, tt, 0)))
    params = dict(objective="binary", learning_rate=0.05, num_leaves=31, min_data_in_leaf=50, verbose=-1, seed=7)
    rows, diag = [], []
    for k, d in enumerate(data):
        P = prep[k]
        base_team = to25(d, decode(P["ls"], P["lr"], d, P["alive5"]))
        alive = (d["period"] > 0) & ~d["dead_true"]
        err = base_team != d["gt_team"]
        rows.append({"match": d["m"], "tau": "none", "spellF1_dfl": spell(d["gt_team"], base_team, d),
                     "spellF1_touch": spell(P["touch"], base_team, d)})
        for tau in TAUS:
            contested5 = P["alive5"] & (np.abs(2 * P["p_home"] - 1) < tau)
            contested = np.repeat(contested5, pc.STEP)[:len(d["period"])]
            diag.append({"match": d["m"], "tau": tau, "contested_share": float(contested[alive].mean()),
                         "errors_in_contested": float((err & contested)[alive].sum() / max(err[alive].sum(), 1)),
                         "err_rate_contested": float(err[alive & contested].mean()) if (alive & contested).any() else np.nan,
                         "err_rate_rest": float(err[alive & ~contested].mean())})
            Xs, ys = [], []
            for j, Q in enumerate(prep):
                if j == k:
                    continue
                m = Q["alive5"] & (np.abs(2 * Q["p_home"] - 1) < tau) & (Q["y5"] >= 0)
                Xs.append(Q["X"][m])
                ys.append((Q["y5"][m] == 0).astype(np.int8))
            booster = lgb.train(params, lgb.Dataset(np.concatenate(Xs), np.concatenate(ys)), num_boost_round=200)
            q_home = np.clip(booster.predict(P["X"]), 1e-4, 1 - 1e-4)
            ls = P["ls"].copy()
            rows_c = np.flatnonzero(contested5)
            home = d["team"] == 0
            ls[np.ix_(rows_c, np.flatnonzero(home))] += np.log(q_home[rows_c])[:, None]
            ls[np.ix_(rows_c, np.flatnonzero(~home))] += np.log(1 - q_home[rows_c])[:, None]
            team = to25(d, decode(ls, P["lr"], d, P["alive5"]))
            rows.append({"match": d["m"], "tau": tau, "spellF1_dfl": spell(d["gt_team"], team, d),
                         "spellF1_touch": spell(P["touch"], team, d)})
        print("done", d["m"], flush=True)
    t = pd.DataFrame(rows)
    g = pd.DataFrame(diag)
    print("\n== diagnosis: where do the hybrid's team errors fall? (DFL truth) ==")
    print(g.groupby("tau")[["contested_share", "errors_in_contested", "err_rate_contested", "err_rate_rest"]]
          .mean().round(3).to_string())
    print("\n== spellF1 by tau (mean over 7 test matches) ==")
    print(t.groupby("tau", sort=False)[["spellF1_dfl", "spellF1_touch"]].mean().round(4).to_string())
    pick = []
    sc = t[t["tau"] != "none"].pivot(index="match", columns="tau", values="spellF1_dfl")
    for m in sc.index:
        best = sc.drop(index=m).mean().idxmax()
        pick.append({"match": m, "tau": best, "spellF1_dfl": sc.loc[m, best],
                     "base": float(t[(t["match"] == m) & (t["tau"] == "none")]["spellF1_dfl"].iloc[0])})
    pk = pd.DataFrame(pick)
    print("\n== tau chosen on the other 6 matches ==")
    print(pk.round(4).to_string(index=False))
    print("mean: hybrid {:.4f} -> hybrid + contest model {:.4f}".format(pk["base"].mean(), pk["spellF1_dfl"].mean()))


if __name__ == "__main__":
    main()
