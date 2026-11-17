"""Contest round 3: do non-contested errors come from a wrong contest decision carried on? And a contest-outcome
model that predicts who has the ball after each contest (DFL, leave-one-match-out, ball available).

    python possession_pathcrf/eval_contest3.py

Base = hybrid PathCRF-lite + contest model (tau 0.3, penalty 8; eval_contest.py).
Diagnosis: every run of wrong-team frames (DFL truth, in play) of the base is
  propagated    it starts inside a contested stretch or within CARRY_S after one ends
  independent   otherwise
Outcome model: contest episodes = runs of contested steps (gaps up to 2 steps merged). Target = team of the
sender in our touch labels OUTCOME_S after the episode ends. Features: the episode (length, mean P(H), mean
nearest-distance gap) and the OUTCOME_S after it (share of steps the ball is within CONTROL_OBS_M of a home /
away player, nearest distances per team at +0.4 s / +1 s / +2 s, base team at the end). Offline, so the
future is allowed. The outcome probability is added (log) to every player's sender score from the episode end
for OUTCOME_S, then Viterbi again.
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
from eval_dfl3 import CONTROL_OBS_M, LAM  # noqa: E402
from eval_contest import hybrid_scores, contest_features, touch_team, to25, spell, decode  # noqa: E402

TAU = 0.3
CARRY_S = 1.0       # an error run starting this soon after a contest counts as carried on
OUTCOME_S = 1.0     # the outcome is read this long after the episode
STEP_S = pc.STEP / 25


def runs(mask):
    e = np.diff(np.r_[False, mask, False].astype(np.int8))
    return list(zip(np.flatnonzero(e == 1), np.flatnonzero(e == -1)))


def episodes(contested5, alive5):
    out, cur = [], None
    for a, b in runs(contested5):
        if cur and a - cur[1] <= 2 and alive5[cur[1]:a].all():
            cur = (cur[0], b)
        else:
            if cur:
                out.append(cur)
            cur = (a, b)
    if cur:
        out.append(cur)
    return out


def ball_team_dist(d):
    """Per 5-fps step: nearest distance of each team's players to the ball (inf when unseen)."""
    p = d["players"][d["idx"]]
    ball = d["ball"][d["idx"]]
    dist = np.linalg.norm(p - ball[:, None, :], axis=2)
    out = []
    for k in (0, 1):
        dk = np.where((d["team"] == k)[None, :] & np.isfinite(dist), dist, np.inf)
        out.append(dk.min(axis=1))
    return np.stack(out, axis=1)


def episode_rows(d, P, eps, base5, td):
    n = len(base5)
    look = int(round(OUTCOME_S / STEP_S))
    feats, ys, ends = [], [], []
    near = td <= CONTROL_OBS_M
    gap = np.where(np.isfinite(td).all(axis=1), td[:, 0] - td[:, 1], 0.0)
    for a, b in eps:
        e = min(b + look, n - 1)
        f = [b - a, float(P["p_home"][a:b].mean()), float(gap[a:b].mean()),
             float(near[b:e, 0].mean()) if e > b else 0.0, float(near[b:e, 1].mean()) if e > b else 0.0]
        for off in (2, 5, 10):
            t = min(b + off, n - 1)
            f += [min(td[t, 0], 60.0), min(td[t, 1], 60.0)]
        f.append(float(base5[b - 1]) if b > 0 else 0.5)
        feats.append(f)
        y = P["y5"][e] if P["y5"][e] >= 0 else -1
        ys.append(y)
        ends.append((b, e))
    return np.array(feats, dtype=np.float32), np.array(ys), ends


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
                         y5=tt[d["idx"]], touch=np.where(tt >= 0, tt, 0), td=ball_team_dist(d)))
    p1 = dict(objective="binary", learning_rate=0.05, num_leaves=31, min_data_in_leaf=50, verbose=-1, seed=7)
    p2 = dict(objective="binary", learning_rate=0.05, num_leaves=15, min_data_in_leaf=20, verbose=-1, seed=7)
    # base = hybrid + contest model, per fold
    for k, d in enumerate(data):
        P = prep[k]
        Xs, ys = [], []
        for j, Q in enumerate(prep):
            if j != k:
                m = Q["alive5"] & (np.abs(2 * Q["p_home"] - 1) < TAU) & (Q["y5"] >= 0)
                Xs.append(Q["X"][m])
                ys.append((Q["y5"][m] == 0).astype(np.int8))
        q = np.clip(lgb.train(p1, lgb.Dataset(np.concatenate(Xs), np.concatenate(ys)), 200).predict(P["X"]),
                    1e-4, 1 - 1e-4)
        c5 = P["alive5"] & (np.abs(2 * P["p_home"] - 1) < TAU)
        home = d["team"] == 0
        ls = P["ls"].copy()
        rc = np.flatnonzero(c5)
        ls[np.ix_(rc, np.flatnonzero(home))] += np.log(q[rc])[:, None]
        ls[np.ix_(rc, np.flatnonzero(~home))] += np.log(1 - q[rc])[:, None]
        s5 = decode(ls, P["lr"], d, P["alive5"])
        P.update(ls_base=ls, c5=c5, base5=np.where(s5 >= 0, d["team"][np.maximum(s5, 0)], 0),
                 eps=episodes(c5, P["alive5"]))
        print("base", d["m"], flush=True)
    rows, diag = [], []
    for k, d in enumerate(data):
        P = prep[k]
        base25 = np.repeat(P["base5"], pc.STEP)[:len(d["period"])]
        alive = (d["period"] > 0) & ~d["dead_true"]
        contested = np.repeat(P["c5"], pc.STEP)[:len(d["period"])]
        after = contested.copy()
        carry = int(CARRY_S * 25)
        for a, b in runs(contested):
            after[b:b + carry] = True
        wrong = alive & (base25 != d["gt_team"])
        prop = ind = 0
        for a, b in runs(wrong):
            if after[a]:
                prop += b - a
            else:
                ind += b - a
        diag.append({"match": d["m"], "error_frames": int(wrong.sum()), "propagated_share": prop / max(prop + ind, 1),
                     "contested_share_of_errors": float((wrong & contested).sum() / max(wrong.sum(), 1))})
        # outcome model trained on the other matches' episodes
        Xs, ys = [], []
        for j, Q in enumerate(prep):
            if j != k:
                X, y, _ = episode_rows(data[j], Q, Q["eps"], Q["base5"], Q["td"])
                ok = y >= 0
                Xs.append(X[ok])
                ys.append((y[ok] == 0).astype(np.int8))
        booster = lgb.train(p2, lgb.Dataset(np.concatenate(Xs), np.concatenate(ys)), 200)
        X, y, ends = episode_rows(d, P, P["eps"], P["base5"], P["td"])
        qo = np.clip(booster.predict(X), 1e-4, 1 - 1e-4)
        ls = P["ls_base"].copy()
        home = d["team"] == 0
        for (b, e), qh in zip(ends, qo):
            ls[b:e, home] += np.log(qh)
            ls[b:e, ~home] += np.log(1 - qh)
        team_out = to25(d, decode(ls, P["lr"], d, P["alive5"]))
        ok = y >= 0
        rows.append({"match": d["m"], "episodes": len(ends),
                     "outcome_acc": float(((qo[ok] > .5) == (y[ok] == 0)).mean()) if ok.any() else np.nan,
                     "base_dfl": spell(d["gt_team"], base25, d), "outcome_dfl": spell(d["gt_team"], team_out, d),
                     "base_touch": spell(P["touch"], base25, d), "outcome_touch": spell(P["touch"], team_out, d)})
        print("done", d["m"], flush=True)
    g = pd.DataFrame(diag)
    print("\n== where the base's errors come from (DFL truth, in play) ==")
    print(g.round(3).to_string(index=False))
    print("mean propagated share {:.3f}, inside contests {:.3f}".format(g["propagated_share"].mean(),
                                                                       g["contested_share_of_errors"].mean()))
    t = pd.DataFrame(rows)
    print("\n== contest-outcome model ==")
    print(t.round(4).to_string(index=False))
    print("mean: base {:.4f} -> +outcome {:.4f} (DFL truth); base {:.4f} -> {:.4f} (touch labels); better {} / worse {}"
          .format(t["base_dfl"].mean(), t["outcome_dfl"].mean(), t["base_touch"].mean(), t["outcome_touch"].mean(),
                  int((t["outcome_dfl"] > t["base_dfl"]).sum()), int((t["outcome_dfl"] < t["base_dfl"]).sum())))


if __name__ == "__main__":
    main()
