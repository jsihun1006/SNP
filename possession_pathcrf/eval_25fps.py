"""Hybrid PathCRF-lite + team model decoded at the full 25 fps (DFL, leave-one-match-out, ball available).

    python possession_pathcrf/eval_25fps.py

Node models (sender, receiver) and the team model are trained on 5-fps rows as before; pathcrf_lite.node_features
and team_features give the same feature meaning on any grid, so the models are applied to every 25-fps frame.
Viterbi then runs per frame. Per-frame scores accumulate 5x faster than at 5 fps, so the switch penalty is chosen
again on DFL from PENALTIES (for each test match, the best mean spellF1 on the other 6 matches).
Reported: spellF1 (DFL truth, touch labels), frame accuracy, predicted / true spell counts, and short-spell recall.
"""

import os
import sys
import time

import numpy as np
import pandas as pd

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
import pathcrf_lite as pc  # noqa: E402
import possession_metrics as pm  # noqa: E402
from eval_dfl import prepare, rows, codes  # noqa: E402
from eval_dfl3 import CONTROL_OBS_M, HYBRID_BONUS, nearest_with_dist  # noqa: E402
from eval_contest import touch_team  # noqa: E402

CACHE = os.path.join(HERE, "cache")
OUT = os.path.join(HERE, "outputs", "25fps")   # parameters, per-frame and per-player probabilities (CSV)


def save_params():
    import eval_dfl3
    rows = [("STEP_training_frames", pc.STEP, "node and team models are trained on every 5th frame"),
            ("CONTROL_M", pc.CONTROL_M, "training labels: ball within this of a player = control"),
            ("CONTROL_Z", pc.CONTROL_Z, "training labels: ball below this height"),
            ("BIG", pc.BIG, "penalty of an illegal edge transition (soft mask)"),
            ("CONTEXT_s", "-2,-1,+1,+2", "context offsets of the node features"),
            ("CONTROL_OBS_M", eval_dfl3.CONTROL_OBS_M, "hybrid: ball at someone's feet within this distance"),
            ("HYBRID_BONUS", eval_dfl3.HYBRID_BONUS, "hybrid: log-score bonus for that player"),
            ("PENALTIES", ",".join(str(x) for x in PENALTIES), "Viterbi kick/receive penalty grid at 25 fps"),
            ("CHUNK", CHUNK, "feature chunk length (frames)"), ("MARGIN", MARGIN, "context margin per chunk")]
    rows += [("node_lgbm_" + k, v, "") for k, v in NODE.items()] + [("node_rounds", 300, "")]
    rows += [("team_lgbm_" + k, v, "") for k, v in TEAM.items()] + [("team_rounds", 200, "")]
    os.makedirs(OUT, exist_ok=True)
    pd.DataFrame(rows, columns=["parameter", "value", "note"]).to_csv(os.path.join(OUT, "parameters.csv"), index=False)
PENALTIES = (20.0, 40.0, 80.0)
CHUNK, MARGIN = 4000, 60
NODE = dict(objective="binary", learning_rate=0.05, num_leaves=63, min_data_in_leaf=50, feature_fraction=0.8,
            bagging_fraction=0.8, bagging_freq=1, verbose=-1, seed=7)
TEAM = dict(objective="binary", learning_rate=0.05, num_leaves=31, min_data_in_leaf=50, verbose=-1, seed=7)


def node_boosters(data, k):
    import lightgbm as lgb
    out = {}
    for which in ("s", "r"):
        path = os.path.join(CACHE, "booster_{}_{}.txt".format(data[k]["m"], which))
        if os.path.exists(path):
            out[which] = lgb.Booster(model_file=path)
            continue
        Xs, ys = zip(*[rows(x, which) for j, x in enumerate(data) if j != k])
        b = lgb.train(NODE, lgb.Dataset(np.concatenate(Xs), np.concatenate(ys)), num_boost_round=300)
        b.save_model(path)
        out[which] = b
    return out


def node_probs_25(d, boosters):
    """Sender / receiver probabilities for every 25-fps frame (features in chunks with a context margin)."""
    T = len(d["period"])
    probs = {w: np.zeros((T, len(d["team"])), dtype=np.float32) for w in boosters}
    present = np.zeros((T, len(d["team"])), dtype=bool)
    for a in range(0, T, CHUNK):
        b = min(a + CHUNK, T)
        lo, hi = max(a - MARGIN, 0), min(b + MARGIN, T)
        X, _, pres, _, _ = pc.node_features(d["players"], d["team"], d["period"], d["match"].pitch,
                                            idx=np.arange(lo, hi))
        keep = slice(a - lo, a - lo + (b - a))
        F = X.shape[2]
        for w, booster in boosters.items():
            probs[w][a:b] = booster.predict(X[keep].reshape(-1, F)).reshape(b - a, -1)
        present[a:b] = pres[keep]
    return probs, present


def hybrid_25(d, probs, present):
    eps = 1e-6
    who, dmin = nearest_with_dist(d["players"], d["ball"], d["team"])
    ls = np.log(np.clip(probs["s"], eps, 1)).astype(np.float64)
    lr = np.log(np.clip(probs["r"], eps, 1)).astype(np.float64)
    r = np.flatnonzero(dmin <= CONTROL_OBS_M)
    ls[r, who[r]] += HYBRID_BONUS
    lr[r, who[r]] += HYBRID_BONUS
    m = np.where(present, ls, -np.inf)
    w = np.where(present, np.exp(ls - m.max(axis=1, keepdims=True)), 0)
    w = w / np.clip(w.sum(axis=1, keepdims=True), eps, None)
    p_home = (w * (d["team"] == 0)[None, :]).sum(axis=1)
    return ls, lr, p_home


def team_features(d, p_home):
    """Per 25-fps frame: the two teams around the ball (velocities over 0.2 s)."""
    p, ball = d["players"], d["ball"]
    lag = 5
    vel = np.full_like(p, np.nan)
    vel[lag:] = (p[lag:] - p[:-lag]) / 0.2
    bvel = np.full_like(ball, np.nan)
    bvel[lag:] = (ball[lag:] - ball[:-lag]) / 0.2
    to_ball = ball[:, None, :] - p
    dist = np.linalg.norm(to_ball, axis=2)
    with np.errstate(invalid="ignore", divide="ignore"):
        closing = np.nansum(vel * to_ball, axis=2) / dist
    cols = []
    for k in (0, 1):
        dk = np.where((d["team"] == k)[None, :] & np.isfinite(dist), dist, np.inf)
        srt = np.sort(dk, axis=1)
        i = dk.argmin(axis=1)
        cols += [np.where(np.isfinite(srt[:, 0]), srt[:, 0], 60), np.where(np.isfinite(srt[:, 1]), srt[:, 1], 60),
                 closing[np.arange(len(i)), i], (dk < 3).sum(axis=1)]
    cols += [cols[0] - cols[4], np.linalg.norm(bvel, axis=1), p_home]
    return np.stack(cols, axis=1).astype(np.float32)


def viterbi_25(ls, lr, present, alive, lam):
    sender = np.full(len(ls), -1)
    e = np.diff(np.r_[False, alive, False].astype(np.int8))
    for a, b in zip(np.flatnonzero(e == 1), np.flatnonzero(e == -1)):
        su, _ = pc.viterbi_edges(ls[a:b], lr[a:b], present[a:b], lam, lam)
        sender[a:b] = su
    return sender


def short_recall(gt, pred, limit_s=2.0):
    runs = pm.valid_runs(gt)
    gs, ps = pm.possession_spells(gt, runs), pm.possession_spells(pred, runs)
    best = {}
    for iou, i, j in pm.spell_overlaps(gs, ps):
        best[i] = max(best.get(i, 0), iou)
    short = [i for i, (a, b, _) in enumerate(gs) if (b - a) / 25 < limit_s]
    return float(np.mean([best.get(i, 0) >= 0.5 for i in short])) if short else np.nan


def main():
    import lightgbm as lgb
    import dfl_load
    pd.set_option("display.width", 200)
    t0 = time.time()
    save_params()
    data = [prepare(m) for m in dfl_load.MATCHES]
    prep = []
    for k, d in enumerate(data):
        probs, present = node_probs_25(d, node_boosters(data, k))
        ls, lr, p_home = hybrid_25(d, probs, present)
        tt = touch_team(d)
        alive = (d["period"] > 0) & ~d["dead_true"]
        prep.append(dict(ls=ls, lr=lr, present=present, alive=alive, X=team_features(d, p_home), tt=tt,
                         touch=np.where(tt >= 0, tt, 0), p_home=p_home, probs=probs))
        print("prepared", d["m"], "{:.0f}s".format(time.time() - t0), flush=True)
    for d in data:
        d["X"] = None        # 5-fps training cubes are no longer needed (boosters are trained and saved)
    out = []
    for k, d in enumerate(data):
        P = prep[k]
        Xs, ys = [], []
        for j, Q in enumerate(prep):
            if j != k:
                m = Q["alive"] & (Q["tt"] >= 0)
                m[np.arange(len(m)) % pc.STEP != 0] = False          # train on 5-fps rows, as before
                Xs.append(Q["X"][m])
                ys.append((Q["tt"][m] == 0).astype(np.int8))
        q = np.clip(lgb.train(TEAM, lgb.Dataset(np.concatenate(Xs), np.concatenate(ys)), 200).predict(P["X"]),
                    1e-4, 1 - 1e-4)
        home = d["team"] == 0
        ls = P["ls"].copy()
        r = np.flatnonzero(P["alive"])
        ls[np.ix_(r, np.flatnonzero(home))] += np.log(q[r])[:, None]
        ls[np.ix_(r, np.flatnonzero(~home))] += np.log(1 - q[r])[:, None]
        gt_d = codes(d["gt_team"], d["dead_true"], d["period"])
        gt_t = codes(P["touch"], d["dead_true"], d["period"])
        frames = pd.DataFrame({"frame": np.arange(len(d["period"])), "period": d["period"],
                               "alive": P["alive"].astype(np.int8), "truth_dfl": np.where(d["gt_team"] == 0, "H", "A"),
                               "truth_touch": np.where(P["tt"] < 0, "", np.where(P["tt"] == 0, "H", "A")),
                               "p_home_hybrid": np.round(P["p_home"], 4), "q_home_team_model": np.round(q, 4)})
        alive_idx = np.flatnonzero(P["alive"])
        t_i, p_i = np.nonzero(P["present"][alive_idx])
        t_i = alive_idx[t_i]
        pd.DataFrame({"frame": t_i, "player": p_i, "team": np.where(d["team"][p_i] == 0, "H", "A"),
                      "p_sender": np.round(P["probs"]["s"][t_i, p_i], 4),
                      "p_receiver": np.round(P["probs"]["r"][t_i, p_i], 4)}).to_csv(
            os.path.join(OUT, "player_probs_{}.csv.gz".format(d["m"])), index=False)
        for lam in PENALTIES:
            s = viterbi_25(ls, P["lr"], P["present"], P["alive"], lam)
            team = np.where(s >= 0, d["team"][np.maximum(s, 0)], 0)
            frames["sender_lam{:g}".format(lam)] = s
            frames["pred_lam{:g}".format(lam)] = np.where(~P["alive"], "D", np.where(team == 0, "H", "A"))
            pred = codes(team, d["dead_true"], d["period"])
            sd = pm.spell_scores(gt_d, pred, pm.valid_runs(gt_d))
            out.append({"match": d["m"], "lam": lam, "spellF1_dfl": sd["mean_f1"],
                        "spellF1_touch": pm.spell_scores(gt_t, pred, pm.valid_runs(gt_t))["mean_f1"],
                        "frame_acc": float((team[P["alive"]] == d["gt_team"][P["alive"]]).mean()),
                        "pred_spells": sd["n_pred"], "true_spells": sd["n_gt"],
                        "short_recall_dfl": short_recall(gt_d, pred)})
        frames.to_csv(os.path.join(OUT, "frames_{}.csv.gz".format(d["m"])), index=False)
        print("done", d["m"], "{:.0f}s".format(time.time() - t0), flush=True)
    t = pd.DataFrame(out)
    t.to_csv(os.path.join(OUT, "results_per_match.csv"), index=False)
    print("\n== 25 fps, mean over 7 test matches ==")
    print(t.groupby("lam")[["spellF1_dfl", "spellF1_touch", "frame_acc", "pred_spells", "true_spells",
                            "short_recall_dfl"]].mean().round(4).to_string())
    sc = t.pivot(index="match", columns="lam", values="spellF1_dfl")
    pick = [{"match": m, "lam": sc.drop(index=m).mean().idxmax(), "spellF1_dfl": sc.loc[m, sc.drop(index=m).mean().idxmax()]}
            for m in sc.index]
    pk = pd.DataFrame(pick)
    pk.to_csv(os.path.join(OUT, "chosen_penalty.csv"), index=False)
    print("\npenalty chosen on the other 6 matches:")
    print(pk.round(4).to_string(index=False))
    print("mean spellF1 (DFL truth) {:.4f}  (5-fps best was 0.5544)".format(pk["spellF1_dfl"].mean()))


if __name__ == "__main__":
    main()
