"""How well does each possession method do on DFL? Several metrics, ball visible and ball hidden.

    python possession_pathcrf/eval_dfl2.py

Same leave-one-match-out node models as eval_dfl.py (probabilities cached in possession_pathcrf/cache/).
Ground truth: DFL ball_owning_team where in play, D where the DFL dead-ball label is dead.
Dead frames for every method: truth (so only possession is judged) and model (full pipeline).
Ball conditions for the ball-based baseline (PathCRF-lite never reads the ball):
  ball    original DFL ball
  hidden  ball deleted during dead balls, while it is above 3 m, and while a goalkeeper holds it
          (the situations where the provided tracking tends to lose it)
Metrics: spellF1 (official, ranking), team accuracy on in-play frames, macroF1 (H/A/D), possession-share
error (|predicted home share - true home share| over in-play frames), sender accuracy (PathCRF-lite).
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
from eval_dfl import prepare, rows, codes, load_model1  # noqa: E402

CACHE = os.path.join(HERE, "cache")
GRID = [(8.0, 8.0), (16.0, 16.0), (32.0, 32.0), (16.0, 8.0), (8.0, 16.0)]
CFG = {"fps": 25, "long_missing_seconds": 3.0, "opposite_team_cost_per_second": 2.0,
       "dead_when_observed_cost_per_second": 2.0, "dead_in_short_gap_cost_per_second": 1.0,
       "alive_in_long_gap_cost_per_second": 0.5, "team_switch_cost": 1.2, "dead_switch_cost": 0.8}


def metrics(gt, pred, alive_truth, team_pred, team_true):
    runs = pm.valid_runs(gt)
    cm = pm.scored_confusion(pm.confusion(gt, pred))
    ok = alive_truth
    return {"spellF1": pm.spell_scores(gt, pred, runs)["mean_f1"], "macroF1": pm.macro_f1(cm),
            "team_acc": float((team_pred[ok] == team_true[ok]).mean()),
            "share_err": abs(float((team_pred[ok] == pm.H).mean()) - float((team_true[ok] == pm.H).mean()))}


def node_probs(data, k):
    import lightgbm as lgb
    path = os.path.join(CACHE, "{}_probs.npz".format(data[k]["m"]))
    if os.path.exists(path):
        z = np.load(path)
        return {"s": z["s"], "r": z["r"]}
    params = dict(objective="binary", learning_rate=0.05, num_leaves=63, min_data_in_leaf=50,
                  feature_fraction=0.8, bagging_fraction=0.8, bagging_freq=1, verbose=-1, seed=7)
    train = [x for j, x in enumerate(data) if j != k]
    out = {}
    for which in ("s", "r"):
        Xs, ys = zip(*[rows(x, which) for x in train])
        booster = lgb.train(params, lgb.Dataset(np.concatenate(Xs), np.concatenate(ys)), num_boost_round=300)
        T, N, F = data[k]["X"].shape
        out[which] = booster.predict(data[k]["X"].reshape(-1, F)).reshape(T, N)
    os.makedirs(CACHE, exist_ok=True)
    np.savez_compressed(path, **out)
    return out


def nearest(players, ball, team):
    dist = np.linalg.norm(players - ball[:, None, :], axis=2)
    dist = np.where(np.isfinite(dist), dist, np.inf)
    who = dist.argmin(axis=1)
    seen = np.isfinite(dist[np.arange(len(dist)), who])
    return np.where(seen, team[who], -1)


def main():
    import dfl_load
    from eval_dfl_current import run as dead_pipeline
    from eval_keeper_hold_size import keeper_hold_spans
    pd.set_option("display.width", 220)
    t0 = time.time()
    data = [prepare(m) for m in dfl_load.MATCHES]
    m1 = load_model1()
    out = []
    for k, d in enumerate(data):
        probs = node_probs(data, k)
        import dfl_load as dl
        tracking, _, _ = dl.load_match(d["m"])
        z = tracking["ball_z"].to_numpy(np.float64)
        graded = d["period"] > 0
        hidden = d["ball"].copy()
        hide = (graded & d["dead_true"]) | (graded & ~d["dead_true"] & (z > 3.0))
        for a, b in keeper_hold_spans(d["match"], d["dead_true"]):
            hide[a:b] = True
        hidden[hide] = np.nan
        balls = {"ball": d["ball"], "hidden": hidden}
        dead_model = dead_pipeline(d["match"], d["score"]).astype(bool)
        gt = codes(d["gt_team"], d["dead_true"], d["period"])
        alive_truth_full = graded & ~d["dead_true"]
        alive_truth = np.insert(alive_truth_full, np.flatnonzero(np.diff(d["period"]) != 0) + 1, False)
        team_true = np.insert(d["gt_team"], np.flatnonzero(np.diff(d["period"]) != 0) + 1, pm.X)
        lab = d["s5"]
        okl = d["alive5"] & (lab >= 0)
        decoded = {}
        for dead_name, dead in (("truth", d["dead_true"]), ("model", dead_model)):
            alive = graded & ~dead
            alive5 = alive[d["idx"]]
            preds = {}
            for bname, ball in balls.items():
                near = nearest(d["players"], ball, d["team"])
                obs = np.where(near == 0, 0, np.where(near == 1, 1, -1)).astype(np.int8)
                lab_m1 = m1.viterbi(m1.constrained_costs(m1.emission_costs(obs, CFG), graded, dead), 1.2, 0.8)
                preds["baseline_model1/" + bname] = np.where(lab_m1 == 2, 0, lab_m1)
            for lk, lr in GRID:
                sender5 = pc.decode_match(probs["s"], probs["r"], d["present"], alive5, lk, lr)
                if dead_name == "truth":
                    decoded[(lk, lr)] = float((sender5[okl] == lab[okl]).mean())
                team5 = np.where(sender5 >= 0, d["team"][np.maximum(sender5, 0)], 0)
                preds["pathcrf_{:g}_{:g}".format(lk, lr)] = np.repeat(team5, pc.STEP)[:len(d["period"])]
            for name, team in preds.items():
                pred = codes(team, dead, d["period"])
                team_pred = np.insert(team, np.flatnonzero(np.diff(d["period"]) != 0) + 1, pm.X)
                row = {"match": d["m"], "dead": dead_name, "method": name}
                row.update(metrics(gt, pred, alive_truth, team_pred, team_true))
                if name.startswith("pathcrf"):
                    key = tuple(float(x) for x in name.split("_")[1:])
                    row["sender_acc"] = decoded[key]
                out.append(row)
        print("done", d["m"], "{:.0f}s".format(time.time() - t0), flush=True)
    t = pd.DataFrame(out)
    t.to_csv(os.path.join(CACHE, "eval_dfl2.csv"), index=False)
    for dead_name in ("truth", "model"):
        print("\n== mean over 7 matches, dead = {} ==".format(dead_name))
        print(t[t["dead"] == dead_name].groupby("method")[["spellF1", "team_acc", "macroF1", "share_err",
                                                            "sender_acc"]].mean().round(4).to_string())


if __name__ == "__main__":
    main()
