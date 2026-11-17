"""DFL leave-one-match-out evaluation of PathCRF-lite against simple baselines (official spellF1).

    python possession_pathcrf/eval_dfl.py

Ground truth per 25-fps frame: D where the DFL dead-ball label is dead, else DFL ball_owning_team (home = H).
Methods (dead frames always from the given dead source):
  nearest_raw      team of the player nearest to the ball (uses the ball), no smoothing
  nearest_model1   same observation + the fixed-cost decoder of model1_dead_ball (uses the ball)
  touch_oracle     team of the sender from our own ball-control labels (label-definition upper bound)
  pathcrf_k_r      PathCRF-lite, no ball, Viterbi with kick / receive penalties k, r
Dead sources: truth (isolates possession) and model (the full dead-ball pipeline, OOF scores).
"""

import importlib.util
import os
import sys
import time

import numpy as np
import pandas as pd

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
import pathcrf_lite as pc  # noqa: E402
import features as fx  # noqa: E402
import possession_metrics as pm  # noqa: E402

GRID = [(1.0, 1.0), (2.0, 2.0), (4.0, 4.0), (8.0, 8.0), (4.0, 2.0), (2.0, 4.0)]
P2 = os.path.join(os.path.dirname(pc.ROOT), "SNP_submission", "model", "phase2_possession", "model1_dead_ball",
                  "possession.py")


def load_model1():
    spec = importlib.util.spec_from_file_location("_m1", P2)
    mod = importlib.util.module_from_spec(spec)
    sys.argv_backup = sys.argv
    spec.loader.exec_module(mod)
    return mod


def codes(team_code, dead, period):
    """H/A/D codes with an X between halves and X outside graded periods."""
    c = np.where(dead, pm.D, team_code).astype(np.int8)
    c[period <= 0] = pm.X
    breaks = np.flatnonzero(np.diff(period) != 0) + 1
    return np.insert(c, breaks, pm.X)


def spell(gt, pred):
    return pm.spell_scores(gt, pred, pm.valid_runs(gt))["mean_f1"]


def prepare(m):
    import dfl_load
    import train_dfl
    tracking, _, meta = dfl_load.load_match(m)
    raw = fx.match_from_dfl(tracking, meta)
    oof = pd.read_parquet(os.path.join(train_dfl.oof_dir(), "{}_sim.parquet".format(m)))
    period = oof["period"].to_numpy().astype(np.int64)
    dead_true = oof["dead"].to_numpy().astype(bool)
    alive = (period > 0) & ~dead_true
    players = np.asarray(raw.players, dtype=np.float64)
    ball = np.asarray(raw.ball, dtype=np.float64)
    owner = tracking["ball_owning_team"].to_numpy()
    gt_team = np.where(owner == "away", pm.A, pm.H)
    ctrl = pc.control_labels(players, ball, tracking["ball_z"].to_numpy(np.float64), alive)
    s, r = pc.edge_labels(ctrl, alive)
    X, names, present, idx, per5 = pc.node_features(players, raw.team, period, raw.pitch)
    match = fx.Match(players=raw.players, team=raw.team, ball=raw.ball, period=period, pitch=raw.pitch)
    return dict(m=m, X=X, names=names, present=present, idx=idx, alive5=alive[idx], s5=s[idx], r5=r[idx],
                s25=s, team=raw.team, period=period, dead_true=dead_true, gt_team=gt_team, ball=ball,
                players=players, score=oof["score"].to_numpy(np.float64), match=match)


def rows(d, which):
    """Training rows: present players at alive steps with a known label."""
    lab = d["s5"] if which == "s" else d["r5"]
    ok = d["alive5"] & (lab >= 0)
    t_idx, p_idx = np.nonzero(d["present"] & ok[:, None])
    y = (p_idx == lab[t_idx]).astype(np.int8)
    return d["X"][t_idx, p_idx], y


def nearest_team(d):
    dist = np.linalg.norm(d["players"] - d["ball"][:, None, :], axis=2)
    dist = np.where(np.isfinite(dist), dist, np.inf)
    who = dist.argmin(axis=1)
    seen = np.isfinite(dist[np.arange(len(dist)), who])
    return np.where(seen, d["team"][who], -1)


def main():
    import lightgbm as lgb
    import dfl_load
    from eval_dfl_current import run as dead_pipeline
    pd.set_option("display.width", 200)
    t0 = time.time()
    data = [prepare(m) for m in dfl_load.MATCHES]
    print("prepared {} matches in {:.0f}s, {} features".format(len(data), time.time() - t0, data[0]["X"].shape[2]),
          flush=True)
    m1 = load_model1()
    cfg = {"fps": 25, "long_missing_seconds": 3.0, "opposite_team_cost_per_second": 2.0,
           "dead_when_observed_cost_per_second": 2.0, "dead_in_short_gap_cost_per_second": 1.0,
           "alive_in_long_gap_cost_per_second": 0.5, "team_switch_cost": 1.2, "dead_switch_cost": 0.8}
    params = dict(objective="binary", learning_rate=0.05, num_leaves=63, min_data_in_leaf=50,
                  feature_fraction=0.8, bagging_fraction=0.8, bagging_freq=1, verbose=-1, seed=7)
    out = []
    for k, d in enumerate(data):
        train = [x for j, x in enumerate(data) if j != k]
        probs = {}
        for which in ("s", "r"):
            Xs, ys = zip(*[rows(x, which) for x in train])
            booster = lgb.train(params, lgb.Dataset(np.concatenate(Xs), np.concatenate(ys)), num_boost_round=300)
            T, N, F = d["X"].shape
            probs[which] = booster.predict(d["X"].reshape(-1, F)).reshape(T, N)
        dead_model = dead_pipeline(d["match"], d["score"]).astype(bool)
        gt = codes(d["gt_team"], d["dead_true"], d["period"])
        near = nearest_team(d)
        # sender accuracy at labelled alive steps (diagnostic)
        lab = d["s5"]
        okl = d["alive5"] & (lab >= 0)
        for dead_name, dead in (("truth", d["dead_true"]), ("model", dead_model)):
            alive = (d["period"] > 0) & ~dead
            res = {"match": d["m"], "dead": dead_name}
            # baselines
            raw_team = np.where(near >= 0, near, 0)
            res["nearest_raw"] = spell(gt, codes(raw_team, dead, d["period"]))
            obs = np.where(near == 0, 0, np.where(near == 1, 1, -1)).astype(np.int8)
            lab_m1 = m1.viterbi(m1.constrained_costs(m1.emission_costs(obs, cfg), d["period"] > 0, dead), 1.2, 0.8)
            res["nearest_model1"] = spell(gt, codes(np.where(lab_m1 == 2, 0, lab_m1), dead, d["period"]))
            s25 = d["s25"]
            oracle = np.where(s25 >= 0, d["team"][np.maximum(s25, 0)], 0)
            res["touch_oracle"] = spell(gt, codes(oracle, dead, d["period"]))
            alive5 = alive[d["idx"]]
            for lk, lr in GRID:
                sender5 = pc.decode_match(probs["s"], probs["r"], d["present"], alive5, lk, lr)
                if dead_name == "truth" and (lk, lr) == GRID[2]:
                    res_acc = float((sender5[okl] == lab[okl]).mean())
                team5 = np.where(sender5 >= 0, d["team"][np.maximum(sender5, 0)], 0)
                team25 = np.repeat(team5, pc.STEP)[:len(d["period"])]
                res["pathcrf_{:g}_{:g}".format(lk, lr)] = spell(gt, codes(team25, dead, d["period"]))
            res["sender_acc_4_4"] = res_acc
            out.append(res)
        print("done", d["m"], "{:.0f}s".format(time.time() - t0), flush=True)
    t = pd.DataFrame(out)
    print("\n== spellF1 (H/A/D) per match ==")
    print(t.round(4).to_string(index=False))
    print("\n== mean over 7 matches ==")
    print(t.drop(columns="match").groupby("dead").mean().round(4).T.to_string())


if __name__ == "__main__":
    main()
