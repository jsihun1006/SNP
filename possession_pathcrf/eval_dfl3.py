"""Possession on DFL, round 3: spell-count diagnosis, a control-only baseline and a hybrid PathCRF-lite.

    python possession_pathcrf/eval_dfl3.py

Dead frames = DFL truth (only possession is judged). Node probabilities from eval_dfl2 (cache/).
Methods
  baseline_model1     nearest player's team every frame + model1 decoder (current submission logic)
  control_model1      nearest player's team only when the ball is within CONTROL_OBS_M of that player, else
                      missing (a ball in flight says nothing), + the same decoder
  pathcrf_8_8         PathCRF-lite, no ball
  hybrid_8_8          PathCRF-lite + the ball where it is seen at someone's feet: the nearest player within
                      CONTROL_OBS_M gets +HYBRID_BONUS on both sender and receiver log-scores
Ball conditions: ball (original) and hidden (deleted while dead, above 3 m and while a keeper holds it).
"""

import os
import sys

import numpy as np
import pandas as pd

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
import pathcrf_lite as pc  # noqa: E402
import possession_metrics as pm  # noqa: E402
from eval_dfl import prepare, codes, load_model1  # noqa: E402
from eval_dfl2 import node_probs, nearest, CFG  # noqa: E402

CONTROL_OBS_M = 1.5      # "at someone's feet" (fixed before the run; DFL label control radius is 1.0 m)
HYBRID_BONUS = 3.0       # log-score bonus for the player at the ball (fixed before the run)
LAM = (8.0, 8.0)


def nearest_with_dist(players, ball, team):
    dist = np.linalg.norm(players - ball[:, None, :], axis=2)
    dist = np.where(np.isfinite(dist), dist, np.inf)
    who = dist.argmin(axis=1)
    dmin = dist[np.arange(len(dist)), who]
    return who, dmin


def decode_hybrid(probs, present, alive5, who5, near5, lam):
    T, N = probs["s"].shape
    eps = 1e-6
    ls = np.log(np.clip(probs["s"], eps, 1))
    lr = np.log(np.clip(probs["r"], eps, 1))
    rows = np.flatnonzero(near5)
    ls[rows, who5[rows]] += HYBRID_BONUS
    lr[rows, who5[rows]] += HYBRID_BONUS
    sender = np.full(T, -1)
    edges = np.diff(np.r_[False, alive5, False].astype(np.int8))
    for a, b in zip(np.flatnonzero(edges == 1), np.flatnonzero(edges == -1)):
        su, _ = pc.viterbi_edges(ls[a:b], lr[a:b], present[a:b], *lam)
        sender[a:b] = su
    return sender


def main():
    import dfl_load
    from eval_keeper_hold_size import keeper_hold_spans
    pd.set_option("display.width", 200)
    data = [prepare(m) for m in dfl_load.MATCHES]
    m1 = load_model1()
    out = []
    for k, d in enumerate(data):
        probs = node_probs(data, k)
        tracking, _, _ = dfl_load.load_match(d["m"])
        z = tracking["ball_z"].to_numpy(np.float64)
        graded = d["period"] > 0
        dead = d["dead_true"]
        hidden = d["ball"].copy()
        hide = (graded & dead) | (graded & ~dead & (z > 3.0))
        for a, b in keeper_hold_spans(d["match"], dead):
            hide[a:b] = True
        hidden[hide] = np.nan
        gt = codes(d["gt_team"], dead, d["period"])
        alive5 = (graded & ~dead)[d["idx"]]
        preds = {}
        team5 = lambda s5: np.repeat(np.where(s5 >= 0, d["team"][np.maximum(s5, 0)], 0), pc.STEP)[:len(d["period"])]
        sender5 = pc.decode_match(probs["s"], probs["r"], d["present"], alive5, *LAM)
        preds["pathcrf_8_8"] = team5(sender5)
        for bname, ball in (("ball", d["ball"]), ("hidden", hidden)):
            near = nearest(d["players"], ball, d["team"])
            who, dmin = nearest_with_dist(d["players"], ball, d["team"])
            for name, obs_team in (("baseline_model1", near),
                                   ("control_model1", np.where(dmin <= CONTROL_OBS_M, d["team"][who], -1))):
                obs = np.where(obs_team == 0, 0, np.where(obs_team == 1, 1, -1)).astype(np.int8)
                lab = m1.viterbi(m1.constrained_costs(m1.emission_costs(obs, CFG), graded, dead), 1.2, 0.8)
                preds["{}/{}".format(name, bname)] = np.where(lab == 2, 0, lab)
            near5 = (dmin <= CONTROL_OBS_M)[d["idx"]]
            preds["hybrid_8_8/" + bname] = team5(decode_hybrid(probs, d["present"], alive5, who[d["idx"]], near5, LAM))
        for name, team in preds.items():
            sc = pm.spell_scores(gt, codes(team, dead, d["period"]), pm.valid_runs(gt))
            alive = graded & ~dead
            out.append({"match": d["m"], "method": name, "spellF1": sc["mean_f1"],
                        "team_acc": float((team[alive] == d["gt_team"][alive]).mean()),
                        "pred_spells": sc["n_pred"], "true_spells": sc["n_gt"]})
        print("done", d["m"], flush=True)
    t = pd.DataFrame(out)
    t.to_csv(os.path.join(HERE, "cache", "eval_dfl3.csv"), index=False)
    print("\n== mean over 7 matches (dead = truth) ==")
    print(t.groupby("method")[["spellF1", "team_acc", "pred_spells", "true_spells"]].mean().round(4).to_string())


if __name__ == "__main__":
    main()
