"""Published PathCRF model vs our pipeline on J03WR9, the one DFL match it never saw (spellF1, dead = truth).

    python possession_pathcrf/eval_paper_j03wr9.py

Inputs: outputs/pathcrf_repo/J03WR9_probs.csv.gz and _decoded.csv.gz (run_pathcrf_repo.py, no event information),
our saved 25-fps hybrid probabilities (outputs/25fps) and the DFL data.
Variants (penalty 40, illegal jump 4 x penalty, 25 fps on contested frames and 5 fps elsewhere):
  paper_crf            the paper model's own CRF decoding; team = team of the sender (outside nodes keep the last team)
  paper_probs          the paper's sender / receiver probabilities + the ball bonus at someone's feet, our decoder
  paper_probs_contest  paper_probs + contest model v2 (weight 2) + contest-outcome model (trained on the other 6 DFL
                       matches with our hybrid; the contested frames come from the paper's P(H))
  ours_contest         our current best: our LightGBM hybrid + contest v2 + outcome
Outputs: outputs/pathcrf_repo/J03WR9_comparison.csv
"""

import os
import sys

import numpy as np
import pandas as pd

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
import eval_25fps_step as st  # noqa: E402
from eval_dfl3 import CONTROL_OBS_M, HYBRID_BONUS  # noqa: E402
from eval_contest25 import contest_features, episodes, episode_rows, OUTCOME  # noqa: E402
from eval_contest_tune import rule_features, train_contest, TAU, LAM  # noqa: E402

TEST = "J03WR9"
SRC = os.path.join(HERE, "outputs", "pathcrf_repo")


def paper_scores(d):
    pp = pd.read_csv(os.path.join(SRC, "{}_probs.csv.gz".format(TEST)))
    T, N = d["ls"].shape
    ps, pr = np.full((T, N), 1e-6), np.full((T, N), 1e-6)
    f, j = pp["frame"].to_numpy(), pp["player"].to_numpy()
    ps[f, j], pr[f, j] = pp["p_sender"].to_numpy(), pp["p_receiver"].to_numpy()
    ls, lr = np.log(np.clip(ps, 1e-6, 1)), np.log(np.clip(pr, 1e-6, 1))
    who, dmin = st.nearest_with_dist(d["players"], d["ball"], d["team"])
    r = np.flatnonzero(dmin <= CONTROL_OBS_M)
    ls[r, who[r]] += HYBRID_BONUS
    lr[r, who[r]] += HYBRID_BONUS
    w = np.where(d["present"], np.exp(ls - np.where(d["present"], ls, -np.inf).max(axis=1, keepdims=True)), 0)
    w = w / np.clip(w.sum(axis=1, keepdims=True), 1e-9, None)
    return ls, lr, (w * (d["team"] == 0)[None, :]).sum(axis=1)


def add_contest(d, ls, data, k):
    import lightgbm as lgb
    home = d["team"] == 0
    cf = np.flatnonzero(d["contested"])
    q = np.clip(train_contest(data, k, "X2").predict(d["X2"]), 1e-4, 1 - 1e-4)
    out = ls.copy()
    out[np.ix_(cf, np.flatnonzero(home))] += 2.0 * np.log(q[cf])[:, None]
    out[np.ix_(cf, np.flatnonzero(~home))] += 2.0 * np.log(1 - q[cf])[:, None]
    Xs, ys = [], []
    for j, o in enumerate(data):
        if j != k:
            X, y, _ = episode_rows(o, episodes(o["contested"], o["alive"]), o["rule"], o["X2"])
            Xs.append(X[y >= 0])
            ys.append((y[y >= 0] == 0).astype(np.int8))
    ob = lgb.train(OUTCOME, lgb.Dataset(np.concatenate(Xs), np.concatenate(ys)), 200)
    X, _, spans = episode_rows(d, episodes(d["contested"], d["alive"]), d["rule"], d["X2"])
    for (b, e), qh in zip(spans, np.clip(ob.predict(X), 1e-4, 1 - 1e-4)):
        out[b:e, home] += np.log(qh)
        out[b:e, ~home] += np.log(1 - qh)
    return out


def prepare(d, p_home):
    d = dict(d)
    d["p_home"] = p_home
    d["contested"] = d["alive"] & (np.abs(2 * p_home - 1) < TAU)
    d["X1"] = contest_features(d)
    d["rule"], extra = rule_features(d)
    d["X2"] = np.concatenate([d["X1"], extra], axis=1)
    return d


def main():
    import dfl_load
    pd.set_option("display.width", 200)
    data = []
    for m in dfl_load.MATCHES:
        d = st.load(m, add_team=False)
        d["tt"] = np.asarray(d["tt"], dtype=object)
        data.append(prepare(d, d["p_home"]))
    k = dfl_load.MATCHES.index(TEST)
    ours = data[k]
    rows = []
    dec = pd.read_csv(os.path.join(SRC, "{}_decoded.csv.gz".format(TEST)))
    sender = np.full(len(ours["period"]), -1)
    sender[dec["frame"].to_numpy()] = dec["sender"].to_numpy()
    idx = np.where(sender >= 0, np.arange(len(sender)), -1)
    last = np.maximum.accumulate(idx)
    team = np.where(last >= 0, ours["team"][sender[np.maximum(last, 0)]], 0)
    rows.append(dict(variant="paper_crf", **st.score(ours, team)))
    ls_p, lr_p, ph_p = paper_scores(ours)
    dp = prepare(ours, ph_p)
    team, _ = st.decode(dp, ls_p, lr_p, LAM, fine=dp["contested"])
    rows.append(dict(variant="paper_probs", **st.score(dp, team)))
    team, _ = st.decode(dp, add_contest(dp, ls_p, data, k), lr_p, LAM, fine=dp["contested"])
    rows.append(dict(variant="paper_probs_contest", **st.score(dp, team)))
    team, _ = st.decode(ours, add_contest(ours, ours["ls"], data, k), ours["lr"], LAM, fine=ours["contested"])
    rows.append(dict(variant="ours_contest", **st.score(ours, team)))
    t = pd.DataFrame(rows)
    t.to_csv(os.path.join(SRC, "{}_comparison.csv".format(TEST)), index=False)
    print("== {} (held out of the paper model), dead = truth ==".format(TEST))
    print(t.round(4).to_string(index=False))


if __name__ == "__main__":
    main()
