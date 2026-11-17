"""PathCRF-lite (hybrid) as the base, tuning only the contested frames (DFL, 25 fps, LOO, ball available).

    python possession_pathcrf/eval_contest_tune.py

Fixed: TAU 0.3 (contested = |P(H) - P(A)| < 0.3), switch penalty 40, illegal jump 4 x penalty.
Tuned, each chosen per test match on the other 6 matches (DFL truth, spellF1):
  features   v1 = eval_contest25 contest features
             v2 = v1 + last-touch rule team (R 1.0 m, real touches), number of rule team changes in the last
                  1 s, and whether the rule agrees with the hybrid
             (first compared by contested-frame accuracy, then the better one is decoded)
  w_contest  weight of the contest-model log probability added on contested frames: 0.5, 1, 2
  outcome    contest-outcome model off / on
Outputs: outputs/contest_tune/ (parameters, contest-model accuracy, per-match results, chosen settings).
"""

import os
import sys
import time

import numpy as np
import pandas as pd

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
import eval_25fps_step as st  # noqa: E402
from check_touch_rule import touches  # noqa: E402
from eval_contest25 import (contest_features, episodes, episode_rows, add_team, CONTEST, OUTCOME,  # noqa: E402
                            OUTCOME_S)

TAU = 0.3
LAM = 40.0
W_CONTEST = (0.5, 1.0, 2.0)
OUT = os.path.join(HERE, "outputs", "contest_tune")


def rule_features(d):
    rule = touches(d, 1.0, True)
    change = np.r_[0, (rule[1:] != rule[:-1]).astype(int)]
    flips_1s = np.convolve(change, np.ones(25, int), mode="full")[:len(change)]
    agree = (rule == np.where(d["p_home"] > 0.5, 0, 1)).astype(int)
    return rule, np.stack([rule, flips_1s, agree], axis=1).astype(np.float32)


def train_contest(data, k, key):
    import lightgbm as lgb
    Xs, ys = [], []
    for j, o in enumerate(data):
        if j != k:
            m = o["contested"] & np.isin(o["tt"], ["H", "A"]) & (np.arange(len(o["tt"])) % 5 == 0)
            Xs.append(o[key][m])
            ys.append((o["tt"][m] == "H").astype(np.int8))
    return lgb.train(CONTEST, lgb.Dataset(np.concatenate(Xs), np.concatenate(ys)), 200)


def main():
    import lightgbm as lgb
    import dfl_load
    pd.set_option("display.width", 200)
    os.makedirs(OUT, exist_ok=True)
    t0 = time.time()
    data = []
    for m in dfl_load.MATCHES:
        d = st.load(m, add_team=False)
        d["tt"] = np.asarray(d["tt"], dtype=object)
        d["contested"] = d["alive"] & (np.abs(2 * d["p_home"] - 1) < TAU)
        d["X1"] = contest_features(d)
        d["rule"], extra = rule_features(d)
        d["X2"] = np.concatenate([d["X1"], extra], axis=1)
        data.append(d)
    print("loaded {:.0f}s".format(time.time() - t0), flush=True)
    # 1. contest-model accuracy on contested frames, v1 vs v2
    acc = []
    for k, d in enumerate(data):
        cf = np.flatnonzero(d["contested"])
        lab = np.isin(d["tt"][cf], ["H", "A"])
        r = {"match": d["m"], "hybrid": float(((d["p_home"][cf][lab] > .5) == (d["tt"][cf][lab] == "H")).mean()),
             "rule": float(((d["rule"][cf][lab] == 0) == (d["tt"][cf][lab] == "H")).mean())}
        for key in ("X1", "X2"):
            q = np.clip(train_contest(data, k, key).predict(d[key]), 1e-4, 1 - 1e-4)
            d["q_" + key] = q
            r["contest_" + key] = float(((q[cf][lab] > .5) == (d["tt"][cf][lab] == "H")).mean())
        acc.append(r)
    a = pd.DataFrame(acc)
    a.to_csv(os.path.join(OUT, "contest_model_accuracy.csv"), index=False)
    print("\n== accuracy on contested frames (touch labels), mean over 7 ==")
    print(a.drop(columns="match").mean().round(4).to_string())
    best = "X2" if a["contest_X2"].mean() > a["contest_X1"].mean() else "X1"
    print("decoding with features", best, flush=True)
    # 2. decode grid
    rows = []
    for k, d in enumerate(data):
        q = d["q_" + best]
        cf = np.flatnonzero(d["contested"])
        # outcome model (episode features from the rule team at the episode end)
        Xs, ys = [], []
        for j, o in enumerate(data):
            if j != k:
                X, y, _ = episode_rows(o, episodes(o["contested"], o["alive"]), o["rule"], o[best])
                Xs.append(X[y >= 0])
                ys.append((y[y >= 0] == 0).astype(np.int8))
        ob = lgb.train(OUTCOME, lgb.Dataset(np.concatenate(Xs), np.concatenate(ys)), 200)
        X, y, spans = episode_rows(d, episodes(d["contested"], d["alive"]), d["rule"], d[best])
        qo = np.clip(ob.predict(X), 1e-4, 1 - 1e-4)
        home = d["team"] == 0
        for w in W_CONTEST:
            base = d["ls"].copy()
            base[np.ix_(cf, np.flatnonzero(home))] += w * np.log(q[cf])[:, None]
            base[np.ix_(cf, np.flatnonzero(~home))] += w * np.log(1 - q[cf])[:, None]
            for outcome in (False, True):
                ls = base.copy()
                if outcome:
                    for (b, e), qh in zip(spans, qo):
                        ls[b:e, home] += np.log(qh)
                        ls[b:e, ~home] += np.log(1 - qh)
                team, _ = st.decode(d, ls, d["lr"], LAM)
                r = {"match": d["m"], "w_contest": w, "outcome": int(outcome)}
                r.update(st.score(d, team))
                rows.append(r)
        print("done", d["m"], "{:.0f}s".format(time.time() - t0), flush=True)
    t = pd.DataFrame(rows)
    t.to_csv(os.path.join(OUT, "results_per_match.csv"), index=False)
    pd.DataFrame([("TAU", TAU), ("LAM", LAM), ("BIG_RATIO", st.BIG_RATIO), ("W_CONTEST", ",".join(map(str, W_CONTEST))),
                  ("OUTCOME_S", OUTCOME_S), ("features", best)], columns=["parameter", "value"]).to_csv(
        os.path.join(OUT, "parameters.csv"), index=False)
    print("\n== mean over 7 test matches ==")
    print(t.groupby(["w_contest", "outcome"])[["spellF1_dfl", "spellF1_touch", "frame_acc", "pred_spells",
                                               "true_spells"]].mean().round(4).to_string())
    pk = st.choose(t, ["w_contest", "outcome"])
    pk.to_csv(os.path.join(OUT, "chosen.csv"), index=False)
    print("\nchosen on the other 6 matches:")
    print(pk[["match", "w_contest", "outcome", "spellF1_dfl", "spellF1_touch", "frame_acc"]].round(4).to_string(index=False))
    print("mean spellF1 DFL {:.4f} (previous contest structure + outcome: 0.5956)".format(pk["spellF1_dfl"].mean()))


if __name__ == "__main__":
    main()
