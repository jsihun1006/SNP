"""Frame-dependent switch penalty: cheaper changes on contested frames, dearer on calm frames (DFL, LOO).

    python possession_pathcrf/eval_lam_frame.py

Base model = the current best: PathCRF-lite hybrid + contest model v2 (weight 2) + contest-outcome model, 25 fps on
contested frames and 5 fps elsewhere, base penalty 40, illegal jump 4 x penalty.
Penalty per frame = 40 x M_CONTEST on contested frames (|P(H) - P(A)| < 0.3) and 40 x M_CALM elsewhere.
Grid M_CONTEST x M_CALM; (1, 1) is the base. For each test match the pair with the best mean spellF1 on the other
6 matches is used (DFL truth); touch-label spellF1 and spell counts are reported too.
Outputs: outputs/lam_frame/ (parameters, per-match results, chosen).
"""

import os
import sys
import time

import numpy as np
import pandas as pd

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
import eval_25fps_step as st  # noqa: E402
from eval_contest25 import contest_features, episodes, episode_rows, OUTCOME  # noqa: E402
from eval_contest_tune import rule_features, train_contest, TAU, LAM  # noqa: E402

M_CONTEST = (0.25, 0.5, 1.0)
M_CALM = (1.0, 1.5, 2.0)
OUT = os.path.join(HERE, "outputs", "lam_frame")


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
    rows = []
    for k, d in enumerate(data):
        home = d["team"] == 0
        cf = np.flatnonzero(d["contested"])
        q = np.clip(train_contest(data, k, "X2").predict(d["X2"]), 1e-4, 1 - 1e-4)
        ls = d["ls"].copy()
        ls[np.ix_(cf, np.flatnonzero(home))] += 2.0 * np.log(q[cf])[:, None]
        ls[np.ix_(cf, np.flatnonzero(~home))] += 2.0 * np.log(1 - q[cf])[:, None]
        Xs, ys = [], []
        for j, o in enumerate(data):
            if j != k:
                X, y, _ = episode_rows(o, episodes(o["contested"], o["alive"]), o["rule"], o["X2"])
                Xs.append(X[y >= 0])
                ys.append((y[y >= 0] == 0).astype(np.int8))
        ob = lgb.train(OUTCOME, lgb.Dataset(np.concatenate(Xs), np.concatenate(ys)), 200)
        X, _, spans = episode_rows(d, episodes(d["contested"], d["alive"]), d["rule"], d["X2"])
        for (b, e), qh in zip(spans, np.clip(ob.predict(X), 1e-4, 1 - 1e-4)):
            ls[b:e, home] += np.log(qh)
            ls[b:e, ~home] += np.log(1 - qh)
        for mc in M_CONTEST:
            for mn in M_CALM:
                mult = np.where(d["contested"], mc, mn)
                team, _ = st.decode(d, ls, d["lr"], LAM, fine=d["contested"], lam_frame=mult)
                r = {"match": d["m"], "m_contest": mc, "m_calm": mn}
                r.update(st.score(d, team))
                rows.append(r)
        print("done", d["m"], "{:.0f}s".format(time.time() - t0), flush=True)
    t = pd.DataFrame(rows)
    t.to_csv(os.path.join(OUT, "results_per_match.csv"), index=False)
    pd.DataFrame([("TAU", TAU), ("LAM", LAM), ("BIG_RATIO", st.BIG_RATIO), ("M_CONTEST", ",".join(map(str, M_CONTEST))),
                  ("M_CALM", ",".join(map(str, M_CALM))), ("contest_weight", 2.0), ("grid", "25 fps contested, 5 fps else")],
                 columns=["parameter", "value"]).to_csv(os.path.join(OUT, "parameters.csv"), index=False)
    print("\n== mean over 7 test matches ==")
    print(t.groupby(["m_contest", "m_calm"])[["spellF1_dfl", "spellF1_touch", "frame_acc", "pred_spells",
                                              "true_spells"]].mean().round(4).to_string())
    pk = st.choose(t, ["m_contest", "m_calm"])
    pk.to_csv(os.path.join(OUT, "chosen.csv"), index=False)
    base = t[(t["m_contest"] == 1.0) & (t["m_calm"] == 1.0)].set_index("match")["spellF1_dfl"].sort_index()
    pk = pk.set_index("match").sort_index()
    print("\nchosen on the other 6 matches:")
    print(pk[["m_contest", "m_calm", "spellF1_dfl", "spellF1_touch", "frame_acc"]].round(4).to_string())
    print("mean spellF1 DFL: base (1, 1) {:.4f} -> chosen {:.4f}; better {} / worse {}".format(
        base.mean(), pk["spellF1_dfl"].mean(), int((pk["spellF1_dfl"] > base).sum()), int((pk["spellF1_dfl"] < base).sum())))


if __name__ == "__main__":
    main()
