"""Does a contest decision carry into the long possession after it? Error-origin diagnosis (DFL, 25 fps, LOO).

    python possession_pathcrf/eval_propagation.py

Variants (one Viterbi over each in-play stretch, penalty 40, illegal jump 4 x penalty; 25 fps on contested frames,
5 fps elsewhere, contest_v2_full25 = 25 fps everywhere for comparison):
  hybrid      PathCRF-lite hybrid only
  contest_v1  + contest model v1 (weight 1) + contest-outcome model   (eval_contest25 "contest_outcome")
  contest_v2  + contest model v2 (weight 2) + contest-outcome model   (eval_contest_tune best)
Every wrong-team in-play frame (DFL truth) is put in one of three groups:
  in_contest    the frame itself is contested (|P(H) - P(A)| < 0.3, hybrid P, fixed before decoding)
  carried       not contested, but its run of wrong frames started inside a contest or within CARRY_S after one
  independent   the rest
If contest decisions carry into what follows, the 'carried' share should fall together with 'in_contest'.
Outputs: outputs/propagation/ (parameters, error groups per match, spellF1 per match).
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

CARRY_S = 1.0
OUT = os.path.join(HERE, "outputs", "propagation")
VARIANTS = {"hybrid": None, "contest_v1": ("X1", 1.0), "contest_v2": ("X2", 2.0), "contest_v2_full25": ("X2", 2.0)}
# Decoding: 25 fps only on contested frames, 5 fps elsewhere (eval_25fps_step.decode fine=...).
# contest_v2_full25 decodes every frame at 25 fps, to check that the mixed grid loses nothing.


def runs(mask):
    e = np.diff(np.r_[False, mask, False].astype(np.int8))
    return list(zip(np.flatnonzero(e == 1), np.flatnonzero(e == -1)))


def error_groups(d, team):
    alive, contested = d["alive"], d["contested"]
    after = contested.copy()
    carry = int(CARRY_S * 25)
    for a, b in runs(contested):
        after[b:b + carry] = True
    wrong = alive & (team != d["gt"])
    inside = int((wrong & contested).sum())
    carried = independent = 0
    for a, b in runs(wrong):
        outside = int((~contested[a:b]).sum())
        if after[a]:
            carried += outside
        else:
            independent += outside
    total = max(int(wrong.sum()), 1)
    return {"error_frames": int(wrong.sum()), "error_rate": float(wrong[alive].mean()),
            "in_contest": inside / total, "carried": carried / total, "independent": independent / total,
            "contest_error_rate": float(wrong[contested].mean()),
            "outside_error_rate": float(wrong[alive & ~contested].mean())}


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
        for name, spec in VARIANTS.items():
            ls = d["ls"].copy()
            if spec is not None:
                key, w = spec
                q = np.clip(train_contest(data, k, key).predict(d[key]), 1e-4, 1 - 1e-4)
                ls[np.ix_(cf, np.flatnonzero(home))] += w * np.log(q[cf])[:, None]
                ls[np.ix_(cf, np.flatnonzero(~home))] += w * np.log(1 - q[cf])[:, None]
                Xs, ys = [], []
                for j, o in enumerate(data):
                    if j != k:
                        X, y, _ = episode_rows(o, episodes(o["contested"], o["alive"]), o["rule"], o[key])
                        Xs.append(X[y >= 0])
                        ys.append((y[y >= 0] == 0).astype(np.int8))
                ob = lgb.train(OUTCOME, lgb.Dataset(np.concatenate(Xs), np.concatenate(ys)), 200)
                X, _, spans = episode_rows(d, episodes(d["contested"], d["alive"]), d["rule"], d[key])
                for (b, e), qh in zip(spans, np.clip(ob.predict(X), 1e-4, 1 - 1e-4)):
                    ls[b:e, home] += np.log(qh)
                    ls[b:e, ~home] += np.log(1 - qh)
            fine = None if name.endswith("full25") else d["contested"]
            team, _ = st.decode(d, ls, d["lr"], LAM, fine=fine)
            r = {"match": d["m"], "variant": name}
            r.update(error_groups(d, team))
            r.update({k2: v for k2, v in st.score(d, team).items() if k2.startswith("spellF1")})
            rows.append(r)
        print("done", d["m"], "{:.0f}s".format(time.time() - t0), flush=True)
    t = pd.DataFrame(rows)
    t.to_csv(os.path.join(OUT, "error_groups_per_match.csv"), index=False)
    pd.DataFrame([("TAU", TAU), ("LAM", LAM), ("BIG_RATIO", st.BIG_RATIO), ("CARRY_S", CARRY_S)],
                 columns=["parameter", "value"]).to_csv(os.path.join(OUT, "parameters.csv"), index=False)
    print("\n== error origin, mean over 7 matches (DFL truth, in-play frames) ==")
    cols = ["error_rate", "contest_error_rate", "outside_error_rate", "in_contest", "carried", "independent",
            "spellF1_dfl", "spellF1_touch"]
    print(t.groupby("variant", sort=False)[cols].mean().round(4).to_string())
    print("\nerror frames per match (mean): " + ", ".join(
        "{} {:.0f}".format(v, g["error_frames"].mean()) for v, g in t.groupby("variant", sort=False)))


if __name__ == "__main__":
    main()
