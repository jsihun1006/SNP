"""Which possession spells does the best model get wrong, and how? (DFL, leave-one-match-out, dead = truth)

    python possession_pathcrf/eval_spells.py

Model: hybrid PathCRF-lite + team model on every step, penalty 8 (best in eval_contest2.py).
Truth: DFL ball_owning_team and our touch labels. A true spell is matched when a predicted spell of the same team
has IoU >= 0.5 with it (the middle of the official 0.3-0.7 range). Unmatched true spells are classified:
  missed      the prediction is the other team for most of it (the change was not seen: merged into neighbours)
  boundary    same team for most of it, but the predicted spell is longer/shorter/shifted (IoU < 0.5)
  split       same team for most of it, but more than one predicted spell covers it (a false change inside)
Unmatched predicted spells: fake (other team for most of it in the truth) or boundary.
Also: length distribution of matched and unmatched true spells, and frame accuracy.
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
from eval_contest import hybrid_scores, contest_features, touch_team, to25, decode  # noqa: E402

BINS = [0, 1, 2, 4, 8, 16, 1e9]
LABELS = ["<1s", "1-2s", "2-4s", "4-8s", "8-16s", "16s+"]


def classify(gt, pred, fps=25):
    runs = pm.valid_runs(gt)
    gs, ps = pm.possession_spells(gt, runs), pm.possession_spells(pred, runs)
    ov = pm.spell_overlaps(gs, ps)
    best_g = {}
    best_p = {}
    for iou, i, j in ov:
        best_g[i] = max(best_g.get(i, 0), iou)
        best_p[j] = max(best_p.get(j, 0), iou)
    rows = []
    for i, (a, b, k) in enumerate(gs):
        seg = pred[a:b]
        same = float((seg == k).mean())
        covering = len({j for j, (c, d, kk) in enumerate(ps) if kk == k and c < b and a < d})
        cover = [(c, e) for c, e, kk in ps if kk == k and c < b and a < e]
        big = max(cover, key=lambda ce: min(ce[1], b) - max(ce[0], a)) if cover else None
        across = big is not None and sum(1 for g0, g1, gk in gs if gk == k and g0 < big[1] and big[0] < g1) >= 2
        if best_g.get(i, 0) >= 0.5:
            kind = "matched"
        elif same < 0.5:
            kind = "missed"
        elif covering > 1:
            kind = "split"
        elif across:
            kind = "boundary_merged"   # the predicted spell also covers another true spell of the same team
        else:
            kind = "boundary_shift"
        rows.append({"side": "true", "kind": kind, "seconds": (b - a) / fps})
    for j, (a, b, k) in enumerate(ps):
        if best_p.get(j, 0) >= 0.5:
            continue
        same = float((gt[a:b] == k).mean())
        rows.append({"side": "pred", "kind": "fake" if same < 0.5 else "boundary", "seconds": (b - a) / fps})
    return rows


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
                         y5=tt[d["idx"]], touch=np.where(tt >= 0, tt, 0)))
    params = dict(objective="binary", learning_rate=0.05, num_leaves=31, min_data_in_leaf=50, verbose=-1, seed=7)
    rows, acc = [], []
    for k, d in enumerate(data):
        P = prep[k]
        Xs, ys = [], []
        for j, Q in enumerate(prep):
            if j != k:
                m = Q["alive5"] & (Q["y5"] >= 0)
                Xs.append(Q["X"][m])
                ys.append((Q["y5"][m] == 0).astype(np.int8))
        q = np.clip(lgb.train(params, lgb.Dataset(np.concatenate(Xs), np.concatenate(ys)), 200).predict(P["X"]),
                    1e-4, 1 - 1e-4)
        home = d["team"] == 0
        ls = P["ls"].copy()
        rc = np.flatnonzero(P["alive5"])
        ls[np.ix_(rc, np.flatnonzero(home))] += np.log(q[rc])[:, None]
        ls[np.ix_(rc, np.flatnonzero(~home))] += np.log(1 - q[rc])[:, None]
        team = to25(d, decode(ls, P["lr"], d, P["alive5"]))
        alive = (d["period"] > 0) & ~d["dead_true"]
        pred = codes(team, d["dead_true"], d["period"])
        for truth_name, truth_team in (("dfl", d["gt_team"]), ("touch", P["touch"])):
            gt = codes(truth_team, d["dead_true"], d["period"])
            for r in classify(gt, pred):
                r.update(match=d["m"], truth=truth_name)
                rows.append(r)
            acc.append({"match": d["m"], "truth": truth_name, "frame_acc": float((team[alive] == truth_team[alive]).mean()),
                        "spellF1": pm.spell_scores(gt, pred, pm.valid_runs(gt))["mean_f1"]})
        print("done", d["m"], flush=True)
    t = pd.DataFrame(rows)
    t["length"] = pd.cut(t["seconds"], BINS, labels=LABELS, right=False)
    a = pd.DataFrame(acc)
    print("\n== frame accuracy and spellF1 (mean over 7) ==")
    print(a.groupby("truth")[["frame_acc", "spellF1"]].mean().round(4).to_string())
    for truth_name in ("dfl", "touch"):
        s = t[t["truth"] == truth_name]
        tr, pr = s[s["side"] == "true"], s[s["side"] == "pred"]
        print("\n== truth = {} : true spells by outcome (count per match) ==".format(truth_name))
        print((tr.groupby("kind").size() / 7).round(1).to_string())
        print("\ntrue spells: outcome share by length")
        print(pd.crosstab(tr["length"], tr["kind"], normalize="index").round(3).assign(
            n_per_match=(tr.groupby("length", observed=False).size() / 7).round(1)).to_string())
        print("\nunmatched predicted spells (count per match)")
        print((pr.groupby(["kind", "length"], observed=True).size() / 7).round(1).to_string())


if __name__ == "__main__":
    main()
