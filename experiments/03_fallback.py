"""Experiment A: team-shape fallback evidence for frames where the ball is missing.

1. Pseudo-label visible-ball frames with experiment C's decoder output (no ground truth involved).
2. Train a gradient-boosted classifier P(home in possession) on ball-free team-shape features,
   fitted on one half and tested on the other (and on the public GT, as a check only).
3. Feed its log-odds into the decoder as weak per-frame evidence wherever the ball is missing.
Also evaluated with the ball artificially masked in the public window using real gap patterns.
"""
import warnings
import numpy as np
from sklearn.ensemble import HistGradientBoostingClassifier

import common as c
import decoder as dec
import features as ft

warnings.filterwarnings("ignore")
d = c.load_tracking()
gt = c.load_gt()
vis = np.isfinite(d["ball"][:, 0])
dh, da = dec.nearest_team_distances(d["pos"], d["ball"], d["team"])
e = dec.nearest_evidence(dh, da)
X, signs = ft.build(d)
print("attack signs (team, half):", signs)

# 1. pseudo labels = decoded possession on visible frames
base = dec.decode(dec.make_scores(e, d_bias=None), 25, 25)
y = (base == "H").astype(int)
g = d["graded"]
rng = np.random.default_rng(0)


def fit(mask):
    idx = np.where(mask)[0]
    idx = idx[rng.random(len(idx)) < 0.2]       # subsample: neighbouring frames are near duplicates
    m = HistGradientBoostingClassifier(max_depth=4, max_iter=150, learning_rate=0.06,
                                       l2_regularization=1.0, random_state=0)
    return m.fit(X[idx], y[idx])


# 2. cross-half validation on pseudo labels, and check against the public GT
for tr, te in ((1, 2), (2, 1)):
    m = fit(g & vis & (d["half"] == tr))
    sel = g & vis & (d["half"] == te)
    p = m.predict_proba(X[sel])[:, 1]
    print("train half {} -> test half {}: accuracy on pseudo-labels {:.3f} (majority {:.3f})".format(
        tr, te, np.mean((p > .5) == y[sel]), max(y[sel].mean(), 1 - y[sel].mean())))
# the public window (plus 30 s either side) is held out of training so the checks below are not in-sample
holdout = np.zeros(c.N_FRAMES, bool)
holdout[11185 - 750:18684 + 750 + 1] = True
model = fit(g & vis & ~holdout)
p_all = model.predict_proba(X)[:, 1]
alive = np.isin(gt, ["H", "A"])
print("public GT alive frames: shape-only accuracy {:.3f} on all, {:.3f} where ball missing (n={})".format(
    np.mean((p_all[alive] > .5) == (gt[alive] == "H")),
    np.mean((p_all[alive & ~vis] > .5) == (gt[alive & ~vis] == "H")) if (alive & ~vis).any() else float("nan"),
    int((alive & ~vis).sum())))

logit = np.clip(np.log((p_all + 1e-3) / (1 - p_all + 1e-3)), -3, 3)


def run(ball_vis, w, lam=25):
    """Decode with nearest-player evidence where the ball is visible, w * shape logit where it is not."""
    ev = np.where(ball_vis, e, w * logit)
    return dec.decode(dec.make_scores(ev, d_bias=None), lam, lam)


def row(name, labels):
    r = c.score(labels)
    print("{:<30} spellF1 {:.4f} macroF1 {:.4f} poss {:>3} dur {:>5.1f}".format(
        name, r["spell_f1_mean"], r["macro_f1"], r["n_pred_spells"], r["mean_spell_duration_s"]))
    return r["spell_f1_mean"]


print("\n-- real ball (public window almost never misses it: fallback barely used) --")
for w in (0.0, 0.25, 0.5, 1.0, 2.0):
    lab = run(vis, w)
    row("A_w{}".format(w), lab)
    if w == 0.5:
        c.write_csv("{}/A_fallback_w0.5.csv".format(c.RESULTS), lab)

print("\n-- ball masked in public window with real gap patterns from other chunks (mean over 8) --")
a0, b0 = 11185, 18684
L = b0 - a0 + 1
starts = [30000, 45000, 60000, 75000, 125000, 140000, 155000, 170000]
res = {w: [] for w in (0.0, 0.25, 0.5, 1.0, 2.0)}
for s0 in starts:
    mask = vis.copy()
    mask[a0:b0 + 1] = vis[s0:s0 + L]                    # visibility pattern borrowed from elsewhere
    mask[a0:b0 + 1] &= vis[a0:b0 + 1]
    for w in res:
        r = c.score(run(mask, w))
        res[w].append((r["spell_f1_mean"], r["macro_f1"], 1 - mask[a0:b0 + 1].mean()))
for w, v in res.items():
    v = np.array(v)
    print("w={:<4} masked-missing {:.0%}  spellF1 {:.4f} +/- {:.4f}  macroF1 {:.4f}".format(
        w, v[:, 2].mean(), v[:, 0].mean(), v[:, 0].std(), v[:, 1].mean()))
