"""Experiment B: dead-ball evidence from ball and player behaviour.

Logistic regression (strongly regularised, six-ish windowed features) fitted to the public GT's dead-ball
vs live frames. With only 4 dead stretches this is validated leave-one-stretch-out, and reported as
weak evidence. The D log-odds enter the decoder as an emission score for the D state.
"""
import warnings
import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import make_pipeline
from sklearn.metrics import roc_auc_score

import common as c
import decoder as dec
import features as ft
import deadball as db

warnings.filterwarnings("ignore")
d = c.load_tracking()
gt = c.load_gt()
vis = np.isfinite(d["ball"][:, 0])
dh, da = dec.nearest_team_distances(d["pos"], d["ball"], d["team"])
e = dec.nearest_evidence(dh, da)
F = np.nan_to_num(db.build(d, dh, da), nan=0.0)   # NaN only outside the playing halves
names = ["pspd2s", "pspd10s", "slowfrac10s", "vis2s", "vis10s", "ballspd", "nearest", "edge", "oob"]

a, b = 11185, 18684
lab = np.zeros(c.N_FRAMES, bool)
lab[a:b + 1] = gt[a:b + 1] == "D"
win = np.zeros(c.N_FRAMES, bool)
win[a:b + 1] = True
stretches = [(12851, 12988), (15889, 16399), (17380, 17837), (18027, 18388)]


CLASS_WEIGHT = None      # None keeps the public window's dead-ball prior (19.6%); "balanced" inflates D


def model():
    return make_pipeline(StandardScaler(), LogisticRegression(C=0.1, class_weight=CLASS_WEIGHT, max_iter=500))


print("leave-one-stretch-out (hold out each dead stretch + 5 s margin, train on the rest of the window):")
aucs = []
for s, t in stretches:
    ho = np.zeros(c.N_FRAMES, bool)
    ho[s - 125:t + 126] = True
    m = model().fit(F[win & ~ho], lab[win & ~ho])
    sel = np.zeros(c.N_FRAMES, bool)
    sel[s - 400:t + 401] = True          # stretch plus surrounding live play
    sel &= win
    p = m.predict_proba(F[sel])[:, 1]
    auc = roc_auc_score(lab[sel], p)
    inside = p[lab[sel]].mean()
    print("  {}-{}  AUC {:.3f}  mean P(dead) inside {:.2f} vs outside {:.2f}".format(s, t, auc, inside, p[~lab[sel]].mean()))
    aucs.append(auc)
print("  mean AUC {:.3f}".format(np.mean(aucs)))

full = model().fit(F[win], lab[win])
coef = full[-1].coef_[0]
print("standardised coefficients:", {n: round(float(v), 2) for n, v in zip(names, coef)})
p_dead = full.predict_proba(F)[:, 1]
# honest public-window scores: cross-fit, each block predicted by a model that never saw it (5 s margin)
p_dead_cv = p_dead.copy()
edges = [a, 14400, 16900, 17900, b + 1]
for lo, hi in zip(edges[:-1], edges[1:]):
    tr = win.copy()
    tr[lo - 125:hi + 125] = False
    p_dead_cv[lo:hi] = model().fit(F[tr], lab[tr]).predict_proba(F[lo:hi])[:, 1]
g = d["graded"]
print("P(dead)>0.5 share over graded frames: {:.3f} (public GT dead share {:.3f})".format(
    np.mean(p_dead[g] > .5), lab[win].mean()))

# shape-only possession evidence (reuse A's model, public window held out)
from sklearn.ensemble import HistGradientBoostingClassifier
X, _ = ft.build(d)
base = dec.decode(dec.make_scores(e, d_bias=None), 25, 25)
y = (base == "H").astype(int)
rng = np.random.default_rng(0)
holdout = np.zeros(c.N_FRAMES, bool)
holdout[a - 750:b + 751] = True
idx = np.where(g & vis & ~holdout)[0]
idx = idx[rng.random(len(idx)) < 0.2]
gb = HistGradientBoostingClassifier(max_depth=4, max_iter=150, learning_rate=0.06, l2_regularization=1.0,
                                    random_state=0).fit(X[idx], y[idx])
p_h = gb.predict_proba(X)[:, 1]
logit_h = np.clip(np.log((p_h + 1e-3) / (1 - p_h + 1e-3)), -3, 3)
logit_d = np.clip(np.log((p_dead + 1e-3) / (1 - p_dead + 1e-3)), -4, 4)
logit_d_cv = np.clip(np.log((p_dead_cv + 1e-3) / (1 - p_dead_cv + 1e-3)), -4, 4)


def run(ball_vis, w_fb, w_d, d_off, lam, ld=None):
    ev = np.where(ball_vis, e, w_fb * logit_h)
    s = dec.make_scores(ev, d_bias=-d_off, d_evidence=w_d * (logit_d if ld is None else ld))
    return dec.decode(s, lam, lam)


def row(name, labels, write=None):
    r = c.score(labels)
    gm = d["graded"]
    print("{:<34} spellF1 {:.4f} macroF1 {:.4f} poss {:>3} dur {:>5.1f} | D share (full match) {:.2f}".format(
        name, r["spell_f1_mean"], r["macro_f1"], r["n_pred_spells"], r["mean_spell_duration_s"], np.mean(labels[gm] == "D")))
    if write:
        c.write_csv("{}/{}.csv".format(c.RESULTS, write), labels)
    return r


print("\n-- public-window rows use the cross-fitted dead-ball evidence (not in-sample); D share is over the full match --")
for w_d in (0.0, 0.25, 0.5, 1.0):
    lab_cv = run(vis, 1.0, w_d, 0.0, 25, ld=logit_d_cv)
    row("B_cv_wd{}".format(w_d), lab_cv)
    if w_d == 0.5:
        # file written for the full match uses the single full-window model
        row("B_full_wd0.5", run(vis, 1.0, 0.5, 0.0, 25), write="B_deadball_wd0.5")
