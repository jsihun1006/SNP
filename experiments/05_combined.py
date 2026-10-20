"""Combined candidate: C (segment decoder) + A (team-shape fallback) + B (dead-ball evidence).

Fixed configuration, chosen from the football reasoning in the notes and NOT tuned on the public score:
  switch cost 25 frames (~1 s of unit evidence), fallback weight 1.0, dead-ball weight 0.25.
The shape model never trains on the public window (+/-30 s); the dead-ball model is cross-fitted inside it.
Writes evaluation/results/VARcelona_candidate.csv. Variants below are sensitivity checks only.
"""
import warnings
import numpy as np
from sklearn.ensemble import HistGradientBoostingClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import make_pipeline

import common as c
import decoder as dec
import features as ft
import deadball as db

warnings.filterwarnings("ignore")
CFG = dict(lam=25, w_fb=1.0, w_d=0.25)
a, b = 11185, 18684
d = c.load_tracking()
gt = c.load_gt()
g = d["graded"]
vis = np.isfinite(d["ball"][:, 0])
dh, da = dec.nearest_team_distances(d["pos"], d["ball"], d["team"])
e = dec.nearest_evidence(dh, da)


def logit(p, k):
    return np.clip(np.log((p + 1e-3) / (1 - p + 1e-3)), -k, k)


# --- possession shape model (pseudo-labels from the visible-ball decode; public window held out) ---
X, _ = ft.build(d)
y = (dec.decode(dec.make_scores(e, d_bias=None), CFG["lam"], CFG["lam"]) == "H").astype(int)
rng = np.random.default_rng(0)
holdout = np.zeros(c.N_FRAMES, bool)
holdout[a - 750:b + 751] = True
idx = np.where(g & vis & ~holdout)[0]
idx = idx[rng.random(len(idx)) < 0.2]
shape = HistGradientBoostingClassifier(max_depth=4, max_iter=150, learning_rate=0.06, l2_regularization=1.0,
                                       random_state=0).fit(X[idx], y[idx])
logit_h = logit(shape.predict_proba(X)[:, 1], 3)

# --- dead-ball model: fitted on the public window (4 stretches); cross-fitted for scoring inside it ---
F = np.nan_to_num(db.build(d, dh, da), nan=0.0)
lab = np.zeros(c.N_FRAMES, bool)
lab[a:b + 1] = gt[a:b + 1] == "D"
win = np.zeros(c.N_FRAMES, bool)
win[a:b + 1] = True


def dead_model():
    return make_pipeline(StandardScaler(), LogisticRegression(C=0.1, max_iter=500))


p_full = dead_model().fit(F[win], lab[win]).predict_proba(F)[:, 1]
p_cv = p_full.copy()
edges = [a, 14400, 16900, 17900, b + 1]
for lo, hi in zip(edges[:-1], edges[1:]):
    tr = win.copy()
    tr[lo - 125:hi + 125] = False
    p_cv[lo:hi] = dead_model().fit(F[tr], lab[tr]).predict_proba(F[lo:hi])[:, 1]
ld_full, ld_cv = logit(p_full, 4), logit(p_cv, 4)


def pipeline(ball_vis, ld, lam=CFG["lam"], w_fb=CFG["w_fb"], w_d=CFG["w_d"], always=0.0):
    ev = np.where(ball_vis, e + always * logit_h, w_fb * logit_h)
    return dec.decode(dec.make_scores(ev, d_bias=0.0, d_evidence=w_d * ld), lam, lam)


def row(name, labels):
    r = c.score(labels)
    gm = labels[g]
    print("{:<36} spellF1 {:.4f} macroF1 {:.4f} poss {:>3} dur {:>5.1f} | full match H {:.2f} A {:.2f} D {:.2f}".format(
        name, r["spell_f1_mean"], r["macro_f1"], r["n_pred_spells"], r["mean_spell_duration_s"],
        np.mean(gm == "H"), np.mean(gm == "A"), np.mean(gm == "D")))


if __name__ == "__main__":
    print("public window (dead-ball cross-fitted; shape model held out). Baselines: smooth125 0.4242/0.5812, smooth50 0.3256/0.6207")
    final = pipeline(vis, ld_full)                     # file for the whole match: dead model fitted on the full public window
    c.write_csv("{}/VARcelona_candidate.csv".format(c.RESULTS), final)
    row("candidate (cross-fit on public)", pipeline(vis, ld_cv))
    row("candidate full-match file", final)
    print("\nsensitivity (cross-fit):")
    for lam in (12, 25, 40):
        row("  lam={}".format(lam), pipeline(vis, ld_cv, lam=lam))
    for w_d in (0.0, 0.5):
        row("  w_d={}".format(w_d), pipeline(vis, ld_cv, w_d=w_d))
    row("  shape evidence also when ball seen", pipeline(vis, ld_cv, always=0.5))

    print("\nball masked with real gap patterns from 8 other chunks (mean +/- sd):")
    L = b - a + 1
    out = {k: [] for k in ("candidate", "no fallback (w_fb=0)", "no dead-ball (w_d=0)")}
    for s0 in [30000, 45000, 60000, 75000, 125000, 140000, 155000, 170000]:
        m = vis.copy()
        m[a:b + 1] = vis[s0:s0 + L] & vis[a:b + 1]
        for k, kw in (("candidate", {}), ("no fallback (w_fb=0)", dict(w_fb=0.0)), ("no dead-ball (w_d=0)", dict(w_d=0.0))):
            r = c.score(pipeline(m, ld_cv, **kw))
            out[k].append((r["spell_f1_mean"], r["macro_f1"]))
    for k, v in out.items():
        v = np.array(v)
        print("  {:<24} spellF1 {:.4f} +/- {:.4f}  macroF1 {:.4f}".format(k, v[:, 0].mean(), v[:, 0].std(), v[:, 1].mean()))
