"""How much of the match is dead ball? Estimate the class prior on the full match under label shift.

Public window dead share is 19.6% from 4 stretches; the full match may differ. Three checks:
 1. Do the ball-missing rates in the public window explain the full-match 41% missing? (they cannot)
 2. Is the public window typical in player activity? (10 s mean player speed, window vs rest of the match)
 3. Saerens-style EM prior estimate: train P(dead | x) on the public window, then re-estimate the class prior on the
    whole match. Done with ball-free features only (player speeds) and with all features.
"""
import warnings
import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import make_pipeline

import common as c
import decoder as dec
import deadball as db
from features import _rolling_mean

warnings.filterwarnings("ignore")
d = c.load_tracking(); gt = c.load_gt()
g = d["graded"]
vis = np.isfinite(d["ball"][:, 0])
dh, da = dec.nearest_team_distances(d["pos"], d["ball"], d["team"])
F = np.nan_to_num(db.build(d, dh, da), nan=0.0)
a, b = 11185, 18684
win = np.zeros(c.N_FRAMES, bool); win[a:b + 1] = True
lab = np.zeros(c.N_FRAMES, bool); lab[a:b + 1] = gt[a:b + 1] == "D"
rest = g & ~win

# 1. missing-ball accounting
m_d, m_a = (~vis[win & lab]).mean(), (~vis[win & ~lab]).mean()
print("1. ball missing in public window: {:.0%} of dead frames, {:.0%} of live frames; full match graded {:.0%}".format(m_d, m_a, (~vis[g]).mean()))
print("   if public rates held, dead share would have to be {:.0%} (>100% = impossible): ball is lost far more in live play elsewhere".format(
    ((~vis[g]).mean() - m_a) / (m_d - m_a)))

# 2. is the public window typical?
spd10 = F[:, 1]   # 10 s mean player speed
print("2. 10 s mean player speed: public window mean {:.2f} m/s ({:.0f}th pct of the match); rest of match mean {:.2f}".format(
    spd10[win].mean(), (spd10[rest] < spd10[win].mean()).mean() * 100, spd10[rest].mean()))
print("   share of frames with 10 s speed < 1.6 m/s: window {:.2f}, rest {:.2f}, dead frames in window {:.2f}".format(
    (spd10[win] < 1.6).mean(), (spd10[rest] < 1.6).mean(), (spd10[win & lab] < 1.6).mean()))
chunks = np.array_split(np.where(g)[0], 21)
print("   per-5-min chunk, share of frames with speed < 1.6: " + " ".join("{:.2f}".format((spd10[ix] < 1.6).mean()) for ix in chunks))


def em_prior(p, prior0, iters=200):
    pi = prior0
    for _ in range(iters):
        adj = p * pi / prior0
        adj = adj / (adj + (1 - p) * (1 - pi) / (1 - prior0))
        pi = adj.mean()
    return pi


names = ["pspd2s", "pspd10s", "slowfrac10s", "vis2s", "vis10s", "ballspd", "nearest", "edge", "oob"]
print("3. EM prior estimate of match dead share (public prior {:.3f}):".format(lab[win].mean()))
for label, cols in (("player speed only (ball-free)", [0, 1, 2]), ("all features", list(range(9)))):
    mdl = make_pipeline(StandardScaler(), LogisticRegression(C=0.1, max_iter=500)).fit(F[win][:, cols], lab[win])
    p = mdl.predict_proba(F[rest][:, cols])[:, 1]
    prior0 = lab[win].mean()
    print("   {:<30} mean P(dead) {:.3f}   EM-adjusted prior {:.3f}   hard share(p>.5) {:.3f}".format(
        label, p.mean(), em_prior(np.clip(p, 1e-4, 1 - 1e-4), prior0), (p > .5).mean()))
