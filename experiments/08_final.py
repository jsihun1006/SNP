"""Frozen candidate: segment decoder + team-shape fallback + dead-ball evidence + Smart Tagging set-piece prior.

Configuration fixed from whole-match structure and dead-share evidence (experiments 06/07/09), not from the public score:
  switch cost 25 frames, fallback weight 1.0, dead-ball weight 0.5, set-piece prior weight 1.0 over tag start +/-5 s.
Writes evaluation/results/VARcelona_candidate.csv. Run: python experiments/08_final.py
"""
import importlib
import numpy as np
from kloppy import sportscode

import common as c
import decoder as dec

CFG = dict(lam=25, w_fb=1.0, w_d=0.5, w_tag=1.0, tag_pre=5, tag_post=5)
combined = importlib.import_module("05_combined")
d, g, vis, e, logit_h = combined.d, combined.g, combined.vis, combined.e, combined.logit_h
FPS = c.FPS


def tag_prior():
    ev = sportscode.load(c.asset_paths()["smart_tagging"]).to_df()
    sp = ev[(ev["code"].str.upper() == "SET PIECES") & (~ev["Type"].astype(str).str.contains("Start"))]
    starts = (sp["timestamp"].dt.total_seconds() * FPS).round().astype(int).to_numpy()
    prior = np.zeros(c.N_FRAMES)
    for s in starts:
        prior[max(s - CFG["tag_pre"] * FPS, 0):s + CFG["tag_post"] * FPS + 1] = 1.0
    return prior


PRIOR = tag_prior()


def run(ball_vis, ld):
    team_ev = np.where(ball_vis, e, CFG["w_fb"] * logit_h)
    scores = dec.make_scores(team_ev, d_bias=0.0, d_evidence=CFG["w_d"] * ld + CFG["w_tag"] * PRIOR)
    return dec.decode(scores, CFG["lam"], CFG["lam"])


def show(name, lab):
    r = c.score(lab)
    print("{:<30} spellF1 {:.4f} macroF1 {:.4f} poss {:>3} dur {:>5.1f}".format(
        name, r["spell_f1_mean"], r["macro_f1"], r["n_pred_spells"], r["mean_spell_duration_s"]))


if __name__ == "__main__":
    print("config:", CFG)
    final = run(vis, combined.ld_full)
    c.write_csv("{}/VARcelona_candidate.csv".format(c.RESULTS), final)
    show("public, dead-ball cross-fitted", run(vis, combined.ld_cv))
    show("public, full-match file", final)
    gm = final[g]
    print("full match graded: H {:.3f} A {:.3f} D {:.3f}".format(*(np.mean(gm == k) for k in "HAD")))
    a, b = 11185, 18684
    L = b - a + 1
    res = []
    for s0 in [30000, 45000, 60000, 75000, 125000, 140000, 155000, 170000]:
        m = vis.copy(); m[a:b + 1] = vis[s0:s0 + L] & vis[a:b + 1]
        r = c.score(run(m, combined.ld_cv)); res.append((r["spell_f1_mean"], r["macro_f1"]))
    res = np.array(res)
    print("ball masked (8 real gap patterns): spellF1 {:.4f} +/- {:.4f}  macroF1 {:.4f}".format(res[:, 0].mean(), res[:, 0].std(), res[:, 1].mean()))
