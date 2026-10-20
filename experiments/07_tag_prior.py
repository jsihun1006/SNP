"""Smart Tagging set-piece prior for dead ball: extra D evidence in a window around each set-piece tag start.

Checks (1) how often a tag start lies inside a predicted dead stretch with no prior, (2) the effect of the prior on
structure, and (3) the public score as a sanity check only (the public window holds just 2 relevant tags).
"""
import importlib
import numpy as np
from kloppy import sportscode

import common as c

combined = importlib.import_module("05_combined")
d, g, vis, ld_full, ld_cv, e = (combined.d, combined.g, combined.vis, combined.ld_full, combined.ld_cv, combined.e)
import decoder as dec

FPS = c.FPS
ev = sportscode.load(c.asset_paths()["smart_tagging"]).to_df()
ev["sf"] = (ev["timestamp"].dt.total_seconds() * FPS).round().astype(int)
sp = ev[(ev["code"].str.upper() == "SET PIECES") & (~ev["Type"].astype(str).str.contains("Start"))]
starts = sp["sf"].to_numpy()
starts = starts[g[np.clip(starts, 0, c.N_FRAMES - 1)]]
print("set-piece tag starts inside graded halves (excl. match kick-off):", len(starts))


# rebuild the candidate decode with the tag prior added to the D emission
logit_h = combined.logit_h


def decode(ball_vis, ld, lam, w_d, w_tag, pre=5, post=5, w_fb=1.0):
    prior = np.zeros(c.N_FRAMES)
    for s in starts:
        prior[max(s - pre * FPS, 0):s + post * FPS + 1] = 1.0
    team_ev = np.where(ball_vis, e, w_fb * logit_h)
    s_ = dec.make_scores(team_ev, d_bias=0.0, d_evidence=w_d * ld + w_tag * prior)
    return dec.decode(s_, lam, lam)


def stats(lab):
    inside = np.mean([lab[s] == "D" for s in starts])
    spells = []
    for _, a, b in c.half_video_blocks(c.GAME):
        seg = lab[a:b + 1]; st = 0
        for i in range(1, len(seg) + 1):
            if i == len(seg) or seg[i] != seg[st]:
                if seg[st] in "HA": spells.append(i - st)
                st = i
    return inside, len(spells), np.mean(spells) / FPS, np.mean(lab[g] == "D")


print("{:>5} {:>5} {:>6} | {:>16} {:>7} {:>8} {:>7} | public cross-fit: {:>8} {:>8} {:>5}".format(
    "lam", "w_d", "w_tag", "tag start is D", "spells", "mean_s", "D share", "spellF1", "macroF1", "poss"))
for lam, w_d in ((15, 0.25), (25, 0.25), (15, 0.1), (25, 0.5)):
    for w_tag in (0.0, 0.5, 1.0, 2.0):
        lab = decode(vis, ld_full, lam, w_d, w_tag)
        ins, n, m, ds = stats(lab)
        r = c.score(decode(vis, ld_cv, lam, w_d, w_tag))
        print("{:>5} {:>5} {:>6} | {:>16.2f} {:>7} {:>8.1f} {:>7.2f} | {:>25.4f} {:>8.4f} {:>5}".format(
            lam, w_d, w_tag, ins, n, m, ds, r["spell_f1_mean"], r["macro_f1"], r["n_pred_spells"]))
