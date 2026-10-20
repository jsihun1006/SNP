"""Experiment C: replace the fixed majority-vote window with a Viterbi segment decoder.

Same evidence as the baseline (nearest player to the ball); what changed is the temporal model:
H<->A switches and entering/leaving D each cost a fixed penalty (in units of per-frame evidence).
Missing ball = no team evidence, so the decoder carries the previous possession through the gap.
"""
import numpy as np
import common as c
import decoder as dec

d = c.load_tracking()
gt = c.load_gt()
dh, da = dec.nearest_team_distances(d["pos"], d["ball"], d["team"])
e = dec.nearest_evidence(dh, da)
vis = np.isfinite(d["ball"][:, 0])


def report(name, labels, write=True):
    r = c.score(labels)
    g = d["graded"]
    shares = {k: np.mean(labels[g] == k) for k in "HAD"}
    full_poss = sum(((labels[a:b + 1][1:] != labels[a:b + 1][:-1]) & (labels[a:b + 1][1:] != "D")
                     & (labels[a:b + 1][1:] != "X")).sum() for _, a, b in c.half_video_blocks(c.GAME))
    print("{:<34} spellF1 {:.4f} macroF1 {:.4f} poss {:>3} dur {:>5.1f} | full match: H {:.2f} A {:.2f} D {:.2f} ~{} poss starts".format(
        name, r["spell_f1_mean"], r["macro_f1"], r["n_pred_spells"], r["mean_spell_duration_s"],
        shares["H"], shares["A"], shares["D"], full_poss))
    if write:
        c.write_csv("{}/{}.csv".format(c.RESULTS, name), labels)
    return r


print("public GT: {} possessions, mean {:.1f}s".format(17, 14.2))
# fixed grid chosen from football reasoning (switch ~ 0.5-4 s of evidence), reported in full, not picked on the public score
for lam in (12, 25, 50, 100):
    # ball missing -> D (baseline behaviour) vs missing -> carry previous possession
    s = dec.make_scores(e, d_bias=None)
    report("C_carry_lam{}".format(lam), dec.decode(s, lam, lam))
    s = dec.make_scores(e, d_evidence=np.where(vis, -0.5, 0.5))
    report("C_missingD_lam{}".format(lam), dec.decode(s, lam, lam))
