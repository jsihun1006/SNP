"""Settle switch cost and dead-ball weight from whole-match structure, not the public score; test the Smart Tagging prior.

Part 1: for each (lam, w_d) decode the FULL match and report structural statistics only -- possession count and
mean duration (public GT: 14.2 s), dead-ball share, number of dead stretches >= 2 s -- plus how well the predicted
dead stretches line up with Smart Tagging set pieces (an independent, inferred source).
Part 2: timing relation between each set-piece tag and the nearest predicted dead stretch.
"""
import numpy as np
from kloppy import sportscode

import common as c
import importlib
combined = importlib.import_module("05_combined")
d, g, vis, ld_full, pipeline = (combined.d, combined.g, combined.vis, combined.ld_full, combined.pipeline)

FPS = c.FPS
P = c.asset_paths()
ev = sportscode.load(P["smart_tagging"]).to_df()
ev["sf"] = (ev["timestamp"].dt.total_seconds() * FPS).round().astype(int)
ev["ef"] = (ev["end_timestamp"].dt.total_seconds() * FPS).round().astype(int)
sp = ev[ev["code"].str.upper() == "SET PIECES"].copy()
sp["Type"] = sp["Type"].astype(str)
in_half = lambda f: any(a <= f <= b for _, a, b in c.half_video_blocks(c.GAME))
sp = sp[sp.sf.map(in_half) | sp.ef.map(in_half)]
print("set-piece tags inside the halves:", len(sp), sp["Type"].value_counts().to_dict())


def runs(labels, which):
    out = []
    for _, a, b in c.half_video_blocks(c.GAME):
        x = labels[a:b + 1] == which
        dd = np.diff(np.concatenate(([0], x.astype(int), [0])))
        out += [(a + s, a + t - 1) for s, t in zip(np.where(dd == 1)[0], np.where(dd == -1)[0])]
    return out


def structure(labels):
    spells = []
    for _, a, b in c.half_video_blocks(c.GAME):
        seg = labels[a:b + 1]
        start = 0
        for i in range(1, len(seg) + 1):
            if i == len(seg) or seg[i] != seg[start]:
                if seg[start] in "HA":
                    spells.append(i - start)
                start = i
    dead = runs(labels, "D")
    return len(spells), np.mean(spells) / FPS, dead


def tag_alignment(dead):
    dead_long = [(s, t) for s, t in dead if t - s >= 2 * FPS]
    starts = np.array([s for s, _ in dead_long]); ends = np.array([t for _, t in dead_long])
    hit = 0
    for _, r in sp.iterrows():
        if any((s <= r.ef) and (t >= r.sf - 10 * FPS) for s, t in dead_long):
            hit += 1
    near = sum(any((r.sf - 10 * FPS) <= s <= r.ef for _, r in sp.iterrows()) for s in starts)
    return len(dead_long), hit / len(sp), near / max(len(dead_long), 1)


print("\nPart 1 -- full-match structure (target mean possession ~14.2 s from the public truth)")
print("{:>5} {:>5} | {:>6} {:>8} {:>7} | {:>8} {:>13} {:>14}".format(
    "lam", "w_d", "spells", "mean_s", "D share", "D>=2s", "tag recall", "D near a tag"))
table = {}
for w_d in (0.0, 0.1, 0.25, 0.5):
    for lam in (10, 15, 20, 25, 30, 40, 60):
        lab = pipeline(vis, ld_full, lam=lam, w_d=w_d)
        n, m, dead = structure(lab)
        nd, rec, prec = tag_alignment(dead)
        table[(lam, w_d)] = (n, m)
        print("{:>5} {:>5} | {:>6} {:>8.1f} {:>7.2f} | {:>8} {:>13.2f} {:>14.2f}".format(
            lam, w_d, n, m, np.mean(lab[g] == "D"), nd, rec, prec))

# Part 2: timing between tags and predicted dead stretches (w_d=0.25, lam=25)
lab = pipeline(vis, ld_full, lam=25, w_d=0.25)
dead = [(s, t) for s, t in runs(lab, "D") if t - s >= 2 * FPS]
print("\nPart 2 -- tag start/end relative to nearest predicted dead stretch (seconds; negative = tag earlier than D)")
for typ, grp in sp.groupby("Type"):
    ds, de = [], []
    for _, r in grp.iterrows():
        if dead:
            j = int(np.argmin([abs(s - r.sf) for s, _ in dead]))
            ds.append((dead[j][0] - r.sf) / FPS); de.append((dead[j][1] - r.sf) / FPS)
    if ds:
        print("  {:<22} n={:<3} D-start - tag-start: median {:+6.1f}s IQR [{:+.1f}, {:+.1f}]   D-end - tag-start: median {:+6.1f}s".format(
            typ, len(ds), np.median(ds), np.percentile(ds, 25), np.percentile(ds, 75), np.median(de)))
