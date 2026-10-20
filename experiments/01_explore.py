"""Step 1 exploration: tracking coverage, missing-ball structure, Smart Tagging vs public GT."""
import os, sys
import numpy as np, pandas as pd
ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)
from games import asset_paths, half_video_blocks
from kloppy import metrica, sportscode

FPS = 25
P = asset_paths()
ds = metrica.load_tracking_epts(meta_data=P["atd_meta"], raw_data=P["atd_raw"])
df = ds.to_df()
n = len(df)
vf = df["frame_id"].to_numpy() - 1
xcols = [c for c in df.columns if c.endswith("_x") and c != "ball_x"]
ycols = [c.replace("_x", "_y") for c in xcols]
print("frames", n, "frame_id range", df.frame_id.min(), df.frame_id.max(), "player slots", len(xcols))
nt = df[xcols].notna().sum(axis=1).to_numpy()
graded = np.zeros(n, bool)
for _, a, b in half_video_blocks("BAR-ATH"):
    graded[a:b + 1] = True
print("graded frames", graded.sum())
for name, m in (("all", np.ones(n, bool)), ("graded", graded)):
    print(f"[{name}] mean tracked {nt[m].mean():.2f}  >22: {(nt[m] > 22).mean()*100:.2f}%  >22 count {(nt[m]>22).sum()}  max {nt[m].max()}")

ball = df["ball_x"].notna().to_numpy()
miss = ~ball
print(f"ball missing: all {miss.mean()*100:.1f}%  graded {miss[graded].mean()*100:.1f}%")
for h, a, b in half_video_blocks("BAR-ATH"):
    print(f"  half {h}: missing {miss[a:b+1].mean()*100:.1f}%")

# gap lengths inside graded halves
gaps = []
for _, a, b in half_video_blocks("BAR-ATH"):
    m = miss[a:b + 1].astype(int)
    d = np.diff(np.concatenate(([0], m, [0])))
    s, e = np.where(d == 1)[0], np.where(d == -1)[0]
    gaps += list(zip(s + a, e - s))
L = np.array([g[1] for g in gaps])
print(f"gaps {len(L)}  median {np.median(L):.0f}f  mean {L.mean():.1f}f  max {L.max()}f ({L.max()/FPS:.1f}s)")
for lo, hi in [(1, 5), (6, 12), (13, 25), (26, 75), (76, 250), (251, 10**9)]:
    k = (L >= lo) & (L <= hi)
    print(f"  {lo:>4}-{hi if hi<10**9 else 'inf':>5} frames: {k.sum():5d} gaps ({k.mean()*100:4.1f}%)  covering {L[k].sum()/L.sum()*100:4.1f}% of missing frames")

# where does the ball go missing? use last known ball position before each gap
bx = df["ball_x"].to_numpy(); by = df["ball_y"].to_numpy()
last = pd.Series(bx).ffill().to_numpy(); lasty = pd.Series(by).ffill().to_numpy()
nxt = pd.Series(bx).bfill().to_numpy()
def zone(x, y):
    # x 0..1 length; penalty area ~16.5m of 105.3 => 0.157 ; width 40.3 of 68 => y .204-.796
    inbox = ((x < 0.157) | (x > 0.843)) & (y > 0.204) & (y < 0.796)
    return inbox
pk = zone(last, lasty)
edge = (lasty < 0.04) | (lasty > 0.96)
print(f"graded frames w/ last-known ball in a penalty area: {pk[graded].mean()*100:.1f}% of frames; missing rate there {miss[graded & pk].mean()*100:.1f}% vs elsewhere {miss[graded & ~pk].mean()*100:.1f}%")
print(f"near touchline (<4% width): missing {miss[graded & edge].mean()*100:.1f}% (n={int((graded&edge).sum())})")
# ball speed before gap onset (aerial proxy: fast ball disappearing)
sp = np.hypot(np.diff(bx, prepend=np.nan) * 105.3, np.diff(by, prepend=np.nan) * 68) * FPS
onset = [s for s, _ in gaps if s > 0 and not np.isnan(sp[s - 1])]
print(f"ball speed on frame before gap onset: median {np.nanmedian(sp[np.array(onset)-1]):.1f} m/s vs overall visible median {np.nanmedian(sp[graded & ball]):.1f} m/s;"
      f" share >15 m/s: {np.mean(sp[np.array(onset)-1]>15)*100:.1f}% vs {np.nanmean(sp[graded&ball]>15)*100:.1f}%")
# players near ball-exit: long gaps with ball re-appearing far away => jump
jump = []
for s, l in gaps:
    e = s + l
    if s > 0 and e < n and not np.isnan(bx[s - 1]) and not np.isnan(bx[e]):
        jump.append(np.hypot((bx[e]-bx[s-1])*105.3, (by[e]-by[s-1])*68))
jump = np.array(jump)
print(f"gap bridge distance (m): median {np.median(jump):.1f}  p90 {np.percentile(jump,90):.1f}; gaps >=25f with bridge >20m: {np.mean([j>20 for j,(s,l) in zip(jump,[g for g in gaps if g[0]>0]) if l>=25])*100:.1f}%")

# Smart tagging
ev = sportscode.load(P["smart_tagging"]).to_df()
ev["sf"] = (ev["timestamp"].dt.total_seconds() * FPS).round().astype(int)
ev["ef"] = (ev["end_timestamp"].dt.total_seconds() * FPS).round().astype(int)
print("\ntagging rows", len(ev), "cols", list(ev.columns))
print(ev["code"].value_counts().to_string())
print(pd.crosstab(ev["code"], ev["Team"]).to_string())
print(ev.groupby("Type")["Team"].value_counts().to_string() if "Type" in ev else "")
sp_ = ev[ev["code"].str.upper() == "SET PIECES"]
print("set pieces: duration median %.1fs" % ((sp_.ef - sp_.sf).median() / FPS))

gt = pd.read_csv(os.path.join(ROOT, "evaluation/gt/BAR-ATH_possession_gt_public.csv"))
g = gt[gt.label != "X"]
a, b = g.frame.min(), g.frame.max()
print(f"\npublic GT frames {a}..{b}  {g.label.value_counts().to_dict()}")
lab = gt.label.to_numpy()
runs = []
i = a
while i <= b:
    j = i
    while j + 1 <= b and lab[j + 1] == lab[i]: j += 1
    runs.append((lab[i], i, j)); i = j + 1
print("GT runs:", len(runs))
for l, s, e in runs: print(f"  {l} {s}-{e} ({(e-s+1)/FPS:.1f}s)  ball missing {miss[s:e+1].mean()*100:.0f}%")
w = ev[(ev.ef >= a) & (ev.sf <= b)].sort_values("sf")
print("\ntagging intervals overlapping the public window:")
cols = [c for c in ["code", "Team", "Type"] if c in w.columns]
for _, r in w.iterrows():
    print(f"  {r.sf}-{r.ef}  " + " | ".join(str(r[c]) for c in cols))
# how often does a tag's team agree with GT majority within interval
agree = tot = 0
for _, r in w.iterrows():
    if r["Team"] in ("Barcelona", "Atletic Club"):
        seg = lab[max(r.sf, a):min(r.ef, b) + 1]; seg = seg[(seg == "H") | (seg == "A")]
        if len(seg):
            maj = "H" if (seg == "H").mean() >= .5 else "A"
            tot += 1; agree += maj == ("H" if r["Team"] == "Barcelona" else "A")
print(f"team-tagged phases in window: {tot}, majority-GT-possession agrees: {agree}")
dead = [(s, e) for l, s, e in runs if l == "D"]
print("GT dead stretches:", dead)
