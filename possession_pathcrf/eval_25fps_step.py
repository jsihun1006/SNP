"""25-fps possession, step-by-step improvements re-decoded from the saved probabilities (no training, no features).

    python possession_pathcrf/eval_25fps_step.py big        # step 1: illegal-jump penalty = BIG_RATIO x switch penalty
    python possession_pathcrf/eval_25fps_step.py outcome    # step 2: + contest-outcome model
    python possession_pathcrf/eval_25fps_step.py short      # step 3: + merge predicted spells shorter than a minimum

Inputs: outputs/25fps/player_probs_<m>.csv.gz (sender / receiver probability per in-play player-frame) and
frames_<m>.csv.gz (team-model probability, truths). The hybrid ball bonus is recomputed from the DFL ball.
Every choice (penalty, minimum length) is made per test match on the other 6 matches (DFL truth, spellF1).
Outputs: outputs/25fps_<step>/ (results_per_match.csv, chosen.csv, parameters.csv, frames_<m>.csv.gz).
"""

import os
import sys
import time

import numpy as np
import pandas as pd

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
import pathcrf_lite as pc  # noqa: E402
import possession_metrics as pm  # noqa: E402
import features as fx  # noqa: E402
from eval_dfl import codes  # noqa: E402
from eval_dfl3 import CONTROL_OBS_M, HYBRID_BONUS, nearest_with_dist  # noqa: E402

SRC = os.path.join(HERE, "outputs", "25fps")
PENALTIES = (10.0, 20.0, 40.0)
BIG_RATIO = 4.0          # an illegal jump costs this many switch penalties (fixed before the run)


def load(m, add_team=True):
    import dfl_load
    import train_dfl
    tracking, _, meta = dfl_load.load_match(m)
    raw = fx.match_from_dfl(tracking, meta)
    oof = pd.read_parquet(os.path.join(train_dfl.oof_dir(), "{}_sim.parquet".format(m)))
    period = oof["period"].to_numpy().astype(np.int64)
    dead = oof["dead"].to_numpy().astype(bool)
    players = np.asarray(raw.players, dtype=np.float64)
    ball = np.asarray(raw.ball, dtype=np.float64)
    T, N = players.shape[:2]
    fr = pd.read_csv(os.path.join(SRC, "frames_{}.csv.gz".format(m)), keep_default_na=False)
    pp = pd.read_csv(os.path.join(SRC, "player_probs_{}.csv.gz".format(m)))
    ps, pr = np.full((T, N), 1e-6), np.full((T, N), 1e-6)
    present = np.zeros((T, N), dtype=bool)
    f, j = pp["frame"].to_numpy(), pp["player"].to_numpy()
    ps[f, j], pr[f, j], present[f, j] = pp["p_sender"].to_numpy(), pp["p_receiver"].to_numpy(), True
    ls, lr = np.log(np.clip(ps, 1e-6, 1)), np.log(np.clip(pr, 1e-6, 1))
    who, dmin = nearest_with_dist(players, ball, raw.team)
    r = np.flatnonzero(dmin <= CONTROL_OBS_M)
    ls[r, who[r]] += HYBRID_BONUS
    lr[r, who[r]] += HYBRID_BONUS
    q = np.clip(fr["q_home_team_model"].to_numpy(np.float64), 1e-4, 1 - 1e-4)
    alive = fr["alive"].to_numpy() == 1
    home = raw.team == 0
    a = np.flatnonzero(alive)
    if add_team:      # the all-frames team model of eval_25fps.py
        ls[np.ix_(a, np.flatnonzero(home))] += np.log(q[a])[:, None]
        ls[np.ix_(a, np.flatnonzero(~home))] += np.log(1 - q[a])[:, None]
    gt_team = np.where(fr["truth_dfl"].to_numpy() == "A", pm.A, pm.H)
    tt = fr["truth_touch"].to_numpy()
    touch = np.where(tt == "A", pm.A, pm.H)
    return dict(m=m, team=raw.team, period=period, dead=dead, alive=alive, present=present, ls=ls, lr=lr, tt=tt,
                gt=gt_team, touch=touch, p_home=fr["p_home_hybrid"].to_numpy(np.float64), q=q,
                players=players, ball=ball)


def decode(d, ls, lr, lam, fine=None, lam_frame=None):
    """Edge Viterbi per in-play stretch. fine: frames decoded at 25 fps (e.g. contested); elsewhere every
    pc.STEP-th frame, its scores multiplied by pc.STEP so a step weighs as much evidence per second as 25 fps.
    fine=None decodes every frame at 25 fps. Each step's state fills the frames up to the next step.
    lam_frame: optional per-frame multiplier of the switch penalty (illegal jump scales with it)."""
    sender = np.full(len(ls), -1)
    e = np.diff(np.r_[False, d["alive"], False].astype(np.int8))
    for a, b in zip(np.flatnonzero(e == 1), np.flatnonzero(e == -1)):
        mult = np.ones(b - a) if lam_frame is None else np.asarray(lam_frame[a:b], dtype=np.float64)
        if fine is None:
            su, _ = pc.viterbi_edges(ls[a:b], lr[a:b], d["present"][a:b], lam * mult, lam * mult,
                                     big=BIG_RATIO * lam * mult)
            sender[a:b] = su
            continue
        f = np.asarray(fine[a:b], dtype=bool)
        keep = f | (np.arange(b - a) % pc.STEP == 0)
        keep[0] = True
        steps = np.flatnonzero(keep)
        weight = np.diff(np.r_[steps, b - a]).astype(np.float64)[:, None]   # frames each step stands for
        m = mult[steps]
        su, _ = pc.viterbi_edges(ls[a:b][steps] * weight, lr[a:b][steps] * weight, d["present"][a:b][steps],
                                 lam * m, lam * m, big=BIG_RATIO * lam * m)
        sender[a:b] = np.repeat(su, weight[:, 0].astype(int))
    return np.where(sender >= 0, d["team"][np.maximum(sender, 0)], 0), sender


def score(d, team):
    pred = codes(team, d["dead"], d["period"])
    out = {}
    for name, truth in (("dfl", d["gt"]), ("touch", d["touch"])):
        gt = codes(truth, d["dead"], d["period"])
        sc = pm.spell_scores(gt, pred, pm.valid_runs(gt))
        out["spellF1_" + name] = sc["mean_f1"]
        if name == "dfl":
            out["pred_spells"], out["true_spells"] = sc["n_pred"], sc["n_gt"]
    out["frame_acc"] = float((team[d["alive"]] == d["gt"][d["alive"]]).mean())
    return out


def choose(t, key_cols):
    """Per test match: the setting with the best mean DFL spellF1 on the other 6 matches."""
    t = t.copy()
    t["key"] = t[key_cols].astype(str).agg("/".join, axis=1)
    sc = t.pivot(index="match", columns="key", values="spellF1_dfl")
    rows = []
    for m in sc.index:
        best = sc.drop(index=m).mean().idxmax()
        r = t[(t["match"] == m) & (t["key"] == best)].iloc[0].to_dict()
        rows.append(r)
    return pd.DataFrame(rows)


def run_big(data, out_dir):
    rows = []
    for d in data:
        for lam in PENALTIES:
            team, _ = decode(d, d["ls"], d["lr"], lam)
            r = {"match": d["m"], "lam": lam}
            r.update(score(d, team))
            rows.append(r)
        print("done", d["m"], flush=True)
    t = pd.DataFrame(rows)
    return t, choose(t, ["lam"])


def main():
    import dfl_load
    step = sys.argv[1] if len(sys.argv) > 1 else "big"
    pd.set_option("display.width", 200)
    out_dir = os.path.join(HERE, "outputs", "25fps_" + step)
    os.makedirs(out_dir, exist_ok=True)
    t0 = time.time()
    data = [load(m) for m in dfl_load.MATCHES]
    print("loaded in {:.0f}s".format(time.time() - t0), flush=True)
    t, pk = {"big": run_big}[step](data, out_dir)
    pd.DataFrame([("PENALTIES", ",".join(map(str, PENALTIES))), ("BIG_RATIO", BIG_RATIO),
                  ("CONTROL_OBS_M", CONTROL_OBS_M), ("HYBRID_BONUS", HYBRID_BONUS)],
                 columns=["parameter", "value"]).to_csv(os.path.join(out_dir, "parameters.csv"), index=False)
    t.to_csv(os.path.join(out_dir, "results_per_match.csv"), index=False)
    pk.to_csv(os.path.join(out_dir, "chosen.csv"), index=False)
    cols = ["spellF1_dfl", "spellF1_touch", "frame_acc", "pred_spells", "true_spells"]
    print("\n== mean over 7 test matches by setting ==")
    print(t.drop(columns="match").groupby([c for c in t.columns if c not in cols + ["match"]]).mean().round(4).to_string())
    print("\n== chosen on the other 6 matches ==")
    print(pk.round(4).to_string(index=False))
    print("mean: spellF1 DFL {:.4f}, touch {:.4f}, frame acc {:.4f}  (previous 25 fps: 0.5636)".format(
        pk["spellF1_dfl"].mean(), pk["spellF1_touch"].mean(), pk["frame_acc"].mean()))


if __name__ == "__main__":
    main()
