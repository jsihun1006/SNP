"""The contest structure at 25 fps (DFL, leave-one-match-out, ball available).

    python possession_pathcrf/eval_contest25.py

Structure (as proposed):
  1. Per frame, P(H) from the hybrid PathCRF-lite node scores (saved in outputs/25fps).
  2. Contested = in play and |P(H) - P(A)| < TAU (0.3, chosen on DFL at 5 fps). The rest is not contested.
  3. Contest model, trained only on contested frames (5-fps rows of the other 6 matches, target = touch-label team),
     with information that matters in a contest:
       per team: nearest and 2nd-nearest distance to the ball, closing speed, time to reach the ball
                 (distance / closing speed), players within 3 m, whether the ball is heading to its nearest player
       duel distance (nearest home player to nearest away player), ball speed, ball direction change over 0.2 s
       last team at the ball and seconds since, next team at the ball and seconds until (offline)
       P(H) of the hybrid
     Applied only on contested frames (log probability added to every player's sender score of that team).
  4. Not contested: the hybrid scores alone.
  5. Contest-outcome model: per contest episode, the team holding the ball OUTCOME_S after it ends (touch labels),
     from the episode and the OUTCOME_S after it; added for OUTCOME_S from the episode end.
Variants compared on the same decoder (switch penalty from PENALTIES, illegal jump = 4 x penalty):
  hybrid            no team or contest model
  allframes_team    the all-frames team model of eval_25fps.py (the previous best)
  contest           structure 1-4
  contest_outcome   structure 1-5
For each test match the penalty is chosen on the other 6 matches (DFL truth); touch-label scores are reported too.
Outputs: outputs/25fps_contest/ (parameters, per-match results, chosen settings, per-frame contest probabilities).
"""

import os
import sys
import time

import numpy as np
import pandas as pd

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
import eval_25fps_step as st  # noqa: E402
from eval_dfl3 import CONTROL_OBS_M  # noqa: E402

TAU = 0.3
OUTCOME_S = 1.0
PENALTIES = (20.0, 40.0)
OUT = os.path.join(HERE, "outputs", "25fps_contest")
CONTEST = dict(objective="binary", learning_rate=0.05, num_leaves=31, min_data_in_leaf=50, verbose=-1, seed=7)
OUTCOME = dict(objective="binary", learning_rate=0.05, num_leaves=15, min_data_in_leaf=20, verbose=-1, seed=7)


def ffill_index(mask):
    """For each frame, index of the last frame where mask is True (-1 before any)."""
    idx = np.where(mask, np.arange(len(mask)), -1)
    return np.maximum.accumulate(idx)


def bfill_index(mask):
    n = len(mask)
    idx = np.where(mask, np.arange(n), n)
    return np.minimum.accumulate(idx[::-1])[::-1]


def contest_features(d):
    p, ball, team = d["players"], d["ball"], d["team"]
    n, lag = len(ball), 5
    vel = np.full_like(p, np.nan)
    vel[lag:] = (p[lag:] - p[:-lag]) / 0.2
    bvel = np.full_like(ball, np.nan)
    bvel[lag:] = (ball[lag:] - ball[:-lag]) / 0.2
    to_ball = ball[:, None, :] - p
    dist = np.linalg.norm(to_ball, axis=2)
    with np.errstate(invalid="ignore", divide="ignore"):
        closing = np.nansum(vel * to_ball, axis=2) / dist
    bspeed = np.linalg.norm(bvel, axis=1)
    prev = np.full_like(bvel, np.nan)
    prev[lag:] = bvel[:-lag]
    with np.errstate(invalid="ignore", divide="ignore"):
        turn = np.degrees(np.arccos(np.clip(np.nansum(bvel * prev, axis=1) /
                                            (np.linalg.norm(bvel, axis=1) * np.linalg.norm(prev, axis=1)), -1, 1)))
    cols, near_idx = [], []
    for k in (0, 1):
        dk = np.where((team == k)[None, :] & np.isfinite(dist), dist, np.inf)
        srt = np.sort(dk, axis=1)
        i = dk.argmin(axis=1)
        near_idx.append(i)
        d1 = np.where(np.isfinite(srt[:, 0]), srt[:, 0], 60)
        c1 = closing[np.arange(n), i]
        with np.errstate(invalid="ignore", divide="ignore"):
            to_player = -to_ball[np.arange(n), i]                       # ball -> that player
            heading = np.nansum(bvel * to_player, axis=1) / (bspeed * np.linalg.norm(to_player, axis=1))
        cols += [d1, np.where(np.isfinite(srt[:, 1]), srt[:, 1], 60), c1,
                 np.clip(d1 / np.maximum(np.nan_to_num(c1, nan=0), 0.5), 0, 60), (dk < 3).sum(axis=1), heading]
    duel = np.linalg.norm(p[np.arange(n), near_idx[0]] - p[np.arange(n), near_idx[1]], axis=1)
    who, dmin = st.nearest_with_dist(p, ball, team)
    at_ball = dmin <= CONTROL_OBS_M
    last = ffill_index(at_ball)
    nxt = bfill_index(at_ball)
    last_team = np.where(last >= 0, team[who[np.maximum(last, 0)]], -1)
    next_team = np.where(nxt < n, team[who[np.minimum(nxt, n - 1)]], -1)
    since = np.where(last >= 0, (np.arange(n) - last) / 25, 60)
    until = np.where(nxt < n, (nxt - np.arange(n)) / 25, 60)
    cols += [duel, bspeed, turn, last_team, np.clip(since, 0, 60), next_team, np.clip(until, 0, 60), d["p_home"]]
    return np.stack(cols, axis=1).astype(np.float32)


def episodes(contested, alive):
    e = np.diff(np.r_[False, contested, False].astype(np.int8))
    out, cur = [], None
    for a, b in zip(np.flatnonzero(e == 1), np.flatnonzero(e == -1)):
        if cur and a - cur[1] <= 10 and alive[cur[1]:a].all():
            cur = (cur[0], b)
        else:
            if cur:
                out.append(cur)
            cur = (a, b)
    if cur:
        out.append(cur)
    return out


def episode_rows(d, eps, base_team, X):
    n, look = len(base_team), int(OUTCOME_S * 25)
    feats, ys, spans = [], [], []
    for a, b in eps:
        e = min(b + look, n - 1)
        after = X[b:e] if e > b else X[b - 1:b]
        f = [(b - a) / 25, float(d["p_home"][a:b].mean()), float(X[a:b, 0].mean() - X[a:b, 6].mean()),
             float(base_team[b - 1]) if b > 0 else 0.5]
        f += list(np.nanmean(after, axis=0))
        feats.append(f)
        ys.append(d["tt"][e] if d["tt"][e] in ("H", "A") else "")
        spans.append((b, e))
    y = np.array([0 if v == "H" else 1 if v == "A" else -1 for v in ys])
    return np.array(feats, dtype=np.float32), y, spans


def add_team(ls, team, frames, q_home):
    home = team == 0
    out = ls.copy()
    out[np.ix_(frames, np.flatnonzero(home))] += np.log(q_home)[:, None]
    out[np.ix_(frames, np.flatnonzero(~home))] += np.log(1 - q_home)[:, None]
    return out


def main():
    import lightgbm as lgb
    import dfl_load
    pd.set_option("display.width", 200)
    os.makedirs(OUT, exist_ok=True)
    t0 = time.time()
    data = []
    for m in dfl_load.MATCHES:
        d = st.load(m, add_team=False)
        d["tt"] = np.asarray(d["tt"], dtype=object)
        d["X"] = contest_features(d)
        d["contested"] = d["alive"] & (np.abs(2 * d["p_home"] - 1) < TAU)
        data.append(d)
    print("loaded {:.0f}s".format(time.time() - t0), flush=True)
    rows, diag = [], []
    for k, d in enumerate(data):
        # contest model: contested 5-fps rows of the other matches, touch-label target
        Xs, ys = [], []
        for j, o in enumerate(data):
            if j != k:
                m = o["contested"] & np.isin(o["tt"], ["H", "A"]) & (np.arange(len(o["tt"])) % 5 == 0)
                Xs.append(o["X"][m])
                ys.append((o["tt"][m] == "H").astype(np.int8))
        booster = lgb.train(CONTEST, lgb.Dataset(np.concatenate(Xs), np.concatenate(ys)), 200)
        q = np.clip(booster.predict(d["X"]), 1e-4, 1 - 1e-4)
        d["q_contest"] = q
        cf = np.flatnonzero(d["contested"])
        lab = np.isin(d["tt"][cf], ["H", "A"])
        diag.append({"match": d["m"], "contested_share": float(d["contested"][d["alive"]].mean()),
                     "contest_model_acc": float(((q[cf][lab] > .5) == (d["tt"][cf][lab] == "H")).mean()),
                     "hybrid_acc_in_contest": float(((d["p_home"][cf][lab] > .5) == (d["tt"][cf][lab] == "H")).mean())})
        variants = {"hybrid": d["ls"], "allframes_team": add_team(d["ls"], d["team"], np.flatnonzero(d["alive"]),
                                                                  d["q"][d["alive"]]),
                    "contest": add_team(d["ls"], d["team"], cf, q[cf])}
        for name, ls in variants.items():
            for lam in PENALTIES:
                team, _ = st.decode(d, ls, d["lr"], lam)
                r = {"match": d["m"], "variant": name, "lam": lam}
                r.update(st.score(d, team))
                rows.append(r)
                if name == "contest" and lam == PENALTIES[0]:
                    d["contest_team"] = team
        print("contest", d["m"], "{:.0f}s".format(time.time() - t0), flush=True)
        pd.DataFrame({"frame": np.arange(len(q)), "contested": d["contested"].astype(np.int8),
                      "p_home_hybrid": np.round(d["p_home"], 4), "q_home_contest": np.round(q, 4)}).to_csv(
            os.path.join(OUT, "contest_probs_{}.csv.gz".format(d["m"])), index=False)
    # contest-outcome model (needs every match's contest decode)
    for k, d in enumerate(data):
        Xs, ys = [], []
        for j, o in enumerate(data):
            if j != k:
                X, y, _ = episode_rows(o, episodes(o["contested"], o["alive"]), o["contest_team"], o["X"])
                Xs.append(X[y >= 0])
                ys.append((y[y >= 0] == 0).astype(np.int8))
        booster = lgb.train(OUTCOME, lgb.Dataset(np.concatenate(Xs), np.concatenate(ys)), 200)
        X, y, spans = episode_rows(d, episodes(d["contested"], d["alive"]), d["contest_team"], d["X"])
        qo = np.clip(booster.predict(X), 1e-4, 1 - 1e-4)
        cf = np.flatnonzero(d["contested"])
        q = d["q_contest"]
        ls = add_team(d["ls"], d["team"], cf, q[cf])
        home = d["team"] == 0
        for (b, e), qh in zip(spans, qo):
            ls[b:e, home] += np.log(qh)
            ls[b:e, ~home] += np.log(1 - qh)
        ok = y >= 0
        diag[k]["outcome_acc"] = float(((qo[ok] > .5) == (y[ok] == 0)).mean()) if ok.any() else np.nan
        diag[k]["episodes"] = len(spans)
        for lam in PENALTIES:
            team, _ = st.decode(d, ls, d["lr"], lam)
            r = {"match": d["m"], "variant": "contest_outcome", "lam": lam}
            r.update(st.score(d, team))
            rows.append(r)
        print("outcome", d["m"], "{:.0f}s".format(time.time() - t0), flush=True)
    t = pd.DataFrame(rows)
    t.to_csv(os.path.join(OUT, "results_per_match.csv"), index=False)
    g = pd.DataFrame(diag)
    g.to_csv(os.path.join(OUT, "contest_diagnosis.csv"), index=False)
    pd.DataFrame([("TAU", TAU), ("OUTCOME_S", OUTCOME_S), ("PENALTIES", ",".join(map(str, PENALTIES))),
                  ("BIG_RATIO", st.BIG_RATIO), ("CONTROL_OBS_M", CONTROL_OBS_M)] +
                 [("contest_lgbm_" + a, b) for a, b in CONTEST.items()] +
                 [("outcome_lgbm_" + a, b) for a, b in OUTCOME.items()],
                 columns=["parameter", "value"]).to_csv(os.path.join(OUT, "parameters.csv"), index=False)
    print("\n== contest diagnosis (mean over 7) ==")
    print(g.drop(columns="match").mean().round(4).to_string())
    print("\n== mean over 7 test matches by variant and penalty ==")
    print(t.groupby(["variant", "lam"], sort=False)[["spellF1_dfl", "spellF1_touch", "frame_acc", "pred_spells",
                                                    "true_spells"]].mean().round(4).to_string())
    print("\n== penalty chosen on the other 6 matches (DFL truth), per variant ==")
    chosen = []
    for name, tv in t.groupby("variant", sort=False):
        pk = st.choose(tv, ["lam"])
        chosen.append(pk.assign(variant=name))
        print("{:16s} spellF1 DFL {:.4f}  touch {:.4f}  frame acc {:.4f}  better/worse vs hybrid: computed below".format(
            name, pk["spellF1_dfl"].mean(), pk["spellF1_touch"].mean(), pk["frame_acc"].mean()))
    pd.concat(chosen).to_csv(os.path.join(OUT, "chosen.csv"), index=False)


if __name__ == "__main__":
    main()
