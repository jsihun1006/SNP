"""A richer distance rule: 'the last team at the ball keeps it' (DFL, 25 fps, ball available, no learning).

    python possession_pathcrf/check_touch_rule.py

Rules (team per in-play frame; dead frames = DFL truth):
  nearest            team of the nearest player to the ball every frame (the plain distance rule)
  last_touch_R       a player within R m of the ball touches it; the ball stays with that team until a player
                     of the other team touches it (in flight / loose the last team keeps it)
  last_touch_R_slow  as last_touch_R, but a pass-by only counts as a touch when the ball also slows down or
                     turns there (speed drop > 20 % or turn > 30 deg within 0.2 s) or stays within R for 0.2 s
Scored on every in-play frame and on the non-contested frames of the hybrid (|P(H)-P(A)| >= 0.3):
frame accuracy, and spellF1 over the whole match (DFL truth and touch labels).
"""

import os
import sys

import numpy as np
import pandas as pd

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
import eval_25fps_step as st  # noqa: E402

RADII = (1.0, 1.5, 2.0)


def touches(d, radius, slow):
    who, dmin = st.nearest_with_dist(d["players"], d["ball"], d["team"])
    near = np.isfinite(dmin) & (dmin <= radius)
    if slow:
        ball = d["ball"]
        lag = 5
        v = np.full_like(ball, np.nan)
        v[lag:] = (ball[lag:] - ball[:-lag]) / 0.2
        sp = np.linalg.norm(v, axis=1)
        prev_v = np.full_like(v, np.nan)
        prev_v[lag:] = v[:-lag]
        prev_sp = np.linalg.norm(prev_v, axis=1)
        with np.errstate(invalid="ignore", divide="ignore"):
            turn = np.degrees(np.arccos(np.clip(np.nansum(v * prev_v, axis=1) / (sp * prev_sp), -1, 1)))
            drop = (prev_sp - sp) / prev_sp
        stay = np.convolve(near.astype(int), np.ones(lag, int), mode="same") >= lag
        near &= (np.nan_to_num(drop) > 0.2) | (np.nan_to_num(turn) > 30) | stay | (np.nan_to_num(sp) < 2)
    team_at = np.where(near, d["team"][who], -1)
    idx = np.where(team_at >= 0, np.arange(len(team_at)), -1)
    last = np.maximum.accumulate(idx)
    out = np.where(last >= 0, team_at[np.maximum(last, 0)], 0)
    return out


def main():
    import dfl_load
    import possession_metrics as pm
    from eval_dfl import codes
    pd.set_option("display.width", 200)
    rows = []
    for m in dfl_load.MATCHES:
        d = st.load(m, add_team=False)
        who, dmin = st.nearest_with_dist(d["players"], d["ball"], d["team"])
        rules = {"nearest": np.where(np.isfinite(dmin), d["team"][who], 0)}
        for r in RADII:
            rules["last_touch_{:g}".format(r)] = touches(d, r, False)
            rules["last_touch_{:g}_slow".format(r)] = touches(d, r, True)
        calm = d["alive"] & (np.abs(2 * d["p_home"] - 1) >= 0.3)
        tt = np.asarray(d["tt"])
        touch_ok = np.isin(tt, ["H", "A"])
        touch_ref = np.where(tt == "A", 1, 0)
        for name, team in rules.items():
            r = {"match": m, "rule": name,
                 "acc_all_dfl": float((team[d["alive"]] == d["gt"][d["alive"]]).mean()),
                 "acc_calm_dfl": float((team[calm] == d["gt"][calm]).mean()),
                 "acc_all_touch": float((team[d["alive"] & touch_ok] == touch_ref[d["alive"] & touch_ok]).mean())}
            r.update({k: v for k, v in st.score(d, team).items() if k.startswith("spellF1") or k.endswith("spells")})
            rows.append(r)
        print("done", m, flush=True)
    t = pd.DataFrame(rows)
    print("\n== rules, mean over 7 matches (dead = truth) ==")
    print(t.groupby("rule", sort=False)[["acc_all_dfl", "acc_calm_dfl", "acc_all_touch", "spellF1_dfl",
                                         "spellF1_touch", "pred_spells", "true_spells"]].mean().round(4).to_string())
    print("\nreference (models): hybrid+team 25 fps spellF1 DFL 0.5636 / touch 0.55; penalty 40 + 4x jump 0.5828 / 0.5238")


if __name__ == "__main__":
    main()
