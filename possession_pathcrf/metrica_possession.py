"""Possession ground truth from the Metrica sample events (games 1-3), for validating possession models.

    python possession_pathcrf/metrica_possession.py          # build and summarise the truth

Per tracking frame (25 fps, pitch 105 x 68 m):
  dead       the existing Metrica dead-ball label (src/metrica_load.dead_spells / eval_metrica3.load_game3)
  team       in play: the team of the latest ball event (PASS, RECOVERY, BALL LOST, SHOT, SET PIECE, CHALLENGE won,
             BALL OUT) that started at or before the frame. 0 = home, 1 = away, -1 = before the first event.
             "Last team to touch the ball": a PASS / BALL LOST keeps the actor's team until the other team's next
             event. A CHALLENGE only counts for the team that won it (subtype contains WON); a lost challenge is
             skipped. CARD and FAULT RECEIVED do not move possession.
  challenge  frames within CHALLENGE_PAD of a CHALLENGE event: the truth itself is uncertain there and is
             reported separately.
  near_change frames within CHANGE_TOL of a change of the truth team: event frames are hand-tagged and can be
             off by a few frames, so frame accuracy is also given without them.
"""

import json
import os
import sys

import numpy as np
import pandas as pd

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(HERE)
sys.path[:0] = [HERE, os.path.join(ROOT, "src"), os.path.join(ROOT, "starter"), os.path.join(ROOT, "evaluation"), ROOT]
import metrica_load as ml  # noqa: E402

GAMES = (1, 2, 3)
CHALLENGE_PAD = 10      # frames (0.4 s)
CHANGE_TOL = 10         # frames (0.4 s)
KEEP_TYPES = {"PASS", "RECOVERY", "BALL LOST", "SHOT", "SET PIECE", "BALL OUT", "CHALLENGE"}


def _events_12(game):
    ev = pd.read_csv(os.path.join(ml.RAW, "Sample_Game_{}_RawEventsData.csv".format(game)))
    out = pd.DataFrame({"type": ev["Type"], "sub": ev["Subtype"].fillna(""), "start": ev["Start Frame"].astype(int),
                        "team": np.where(ev["Team"] == "Home", 0, 1)})
    return out


def _events_3(home_id):
    data = json.load(open(os.path.join(ml.RAW, "Sample_Game_3_events.json"), encoding="utf-8"))["data"]
    rows = []
    for e in data:
        sub = e.get("subtypes")
        names = [sub.get("name") or ""] if isinstance(sub, dict) else [s.get("name") or "" for s in (sub or [])]
        rows.append({"type": e["type"]["name"], "sub": "-".join(names), "start": int(e["start"]["frame"]),
                     "team": 0 if e["team"]["id"] == home_id else 1})
    return pd.DataFrame(rows)


def load_game(game):
    """dict(match, period, dead, team, challenge, near_change) on the tracking frame grid."""
    if game in (1, 2):
        tracking, _ = ml.load_game(game)
        match = ml.match_from_metrica(tracking)
        dead = ml.dead_label(tracking, ml.dead_spells(tracking, ml.load_game(game)[1])).astype(bool)
        frames = tracking["frame"].to_numpy()
        events = _events_12(game)
    else:
        from kloppy import metrica
        import eval_metrica3 as e3
        match, label, _ = e3.load_game3()
        dead = label.astype(bool)
        frames = np.asarray(match.frame)
        ds = metrica.load_tracking_epts(meta_data=e3.RAW3 + "metadata.xml", raw_data=e3.RAW3 + "tracking.txt",
                                        limit=1)
        events = _events_3(ds.metadata.teams[0].team_id)
    row_of = {int(f): i for i, f in enumerate(frames)}
    n = len(frames)
    team = np.full(n, -1, dtype=np.int8)
    challenge = np.zeros(n, dtype=bool)
    events = events[events["type"].isin(KEEP_TYPES)].sort_values("start", kind="stable")
    marks = []
    for e in events.itertuples():
        r = row_of.get(int(e.start))
        if r is None:
            continue
        if e.type == "CHALLENGE":
            challenge[max(r - CHALLENGE_PAD, 0):r + CHALLENGE_PAD + 1] = True
            if "WON" not in str(e.sub).upper():
                continue
        marks.append((r, int(e.team)))
    for (r, t), nxt in zip(marks, [m[0] for m in marks[1:]] + [n]):
        team[r:nxt] = t
    change = np.flatnonzero(np.diff(team.astype(int)) != 0) + 1
    near = np.zeros(n, dtype=bool)
    for c in change:
        near[max(c - CHANGE_TOL, 0):c + CHANGE_TOL] = True
    period = np.asarray(match.period).astype(np.int64)
    return dict(game=game, match=match, period=period, dead=dead, team=team, challenge=challenge, near_change=near)


def main():
    rows = []
    for g in GAMES:
        d = load_game(g)
        alive = (d["period"] > 0) & ~d["dead"] & (d["team"] >= 0)
        runs = np.flatnonzero(np.diff(np.r_[-9, np.where(alive, d["team"], -9)]) != 0)
        rows.append({"game": g, "frames": len(d["team"]), "in_play_frames": int(alive.sum()),
                     "home_share": float((d["team"][alive] == 0).mean()), "challenge_share": float(d["challenge"][alive].mean()),
                     "team_changes_in_play": int(len(runs))})
    print(pd.DataFrame(rows).round(3).to_string(index=False))


if __name__ == "__main__":
    main()
