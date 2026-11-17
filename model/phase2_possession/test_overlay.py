"""Synthetic tests of the shared dead-ball overlay (no match data).

    python model/phase2_possession/test_overlay.py
"""
from pathlib import Path
import sys

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))
import overlay  # noqa: E402

N = overlay.N_FRAMES


def frames(**spans):
    graded = np.zeros(N, bool)
    dead = np.zeros(N, bool)
    graded[spans.get("graded", (0, 0))[0]:spans.get("graded", (0, 0))[1]] = True
    for a, b in spans.get("dead", []):
        dead[a:b] = True
    return graded, dead


def test_dead_wins_inside_graded():
    graded, dead = frames(graded=(100, 200), dead=[(120, 130)])
    team = np.array(["H"] * N, dtype=object)
    labels = overlay.overlay(team, graded, dead)
    assert (labels[120:130] == "D").all() and (labels[100:120] == "H").all() and (labels[130:200] == "H").all()


def test_missing_team_in_play_is_an_error():
    graded, dead = frames(graded=(100, 200), dead=[(120, 130)])
    team = np.array(["A"] * N, dtype=object)
    team[150] = ""
    try:
        overlay.overlay(team, graded, dead)
    except ValueError as error:
        assert "150" in str(error)
    else:
        raise AssertionError("expected an error for a missing team on an in-play graded frame")


def test_no_call_allowed_on_dead_and_ungraded():
    graded, dead = frames(graded=(100, 200), dead=[(120, 130)])
    team = np.array(["A"] * N, dtype=object)
    team[120:130] = ""
    team[:100] = ""
    labels = overlay.overlay(team, graded, dead)
    assert (labels[:100] == "D").all() and (labels[120:130] == "D").all() and (labels[200:] == "A").all()


def test_bad_value_rejected():
    graded, dead = frames(graded=(100, 200))
    team = np.array(["H"] * N, dtype=object)
    team[5] = "X"
    try:
        overlay.overlay(team, graded, dead)
    except ValueError:
        pass
    else:
        raise AssertionError("expected an error for a team value other than H/A/''")


if __name__ == "__main__":
    for name, test in list(globals().items()):
        if name.startswith("test_"):
            test()
    print("OK")
