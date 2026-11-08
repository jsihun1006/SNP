"""DFL 경기의 데드볼 구간과 그 원인(재개 종류).

라벨은 DFL BallStatus(ball_state 0 = 데드)다. 원인은 구간 끝 근처(±max_gap_s)의 세트피스 이벤트로 정한다.
DFL 데드볼 끝은 재개 킥 이벤트의 중앙값 0.5초 뒤다(7경기에서 잰 값). 이벤트가 없으면 "other".
"""

import os
import sys

import numpy as np
import pandas as pd

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
from config import CONFIG  # noqa: E402
from features import runs_of  # noqa: E402

CAUSE_NAMES = {"THROW_IN": "throw_in", "FREE_KICK": "free_kick", "GOAL_KICK": "goal_kick",
               "CORNER_KICK": "corner", "KICK_OFF": "kick_off", "PENALTY": "penalty"}


def dead_label(tracking):
    """(n,) 1 데드 / 0 인플레이."""
    return (tracking["ball_state"].to_numpy() == 0).astype(np.int8)


def dead_spells(tracking, events, max_gap_s=None):
    """DataFrame[start, stop(half-open, 행 번호), period, seconds, cause]."""
    max_gap_s = CONFIG["cause_max_gap_s"] if max_gap_s is None else max_gap_s
    dead = dead_label(tracking).astype(bool)
    period = tracking["period"].to_numpy()
    stamp = tracking["timestamp_s"].to_numpy()
    rows = []
    for p in np.unique(period):
        rows_p = np.flatnonzero(period == p)
        pieces = events[(events["period_id"] == p) & events["set_piece_type"].notna()]
        times = pieces["timestamp_s"].to_numpy()
        kinds = pieces["set_piece_type"].to_numpy()
        for a, b in runs_of(dead[rows_p]):
            start, stop = rows_p[a], rows_p[b - 1] + 1
            cause = "other"
            if len(times):
                end_time = stamp[stop - 1]
                k = int(np.argmin(np.abs(times - end_time)))
                if abs(times[k] - end_time) <= max_gap_s:
                    cause = CAUSE_NAMES.get(kinds[k], str(kinds[k]).lower())
            rows.append((start, stop, int(p), (stop - start) / float(CONFIG["fps"]), cause))
    return pd.DataFrame(rows, columns=["start", "stop", "period", "seconds", "cause"])
