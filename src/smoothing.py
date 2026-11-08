"""프레임별 데드볼 확률을 구간으로 다듬는다. 모두 하프(period) 안에서만 계산한다.

    median_filter   확률을 width 프레임 이동 중앙값으로 편 뒤 0.5로 자른다
    viterbi         2상태(인플레이/데드) 최소 비용 경로. 프레임 비용 -log p, 상태가 바뀔 때마다 penalty
    min_duration    min_frames보다 짧은 구간을 짧은 것부터 이웃 상태로 메운다

spellF1은 가짜 전환에 가장 민감하다(메모리 mtah-metric-and-labels). 그래서 전환 하나에 비용을 매기는
viterbi가 프레임마다 독립적으로 자르는 방식보다 맞을 것이라는 가설을 검증한다.
"""

import os
import sys

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
from features import runs_of  # noqa: E402

EPS = 1e-6


def _per_period(function, values, period, *args):
    out = np.empty(len(values), dtype=np.int8)
    for p in np.unique(period):
        rows = np.flatnonzero(period == p)
        out[rows] = function(values[rows], *args)
    return out


def _median_one(score, width):
    if width <= 1:
        return (score >= 0.5).astype(np.int8)
    half = width // 2
    padded = np.pad(score, half, mode="edge")
    windows = np.lib.stride_tricks.sliding_window_view(padded, 2 * half + 1)
    return (np.median(windows, axis=1) >= 0.5).astype(np.int8)


def median_filter(score, period, width):
    return _per_period(_median_one, np.asarray(score, dtype=np.float64), period, int(width))


def _viterbi_one(score, penalty, bias, free_entry=None):
    """free_entry (n,) bool: True인 프레임에서는 인플레이 → 데드 전환에 벌점이 없다(공이 라인을 넘은 순간 등)."""
    p = np.clip(score, EPS, 1 - EPS)
    # bias > 0이면 데드 쪽 비용을 늘린다(로그 오즈 이동).
    cost = np.stack([-np.log(1 - p), -np.log(p) + bias], axis=1)
    n = len(p)
    back = np.zeros((n, 2), dtype=bool)  # True면 이전 상태가 반대 상태였다
    total = cost[0].copy()
    for t in range(1, n):
        cost_t = penalty[t] if np.ndim(penalty) else penalty  # 프레임마다 다른 벌점도 받는다
        stay0, switch0 = total[0], total[1] + cost_t
        stay1, switch1 = total[1], total[0] + (0.0 if free_entry is not None and free_entry[t] else cost_t)
        back[t, 0] = switch0 < stay0
        back[t, 1] = switch1 < stay1
        total = np.array([min(stay0, switch0), min(stay1, switch1)]) + cost[t]
    state = np.empty(n, dtype=np.int8)
    state[-1] = int(np.argmin(total))
    for t in range(n - 1, 0, -1):
        state[t - 1] = 1 - state[t] if back[t, state[t]] else state[t]
    return state


def viterbi(score, period, penalty, bias=0.0, free_entry=None):
    """free_entry가 있으면 그 프레임들에서 데드로 들어가는 전환은 공짜다(데드에서 나오는 전환 벌점은 그대로)."""
    score = np.asarray(score, dtype=np.float64)
    if free_entry is None and np.ndim(penalty) == 0:
        return _per_period(_viterbi_one, score, period, float(penalty), float(bias))
    out = np.empty(len(score), dtype=np.int8)
    for p in np.unique(period):
        rows = np.flatnonzero(period == p)
        pen = np.asarray(penalty, dtype=np.float64)[rows] if np.ndim(penalty) else float(penalty)
        free = np.asarray(free_entry)[rows] if free_entry is not None else None
        out[rows] = _viterbi_one(score[rows], pen, float(bias), free)
    return out


def _min_duration_one(state, min_frames):
    state = state.copy()
    if min_frames <= 1:
        return state
    while True:
        runs = [(a, b) for a, b in runs_of(state == 1)] + [(a, b) for a, b in runs_of(state == 0)]
        # 하프의 처음·끝 구간은 잘린 것일 수 있어 건드리지 않는다
        inner = [(b - a, a, b) for a, b in runs if a > 0 and b < len(state) and b - a < min_frames]
        if not inner:
            return state
        _, a, b = min(inner)
        state[a:b] = 1 - state[a]


def min_duration(state, period, min_frames):
    return _per_period(_min_duration_one, np.asarray(state, dtype=np.int8), period, int(min_frames))
