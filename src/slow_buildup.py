"""느린 빌드업 보정(JSH ⑤): 패스·수신이나 공과 선수의 동행이 보이면 데드볼로 들어가기 어렵게 한다.

출처: JSH 브랜치 model/영상추출/모델/dead_ball/LightGBM_deadball_corrections.ipynb (커밋 b6b307d)의
movement_features·viterbi_cost·decode 중 ⑤에 필요한 부분을 그대로 옮겼다. 상수는 노트북 값이며 노트북에 따르면
DFL 학습 쪽 내부 검증 경기로 골랐다(대회 영상이나 공개 정답 5분에 맞춘 값이 아니다). 대회 설정은 노트북의
COMPETITION_RULES(margin 0.0, strength 1.0, bonus 0.85)와 같다(config.py의 slow_buildup).

검증(인플레이 구간 F1, 지금 파이프라인 대비):
  Metrica 3경기(학습에 안 씀)   0.809 -> 0.820, 2경기 개선 / 악화 0 (src/eval_jsh_corrections_metrica.py)
  SkillCorner 20경기            0.729 -> 0.748, 18경기 개선 / 2경기 악화 (src/eval_jsh_corrections_skillcorner.py)
    데드 -> 인플레이로 바꾼 프레임 중 실제 데드는 24%(데드볼 동안 진짜 공이 있는 데이터로 잰 손해)
  DFL 7경기(JSH 보고)           0.877 -> 0.894
"""

import numpy as np
from scipy.ndimage import maximum_filter1d, minimum_filter1d, uniform_filter1d
from scipy.special import logit

from config import CONFIG
from features import runs_of
import smoothing

FPS = CONFIG["fps"]


def halves(period):
    cuts = np.r_[0, np.flatnonzero(period[1:] != period[:-1]) + 1, len(period)]
    return [slice(a, b) for a, b in zip(cuts[:-1], cuts[1:])]


def shift(x, frames):
    out = np.full(x.shape, np.nan, dtype=np.float64)
    if not frames:
        return np.asarray(x, dtype=np.float64)
    if 0 < frames < len(x):
        out[frames:] = x[:-frames]
    elif -len(x) < frames < 0:
        out[:frames] = x[-frames:]
    return out


def spread(mask, radius):
    return maximum_filter1d(mask.astype(np.uint8), 2 * radius + 1, mode="constant") != 0


def extend_from(indices, n, before, after):
    counts = np.zeros(n + 1, dtype=np.int32)
    np.add.at(counts, np.maximum(0, indices - before), 1)
    np.add.at(counts, np.minimum(n, indices + after + 1), -1)
    return np.cumsum(counts[:-1]) > 0


def peak_indices(mask, strength, gap):
    peaks = np.array([a + np.argmax(strength[a:b]) for a, b in runs_of(mask)], dtype=int)
    return peaks[np.r_[True, np.diff(peaks) >= gap]] if peaks.size else peaks


def movement_signals(players, ball, pitch, margin):
    """한 하프의 공 신호(노트북 movement_features 중 ⑤에 쓰는 것): reliable, pass, carry, restart, cross."""
    n = len(ball)
    observed = np.isfinite(ball).all(axis=1)
    x, y = ball.T
    length, width = pitch
    distance = np.minimum.reduce((x, length - x, y, width - y))
    delta = ball - shift(ball, 1)
    jumps = np.linalg.norm(delta, axis=1) > 45 / FPS
    continuous = minimum_filter1d(observed.astype(np.uint8), 7, mode="constant").astype(bool)
    reliable = continuous & ~spread(jumps, 4)
    velocity = (shift(ball, -3) - shift(ball, 3)) * FPS / 6
    speed = np.linalg.norm(velocity, axis=1)
    speed[~reliable] = np.nan
    dist = np.linalg.norm(players - ball[:, None, :], axis=2)
    dist = np.where(np.isfinite(dist), dist, np.inf)
    near = dist.min(axis=1)
    owner = np.argmin(dist, axis=1)
    old_owner = np.nan_to_num(shift(owner, 4), nan=-1).astype(int)
    same = owner == old_owner
    rows = np.arange(n)
    dr = players[rows, owner] - shift(players, 4)[rows, owner]
    ball_motion = ball - shift(ball, 4)
    co_motion = np.linalg.norm(dr - ball_motion, axis=1)
    # 공과 같은 선수가 함께 움직임(드리블·공을 들고 감)
    track_carry = reliable & same & (near < 1.1) & (co_motion < .40) & (speed > .35) & (speed < 7)
    carry = uniform_filter1d(track_carry.astype(float), 15, mode="constant") >= .65
    pass_windows = np.zeros(n, dtype=bool)
    launch = np.nan_to_num(shift(speed, -5), nan=0)
    before = np.nan_to_num(shift(speed, 5), nan=0)
    persistent = uniform_filter1d((speed > 3.0).astype(float), 9, mode="constant")
    departure = (launch > 3.8) & (launch - before > 1.3) & (near < 2.0) & reliable & (persistent > .35)
    departures = peak_indices(departure, launch - before, 10)
    # 공이 떠난 뒤 다른 선수가 받음(패스·수신)
    for a in departures:
        b = min(n, a + 51)
        window = np.arange(a + 7, b)
        hits = window[reliable[window] & (near[window] < 1.35) & (owner[window] != owner[a])]
        if hits.size:
            h = int(hits[0])
            flight = ball[a:h + 1]
            span = np.linalg.norm(ball[h] - ball[a])
            if (span >= 4 and np.isfinite(flight).all() and not jumps[a:h + 1].any()
                    and np.min(distance[a:h + 1]) >= 0
                    and (np.nanmedian(speed[max(a, h - 4):min(n, h + 5)]) < launch[a] * .8 or near[h] < .7)):
                pass_windows[a:h + 1] = True
    restart = departure & (np.nan_to_num(shift(persistent, -6), nan=0) > .48)
    restart_at = np.zeros(n, dtype=bool)
    restart_at[peak_indices(restart, launch - before, 13)] = True
    before_dist = shift(distance, 5)
    after_dist = shift(distance, -4)
    crossing = reliable & (shift(distance, 1) >= -margin) & (distance < -margin)
    crossing &= (before_dist - distance > .15) & (distance - after_dist > -.06)
    cross = peak_indices(crossing, np.nan_to_num(before_dist - distance, nan=0), 10)
    return {"reliable": reliable, "distance": distance, "restart": restart_at, "pass": pass_windows,
            "carry": carry, "cross": cross}


def viterbi_cost(log_odds, enter, leave):
    """smoothing.viterbi와 같은 2상태 재귀에 프레임별 진입(enter)·이탈(leave) 비용을 둔 것."""
    alive, dead = 0., float(-log_odds[0])
    back_a = np.zeros(len(log_odds), dtype=bool)
    back_d = np.zeros(len(log_odds), dtype=bool)
    for i in range(1, len(log_odds)):
        switch_a = dead + leave[i]
        switch_d = alive + enter[i]
        back_a[i] = switch_a < alive
        back_d[i] = switch_d < dead
        alive, dead = min(alive, switch_a), min(dead, switch_d) - log_odds[i]
    state = np.empty(len(log_odds), dtype=bool)
    current = dead < alive
    for i in range(len(state) - 1, -1, -1):
        state[i] = current
        current = not current if (back_d[i] if current else back_a[i]) else current
    return state


def decode(score, players, ball, pitch, period, crossings, setting=None):
    """라인 아웃 순간 진입 비용 0(지금 파이프라인 1단계) + ⑤ 보호 구간. force_line_outs 전 상태를 돌려준다."""
    setting = setting or CONFIG["slow_buildup"]
    penalty, bias = CONFIG["smooth_choice"]["penalty"], CONFIG["smooth_choice"]["bias"]
    strength, bonus, margin = setting["strength"], setting["bonus"], setting["margin"]
    period = np.asarray(period)
    result = np.zeros(len(score), dtype=bool)
    for half in halves(period):
        p = score[half]
        n = len(p)
        sig = movement_signals(players[half], ball[half], pitch, margin)
        old = crossings[(crossings >= half.start) & (crossings < half.stop)] - half.start
        if not sig["reliable"].any():
            result[half] = smoothing.viterbi(p, np.zeros(n), penalty, bias).astype(bool)
            continue
        log_odds = logit(np.clip(p, 1e-6, 1 - 1e-6))
        enter = np.full(n, penalty)
        leave = np.full(n, penalty)
        enter[old] = 0
        restart_ids = np.flatnonzero(sig["restart"])
        certified = []
        for a in sig["cross"]:
            rs = restart_ids[(restart_ids > a + round(.32 * FPS)) & (restart_ids < a + 25 * FPS)]
            evidence = p[a:min(n, a + 2 * FPS)]
            outside = sig["distance"][a:min(n, a + 8)] < -margin
            if len(rs) and (outside.sum() >= 4) and (evidence.max() >= .40 or np.any(old == a)):
                certified.append(int(a))
        confirmed = np.unique(np.r_[old, np.array(certified, dtype=int)])
        protected = (sig["pass"] | sig["carry"]) & ~extend_from(confirmed, n, 2, 4)
        enter[protected] += 28 * strength
        log_odds[protected] -= 2.5 * bonus
        enter[old] = 0
        result[half] = viterbi_cost(log_odds, enter, leave)
    return result
