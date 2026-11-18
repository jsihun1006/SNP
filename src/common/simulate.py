"""DFL 경기를 대회 데이터(Metrica ATD) 조건처럼 망가뜨린다(트래킹 노이즈 시뮬레이션).

모델이 선수 특징만 쓰므로 두 가지를 흉내 낸다.
  1. 선수 누락: DFL은 매 프레임 22명, 대회 데이터는 평균 20.5명이다. 선수마다 보임/안 보임을 번갈아 만드는
     구간 과정으로 지운다(무작위 프레임이 아니라 구간 단위).
  2. 좌표 흔들림: 작은 가우시안 잡음을 상관 시간만큼 부드럽게 이어서 더한다. 프레임마다 독립적인 잡음은
     0.16초 차분 속도에서 크게 부풀어 실제 트래킹과 다르다.
공은 건드리지 않는다(모델이 공 특징을 쓰지 않는다).
"""

import os
import sys

import numpy as np

HERE = os.path.dirname(os.path.realpath(__file__))
sys.path.insert(0, HERE)
from config import CONFIG  # noqa: E402
from features import Match  # noqa: E402

FPS = CONFIG["fps"]


def gap_length(rng, mean_frames, sigma):
    """빈 구간 길이(프레임). sigma > 0이면 평균 mean_frames의 로그정규(꼬리가 긴 분포), 0이면 지수 분포."""
    if sigma > 0:
        mu = np.log(mean_frames) - sigma ** 2 / 2.0
        return max(1, int(round(rng.lognormal(mu, sigma))))
    return max(1, int(round(rng.exponential(mean_frames))))


def dropout_mask(n, slots, present, rng, keep_share, mean_off_frames, sigma=0.0):
    """(n, slots) True면 지운다. present(이미 있는 프레임) 안에서 평균 (1 - keep_share)만큼, 평균 길이
    mean_off_frames의 구간으로 지운다. 보임/안 보임 구간 길이는 지수 분포를 따른다."""
    mask = np.zeros((n, slots), dtype=bool)
    off_share = 1.0 - keep_share
    if off_share <= 0:
        return mask
    mean_on = mean_off_frames * keep_share / off_share
    for slot in range(slots):
        t = int(rng.exponential(mean_on))
        while t < n:
            length = gap_length(rng, mean_off_frames, sigma)
            mask[t:t + length, slot] = True
            t += length + max(1, int(round(rng.exponential(mean_on))))
    return mask & present


def _box(values, width):
    """시간축(0번 축) 이동 평균. 길이가 width - 1만큼 줄어든다."""
    total = np.cumsum(np.concatenate((np.zeros((1,) + values.shape[1:]), values)), axis=0)
    return (total[width:] - total[:-width]) / float(width)


def group_dropout_mask(n, slots, present, rng, off_share, mean_frames, size_range, sigma=0.0):
    """(n, slots) 여러 선수가 함께 빠지는 구간. 전체 (선수×프레임) 중 평균 off_share만큼 지운다."""
    mask = np.zeros((n, slots), dtype=bool)
    if off_share <= 0:
        return mask
    low, high = size_range
    mean_size = (low + high) / 2.0
    per_slot_present = present.sum(axis=0).mean() / float(n)  # 프레임당 실제로 있는 선수 비율
    events = int(round(off_share * n * slots * per_slot_present / (mean_size * mean_frames)))
    for _ in range(events):
        start = int(rng.integers(0, n))
        length = gap_length(rng, mean_frames, sigma)
        alive = np.flatnonzero(present[min(start, n - 1)])
        if not len(alive):
            continue
        size = min(int(rng.integers(low, high + 1)), len(alive))
        chosen = rng.choice(alive, size=size, replace=False)
        mask[start:start + length, chosen] = True
    return mask & present


def smooth_noise(n, slots, sigma, smooth_frames, rng):
    """(n, slots, 2) 상관 시간 약 smooth_frames인 가우시안 잡음. 표준편차는 sigma로 맞춘다.

    상자 평균을 세 번 겹쳐(가우시안에 가까운 평활) 프레임 사이 변화가 작게 한다. 한 번만 하면 잡음의
    프레임 간 차이가 커서 서 있는 선수도 1 m/s가 넘는 가짜 속도를 갖게 된다.
    """
    width = max(int(smooth_frames), 1)
    if width == 1:
        noise = rng.normal(size=(n, slots, 2))
    else:
        noise = rng.normal(size=(n + 3 * (width - 1), slots, 2))
        for _ in range(3):
            noise = _box(noise, width)
        noise = noise[:n]
    std = noise.std()
    return noise * (sigma / std if std > 0 else 0.0)


def simulate(match, seed=None, target_players=None, dropout_mean_s=None, jitter_m=None, smooth_s=None,
             group_share=None, group_mean_s=None, gap_sigma=None):
    """망가뜨린 Match 사본. 같은 seed면 같은 결과다."""
    rng = np.random.default_rng(CONFIG["sim_seed"] if seed is None else seed)
    target = CONFIG["sim_target_players"] if target_players is None else target_players
    mean_off = (CONFIG["sim_dropout_mean_s"] if dropout_mean_s is None else dropout_mean_s) * FPS
    sigma = CONFIG["sim_jitter_m"] if jitter_m is None else jitter_m
    smooth = int(round((CONFIG["sim_jitter_smooth_s"] if smooth_s is None else smooth_s) * FPS))

    players = match.players.astype(np.float64).copy()
    n, slots, _ = players.shape
    present = np.isfinite(players[..., 0])
    mean_present = present.sum(axis=1).mean()
    off_share = max(0.0, 1.0 - target / mean_present) if mean_present else 0.0
    group_share = CONFIG["sim_group_share"] if group_share is None else group_share
    players += smooth_noise(n, slots, sigma, smooth, rng)
    group_mean = (CONFIG["sim_group_mean_s"] if group_mean_s is None else group_mean_s) * FPS
    sigma_gap = CONFIG["sim_gap_sigma"] if gap_sigma is None else gap_sigma
    single = dropout_mask(n, slots, present, rng, 1.0 - off_share * (1.0 - group_share), mean_off, sigma_gap)
    group = group_dropout_mask(n, slots, present, rng, off_share * group_share, group_mean,
                               CONFIG["sim_group_size"], sigma_gap)
    removed = single | group
    # 구간이 겹치면 목표보다 덜 지워진다. 모자란 만큼 개별 누락을 한 번 더 넣는다.
    remaining = present & ~removed
    have = remaining.sum(axis=1).mean()
    if have > target and have > 0:
        removed |= dropout_mask(n, slots, remaining, rng, target / have, mean_off, sigma_gap)
    players[removed] = np.nan
    return Match(players=players.astype(np.float32), team=match.team.copy(), ball=match.ball.copy(),
                 period=match.period.copy(), pitch=match.pitch, frame=match.frame)


def ball_dropout(ball, rng, missing_share, gap_mean_s):
    """공 좌표 사본에서 평균 missing_share만큼을 평균 gap_mean_s초 구간들로 지운다(보임/안 보임 길이는 지수 분포)."""
    out = np.asarray(ball, dtype=np.float64).copy()
    if missing_share <= 0:
        return out
    present = np.isfinite(out).all(axis=1)[:, None]
    mask = dropout_mask(len(out), 1, present, rng, 1.0 - missing_share, gap_mean_s * FPS)[:, 0]
    out[mask] = np.nan
    return out
