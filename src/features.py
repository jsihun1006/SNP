"""DFL과 대회 데이터(Metrica ATD)에 똑같이 쓰는 프레임별 특징.

공통 포맷(Match):
    players  (n, k, 2) 미터, 없으면 NaN. k는 선수 슬롯 수(DFL은 선수, 대회는 트랙 조각).
    team     (k,) 0 홈 / 1 원정
    ball     (n, 2) 미터, 없으면 NaN
    period   (n,) 하프 번호. 0이면 특징을 계산하되 시간 문맥을 다른 하프와 잇지 않는다.
    pitch    (길이, 너비) 미터

선수 개별 좌표·ID는 특징에 넣지 않는다. 두 데이터셋에서 똑같이 계산되는 팀 단위 요약만 쓴다.
선수 신원이 없는 대회 데이터와, 신원이 고정된 DFL 사이의 차이를 줄이기 위해서다.
"""

import os
import sys
import warnings
from dataclasses import dataclass

import numpy as np
import pandas as pd

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
from config import CONFIG  # noqa: E402

FPS = CONFIG["fps"]


@dataclass
class Match:
    players: np.ndarray
    team: np.ndarray
    ball: np.ndarray
    period: np.ndarray
    pitch: tuple
    frame: np.ndarray = None  # 원래 프레임 번호(보고용)


def runs_of(mask):
    """[(start, stop)] half-open runs where a boolean array is True."""
    edges = np.diff(np.concatenate(([0], np.asarray(mask, dtype=np.int8), [0])))
    return list(zip(np.flatnonzero(edges == 1).tolist(), np.flatnonzero(edges == -1).tolist()))


def interpolate_short_gaps(ball, period, max_frames):
    """같은 하프 안에서, 양쪽에 관측이 있는 max_frames 미만의 결측만 선형으로 메운다."""
    out = ball.copy()
    seen = np.isfinite(ball).all(axis=1)
    for start, stop in runs_of(~seen):
        if stop - start >= max_frames or start == 0 or stop >= len(ball):
            continue
        if period[start - 1] != period[stop]:
            continue
        weights = (np.arange(1, stop - start + 1) / float(stop - start + 1))[:, None]
        out[start:stop] = ball[start - 1] + weights * (ball[stop] - ball[start - 1])
    return out


def central_speed(positions, period, half, cap):
    """앞뒤 half 프레임 차분 속도(m/s). 위치가 없거나 하프가 바뀌거나 cap을 넘으면 NaN. positions (n, ..., 2)."""
    n = len(positions)
    speed = np.full(positions.shape[:-1], np.nan)
    if n <= 2 * half:
        return speed
    step = positions[2 * half:] - positions[:-2 * half]
    value = np.hypot(step[..., 0], step[..., 1]) * FPS / (2.0 * half)
    same = (period[2 * half:] == period[:-2 * half])
    value = np.where(same.reshape((-1,) + (1,) * (value.ndim - 1)), value, np.nan)
    with np.errstate(invalid="ignore"):
        value[value > cap] = np.nan
    speed[half:-half] = value
    return speed


def seconds_since_seen(seen, period):
    """프레임마다 공이 마지막으로 보인 뒤 지난 초. 보이면 0. 하프 시작부터 안 보이면 하프 시작부터 센다."""
    out = np.zeros(len(seen))
    anchor = 0  # 마지막으로 공이 보인 프레임, 또는 하프 시작
    for t in range(len(seen)):
        if t == 0 or period[t] != period[t - 1] or seen[t]:
            anchor = t
        out[t] = 0.0 if seen[t] else (t - anchor) / float(FPS)
    return out


def _nan_stat(function, values, axis):
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", category=RuntimeWarning)
        return function(values, axis=axis)


def nan_percentile(values, q):
    """(n, k)의 행마다 NaN을 뺀 q 백분위(np.nanpercentile 선형 보간과 같은 값). 정렬 한 번으로 계산해 빠르다."""
    ordered = np.sort(values, axis=1)  # NaN은 뒤로 간다
    count = np.isfinite(values).sum(axis=1)
    position = (count - 1).clip(min=0) * (q / 100.0)
    low = np.floor(position).astype(int)
    high = np.minimum(low + 1, (count - 1).clip(min=0))
    rows = np.arange(len(values))
    weight = position - low
    out = ordered[rows, low] * (1 - weight) + ordered[rows, high] * weight
    out[count == 0] = np.nan
    return out


def nearest_neighbour_stats(players, chunk=20000):
    """프레임마다 보이는 선수들의 최근접 이웃 거리의 평균과 최솟값(팀 무관). players (n, k, 2)."""
    n = len(players)
    mean, low = np.full(n, np.nan), np.full(n, np.nan)
    for a in range(0, n, chunk):
        p = players[a:a + chunk]
        d = np.hypot(p[:, :, None, 0] - p[:, None, :, 0], p[:, :, None, 1] - p[:, None, :, 1])
        k = d.shape[1]
        d[:, np.arange(k), np.arange(k)] = np.nan
        nearest = _nan_stat(np.nanmin, d, 2)
        mean[a:a + chunk] = _nan_stat(np.nanmean, nearest, 1)
        low[a:a + chunk] = _nan_stat(np.nanmin, nearest, 1)
    return mean, low


def base_features(match):
    """프레임별 기준 특징 DataFrame(시간 문맥 전)."""
    length, width = match.pitch
    players = match.players.astype(np.float64)
    period = np.asarray(match.period)
    half = CONFIG["speed_half_window"]

    ball = interpolate_short_gaps(match.ball.astype(np.float64), period, int(CONFIG["ball_interp_max_s"] * FPS))
    seen = np.isfinite(ball).all(axis=1)
    player_speed = central_speed(players, period, half, CONFIG["max_player_speed"])
    ball_speed = central_speed(ball, period, half, CONFIG["max_ball_speed"])

    out = {}
    for code, name in ((0, "home"), (1, "away")):
        mask = match.team == code
        speeds = player_speed[:, mask]
        out[name + "_speed_mean"] = _nan_stat(np.nanmean, speeds, 1)
        out[name + "_speed_max"] = _nan_stat(np.nanmax, speeds, 1)
        out[name + "_speed_std"] = _nan_stat(np.nanstd, speeds, 1)
        out[name + "_count"] = np.isfinite(players[:, mask, 0]).sum(axis=1).astype(np.float64)
        xs, ys = players[:, mask, 0], players[:, mask, 1]
        # 대형 크기와 중심은 이상치에 강하게 잡는다: 10~90% 범위와 중앙값. 대회 데이터는 팀 분류가 틀린 트랙이나
        # 엉뚱한 위치의 트랙 하나가 최댓값-최솟값을 크게 늘린다(팀 길이 평균 55 m 대 DFL 49 m).
        low, high = CONFIG["formation_quantiles"]
        out[name + "_length"] = nan_percentile(xs, high) - nan_percentile(xs, low)
        out[name + "_width"] = nan_percentile(ys, high) - nan_percentile(ys, low)
        out[name + "_cx"] = _nan_stat(np.nanmedian, xs, 1)
        out[name + "_cy"] = _nan_stat(np.nanmedian, ys, 1)
    out["all_speed_mean"] = _nan_stat(np.nanmean, player_speed, 1)
    # v2 특징: 기준값 없이 분포만 요약한다(어디서 나눌지는 모델이 배운다).
    for q in CONFIG["speed_quantiles"]:
        out["all_speed_p{}".format(q)] = nan_percentile(player_speed, q)
    out["team_speed_gap"] = np.abs(out["home_speed_mean"] - out["away_speed_mean"])
    xs_all, ys_all = players[..., 0], players[..., 1]
    with np.errstate(invalid="ignore"):
        out["touchline_min"] = _nan_stat(np.nanmin, np.minimum(ys_all, width - ys_all), 1)
        out["goalline_min"] = _nan_stat(np.nanmin, np.minimum(xs_all, length - xs_all), 1)
    out["nn_mean"], out["nn_min"] = nearest_neighbour_stats(players)
    out["centroid_distance"] = np.hypot(out["home_cx"] - out["away_cx"], out["home_cy"] - out["away_cy"])
    # 중심 위치는 경기장 방향과 하프마다 바뀌는 공격 방향에 묶여 있어 그대로 쓰지 않는다(거리만 남긴다).
    for name in ("home_cx", "home_cy", "away_cx", "away_cy"):
        out.pop(name)

    out["ball_missing"] = (~seen).astype(np.float64)
    out["ball_missing_s"] = seconds_since_seen(seen, period)
    out["ball_speed"] = ball_speed
    x, y = ball[:, 0], ball[:, 1]
    with np.errstate(invalid="ignore"):
        inside_x = np.minimum(x, length - x)
        inside_y = np.minimum(y, width - y)
        out["ball_boundary_distance"] = np.minimum(inside_x, inside_y)  # 밖이면 음수
        out["ball_outside"] = np.where(seen, (out["ball_boundary_distance"] < 0).astype(np.float64), np.nan)

    offset = players - ball[:, None]
    distance = np.hypot(offset[..., 0], offset[..., 1])
    filled = np.where(np.isfinite(distance), distance, np.inf)
    nearest_slot = np.argmin(filled, axis=1)
    nearest = filled[np.arange(len(filled)), nearest_slot]
    nearest[~np.isfinite(nearest)] = np.nan
    out["ball_nearest_player"] = nearest
    nearest_team = match.team[nearest_slot]
    home_min = _nan_stat(np.nanmin, np.where(match.team[None, :] == 0, distance, np.nan), 1)
    away_min = _nan_stat(np.nanmin, np.where(match.team[None, :] == 1, distance, np.nan), 1)
    out["ball_nearest_opponent"] = np.where(nearest_team == 0, away_min, home_min)
    out["ball_nearest_opponent"][~np.isfinite(nearest)] = np.nan

    frame = pd.DataFrame(out)
    frame.insert(0, "period", period)
    if match.frame is not None:
        frame.insert(0, "frame", match.frame)
    return frame


def add_context(frame, columns, offsets_s=None):
    """기준 특징의 과거·미래 시점 값을 붙인다. 하프(period)를 넘지 않는다. 이름은 '<특징>@<초>s'."""
    offsets_s = CONFIG["context_offsets_s"] if offsets_s is None else offsets_s
    grouped = frame.groupby("period", sort=False)[columns]
    pieces = [frame]
    for seconds in offsets_s:
        shift = -int(round(seconds * FPS))  # 미래(+)를 보려면 뒤의 값을 당겨 온다
        shifted = grouped.shift(shift)
        shifted.columns = ["{}@{:+g}s".format(c, seconds) for c in columns]
        pieces.append(shifted)
    return pd.concat(pieces, axis=1)


def compute_features(match, offsets_s=None):
    """기준 특징 + 시간 문맥. 열 이름 목록은 feature_columns()로 얻는다."""
    base = base_features(match)
    columns = [c for c in base.columns if c not in ("frame", "period")]
    return add_context(base, columns, offsets_s)


def feature_columns(frame):
    return [c for c in frame.columns if c not in ("frame", "period")]


# --- DFL(IDSSE) → 공통 포맷 ---

def match_from_dfl(tracking, meta):
    """dfl_load.load_match의 결과를 공통 포맷으로. 좌표는 0~1 정규화 값에 피치 크기를 곱한다."""
    length = float(meta["pitch_length"].iloc[0])
    width = float(meta["pitch_width"].iloc[0])
    team_of = dict(zip(meta["player_id"], meta["team"]))
    ids = [c[:-2] for c in tracking.columns if c.endswith("_x") and not c.startswith("ball")]
    ids = [i for i in ids if i in team_of]
    players = np.stack([np.stack([tracking[i + "_x"].to_numpy(np.float64) * length,
                                  tracking[i + "_y"].to_numpy(np.float64) * width], axis=1) for i in ids], axis=1)
    team = np.array([0 if team_of[i] == "home" else 1 for i in ids], dtype=np.int8)
    ball = np.stack([tracking["ball_x"].to_numpy(np.float64) * length,
                     tracking["ball_y"].to_numpy(np.float64) * width], axis=1)
    return Match(players=players.astype(np.float32), team=team, ball=ball,
                 period=tracking["period"].to_numpy(), pitch=(length, width),
                 frame=tracking["frame_id"].to_numpy())


if __name__ == '__main__':
    import time
    import dfl_load
    match_id = sys.argv[1] if len(sys.argv) > 1 else "J03WPY"
    tracking, _, meta = dfl_load.load_match(match_id)
    started = time.time()
    features = compute_features(match_from_dfl(tracking, meta))
    print("{}: {} frames x {} feature columns in {:.1f}s".format(
        match_id, len(features), len(feature_columns(features)), time.time() - started))
