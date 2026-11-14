"""[보류한 실험] 결과(DFL, 공 누락 흉내): 인플레이 구간 F1 0.872 → 0.874(패스 1번) / 0.876(패스 2번),
되돌린 구간 중 실제 인플레이 64% / 69%. 패스 횟수는 드리블·공 소유를 못 보는 잘못된 기준이었고, 손해(진짜 데드볼을
되돌림)를 검증할 데이터가 없어 보류(2026-11-11). 데드볼 동안 공이 트래킹된 정답 데이터(PFF 월드컵 2022 후보)가 생기면 다시 본다.

짧은 데드볼 구간을 공으로 다시 보는 단계(final_dead_balls.ball_play_check)를 DFL로 검증한다.

    python src/eval_ball_play.py

1. 기준 길이: 지금 최종 단계(비터비 + 라인 아웃 처리)가 DFL에서 만든 가짜 데드볼 구간(정답 인플레이)의 길이 p90.
   DFL 점수로 고르면 손해 쪽을 잴 수 없어 항상 '제한 없음'이 뽑히므로, 점수가 아니라 가짜 데드볼의 길이로 정한다.
2. 효과: 그 길이 미만 구간에 규칙을 적용했을 때 인플레이 구간 F1과, 되돌린 구간이 실제로 인플레이였는지.
   DFL은 데드볼 동안 공을 고정해 진짜 데드볼 안에서는 패스가 생기지 않으므로, 진짜 데드볼을 잘못 되돌리는 손해는
   과소평가된다(구간 일부만 인플레이였던 경우만 손해로 잡힌다).
공 누락은 대회 데이터처럼 흉내 낸다(simulate.ball_dropout, 범위의 가운데 값). 대회 데이터는 쓰지 않는다.
"""

import os
import sys

import numpy as np
import pandas as pd

HERE = os.path.dirname(os.path.abspath(__file__))
HERE = os.path.dirname(HERE)  # src/ — 이 파일은 src/experiments_dropped/ 안에 있다
sys.path[:0] = [HERE, os.path.join(os.path.dirname(HERE), "starter")]
from config import CONFIG  # noqa: E402
import compare_smoothing as cs  # noqa: E402
import dfl_load  # noqa: E402
import features as fx  # noqa: E402
import final_dead_balls as fd  # noqa: E402
import simulate  # noqa: E402
import smoothing  # noqa: E402
import train_dfl  # noqa: E402

FPS = CONFIG["fps"]
F = CONFIG["final"]
REACH_M = 1.5  # 발이 닿는 거리
from features import runs_of  # noqa: E402
from final_dead_balls import first_in, overlaps  # noqa: E402


def ball_touches(match, sig):
    """선수가 공을 건드린 프레임: 공이 보이고, 공 가속도(앞뒤 속도 벡터 차이)가 앞뒤 speed_half_window 프레임 중
    가장 크며, 공에서 reach_m 안에 선수가 있다. 반환: (터치 프레임 배열, 프레임별 공 위치(이동평균))."""
    ball = match.ball.astype(np.float64)
    n = len(ball)
    seen = sig["seen"]
    smooth = F["ball_smooth_frames"]
    half_s = smooth // 2
    values = np.where(seen[:, None], ball, 0.0)
    total = np.vstack((np.zeros(2), np.cumsum(values, axis=0)))
    count = np.concatenate(([0], np.cumsum(seen)))
    index = np.arange(n)
    low, high = np.maximum(index - half_s, 0), np.minimum(index + half_s + 1, n)
    full = (count[high] - count[low]) == smooth
    pos = np.full((n, 2), np.nan)
    pos[full] = (total[high] - total[low])[full] / smooth
    h = CONFIG["speed_half_window"]
    vel = np.full((n, 2), np.nan)
    vel[h:-h] = (pos[2 * h:] - pos[:-2 * h]) * FPS / (2.0 * h)
    acc = np.full(n, np.nan)
    acc[h:-h] = np.hypot(*(vel[2 * h:] - vel[:-2 * h]).T)
    filled = np.where(np.isfinite(acc), acc, -np.inf)
    padded = np.concatenate([np.full(h, -np.inf), filled, np.full(h, -np.inf)])
    local_max = np.lib.stride_tricks.sliding_window_view(padded, 2 * h + 1).max(axis=1)
    with np.errstate(invalid="ignore"):
        offset = match.players.astype(np.float64) - ball[:, None]
        near = np.fmin.reduce(np.hypot(offset[..., 0], offset[..., 1]), axis=1)
        touch = seen & np.isfinite(acc) & (acc > 0) & (filled >= local_max) & (near <= REACH_M)
    return np.flatnonzero(touch), pos


def ball_play_passes(touches, pos, sig, start, stop):
    """[start, stop) 안의 패스 수. 패스 = 이어진 두 터치 사이에 공이 빠르게(ball_moving_mps 초과) 움직였고,
    공이 두 터치 사이에 2 × reach_m보다 멀리 이동했다(한 선수가 몰고 가는 터치는 세지 않는다)."""
    t = touches[(touches >= start) & (touches < stop)]
    moving = np.concatenate(([0], np.cumsum(sig["moving"])))
    passes = 0
    for a, b in zip(t[:-1], t[1:]):
        if moving[b] - moving[a] <= 0:
            continue
        if np.isfinite(pos[a]).all() and np.isfinite(pos[b]).all() and \
                np.hypot(*(pos[b] - pos[a])) > 2 * REACH_M:
            passes += 1
    return passes


def ball_play_check(state, period, touches, pos, sig, crossings, tag_spans, protected, max_frames, min_passes,
                    window):
    """짧은 데드볼 구간을 공으로 다시 본다: 공이 선수 사이를 오가는 패스가 min_passes번 이상 보이면 인플레이로 되돌린다.

    대상: 길이 max_frames 미만이고, 라인 아웃(시작 앞뒤 window 안)·SET PIECES 태그·protected 구간과 겹치지 않는 데드 구간.
    긴 데드볼은 공이 대부분 안 보여 모델과 공 공백이 같은 쪽을 가리키므로 건드리지 않는다.
    반환: (새 state, 되돌린 구간 [(start, stop, passes)])."""
    out = state.copy()
    reverted = []
    for p in np.unique(period):
        rows = np.flatnonzero(period == p)
        for a, b in runs_of(state[rows]):
            s, e = rows[a], rows[b - 1] + 1
            if e - s >= max_frames or overlaps(s, e, tag_spans) or overlaps(s, e, protected):
                continue
            if first_in(crossings, s - window, min(s + window, e)) is not None:
                continue
            k = ball_play_passes(touches, pos, sig, s, e)
            if k >= min_passes:
                out[s:e] = False
                reverted.append((s, e, k))
    return out, reverted


def pipeline_state(m, i):
    tracking, _, meta = dfl_load.load_match(m)
    match = simulate.simulate(fx.match_from_dfl(tracking, meta), seed=CONFIG["sim_seed"])
    share = float(np.mean(CONFIG["sim_ball_missing_share"]))
    gap = float(np.mean(CONFIG["sim_ball_gap_mean_s"]))
    match.ball = simulate.ball_dropout(match.ball, np.random.default_rng(CONFIG["sim_seed"] + 100 + i), share, gap)
    sig = fd.ball_signals(match)
    crossings = fd.line_crossings(sig)
    oof = pd.read_parquet(os.path.join(train_dfl.oof_dir(), "{}_sim.parquet".format(m)))
    score, period, label = oof["score"].to_numpy(np.float64), oof["period"].to_numpy(), oof["dead"].to_numpy()
    free = np.zeros(len(score), dtype=bool)
    free[crossings] = True
    choice = CONFIG["smooth_choice"]
    state = smoothing.viterbi(score, period, choice["penalty"], choice["bias"], free_entry=free).astype(bool)
    state, forced = fd.force_line_outs(state, crossings, sig, score, period)
    return match, sig, crossings, state, forced, period, label


def main():
    pd.set_option("display.width", 200)
    train_dfl.set_feature_set(CONFIG["default_feature_set"])
    cache, false_len = {}, []
    for i, m in enumerate(dfl_load.MATCHES):
        match, sig, crossings, state, forced, period, label = pipeline_state(m, i)
        cache[m] = (match, sig, crossings, state, forced, period, label)
        for p in np.unique(period):
            rows = np.flatnonzero(period == p)
            for a, b in fx.runs_of(state[rows]):
                s, e = rows[a], rows[b - 1] + 1
                if label[s:e].mean() < 0.5:
                    false_len.append((e - s) / FPS)
    false_len = np.array(false_len)
    cut = float(np.percentile(false_len, 90))
    print("false dead segments after the current final step: {} in 7 matches; length s p50 {:.1f} p90 {:.1f} max {:.1f}"
          .format(len(false_len), np.median(false_len), cut, false_len.max()), flush=True)

    rows = []
    for m, (match, sig, crossings, state, forced, period, label) in cache.items():
        touches, pos = ball_touches(match, sig)
        base = cs.score_one(state.astype(np.int8), pd.DataFrame({"period": period, "dead": label}))
        rows.append(dict(match=m, variant="current", alive_spell_f1=base["alive_spell_f1"], dead_f1=base["dead_f1"],
                         reverted=0, reverted_truly_alive=np.nan))
        for min_passes in (1, 2):
            new, reverted = ball_play_check(state, period, touches, pos, sig, crossings, [], forced,
                                               int(round(cut * FPS)), min_passes,
                                               int(CONFIG["final"]["out_start_window_s"] * FPS))
            r = cs.score_one(new.astype(np.int8), pd.DataFrame({"period": period, "dead": label}))
            alive_share = [1.0 - label[s:e].mean() for s, e, _ in reverted]
            rows.append(dict(match=m, variant="check_{}_pass".format(min_passes), alive_spell_f1=r["alive_spell_f1"],
                             dead_f1=r["dead_f1"], reverted=len(reverted),
                             reverted_truly_alive=np.mean(alive_share) if alive_share else np.nan))
        print("done", m, flush=True)
    t = pd.DataFrame(rows)
    print("\n== alive-spell F1 per match ==")
    print(t.pivot(index="match", columns="variant", values="alive_spell_f1").round(3).to_string())
    print("\n== mean ==")
    print(t.groupby("variant")[["alive_spell_f1", "dead_f1", "reverted", "reverted_truly_alive"]].mean()
          .round(3).to_string())


if __name__ == '__main__':
    main()
