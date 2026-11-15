"""진단: 골키퍼가 공을 쥔 인플레이 구간을 지금 파이프라인이 데드로 잘못 보는지, 공이 안 보이면 얼마나 더 그런지 잰다.

    python src/eval_keeper_hold_size.py

DFL 7경기(공이 늘 보이고 데드볼 동안만 고정)로 정답 구간을 만든다:
  골키퍼 = 하프마다 각 팀에서 자기 골라인까지 평균 거리가 가장 짧은 선수(트랙이 한 선수를 따라가는 DFL에서만 쓸 수 있다)
  쥠 = 인플레이(정답)이고, 공이 그 골키퍼에서 HOLD_DIST_M 안, 골키퍼가 자기 페널티 박스 안, HOLD_MIN_S초 이상 이어짐
파이프라인 입력: 모델 OOF 확률 + 원본 선수 좌표, 공은 원래대로(ball)와 전부 지움(no_ball).
출력: 쥠 구간 수, 그중 프레임 절반 이상을 데드로 본 구간 비율, 쥠 프레임 중 데드로 본 비율, 쥠 길이 분포.
"""

import os
import sys

import numpy as np
import pandas as pd

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(HERE)
sys.path[:0] = [HERE, os.path.join(ROOT, "starter")]
from config import CONFIG  # noqa: E402
import features as fx  # noqa: E402
from eval_kickoff_fill import pipeline  # noqa: E402

FPS = CONFIG["fps"]
HOLD_DIST_M, HOLD_MIN_S, BOX_DEPTH, BOX_HALF = 1.5, 1.0, 16.5, 20.16


def keeper_hold_spans(match, label):
    length, width = match.pitch
    p = match.players.astype(np.float64)
    ball = np.asarray(match.ball, dtype=np.float64)
    period = np.asarray(match.period)
    team = np.asarray(match.team)
    hold = np.zeros(len(period), dtype=bool)
    for per in np.unique(period[period > 0]):
        rows = period == per
        for t in (0, 1):
            cols = np.flatnonzero(team == t)
            mean_x = np.nanmean(p[rows][:, cols, 0], axis=0)
            team_x = np.nanmean(mean_x)
            side = 0 if team_x < length / 2 else 1
            depth_mean = mean_x if side == 0 else length - mean_x
            k = cols[int(np.nanargmin(depth_mean))]
            kx, ky = p[:, k, 0], p[:, k, 1]
            depth = kx if side == 0 else length - kx
            with np.errstate(invalid="ignore"):
                near = np.hypot(ball[:, 0] - kx, ball[:, 1] - ky) <= HOLD_DIST_M
                in_box = (depth <= BOX_DEPTH) & (depth >= -1) & (np.abs(ky - width / 2) <= BOX_HALF)
            hold |= rows & near & in_box & ~label
    return [(a, b) for a, b in fx.runs_of(hold) if b - a >= HOLD_MIN_S * FPS]


def main():
    import dfl_load
    import train_dfl
    pd.set_option("display.width", 200)
    rows, lengths = [], []
    for m in dfl_load.MATCHES:
        tracking, events, meta = dfl_load.load_match(m)
        raw = fx.match_from_dfl(tracking, meta)
        oof = pd.read_parquet(os.path.join(train_dfl.oof_dir(), "{}_sim.parquet".format(m)))
        match = fx.Match(players=raw.players, team=raw.team, ball=raw.ball,
                         period=oof["period"].to_numpy().astype(np.int64), pitch=raw.pitch)
        label = oof["dead"].to_numpy().astype(bool)
        score = oof["score"].to_numpy(np.float64)
        spans = keeper_hold_spans(match, label)
        lengths += [(b - a) / FPS for a, b in spans]
        for cond in ("ball", "no_ball"):
            mm = match if cond == "ball" else fx.Match(players=match.players, team=match.team,
                                                       ball=np.full_like(np.asarray(match.ball, dtype=np.float64), np.nan),
                                                       period=match.period, pitch=match.pitch)
            state = pipeline(mm, score)
            frac = [state[a:b].mean() for a, b in spans]
            rows.append({"match": m, "ball": cond, "holds": len(spans),
                         "holds_mostly_dead": int(sum(f >= 0.5 for f in frac)),
                         "hold_frames_dead": float(np.mean(np.concatenate([state[a:b] for a, b in spans]))) if spans else np.nan})
        print("done", m, flush=True)
    t = pd.DataFrame(rows)
    print("\n== per match ==")
    print(t.round(3).to_string(index=False))
    print("\n== total ==")
    g = t.groupby("ball")[["holds", "holds_mostly_dead"]].sum()
    g["share_mostly_dead"] = (g["holds_mostly_dead"] / g["holds"]).round(3)
    print(g.to_string())
    print("\nhold length (s): p10/p50/p90/max", np.round(np.percentile(lengths, [10, 50, 90, 100]), 1) if lengths else "-")


if __name__ == '__main__':
    main()
