"""[채택하지 않은 실험] 결과: DFL 골키퍼 쥔 시간 p99 17.8초(6초 초과 26%). 골키퍼 처리는 효과가 없어 최종 단계에서 뺐다.

DFL 7경기에서 골키퍼가 박스 안에서 공을 쥐고 있던 시간의 분포를 잰다(인플레이 중, 공 좌표는 진짜).

    python src/keeper_holds.py

쥐고 있음 = 인플레이 프레임에서 공이 그 골키퍼의 REACH_M 안이고 공이 페널티 박스 안이며, 공 속도가
선수 최고 속도(max_player_speed) 이하(공이 골키퍼와 함께 움직임). 연속 구간의 길이를 센다.
경기 규칙(이 대회 경기 시즌): 6초 넘게 쥐면 반칙(간접 프리킥). 실제로는 잘 적용되지 않는다(2025/26부터 8초).
"""

import os
import sys

import numpy as np
import pandas as pd

HERE = os.path.dirname(os.path.abspath(__file__))
HERE = os.path.dirname(HERE)  # src/ — 이 파일은 src/experiments_dropped/ 안에 있다(채택하지 않은 실험, 기록용)
sys.path[:0] = [HERE, os.path.join(os.path.dirname(HERE), "starter")]
from config import CONFIG  # noqa: E402
import dfl_load  # noqa: E402
import features as fx  # noqa: E402

FPS = CONFIG["fps"]
REACH_M = 1.5  # 발·손이 닿는 거리
KEEPER_BOX_M = (16.5, 40.32)  # 페널티 박스(IFAB): 골라인에서 깊이, 너비


def keeper_hold_runs(tracking, meta):
    match = fx.match_from_dfl(tracking, meta)
    length, width = match.pitch
    alive = tracking["ball_state"].to_numpy() == 1
    ids = [c[:-2] for c in tracking.columns if c.endswith("_x") and not c.startswith("ball")]
    team_of = dict(zip(meta["player_id"], meta["position"]))
    keepers = [k for k, pid in enumerate(i for i in ids if i in team_of) if team_of[pid] == "Goalkeeper"]
    ball = match.ball
    depth, box_w = KEEPER_BOX_M
    with np.errstate(invalid="ignore"):
        box = ((ball[:, 0] <= depth) | (ball[:, 0] >= length - depth)) & (np.abs(ball[:, 1] - width / 2) <= box_w / 2)
        speed = fx.central_speed(ball, match.period, CONFIG["speed_half_window"], np.inf)
        held = np.zeros(len(ball), dtype=bool)
        for k in keepers:
            d = np.hypot(*(match.players[:, k].astype(np.float64) - ball).T)
            held |= d <= REACH_M
        held &= box & alive & (speed <= CONFIG["max_player_speed"])
    runs = fx.runs_of(held)
    return [(b - a) / FPS for a, b in runs]


def main():
    lengths = []
    for m in dfl_load.MATCHES:
        tracking, _, meta = dfl_load.load_match(m)
        r = keeper_hold_runs(tracking, meta)
        lengths += r
        print(m, "holds", len(r), flush=True)
    v = np.array(lengths)
    long = v[v >= 1.0]  # 1초 미만은 스쳐 지나가는 공(받자마자 차기 등). 분포만 따로 본다
    print("\nall runs {}, of which >= 1 s: {}".format(len(v), len(long)))
    print("runs >= 1 s, seconds: p50 {:.1f} p75 {:.1f} p90 {:.1f} p95 {:.1f} p99 {:.1f} max {:.1f}".format(
        *np.percentile(long, [50, 75, 90, 95, 99]), long.max()))
    for s in (6, 8, 10, 15, 20):
        print("  longer than {:>2} s: {} ({:.1%})".format(s, int((long > s).sum()), (long > s).mean()))


if __name__ == '__main__':
    main()
