"""공개 5분 정답에서 D가 언제 시작하는지 이해하기 위한 확인 도구(직접 연 터미널에서 실행, 결과는 AI에 올리지 않는다).

    python src/inspect_gt_dead_start.py

공개 정답(evaluation/gt/BAR-ATH_possession_gt_public.csv)의 D 구간마다, 대회 ATD에서 공이 라인을 넘은 가장 가까운
프레임(final_dead_balls.line_crossings)과의 시간 차를 보여 준다.
  offset_s = D 시작 − 라인 넘음.  0 근처 = 넘는 순간 D,  + = 넘은 뒤 늦게 D,  − = 넘기 전에 이미 D
라인 넘음이 ±10초 안에 없으면(파울·오프사이드 등 라인 아웃이 아닌 데드볼) 'no crossing'으로 표시한다.
같은 방식으로 D 끝과 그 뒤 공 첫 움직임(재개 킥 후보)의 차이도 보여 준다.
이 결과로 설정값을 고르지 않는다. D의 정의(시작·끝을 어디로 보는지)를 이해하는 데만 쓴다.
"""

import os
import sys

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(HERE)
sys.path[:0] = [HERE, ROOT, os.path.join(ROOT, "starter"), os.path.join(ROOT, "evaluation")]
import possession_metrics as pm  # noqa: E402
import final_dead_balls as fd  # noqa: E402
import competition_data  # noqa: E402

FPS = 25
WINDOW_S = 10


def main():
    gt = pm.label_codes(pm.read_label_csv(os.path.join(ROOT, "evaluation", "gt", "BAR-ATH_possession_gt_public.csv")))
    match, _ = competition_data.load_competition_match()
    sig = fd.ball_signals(match)
    crossings = np.asarray(fd.line_crossings(sig))
    ball = np.asarray(match.ball, dtype=np.float64)
    speed = np.full(len(ball), np.nan)
    speed[1:] = np.linalg.norm(np.diff(ball, axis=0), axis=1) * FPS
    dead = gt == pm.D
    scored = gt != pm.X
    edges = np.diff(np.r_[False, dead, False].astype(np.int8))
    starts, stops = np.flatnonzero(edges == 1), np.flatnonzero(edges == -1)
    print("public D spells: {}\n".format(len(starts)))
    print("{:>8} {:>8} {:>7} | {:>18} | {:>20}".format("start", "end", "len_s", "start - crossing s",
                                                          "first ball move - end s"))
    offsets = []
    for a, b in zip(starts, stops):
        if not scored[a]:
            continue
        near = crossings[np.abs(crossings - a) <= WINDOW_S * FPS]
        if len(near):
            c = near[np.argmin(np.abs(near - a))]
            off = (a - c) / FPS
            offsets.append(off)
            cross_txt = "{:+.2f}".format(off)
        else:
            cross_txt = "no crossing"
        after = np.flatnonzero(np.nan_to_num(speed[max(b - 3 * FPS, 0):b + 3 * FPS], nan=0) > 3.0)
        move_txt = "{:+.2f}".format((max(b - 3 * FPS, 0) + after[0] - b) / FPS) if len(after) else "ball not seen"
        print("{:>8} {:>8} {:>7.1f} | {:>18} | {:>20}".format(a, b, (b - a) / FPS, cross_txt, move_txt))
    if offsets:
        o = np.array(offsets)
        print("\nD start - line crossing, over {} line-out D spells: median {:+.2f} s, range {:+.2f} .. {:+.2f} s".format(
            len(o), np.median(o), o.min(), o.max()))


if __name__ == "__main__":
    main()
