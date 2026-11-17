"""진단: 최종 데드볼 파이프라인의 단계마다 판정이 어떻게 바뀌는지 한 지점 주변에서 보여 준다.
직접 연 터미널에서 실행할 것(대회 트래킹을 읽는다). 먼저 python src/apply_competition.py 결과가 있어야 한다.

    python src/trace_stages.py --frame <프레임>            # 앞뒤 20초
    python src/trace_stages.py --frame <프레임> --window 30

한 글자 = 1초(25프레임), D = 그 1초의 절반 이상이 데드, . = 인플레이. 맨 아래 줄은 가운데(--frame) 표시.
단계(final_dead_balls.main과 같은 순서):
  0 model_only      모델 확률 + 비터비(벌점 40)만, 라인 아웃·느린 빌드업 없음
  1 no_slow_buildup 라인 아웃 순간 데드 진입 벌점 0까지(느린 빌드업 ⑤ 끔)
  2 slow_buildup    + 느린 빌드업 ⑤(패스·수신·공 동행 구간은 데드 진입을 어렵게)
  3 line_outs       + 확실한 라인 아웃은 무조건 데드
  4 penalty         + 페널티킥 순간 끊기, 페널티 골이면 킥오프까지 데드
  5 keeper          + 골키퍼 쥠은 인플레이(keeper_hold)
바뀐 단계 옆에 그 단계가 이 범위에서 바꾼 초 수를 적는다. 시작·끝 시각(경기 시계)도 함께 출력한다.
"""

import argparse
import os
import sys

import numpy as np
import pandas as pd

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(HERE)
sys.path[:0] = [HERE, ROOT, os.path.join(ROOT, "starter")]
from config import CONFIG  # noqa: E402
import final_dead_balls as fd  # noqa: E402
import slow_buildup as sb  # noqa: E402
import smoothing  # noqa: E402

FPS = CONFIG["fps"]


def strip(state, a, b):
    return "".join("D" if state[i:min(i + FPS, b)].mean() >= 0.5 else "." for i in range(a, b, FPS))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--frame", type=int, required=True)
    ap.add_argument("--window", type=float, default=20.0, help="앞뒤 몇 초")
    args = ap.parse_args()
    import competition_data
    import kickoff
    import keeper_hold
    from apply_competition import OUT_DIR as model_dir, clock
    match, blocks = competition_data.load_competition_match()
    n = len(match.period)
    graded = match.period > 0
    sig = fd.ball_signals(match)
    crossings = fd.line_crossings(sig)
    cross_in = crossings[(crossings > 0) & (crossings < n)]
    cross_in = cross_in[match.period[cross_in] == match.period[cross_in - 1]]
    score = pd.read_csv(os.path.join(model_dir, "frames.csv"), usecols=["score"])["score"].to_numpy(np.float64)
    choice = CONFIG["smooth_choice"]
    stages = []
    s0 = smoothing.viterbi(score, match.period, choice["penalty"], choice["bias"]).astype(bool) & graded
    stages.append(("0 model_only", s0))
    s1 = sb.decode(score, match.players, match.ball, match.pitch, match.period, cross_in,
                   setting={"margin": 0.0, "strength": 0.0, "bonus": 0.0}) & graded
    stages.append(("1 no_slow_buildup", s1))
    s2 = sb.decode(score, match.players, match.ball, match.pitch, match.period, cross_in, team=match.team) & graded
    stages.append(("2 slow_buildup", s2))
    # 비교용(파이프라인에는 안 씀): ⑤가 진행 중인 데드볼을 끝내지 못하게 한 것, 같은 팀 패스만 보호한 것
    alt = {"2b no_exit": dict(CONFIG["slow_buildup"], no_exit=True),
           "2c same_team": dict(CONFIG["slow_buildup"], same_team_pass=True),
           "2d short_islands": dict(CONFIG["slow_buildup"], revert_short_islands=True)}
    alt_states = {k: sb.decode(score, match.players, match.ball, match.pitch, match.period, cross_in,
                               setting=v, team=match.team) & graded for k, v in alt.items()}
    s3, _ = fd.force_line_outs(s2, cross_in, sig, score, match.period)
    s3 &= graded
    stages.append(("3 line_outs", s3))
    pen = [k for k, _ in fd.penalty_kicks(match) if graded[k]]
    s4, _ = fd.split_at_penalty_kicks(s3, pen, match.period, cross_in)
    s4, _ = kickoff.hold_after_penalty_goals(s4, pen, kickoff.kickoffs(match), match.period)
    s4 &= graded
    stages.append(("4 penalty", s4))
    s5, holds = keeper_hold.keep_holds_alive(s4, match, sig, cross_in)
    s5 &= graded
    stages.append(("5 keeper", s5))
    a = max(args.frame - int(args.window * FPS), 0)
    b = min(args.frame + int(args.window * FPS), n)
    print("frames {}..{}  ({} .. {})".format(a, b - 1, clock(a), clock(b - 1)))
    prev = None
    for name, st in stages:
        changed = "" if prev is None else "  changed {:.1f} s".format((st[a:b] != prev[a:b]).sum() / FPS)
        print("{:18s} {}{}".format(name, strip(st, a, b), changed))
        prev = st
    mid = (args.frame - a) // FPS
    print("{:18s} {}^".format("", " " * mid))
    print("\ncompare stage 2 (not used by the pipeline):")
    for name, st in alt_states.items():
        print("{:18s} {}  vs stage 2 changed {:.1f} s".format(name, strip(st, a, b), (st[a:b] != s2[a:b]).sum() / FPS))
    near = [(s, e) for s, e in holds if s < b and a < e]
    if near:
        print("keeper spans in range: {}".format(", ".join("{}-{}".format(s, e) for s, e in near)))


if __name__ == '__main__':
    main()
