"""대회 데이터에서 지금 파이프라인과 느린 빌드업(⑤) 변형의 판정 결과만 비교한다(정답은 쓰지 않는다).
직접 연 터미널에서 실행할 것(대회 트래킹을 읽는다). 먼저 python src/apply_competition.py 결과가 있어야 한다.

    python src/compare_variants_competition.py                          # 기본: short_islands
    python src/compare_variants_competition.py --variant no_exit
    python src/compare_variants_competition.py --variant same_team

출력: 채점 구간의 데드볼 비율, 데드볼 구간 수, 5초 미만 데드볼 구간 수, 바뀐 시간(데드→인플레이, 인플레이→데드),
바뀐 구간 목록(시작 프레임, 시각, 길이, 바뀐 방향). 파이프라인 단계는 final_dead_balls.main과 같다
(⑤ → 라인 아웃 → 페널티킥 → 골키퍼 쥠). 공개 킥오프·시작 보정 단계는 두 쪽에 똑같아서 생략한다.
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
from features import runs_of  # noqa: E402
import final_dead_balls as fd  # noqa: E402
import slow_buildup as sb  # noqa: E402

FPS = CONFIG["fps"]
VARIANTS = {"short_islands": {"revert_short_islands": True}, "no_exit": {"no_exit": True},
            "same_team": {"same_team_pass": True}}


def run(match, score, sig, cross_in, setting):
    import kickoff
    import keeper_hold
    graded = match.period > 0
    st = sb.decode(score, match.players, match.ball, match.pitch, match.period, cross_in, setting=setting, team=match.team)
    st, _ = fd.force_line_outs(st, cross_in, sig, score, match.period)
    st &= graded
    pen = [k for k, _ in fd.penalty_kicks(match) if graded[k]]
    st, _ = fd.split_at_penalty_kicks(st, pen, match.period, cross_in)
    st, _ = kickoff.hold_after_penalty_goals(st, pen, kickoff.kickoffs(match), match.period)
    st &= graded
    st, _ = keeper_hold.keep_holds_alive(st, match, sig, cross_in)
    return st & graded


def stats(st, graded, blocks):
    segs = [(a0 + a, a0 + b) for a0, b0 in blocks for a, b in runs_of(st[a0:b0])]
    return {"dead_share": float(st[graded].mean()), "dead_segments": len(segs),
            "short_dead_<5s": sum(1 for a, b in segs if b - a < 5 * FPS)}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--variant", choices=sorted(VARIANTS), default="short_islands")
    args = ap.parse_args()
    import competition_data
    from apply_competition import OUT_DIR as model_dir, clock
    match, blocks = competition_data.load_competition_match()
    n = len(match.period)
    graded = match.period > 0
    sig = fd.ball_signals(match)
    crossings = fd.line_crossings(sig)
    cross_in = crossings[(crossings > 0) & (crossings < n)]
    cross_in = cross_in[match.period[cross_in] == match.period[cross_in - 1]]
    score = pd.read_csv(os.path.join(model_dir, "frames.csv"), usecols=["score"])["score"].to_numpy(np.float64)
    base = run(match, score, sig, cross_in, dict(CONFIG["slow_buildup"]))
    alt = run(match, score, sig, cross_in, dict(CONFIG["slow_buildup"], **VARIANTS[args.variant]))
    t = pd.DataFrame([dict(version="current (keeper v8)", **stats(base, graded, blocks)),
                      dict(version="current + " + args.variant, **stats(alt, graded, blocks))])
    pd.set_option("display.width", 200)
    print(t.round(4).to_string(index=False))
    to_dead = (~base & alt & graded).sum() / FPS
    to_alive = (base & ~alt & graded).sum() / FPS
    print("\nchanged: in play -> dead {:.1f} s, dead -> in play {:.1f} s".format(to_dead, to_alive))
    rows = []
    for a, b in runs_of((base != alt) & graded):
        rows.append({"start": a, "clock": clock(a), "seconds": round((b - a) / FPS, 1),
                     "change": "to_dead" if alt[a] else "to_in_play"})
    if rows:
        print("\nchanged stretches:")
        print(pd.DataFrame(rows).to_string(index=False))


if __name__ == '__main__':
    main()
