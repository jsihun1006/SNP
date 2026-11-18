"""대회 데이터에서 공이 안 보이는 구간(공백)의 길이 분포를 집계한다. 채점 구간(전·후반)만 센다.

    python src/ball_gaps.py      # 직접 연 터미널에서 실행할 것

출력은 집계값뿐이다(개수, 분위수, 길이 구간별 비율). 프레임 번호는 내보내지 않는다.
artifacts/deadball/lightgbm/bar_ath/final_model/frames.csv(src/deadball/final_dead_balls.py 결과)가 있으면, 공백 프레임 중 최종 판정이
데드인 비율도 길이 구간별로 보여 준다(공백이 데드볼 신호인지 보려는 것. 정답이 아니라 우리 판정 기준이다).
"""

import os
import sys

import numpy as np
import pandas as pd

HERE = os.path.dirname(os.path.realpath(__file__))
ROOT = os.path.dirname(os.path.dirname(HERE))
sys.path[:0] = [HERE, os.path.join(ROOT, "src"), ROOT]
from config import CONFIG  # noqa: E402
from features import runs_of  # noqa: E402

FPS = CONFIG["fps"]
BUCKETS_S = [0, 0.5, 1, 2, 3, 5, 10, 30, np.inf]
FINAL_PATH = os.path.join(ROOT, "artifacts", "deadball", "lightgbm", "bar_ath", "final_model", "frames.csv")


def gap_table(seen, blocks):
    """[(start, stop)] 채점 구간 안의 공백. 구간 경계에서 잘린 공백도 그 안의 길이만 센다."""
    gaps = []
    for a, b in blocks:
        gaps += [(a + s, a + e) for s, e in runs_of(~seen[a:b])]
    return gaps


def main():
    import competition_data
    match, blocks = competition_data.load_competition_match()
    seen = np.isfinite(match.ball).all(axis=1)
    graded = np.zeros(len(seen), dtype=bool)
    for a, b in blocks:
        graded[a:b] = True
    gaps = gap_table(seen, blocks)
    lengths = np.array([(b - a) / FPS for a, b in gaps])
    missing = (~seen & graded).sum()

    dead = None
    if os.path.exists(FINAL_PATH):
        dead = pd.read_csv(FINAL_PATH)["dead"].to_numpy().astype(bool)[:len(seen)]

    lines = ["== ball gaps in graded halves ==",
             "graded frames {}, ball missing {} ({:.1%}), gaps {}".format(
                 graded.sum(), missing, missing / graded.sum(), len(gaps)),
             "gap length s: p10 {:.2f} p25 {:.2f} p50 {:.2f} p75 {:.2f} p90 {:.2f} p99 {:.1f} max {:.1f}".format(
                 *np.percentile(lengths, [10, 25, 50, 75, 90, 99]), lengths.max()),
             "", "by gap length (share of gaps / share of missing frames{}):".format(
                 " / share of those frames our final call marks dead" if dead is not None else "")]
    for low, high in zip(BUCKETS_S[:-1], BUCKETS_S[1:]):
        pick = [(a, b) for (a, b), s in zip(gaps, lengths) if low <= s < high]
        frames = sum(b - a for a, b in pick)
        row = "  {:>4}-{:<4} s  gaps {:5d} ({:5.1%})  frames {:6d} ({:5.1%})".format(
            low, "inf" if np.isinf(high) else high, len(pick), len(pick) / len(gaps), frames, frames / missing)
        if dead is not None and frames:
            row += "  dead {:5.1%}".format(sum(dead[a:b].sum() for a, b in pick) / frames)
        lines.append(row)
    if dead is not None:
        lines.append("")
        lines.append("for reference: our final call marks {:.1%} of graded frames dead; {:.1%} of frames with the "
                     "ball visible".format(dead[graded].mean(), dead[graded & seen].mean()))
    print("\n".join(lines))


if __name__ == '__main__':
    main()
