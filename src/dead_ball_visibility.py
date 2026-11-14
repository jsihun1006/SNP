"""최종 데드볼 구간 안에서 공이 얼마나 보이는지 집계한다(대회 데이터, 집계값만 출력).

    python src/dead_ball_visibility.py      # 직접 연 터미널에서 실행할 것

입력: data/dead_balls/final_model/segments.csv, frame_signals.csv (src/final_dead_balls.py 결과)
보는 것:
  - 데드볼 프레임과 인플레이 프레임에서 공이 보이는 비율(채점 구간)
  - 데드볼 구간마다 공이 보인 비율의 분포: 전혀 안 보임 / 일부 / 대부분 보임
  - 구간 길이별, 판정 이유(kind)별로 같은 집계
프레임 번호는 출력하지 않는다.
"""

import os
import sys

import numpy as np
import pandas as pd

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(HERE)
sys.path[:0] = [HERE, ROOT]
from config import CONFIG  # noqa: E402

FPS = CONFIG["fps"]
FINAL = os.path.join(ROOT, "data", "dead_balls", "final_model")
LENGTH_BINS = [0, 5, 10, 20, 40, np.inf]
SEEN_BINS = [(0.0, 0.0, "0% (never seen)"), (0.0, 0.25, "0-25%"), (0.25, 0.5, "25-50%"), (0.5, 0.75, "50-75%"),
             (0.75, 1.0, "75-100%")]


def length_bin(seconds):
    for low, high in zip(LENGTH_BINS[:-1], LENGTH_BINS[1:]):
        if low <= seconds < high:
            return "{}-{}s".format(low, "inf" if np.isinf(high) else high)


def seen_bin(share):
    if share == 0:
        return SEEN_BINS[0][2]
    for low, high, name in SEEN_BINS[1:]:
        if low < share <= high or (low == 0 and share <= high):
            return name
    return SEEN_BINS[-1][2]


def main():
    from games import GAME, half_video_blocks
    pd.set_option("display.width", 200)
    segments = pd.read_csv(os.path.join(FINAL, "segments.csv"))
    signals = pd.read_csv(os.path.join(FINAL, "frame_signals.csv"), usecols=["frame", "seen"])
    dead = pd.read_csv(os.path.join(FINAL, "frames.csv"))["dead"].to_numpy().astype(bool)
    seen = signals["seen"].to_numpy().astype(bool)
    n = min(len(dead), len(seen))
    dead, seen = dead[:n], seen[:n]
    graded = np.zeros(n, dtype=bool)
    for _, first, last in half_video_blocks(GAME):
        graded[first:last + 1] = True

    lines = ["== ball visibility, graded halves ==",
             "dead frames: ball seen {:.1%} of {} frames".format(seen[dead & graded].mean(), int((dead & graded).sum())),
             "in-play frames: ball seen {:.1%} of {} frames".format(seen[~dead & graded].mean(),
                                                                    int((~dead & graded).sum()))]
    rows = []
    for s in segments.itertuples():
        a, b = int(s.start), int(s.end) + 1
        if not graded[a:b].any():
            continue
        rows.append({"seconds": s.seconds, "kind": s.kind, "seen_share": seen[a:b].mean(),
                     "length": length_bin(s.seconds)})
    t = pd.DataFrame(rows)
    t["seen_bin"] = t["seen_share"].apply(seen_bin)
    order = [name for _, _, name in SEEN_BINS]
    lines += ["", "== final dead segments ({}) by share of frames with the ball seen ==".format(len(t)),
              t["seen_bin"].value_counts().reindex(order).fillna(0).astype(int).to_string(),
              "", "segment seen share: p10 {:.0%} p50 {:.0%} p90 {:.0%}".format(
                  *np.percentile(t["seen_share"], [10, 50, 90])),
              "", "== by segment length: segments, mean seen share, segments with the ball seen > 75% =="]
    by_len = t.groupby("length").agg(segments=("seen_share", "size"), mean_seen=("seen_share", "mean"),
                                     mostly_seen=("seen_share", lambda v: int((v > 0.75).sum())))
    len_order = [length_bin(x + 1e-9) for x in LENGTH_BINS[:-1]]
    lines.append(by_len.reindex([o for o in len_order if o in by_len.index]).round(2).to_string())
    lines += ["", "== by kind =="]
    lines.append(t.groupby("kind").agg(segments=("seen_share", "size"), mean_seen=("seen_share", "mean"),
                                       mostly_seen=("seen_share", lambda v: int((v > 0.75).sum())))
                 .round(2).to_string())
    print("\n".join(lines))


if __name__ == '__main__':
    main()
