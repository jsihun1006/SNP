"""최종 데드볼 구간 중 짧은 구간의 프레임 목록을 뽑는다(대회 데이터, 직접 확인용).

    python src/list_short_dead.py            # 10초 미만
    python src/list_short_dead.py --max-s 5  # 5초 미만

입력: data/dead_balls/final_model/segments.csv, frame_signals.csv (src/final_dead_balls.py 결과),
      data/dead_balls/model_v2/frames.csv의 score (src/apply_competition.py 결과)
출력: data/dead_balls/final_model/short_segments.csv 와 터미널 표
  start, end, seconds, half, half_clock(해당 하프 채점 시작부터 mm:ss), video_clock(프레임/25),
  kind, mean_score(모델 데드 확률 평균), seen_share(공이 보인 프레임 비율)
이 목록은 확인용이다. 본 결과로 기준값이나 규칙을 고르지 않는다.
"""

import argparse
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
MODEL = os.path.join(ROOT, "data", "dead_balls", "model_v2")  # 모델 확률(score)은 apply_competition.py 결과에 있다


def mmss(seconds):
    seconds = int(round(seconds))
    return "{:02d}:{:02d}".format(seconds // 60, seconds % 60)


def main():
    from games import GAME, half_video_blocks
    ap = argparse.ArgumentParser()
    ap.add_argument("--max-s", type=float, default=10.0)
    args = ap.parse_args()
    pd.set_option("display.width", 200)
    pd.set_option("display.max_rows", 500)

    segments = pd.read_csv(os.path.join(FINAL, "segments.csv"))
    score = pd.read_csv(os.path.join(MODEL, "frames.csv"), usecols=["score"])["score"].to_numpy(np.float64)
    seen = pd.read_csv(os.path.join(FINAL, "frame_signals.csv"), usecols=["seen"])["seen"].to_numpy().astype(bool)
    blocks = list(half_video_blocks(GAME))

    rows = []
    for s in segments.itertuples():
        if s.seconds >= args.max_s:
            continue
        a, b = int(s.start), int(s.end)
        block = next(((h, first) for h, first, last in blocks if first <= a <= last), None)
        if block is None:  # 채점 구간 밖
            continue
        half, first = block
        rows.append({"start": a, "end": b, "seconds": round(s.seconds, 1), "half": half,
                     "half_clock": mmss((a - first) / FPS), "video_clock": mmss(a / FPS), "kind": s.kind,
                     "mean_score": round(float(np.nanmean(score[a:b + 1])), 2),
                     "seen_share": round(float(seen[a:b + 1].mean()), 2)})
    t = pd.DataFrame(rows)
    out = os.path.join(FINAL, "short_segments.csv")
    t.to_csv(out, index=False)
    print("short dead segments (< {:g}s): {}".format(args.max_s, len(t)))
    print(t.to_string(index=False))
    print("\nsaved:", out)


if __name__ == '__main__':
    main()
