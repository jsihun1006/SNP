"""프레임 번호를 주면 그 근처의 최종 데드볼 구간을 보여 준다(대회 데이터, 직접 확인용).

    python src/lookup_segments.py 20539 20687 30803
    python src/lookup_segments.py --pad-s 10 135032

각 프레임마다, 그 프레임 앞뒤 pad-s초 안에 걸치는 최종 데드볼 구간(data/dead_balls/final_model/segments.csv):
  start, end, seconds, kind, 모델 확률 평균(mean_score), 공이 보인 비율(seen_share), 프레임이 구간 안인지(contains)
이 목록은 확인용이다. 영상에서 본 판단을 라벨이나 기준값으로 쓰지 않는다.
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
MODEL = os.path.join(ROOT, "data", "dead_balls", "model_" + CONFIG["default_feature_set"])


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("frames", type=int, nargs="+")
    ap.add_argument("--pad-s", type=float, default=5.0)
    args = ap.parse_args()
    pd.set_option("display.width", 200)
    segments = pd.read_csv(os.path.join(FINAL, "segments.csv"))
    score = pd.read_csv(os.path.join(MODEL, "frames.csv"), usecols=["score"])["score"].to_numpy(np.float64)
    seen = pd.read_csv(os.path.join(FINAL, "frame_signals.csv"), usecols=["seen"])["seen"].to_numpy().astype(bool)
    pad = int(args.pad_s * FPS)
    rows = []
    for f in args.frames:
        near = segments[(segments["end"] >= f - pad) & (segments["start"] <= f + pad)]
        if not len(near):
            rows.append({"frame": f, "start": "-", "end": "-", "seconds": "-", "kind": "no dead segment nearby"})
        for s in near.itertuples():
            a, b = int(s.start), int(s.end)
            rows.append({"frame": f, "start": a, "end": b, "seconds": s.seconds, "kind": s.kind,
                         "mean_score": round(float(np.nanmean(score[a:b + 1])), 2),
                         "seen_share": round(float(seen[a:b + 1].mean()), 2), "contains": a <= f <= b})
    print(pd.DataFrame(rows).to_string(index=False))


if __name__ == '__main__':
    main()
