"""대회 경기에 데드볼 모델을 적용한다(파이프라인 1/2). 다음 단계는 src/final_dead_balls.py.

    python src/apply_competition.py      # 직접 연 터미널에서 실행할 것(대회 트래킹을 읽는다)

입력: 대회 ATD(data/tactical/), 학습된 모델 data/external/dfl/model_all_<특징 세트>.txt
      (만드는 법: python src/train_dfl.py --save-final)
출력: data/dead_balls/model_<특징 세트>/ (커밋 안 함)
    frames.csv     frame, score(프레임별 데드볼 확률), dead(비터비 평활화한 판정, 채점 구간만)
    segments.csv   데드볼 구간: start, end(포함), start_clock, seconds, mean_score, tagged(근처 SET PIECES 태그 여부)
    summary.txt    화면 출력과 같은 집계

스마트 태깅은 결과 보고(태그를 찾은 비율)에만 쓴다. 정답이 아니라 참고용이다.
공개 정답 5분은 읽지 않는다.
"""

import os
import sys

import numpy as np
import pandas as pd

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = __import__("repo_root").ROOT
sys.path[:0] = [HERE, ROOT, os.path.join(ROOT, "starter")]
from config import CONFIG  # noqa: E402
import features as fx  # noqa: E402
import smoothing  # noqa: E402

FPS = CONFIG["fps"]
OUT_DIR = os.path.join(ROOT, "data", "dead_balls", "model_" + CONFIG["default_feature_set"])


def smooth(score, period):
    choice = CONFIG["smooth_choice"]
    state = smoothing.viterbi(score, period, choice["penalty"], choice["bias"])
    return smoothing.min_duration(state, period, int(choice["min_s"] * FPS))


def segments_of(mask, blocks):
    out = []
    for start, stop in blocks:
        out += [(start + a, start + b) for a, b in fx.runs_of(mask[start:stop])]
    return out


def clock(frame):
    seconds = int(frame // FPS)
    return "{}:{:02d}:{:02d}".format(seconds // 3600, seconds // 60 % 60, seconds % 60)


def overlaps(start, stop, spans):
    return any(a < stop and start < b for a, b in spans)


def quantiles(values):
    return "p10 {:.1f} p50 {:.1f} p90 {:.1f}".format(*np.percentile(values, [10, 50, 90])) if len(values) else "-"


def set_piece_tags(blocks):
    """채점 구간 안의 스마트 태깅 SET PIECES 태그 [{start, end, type}]."""
    import load_smart_tagging
    out = []
    for event in load_smart_tagging.load():
        if (event["code"] or "").strip().lower() != "set pieces":
            continue
        if any(a <= event["start_frame"] < b for a, b in blocks):
            out.append({"start": event["start_frame"], "end": event["end_frame"],
                        "type": (event["tags"].get("Type") or "").strip()})
    return sorted(out, key=lambda t: t["start"])


def apply():
    import lightgbm as lgb
    import competition_data
    import train_dfl

    match, blocks = competition_data.load_competition_match()
    model = lgb.Booster(model_file=train_dfl.model_path(CONFIG["default_feature_set"]))
    columns = model.feature_name()
    frame = fx.compute_features(match)
    score = model.predict(frame[columns].astype(np.float32))
    period = match.period
    graded = period > 0
    state = smooth(score, period).astype(bool) & graded
    n = len(state)

    segs = segments_of(state, blocks)
    lengths = np.array([(b - a) / FPS for a, b in segs])
    lines = ["== dead-ball model on the competition match (graded halves) ==",
             "predicted dead share {:.1%} (0.5 threshold before smoothing: {:.1%})".format(
                 state[graded].mean(), (score[graded] >= 0.5).mean()),
             "dead segments {}, length s {}".format(len(segs), quantiles(lengths))]

    # 스마트 태깅 SET PIECES 태그(참고용). 태그 창은 부정확해서 앞뒤로 넓혀 겹치는지만 본다.
    pad = int(CONFIG["tag_pad_s"] * FPS)
    tags = set_piece_tags(blocks)
    tag_spans = [(t["start"] - pad, t["end"] + pad) for t in tags]
    found = [overlaps(a, b, segs) for a, b in tag_spans]
    lines += ["", "== SET PIECES tags in graded halves ({} tags, window padded {:.0f} s; reference only) ==".format(
        len(tags), CONFIG["tag_pad_s"]),
              "tags with a model dead segment: {}/{}".format(sum(found), len(tags))]
    if tags:
        by_type = pd.DataFrame({"type": [t["type"] or "?" for t in tags], "model": found})
        lines.append(by_type.groupby("type").agg(tags=("model", "size"), model=("model", "sum")).to_string())

    table = pd.DataFrame([{"start": a, "end": b - 1, "start_clock": clock(a), "seconds": round((b - a) / FPS, 1),
                           "mean_score": round(float(score[a:b].mean()), 2), "tagged": overlaps(a, b, tag_spans)}
                          for a, b in segs])
    os.makedirs(OUT_DIR, exist_ok=True)
    pd.DataFrame({"frame": np.arange(n), "score": np.round(score, 3), "dead": state.astype(np.int8)}).to_csv(
        os.path.join(OUT_DIR, "frames.csv"), index=False)
    table.to_csv(os.path.join(OUT_DIR, "segments.csv"), index=False)
    with open(os.path.join(OUT_DIR, "summary.txt"), "w", encoding="utf-8") as handle:
        handle.write("\n".join(lines) + "\n")
    print("\n".join(lines))
    print("\nwrote {}".format(OUT_DIR))


if __name__ == '__main__':
    pd.set_option("display.width", 200)
    apply()
