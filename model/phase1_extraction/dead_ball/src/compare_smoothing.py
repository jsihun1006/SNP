"""평활화 방법 비교(DFL). train_dfl.py가 저장한 평가 경기 확률(data/external/dfl/oof/)을 쓴다.

    python src/compare_smoothing.py

설정 선택도 경기 하나씩 빼고 한다: 평가할 경기를 뺀 6경기 평균 점수로 설정을 고른 뒤, 빠진 경기에서 잰다.
주 점수는 대회 채점기(evaluation/possession_metrics.py)의 spell_scores를 인플레이=H, 데드=D로 둔 값이다.
즉 데드볼 사이 인플레이 구간을 IoU로 맞춘 F1이다(팀 구분 없이, 데드볼 판정이 점유 구간을 얼마나 잘 나누는지).
"""

import itertools
import os
import sys

import numpy as np
import pandas as pd

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
sys.path.insert(0, os.path.join(__import__("repo_root").ROOT, "evaluation"))
from config import CONFIG  # noqa: E402
import dfl_load  # noqa: E402
import metrics  # noqa: E402
import possession_metrics as pm  # noqa: E402
import smoothing  # noqa: E402

FPS = CONFIG["fps"]
OOF_DIR = os.path.join(CONFIG["dfl_cache"], "oof")


def candidates():
    out = []
    for m in CONFIG["smooth_min_s"]:
        out.append(("threshold", 0, 0.0, m))
        out += [("median", w, 0.0, m) for w in CONFIG["smooth_median_s"]]
        out += [("viterbi", pen, bias, m) for pen, bias in itertools.product(CONFIG["smooth_viterbi_penalty"],
                                                                             CONFIG["smooth_viterbi_bias"])]
    return out


def apply(method, a, b, min_s, score, period):
    if method == "threshold":
        state = (score >= 0.5).astype(np.int8)
    elif method == "median":
        state = smoothing.median_filter(score, period, int(a * FPS) | 1)
    else:
        state = smoothing.viterbi(score, period, a, b)
    return smoothing.min_duration(state, period, int(min_s * FPS))


def codes_of(dead, period):
    """하프 사이에 X를 한 프레임 끼워 하프를 넘는 구간이 생기지 않게 한다."""
    codes = np.where(dead == 1, pm.D, pm.H).astype(np.int8)
    breaks = np.flatnonzero(np.diff(period) != 0) + 1
    return np.insert(codes, breaks, pm.X)


def score_one(state, frame):
    period, label = frame["period"].to_numpy(), frame["dead"].to_numpy()
    gt, pred = codes_of(label, period), codes_of(state, period)
    runs = pm.valid_runs(gt)
    spell = pm.spell_scores(gt, pred, runs)
    out = {"alive_spell_f1": spell["mean_f1"], "pred_spells": spell["n_pred"], "true_spells": spell["n_gt"]}
    out.update({k: v for k, v in metrics.frame_metrics(label, state).items() if k in ("dead_f1",)})
    return out


def main():
    pd.set_option("display.width", 200)
    kind = sys.argv[1] if len(sys.argv) > 1 else "sim"
    frames = {m: pd.read_parquet(os.path.join(OOF_DIR, "{}_{}.parquet".format(m, kind))) for m in dfl_load.MATCHES}
    grid = candidates()
    results = {}
    for cand in grid:
        for m, f in frames.items():
            state = apply(*cand, f["score"].to_numpy(), f["period"].to_numpy())
            results[(cand, m)] = score_one(state, f)
        print("done", cand, flush=True)
    table = pd.DataFrame([dict(method=c[0], a=c[1], b=c[2], min_s=c[3], match=m, **r)
                          for (c, m), r in results.items()])
    keys = ["method", "a", "b", "min_s"]
    mean = table.groupby(keys)[["alive_spell_f1", "dead_f1", "pred_spells", "true_spells"]].mean()
    print("\n== mean over 7 matches, best per method and min_s ==")
    best = mean.reset_index().sort_values("alive_spell_f1", ascending=False).groupby(["method", "min_s"]).head(1)
    print(best.sort_values(["method", "min_s"]).round(3).to_string(index=False))

    print("\n== nested choice: settings picked on the other 6 matches, scored on the held one ==")
    rows = []
    for method in ("threshold", "median", "viterbi", "any"):
        for held in dfl_load.MATCHES:
            others = table[(table.match != held) & ((table.method == method) | (method == "any"))]
            pick = others.groupby(keys)["alive_spell_f1"].mean().idxmax()
            hit = table[(table.match == held) & (table[keys].apply(tuple, axis=1) == pick)].iloc[0]
            rows.append(dict(family=method, held=held, pick=pick, **hit[["alive_spell_f1", "dead_f1",
                                                                         "pred_spells", "true_spells"]]))
    nested = pd.DataFrame(rows)
    print(nested.groupby("family")[["alive_spell_f1", "dead_f1", "pred_spells", "true_spells"]].mean().round(3))
    print("\npicks:")
    print(nested.groupby("family")["pick"].agg(lambda s: s.value_counts().to_dict()).to_string())


if __name__ == '__main__':
    main()
