"""세 데드볼 모델을 대회 경기에 적용해, 어디서 다르게 판정하는지 뷰어로 직접 보는 파일을 만든다.

채택 안 함: Metrica 3경기에서 비중 맞춘 모델 0.811 대 지금 모델 0.809(1경기만 개선), DFL 0.869 대 0.878.

    python src/compare_models_competition.py      # 직접 연 터미널에서 실행할 것(대회 데이터를 읽는다)

모델(src/train_with_skillcorner.py 실험):
  current          지금 모델(DFL 7경기, data/external/dfl/model_all_v2.txt). 채택된 모델.
  dfl_sc           DFL 7경기 + SkillCorner 20경기. 채택하지 않음(DFL·Metrica에서 손해).
  dfl_sc_balanced  같은 데이터, 두 데이터의 비중을 같게. 채택하지 않음.
후처리는 모두 final_dead_balls.py와 같다(비터비, 라인 아웃, 시작 보정, 공개 킥오프).
확인용이다: 영상에서 본 것으로 모델을 고르거나 기준값을 바꾸지 않는다(모델 선택은 DFL·Metrica로 이미 했다).

출력: data/dead_balls/compare_models/ (커밋 안 함)
  <모델>/viewer.csv    뷰어(dead_balls.csv 칸)에 넣는 구간 목록
  <모델>/segments.csv  구간 목록
  differences.csv     current와 다른 모델이 다르게 판정한 구간: start, end, seconds, model, change
                      change = only_in_<모델>(그 모델만 데드) | only_in_current(지금 모델만 데드)
  viewer_differences.csv  위 목록을 뷰어 형식으로(종류 칸에 change가 나온다)
"""

import os
import sys

import numpy as np
import pandas as pd

HERE = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))  # src/ (이 파일은 experiments_dropped/ 안)
ROOT = os.path.dirname(HERE)
sys.path[:0] = [HERE, ROOT, os.path.join(ROOT, "starter")]
from config import CONFIG  # noqa: E402
import features as fx  # noqa: E402
import final_dead_balls as fd  # noqa: E402
import smoothing  # noqa: E402
import train_dfl  # noqa: E402

FPS = CONFIG["fps"]
F = CONFIG["final"]
OUT_DIR = os.path.join(ROOT, "data", "dead_balls", "compare_models")
SC_DIR = os.path.join(ROOT, "data", "external", "skillcorner")
MODELS = {"current": train_dfl.model_path(CONFIG["default_feature_set"]),
          "dfl_sc": os.path.join(SC_DIR, "model_dfl_sc_v2.txt"),
          "dfl_sc_balanced": os.path.join(SC_DIR, "model_dfl_sc_balanced_v2.txt")}


def final_segments(score, match, blocks, sig, crossings):
    """final_dead_balls.main의 1~4단계. [(start, stop, kind)]."""
    choice = CONFIG["smooth_choice"]
    n = len(score)
    graded = match.period > 0
    free = np.zeros(n, dtype=bool)
    free[crossings[crossings < n]] = True
    state = smoothing.viterbi(score, match.period, choice["penalty"], choice["bias"], free_entry=free).astype(bool)
    state, forced = fd.force_line_outs(state, crossings[crossings < n], sig, score, match.period)
    state &= graded
    model_segments = [(a0 + a, a0 + b) for a0, b0 in blocks for a, b in fx.runs_of(state[a0:b0])]
    final = fd.build(model_segments, crossings, int(F["out_start_window_s"] * FPS))
    first, second_start, second_kick = F["public_kickoffs"]
    final = fd.apply_public_kickoffs(final, first, second_start, second_kick)
    return [(a, b, "forced_line_out" if any(fa < b and a < fb for fa, fb in forced) else k) for a, b, k in final]


def table_of(segments, clock):
    return pd.DataFrame([{"start": a, "end": b - 1, "start_clock": clock(a), "seconds": round((b - a) / FPS, 1),
                          "kind": k} for a, b, k in segments]).sort_values("start").reset_index(drop=True)


def only_in(a_segs, b_segs):
    """a에는 있고 b와는 전혀 겹치지 않는 구간."""
    return [(a, b, k) for a, b, k in a_segs if not any(x < b and a < y for x, y, _ in b_segs)]


def main():
    import lightgbm as lgb
    import competition_data
    from apply_competition import clock
    missing = [name for name, path in MODELS.items() if not os.path.exists(path)]
    if missing:
        raise SystemExit("model files missing: {} (run src/train_with_skillcorner.py first)".format(missing))
    match, blocks = competition_data.load_competition_match()
    sig = fd.ball_signals(match)
    crossings = fd.line_crossings(sig)
    frame = fx.compute_features(match)
    graded = match.period > 0
    results = {}
    lines = ["== three models on the competition match (graded halves, same post-processing) =="]
    for name, path in MODELS.items():
        model = lgb.Booster(model_file=path)
        score = model.predict(frame[model.feature_name()].astype(np.float32)).astype(np.float64)
        segs = final_segments(score, match, blocks, sig, crossings)
        results[name] = segs
        mask = np.zeros(len(score), dtype=bool)
        for a, b, _ in segs:
            mask[a:b] = True
        lengths = np.array([(b - a) / FPS for a, b, _ in segs])
        lines.append("{:16s} segments {:4d}, dead share {:.1%}, shorter than 5 s {}, length s p50 {:.1f}".format(
            name, len(segs), mask[graded].mean(), int((lengths < 5).sum()), np.median(lengths)))
        out = os.path.join(OUT_DIR, name)
        os.makedirs(out, exist_ok=True)
        table = table_of(segs, clock)
        table.to_csv(os.path.join(out, "segments.csv"), index=False)
        fd.viewer_rows(table, blocks).to_csv(os.path.join(out, "viewer.csv"), index=False)

    rows = []
    for name in ("dfl_sc", "dfl_sc_balanced"):
        for a, b, _ in only_in(results[name], results["current"]):
            rows.append({"start": a, "stop": b, "model": name, "change": "only_in_" + name})
        for a, b, _ in only_in(results["current"], results[name]):
            rows.append({"start": a, "stop": b, "model": name, "change": "only_in_current"})
    diff = pd.DataFrame(rows, columns=["start", "stop", "model", "change"])
    lines.append("")
    lines.append("== segments that do not overlap any segment of the current model (and the reverse) ==")
    if len(diff):
        diff["seconds"] = ((diff["stop"] - diff["start"]) / FPS).round(1)
        lines.append(diff.groupby(["model", "change"])["seconds"].agg(segments="size", median_s="median")
                     .round(1).to_string())
        # 같은 구간이 두 모델에서 같은 변화로 나오면 한 번만 보여 준다
        view = diff.drop_duplicates(["start", "stop", "change"]).sort_values("start")
        view_table = pd.DataFrame({"start": view["start"], "end": view["stop"] - 1,
                                   "start_clock": [clock(a) for a in view["start"]], "seconds": view["seconds"],
                                   "kind": [c + ("" if c == "only_in_current" else "") for c in view["change"]]})
        fd.viewer_rows(view_table.reset_index(drop=True), blocks).to_csv(
            os.path.join(OUT_DIR, "viewer_differences.csv"), index=False)
        diff.assign(end=diff["stop"] - 1).drop(columns="stop")[["start", "end", "seconds", "model", "change"]] \
            .sort_values(["start", "model"]).to_csv(os.path.join(OUT_DIR, "differences.csv"), index=False)
    else:
        lines.append("none")
    with open(os.path.join(OUT_DIR, "summary.txt"), "w", encoding="utf-8") as handle:
        handle.write("\n".join(lines) + "\n")
    print("\n".join(lines))
    print("\nwrote", OUT_DIR)


if __name__ == '__main__':
    main()
