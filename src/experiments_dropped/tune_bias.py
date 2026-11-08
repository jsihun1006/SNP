"""[채택하지 않은 실험] 결과: 비터비 보정값은 점수를 올리지 못했고, 끝 0.5초 당기기는 사용자 판단으로 넣지 않았다.

고도화 1: 모델이 데드볼을 넓게 잡는 경향(Metrica 예측 45% 대 정답 40%, 대회 47%)을 줄이는 후처리 설정 찾기.

    python src/tune_bias.py

후보: 비터비 데드 쪽 보정값(bias, 로그 오즈. 클수록 데드를 덜 잡는다), 전환 벌점(penalty),
      구간 끝 당기기(end_shift_s, DFL 데드볼은 재개 킥 약 0.5초 뒤에 끝나 모델 끝이 늦다).
평가: DFL 7경기(경기 하나씩 빼고 학습한 확률, 시뮬레이션 버전)와 Metrica 2경기(DFL 전체로 학습한 모델).
선택: Metrica는 한 경기에서 고르고 다른 경기에서 잰다(두 번). DFL 평균이 기준보다 크게 나빠지지 않는지도 본다.
대회 데이터는 쓰지 않는다.
"""

import itertools
import os
import sys

import numpy as np
import pandas as pd

HERE = os.path.dirname(os.path.abspath(__file__))
HERE = os.path.dirname(HERE)  # src/ — 이 파일은 src/experiments_dropped/ 안에 있다(채택하지 않은 실험, 기록용)
sys.path.insert(0, HERE)
from config import CONFIG  # noqa: E402
import compare_smoothing as cs  # noqa: E402
import dfl_load  # noqa: E402
import metrica_load as ml  # noqa: E402
import smoothing  # noqa: E402
from features import runs_of  # noqa: E402

FPS = CONFIG["fps"]
BIASES = [-0.5, 0.0, 0.5, 1.0, 1.5, 2.0, 3.0]
PENALTIES = [20.0, 40.0, 80.0]
END_SHIFTS_S = [0.0, 0.5, 1.0]


def shift_ends(state, period, frames):
    """데드 구간의 끝을 frames만큼 앞당긴다(구간이 사라지지 않게 최소 1프레임은 남긴다)."""
    out = state.copy()
    if frames <= 0:
        return out
    for p in np.unique(period):
        rows = np.flatnonzero(period == p)
        for a, b in runs_of(state[rows] == 1):
            if b < len(rows):  # 하프 끝에서 잘린 구간은 그대로
                cut = min(frames, b - a - 1)
                out[rows[b - cut:b]] = 0
    return out


def datasets():
    out = {}
    for m in dfl_load.MATCHES:
        f = pd.read_parquet(os.path.join(CONFIG["dfl_cache"], "oof", "{}_sim.parquet".format(m)))
        out[("dfl", m)] = (f["score"].to_numpy(np.float64), f["period"].to_numpy(), f["dead"].to_numpy())
    for g in ml.GAMES:
        tracking, events = ml.load_game(g)
        label = ml.dead_label(tracking, ml.dead_spells(tracking, events))
        score = np.load(os.path.join(ml.CACHE, "game{}_score.npy".format(g))).astype(np.float64)
        out[("metrica", g)] = (score, tracking["period"].to_numpy(), label)
    return out


def main():
    pd.set_option("display.width", 220)
    data = datasets()
    rows = []
    for bias, penalty in itertools.product(BIASES, PENALTIES):
        for key, (score, period, label) in data.items():
            base = smoothing.viterbi(score, period, penalty, bias)
            for shift in END_SHIFTS_S:
                state = shift_ends(base, period, int(shift * FPS))
                r = cs.score_one(state, pd.DataFrame({"period": period, "dead": label}))
                rows.append(dict(source=key[0], match=key[1], bias=bias, penalty=penalty, end_shift_s=shift,
                                 pred_share=state.mean(), true_share=label.mean(), **r))
        print("done bias {} penalty {}".format(bias, penalty), flush=True)
    table = pd.DataFrame(rows)
    keys = ["bias", "penalty", "end_shift_s"]
    mean = table.groupby(keys + ["source"])[["alive_spell_f1", "dead_f1", "pred_share", "true_share"]].mean()
    wide = mean.unstack("source")
    wide.columns = ["{}_{}".format(a, b) for a, b in wide.columns]
    current = wide.loc[(0.0, 40.0, 0.0)]
    print("\n== current setting (bias 0, penalty 40, no end shift) ==")
    print(current.round(3).to_string())

    print("\n== top 12 by Metrica alive_spell_f1 (mean of 2 games), with DFL alongside ==")
    cols = ["alive_spell_f1_metrica", "alive_spell_f1_dfl", "dead_f1_metrica", "dead_f1_dfl",
            "pred_share_metrica", "true_share_metrica", "pred_share_dfl", "true_share_dfl"]
    print(wide.sort_values("alive_spell_f1_metrica", ascending=False)[cols].head(12).round(3).to_string())

    print("\n== nested on Metrica: pick on one game, score on the other ==")
    met = table[table.source == "metrica"]
    for pick_game, test_game in ((1, 2), (2, 1)):
        best = met[met.match == pick_game].groupby(keys)["alive_spell_f1"].mean().idxmax()
        hit = met[(met.match == test_game) & (met[keys].apply(tuple, axis=1) == best)].iloc[0]
        base = met[(met.match == test_game) & (met.bias == 0) & (met.penalty == 40) & (met.end_shift_s == 0)].iloc[0]
        dfl = wide.loc[best, "alive_spell_f1_dfl"]
        print("pick on game {} -> {}: game {} alive_spell_f1 {:.3f} (current {:.3f}), dead share {:.1%} (true {:.1%}); "
              "DFL mean {:.3f} (current {:.3f})".format(pick_game, best, test_game, hit.alive_spell_f1,
                                                         base.alive_spell_f1, hit.pred_share, hit.true_share, dfl,
                                                         current["alive_spell_f1_dfl"]))

    print("\n== bias alone (penalty 40, no end shift) ==")
    print(wide.xs((40.0, 0.0), level=("penalty", "end_shift_s"))[cols].round(3).to_string())
    print("\n== end shift alone (bias 0, penalty 40) ==")
    print(wide.xs((0.0, 40.0), level=("bias", "penalty"))[cols].round(3).to_string())


if __name__ == '__main__':
    main()
