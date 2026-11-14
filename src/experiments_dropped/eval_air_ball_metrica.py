"""멈춤 확인(eval_air_ball.flew_back)을 Metrica 샘플 2경기의 진짜 공 좌표에서 시험한다(손해가 없는지).

    python src/experiments_dropped/eval_air_ball_metrica.py

Metrica에는 공 높이가 없어 투영 재현은 못 한다. 데드볼 동안 공을 지우지만, 라인을 넘는 순간과 그 직후는
인플레이 공이라 라인 아웃 처리의 손해를 볼 수 있다. 확률은 metrica_eval.py가 저장한 DFL 전용 모델 확률을 쓴다.
"""

import os
import sys

import numpy as np
import pandas as pd

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path[:0] = [HERE, os.path.dirname(HERE)]  # experiments_dropped/와 src/
from config import CONFIG  # noqa: E402
import compare_smoothing as cs  # noqa: E402
import eval_air_ball as ea  # noqa: E402
import final_dead_balls as fd  # noqa: E402
import metrica_load as ml  # noqa: E402
import metrics  # noqa: E402
import smoothing  # noqa: E402


def main():
    pd.set_option("display.width", 200)
    choice = CONFIG["smooth_choice"]
    rows = []
    for g in ml.GAMES:
        tracking, events = ml.load_game(g)
        spells = ml.dead_spells(tracking, events)
        label = ml.dead_label(tracking, spells)
        match = ml.match_from_metrica(tracking)
        period = match.period
        path = os.path.join(ml.CACHE, "game{}_score_{}.npy".format(g, CONFIG["default_feature_set"]))
        score = np.load(path).astype(np.float64)
        sig = fd.ball_signals(match)
        crossings = fd.line_crossings(sig)
        back = ea.flew_back(crossings, sig, period)
        truth = pd.DataFrame({"period": period, "dead": label})
        for name, cr in (("model_only", np.array([], dtype=np.int64)), ("current", crossings),
                         ("motion", crossings[~back])):
            if len(cr):
                state = ea.run(score, period, sig, cr)
            else:
                state = smoothing.viterbi(score, period, choice["penalty"], choice["bias"]).astype(np.int8)
            r = cs.score_one(state, truth)
            seg = metrics.segment_metrics(spells, state, period)
            rows.append({"game": g, "variant": name, "alive_spell_f1": r["alive_spell_f1"],
                         "spell_recall": seg["spell_recall"], "segment_precision": seg["segment_precision"],
                         "crossings": len(cr)})
        real = label[np.minimum(crossings + CONFIG["fps"], len(label) - 1)] == 1
        print("game {}: crossings {}, real {:.1%}; removed by motion check {} (of which real {})".format(
            g, len(crossings), real.mean() if len(real) else np.nan, int(back.sum()), int((back & real).sum())))
    t = pd.DataFrame(rows)
    print(t.round(4).to_string(index=False))


if __name__ == '__main__':
    main()
