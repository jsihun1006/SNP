"""라인 아웃이 보인 프레임에서 데드로 들어가는 전환 벌점을 없앤 비터비가 나은지 DFL로 검증한다.

    python src/eval_crossing.py

DFL은 데드볼 동안 공을 고정하지만, 데드볼 직전까지는 공이 실제로 움직이므로 라인을 넘는 순간은 진짜 신호다.
쓰는 확률은 train_dfl.py가 저장한 경기 하나씩 빼고 학습한 모델의 확률(기본 특징 세트, 시뮬레이션 버전)이다.
라인 아웃 판정은 final_dead_balls.py와 같은 함수를 쓴다. 새 상수는 없다.
"""

import os
import sys

import numpy as np
import pandas as pd

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path[:0] = [HERE, os.path.join(os.path.dirname(HERE), "starter")]
from config import CONFIG  # noqa: E402
import compare_smoothing as cs  # noqa: E402
import dfl_labels  # noqa: E402
import dfl_load  # noqa: E402
import features as fx  # noqa: E402
import final_dead_balls as fd  # noqa: E402
import metrics  # noqa: E402
import smoothing  # noqa: E402
import train_dfl  # noqa: E402

FPS = CONFIG["fps"]


def main():
    pd.set_option("display.width", 200)
    choice = CONFIG["smooth_choice"]
    rows, causes, cross_rows = [], [], []
    for m in dfl_load.MATCHES:
        tracking, events, meta = dfl_load.load_match(m)
        match = fx.match_from_dfl(tracking, meta)
        sig = fd.ball_signals(match)
        crossings = fd.line_crossings(sig)
        oof = pd.read_parquet(os.path.join(train_dfl.oof_dir(), "{}_sim.parquet".format(m)))
        score, period, label = oof["score"].to_numpy(np.float64), oof["period"].to_numpy(), oof["dead"].to_numpy()
        spells = dfl_labels.dead_spells(tracking, events)
        starts = spells["start"].to_numpy()
        for c in crossings:
            k = np.argmin(np.abs(starts - c))
            cross_rows.append({"match": m, "dead_after": label[min(c + FPS, len(label) - 1)] == 1,
                               "offset_s": (c - starts[k]) / FPS})
        free = np.zeros(len(score), dtype=bool)
        free[crossings] = True
        free_state = smoothing.viterbi(score, period, choice["penalty"], choice["bias"], free_entry=free).astype(bool)
        forced_state, _ = fd.force_line_outs(free_state, crossings, sig, score, period)
        for name, state in (("baseline", smoothing.viterbi(score, period, choice["penalty"], choice["bias"])),
                            ("free_entry_at_line_out", free_state.astype(np.int8)),
                            ("free_entry_plus_forced_line_out", forced_state.astype(np.int8))):
            r = cs.score_one(state, pd.DataFrame({"period": period, "dead": label}))
            seg = metrics.segment_metrics(spells, state, period)
            found = seg.pop("found")
            rows.append(dict(match=m, variant=name, alive_spell_f1=r["alive_spell_f1"], dead_f1=r["dead_f1"],
                             pred_spells=r["pred_spells"], true_spells=r["true_spells"],
                             spell_recall=seg["spell_recall"], segment_precision=seg["segment_precision"],
                             start_abs_med=seg["start_abs_med"]))
            causes.append(spells.assign(found=found, variant=name))
        print("done", m, "crossings", len(crossings), flush=True)

    cross = pd.DataFrame(cross_rows)
    print("\n== line crossings on DFL ({} in 7 matches) ==".format(len(cross)))
    print("followed by a dead ball 1 s later: {:.1%}".format(cross["dead_after"].mean()))
    near = cross["offset_s"].abs() <= 2
    print("crossing minus nearest dead start (s), |offset| <= 2 s ({:.0%} of crossings): p10 {:.2f} p50 {:.2f} "
          "p90 {:.2f}".format(near.mean(), *np.percentile(cross.loc[near, "offset_s"], [10, 50, 90])))

    table = pd.DataFrame(rows)
    print("\n== per match ==")
    print(table.pivot(index="match", columns="variant", values="alive_spell_f1").round(3).to_string())
    print("\n== mean over 7 matches ==")
    print(table.groupby("variant")[["alive_spell_f1", "dead_f1", "spell_recall", "segment_precision",
                                     "start_abs_med", "pred_spells", "true_spells"]].mean().round(3).T.to_string())
    by = pd.concat(causes).groupby(["cause", "variant"])["found"].mean().unstack().round(3)
    print("\n== spell recall by restart ==")
    print(by.to_string())


if __name__ == '__main__':
    main()
