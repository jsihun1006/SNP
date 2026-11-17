"""지금 최종 파이프라인의 DFL 7경기 성능(인플레이 구간 F1). final_dead_balls.main과 같은 단계
(⑤ → 라인 아웃 → 페널티킥 끊기·페널티 골 → 골키퍼 쥠, keeper_hold.SETTING). 모델 확률은 경기별 OOF.

    python src/eval_dfl_current.py

조건: ball(원본 좌표), hidden(골키퍼 쥠 구간 공 삭제), flight(쥠 시작 1초 전부터 공 삭제).
함께 출력: 정답·판정의 데드볼 비율, 데드볼 구간 수, 골키퍼 규칙 없이 돌린 F1(규칙의 몫).
"""

import os
import sys

import numpy as np
import pandas as pd

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = __import__("repo_root").ROOT
sys.path[:0] = [HERE, os.path.join(ROOT, "starter")]
from config import CONFIG  # noqa: E402
import features as fx  # noqa: E402
import final_dead_balls as fd  # noqa: E402
import slow_buildup as sb  # noqa: E402
import compare_smoothing as cs  # noqa: E402
import kickoff  # noqa: E402
import keeper_hold as kh  # noqa: E402
from eval_keeper_hold_size import keeper_hold_spans  # noqa: E402

FPS = CONFIG["fps"]


def run(match, score, keeper=True):
    graded = np.asarray(match.period) > 0
    sig = fd.ball_signals(match)
    cr = fd.line_crossings(sig)
    cr = cr[(cr > 0) & (match.period[cr] == match.period[cr - 1])]
    sup = {}
    st = sb.decode(score, match.players, match.ball, match.pitch, match.period, cr, team=match.team, support=sup)
    st, _ = fd.force_line_outs(st, cr, sig, score, match.period)
    if CONFIG["slow_buildup"].get("restart_snap", False):
        st, _ = sb.snap_dead_ends(st, sup["restart"], match.period)
    st &= graded
    pen = [k for k, _ in fd.penalty_kicks(match) if graded[k]]
    st, _ = fd.split_at_penalty_kicks(st, pen, match.period, cr)
    st, _ = kickoff.hold_after_penalty_goals(st, pen, kickoff.kickoffs(match), match.period)
    st &= graded
    if keeper:
        st, _ = kh.keep_holds_alive(st, match, sig, cr)
    return st & graded


def main():
    import dfl_load
    import train_dfl
    pd.set_option("display.width", 200)
    rows = []
    for m in dfl_load.MATCHES:
        tracking, events, meta = dfl_load.load_match(m)
        raw = fx.match_from_dfl(tracking, meta)
        oof = pd.read_parquet(os.path.join(train_dfl.oof_dir(), "{}_sim.parquet".format(m)))
        match = fx.Match(players=raw.players, team=raw.team, ball=raw.ball,
                         period=oof["period"].to_numpy().astype(np.int64), pitch=raw.pitch)
        label = oof["dead"].to_numpy().astype(bool)
        score = oof["score"].to_numpy(np.float64)
        graded = np.asarray(match.period) > 0
        truth = pd.DataFrame({"period": match.period, "dead": label.astype(np.int8)})
        spans = keeper_hold_spans(match, label)
        conds = {"ball": match}
        for name, lead in (("hidden", 0), ("flight", FPS)):
            ball = np.asarray(match.ball, dtype=np.float64).copy()
            for a, b in spans:
                ball[max(a - lead, 0):b] = np.nan
            conds[name] = fx.Match(players=raw.players, team=raw.team, ball=ball, period=match.period, pitch=raw.pitch)
        for cond, mm in conds.items():
            st = run(mm, score)
            no_keeper = run(mm, score, keeper=False)
            r = cs.score_one(st.astype(np.int8), truth)
            rows.append({"cond": cond, "match": m, "f1": r["alive_spell_f1"],
                         "f1_without_keeper_rule": cs.score_one(no_keeper.astype(np.int8), truth)["alive_spell_f1"],
                         "dead_share_truth": float(label[graded].mean()), "dead_share_pred": float(st[graded].mean()),
                         "spells_truth": r["true_spells"], "spells_pred": r["pred_spells"]})
        print("done", m, flush=True)
    t = pd.DataFrame(rows)
    print("\n== per match (ball) ==")
    print(t[t["cond"] == "ball"].round(4).to_string(index=False))
    print("\n== mean by condition ==")
    print(t.groupby("cond")[["f1", "f1_without_keeper_rule", "dead_share_truth", "dead_share_pred",
                              "spells_truth", "spells_pred"]].mean().round(4).to_string())


if __name__ == '__main__':
    main()
