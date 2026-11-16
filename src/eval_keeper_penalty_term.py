"""실험: 골키퍼 쥠을 강제 덮어쓰기(keeper_hold.keep_holds_alive) 대신, 느린 빌드업(⑤)과 같은 벌점항으로 다루면 어떤가
(사용자 제안, 2026-11-15). 보호 구간은 keeper_hold.protect_mask, 벌점은 ⑤의 값(진입 +28, 데드 점수 −2.125) 그대로.

    python src/eval_keeper_penalty_term.py

조건(eval_keeper_hold_rule과 같음): dfl(원본), dfl_hidden(쥠 구간 공 삭제), metrica, skillcorner.
비교:
  hard        지금 파이프라인(강제 규칙, keeper_hold.SETTING)
  soft        벌점항만(강제 규칙 없음)
  soft_hard   벌점항 + 강제 규칙
판정 기준(결과 보기 전에 정함): hard 대비, dfl·dfl_hidden이 떨어지지 않고 둘 중 하나는 오르며,
metrica·skillcorner는 평균이 떨어지지 않고 나빠진 경기가 좋아진 경기보다 적을 것.
"""

import os
import sys

import numpy as np
import pandas as pd

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(HERE)
sys.path[:0] = [HERE, os.path.join(ROOT, "starter")]
from config import CONFIG  # noqa: E402
import features as fx  # noqa: E402
import final_dead_balls as fd  # noqa: E402
import slow_buildup as sb  # noqa: E402
import compare_smoothing as cs  # noqa: E402
import keeper_hold as kh  # noqa: E402
from eval_goal_kickoff import sources  # noqa: E402
from eval_keeper_hold_size import keeper_hold_spans  # noqa: E402

FPS = CONFIG["fps"]


def pipeline(match, score, protect=None):
    sig = fd.ball_signals(match)
    cr = fd.line_crossings(sig)
    cr = cr[(cr > 0) & (match.period[cr] == match.period[cr - 1])]
    state = sb.decode(score, match.players, match.ball, match.pitch, match.period, cr,
                      extra_protect=protect(match, sig, cr) if protect else None, team=match.team)
    state, _ = fd.force_line_outs(state, cr, sig, score, match.period)
    state, _ = fd.split_at_penalty_kicks(state, [k for k, _ in fd.penalty_kicks(match)], match.period, cr)
    return state, sig, cr


def run(cond, name, match, score, label):
    graded = np.asarray(match.period) > 0
    truth = pd.DataFrame({"period": match.period, "dead": label.astype(np.int8)})
    plain, sig, cr = pipeline(match, score)
    hard, _ = kh.keep_holds_alive(plain, match, sig, cr)
    soft, _, _ = pipeline(match, score, kh.protect_mask)
    soft_hard, _ = kh.keep_holds_alive(soft, match, sig, cr)
    rows = []
    for variant, st in (("hard", hard), ("soft", soft), ("soft_hard", soft_hard)):
        changed = plain & ~st
        rows.append({"variant": variant, "cond": cond, "match": name.split(":")[1],
                     "f1": cs.score_one((st & graded).astype(np.int8), truth)["alive_spell_f1"],
                     "to_alive_s": changed.sum() / FPS,
                     "truly_alive": float((~label[changed]).mean()) if changed.any() else np.nan})
    return rows


def main():
    pd.set_option("display.width", 220)
    rows = []
    for name, match, score, label in sources():
        label = np.asarray(label).astype(bool)
        data = name.split(":")[0]
        if data == "dfl":
            rows += run("dfl", name, match, score, label)
            ball = np.asarray(match.ball, dtype=np.float64).copy()
            for a, b in keeper_hold_spans(match, label):
                ball[a:b] = np.nan
            hidden = fx.Match(players=match.players, team=match.team, ball=ball, period=match.period, pitch=match.pitch)
            rows += run("dfl_hidden", name, hidden, score, label)
        else:
            rows += run(data, name, match, score, label)
        print("done", name, flush=True)
    t = pd.DataFrame(rows)
    pv = t.pivot_table(index=["cond", "match"], columns="variant", values="f1")
    print("\n== mean F1 and changed time by variant ==")
    print(t.groupby(["variant", "cond"]).agg(f1=("f1", "mean"), to_alive_s=("to_alive_s", "sum"),
                                              truly_alive=("truly_alive", "mean")).round(4).to_string())
    for variant in ("soft", "soft_hard"):
        d = pv[variant] - pv["hard"]
        s = d.groupby("cond").agg(mean="mean", better=lambda x: int((x > 1e-9).sum()), worse=lambda x: int((x < -1e-9).sum()))
        print("\n== {} vs hard ==".format(variant))
        print(s.round(4).to_string())
        dfl_ok = s.loc["dfl", "mean"] >= -1e-9 and s.loc["dfl_hidden", "mean"] >= -1e-9 and \
            max(s.loc["dfl", "mean"], s.loc["dfl_hidden", "mean"]) > 0
        rest = d.loc[["metrica", "skillcorner"]]
        rest_ok = (s.loc[["metrica", "skillcorner"], "mean"] >= -1e-9).all() and \
            int((rest < -1e-9).sum()) < int((rest > 1e-9).sum())
        print("{}: {}".format(variant, "PASS" if dfl_ok and rest_ok else "not passed"))


if __name__ == '__main__':
    main()
