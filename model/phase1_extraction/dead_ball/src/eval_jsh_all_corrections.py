"""실험: JSH 노트북의 데드볼 보정 ①~⑥을 지금 파이프라인(⑤ + 라인 아웃 + 페널티킥 + 골키퍼 쥠) 위에서 하나씩, 그리고
전부 켜서 비교한다. 노트북 코드는 git에서 그대로 읽는다(eval_jsh_corrections_metrica.load_notebook_code).

    python src/eval_jsh_all_corrections.py

변형(노트북 decode의 cases, 설정은 COMPETITION_RULES): 5(지금), 5+1, 5+2, 5+3, 5+4, 5+6, 1~6 전부.
노트북 decode는 라인 아웃 강제까지 한다. 그 뒤 페널티킥 끊기·페널티 골·골키퍼 쥠(keeper_hold.SETTING)을 똑같이 얹는다.
데이터: eval_goal_kickoff.sources(DFL 7 원본 좌표 + OOF 확률, Metrica 3, SkillCorner 20).
판정 기준(결과 보기 전에 정함, 2026-11-16): ⑤만(지금) 대비
  주 기준  DFL 평균이 오르고, DFL에서 나빠진 경기가 좋아진 경기보다 적을 것.
  보조 기준 Metrica·SkillCorner 평균이 0.002 넘게 떨어지지 않을 것.
"""

import os
import sys

import numpy as np
import pandas as pd

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = __import__("repo_root").ROOT
sys.path[:0] = [HERE, os.path.join(ROOT, "starter")]
from config import CONFIG  # noqa: E402
import final_dead_balls as fd  # noqa: E402
import compare_smoothing as cs  # noqa: E402
import kickoff  # noqa: E402
import keeper_hold as kh  # noqa: E402
from eval_goal_kickoff import sources  # noqa: E402
from eval_jsh_corrections_metrica import load_notebook_code  # noqa: E402

FPS = CONFIG["fps"]
# 1차(하나씩, 전부): 5+1 DFL +0.0011(2/0) 통과, 5+6 DFL +0.0008(1/0)·Metrica +0.0042 통과, 5+2·5+3·5+4·전부는 불통과
# (전부: DFL +0.0002(3/4), SkillCorner −0.0113). 2차: 통과한 둘을 합친 5+1+6(같은 기준).
CASES = {"5_only": (5,), "5+1+6": (1, 5, 6)}


def later_stages(state, match, sig, cr):
    graded = np.asarray(match.period) > 0
    st = state & graded
    pen = [k for k, _ in fd.penalty_kicks(match) if graded[k]]
    st, _ = fd.split_at_penalty_kicks(st, pen, match.period, cr)
    st, _ = kickoff.hold_after_penalty_goals(st, pen, kickoff.kickoffs(match), match.period)
    st &= graded
    st, _ = kh.keep_holds_alive(st, match, sig, cr)
    return st & graded


def main():
    pd.set_option("display.width", 200)
    jsh = load_notebook_code()
    rows = []
    for name, match, score, label in sources():
        label = np.asarray(label).astype(np.int8)
        truth = pd.DataFrame({"period": match.period, "dead": label})
        geometry = fd.ball_signals(match)
        crossings = fd.line_crossings(geometry)
        crossings = crossings[(crossings > 0) & (match.period[crossings] == match.period[crossings - 1])]
        signals = [jsh["movement_features"](match.players[h], match.team, match.ball[h], match.pitch)
                   for h in jsh["halves"](match.period)]
        game = dict(period=match.period, conditions={"original": (geometry, crossings, signals)})
        for variant, cases in CASES.items():
            decoded, _ = jsh["decode"](score, game, "original", cases, jsh["COMPETITION_RULES"])
            st = later_stages(np.asarray(decoded, dtype=bool), match, geometry, crossings)
            r = cs.score_one(st.astype(np.int8), truth)
            rows.append({"data": name.split(":")[0], "match": name.split(":")[1], "variant": variant,
                         "f1": r["alive_spell_f1"], "spells_pred": r["pred_spells"], "spells_truth": r["true_spells"],
                         "dead_share": float(st[np.asarray(match.period) > 0].mean())})
        print("done", name, flush=True)
    t = pd.DataFrame(rows)
    print("\n== mean by data set ==")
    print(t.pivot_table(index="variant", columns="data", values="f1").round(4).to_string())
    print("\n== predicted in-play spells (truth in brackets) ==")
    print(t.pivot_table(index="variant", columns="data", values="spells_pred").round(1).to_string())
    print(t.groupby("data")["spells_truth"].mean().round(1).to_string())
    pv = t.pivot_table(index=["data", "match"], columns="variant", values="f1")
    print()
    for variant in [v for v in CASES if v != "5_only"]:
        d = pv[variant] - pv["5_only"]
        s = d.groupby("data").agg(mean="mean", better=lambda x: int((x > 1e-9).sum()), worse=lambda x: int((x < -1e-9).sum()))
        main_ok = s.loc["dfl", "mean"] > 0 and s.loc["dfl", "worse"] < s.loc["dfl", "better"]
        aux_ok = (s.loc[["metrica", "skillcorner"], "mean"] >= -0.002).all()
        print("{:8s} DFL {:+.4f} ({}/{})  Metrica {:+.4f}  SkillCorner {:+.4f} ({}/{})  -> {}".format(
            variant, s.loc["dfl", "mean"], s.loc["dfl", "better"], s.loc["dfl", "worse"], s.loc["metrica", "mean"],
            s.loc["skillcorner", "mean"], s.loc["skillcorner", "better"], s.loc["skillcorner", "worse"],
            "PASS" if main_ok and aux_ok else "not passed"))


if __name__ == '__main__':
    main()
