"""실험: 감지된 킥오프 앞의 세리머니 중 짧은 인플레이를 데드로 메운다(kickoff.fill_before_kickoffs).

    python src/eval_kickoff_fill.py

지금 파이프라인(느린 빌드업 + 라인 아웃 강제 + 페널티킥 끊기)과 인플레이 구간 F1을 비교한다. 데이터는
eval_goal_kickoff.sources(DFL 7, Metrica 3, SkillCorner 20). 공 조건 두 가지(원래 공, 공 전체 없음)이고
기준선도 같은 공 조건으로 다시 계산한다(공은 느린 빌드업 보정과 라인 아웃에 쓰인다).
킥오프 감지 자체(src/eval_kickoff_detect.py): 경기 중 킥오프 DFL 11/14, Metrica 16/18. 하프 시작 킥오프는
트래킹이 킥오프에서 시작해 배치가 안 보여 못 찾는다(메우기와 무관). DFL 가짜 7개 중 6개는 데드볼 안.
채택 기준(결과 보기 전에 정함, 2026-11-15): 두 공 조건 모두 데이터별 평균이 떨어지지 않고, 나빠진 경기가 좋아진 경기보다 적을 것.
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
import kickoff as ko  # noqa: E402
from eval_goal_kickoff import sources  # noqa: E402

FPS = CONFIG["fps"]


def pipeline(match, score):
    sig = fd.ball_signals(match)
    cr = fd.line_crossings(sig)
    cr = cr[(cr > 0) & (match.period[cr] == match.period[cr - 1])]
    state = sb.decode(score, match.players, match.ball, match.pitch, match.period, cr, team=match.team)
    state, _ = fd.force_line_outs(state, cr, sig, score, match.period)
    state, _ = fd.split_at_penalty_kicks(state, [k for k, _ in fd.penalty_kicks(match)], match.period, cr)
    return state


def main():
    pd.set_option("display.width", 220)
    rows = []
    for name, match, score, label in sources():
        graded = np.asarray(match.period) > 0
        truth = pd.DataFrame({"period": match.period, "dead": np.asarray(label).astype(np.int8)})
        kicks = ko.kickoffs(match)
        for cond in ("ball", "no_ball"):
            m = match if cond == "ball" else fx.Match(players=match.players, team=match.team,
                                                      ball=np.full_like(np.asarray(match.ball, dtype=np.float64), np.nan),
                                                      period=match.period, pitch=match.pitch)
            base = pipeline(m, score)
            new, filled = ko.fill_before_kickoffs(base, kicks, match.period)
            changed = new & ~base
            rb = cs.score_one((base & graded).astype(np.int8), truth)
            rn = cs.score_one((new & graded).astype(np.int8), truth)
            rows.append({"data": name.split(":")[0], "match": name.split(":")[1], "ball": cond, "kickoffs": len(kicks),
                         "filled_s": filled / FPS,
                         "filled_truly_dead": float(np.asarray(label)[changed].mean()) if changed.any() else np.nan,
                         "base": rb["alive_spell_f1"], "fill": rn["alive_spell_f1"],
                         "base_spells": rb["pred_spells"], "fill_spells": rn["pred_spells"]})
        print("done", name, flush=True)
    t = pd.DataFrame(rows)
    t["diff"] = t["fill"] - t["base"]
    print("\n== per match ==")
    print(t.round(4).to_string(index=False))
    print("\n== mean by data set ==")
    print(t.groupby(["ball", "data"])[["base", "fill", "diff", "filled_s", "filled_truly_dead"]].mean().round(4).to_string())
    for cond, g in t.groupby("ball"):
        better, worse = int((g["diff"] > 1e-9).sum()), int((g["diff"] < -1e-9).sum())
        ok = (g.groupby("data")["diff"].mean() >= -1e-9).all() and worse < better
        print("{}: better {} / worse {}, sum {:+.4f} -> {}".format(cond, better, worse, g["diff"].sum(),
                                                                   "PASS" if ok else "not passed"))


if __name__ == '__main__':
    main()
