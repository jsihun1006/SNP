"""실험: 페널티킥 뒤 킥오프가 감지되면 골로 보고 킥오프까지 데드로 묶는다(kickoff.hold_after_penalty_goals).
공이 안 보여도 작동하는지가 핵심이다(대회 페널티 골 장면은 공이 안 보였고 세리머니 중 데드볼이 끊겼다).

    python src/eval_penalty_goal_hold.py

지금 파이프라인(느린 빌드업 + 라인 아웃 강제 + 페널티킥 끊기)과 인플레이 구간 F1을 비교한다. 데이터는
eval_goal_kickoff.sources(DFL 7, Metrica 3, SkillCorner 20), 공 조건 두 가지(원래 공, 공 전체 없음, 기준선도 같은 조건).
공개 페널티 골: DFL J03WMX·J03WOH(다음 데드 kick_off), SkillCorner 1874553·2006229.
채택 기준(결과 보기 전에 정함, 2026-11-15): 두 공 조건 모두 나빠진 경기가 없을 것.
결과: 통과(나빠진 경기 0, SkillCorner 1874553 +0.0019). 묶음 4건(DFL 2, Metrica 1, SkillCorner 1), J03WR9은 킥오프 없음 → 채택.
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
import compare_smoothing as cs  # noqa: E402
import kickoff as ko  # noqa: E402
from eval_goal_kickoff import sources  # noqa: E402
from eval_kickoff_fill import pipeline  # noqa: E402

FPS = CONFIG["fps"]


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
            pens = [k for k, _ in fd.penalty_kicks(m)]
            new, made = ko.hold_after_penalty_goals(base, pens, kicks, match.period)
            changed = new & ~base
            rb = cs.score_one((base & graded).astype(np.int8), truth)["alive_spell_f1"]
            rn = cs.score_one((new & graded).astype(np.int8), truth)["alive_spell_f1"]
            rows.append({"data": name.split(":")[0], "match": name.split(":")[1], "ball": cond, "penalties": len(pens),
                         "held": [(round((k - p) / FPS, 1)) for p, k in made], "changed_s": changed.sum() / FPS,
                         "changed_truly_dead": float(np.asarray(label)[changed].mean()) if changed.any() else np.nan,
                         "base": rb, "hold": rn})
        print("done", name, flush=True)
    t = pd.DataFrame(rows)
    t["diff"] = t["hold"] - t["base"]
    print("\n== matches with a penalty kick ==")
    print(t[t["penalties"] > 0].round(4).to_string(index=False))
    for cond, g in t.groupby("ball"):
        better, worse = int((g["diff"] > 1e-9).sum()), int((g["diff"] < -1e-9).sum())
        print("{}: better {} / worse {}, sum {:+.4f} -> {}".format(cond, better, worse, g["diff"].sum(),
                                                                   "PASS" if worse == 0 else "not passed"))


if __name__ == '__main__':
    main()
