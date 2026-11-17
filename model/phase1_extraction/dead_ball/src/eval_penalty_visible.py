"""실험: 페널티킥 배치의 '보이는 선수 MIN_VISIBLE명 이상' 조건을 빼도 되는지 본다.

    python src/eval_penalty_visible.py

대회 트래킹은 선수가 겹치면 한 명으로 잡혀 보이는 수가 줄 수 있다(사용자 확인, 장면 수치는 보지 않음).
특정 장면이 통과하는 값으로 낮추지 않고, 지금 값(16)과 조건 없음(0) 두 가지만 비교한다(결과 보기 전에 정함, 2026-11-15).
채택 기준: 조건 없음에서 가짜 감지가 늘지 않을 것. 결과: 통과(가짜 0 그대로, 2006229 페널티 골 새로 찾음) → 0 채택. 선수가 적게 잡히는 SkillCorner(검출 선수만)와 DFL 시뮬레이션이
가짜 감지를 가장 엄하게 시험한다. 공 조건은 원래 공과 공 전체 없음(eval_penalty_noball과 같음).
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
import eval_penalty_combo as pk  # noqa: E402
import eval_penalty_detect as epd  # noqa: E402

FPS = CONFIG["fps"]
METHODS = {"setup_then_kick_v2": pk.setup_then_kick_v2, "run_up_v2": pk.run_up_v2}
THRESHOLDS = (16, 0)


def main():
    pd.set_option("display.width", 220)
    rows = []
    for gen in (epd.dfl_sets, epd.skillcorner_sets, epd.metrica_sets):
        for name, match, pens in gen():
            kicks = [stop for _, stop in pens]
            for cond in ("ball", "no_ball"):
                ball = match.ball if cond == "ball" else np.full_like(np.asarray(match.ball, dtype=np.float64), np.nan)
                m = fx.Match(players=match.players, team=match.team, ball=ball, period=match.period, pitch=match.pitch)
                for th in THRESHOLDS:
                    pk.FORMATION_MIN_VISIBLE = th
                    for method, fn in METHODS.items():
                        found = fn(m)
                        good = [k for k in found if any(abs(k - t) <= 5 * FPS for t in kicks)]
                        rows.append({"min_visible": th, "ball": cond, "method": method, "set": name.split(":")[0],
                                     "true": len(kicks),
                                     "found": sum(1 for t in kicks if any(abs(k - t) <= 5 * FPS for k in found)),
                                     "false": len(found) - len(good)})
            pk.FORMATION_MIN_VISIBLE = 0
            print("done", name, flush=True)
    t = pd.DataFrame(rows)
    print("\n== found / false ==")
    print(t.groupby(["ball", "method", "set", "min_visible"])[["true", "found", "false"]].sum()
          .unstack("min_visible").to_string())


if __name__ == '__main__':
    main()
