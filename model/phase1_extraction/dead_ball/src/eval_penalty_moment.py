"""실험: run_up_v2가 차는 순간으로 내놓는 프레임을 무엇으로 할지 비교한다.

    python src/eval_penalty_moment.py

  arrival  키커가 마크 ARRIVE_M(1.5 m) 안에 들어온 순간(지금). 천천히 들어오는 키커는 일찍 잡힌다(대회 한 장면 0.4초).
  break    그 뒤 배치가 처음 깨진 순간(차면 다른 선수들이 박스로 들어온다).
공개 실제 페널티킥(DFL 3, SkillCorner 3, Metrica 1; DFL 시뮬레이션과 SkillCorner 검출 선수만 조건 포함)에서
실제 킥과의 오차를 본다. 고르는 기준(결과 보기 전에 정함, 2026-11-15): 원본 데이터(dfl, metrica, skillcorner_all,
skillcorner_detected) 오차 절댓값의 중앙값이 작은 쪽. 같으면 arrival 유지. 새 상수 없음.
결과: arrival 0.20초, break 0.08초(5개), 가짜 감지 둘 다 0 → break 채택(final_dead_balls.penalty_kicks).
"""

import os
import sys

import numpy as np
import pandas as pd

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = __import__("repo_root").ROOT
sys.path[:0] = [HERE, os.path.join(ROOT, "starter")]
from config import CONFIG  # noqa: E402
import eval_penalty_combo as pk  # noqa: E402
import eval_penalty_detect as epd  # noqa: E402

FPS = CONFIG["fps"]


def main():
    pd.set_option("display.width", 200)
    rows, false = [], []
    for gen in (epd.dfl_sets, epd.skillcorner_sets, epd.metrica_sets):
        for name, match, pens in gen():
            kicks = [stop for _, stop in pens]
            for moment in ("arrival", "break"):
                found = pk.run_up_v2(match, moment)
                false.append({"set": name.split(":")[0], "moment": moment,
                              "false": sum(1 for k in found if not any(abs(k - t) <= 5 * FPS for t in kicks))})
                for t in kicks:
                    near = [k for k in found if abs(k - t) <= 5 * FPS]
                    if near:
                        rows.append({"data": name, "set": name.split(":")[0], "moment": moment,
                                     "error_s": (min(near, key=lambda k: abs(k - t)) - t) / FPS})
            print("done", name, flush=True)
    t = pd.DataFrame(rows)
    print("\n== kick time error (s), detection minus true kick ==")
    print(t.pivot_table(index="data", columns="moment", values="error_s").round(2).to_string())
    main_sets = t[t["set"] != "dfl_sim"]
    print("\n== |error| median, original data (sim excluded) ==")
    print(main_sets.assign(abs_err=main_sets["error_s"].abs()).groupby("moment")["abs_err"].agg(["count", "median", "max"]).round(2).to_string())
    print("\n== false detections ==")
    print(pd.DataFrame(false).groupby(["moment", "set"])["false"].sum().unstack().to_string())


if __name__ == '__main__':
    main()
