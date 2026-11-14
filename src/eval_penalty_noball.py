"""실험: 공이 안 보일 때 페널티킥 감지가 버티는지 본다. DFL은 공이 늘 보여서 이 조건을 시험받은 적이 없다.

    python src/eval_penalty_noball.py

공 조건(결과 보기 전에 정함, 2026-11-15):
  ball         원래 공
  gap_at_kick  실제 페널티킥마다 차기 10초 전 ~ 5초 뒤 공을 지우고, 나머지는 simulate.ball_dropout(누락 비율 25%,
               평균 6초 = config의 sim_ball_* 범위 위쪽 끝)으로 지운다. 대회에서 본 상황(차는 순간 공이 안 보임)을 흉내 낸다.
  no_ball      공 전체를 지운다(가장 나쁜 경우, 가짜 감지가 얼마나 늘어나는지 본다).
방식: setup_then_kick_v2(공 필요), setup_then_kick_v3(공이 없으면 배치가 깨진 순간), run_up_v2(공 안 씀).
정답과 데이터는 eval_penalty_combo와 같다. 5초 안이면 맞은 것으로 센다.
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
import simulate  # noqa: E402
import eval_penalty_combo as pk  # noqa: E402
import eval_penalty_detect as epd  # noqa: E402

FPS = CONFIG["fps"]
METHODS = {"setup_then_kick_v2": pk.setup_then_kick_v2, "setup_then_kick_v3": pk.setup_then_kick_v3,
           "run_up_v2": pk.run_up_v2}


def with_ball(match, ball):
    return fx.Match(players=match.players, team=match.team, ball=ball, period=match.period, pitch=match.pitch)


def ball_variants(match, kicks, seed):
    ball = np.asarray(match.ball, dtype=np.float64)
    rng = np.random.default_rng(seed)
    gap = simulate.ball_dropout(ball, rng, CONFIG["sim_ball_missing_share"][1], CONFIG["sim_ball_gap_mean_s"][1])
    for k in kicks:
        gap[max(k - 10 * FPS, 0):k + 5 * FPS] = np.nan
    return {"ball": ball, "gap_at_kick": gap, "no_ball": np.full_like(ball, np.nan)}


def main():
    pd.set_option("display.width", 220)
    rows, detections = [], []
    seed = CONFIG["sim_seed"]
    for gen in (epd.dfl_sets, epd.skillcorner_sets, epd.metrica_sets):
        for name, match, pens in gen():
            kicks = [stop for _, stop in pens]
            seed += 1
            for cond, ball in ball_variants(match, kicks, seed).items():
                m = with_ball(match, ball)
                for method, fn in METHODS.items():
                    found = fn(m)
                    good = [k for k in found if any(abs(k - t) <= 5 * FPS for t in kicks)]
                    rows.append({"ball": cond, "method": method, "set": name.split(":")[0], "true": len(kicks),
                                 "found": sum(1 for t in kicks if any(abs(k - t) <= 5 * FPS for k in found)),
                                 "false": len(found) - len(good)})
                    for k in good:
                        t = min(kicks, key=lambda t: abs(k - t))
                        detections.append({"ball": cond, "method": method, "data": name, "error_s": (k - t) / FPS})
            print("done", name, flush=True)
    t = pd.DataFrame(rows)
    print("\n== found / false by ball condition ==")
    print(t.groupby(["ball", "method", "set"])[["true", "found", "false"]].sum().to_string())
    print("\n== totals (sim and detected-only variants excluded) ==")
    main_sets = t[t["set"].isin(["dfl", "metrica", "skillcorner_all"])]
    print(main_sets.groupby(["ball", "method"])[["true", "found", "false"]].sum().to_string())
    d = pd.DataFrame(detections)
    if len(d):
        print("\n== kick time error of correct detections (s) ==")
        print(d.groupby(["ball", "method"])["error_s"].describe()[["count", "min", "50%", "max"]].round(2).to_string())


if __name__ == '__main__':
    main()
