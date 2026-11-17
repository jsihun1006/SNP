"""실험: 감지된 페널티킥 순간에 데드볼을 끊는다(final_dead_balls.split_at_penalty_kicks). 지금 파이프라인
(느린 빌드업 보정 + 라인 아웃 강제)과 인플레이 구간 F1을 비교한다.

    python src/eval_penalty_split.py

데이터와 파이프라인은 eval_goal_kickoff와 같다(DFL 7, Metrica 3, SkillCorner 20). 공 조건 두 가지:
원래 공, 공 전체 없음(대회처럼 차는 순간 공이 안 보일 때. 이때는 run_up_v2만 감지한다).
채택 기준(결과 보기 전에 정함, 2026-11-15): 두 공 조건 모두 나빠진 경기가 없을 것(페널티킥이 없는 경기는 바뀌지 않는다).
1차(고정 0.6초): 공 없음 통과(4/0), 원래 공 불통과(SkillCorner 1874553 −0.0035, 킥 → 골 1.5초).
2차: 공이 보이면 킥 뒤 3초 안의 첫 라인 아웃까지 인플레이.
   결과: DFL·Metrica 4경기 +0.0015~+0.0039, SkillCorner 1874553 −0.0035(라벨이 킥 6.4초 전부터 인플레이) → 사용자 결정으로 채택.
3차: run_up_v2 차는 순간을 배치 깨짐으로(eval_penalty_moment). 합계 공 있음 +0.0095 → +0.0159, 공 없음 +0.0066 → +0.0127.
   Metrica game2 공 없음 −0.0003(감지가 킥 0.24초 뒤, 0.6초 창과 겹침 감소) → 사용자 결정으로 유지.
4차(diff_vit): 고정 0.6초 대신 킥 프레임 인플레이 고정 + 데드 이탈 비용 0으로 비터비 재계산(slow_buildup.decode의
   restarts). 킥 뒤 인플레이 길이를 모델 확률이 정한다. 비교 기준(결과 보기 전에 정함): 공 없음 조건의 합계와
   나빠진 경기 수가 3차보다 나쁘지 않을 것.
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
from eval_goal_kickoff import sources  # noqa: E402

FPS = CONFIG["fps"]


def main():
    pd.set_option("display.width", 220)
    rows = []
    for name, match, score, label in sources():
        sig = fd.ball_signals(match)
        cr = fd.line_crossings(sig)
        cr = cr[(cr > 0) & (match.period[cr] == match.period[cr - 1])]
        base = sb.decode(score, match.players, match.ball, match.pitch, match.period, cr, team=match.team)
        base, _ = fd.force_line_outs(base, cr, sig, score, match.period)
        graded = np.asarray(match.period) > 0
        truth = pd.DataFrame({"period": match.period, "dead": np.asarray(label).astype(np.int8)})
        b = cs.score_one((base & graded).astype(np.int8), truth)["alive_spell_f1"]
        for cond in ("ball", "no_ball"):
            m = match if cond == "ball" else fx.Match(players=match.players, team=match.team,
                                                      ball=np.full_like(np.asarray(match.ball, dtype=np.float64), np.nan),
                                                      period=match.period, pitch=match.pitch)
            kicks = [k for k, _ in fd.penalty_kicks(m)]
            cr_m = cr if cond == "ball" else np.array([], dtype=np.int64)   # 공이 없으면 라인 아웃도 안 보인다
            new, changed = fd.split_at_penalty_kicks(base, kicks, match.period, cr_m)
            r = cs.score_one((new & graded).astype(np.int8), truth)
            # 4차: 고정 길이 대신 킥 프레임을 인플레이로 고정하고 데드 이탈 비용 0으로 비터비를 다시 푼다
            vit = sb.decode(score, match.players, m.ball, match.pitch, match.period, cr_m, restarts=kicks, team=match.team)
            vit, _ = fd.force_line_outs(vit, cr_m, fd.ball_signals(m), score, match.period)
            rv = cs.score_one((vit & graded).astype(np.int8), truth)
            alive_len = []
            for k in kicks:
                e = k
                while e < len(vit) and not vit[e]:
                    e += 1
                alive_len.append(round((e - k) / FPS, 2))
            rows.append({"data": name.split(":")[0], "match": name.split(":")[1], "ball": cond, "kicks": len(kicks),
                         "split": len(changed), "base": b, "split_f1": r["alive_spell_f1"],
                         "viterbi_f1": rv["alive_spell_f1"], "viterbi_alive_s": alive_len})
        print("done", name, flush=True)
    t = pd.DataFrame(rows)
    t["diff"] = t["split_f1"] - t["base"]
    t["diff_vit"] = t["viterbi_f1"] - t["base"]
    print("\n== matches with a penalty kick detected ==")
    print(t[t["kicks"] > 0].round(4).to_string(index=False))
    print("\n== mean by data set ==")
    print(t.groupby(["ball", "data"])[["base", "split_f1", "viterbi_f1", "diff", "diff_vit"]].mean().round(4).to_string())
    for cond, g in t.groupby("ball"):
        for col in ("diff", "diff_vit"):
            better, worse = int((g[col] > 1e-9).sum()), int((g[col] < -1e-9).sum())
            print("{} {}: better {} / worse {}, sum {:+.4f}".format(cond, col, better, worse, g[col].sum()))


if __name__ == '__main__':
    main()
