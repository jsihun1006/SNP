"""실험: 느린 빌드업 보정(⑤)의 패스·수신 보호를 같은 팀 선수끼리의 패스로만 제한한다(slow_buildup.decode의 same_team_pass).

    python src/eval_slow_buildup_team.py

이유(2026-11-16, 대회 영상 디버깅): 오프사이드 휘슬 뒤 상대 선수가 골키퍼에게 공을 돌려주는 것을 ⑤가 패스·수신으로 보고,
이미 진행 중인 데드볼을 4.5초 끊었다(src/trace_stages.py로 ⑤ 단계에서 바뀐 것을 확인). 경기 중 빌드업 패스는 대개
같은 팀끼리이고, 휘슬 뒤 돌려주는 공은 대개 상대 팀에게 간다. 새 상수 없음.
데이터: eval_goal_kickoff.sources(DFL 7, Metrica 3, SkillCorner 20). 비교는 전체 파이프라인(라인 아웃·페널티킥·골키퍼 포함).
  ⑤가 데드 -> 인플레이로 바꾼 프레임(⑤ 없을 때 대비) 중 실제 데드 비율도 본다(⑤의 손해).
판정 기준(결과 보기 전에 정함): 데이터별 평균이 떨어지지 않고, 전체에서 나빠진 경기가 좋아진 경기보다 적을 것.
1차 same_team 결과: DFL +0.0008(1/0), Metrica 0, SkillCorner −0.0001(5/5), ⑤ 손해 비율 그대로 → 불통과.
   대회 장면(124415)도 못 고쳤다(그 4.5초는 상대에게 돌려준 패스가 아니었다).
2차 no_exit: DFL −0.0135(1/6), Metrica −0.0104(0/2), SkillCorner −0.0151(3/17) → 불통과. ⑤의 이득 대부분이 모델의
   잘못된 데드를 끊는 데서 나온다.
3차 islands(⑤가 원래 데드였던 곳에 만든, 양쪽이 데드이고 5초 안에 끝나는 인플레이 섬을 되돌림): 같은 기준.
"""

import os
import sys

import numpy as np
import pandas as pd

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(HERE)
sys.path[:0] = [HERE, os.path.join(ROOT, "starter")]
from config import CONFIG  # noqa: E402
import final_dead_balls as fd  # noqa: E402
import slow_buildup as sb  # noqa: E402
import compare_smoothing as cs  # noqa: E402
import kickoff  # noqa: E402
import keeper_hold as kh  # noqa: E402
from eval_goal_kickoff import sources  # noqa: E402

FPS = CONFIG["fps"]


def full(match, score, sig, cr, decoded):
    graded = np.asarray(match.period) > 0
    st, _ = fd.force_line_outs(decoded, cr, sig, score, match.period)
    st &= graded
    pen = [k for k, _ in fd.penalty_kicks(match) if graded[k]]
    st, _ = fd.split_at_penalty_kicks(st, pen, match.period, cr)
    st, _ = kickoff.hold_after_penalty_goals(st, pen, kickoff.kickoffs(match), match.period)
    st &= graded
    st, _ = kh.keep_holds_alive(st, match, sig, cr)
    return st & graded


def main():
    pd.set_option("display.width", 200)
    rows = []
    base_setting = dict(CONFIG["slow_buildup"])
    for name, match, score, label in sources():
        label = np.asarray(label).astype(bool)
        graded = np.asarray(match.period) > 0
        truth = pd.DataFrame({"period": match.period, "dead": label.astype(np.int8)})
        sig = fd.ball_signals(match)
        cr = fd.line_crossings(sig)
        cr = cr[(cr > 0) & (match.period[cr] == match.period[cr - 1])]
        no5 = sb.decode(score, match.players, match.ball, match.pitch, match.period, cr,
                        setting={"margin": 0.0, "strength": 0.0, "bonus": 0.0})
        for variant, setting in (("original", base_setting), ("islands", dict(base_setting, revert_short_islands=True))):
            dec = sb.decode(score, match.players, match.ball, match.pitch, match.period, cr, setting=setting, team=match.team)
            flipped = no5 & ~dec & graded
            st = full(match, score, sig, cr, dec)
            rows.append({"data": name.split(":")[0], "match": name.split(":")[1], "variant": variant,
                         "f1": cs.score_one(st.astype(np.int8), truth)["alive_spell_f1"],
                         "flip_s": flipped.sum() / FPS,
                         "flip_truly_dead": float(label[flipped].mean()) if flipped.any() else np.nan})
        print("done", name, flush=True)
    t = pd.DataFrame(rows)
    print("\n== mean by data set ==")
    print(t.groupby(["data", "variant"]).agg(f1=("f1", "mean"), flip_s=("flip_s", "sum"),
                                              flip_truly_dead=("flip_truly_dead", "mean")).round(4).to_string())
    pv = t.pivot_table(index=["data", "match"], columns="variant", values="f1")
    d = pv["islands"] - pv["original"]
    s = d.groupby("data").agg(mean="mean", better=lambda x: int((x > 1e-9).sum()), worse=lambda x: int((x < -1e-9).sum()))
    print("\n== no_exit vs original ==")
    print(s.round(4).to_string())
    better, worse = int((d > 1e-9).sum()), int((d < -1e-9).sum())
    ok = (s["mean"] >= -1e-9).all() and worse < better
    print("better {} / worse {} -> {}".format(better, worse, "PASS" if ok else "not passed"))


if __name__ == '__main__':
    main()
