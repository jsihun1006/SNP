"""실험: JSH 보정 ③(재개 늦음)을 우리 코드로 옮기고(slow_buildup.decode의 case3 + release_forced, 노트북과 프레임 단위로
같음을 확인), 두 가지 개선을 비교한다.

    python src/eval_restart_late.py

  base        지금 파이프라인(⑤ + ① + ⑥, 라인 아웃, 페널티킥, 골키퍼 쥠)
  nb3         + ③ 노트북 그대로(공이 빠르게 출발하면 재개, 라인 아웃 강제도 풀어 줌)
  accel       + ③, 출발 뒤 1초 안에 양 팀이 함께 속도를 올린 재개만(①의 감속 신호를 대칭으로)
  accel_zone  + accel, 라인 아웃 강제 해제는 재개 자리가 나간 라인에 맞을 때만(15·16·17조)
이유: 노트북 ③은 SkillCorner −0.0072(5/15). DFL은 데드볼 동안 공을 고정해 '공이 빠르게 출발 = 재개'가 저절로 맞지만,
데드볼 중에도 공이 보이는 데이터(SkillCorner·대회)에서는 공을 주고받는 것도 재개로 본다(추정).
판정 기준(결과 보기 전에 정함, 2026-11-16): base 대비 DFL 평균이 오르고 DFL에서 나빠진 경기가 좋아진 경기보다 적으며,
SkillCorner 평균이 0.002 넘게 떨어지지 않을 것(이 보정은 공 기반이라 DFL이 낙관적이고 SkillCorner가 실제 위험을 보인다).
Metrica는 참고(데드볼 동안 공을 지워 DFL과 같은 이유로 낙관적).
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
BASE = dict(CONFIG["slow_buildup"])
# 1차 결과: nb3 DFL −0.0001(4/3)·SkillCorner −0.0076, accel DFL +0.0003(2/3)·SkillCorner −0.0028, accel_zone 같음 → 불통과.
#   accel이 바꾼 프레임은 SkillCorner에서도 90%가 실제 인플레이였다(위치는 맞고, 새 인플레이 섬이 구간을 망가뜨린 것으로 추정).
# 2차(사용자 제안): 비용은 그대로 두고 데드볼 끝만 재개로 당긴다(snap_dead_ends, 새 구간을 만들지 않음).
VARIANTS = {"base": (BASE, False), "snap_accel": (dict(BASE, restart_snap=True, restart_accel=True), False),
            "snap_raw": (dict(BASE, restart_snap=True), False)}


def pipeline(match, score, sig, cr, setting, zone):
    graded = np.asarray(match.period) > 0
    sup = {}
    dec = sb.decode(score, match.players, match.ball, match.pitch, match.period, cr, setting=setting, team=match.team,
                    support=sup)
    st, _ = fd.force_line_outs(dec, cr, sig, score, match.period)
    if setting.get("case3", False):
        st = sb.release_forced(st, dec, sup["restart"], cr, match.period, match.ball, match.pitch, zone)
    if setting.get("restart_snap", False):
        st, _ = sb.snap_dead_ends(st, sup["restart"], match.period)
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
    for name, match, score, label in sources():
        label = np.asarray(label).astype(bool)
        truth = pd.DataFrame({"period": match.period, "dead": label.astype(np.int8)})
        sig = fd.ball_signals(match)
        cr = fd.line_crossings(sig)
        cr = cr[(cr > 0) & (match.period[cr] == match.period[cr - 1])]
        states = {}
        for variant, (setting, zone) in VARIANTS.items():
            st = pipeline(match, score, sig, cr, setting, zone)
            states[variant] = st
            r = cs.score_one(st.astype(np.int8), truth)
            changed = states["base"] & ~st if variant != "base" else np.zeros_like(st)
            rows.append({"data": name.split(":")[0], "match": name.split(":")[1], "variant": variant,
                         "f1": r["alive_spell_f1"], "spells_pred": r["pred_spells"],
                         "to_alive_s": changed.sum() / FPS,
                         "to_alive_truly_dead": float(label[changed].mean()) if changed.any() else np.nan})
        print("done", name, flush=True)
    t = pd.DataFrame(rows)
    print("\n== mean F1 ==")
    print(t.pivot_table(index="variant", columns="data", values="f1").round(4).to_string())
    print("\n== dead -> in play vs base: seconds (truly dead share) ==")
    g = t[t["variant"] != "base"].groupby(["variant", "data"]).agg(to_alive_s=("to_alive_s", "sum"),
                                                                    truly_dead=("to_alive_truly_dead", "mean"))
    print(g.round(3).unstack("data").to_string())
    pv = t.pivot_table(index=["data", "match"], columns="variant", values="f1")
    print()
    for variant in [v for v in VARIANTS if v != "base"]:
        d = pv[variant] - pv["base"]
        s = d.groupby("data").agg(mean="mean", better=lambda x: int((x > 1e-9).sum()), worse=lambda x: int((x < -1e-9).sum()))
        ok = (s.loc["dfl", "mean"] > 0 and s.loc["dfl", "worse"] < s.loc["dfl", "better"]
              and s.loc["skillcorner", "mean"] >= -0.002)
        print("{:10s} DFL {:+.4f} ({}/{})  Metrica {:+.4f}  SkillCorner {:+.4f} ({}/{})  -> {}".format(
            variant, s.loc["dfl", "mean"], s.loc["dfl", "better"], s.loc["dfl", "worse"], s.loc["metrica", "mean"],
            s.loc["skillcorner", "mean"], s.loc["skillcorner", "better"], s.loc["skillcorner", "worse"],
            "PASS" if ok else "not passed"))


if __name__ == '__main__':
    main()
