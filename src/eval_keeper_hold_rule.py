"""실험: 공이 골키퍼 옆에서 사라지면 다시 보일 때까지 인플레이로 둔다(keeper_hold.keep_holds_alive).

    python src/eval_keeper_hold_rule.py

지금 파이프라인(eval_kickoff_fill.pipeline)과 인플레이 구간 F1을 비교한다. 조건:
  dfl_hidden   DFL 7경기, 골키퍼 쥠 구간(eval_keeper_hold_size의 정답)에서만 공을 지움. 대회처럼 골키퍼가 잡으면
               공이 안 보이는 상황을 흉내 낸다(DFL은 공이 늘 보여 이 신호가 없다).
  dfl          DFL 원본(공이 보임). 규칙이 잘못 걸리는지 본다.
  metrica      Metrica 3경기(데드볼 동안 공을 지운다. 골킥 전 공이 사라지는 것을 쥠으로 오인하면 손해).
  skillcorner  SkillCorner 20경기(중계 화면에서 공이 자주 사라진다).
채택 기준(결과 보기 전에 정함, 2026-11-15): dfl_hidden 평균이 오르고, 다른 세 조건은 데이터별 평균이 떨어지지 않으며
나빠진 경기가 좋아진 경기보다 적을 것.
v1(다시 나타난 자리 검사 없음): dfl_hidden +0.0031(5/2), dfl·metrica 0, SkillCorner −0.0038(6/13, 바꾼 시간의 70%가 실제 데드)
→ 불통과. v2: 같은 큰 박스 안, 골 에어리어·페널티 마크 밖에서 다시 나타나야 함(같은 기준으로 다시 판정).
v2 결과: dfl_hidden −0.0001(3/3), SkillCorner +0.0011(6/6), 바꾼 프레임 대부분 실제 인플레이 → 불통과.
v3: + 사라지기 직전 1초가 모두 인플레이로 판정돼 있어야 함(같은 기준으로 다시 판정).
v3 결과: dfl_hidden +0.0008(2/1), SkillCorner +0.0017(6/3), dfl·metrica 0 → 통과, 채택(final_dead_balls 2d).
SkillCorner 10건 중 4건은 바꾼 프레임이 실제 데드였다(효과는 작고 오인도 남아 있다).
v4(대회 진단: 골키퍼 옆 탈락 대부분이 in_play·2 m 조건, 던짐은 박스 밖에서 다시 보임): keeper_nearest·in_play_point·
   zone_any 세 완화. v3·v4와 하나씩 뺀 셋을 한 번에 같은 기준으로 판정(결과 보기 전에 정함).
   결과(dfl_hidden / SkillCorner, 좋아짐/나빠짐): v3 +0.0008 / +0.0017(6/3) 통과, v4 +0.0026 / +0.0013(7/9) 불통과,
   v4_no_nearest +0.0026 / +0.0031(8/6) 통과, v4_no_in_play_point +0.0030 / +0.0028(8/7) 통과,
   v4_no_zone_any +0.0005 / +0.0018(8/4) 통과. DFL 원본·Metrica는 모두 0.
   -> v4_no_nearest 채택(keeper_hold.V4_ADOPTED): 통과한 셋 중 합이 크고 오인이 적다.
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
import keeper_hold as kh  # noqa: E402
from eval_goal_kickoff import sources  # noqa: E402
from eval_kickoff_fill import pipeline  # noqa: E402
from eval_keeper_hold_size import keeper_hold_spans  # noqa: E402

FPS = CONFIG["fps"]


def run(cond, name, match, score, label):
    graded = np.asarray(match.period) > 0
    truth = pd.DataFrame({"period": match.period, "dead": label.astype(np.int8)})
    sig = fd.ball_signals(match)
    cr = fd.line_crossings(sig)
    cr = cr[(cr > 0) & (match.period[cr] == match.period[cr - 1])]
    base = pipeline(match, score)
    rb = cs.score_one((base & graded).astype(np.int8), truth)["alive_spell_f1"]
    out = []
    for variant, setting in VARIANTS.items():
        new, spans = kh.keep_holds_alive(base, match, sig, cr, setting)
        changed = base & ~new
        rn = cs.score_one((new & graded).astype(np.int8), truth)["alive_spell_f1"]
        out.append({"variant": variant, "cond": cond, "match": name.split(":")[1], "vanishes": len(spans),
                    "to_alive_s": changed.sum() / FPS,
                    "to_alive_truly_alive": float((~label[changed]).mean()) if changed.any() else np.nan,
                    "base": rb, "rule": rn})
    return out


# v4: 세 완화를 모두 켠 것과 하나씩 끈 것을 한 번에 같은 기준으로 판정한다(결과 보기 전에 정함, 2026-11-15).
VARIANTS = {"adopted": dict(kh.V4_ADOPTED),
            "v9_pitch": dict(kh.V4_ADOPTED, pitch_check=True),
            "v9_traj": dict(kh.V4_ADOPTED, trajectory=True),
            "v9_both": dict(kh.V4_ADOPTED, pitch_check=True, trajectory=True)}
# v9(2026-11-16): adopted = 지금 파이프라인(v8까지). pitch = 경기장 밖에서 다시 보인 공은 재개 자리로 봄(빈틈 수정),
#   traj = 사라진 위치가 골키퍼 2 m 밖이어도, 마지막 속도로 1초 늘인 경로가 골키퍼 2 m 안을 지나면 인정.
#   조건 dfl_flight 추가: 쥠 시작 1초 전(날아오는 구간)부터 끝까지 공 삭제(대회처럼 날아오는 도중 사라짐).
#   판정 기준(결과 보기 전에 정함): adopted 대비 dfl·dfl_hidden·dfl_flight가 떨어지지 않고 셋 중 하나는 오르며,
#   metrica·skillcorner는 평균이 떨어지지 않고 나빠진 경기가 좋아진 경기보다 적을 것.
# v8(2026-11-16, 효과가 있던 '경기 중 강한 증거' 방향을 다듬음): a = 경기 중 확인을 공이 골키퍼에게 도착한 순간 1초 전
#   모델 판정으로(사라지기 1초 전도 계속 인정), b = 인정된 쥠은 도착 순간부터 인플레이, c = 바꾼 구간 바로 뒤 5초 안에 끝나는
#   데드도 인플레이(구간 쪼개짐 방지). 판정 기준은 v7과 같다(결과 보기 전에 정함).
# v7 결과: v7_arrival dfl_hidden −0.0012(0/2), v7_catch dfl −0.0076(0/6)·실제 인플레이 53% 등 모두 불통과.
# v7 비교(DFL 진단 src/eval_keeper_hold_misses.py: 공 삭제 시 놓친 쥠 13/18이 '모델이 먼저 데드',
#   공이 보일 때 14/16이 '멈춰 있음'): adopted = 지금 파이프라인(v4 + carry + continue_hold + mark_with_keeper),
#   v7_arrival = 사라지기 직전 1초 안에 빠르게 날아온 공이면 경기 중으로 봄, v7_arrival_goal_area = + 그때는 골 에어리어에서
#   다시 나타나도 인정, v7_catch = 공이 보이는 채로 날아온 공을 골키퍼가 받은 순간을 쥠 시작점으로(이어 붙이기가 늘림),
#   v7_all = 모두. 판정 기준(결과 보기 전에 정함, 2026-11-16): adopted 대비 dfl·dfl_hidden이 떨어지지 않고 둘 중 하나는
#   오르며, metrica·skillcorner는 평균이 떨어지지 않고 나빠진 경기가 좋아진 경기보다 적을 것.
# v6 비교(이름은 아래 비교 루프 재사용을 위한 자리 이름): adopted = 지금 파이프라인(v4 + carry + continue_hold),
#   adopted_plus_continue = + near_at_reappear(다시 나타날 때 골키퍼 2 m 안이어도 인정),
#   adopted_plus_visible_hold = + mark_with_keeper(마크 근처에서 다시 나타나도 골키퍼 2 m 안이면 인정),
#   adopted_plus_carry = 둘 다. 판정 기준(결과 보기 전에 정함, 2026-11-15): adopted 대비 dfl·dfl_hidden이 떨어지지 않고
#   둘 중 하나는 오르며, metrica·skillcorner는 평균이 떨어지지 않고 나빠진 경기가 좋아진 경기보다 적을 것.
#   (visible_hold 자리의 '추가 조건'은 이 비교에서 쓰지 않는다.)
# continue_hold(인정된 쥠 뒤 공이 골키퍼 옆에 머무는 동안 이어 붙이기) 판정 기준(결과 보기 전에 정함, 2026-11-15):
# 지금 파이프라인(adopted = v4 + carry) 대비, dfl과 dfl_hidden이 떨어지지 않고 둘 중 하나는 오르며,
# metrica·skillcorner는 평균이 떨어지지 않고 나빠진 경기가 좋아진 경기보다 적을 것.
# 이전 결과: visible_hold(+상대 검사)는 dfl −0.0028(1/6)로 불통과, 파이프라인에서 끔.
# continue_hold 결과: dfl +0.0018(2/0), dfl_hidden 0, metrica 0, skillcorner +0.0011(3/0) → 통과, 채택(V4_ADOPTED).
# 앞 결과: visible_hold(상대 검사 없음)는 dfl −0.0026(2/5), 바꾼 시간 중 실제 인플레이 42%로 불통과.
# 이번 visible_hold는 몰고 나가기(carry) 위에 더하고, 공이 골키퍼를 떠나는 순간 상대가 박스 안에 있어야 한다(opponent_check).
# visible_hold(속도 조건 없이 골키퍼 옆 공 + 공이 붙은 순간 1초 전 인플레이) 판정 기준(결과 보기 전에 정함, 2026-11-15):
# carry와 같은 기준을 adopted 대비로 적용하고, carry보다 dfl 평균이 낮지 않을 것.
# carry(공이 보이는 채로 골키퍼가 굴려 몰고 나감) 판정 기준(결과 보기 전에 정함, 2026-11-15): dfl(공 보임)과 dfl_hidden이
# 떨어지지 않고 둘 중 하나는 오르며, metrica·skillcorner는 평균이 떨어지지 않고 나빠진 경기가 좋아진 경기보다 적을 것.


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
            ball2 = np.asarray(match.ball, dtype=np.float64).copy()
            for a, b in keeper_hold_spans(match, label):
                ball2[max(a - FPS, 0):b] = np.nan
            flight = fx.Match(players=match.players, team=match.team, ball=ball2, period=match.period, pitch=match.pitch)
            rows += run("dfl_flight", name, flight, score, label)
        else:
            rows += run(data, name, match, score, label)
        print("done", name, flush=True)
    t = pd.DataFrame(rows)
    t["diff"] = t["rule"] - t["base"]
    print("\n== by variant and condition ==")
    g = t.groupby(["variant", "cond"]).agg(
        base=("base", "mean"), rule=("rule", "mean"), diff=("diff", "mean"),
        better=("diff", lambda d: int((d > 1e-9).sum())), worse=("diff", lambda d: int((d < -1e-9).sum())),
        vanishes=("vanishes", "sum"), to_alive_s=("to_alive_s", "sum"), truly_alive=("to_alive_truly_alive", "mean"))
    print(g.round(4).to_string())
    print()
    for variant, tv in t.groupby("variant"):
        gv = g.loc[variant]
        others = tv[~tv["cond"].isin(["dfl_hidden", "dfl_flight"])]
        ok = (gv.loc["dfl_hidden", "diff"] > 0 and (others.groupby("cond")["diff"].mean() >= -1e-9).all()
              and int((others["diff"] < -1e-9).sum()) < int((others["diff"] > 1e-9).sum()))
        print("{} vs pipeline without the rule: {}".format(variant, "PASS" if ok else "not passed"))
    pv = t.pivot_table(index=["cond", "match"], columns="variant", values="rule")
    means = {}
    for variant in [v for v in VARIANTS if v != "adopted"]:
        if variant not in pv or "adopted" not in pv:
            continue
        d = pv[variant] - pv["adopted"]
        print("\n== {} vs adopted ==".format(variant))
        s = d.groupby("cond").agg(mean="mean", better=lambda x: int((x > 1e-9).sum()), worse=lambda x: int((x < -1e-9).sum()))
        print(s.round(4).to_string())
        means[variant] = s.loc["dfl", "mean"]
        dfl_conds = [c for c in ("dfl", "dfl_hidden", "dfl_flight") if c in s.index]
        dfl_ok = (s.loc[dfl_conds, "mean"] >= -1e-9).all() and s.loc[dfl_conds, "mean"].max() > 0
        rest = d.loc[["metrica", "skillcorner"]]
        rest_ok = (s.loc[["metrica", "skillcorner"], "mean"] >= -1e-9).all() and \
            int((rest < -1e-9).sum()) < int((rest > 1e-9).sum())
        extra = True
        if False and variant == "adopted_plus_visible_hold" and "adopted_plus_carry" in means:
            extra = means[variant] >= means["adopted_plus_carry"] - 1e-9
        print("{}: {}".format(variant, "PASS" if dfl_ok and rest_ok and extra else "not passed"))


if __name__ == '__main__':
    main()
