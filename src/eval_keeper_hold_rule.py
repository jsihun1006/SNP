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
    new, spans = kh.keep_holds_alive(base, match, sig, cr)
    changed = base & ~new
    rb = cs.score_one((base & graded).astype(np.int8), truth)["alive_spell_f1"]
    rn = cs.score_one((new & graded).astype(np.int8), truth)["alive_spell_f1"]
    return {"cond": cond, "match": name.split(":")[1], "vanishes": len(spans), "to_alive_s": changed.sum() / FPS,
            "to_alive_truly_alive": float((~label[changed]).mean()) if changed.any() else np.nan,
            "base": rb, "rule": rn}


def main():
    pd.set_option("display.width", 220)
    rows = []
    for name, match, score, label in sources():
        label = np.asarray(label).astype(bool)
        data = name.split(":")[0]
        if data == "dfl":
            rows.append(run("dfl", name, match, score, label))
            ball = np.asarray(match.ball, dtype=np.float64).copy()
            for a, b in keeper_hold_spans(match, label):
                ball[a:b] = np.nan
            hidden = fx.Match(players=match.players, team=match.team, ball=ball, period=match.period, pitch=match.pitch)
            rows.append(run("dfl_hidden", name, hidden, score, label))
        else:
            rows.append(run(data, name, match, score, label))
        print("done", name, flush=True)
    t = pd.DataFrame(rows)
    t["diff"] = t["rule"] - t["base"]
    print("\n== per match ==")
    print(t.round(4).to_string(index=False))
    print("\n== by condition ==")
    g = t.groupby("cond").agg(base=("base", "mean"), rule=("rule", "mean"), diff=("diff", "mean"),
                              better=("diff", lambda d: int((d > 1e-9).sum())), worse=("diff", lambda d: int((d < -1e-9).sum())),
                              vanishes=("vanishes", "sum"), to_alive_s=("to_alive_s", "sum"))
    print(g.round(4).to_string())
    others = t[t["cond"] != "dfl_hidden"]
    ok = (g.loc["dfl_hidden", "diff"] > 0 and (others.groupby("cond")["diff"].mean() >= -1e-9).all()
          and int((others["diff"] < -1e-9).sum()) < int((others["diff"] > 1e-9).sum()))
    print("\n-> {}".format("PASS" if ok else "not passed"))


if __name__ == '__main__':
    main()
