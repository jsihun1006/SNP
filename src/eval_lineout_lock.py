"""⑦ 관측 라인 아웃 시작점 고정(slow_buildup.lineout_locks)과 그 변형: 지금 파이프라인(base)과 비교.

    python src/eval_lineout_lock.py

전체 파이프라인은 eval_dfl_current.run(①⑤⑥ + 경계 당기기 + 라인 아웃 + 페널티킥 + 골키퍼, 강제 고정은 최종에도 유지).
데이터: DFL 7경기(원본, 쥠 구간 공 삭제), Metrica 3경기, SkillCorner 20경기.
변형(lock_mode):
  hard             원래 ⑦: 데드 고정 + 인플레이 고정(1차 결과 DFL −0.023, 고정 데드의 92.5%가 실제 인플레이)
  alive_only       인플레이 고정만
  soft             데드 쪽은 강제 대신 ⑥과 같은 점수(+0.45 × bonus), 인플레이 고정은 그대로
  certified        인증된 넘음(재개가 25초 안에 이어지고 모델 데드 확률이 0.4 이상 오름)에서만 두 고정
  alive_certified  인플레이 고정 전부 + 인증된 넘음에서만 데드 고정
판정 기준(결과 보기 전에 정함, 1차와 같음): DFL 원본·쥠 공 삭제 평균이 떨어지지 않고 하나는 오르며 DFL에서 나빠진
경기가 좋아진 경기보다 적을 것. 공을 쓰는 규칙이라 SkillCorner(데드 중에도 공 기록) 평균 ≥ −0.002도 요구한다.
진단: DFL에서 원래 ⑦ 사건마다 넘은 라인, 공 높이, 정답 데드까지 시간, 1초 안에 공이 다시 안으로 들어왔는지.
"""

import os
import sys

import numpy as np
import pandas as pd

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path[:0] = [HERE, os.path.join(os.path.dirname(HERE), "starter")]
from config import CONFIG  # noqa: E402
import features as fx  # noqa: E402
import compare_smoothing as cs  # noqa: E402
import slow_buildup as sb  # noqa: E402
from eval_dfl_current import run  # noqa: E402
from eval_goal_kickoff import sources  # noqa: E402
from eval_keeper_hold_size import keeper_hold_spans  # noqa: E402

BASE = dict(CONFIG["slow_buildup"])
VARIANTS = ("hard", "alive_only", "soft", "certified", "alive_certified")


def diagnose(mid, match, score, label):
    import dfl_load
    tracking, _, _ = dfl_load.load_match(mid)
    z = tracking["ball_z"].to_numpy(np.float64)
    length, width = match.pitch
    ball = np.asarray(match.ball, dtype=np.float64)
    out = []
    for h in sb.halves(np.asarray(match.period)):
        sig = sb.movement_signals(match.players[h], ball[h], match.pitch, 0.0)
        p = score[h]
        for f, end, _ in sb.lineout_events(sig["distance"], sig["reliable"], p):
            g = h.start + f
            x, y = ball[g]
            nxt = np.flatnonzero(label[g:])
            seg = sig["distance"][f:min(len(p), f + 25)]
            out.append({"line": "touch" if min(y, width - y) < min(x, length - x) else "goal", "ball_z": z[g],
                        "dead_at_f": bool(label[g]), "to_dead_s": nxt[0] / 25 if len(nxt) else np.inf,
                        "out_max_m": float(-np.nanmin(sig["distance"][f:end])),
                        "back_in_1s": bool(np.nanmax(seg) > 0) if np.isfinite(seg).any() else False})
    return out


def main():
    pd.set_option("display.width", 200)
    rows, diag = [], []
    for name, match, s, label in sources():
        label = np.asarray(label).astype(bool)
        data, mid = name.split(":")[0], name.split(":")[1]
        truth = pd.DataFrame({"period": match.period, "dead": label.astype(np.int8)})
        conds = {data: match}
        if data == "dfl":
            ball = np.asarray(match.ball, dtype=np.float64).copy()
            for a, b in keeper_hold_spans(match, label):
                ball[a:b] = np.nan
            conds["dfl_hidden"] = fx.Match(players=match.players, team=match.team, ball=ball, period=match.period,
                                           pitch=match.pitch)
        for cond, mm in conds.items():
            CONFIG["slow_buildup"] = dict(BASE)
            base = cs.score_one(run(mm, s).astype(np.int8), truth)["alive_spell_f1"]
            for mode in VARIANTS:
                CONFIG["slow_buildup"] = dict(BASE, lineout_lock=True, lock_mode=mode)
                lock = cs.score_one(run(mm, s).astype(np.int8), truth)["alive_spell_f1"]
                rows.append({"variant": mode, "cond": cond, "match": mid, "base": base, "lock": lock})
            CONFIG["slow_buildup"] = dict(BASE)
            if cond == "dfl":
                diag += diagnose(mid, mm, s, label)
        print("done", name, flush=True)
    t = pd.DataFrame(rows)
    t["diff"] = t["lock"] - t["base"]
    g = t.groupby(["variant", "cond"]).agg(base=("base", "mean"), lock=("lock", "mean"), diff=("diff", "mean"),
                                           better=("diff", lambda d: int((d > 1e-9).sum())),
                                           worse=("diff", lambda d: int((d < -1e-9).sum())))
    print("\n== variant vs base (in-play spell F1) ==")
    print(g.round(4).to_string())
    print()
    for mode in VARIANTS:
        gv = g.loc[mode]
        dfl = gv.loc[["dfl", "dfl_hidden"]]
        ok = ((dfl["diff"] >= -1e-9).all() and dfl["diff"].max() > 1e-9
              and gv.loc["dfl", "worse"] < gv.loc["dfl", "better"] and gv.loc["skillcorner", "diff"] >= -0.002)
        print("{}: {}".format(mode, "PASS" if ok else "not passed"))
    dg = pd.DataFrame(diag)
    if len(dg):
        print("\n== DFL original-⑦ events: were they real line-outs? ==")
        print("events {}, truth dead at the crossing {:.1%}, truth dead within 2 s {:.1%}, within 10 s {:.1%}".format(
            len(dg), dg["dead_at_f"].mean(), (dg["to_dead_s"] <= 2).mean(), (dg["to_dead_s"] <= 10).mean()))
        print(dg.groupby("line").agg(n=("dead_at_f", "size"), dead_at_f=("dead_at_f", "mean"),
                                     dead_within_2s=("to_dead_s", lambda x: (x <= 2).mean()),
                                     ball_z_median=("ball_z", "median"), out_max_median_m=("out_max_m", "median"),
                                     back_in_within_1s=("back_in_1s", "mean")).round(3).to_string())


if __name__ == "__main__":
    main()
