"""페널티킥 감지(eval_penalty_combo의 v2 두 방식) 조건을 프레임마다 CSV로 쓴다. 직접 연 터미널에서 실행할 것
(대회 트래킹을 읽는다). 결과는 Claude에게 붙이지 말고, 어느 조건이 깨졌는지만 일반적인 말로 전한다.

    python src/diagnose_penalty_conditions.py                       # 채점 구간 전체
    python src/diagnose_penalty_conditions.py --start 1000 --end 2000   # 프레임 범위만
    python src/diagnose_penalty_conditions.py --start 1000 --end 2000 --kick 1900   # + 조건별 요약(좌표 없음)

출력: data/dead_balls/final_model/penalty_conditions.csv (커밋 안 함). 한 행 = (프레임, 골 쪽).
  visible / visible_ok      보이는 선수 수, MIN_VISIBLE(16) 이상인지
  keeper_on_line            박스 깊숙이(경계 1 m 안쪽) 골라인 1.5 m 안, 골대 사이에 누가 있는지
  close_count               마크에서 8.15 m 안 인원(키커 후보, 1이어야 함)
  close_slot, close_dist_m  그 중 마크에 가장 가까운 트랙 번호와 거리(트랙이 바뀌면 도움닫기 방식이 깨질 수 있음)
  inside_count              박스 깊숙이 또는 마크 8.15 m 안 인원(골키퍼 + 키커 = 2여야 함)
  formation_ok              위 조건을 모두 만족(v2 배치)
  ball_seen, ball_to_spot_m, ball_moving, ball_toward_goal   준비→킥 방식의 공 조건(마크 2 m 안, 빠르게, 골 쪽)
  ball_kick                 공 조건을 모두 만족
  run_up_v2 / setup_then_kick_v2   그 프레임이 감지된 차는 순간인지
"""

import argparse
import os
import sys

import numpy as np
import pandas as pd

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(HERE)
sys.path[:0] = [HERE, ROOT, os.path.join(ROOT, "starter")]
from config import CONFIG  # noqa: E402
import eval_penalty_combo as pk  # noqa: E402
import eval_penalty_detect as epd  # noqa: E402
import final_dead_balls as fd  # noqa: E402

FPS = CONFIG["fps"]


def conditions(match, start=0, end=None):
    """[start, end) 프레임의 조건 표(DataFrame)."""
    n = len(match.period)
    end = n if end is None else min(end, n)
    length, width = match.pitch
    p = match.players.astype(np.float64)
    x, y = p[..., 0], p[..., 1]
    seen = np.isfinite(x) & np.isfinite(y)
    visible = seen.sum(axis=1)
    sig, per_side = pk.signals(match)
    toward = pk.toward_goal(match)
    kicks = {"run_up_v2": set(pk.run_up_v2(match, "break")), "setup_then_kick_v2": set(pk.setup_then_kick_v2(match))}
    frames = np.arange(start, end)
    gap = int(0.5 * FPS)
    speed = np.full(x.shape, np.nan)
    speed[gap:] = np.hypot(x[gap:] - x[:-gap], y[gap:] - y[:-gap]) / 0.5   # run_up_v2와 같은 속도
    tables = []
    for side, ok, close, d in pk.formation(match):
        depth = x if side == 0 else length - x
        with np.errstate(invalid="ignore"):
            deep = seen & (depth <= epd.BOX_DEPTH - pk.EDGE_M) & (depth >= -1) & (np.abs(y - width / 2) <= epd.BOX_HALF - pk.EDGE_M)
            keeper = (deep & (depth <= epd.GOAL_LINE_M) & (np.abs(y - width / 2) <= epd.POST_HALF)).any(axis=1)
        dd = np.where(close, d, np.inf)
        slot = np.argmin(dd, axis=1)
        best = dd[np.arange(n), slot]
        ball_d = per_side[side][0]
        with np.errstate(invalid="ignore"):
            near = ball_d <= pk.SPOT_TOL_M
        s = slice(start, end)
        tables.append(pd.DataFrame({
            "frame": frames, "side": side, "period": np.asarray(match.period)[s],
            "visible": visible[s], "visible_ok": visible[s] >= epd.MIN_VISIBLE,
            "keeper_on_line": keeper[s], "close_count": close.sum(axis=1)[s],
            "close_slot": np.where(np.isfinite(best), slot, -1)[s],
            "close_dist_m": np.round(np.where(np.isfinite(best), best, np.nan)[s], 2),
            "close_speed_mps": np.round(np.where(np.isfinite(best), speed[np.arange(n), slot], np.nan)[s], 2),
            "inside_count": (deep | close).sum(axis=1)[s], "formation_ok": ok[s],
            "ball_seen": sig["seen"][s], "ball_to_spot_m": np.round(ball_d[s], 2), "ball_moving": sig["moving"][s],
            "ball_toward_goal": toward[side][s], "ball_kick": (near & sig["moving"] & toward[side])[s],
            "run_up_v2": [f in kicks["run_up_v2"] for f in frames],
            "setup_then_kick_v2": [f in kicks["setup_then_kick_v2"] for f in frames]}))
    return pd.concat(tables).sort_values(["frame", "side"]).reset_index(drop=True)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--start", type=int, default=0)
    ap.add_argument("--end", type=int, default=None)
    ap.add_argument("--kick", type=int, default=None, help="차는 순간(대략). 주면 조건별 요약을 출력한다")
    args = ap.parse_args()
    import competition_data
    match, _ = competition_data.load_competition_match()
    table = conditions(match, args.start, args.end)
    table = table[table["period"] > 0]
    out = os.path.join(fd.OUT_DIR, "penalty_conditions.csv")
    os.makedirs(fd.OUT_DIR, exist_ok=True)
    table.to_csv(out, index=False)
    print("wrote {} ({} rows)".format(out, len(table)))
    if args.kick is not None:
        print("\n".join(summary(table, args.kick)))


def run_up_lines(table, side, kick):
    """run_up_v2의 '서 있다가 도착' 조건 진단. 도착 = 차는 순간 앞뒤 2초 안에서 키커 거리가 ARRIVE_M 안으로 처음
    들어온 프레임(그 전부터 계속 안에 있었으면 그 구간의 시작). 서 있음 = 도착 전 LOOK_S초 안에 마크에서 BACK_M 이상,
    STILL_MPS 미만, 배치 성립. 좌표는 출력하지 않는다."""
    t = table[table["side"] == side].set_index("frame")
    inside = t["close_dist_m"] <= pk.ARRIVE_M
    near_kick = inside[(inside.index >= kick - 2 * FPS) & (inside.index <= kick + 2 * FPS)]
    if not near_kick.any():
        return ["run_up: kicker never within {} m of the mark near the kick".format(pk.ARRIVE_M)]
    arrive = int(near_kick[near_kick].index[0])
    while arrive - 1 in inside.index and inside.get(arrive - 1, False):
        arrive -= 1
    look = t[(t.index >= arrive - int(pk.LOOK_S * FPS)) & (t.index < arrive)]
    far = look["close_dist_m"] >= pk.BACK_M
    slow = look["close_speed_mps"] < pk.STILL_MPS
    stood = far & slow & look["formation_ok"]
    after_rows = t[(t.index > arrive) & (t.index <= arrive + int(pk.FORM_S * FPS))]
    broken = after_rows[~after_rows["formation_ok"]]
    if len(broken):
        f = broken.iloc[0]
        why = [name for name, bad in (("keeper_on_line", not f["keeper_on_line"]), ("close_count != 1", f["close_count"] != 1),
                                      ("inside_count != 2", f["inside_count"] != 2)) if bad]
        first_break = ["run_up first formation break vs kick: {:+.2f} s, broken by: {}".format(
            (int(broken.index[0]) - kick) / FPS, ", ".join(why) or "?")]
    else:
        first_break = ["run_up first formation break: none within {:.0f} s after arrival".format(pk.FORM_S)]
    after = after_rows["formation_ok"]
    return first_break + ["run_up arrival vs kick: {:+.2f} s".format((arrive - kick) / FPS),
            "run_up formation frames in {:.0f} s after arrival: {} (old rule needed <= {:.0%})".format(
                pk.FORM_S, pct_of(after), 1 - pk.FORM_FRAC),
            "run_up {:.0f} s before arrival: frames with kicker >= {} m from mark {}, slower than {} m/s {}, both {}".format(
                pk.LOOK_S, pk.BACK_M, pct_of(far), pk.STILL_MPS, pct_of(slow), int(stood.sum())),
            "run_up kicker speed p10 / p50 while >= {} m from mark: {}".format(
                pk.BACK_M, " / ".join("{:.2f}".format(v) for v in np.nanpercentile(look.loc[far, "close_speed_mps"], [10, 50]))
                if far.any() else "-")]


def pct_of(mask):
    return "{:.0%}".format(mask.mean()) if len(mask) else "-"


def summary(table, kick):
    """차는 순간(kick, 대략이면 됨) 전후로 어느 조건이 깨졌는지만 요약한다(좌표는 출력하지 않음).
    골 쪽은 차기 전 5초 동안 keeper_on_line이 더 자주 성립한 쪽으로 고른다."""
    before = table[(table["frame"] >= kick - 5 * FPS) & (table["frame"] < kick)]
    if before.empty:
        return ["no rows before the kick (outside the graded halves or the --start/--end range?)"]
    side = int(before.groupby("side")["keeper_on_line"].mean().idxmax())
    b = before[before["side"] == side]
    run = table[(table["side"] == side) & (table["frame"] >= kick - 2 * FPS) & (table["frame"] <= kick + FPS)]
    after = table[(table["side"] == side) & (table["frame"] > kick) & (table["frame"] <= kick + 2 * FPS)]
    slots = run.loc[run["close_slot"] >= 0, "close_slot"]
    pct = lambda s: "{:.0%}".format(s.mean())  # noqa: E731
    lines = ["== penalty condition summary (side {}, 5 s before the kick unless noted) ==".format(side),
             "formation_ok            {}".format(pct(b["formation_ok"])),
             "  keeper_on_line        {}".format(pct(b["keeper_on_line"])),
             "  close_count == 1      {}  (0: {}, 2+: {})".format(pct(b["close_count"] == 1), pct(b["close_count"] == 0),
                                                                pct(b["close_count"] >= 2)),
             "  inside_count == 2     {}  (<2: {}, 3+: {})".format(pct(b["inside_count"] == 2), pct(b["inside_count"] < 2),
                                                                 pct(b["inside_count"] >= 3)),
             "  visible >= 16         {}".format(pct(b["visible_ok"])),
             "kicker track changes (2 s before to 1 s after): {}".format(int((slots.diff().fillna(0) != 0).sum())),
             "kicker came within 1.5 m of the mark: {}".format(bool((run["close_dist_m"] <= pk.ARRIVE_M).any())),
             "formation broke within 2 s after the kick: {}".format(after["formation_ok"].mean() <= 1 - pk.FORM_FRAC
                                                                     if len(after) else "no rows"),
             "ball seen (2 s before to 1 s after): {}".format(pct(run["ball_seen"])),
             ] + run_up_lines(table, side, kick) + [
             "detected: run_up_v2 {}, setup_then_kick_v2 {}".format(bool(table["run_up_v2"].any()),
                                                                     bool(table["setup_then_kick_v2"].any()))]
    return lines


if __name__ == '__main__':
    main()
