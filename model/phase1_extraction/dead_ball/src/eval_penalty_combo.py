"""실험: 박스 안 선수가 정확히 2명이고, 그 박스의 페널티 마크 근처에서 보이는 공이 빠르게 출발하면 페널티킥으로 본다.

    python src/eval_penalty_combo.py

eval_penalty_detect(배치: 유지 시간·골라인 위 골키퍼·9.15 m 조건)와 eval_penalty_spot(데드 구간 안 + 마크 근처 출발)을
합치되 조건을 줄였다. 배치는 '차는 프레임에 박스 안 2명'만 보고, 데드 구간 조건은 쓰지 않는다(모델 불필요).
감지(프레임): 공이 보이고, 마크에서 SPOT_TOL_M 안이고, 빠르게(ball_moving_mps) 움직이고, 그 박스 안 선수가 2명.
이어진 프레임은 한 번으로 세고 MERGE_S초 안의 감지는 합친다.
SPOT_TOL_M은 eval_penalty_spot과 같은 좌표 흔들림 허용치다(정답이나 대회 장면에 맞춘 값이 아니다).
정답은 eval_penalty_detect와 같다(차는 순간 = 페널티 데드볼의 끝). 5초 안이면 맞은 것으로 센다.
아래쪽 진단 표: 실제 페널티킥마다 차기 전 10초 ~ 뒤 2초 사이에 각 조건이 어땠는지.
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
import eval_penalty_detect as epd  # noqa: E402

FPS = CONFIG["fps"]
SPOT_TOL_M = 2.0
MERGE_S = 5.0
# 도움닫기 감지 허용치. 결과를 보기 전에 정했다(2026-11-14). 정답이나 대회 장면에 맞춘 값이 아니다.
EDGE_M, ARRIVE_M, BACK_M, STILL_MPS, LOOK_S, FORM_S, FORM_FRAC = 1.0, 1.5, 2.5, 1.0, 4.0, 2.0, 0.8
# v2 배치의 보이는 선수 하한. 16 → 0(조건 없음)으로 바꿨다(2026-11-15, src/eval_penalty_visible.py).
# 겹친 선수가 한 명으로 잡히면 수가 준다. 공개 30경기에서 0으로 해도 가짜 감지 0 그대로, SkillCorner(검출 선수만)
# 2006229 페널티 골(보이는 선수 13명)을 새로 찾음. 비교는 16과 0 두 값만 미리 정해 했다.
FORMATION_MIN_VISIBLE = 0
AFTER_S = 3.0   # 준비 구간이 끝난 뒤 킥을 기다리는 시간. 결과를 보기 전에 정했다(2026-11-14).


def signals(match):
    """side(0: x=0 골, 1: x=length 골)별 (마크까지 공 거리, 박스 안 선수 수)와 공 신호."""
    length, width = match.pitch
    p = match.players.astype(np.float64)
    x, y = p[..., 0], p[..., 1]
    seen = np.isfinite(x) & np.isfinite(y)
    ball = match.ball.astype(np.float64)
    sig = fd.ball_signals(match)
    per_side = []
    for side in (0, 1):
        depth = x if side == 0 else length - x
        spot_x = epd.SPOT if side == 0 else length - epd.SPOT
        with np.errstate(invalid="ignore"):
            in_box = seen & (depth <= epd.BOX_DEPTH) & (depth >= -1) & (np.abs(y - width / 2) <= epd.BOX_HALF)
            dist = np.hypot(ball[:, 0] - spot_x, ball[:, 1] - width / 2)
        per_side.append((dist, in_box.sum(axis=1)))
    return sig, per_side


def kicks(match, need_two=True):
    sig, per_side = signals(match)
    live = np.asarray(match.period) > 0
    found = []
    for dist, count in per_side:
        with np.errstate(invalid="ignore"):
            ok = sig["moving"] & (dist <= SPOT_TOL_M) & live
        if need_two:
            ok &= count == 2
        found += [a for a, _ in fx.runs_of(ok)]
    merged = []
    for k in sorted(found):
        if not merged or k - merged[-1] > MERGE_S * FPS:
            merged.append(k)
    return merged


def setup_then_kick(match):
    """배치 준비 구간(eval_penalty_detect.setups) 안이나 끝난 뒤 AFTER_S초 안에 같은 쪽 마크 근처에서 공이 빠르게
    출발하면 페널티킥. 차는 순간 선수들이 이미 박스로 들어와 있어(DFL 3번 모두) 같은 프레임 조건은 맞지 않는다."""
    sig, per_side = signals(match)
    n = len(sig["seen"])
    found = []
    for a, b, side in epd.setups(match):
        dist = per_side[side][0]
        with np.errstate(invalid="ignore"):
            ok = sig["moving"][a:min(b + int(AFTER_S * FPS), n)] & (dist[a:min(b + int(AFTER_S * FPS), n)] <= SPOT_TOL_M)
        hit = np.flatnonzero(ok)
        if len(hit):
            found.append(a + int(hit[0]))
    merged = []
    for k in sorted(found):
        if not merged or k - merged[-1] > MERGE_S * FPS:
            merged.append(k)
    return merged


def run_up(match):
    """공 없이 키커의 도움닫기만 본다: 박스 안쪽 깊이 있는 선수가 골키퍼와 키커뿐인 상태에서, 마크에서 떨어져 서 있던 선수가
    마크에 도착하는 순간을 차는 순간으로 본다. 박스 경계에 늘어선 선수들이 좌표 흔들림으로 안팎을 오가므로
    경계에서 EDGE_M 안쪽만 센다. 골라인 위 골키퍼와 키커 2명이어야 하고(1명 허용은 가짜 감지 400개 이상이라 뺐다), 직전 FORM_S초의 FORM_FRAC만 요구한다.
    서 있다 출발하는 조건(STILL_MPS, BACK_M)은 속공으로 마크를 지나가는 공격수를 거른다."""
    length, width = match.pitch
    p = match.players.astype(np.float64)
    x, y = p[..., 0], p[..., 1]
    seen = np.isfinite(x) & np.isfinite(y)
    n = len(x)
    live = (np.asarray(match.period) > 0) & (seen.sum(axis=1) >= epd.MIN_VISIBLE)
    gap = int(0.5 * FPS)
    speed = np.full(x.shape, np.nan)
    speed[gap:] = np.hypot(x[gap:] - x[:-gap], y[gap:] - y[:-gap]) / 0.5
    form_n, look_n = int(FORM_S * FPS), int(LOOK_S * FPS)
    found = []
    for side in (0, 1):
        depth = x if side == 0 else length - x
        spot_x = epd.SPOT if side == 0 else length - epd.SPOT
        with np.errstate(invalid="ignore"):
            deep = seen & (depth <= epd.BOX_DEPTH - EDGE_M) & (depth >= -1) & (np.abs(y - width / 2) <= epd.BOX_HALF - EDGE_M)
            d = np.hypot(x - spot_x, y - width / 2)
            stood = deep & (speed < STILL_MPS) & (d >= BACK_M)
            arrive = deep & (d <= ARRIVE_M)
        count = deep.sum(axis=1)
        with np.errstate(invalid="ignore"):
            keeper = (deep & (depth <= epd.GOAL_LINE_M) & (np.abs(y - width / 2) <= epd.POST_HALF)).any(axis=1)
        # 골라인 위 골키퍼 + 키커, 정확히 2명. 골키퍼 혼자 박스에 있다가 마크 위를 지나가는 경우를 거른다.
        form = np.concatenate(([0], np.cumsum(live & (count == 2) & keeper)))
        for j in range(x.shape[1]):
            stood_c = np.concatenate(([0], np.cumsum(stood[:, j])))
            for t, _ in fx.runs_of(arrive[:, j]):
                if t < look_n:
                    continue
                if stood_c[t] - stood_c[t - look_n] and form[t + 1] - form[t + 1 - form_n] >= FORM_FRAC * form_n:
                    found.append(t)
    merged = []
    for k in sorted(found):
        if not merged or k - merged[-1] > MERGE_S * FPS:
            merged.append(k)
    return merged


def formation(match):
    """[(side, ok, kicker)] 페널티킥 배치(v2, 2026-11-15). 키커와 다른 선수를 깊이가 아니라 마크까지 거리로 가른다.

    v1은 박스 깊이 하나로 셌다. 키커도 다른 선수도 박스 경계 근처에 서서(DFL J03WR9 키커는 골라인에서 16.0 m),
    경계 여유를 주면 키커가 빠지고(run_up) 여유가 없으면 흔들리는 다른 선수가 들어왔다(setups, 시뮬레이션).
    규칙 14조: 다른 선수는 마크에서 9.15 m 밖. DFL 키커 3명은 마크에서 4.1 / 5.3 / 6.0 m.
      close   마크에서 ARC - EDGE_M 안(키커 후보)
      deep    박스 경계에서 EDGE_M 안쪽
      keeper  deep이면서 골라인 위(GOAL_LINE_M 안, 골대 사이)
      ok      keeper가 있고, close가 정확히 1명, deep 또는 close인 선수가 정확히 2명(골키퍼 + 키커),
              채점 구간(보이는 선수 수 조건 없음: FORMATION_MIN_VISIBLE).
    새 상수 없음(9.15 m와 기존 여유 EDGE_M)."""
    length, width = match.pitch
    p = match.players.astype(np.float64)
    x, y = p[..., 0], p[..., 1]
    seen = np.isfinite(x) & np.isfinite(y)
    live = (np.asarray(match.period) > 0) & (seen.sum(axis=1) >= FORMATION_MIN_VISIBLE)
    out = []
    for side in (0, 1):
        depth = x if side == 0 else length - x
        spot_x = epd.SPOT if side == 0 else length - epd.SPOT
        with np.errstate(invalid="ignore"):
            d = np.hypot(x - spot_x, y - width / 2)
            close = seen & (d < epd.ARC - EDGE_M)
            deep = seen & (depth <= epd.BOX_DEPTH - EDGE_M) & (depth >= -1) & (np.abs(y - width / 2) <= epd.BOX_HALF - EDGE_M)
            keeper = deep & (depth <= epd.GOAL_LINE_M) & (np.abs(y - width / 2) <= epd.POST_HALF)
        ok = live & keeper.any(axis=1) & (close.sum(axis=1) == 1) & ((deep | close).sum(axis=1) == 2)
        out.append((side, ok, close, d))
    return out


def _merge(found):
    merged = []
    for k in sorted(found):
        if not merged or k - merged[-1] > MERGE_S * FPS:
            merged.append(k)
    return merged


def setup_then_kick_v2(match):
    """v2 배치가 MIN_HOLD_S초 이상 이어진 구간 안이나 끝난 뒤 AFTER_S초 안에 같은 쪽 마크 근처에서 공이 빠르게,
    골 쪽으로 출발(toward_goal)."""
    sig, per_side = signals(match)
    n = len(sig["seen"])
    toward = toward_goal(match)
    found = []
    for side, ok, _, _ in formation(match):
        dist = per_side[side][0]
        for a, b in fx.runs_of(ok):
            if b - a < epd.MIN_HOLD_S * FPS:
                continue
            e = min(b + int(AFTER_S * FPS), n)
            with np.errstate(invalid="ignore"):
                hit = np.flatnonzero(sig["moving"][a:e] & (dist[a:e] <= SPOT_TOL_M) & toward[side][a:e])
            if len(hit):
                found.append(a + int(hit[0]))
    return _merge(found)


def setup_then_kick_v3(match):
    """v2 + 공이 안 보일 때의 대체 경로(2026-11-15). 대회 페널티킥 한 건에서 차는 순간 공이 안 보였다(사용자 확인,
    장면 수치는 보지 않음). DFL은 공이 늘 보여서 v2는 공 누락을 시험받은 적이 없다.
    배치 구간 [a, b)가 MIN_HOLD_S초 이상이면:
      - 공이 보이면 v2와 같다(마크 근처, 빠르게, 골 쪽).
      - 공 출발을 못 찾았고 [b - 0.5초, b + AFTER_S초]에 공이 한 번도 안 보이면, 배치가 깨진 b를 차는 순간으로 본다.
        단 b 뒤 FORM_S초 동안 배치 프레임이 1 - FORM_FRAC 이하여야 한다(차면 선수들이 박스로 들어온다. run_up_v2와 같은 조건).
    새 상수 없음."""
    sig, per_side = signals(match)
    n = len(sig["seen"])
    toward = toward_goal(match)
    seen = np.concatenate(([0], np.cumsum(sig["seen"])))
    form_n, half_s = int(FORM_S * FPS), FPS // 2
    found = []
    for side, ok, _, _ in formation(match):
        dist = per_side[side][0]
        form = np.concatenate(([0], np.cumsum(ok)))
        for a, b in fx.runs_of(ok):
            if b - a < epd.MIN_HOLD_S * FPS:
                continue
            e = min(b + int(AFTER_S * FPS), n)
            with np.errstate(invalid="ignore"):
                hit = np.flatnonzero(sig["moving"][a:e] & (dist[a:e] <= SPOT_TOL_M) & toward[side][a:e])
            if len(hit):
                found.append(a + int(hit[0]))
                continue
            no_ball = seen[e] - seen[max(b - half_s, 0)] == 0
            broke = form[min(b + form_n, n)] - form[b] <= (1 - FORM_FRAC) * form_n
            if no_ball and broke and b < n:
                found.append(b)
    return _merge(found)


def toward_goal(match):
    """[side0, side1] 프레임별로 공이 그 골 쪽으로 움직이는지(0.5초 뒤 골라인까지 거리가 줄었는지, 둘 다 보일 때).
    규칙 14조: 페널티킥은 앞으로(골 쪽으로) 찬다. 마크 근처를 골에서 멀어지며 지나가는 골킥을 거른다."""
    length, _ = match.pitch
    bx = match.ball[:, 0].astype(np.float64)
    gap = FPS // 2
    later = np.full(len(bx), np.nan)
    later[:-gap] = bx[gap:]
    with np.errstate(invalid="ignore"):
        return [later < bx, later > bx]


def run_up_v2(match, moment="arrival"):
    """v2 배치에서 키커(close)였던 선수가 직전 LOOK_S초 안에 마크에서 BACK_M 이상 떨어져 서 있다가(STILL_MPS)
    마크 ARRIVE_M 안에 도착하는 순간. 도착 직전 FORM_S초의 FORM_FRAC이 v2 배치여야 하고, 도착 뒤 FORM_S초 안에
    배치가 깨져 그 뒤 FORM_S초 동안 다시 서지 않아야 한다(broke_after). 차고 나면 선수들이 박스로 들어오지만, 공을 놓으러 마크에
    갔다 물러나면 배치가 그대로다(DFL J03WR9, 킥 20초 전). 공은 안 씀."""
    p = match.players.astype(np.float64)
    x, y = p[..., 0], p[..., 1]
    gap = int(0.5 * FPS)
    speed = np.full(x.shape, np.nan)
    speed[gap:] = np.hypot(x[gap:] - x[:-gap], y[gap:] - y[:-gap]) / 0.5
    n = len(x)
    form_n, look_n = int(FORM_S * FPS), int(LOOK_S * FPS)
    found = []
    for side, ok, close, d in formation(match):
        form = np.concatenate(([0], np.cumsum(ok)))
        with np.errstate(invalid="ignore"):
            stood = close & (speed < STILL_MPS) & (d >= BACK_M) & ok[:, None]
            arrive = d <= ARRIVE_M
        for j in range(x.shape[1]):
            stood_c = np.concatenate(([0], np.cumsum(stood[:, j])))
            for t, _ in fx.runs_of(arrive[:, j]):
                if t < look_n:
                    continue
                if not (stood_c[t] - stood_c[t - look_n] and form[t + 1] - form[t + 1 - form_n] >= FORM_FRAC * form_n):
                    continue
                b = broke_after(ok, form, t + 1, form_n)
                if b is not None:
                    found.append(t if moment == "arrival" else b)
    return _merge(found)


def broke_after(ok, form, start, form_n):
    """start부터 form_n 안에 배치가 처음 깨지고, 깨진 순간부터 form_n 동안 배치 프레임이 1 - FORM_FRAC 이하이면
    깨진 프레임, 아니면 None.
    처음엔 도착 순간부터 form_n을 쟀는데, 키커가 마크 ARRIVE_M 안에 들어온 뒤 공을 차기까지 걸리는 시간도 배치
    프레임으로 세어 천천히 들어오는 키커를 놓쳤다(도착이 킥 0.4초 전이면 그것만으로 10프레임). 2026-11-15 수정, 새 상수 없음."""
    n = len(ok)
    gone = np.flatnonzero(~ok[start:min(start + form_n, n)])
    if not len(gone):
        return None
    b = start + int(gone[0])
    return b if form[min(b + form_n, n)] - form[b] <= (1 - FORM_FRAC) * form_n else None


def diagnose(name, match, pens):
    sig, per_side = signals(match)
    n = len(sig["seen"])
    rows = []
    for _, stop in pens:
        a, b = max(stop - 10 * FPS, 0), min(stop + 2 * FPS, n)
        side = int(np.nanmin(np.where(sig["seen"][a:b], per_side[1][0][a:b], np.inf))
                   < np.nanmin(np.where(sig["seen"][a:b], per_side[0][0][a:b], np.inf)))
        dist, count = per_side[side]
        d = np.where(sig["seen"][a:b], dist[a:b], np.nan)
        near = np.nan_to_num(d, nan=np.inf) <= SPOT_TOL_M
        rows.append({"data": name, "kick": stop, "ball_seen_frac": sig["seen"][a:b].mean(),
                     "min_ball_to_spot_m": np.nanmin(d) if np.isfinite(d).any() else np.nan,
                     "near_spot_frames": int(near.sum()),
                     "near_and_moving": int((near & sig["moving"][a:b]).sum()),
                     "box_count_min": int(count[a:b].min()), "box_count_median": float(np.median(count[a:b])),
                     "box2_frac": float((count[a:b] == 2).mean()),
                     "near_moving_box2": int((near & sig["moving"][a:b] & (count[a:b] == 2)).sum())})
    return rows


METHODS = {"setup_then_kick": setup_then_kick, "run_up": run_up,
           "setup_then_kick_v2": setup_then_kick_v2, "run_up_v2": run_up_v2, "setup_then_kick_v3": setup_then_kick_v3}


def main():
    pd.set_option("display.width", 250)
    summary, detections, diag = [], [], []
    for gen in (epd.dfl_sets, epd.skillcorner_sets, epd.metrica_sets):
        for name, match, pens in gen():
            true_kicks = [stop for _, stop in pens]
            for need_two, fn in METHODS.items():
                found = fn(match)
                good = [k for k in found if any(abs(k - t) <= 5 * FPS for t in true_kicks)]
                hit = sum(1 for t in true_kicks if any(abs(k - t) <= 5 * FPS for k in found))
                summary.append({"set": name.split(":")[0], "box_two": need_two, "true": len(true_kicks), "found": hit,
                                "detections": len(found), "false": len(found) - len(good)})
                if need_two.endswith("v2"):
                    detections += [{"method": need_two, "data": name, "kick": k, "true": k in good,
                                    "error_s": min([(k - t) / FPS for t in true_kicks if abs(k - t) <= 5 * FPS],
                                                   key=abs, default=np.nan)} for k in found]
            diag += diagnose(name, match, pens)
            print("done", name, flush=True)
    print("\n== by data set (method) ==")
    print(pd.DataFrame(summary).astype({"box_two": str}).groupby(["box_two", "set"])[["true", "found", "detections", "false"]].sum().to_string())
    print("\n== true penalties: conditions from 10 s before to 2 s after the kick ==")
    print(pd.DataFrame(diag).round(2).to_string(index=False))
    print("\n== detections (v2 methods) ==")
    print(pd.DataFrame(detections).to_string(index=False))


if __name__ == '__main__':
    main()
