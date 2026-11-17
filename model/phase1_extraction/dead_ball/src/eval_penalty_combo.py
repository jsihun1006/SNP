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
# 추론에 쓰는 배치·킥 감지 함수와 상수는 penalty.py에 있다(여기는 공개 데이터 검증과 폐기한 v1·v3 변형).
from penalty import (SPOT_TOL_M, MERGE_S, EDGE_M, ARRIVE_M, BACK_M, STILL_MPS, LOOK_S, FORM_S, FORM_FRAC,  # noqa: E402,F401
                     AFTER_S, signals, formation, _merge, setup_then_kick_v2, toward_goal, run_up_v2, broke_after)


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
