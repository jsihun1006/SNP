"""골키퍼가 공을 쥔 동안은 인플레이다(경기 규칙 12조, 6/8초는 반칙 기준일 뿐). 대회 트래킹에서는 골키퍼가 공을 잡으면
공이 안 보이고 선수들이 느려져 모델이 데드로 보기 쉽다(사용자가 영상에서 관찰한 데이터 성질).

공이 골키퍼 옆에서 사라짐(vanish): 공이 보이다가 다음 프레임부터 안 보이고, 마지막으로 보인 곳이 한쪽 페널티 박스 안이며,
그 박스를 지키는 팀에서 골라인에 가장 가까운 선수(골키퍼)가 공에서 VANISH_R_M 안에 있다.
골킥(16조)이 아님: 사라지기 전 out_start_window_s 안에 라인 아웃이 없고 공이 경기장 밖에서 보이지 않았다.
다시 나타남(v2): HOLD_MAX_S초 안에 같은 페널티 박스 안에서 다시 보이고, 그 자리가 골 에어리어(골킥)나 페널티 마크
(페널티킥)가 아님. 데드볼은 정해진 자리에서만 재개하므로 그 밖의 박스 안에서 다시 나타나면 골키퍼가 들고 있던 것이다.
그러면 사라진 순간부터 다시 보일 때까지 인플레이로 둔다.
상수(결과 보기 전에 정함, 2026-11-15): VANISH_R_M은 페널티 마크 감지와 같은 좌표 허용치 2 m,
HOLD_MAX_S는 DFL 7경기에서 잰 골키퍼 쥠 구간의 최대 길이 22.8초(src/eval_keeper_hold_size.py, 207개, 중앙값 2.6초).
검증: src/eval_keeper_hold_rule.py.
"""

import numpy as np

from config import CONFIG

FPS = CONFIG["fps"]
F = CONFIG["final"]
BOX_DEPTH_M, BOX_HALF_M = 16.5, 20.16
VANISH_R_M, HOLD_MAX_S = 2.0, 22.8
# v2: 다시 나타난 자리 검사(경기 규칙 1조 치수, 허용치는 다른 재개 자리 검사와 같은 2 m). v1(검사 없음)은
# SkillCorner에서 바꾼 시간의 70%가 실제 데드라 불통과(src/eval_keeper_hold_rule.py).
GOAL_AREA_DEPTH_M, GOAL_AREA_HALF_M, SPOT_M, RESTART_TOL_M = 5.5, 9.16, 11.0, 2.0
REAPPEAR_CHECK = True


def defended_side(match):
    """하프마다 {팀: 0(x=0 골) 또는 1(x=length 골)}. 팀 평균 x가 작은 팀이 x=0 골을 지킨다."""
    out = {}
    period = np.asarray(match.period)
    for p in np.unique(period[period > 0]):
        rows = period == p
        mean_x = [np.nanmean(match.players[rows][:, np.asarray(match.team) == k, 0]) for k in (0, 1)]
        low = int(np.argmin(mean_x))
        out[int(p)] = {low: 0, 1 - low: 1}
    return out


def vanishes(match, sig, crossings, state=None):
    """[(v, r)] 공이 골키퍼 옆에서 사라진 프레임 v와 다시 보인 프레임 r(최대 HOLD_MAX_S초 뒤, 같은 하프)."""
    length, width = match.pitch
    ball = np.asarray(match.ball, dtype=np.float64)
    p = match.players.astype(np.float64)
    team = np.asarray(match.team)
    period = np.asarray(match.period)
    seen = sig["seen"]
    n = len(seen)
    side_of = defended_side(match)
    cross = np.zeros(n + 1, dtype=np.int64)
    c = np.asarray(crossings, dtype=np.int64)
    cross[c[(c >= 0) & (c < n)] + 1] = 1
    cross = np.cumsum(cross)
    outside = np.concatenate(([0], np.cumsum(sig["outside"])))
    window = int(F["out_start_window_s"] * FPS)
    cap = int(HOLD_MAX_S * FPS)
    out = []
    starts = np.flatnonzero(seen[:-1] & ~seen[1:]) + 1
    for v in starts:
        last = v - 1
        if period[last] <= 0 or period[v] != period[last]:
            continue
        # v3(사용자 제안 2026-11-15): 골키퍼는 인플레이 중에만 공을 잡는다. 사라지기 직전 1초가 모두 인플레이로
        # 판정돼 있을 때만 쓴다(데드볼 중 공을 줍거나 들고 가는 것은 쥠이 아니다).
        if state is not None and (v < FPS or np.asarray(state[v - FPS:v]).any()):
            continue
        bx, by = ball[last]
        if abs(by - width / 2) > BOX_HALF_M:
            continue
        side = 0 if bx <= BOX_DEPTH_M else 1 if bx >= length - BOX_DEPTH_M else None
        if side is None:
            continue
        lo = max(last - window, 0)
        if cross[last + 1] - cross[lo] > 0 or outside[last + 1] - outside[lo] > 0:
            continue
        defenders = [k for k, s in side_of[int(period[last])].items() if s == side]
        cols = np.flatnonzero(team == defenders[0])
        x, y = p[last, cols, 0], p[last, cols, 1]
        if not np.isfinite(x).any():
            continue
        depth = x if side == 0 else length - x
        k = int(np.nanargmin(depth))
        if not np.hypot(x[k] - bx, y[k] - by) <= VANISH_R_M:
            continue
        r = v
        while r < min(v + cap, n) and not seen[r] and period[r] == period[v]:
            r += 1
        if not REAPPEAR_CHECK:
            out.append((int(v), int(r)))
            continue
        # 다시 나타난 곳이 데드볼 재개 자리가 아니어야 한다(v2, 사용자 제안 2026-11-15): 같은 페널티 박스 안이면서
        # 골 에어리어(골킥, 16조)도 페널티 마크(페널티킥, 14조)도 아님. 안 나타나면 바꾸지 않는다.
        if r >= n or not seen[r] or period[r] != period[v]:
            continue
        rx, ry = ball[r]
        rdepth = rx if side == 0 else length - rx
        in_box = -1 <= rdepth <= BOX_DEPTH_M and abs(ry - width / 2) <= BOX_HALF_M
        goal_area = rdepth <= GOAL_AREA_DEPTH_M + RESTART_TOL_M and abs(ry - width / 2) <= GOAL_AREA_HALF_M + RESTART_TOL_M
        on_spot = np.hypot(rdepth - SPOT_M, ry - width / 2) <= RESTART_TOL_M
        if in_box and not goal_area and not on_spot:
            out.append((int(v), int(r)))
    return out


def keep_holds_alive(state, match, sig, crossings):
    """공이 골키퍼 옆에서 사라진 구간 [v, r)를 인플레이로. 반환 (새 state, 바꾼 구간)."""
    out = np.asarray(state, dtype=bool).copy()
    spans = vanishes(match, sig, crossings, state)
    for v, r in spans:
        out[v:r] = False
    return out, spans
