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
# v4(2026-11-15, 대회 진단에서 골키퍼 옆 탈락 대부분이 in_play와 2 m 조건): 결과 보기 전에 정한 세 가지 완화.
#   keeper_nearest  골키퍼 2 m 대신 '마지막 공 위치에 가장 가까운 선수가 그 골키퍼'(날아오던 공이 먼저 사라지는 경우)
#   in_play_point   직전 1초 '모두' 대신 '1초 전 시점'에 인플레이(잡기 직전 모델이 먼저 데드로 바뀌는 경우)
#   zone_any        같은 박스 대신 '재개 자리가 아닌 곳'(골키퍼가 던져 박스 밖에서 다시 보이는 경우)
V3 = {"keeper_nearest": False, "in_play_point": False, "zone_any": False}
V4 = {"keeper_nearest": True, "in_play_point": True, "zone_any": True}
# 채택: v4에서 keeper_nearest만 뺀 것(골키퍼 2 m 유지 + 1초 전 시점 인플레이 + 재개 자리 아닌 곳).
# src/eval_keeper_hold_rule.py: v4 전체는 SkillCorner 7/9로 불통과. 통과한 셋 중 이것이 DFL 쥠 공 삭제 +0.0026,
# SkillCorner +0.0031(8/6)로 합이 가장 크고, 바꾼 시간 중 실제 인플레이 비율도 가장 높다(57%, 다른 둘 46%·62%이나
# 그 하나는 DFL +0.0005로 효과가 작다). 골키퍼 2 m를 '가장 가까운 선수'로 바꾸면 경기 중 오인이 늘었다.
V4_ADOPTED = dict(V4, keeper_nearest=False)
# + carry(keeper_carries, 공이 보이는 채로 골키퍼가 굴려 몰고 나감): 채택된 규칙 위에 더했을 때 DFL 원본 +0.0016(1/0),
# Metrica +0.0014(1/0), SkillCorner +0.0004(1/0), DFL 쥠 공 삭제 0, 나빠진 경기 0. 바꾼 프레임은 DFL·Metrica에서 모두 실제 인플레이.
V4_ADOPTED["carry"] = True
# + continue_hold(continue_holds, 인정된 쥠 뒤 공이 골키퍼 2 m 안에 멈춰 있는 동안 이어 붙임): carry까지 켠 상태 대비
# DFL 원본 +0.0018(2/0), SkillCorner +0.0011(3/0), DFL 쥠 공 삭제·Metrica 0, 나빠진 경기 0. 바꾼 프레임은 DFL에서 모두 실제 인플레이.
V4_ADOPTED["continue_hold"] = True
# + mark_with_keeper(v6 B, 마크 근처에서 다시 나타나도 골키퍼 2 m 안이면 페널티킥이 아니다, 14조): 위 상태 대비
# DFL 쥠 공 삭제 +0.0022(2/0), SkillCorner +0.0002(1/0), 나빠진 경기 0. v6 A(near_at_reappear)는 Metrica −0.0007로 불통과.
V4_ADOPTED["mark_with_keeper"] = True
# + v8(2026-11-16): arrival_model(공이 골키퍼에게 도착한 순간 1초 전 인플레이도 인정), fill_before(인정된 쥠은 도착 순간부터
# 인플레이), merge_short_dead(바꾼 구간 바로 뒤 5초 안에 끝나는 데드도 인플레이). 위 상태 대비 DFL 원본 −0.0008(0/1),
# DFL 쥠 공 삭제 +0.0021(3/1), Metrica 0, SkillCorner +0.0020(10/3). 미리 정한 기준(DFL 하락 없음)은 불통과였으나
# 하락은 J03WR9 한 경기뿐이고, v8이 바꾼 프레임은 정답상 100% 인플레이였다. 바로 옆 프리킥 데드볼(프레임 87,417~87,852)을
# 파이프라인이 놓쳐서, 원래 있던 잘못된 3.8초 데드가 지워지자 인플레이 구간이 프리킥을 넘어 이어졌다
# (src/eval_keeper_v8_regressions.py). 사용자 결정으로 채택. DFL 쥠 정답 기준 공 삭제 조건에서 바로잡은 쥠 11 → 13/29.
V4_ADOPTED.update(arrival_model=True, fill_before=True, merge_short_dead=True)
# + v9(2026-11-16): trajectory(사라진 위치가 골키퍼 2 m 밖이어도 마지막 속도로 1초 늘인 경로가 골키퍼 2 m 안을 지나면 인정),
# pitch_check(경기장 밖에서 다시 보인 공은 재개 자리). v8 대비 DFL 날아오는 구간부터 공 삭제 +0.0045(4/0, 바꾼 시간 중
# 실제 인플레이 99.98%), DFL 원본·쥠 공 삭제·Metrica 0, SkillCorner −0.0004(3/3). 이때부터 기준을 'DFL 주 기준, Metrica·
# SkillCorner는 뚜렷한 하락이 없는지 보는 보조 기준'으로 둔다(SkillCorner는 10 fps를 늘려 쓰고 정답 정의가 달라 세밀한
# 판단에 덜 믿을 만하다). 이 기준은 SkillCorner 결과를 본 뒤 사용자 결정으로 정했다. pitch_check는 공개 데이터 효과 0.
V4_ADOPTED.update(trajectory=True, pitch_check=True)
SETTING = dict(V4_ADOPTED)
OK_ZONES_ANY = ("same_box", "outside_box:other")
CARRY_MIN_S = 1.0


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


def candidates(match, sig, crossings, state=None, setting=None):
    """박스 안에서 공이 사라진 모든 경우와 조건별 통과 여부(진단용, vanishes가 이것을 쓴다).
    키: v(사라짐), r(다시 보임 또는 상한), in_play, keeper_near, no_line_out, reappeared, zone, accepted.
    zone: same_box(통과) | goal_area | penalty_mark | outside_box:touchline | outside_box:corner | outside_box:center |
    outside_box:other | none(안 나타남)."""
    setting = SETTING if setting is None else setting
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
    for v in np.flatnonzero(seen[:-1] & ~seen[1:]) + 1:
        last = v - 1
        if period[last] <= 0 or period[v] != period[last]:
            continue
        bx, by = ball[last]
        if abs(by - width / 2) > BOX_HALF_M:
            continue
        side = 0 if bx <= BOX_DEPTH_M else 1 if bx >= length - BOX_DEPTH_M else None
        if side is None:
            continue
        rec = {"v": int(v), "side": side}
        # v3(사용자 제안 2026-11-15): 골키퍼는 인플레이 중에만 공을 잡는다. 사라지기 직전 1초가 모두 인플레이로
        # 판정돼 있을 때만 쓴다(데드볼 중 공을 줍거나 들고 가는 것은 쥠이 아니다).
        if setting["in_play_point"]:
            rec["in_play"] = state is None or (v >= FPS and not bool(state[v - FPS]))
        else:
            rec["in_play"] = state is None or (v >= FPS and not np.asarray(state[v - FPS:v]).any())
        # v7-1(2026-11-15, DFL 진단: 놓친 쥠 13/18이 '모델이 먼저 데드'): 공이 사라지기 직전 1초 안에 경기장 안에서 빠르게
        # 움직였으면(슛·크로스·백패스가 날아옴) 모델 판정과 상관없이 경기 중으로 본다. 데드볼 중에는 공이 빠르게 날지 않는다.
        rec["fast_arrival"] = bool((sig["inside"] & sig["moving"])[max(v - FPS, 0):v].any())
        if setting.get("arrival_in_play", False) and rec["fast_arrival"]:
            rec["in_play"] = True
        lo = max(last - window, 0)
        rec["no_line_out"] = not (cross[last + 1] - cross[lo] > 0 or outside[last + 1] - outside[lo] > 0)
        defenders = [k for k, s_ in side_of[int(period[last])].items() if s_ == side]
        cols = np.flatnonzero(team == defenders[0])
        x, y = p[last, cols, 0], p[last, cols, 1]
        if np.isfinite(x).any():
            depth = x if side == 0 else length - x
            k = int(np.nanargmin(depth))
            if setting["keeper_nearest"]:
                d_all = np.hypot(p[last, :, 0] - bx, p[last, :, 1] - by)
                rec["keeper_near"] = bool(np.isfinite(d_all).any() and int(np.nanargmin(d_all)) == int(cols[k]))
            else:
                rec["keeper_near"] = bool(np.hypot(x[k] - bx, y[k] - by) <= VANISH_R_M)
            if setting.get("trajectory", False) and not rec["keeper_near"]:
                # v9(2026-11-16, 대회 디버깅: 못 잡은 쥠 6장면 모두 공이 골키퍼에 닿기 전 날아오는 도중 사라졌다): 마지막으로
                # 보인 위치와 속도(직전 0.2초)로 1초 동안 경로를 늘였을 때, 그 경로가 골키퍼 VANISH_R_M 안을 지나면 골키퍼에게
                # 가던 공으로 본다. 반경을 넓히지 않고 방향을 본다. 새 상수 없음(1초·2 m는 기존 값, 0.2초는 속도 계산 창).
                back = last - FPS // 5
                if back >= 0 and seen[back]:
                    vel = (ball[last] - ball[back]) / (last - back)
                    t = np.arange(1, FPS + 1)
                    path = ball[last][None, :] + t[:, None] * vel[None, :]
                    frames = np.minimum(last + t, n - 1)
                    kx_t, ky_t = p[frames, cols[k], 0], p[frames, cols[k], 1]
                    kx_t = np.where(np.isfinite(kx_t), kx_t, x[k])
                    ky_t = np.where(np.isfinite(ky_t), ky_t, y[k])
                    rec["keeper_near"] = bool((np.hypot(path[:, 0] - kx_t, path[:, 1] - ky_t) <= VANISH_R_M).any())
        else:
            rec["keeper_near"] = False
        # v8-a(2026-11-16): 공이 골키퍼에게 '도착한' 순간을 찾는다. 사라지기 전 공이 보이며 골키퍼 VANISH_R_M 안에 계속
        # 있던 구간을 거슬러 올라간 첫 프레임. 그 1초 전이 인플레이면 경기 중에 받은 공이다(휘슬 뒤 차 준 공은 도착 때
        # 이미 선수들이 멈춰 모델이 데드로 본다).
        a0 = v
        while a0 - 1 >= 0 and seen[a0 - 1] and period[a0 - 1] == period[v]:
            xa, ya = p[a0 - 1, cols, 0], p[a0 - 1, cols, 1]
            if not np.isfinite(xa).any():
                break
            ka = int(np.nanargmin(xa if side == 0 else length - xa))
            if not np.hypot(xa[ka] - ball[a0 - 1, 0], ya[ka] - ball[a0 - 1, 1]) <= VANISH_R_M:
                break
            a0 -= 1
        rec["arrival"] = int(a0)
        if setting.get("arrival_model", False) and state is not None and a0 >= FPS and not bool(state[a0 - FPS]):
            rec["in_play"] = True
        r = v
        while r < min(v + cap, n) and not seen[r] and period[r] == period[v]:
            r += 1
        rec["r"] = int(r)
        rec["reappeared"] = bool(r < n and seen[r] and period[r] == period[v])
        rec["zone"] = reappear_zone(ball[r], match.pitch, side, setting.get("pitch_check", False)) if rec["reappeared"] else "none"
        # v6(2026-11-15, 대회 진단: 날아오던 공이 골키퍼에 닿기 전에 사라지고, 골키퍼는 차기 전 마크 근처까지 걸어 나온다):
        # 다시 나타난 순간 공이 골키퍼 VANISH_R_M 안이면 골키퍼가 들고 있던 것이다.
        rec["keeper_near_at_reappear"] = False
        if rec["reappeared"]:
            xr, yr = p[r, cols, 0], p[r, cols, 1]
            if np.isfinite(xr).any():
                kr = int(np.nanargmin(xr if side == 0 else length - xr))
                rec["keeper_near_at_reappear"] = bool(np.hypot(xr[kr] - ball[r, 0], yr[kr] - ball[r, 1]) <= VANISH_R_M)
        if setting.get("near_at_reappear", False) and rec["keeper_near_at_reappear"]:
            rec["keeper_near"] = True
        # 다시 나타난 곳이 데드볼 재개 자리가 아니어야 한다(v2, 사용자 제안 2026-11-15): 같은 페널티 박스 안이면서
        # 골 에어리어(골킥, 16조)도 페널티 마크(페널티킥, 14조)도 아님. 안 나타나면 바꾸지 않는다.
        zone_ok = (rec["zone"] in OK_ZONES_ANY if setting["zone_any"] else rec["zone"] == "same_box") or not REAPPEAR_CHECK
        # v6: 페널티킥 때 골키퍼는 골라인 위에 있어야 한다(14조). 마크 근처에서 다시 나타난 공이 골키퍼 2 m 안이면 페널티킥이 아니다.
        if setting.get("mark_with_keeper", False) and rec["zone"] == "penalty_mark" and rec["keeper_near_at_reappear"]:
            zone_ok = True
        # v7-3: 골킥은 공이 먼저 나가야 한다(16조). 빠르게 날아온 공이 라인 아웃 없이 골키퍼 옆에서 사라졌다면
        # 골 에어리어에서 다시 나타나도 골킥이 아니다(골 앞에서 잡은 경우).
        if (setting.get("arrival_goal_area", False) and rec["zone"] == "goal_area" and rec["fast_arrival"]
                and rec["no_line_out"]):
            zone_ok = True
        rec["accepted"] = bool(rec["in_play"] and rec["no_line_out"] and rec["keeper_near"]
                               and (zone_ok if REAPPEAR_CHECK else True))
        out.append(rec)
    return out


def reappear_zone(point, pitch, side, pitch_check=False):
    length, width = pitch
    rx, ry = float(point[0]), float(point[1])
    if pitch_check:
        # v9(2026-11-16): 경기장 밖(경계 여유 boundary_margin_m 밖)에서 다시 보이면 재개 자리(골킥·코너킥·스로인·골 뒤)다.
        # 전에는 골대 뒤에서 다시 보인 공도 outside_box:other로 분류돼 쥠으로 인정될 수 있었다.
        m = F["boundary_margin_m"]
        if rx < -m or rx > length + m or ry < -m or ry > width + m:
            return "outside_pitch"
    rdepth = rx if side == 0 else length - rx
    lateral = abs(ry - width / 2)
    if -1 <= rdepth <= BOX_DEPTH_M and lateral <= BOX_HALF_M:
        if rdepth <= GOAL_AREA_DEPTH_M + RESTART_TOL_M and lateral <= GOAL_AREA_HALF_M + RESTART_TOL_M:
            return "goal_area"
        if np.hypot(rdepth - SPOT_M, lateral) <= RESTART_TOL_M:
            return "penalty_mark"
        return "same_box"
    near_end = min(rx, length - rx) <= RESTART_TOL_M + 1.0
    near_side = min(ry, width - ry) <= RESTART_TOL_M
    if near_end and near_side:
        return "outside_box:corner"
    if near_side:
        return "outside_box:touchline"
    if np.hypot(rx - length / 2, ry - width / 2) <= RESTART_TOL_M:
        return "outside_box:center"
    return "outside_box:other"


def vanishes(match, sig, crossings, state=None, setting=None):
    """[(v, r)] 공이 골키퍼 옆에서 사라진 프레임 v와 다시 보인 프레임 r(최대 HOLD_MAX_S초 뒤, 같은 하프)."""
    return [(c["v"], c["r"]) for c in candidates(match, sig, crossings, state, setting) if c["accepted"]]


def keeper_carries(match, sig, crossings, state, detail=False, setting=None):
    """공이 보이는 채로 골키퍼가 굴려 몰고 나가는 경우(2026-11-15, 사용자 관찰: 이때는 공이 보인다).
    [(s, e)] 인플레이로 바꿀 구간: 공이 보이고 자기 박스 안 골키퍼에서 VANISH_R_M 안이며 굴러가는 속도
    (line_out_moving_mps 초과)로 CARRY_MIN_S초 이상 이어진 구간 [a, b)가, 그 전 HOLD_MAX_S초 안에 라인 아웃·밖 공이 없고,
    모델 데드 구간 안에 있으면, 그 데드 구간 시작(HOLD_MAX_S초 안일 때)부터 b까지. 새 상수 없음(CARRY_MIN_S는 다른
    규칙의 1초와 같다)."""
    length, width = match.pitch
    ball = np.asarray(match.ball, dtype=np.float64)
    p = match.players.astype(np.float64)
    team = np.asarray(match.team)
    period = np.asarray(match.period)
    n = len(period)
    side_of = defended_side(match)
    with np.errstate(invalid="ignore"):
        rolling = sig["seen"] & (np.nan_to_num(sig["speed"], nan=0.0) > F["line_out_moving_mps"])
    with_keeper = np.zeros(n, dtype=bool)
    for per in np.unique(period[period > 0]):
        rows = np.flatnonzero(period == per)
        for t, side in side_of[int(per)].items():
            cols = np.flatnonzero(team == t)
            x, y = p[rows][:, cols, 0], p[rows][:, cols, 1]
            depth = x if side == 0 else length - x
            dd = np.where(np.isfinite(depth), depth, np.inf)
            k = np.argmin(dd, axis=1)
            kx, ky = x[np.arange(len(rows)), k], y[np.arange(len(rows)), k]
            kd = dd[np.arange(len(rows)), k]
            with np.errstate(invalid="ignore"):
                in_box = (kd <= BOX_DEPTH_M) & (np.abs(ky - width / 2) <= BOX_HALF_M)
                near = np.hypot(ball[rows, 0] - kx, ball[rows, 1] - ky) <= VANISH_R_M
            with_keeper[rows] |= in_box & near
    visible_hold = (SETTING if setting is None else setting).get("visible_hold", False)
    if visible_hold:
        # 공이 보이는 쥠(2026-11-15, 사용자 관찰: 내려놓고 천천히 두는 경우가 많다): 속도 조건 없이 골키퍼 2 m 안이면 되고,
        # 대신 공이 골키퍼에게 처음 붙은 순간의 1초 전이 인플레이여야 한다(휘슬 뒤 골키퍼가 차는 프리킥을 거른다).
        carry = sig["seen"] & with_keeper & (period > 0)
    else:
        carry = rolling & with_keeper & (period > 0)
    c = np.asarray(crossings, dtype=np.int64)
    cross = np.zeros(n + 1, dtype=np.int64)
    cross[c[(c >= 0) & (c < n)] + 1] = 1
    cross = np.cumsum(cross)
    outside = np.concatenate(([0], np.cumsum(sig["outside"])))
    cap = int(HOLD_MAX_S * FPS)
    out = []
    from features import runs_of
    for a, b in runs_of(carry):
        if b - a < CARRY_MIN_S * FPS or not state[a:b].any():
            continue
        lo = max(a - cap, 0)
        if cross[b] - cross[lo] > 0 or outside[b] - outside[lo] > 0 or (period[lo:b] != period[a]).any():
            continue
        if visible_hold:
            if a < FPS or state[a - FPS]:
                continue
            # 상대 박스 안 검사(2026-11-15): 골킥(16조)·자기 박스 프리킥(13조)은 차기 전 상대가 모두 박스 밖이다.
            # 공이 골키퍼를 떠나는 순간(b) 상대가 박스 안(경계 EDGE_M 안쪽)에 있으면 데드볼 재개일 수 없다.
            # DFL(src/eval_keeper_opponents_in_box.py): 쥠 32%, 골킥 4.6%, 자기 박스 프리킥 0%.
            if (SETTING if setting is None else setting).get("opponent_check", False):
                side = 0 if ball[a, 0] < length / 2 else 1
                defending = [t for t, sd in side_of[int(period[a])].items() if sd == side][0]
                if opponents_in_box(match, min(b, n - 1), side, defending) == 0:
                    continue
            out.append((int(a), int(b)))
            continue
        s = a
        while s > lo and state[s - 1]:
            s -= 1
        if s == lo and state[lo]:
            continue          # 데드가 HOLD_MAX_S보다 오래 이어졌다: 쥠이 아니다
        out.append((int(s), int(b)))
    if detail:
        return out, {"with_keeper": with_keeper, "rolling": rolling, "cross": cross, "outside": outside, "cap": cap}
    return out


EDGE_M = 1.0   # 박스 경계 흔들림 여유(페널티킥 감지와 같은 값)


def opponents_in_box(match, frame, side, defending_team):
    """frame에서 그 박스를 지키지 않는 팀 선수 중 박스 경계 EDGE_M 안쪽에 있는 수."""
    length, width = match.pitch
    p = match.players[frame].astype(np.float64)
    opp = np.asarray(match.team) != defending_team
    x, y = p[opp, 0], p[opp, 1]
    depth = x if side == 0 else length - x
    with np.errstate(invalid="ignore"):
        inside = (depth <= BOX_DEPTH_M - EDGE_M) & (depth >= -1) & (np.abs(y - width / 2) <= BOX_HALF_M - EDGE_M)
    return int(np.nansum(inside))


def carry_candidates(match, sig, crossings, state):
    """진단용: 공이 보이고 박스 안 골키퍼 2 m 안에 1초 이상 붙어 있던 구간마다 몰고 나가기 조건 통과 여부.
    키: a, b, seconds, rolling_s(그중 굴러가는 속도였던 가장 긴 연속 시간), in_dead, no_line_out, dead_started_recently,
    accepted, first_fail."""
    from features import runs_of
    accepted, d = keeper_carries(match, sig, crossings, state, detail=True)
    acc = {e for _, e in accepted}
    period = np.asarray(match.period)
    seen_near = d["with_keeper"] & sig["seen"] & (period > 0)
    out = []
    for a, b in runs_of(seen_near):
        if b - a < CARRY_MIN_S * FPS:
            continue
        roll = [e - s for s, e in runs_of(d["rolling"][a:b])]
        rec = {"a": int(a), "b": int(b), "seconds": (b - a) / FPS, "rolling_s": max(roll, default=0) / FPS,
               "in_dead": bool(np.asarray(state[a:b]).any())}
        lo = max(a - d["cap"], 0)
        rec["no_line_out"] = not (d["cross"][b] - d["cross"][lo] > 0 or d["outside"][b] - d["outside"][lo] > 0)
        s = a
        while s > lo and state[s - 1]:
            s -= 1
        rec["dead_started_recently"] = not (s == lo and bool(state[lo]))
        rec["accepted"] = any(a <= e <= b for e in acc)
        if rec["accepted"]:
            rec["first_fail"] = "-"
        elif not rec["in_dead"]:
            rec["first_fail"] = "already_in_play"
        elif rec["rolling_s"] < CARRY_MIN_S:
            rec["first_fail"] = "ball_not_rolling_1s"
        elif not rec["no_line_out"]:
            rec["first_fail"] = "line_out_before"
        elif not rec["dead_started_recently"]:
            rec["first_fail"] = "dead_longer_than_22.8s"
        else:
            rec["first_fail"] = "other"
        out.append(rec)
    return out


def keep_holds_alive(state, match, sig, crossings, setting=None):
    """공이 골키퍼 옆에서 사라진 구간 [v, r)를 인플레이로. 반환 (새 state, 바꾼 구간)."""
    out = np.asarray(state, dtype=bool).copy()
    st = setting or SETTING
    if st.get("fill_before", False):
        # v8-b: 인정된 쥠은 공이 골키퍼에게 도착한 순간부터(사라지기 전 공이 보이던 구간 포함) 인플레이
        spans = [(c["arrival"], c["r"]) for c in candidates(match, sig, crossings, state, setting) if c["accepted"]]
    else:
        spans = vanishes(match, sig, crossings, state, setting)
    for v, r in spans:
        out[v:r] = False
    if st.get("carry", False):
        carries = keeper_carries(match, sig, crossings, out, setting=dict(st, visible_hold=False))
        for s, e in carries:
            out[s:e] = False
        spans = spans + carries
    if st.get("visible_hold", False):
        holds = keeper_carries(match, sig, crossings, out, setting=st)
        for s, e in holds:
            out[s:e] = False
        spans = spans + holds
    if st.get("catch_start", False):
        seeds = catch_starts(match, sig, crossings)
        spans = spans + seeds
    if st.get("continue_hold", False):
        extra = continue_holds(match, sig, spans)
        for s, e in extra:
            out[s:e] = False
        spans = spans + extra
    if st.get("merge_short_dead", False):
        # v8-c: 바꾼 구간 바로 뒤의 데드가 out_start_window_s(5초) 안에 끝나면 그것도 인플레이(모델의 잘못된 데드가
        # 공이 골키퍼를 떠난 직후 몇 초 남아 구간이 쪼개지는 것을 막는다. v7-1은 이것 때문에 정확했는데도 점수가 내려갔다)
        limit = int(F["out_start_window_s"] * FPS)
        period = np.asarray(match.period)
        n = len(out)
        extra = []
        for _, e in spans:
            if e >= n or not out[e]:
                continue
            d = e
            while d < n and out[d] and period[d] == period[e]:
                d += 1
            if d < n and period[d] == period[e] and d - e <= limit:
                out[e:d] = False
                extra.append((int(e), int(d)))
        spans = spans + extra
    return out, spans


def protect_mask(match, sig, crossings):
    """벌점항 방식(2026-11-15, 사용자 제안)의 골키퍼 보호 구간. 느린 빌드업(⑤)의 보호 구간과 같은 방식으로 비터비에서
    데드 진입 비용을 올리고 데드 쪽 점수를 깎는다(slow_buildup.decode의 extra_protect). 강제로 덮어쓰지 않으므로
    모델이 데드를 강하게 보면(반칙 뒤 프리킥 등) 데드가 남는다.
    보호: (1) 공이 보이고 자기 박스 안 골키퍼 VANISH_R_M 안(멈춰 있어도 됨), (2) 공이 박스 안에서 사라졌다가 HOLD_MAX_S초
    안에 다시 보였고, 사라질 때나 다시 보일 때 골키퍼가 VANISH_R_M 안인 그 사이.
    제외: 그 전 HOLD_MAX_S초 안에 라인 아웃이나 경기장 밖 공이 있으면(골킥·코너킥 준비). 새 상수 없음."""
    n = len(match.period)
    period = np.asarray(match.period)
    _, d = keeper_carries(match, sig, crossings, np.zeros(n, dtype=bool), detail=True, setting=dict(SETTING, visible_hold=False))
    protect = d["with_keeper"] & sig["seen"] & (period > 0)
    for c in candidates(match, sig, crossings, None, dict(SETTING, near_at_reappear=False)):
        if c["reappeared"] and (c["keeper_near"] or c["keeper_near_at_reappear"]):
            protect[c["v"]:c["r"]] = True
    recent = np.zeros(n, dtype=bool)
    cap = int(HOLD_MAX_S * FPS)
    cr = np.asarray(crossings, dtype=np.int64)
    for c0 in cr[(cr >= 0) & (cr < n)]:
        recent[c0:min(c0 + cap, n)] = True
    out_frames = np.flatnonzero(sig["outside"])
    if len(out_frames):
        mark = np.zeros(n + 1, dtype=np.int64)
        np.add.at(mark, out_frames, 1)
        np.add.at(mark, np.minimum(out_frames + cap, n), -1)
        recent |= np.cumsum(mark[:-1]) > 0
    return protect & ~recent


def catch_starts(match, sig, crossings):
    """v7-2(2026-11-15, DFL 진단: 공이 보일 때 놓친 쥠 14/16이 '멈춰 있음'): 공이 보이는 채로 골키퍼가 받은 순간을
    쥠의 시작점으로 만든다. 받음 = 공이 보이고 자기 박스 안 골키퍼 VANISH_R_M 안에 들어온 첫 프레임 c이며, 직전 1초 안에
    공이 경기장 안에서 빠르게 움직였고(날아온 공), c에서는 빠르지 않다(멈춤). 그 전 out_start_window_s 안에 라인 아웃·밖 공 없음.
    반환 [(c, c)]: 길이 0인 시작점. 이어 붙이기(continue_holds)가 공이 골키퍼를 떠날 때까지 인플레이로 늘린다.
    새 상수 없음."""
    n = len(match.period)
    period = np.asarray(match.period)
    _, d = keeper_carries(match, sig, crossings, np.zeros(n, dtype=bool), detail=True, setting=dict(SETTING, visible_hold=False))
    near = d["with_keeper"] & sig["seen"] & (period > 0)
    fast = np.concatenate(([0], np.cumsum(sig["inside"] & sig["moving"])))
    window = int(F["out_start_window_s"] * FPS)
    out = []
    from features import runs_of
    for a, _ in runs_of(near):
        if a < FPS or sig["moving"][a]:
            continue
        if fast[a] - fast[a - FPS] == 0:
            continue
        lo = max(a - window, 0)
        if d["cross"][a + 1] - d["cross"][lo] > 0 or d["outside"][a + 1] - d["outside"][lo] > 0:
            continue
        out.append((int(a), int(a)))
    return out


def continue_holds(match, sig, spans):
    """이미 인정된 쥠(경로 A·B) 뒤를 이어 붙인다(2026-11-15, 사용자 관찰: 골키퍼가 내려놓은 공이 한동안 멈춰 있다).
    쥠 구간 [s, e)가 끝나는 e에 공이 그 박스 골키퍼(골라인에 가장 가까운 수비 선수) VANISH_R_M 안에 보이면, 공이
    골키퍼를 떠날 때까지(VANISH_R_M 밖, 빠르게 움직임 ball_moving_mps, 또는 안 보임) 인플레이로 둔다. 멈춰 있어도 된다.
    잡은 순간 s부터 HOLD_MAX_S초를 넘기지 않는다. 경기 중에 잡은 공을 내려놓는 것은 같은 인플레이의 연속이다.
    새 상수 없음. 반환 [(e, c)] 이어 붙인 구간."""
    length, width = match.pitch
    ball = np.asarray(match.ball, dtype=np.float64)
    p = match.players.astype(np.float64)
    team = np.asarray(match.team)
    period = np.asarray(match.period)
    n = len(period)
    side_of = defended_side(match)
    cap = int(HOLD_MAX_S * FPS)
    out = []
    for s, e in spans:
        if e >= n or period[e] <= 0 or not sig["seen"][e]:
            continue
        side = 0 if ball[e, 0] < length / 2 else 1
        defenders = [t for t, sd in side_of[int(period[e])].items() if sd == side]
        if not defenders:
            continue
        cols = np.flatnonzero(team == defenders[0])
        c = e
        limit = min(s + cap, n)
        while c < limit and period[c] == period[e] and sig["seen"][c] and not sig["moving"][c]:
            x, y = p[c, cols, 0], p[c, cols, 1]
            if not np.isfinite(x).any():
                break
            depth = x if side == 0 else length - x
            k = int(np.nanargmin(depth))
            if not np.hypot(x[k] - ball[c, 0], y[k] - ball[c, 1]) <= VANISH_R_M:
                break
            c += 1
        if c > e:
            out.append((int(e), int(c)))
    return out
