"""느린 빌드업 보정(JSH ⑤): 패스·수신이나 공과 선수의 동행이 보이면 데드볼로 들어가기 어렵게 한다.

출처: JSH 브랜치 model/영상추출/모델/dead_ball/LightGBM_deadball_corrections.ipynb (커밋 b6b307d)의
movement_features·viterbi_cost·decode 중 ⑤에 필요한 부분을 그대로 옮겼다. 상수는 노트북 값이며 노트북에 따르면
DFL 학습 쪽 내부 검증 경기로 골랐다(대회 영상이나 공개 정답 5분에 맞춘 값이 아니다). 대회 설정은 노트북의
COMPETITION_RULES(margin 0.0, strength 1.0, bonus 0.85)와 같다(config.py의 slow_buildup).

검증(인플레이 구간 F1, 지금 파이프라인 대비):
  Metrica 3경기(학습에 안 씀)   0.809 -> 0.820, 2경기 개선 / 악화 0 (src/eval_jsh_corrections_metrica.py)
  SkillCorner 20경기            0.729 -> 0.748, 18경기 개선 / 2경기 악화 (src/eval_jsh_corrections_skillcorner.py)
    데드 -> 인플레이로 바꾼 프레임 중 실제 데드는 24%(데드볼 동안 진짜 공이 있는 데이터로 잰 손해)
  DFL 7경기(JSH 보고)           0.877 -> 0.894
"""

import numpy as np
from scipy.ndimage import maximum_filter1d, minimum_filter1d, uniform_filter1d
from scipy.special import logit

from config import CONFIG
from features import runs_of
import smoothing

FPS = CONFIG["fps"]


def halves(period):
    cuts = np.r_[0, np.flatnonzero(period[1:] != period[:-1]) + 1, len(period)]
    return [slice(a, b) for a, b in zip(cuts[:-1], cuts[1:])]


def shift(x, frames):
    out = np.full(x.shape, np.nan, dtype=np.float64)
    if not frames:
        return np.asarray(x, dtype=np.float64)
    if 0 < frames < len(x):
        out[frames:] = x[:-frames]
    elif -len(x) < frames < 0:
        out[:frames] = x[-frames:]
    return out


def spread(mask, radius):
    return maximum_filter1d(mask.astype(np.uint8), 2 * radius + 1, mode="constant") != 0


def extend_from(indices, n, before, after):
    counts = np.zeros(n + 1, dtype=np.int32)
    np.add.at(counts, np.maximum(0, indices - before), 1)
    np.add.at(counts, np.minimum(n, indices + after + 1), -1)
    return np.cumsum(counts[:-1]) > 0


def peak_indices(mask, strength, gap):
    peaks = np.array([a + np.argmax(strength[a:b]) for a, b in runs_of(mask)], dtype=int)
    return peaks[np.r_[True, np.diff(peaks) >= gap]] if peaks.size else peaks


def movement_signals(players, ball, pitch, margin, team=None, stop_team=None):
    """한 하프의 공 신호(노트북 movement_features 중 ⑤에 쓰는 것): reliable, pass, carry, restart, cross."""
    n = len(ball)
    observed = np.isfinite(ball).all(axis=1)
    x, y = ball.T
    length, width = pitch
    distance = np.minimum.reduce((x, length - x, y, width - y))
    delta = ball - shift(ball, 1)
    jumps = np.linalg.norm(delta, axis=1) > 45 / FPS
    continuous = minimum_filter1d(observed.astype(np.uint8), 7, mode="constant").astype(bool)
    reliable = continuous & ~spread(jumps, 4)
    velocity = (shift(ball, -3) - shift(ball, 3)) * FPS / 6
    speed = np.linalg.norm(velocity, axis=1)
    speed[~reliable] = np.nan
    dist = np.linalg.norm(players - ball[:, None, :], axis=2)
    dist = np.where(np.isfinite(dist), dist, np.inf)
    near = dist.min(axis=1)
    owner = np.argmin(dist, axis=1)
    old_owner = np.nan_to_num(shift(owner, 4), nan=-1).astype(int)
    same = owner == old_owner
    rows = np.arange(n)
    dr = players[rows, owner] - shift(players, 4)[rows, owner]
    ball_motion = ball - shift(ball, 4)
    co_motion = np.linalg.norm(dr - ball_motion, axis=1)
    # 공과 같은 선수가 함께 움직임(드리블·공을 들고 감)
    track_carry = reliable & same & (near < 1.1) & (co_motion < .40) & (speed > .35) & (speed < 7)
    carry = uniform_filter1d(track_carry.astype(float), 15, mode="constant") >= .65
    pass_windows = np.zeros(n, dtype=bool)
    launch = np.nan_to_num(shift(speed, -5), nan=0)
    before = np.nan_to_num(shift(speed, 5), nan=0)
    persistent = uniform_filter1d((speed > 3.0).astype(float), 9, mode="constant")
    departure = (launch > 3.8) & (launch - before > 1.3) & (near < 2.0) & reliable & (persistent > .35)
    departures = peak_indices(departure, launch - before, 10)
    # 공이 떠난 뒤 다른 선수가 받음(패스·수신)
    for a in departures:
        b = min(n, a + 51)
        window = np.arange(a + 7, b)
        hits = window[reliable[window] & (near[window] < 1.35) & (owner[window] != owner[a])]
        if team is not None:
            # 같은 팀끼리의 패스만(2026-11-16): 휘슬 뒤 상대 골키퍼에게 공을 돌려주는 것은 경기 중 빌드업이 아니다
            hits = hits[np.asarray(team)[owner[hits]] == np.asarray(team)[owner[a]]]
        if hits.size:
            h = int(hits[0])
            flight = ball[a:h + 1]
            span = np.linalg.norm(ball[h] - ball[a])
            if (span >= 4 and np.isfinite(flight).all() and not jumps[a:h + 1].any()
                    and np.min(distance[a:h + 1]) >= 0
                    and (np.nanmedian(speed[max(a, h - 4):min(n, h + 5)]) < launch[a] * .8 or near[h] < .7)):
                pass_windows[a:h + 1] = True
    restart = departure & (np.nan_to_num(shift(persistent, -6), nan=0) > .48)
    restart_at = np.zeros(n, dtype=bool)
    restart_at[peak_indices(restart, launch - before, 13)] = True
    before_dist = shift(distance, 5)
    after_dist = shift(distance, -4)
    crossing = reliable & (shift(distance, 1) >= -margin) & (distance < -margin)
    crossing &= (before_dist - distance > .15) & (distance - after_dist > -.06)
    cross = peak_indices(crossing, np.nan_to_num(before_dist - distance, nan=0), 10)
    stop = np.zeros(n, dtype=bool)
    fallback = np.zeros(n, dtype=bool)
    accel = np.zeros(n, dtype=bool)
    if stop_team is not None:
        # 노트북 movement_features의 stop_hint 그대로(①에 씀): 양 팀이 함께 감속하고(같은 선수의 속도 변화, 팀마다 감속한
        # 선수 비율의 작은 쪽 > .32) 선수 3 m 안에서 빠르던 공이 갑자기 느려짐. 멈춰 있는 공만으로는 켜지지 않는다.
        team_arr = np.asarray(stop_team)
        vv = (shift(players, -3) - shift(players, 3)) * FPS / 6
        ps = np.linalg.norm(vv, axis=2)
        pv = shift(vv, 9)
        old_ps = np.linalg.norm(pv, axis=2)
        ds = ps - old_ps
        good = np.isfinite(ds) & (old_ps < 12) & (ps < 12)

        def share(flag, valid):
            return np.divide((flag & valid).sum(axis=1), valid.sum(axis=1), out=np.zeros(n), where=valid.sum(axis=1) > 0)

        with np.errstate(invalid="ignore"):
            team_decel = np.column_stack([share(ds[:, team_arr == k] < -.35, good[:, team_arr == k]) for k in (0, 1)])
            ball_interrupted = reliable & (before > 3.0) & (launch < before * .50) & (near < 3.0)
            stop = ball_interrupted & (team_decel.min(axis=1) > .32) & (distance > 1)
            # ③에 쓰는 신호(노트북 그대로): 공이 안 보이면 마지막 공 위치 6 m 안 선수들의 움직임 변화로 재개를 본다
            last_index = np.maximum.accumulate(np.where(observed, rows, -1))
            last = ball[np.maximum(last_index, 0)]
            near_last = np.linalg.norm(players - last[:, None, :], axis=2) < 6
            change = np.linalg.norm(vv - pv, axis=2)
            local = share((ds > .4) | ((change > 1.25) & (ps > .7)), near_last & good)
            fallback = ~reliable & (last_index >= 0) & (rows - last_index <= FPS * 2) & (local > .35)
            # ③ 개선(2026-11-16): 양 팀이 함께 속도를 올림. ①의 감속 신호를 대칭으로 쓴다(같은 .35·.32, 새 상수 없음).
            team_accel = np.column_stack([share(ds[:, team_arr == k] > .35, good[:, team_arr == k]) for k in (0, 1)])
            accel = team_accel.min(axis=1) > .32
    return {"reliable": reliable, "distance": distance, "restart": restart_at, "pass": pass_windows,
            "carry": carry, "cross": cross, "stop": stop, "fallback": fallback, "accel": accel}


def viterbi_cost(log_odds, enter, leave):
    """smoothing.viterbi와 같은 2상태 재귀에 프레임별 진입(enter)·이탈(leave) 비용을 둔 것."""
    alive, dead = 0., float(-log_odds[0])
    back_a = np.zeros(len(log_odds), dtype=bool)
    back_d = np.zeros(len(log_odds), dtype=bool)
    for i in range(1, len(log_odds)):
        switch_a = dead + leave[i]
        switch_d = alive + enter[i]
        back_a[i] = switch_a < alive
        back_d[i] = switch_d < dead
        alive, dead = min(alive, switch_a), min(dead, switch_d) - log_odds[i]
    state = np.empty(len(log_odds), dtype=bool)
    current = dead < alive
    for i in range(len(state) - 1, -1, -1):
        state[i] = current
        current = not current if (back_d[i] if current else back_a[i]) else current
    return state


def decode(score, players, ball, pitch, period, crossings, setting=None, restarts=(), extra_protect=None, team=None,
           support=None):
    """라인 아웃 순간 진입 비용 0(지금 파이프라인 1단계) + ⑤ 보호 구간. force_line_outs 전 상태를 돌려준다.
    restarts: 확실한 재개 프레임(감지된 페널티킥). 그 프레임은 인플레이로 고정하고 데드에서 나오는 비용을 0으로 둔다
    (라인 아웃의 진입 비용 0과 대칭). 킥 뒤 인플레이가 얼마나 이어질지는 모델 확률과 기존 벌점이 정한다(새 상수 없음)."""
    setting = setting or CONFIG["slow_buildup"]
    if (setting.get("case1", False) or setting.get("case3", False)) and team is None:
        raise ValueError("slow_buildup.decode: 보정 ①·③은 팀 정보가 필요하다. team=match.team을 넘길 것")
    if support is not None:
        support["restart"] = np.zeros(len(score), dtype=bool)
    penalty, bias = CONFIG["smooth_choice"]["penalty"], CONFIG["smooth_choice"]["bias"]
    strength, bonus, margin = setting["strength"], setting["bonus"], setting["margin"]
    period = np.asarray(period)
    result = np.zeros(len(score), dtype=bool)
    for half in halves(period):
        p = score[half]
        n = len(p)
        sig = movement_signals(players[half], ball[half], pitch, margin,
                               team if setting.get("same_team_pass", False) else None,
                               team if (setting.get("case1", False) or setting.get("case3", False)) else None)
        old = crossings[(crossings >= half.start) & (crossings < half.stop)] - half.start
        rs_in = np.asarray([r - half.start for r in restarts if half.start <= r < half.stop], dtype=np.int64)
        if not sig["reliable"].any() and not len(rs_in):
            result[half] = smoothing.viterbi(p, np.zeros(n), penalty, bias).astype(bool)
            continue
        log_odds = logit(np.clip(p, 1e-6, 1 - 1e-6))
        enter = np.full(n, penalty)
        leave = np.full(n, penalty)
        enter[old] = 0
        restart_ids = np.flatnonzero(sig["restart"])
        certified, short_pairs = [], []
        for a in sig["cross"]:
            rs = restart_ids[(restart_ids > a + round(.32 * FPS)) & (restart_ids < a + 25 * FPS)]
            evidence = p[a:min(n, a + 2 * FPS)]
            outside = sig["distance"][a:min(n, a + 8)] < -margin
            if len(rs) and (outside.sum() >= 4) and (evidence.max() >= .40 or np.any(old == a)):
                certified.append(int(a))
                if int(rs[0]) - a < 10 * FPS:
                    short_pairs.append((int(a), int(rs[0])))
        confirmed = np.unique(np.r_[old, np.array(certified, dtype=int)])
        if setting.get("case1", False):
            # ① 데드볼 시작 늦음(JSH 노트북 그대로): 공이 밖으로 넘고 재개가 이어진 넘음, 양 팀 감속 + 공 진행 중단(stop)에서
            # 데드 진입 비용을 낮춘다. 2026-11-16 검증(src/eval_jsh_all_corrections.py): ⑤ 대비 DFL +0.0011(2/0).
            picks = np.array(certified, dtype=int)
            enter[picks] *= (1 - .9 * strength)
            enter[sig["stop"]] *= (1 - .55 * strength)
        if setting.get("case3", False) or setting.get("restart_snap", False):
            # ③ 재개 늦음(JSH 노트북 그대로): 모델이 ±1초 안에 데드로 봤던 곳에서 공이 빠르게 출발하면 데드에서 나오는 비용을
            # 낮추고 출발 뒤 5프레임을 인플레이 쪽으로 민다. 공이 안 보이면 마지막 공 근처 선수 움직임(fallback)으로 약하게.
            # restart_accel(2026-11-16): 출발 뒤 1초 안에 양 팀이 함께 속도를 올린 재개만 쓴다. 데드볼 중 공을 주고받을 때는
            # 대부분 선수가 서 있다. 새 상수 없음(1초는 다른 규칙과 같은 창).
            context = maximum_filter1d(p, 2 * FPS + 1, mode="nearest") >= .5
            restart3 = sig["restart"] & context
            fallback3 = sig["fallback"] & (p < .58) & context
            if setting.get("restart_accel", False):
                soon = maximum_filter1d(sig["accel"].astype(np.uint8), FPS + 1, mode="constant", origin=-(FPS // 2)) != 0
                restart3 &= soon
                fallback3 &= soon
            if setting.get("case3", False):
                leave[restart3] *= 1 - .86 * strength
                after = extend_from(np.flatnonzero(restart3), n, 0, 5)
                log_odds[after & sig["reliable"]] -= bonus
                leave[fallback3] *= 1 - .26 * strength
            if support is not None:
                # restart_snap: 비용은 바꾸지 않고 재개 위치만 넘긴다(snap_dead_ends가 데드볼 끝만 당긴다)
                support["restart"][half] |= restart3
        moving = sig["pass"] | sig["carry"]
        if extra_protect is not None:   # 골키퍼 보호 구간(keeper_hold.protect_mask): ⑤와 같은 벌점항으로 다룬다
            moving = moving | np.asarray(extra_protect)[half]
        protected = moving & ~extend_from(confirmed, n, 2, 4)
        if setting.get("no_exit", False):
            # ⑤가 진행 중인 데드볼을 끝내지 못하게 한다(2026-11-16 실험, 가설: 휘슬 뒤 돌려주는 공을 패스로 보고 데드볼을
            # 끊을 수 있다). ⑤ 없이 푼 판정에서 이미 데드인 프레임에서 시작한 보호 구간은 쓰지 않는다.
            # 데드로 '들어가는' 것을 막는 효과는 남는다. 새 상수 없음.
            plain_enter = np.full(n, penalty)
            plain_enter[old] = 0
            plain = viterbi_cost(logit(np.clip(p, 1e-6, 1 - 1e-6)), plain_enter, np.full(n, penalty))
            for a, b in runs_of(protected):
                if plain[a]:
                    protected[a:b] = False
        enter[protected] += 28 * strength
        log_odds[protected] -= 2.5 * bonus
        if setting.get("case6", False):
            # ⑥ 짧은 데드볼 누락(JSH 노트북 그대로): 10초 안에 라인 넘음과 재개가 함께 확인되면 두 전환 비용을 낮추고 그 사이를
            # 데드 쪽으로 민다. 2026-11-16 검증: ⑤ 대비 DFL +0.0008(1/0), Metrica +0.0042.
            for a, b in short_pairs:
                enter[a] = min(enter[a], 2.5)
                leave[b] = min(leave[b], 2.5)
                log_odds[a:b] += .45 * bonus
        enter[old] = 0
        leave[rs_in] = 0
        log_odds[rs_in] = -1e9
        decoded = viterbi_cost(log_odds, enter, leave)
        if setting.get("revert_short_islands", False):
            # ⑤가 만든 짧은 인플레이 섬을 되돌린다(2026-11-16 실험, 가설: 휘슬 뒤 공 정리를 ⑤가 빌드업으로 보고 데드볼
            # 사이에 짧은 인플레이를 끼울 수 있다). ⑤ 없이 데드였던 곳에서 ⑤가 만든 인플레이가 양쪽이 데드이고
            # out_start_window_s(5초) 안에 끝나면 데드로 되돌린다. 긴 데드를 끊어 긴 인플레이를 만드는 주된 이득은 남는다.
            plain_enter = np.full(n, penalty)
            plain_enter[old] = 0
            plain = viterbi_cost(logit(np.clip(p, 1e-6, 1 - 1e-6)), plain_enter, np.full(n, penalty))
            limit = int(CONFIG["final"]["out_start_window_s"] * FPS)
            for a, b in runs_of(~decoded):
                if a > 0 and b < n and decoded[a - 1] and decoded[b] and b - a <= limit and plain[a:b].mean() >= 0.5:
                    decoded[a:b] = True
        result[half] = decoded
    return result


def snap_dead_ends(state, restart, period):
    """③ 재설계(2026-11-16, 사용자 제안 '새 인플레이는 만들지 말고 경계만 당기기'): 데드볼 [s, e)의 끝 e 앞
    out_start_window_s(5초) 안에 재개(restart, decode의 support)가 있으면 끝을 그 재개로 앞당긴다. 여러 개면 e에 가장
    가까운 것(가장 적게 옮김). 구간을 새로 만들거나 합치지 않는다. 데드볼 시작을 근처 라인 넘음으로 옮기는
    final_dead_balls.build와 대칭이고 같은 5초 창을 쓴다. 새 상수 없음. 반환 (새 state, 옮긴 초 목록)."""
    out = np.asarray(state, dtype=bool).copy()
    period = np.asarray(period)
    window = int(CONFIG["final"]["out_start_window_s"] * FPS)
    rel = np.flatnonzero(restart)
    moved = []
    for s, e in runs_of(out):
        if e >= len(out) or period[e] != period[s]:
            continue
        cand = rel[(rel > s) & (rel >= e - window) & (rel < e)]
        if len(cand):
            r = int(cand[-1])
            out[r:e] = False
            moved.append((e - r) / FPS)
    return out, moved


def release_forced(forced, decoded, restart, crossings, period, ball=None, pitch=None, zone=False):
    """③의 라인 아웃 강제 해제(JSH 노트북 decode 끝부분 그대로): force_line_outs가 강제로 데드로 둔 구간(decode는 인플레이로
    봤던 곳)에서 ③의 재개가 감지되면, 그 사이에 다른 라인 넘음이 없을 때 재개부터 인플레이로 돌린다.
    zone(2026-11-16): 재개 자리가 나간 라인에 맞을 때만 푼다(경기 규칙 15·16·17조). 터치라인으로 나갔으면 그 터치라인
    RESTART_TOL_M(2 m) 안, 골라인으로 나갔으면 그 쪽 골 에어리어(+2 m)나 코너(1 m 호 +2 m). 새 상수 없음."""
    out = np.asarray(forced, dtype=bool).copy()
    period = np.asarray(period)
    cross_mask = np.zeros(len(out), dtype=bool)
    c = np.asarray(crossings, dtype=np.int64)
    cross_mask[c[(c >= 0) & (c < len(out))]] = True
    import keeper_hold as kh
    for h in halves(period):
        rel = np.flatnonzero(restart[h])
        for a, b in runs_of(out[h] & ~np.asarray(decoded)[h]):
            chosen = rel[(rel >= max(0, a - 3)) & (rel < b)]
            if zone and len(chosen):
                chosen = np.array([r for r in chosen if _restart_fits(h.start + a, h.start + r, ball, pitch, kh)],
                                  dtype=np.int64)
            if len(chosen) and not cross_mask[h][max(a, chosen[0]):b].any():
                out[h.start + max(a, chosen[0]):h.start + b] = False
    return out


def _restart_fits(c, r, ball, pitch, kh):
    """c(강제 데드 시작 = 라인 넘음 근처)에 나간 라인과 r(재개)의 공 위치가 맞는지."""
    length, width = pitch
    tol = kh.RESTART_TOL_M
    seg = ball[max(c - 3, 0):c + 4]
    seg = seg[np.isfinite(seg).all(axis=1)]
    if not len(seg) or not np.isfinite(ball[r]).all():
        return False
    ox, oy = seg[-1]
    rx, ry = ball[r]
    if ox < 0 or ox > length:                       # 골라인
        depth = rx if ox < 0 else length - rx
        goal_area = depth <= kh.GOAL_AREA_DEPTH_M + tol and abs(ry - width / 2) <= kh.GOAL_AREA_HALF_M + tol
        corner = np.hypot(depth, abs(ry - width / 2) - width / 2) <= 1.0 + tol   # 코너 호 반지름 1 m(1조)
        return bool(goal_area or corner)
    edge_y = 0.0 if oy < width / 2 else width        # 터치라인
    return bool(abs(ry - edge_y) <= tol)
