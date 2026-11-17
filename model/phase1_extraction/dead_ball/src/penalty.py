"""Penalty kicks from the player formation (Law 14) — inference code used by final_dead_balls.

Formation (v2): goalkeeper on the goal line between the posts, exactly one kicker within ARC - EDGE_M of the
penalty mark, nobody else inside the box (EDGE_M inside its edge) or near the mark.
Kick moment, two methods merged within MERGE_S:
  run_up_v2           (no ball) a kicker standing BACK_M+ from the mark arrives within ARRIVE_M of it and the
                      formation then breaks and does not re-form; the kick = the frame the formation first breaks
  setup_then_kick_v2  (ball) the formation holds MIN_HOLD_S+, then within AFTER_S the ball leaves the mark area fast
                      and towards the goal
Constants: pitch dimensions from Law 1, coordinate tolerances fixed before any result (2026-11-14/15).
Validation on public data: src/eval_penalty_combo.py, eval_penalty_moment.py, eval_penalty_visible.py.
"""

import numpy as np

from config import CONFIG
import features as fx

FPS = CONFIG["fps"]
# Law 1 dimensions (m): box depth, box half-width, penalty mark distance, mark-arc radius, half goal width
BOX_DEPTH, BOX_HALF, SPOT, ARC, POST_HALF = 16.5, 20.16, 11.0, 9.15, 3.66
GOAL_LINE_M = 1.5      # goalkeeper "on the line" tolerance
MIN_HOLD_S = 1.0       # the formation must hold this long before a kick
SPOT_TOL_M = 2.0       # ball "at the mark" tolerance
MERGE_S = 5.0          # detections closer than this are one kick
# run-up tolerances (fixed 2026-11-14, before any result)
EDGE_M, ARRIVE_M, BACK_M, STILL_MPS, LOOK_S, FORM_S, FORM_FRAC = 1.0, 1.5, 2.5, 1.0, 4.0, 2.0, 0.8
# visible-player minimum of the formation. 0 = no condition (overlapping players are merged into one track);
# chosen between 16 and 0 on public data (src/eval_penalty_visible.py). Set from CONFIG["penalty"] in detect().
FORMATION_MIN_VISIBLE = 0
AFTER_S = 3.0          # time after the formation ends to wait for the kick


def _ball_signals(match):
    import final_dead_balls
    return final_dead_balls.ball_signals(match)


def signals(match):
    """side(0: x=0 골, 1: x=length 골)별 (마크까지 공 거리, 박스 안 선수 수)와 공 신호."""
    length, width = match.pitch
    p = match.players.astype(np.float64)
    x, y = p[..., 0], p[..., 1]
    seen = np.isfinite(x) & np.isfinite(y)
    ball = match.ball.astype(np.float64)
    sig = _ball_signals(match)
    per_side = []
    for side in (0, 1):
        depth = x if side == 0 else length - x
        spot_x = SPOT if side == 0 else length - SPOT
        with np.errstate(invalid="ignore"):
            in_box = seen & (depth <= BOX_DEPTH) & (depth >= -1) & (np.abs(y - width / 2) <= BOX_HALF)
            dist = np.hypot(ball[:, 0] - spot_x, ball[:, 1] - width / 2)
        per_side.append((dist, in_box.sum(axis=1)))
    return sig, per_side


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
        spot_x = SPOT if side == 0 else length - SPOT
        with np.errstate(invalid="ignore"):
            d = np.hypot(x - spot_x, y - width / 2)
            close = seen & (d < ARC - EDGE_M)
            deep = seen & (depth <= BOX_DEPTH - EDGE_M) & (depth >= -1) & (np.abs(y - width / 2) <= BOX_HALF - EDGE_M)
            keeper = deep & (depth <= GOAL_LINE_M) & (np.abs(y - width / 2) <= POST_HALF)
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
            if b - a < MIN_HOLD_S * FPS:
                continue
            e = min(b + int(AFTER_S * FPS), n)
            with np.errstate(invalid="ignore"):
                hit = np.flatnonzero(sig["moving"][a:e] & (dist[a:e] <= SPOT_TOL_M) & toward[side][a:e])
            if len(hit):
                found.append(a + int(hit[0]))
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


def detect(match):
    """Detected penalty kicks [(frame, method)]; within MERGE_S the earlier one is kept."""
    global FORMATION_MIN_VISIBLE
    FORMATION_MIN_VISIBLE = CONFIG["penalty"]["min_visible"]
    found = sorted([(k, "run_up_v2") for k in run_up_v2(match, CONFIG["penalty"]["run_up_moment"])]
                   + [(k, "setup_then_kick_v2") for k in setup_then_kick_v2(match)])
    out = []
    for k, method in found:
        if not out or k - out[-1][0] > MERGE_S * FPS:
            out.append((k, method))
    return out
