"""선수 배치로 킥오프를 찾는다(공을 쓰지 않는다). 경기 규칙 8조: 킥오프 때 모든 선수는 자기 진영에 있고,
킥오프를 하지 않는 팀은 공(센터 마크)에서 9.15 m 밖에 있다. 두 팀이 하프라인을 사이에 두고 완전히 갈라선
배치는 경기 중에는 거의 없다.

배치(프레임): 한 팀의 보이는 선수가 모두 하프라인 한쪽(EDGE_M 여유), 다른 팀은 모두 반대쪽, 두 팀 모두
MIN_TEAM_VISIBLE명 이상 보이고, 적어도 한 팀은 센터 마크 CIRCLE_M 안에 아무도 없음.
킥오프 = 배치가 MIN_HOLD_S초 이상 이어진 구간이 끝나는 프레임(차면 선수들이 상대 진영으로 들어간다).
상수는 결과를 보기 전에 정했다(2026-11-15). EDGE_M, MIN_HOLD_S는 페널티킥 감지와 같은 값,
CIRCLE_M은 9.15 m − EDGE_M, MIN_TEAM_VISIBLE은 한 팀의 절반 정도(몇 명만 보일 때 우연히 갈라지는 것을 막는다).
검증: src/eval_kickoff_detect.py.
"""

import numpy as np

from config import CONFIG
import features as fx

FPS = CONFIG["fps"]
EDGE_M, MIN_HOLD_S, CIRCLE_M, MIN_TEAM_VISIBLE, MERGE_S = 1.0, 1.0, 9.15 - 1.0, 5, 5.0


def formation(match):
    length, width = match.pitch
    p = match.players.astype(np.float64)
    x, y = p[..., 0], p[..., 1]
    seen = np.isfinite(x) & np.isfinite(y)
    team = np.asarray(match.team)
    half = length / 2
    with np.errstate(invalid="ignore"):
        near_center = seen & (np.hypot(x - half, y - width / 2) < CIRCLE_M)
    sides = []
    for t in (0, 1):
        cols = team == t
        s, xx = seen[:, cols], x[:, cols]
        count = s.sum(axis=1)
        most = np.where(s, xx, -np.inf).max(axis=1)
        least = np.where(s, xx, np.inf).min(axis=1)
        sides.append({"count": count, "left": most <= half + EDGE_M, "right": least >= half - EDGE_M,
                      "clear": ~near_center[:, cols].any(axis=1)})
    a, b = sides
    split = (a["left"] & b["right"]) | (a["right"] & b["left"])
    enough = (a["count"] >= MIN_TEAM_VISIBLE) & (b["count"] >= MIN_TEAM_VISIBLE)
    return split & enough & (a["clear"] | b["clear"]) & (np.asarray(match.period) > 0)


# 골 뒤 세리머니 메우기(fill_before_kickoffs). 결과를 보기 전에 DFL 7경기 정답으로 정했다(2026-11-15):
#   FILL_WINDOW_S  골 → 킥오프 데드볼 길이의 최댓값 92.7초(17개, 중앙값 60.2초)
#   FILL_MAX_GAP_S 실제 인플레이 구간 길이의 하위 5% 3.2초(699개). 이보다 짧은 인플레이만 메운다.
FILL_WINDOW_S, FILL_MAX_GAP_S = 92.7, 3.2


def fill_before_kickoffs(state, kicks, period):
    """감지된 킥오프 (k, a)마다: 배치 구간 [a, k)는 데드(킥오프 전에는 공이 죽어 있다, 규칙 8조). 그리고 k 앞
    FILL_WINDOW_S초 안에서 양쪽이 데드로 막힌 FILL_MAX_GAP_S초 미만 인플레이를 데드로 메운다(세리머니 중 선수가 몰려
    뛰어 모델이 잠깐 인플레이로 본 것). 인플레이를 새로 만들지는 않는다. 반환 (새 state, 메운 프레임 수)."""
    out = np.asarray(state, dtype=bool).copy()
    period = np.asarray(period)
    before = out.sum()
    w, g = int(FILL_WINDOW_S * FPS), int(FILL_MAX_GAP_S * FPS)
    for k, a in kicks:
        if not (0 <= a < k <= len(out)) or period[a] <= 0:
            continue
        out[a:k] = True
        lo = max(k - w, 0)
        lo = max(lo, int(np.flatnonzero(period[:k] != period[a])[-1]) + 1 if (period[:k] != period[a]).any() else lo)
        for s, e in fx.runs_of(~out[lo:k]):
            s, e = lo + s, lo + e
            if s > lo and e < k and e - s < g:
                out[s:e] = True
    return out, int(out.sum() - before)


def hold_after_penalty_goals(state, penalty_kicks, kicks, period):
    """페널티킥 P 뒤 FILL_WINDOW_S초 안에 킥오프 (k, a)가 감지되면 그 페널티킥은 골이었다(킥오프는 골 뒤에만 온다,
    규칙 8조). 킥 직후 인플레이가 끝난 첫 데드 프레임부터 k까지 데드로 둔다(세리머니 중 끊김을 막는다). 공을 쓰지
    않는다. 막혀서 경기가 이어지면 킥오프가 오지 않아 바뀌지 않는다. 반환 (새 state, [(P, k)] 묶은 구간)."""
    out = np.asarray(state, dtype=bool).copy()
    period = np.asarray(period)
    w = int(FILL_WINDOW_S * FPS)
    made = []
    for p in penalty_kicks:
        after = [(k, a) for k, a in kicks if p < a and k <= p + w and (period[p:k] == period[p]).all()]
        if not after:
            continue
        k = after[0][0]
        # 킥 직후 인플레이(페널티 끊기가 만든 것)는 남기되 PENALTY_OUT_MAX_S(3초)까지만 인정한다. 그 뒤 모델이
        # 세리머니를 인플레이로 이어 봐도 데드로 묶는다.
        import final_dead_balls as fd
        alive = np.flatnonzero(out[p:k])
        r = int(alive[0]) if len(alive) else k - p
        start = p + min(r, int(fd.PENALTY_OUT_MAX_S * FPS))
        if start >= k:
            continue
        out[start:k] = True
        made.append((int(p), int(k)))
    return out, made


def kickoffs(match):
    """[(kick, setup_start)] 킥오프 프레임과 그 배치 구간의 시작."""
    ok = formation(match)
    out = []
    for a, b in fx.runs_of(ok):
        if b - a < MIN_HOLD_S * FPS:
            continue
        if out and b - out[-1][0] <= MERGE_S * FPS:
            out[-1] = (b, out[-1][1])
        else:
            out.append((b, a))
    return out
