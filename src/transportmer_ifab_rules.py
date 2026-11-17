"""IFAB boundary evidence and constraints for one 25 fps tracking half."""

from dataclasses import dataclass
from types import FunctionType

import numpy as np
from scipy.ndimage import minimum_filter1d, uniform_filter1d

from features import runs_of


@dataclass(frozen=True)
class Rules:
    fps: int = 25
    ball_radius: float = .11
    outside_margin: float = .25
    max_ball_speed: float = 45
    outside_frames: int = 2
    gap_frames: int = 10
    throw_seconds: int = 25
    throw_distance: float = 6
    thrower_distance: float = 2
    confirmation_frames: int = 12
    restart_follow_frames: int = 15
    stationary_frames: int = 6
    kick_frames: int = 4
    displacement: float = .30
    kick_onset: float = .07
    kicker_distance: float = 1.5
    corner_radius: float = 1
    goal_area_depth: float = 5.5
    goal_half_width: float = 3.66
    penalty_depth: float = 11
    penalty_area_depth: float = 16.5
    penalty_player_distance: float = 9.15
    mark_tolerance: float = 1
    keeper_line_tolerance: float = 1.5
    players_per_team: int = 6
    setup_separation: float = .8
    goal_kickoff_seconds: int = 40
    kickoff_half_start_seconds: int = 30
    restart_context_seconds: int = 30
    dead_context_frames: int = 12
    carry_frames: int = 10
    decel_frames: int = 10
    decel_delta: float = .35
    decel_share: float = .5
    ball_approach_speed: float = 2
    ball_stationary_speed: float = .6
    wait_frames: int = 50
    wait_middle_band: float = 4
    wait_touchline_band: float = 3
    wait_players: int = 3
    gap_bonus: float = 12
    stop_bonus: float = 18
    waiting_cost: float = 16


STAGES = {"six_rules": 0, "throw_pairs": 1, "confirmed_restarts": 2,
          "play_protection": 3, "all_rules": 4}


@dataclass(frozen=True)
class Lock:
    start: int
    end: int
    dead: bool
    priority: int
    stage: int
    rule: str
    evidence: str


@dataclass(frozen=True)
class Crossing:
    frame: int
    previous: int
    side: int
    along: float
    inward: bool
    exact: bool


def flags_at_least(mask, length):
    result = np.zeros(len(mask), dtype=bool)
    for a, b in runs_of(mask):
        if b - a >= length:
            result[a:b] = True
    return result


def coordinates_on_lines(ball, pitch):
    x, y = ball.T
    length, width = pitch
    return np.column_stack((x, length-x, y, width-y))


def crossings(ball, pitch, cfg):
    dist = coordinates_on_lines(ball, pitch)
    seen = np.isfinite(ball).all(axis=1)
    rows = np.flatnonzero(seen)
    # Remember the last completely inside frame to bound uncertain crossings.
    inside = np.all(dist >= 0, axis=1) & seen
    last_inside = np.maximum.accumulate(np.where(inside, np.arange(len(ball)), -1))
    limit = cfg.ball_radius + cfg.outside_margin
    candidates = {}
    for side in range(4):
        d = dist[:, side]
        along = ball[:, 1 if side < 2 else 0]
        line_length = pitch[1 if side < 2 else 0]
        # Inward crossings use the first observed positive coordinate.
        for a, b in zip(rows[:-1], rows[1:], strict=True):
            if d[a] < 0 <= d[b] and d[a] < d[b] and b-a <= cfg.gap_frames + 1:
                if 0 <= along[b] <= line_length and inside[b]:
                    speed = np.linalg.norm(ball[b]-ball[a])*cfg.fps/(b-a)
                    if speed <= cfg.max_ball_speed:
                        candidates[True, side, b] = Crossing(b, a, side, float(along[b]), True, b == a+1)
        outside = seen & (d <= -limit)
        for begin, end in runs_of(outside):
            if end-begin < cfg.outside_frames or begin == 0:
                continue
            a = last_inside[begin-1]
            if a < 0 or not 0 <= along[begin] <= line_length:
                continue
            prev = np.searchsorted(rows, begin)-1
            prior = rows[prev] if prev >= 0 else a
            last_inside_for_side = last_inside[prior]
            if last_inside_for_side < 0 or d[prior] <= -limit:
                continue
            step_speed = np.linalg.norm(ball[begin]-ball[prior])*cfg.fps/(begin-prior)
            if step_speed > cfg.max_ball_speed or d[prior] <= d[begin]:
                continue
            if step_speed == 0 or not np.isfinite(step_speed):
                continue
            # Reject a near-line bounce with immediate return. Clear inside movement later
            # can still be a fast, separately verified restart.
            if end < len(ball) and end-begin == cfg.outside_frames and inside[end]:
                continue
            exact = seen[a:begin+cfg.outside_frames].all()
            candidates[False, side, begin] = Crossing(begin, int(a), side, float(along[begin]), False, exact)
    return sorted(candidates.values(), key=lambda c: (c.frame, c.side, c.inward))


def observed_kicks(ball, players, cfg):
    """Yield the first directed movement supported by stationary ball and contact proximity."""
    seen = np.isfinite(ball).all(axis=1)
    n, k, f = len(ball), cfg.stationary_frames, cfg.kick_frames
    for t in range(k, n-f+1):
        if not seen[t-k:t+f].all():
            continue
        position = ball[t-1]
        history = ball[t-k:t] - position
        if (np.linalg.norm(history, axis=1) > cfg.displacement).any():
            continue
        vectors = ball[t:t+f] - position
        direction = vectors[-1]
        distance = np.linalg.norm(direction)
        if distance < cfg.displacement or distance*cfg.fps/f > cfg.max_ball_speed:
            continue
        step = np.diff(np.vstack((position, ball[t:t+f])), axis=0)
        projected = step @ (direction/distance)
        if projected[0] < cfg.kick_onset or np.any(projected[1:] < -cfg.kick_onset):
            continue
        prev_motion = ball[t-1]-ball[t-2]
        if prev_motion @ (direction/distance) >= cfg.kick_onset:
            continue
        contacts = np.linalg.norm(players[t-1]-position, axis=1)
        contacts[~np.isfinite(contacts)] = np.inf
        actor = int(contacts.argmin())
        if contacts[actor] <= cfg.kicker_distance:
            yield t, position, actor, direction


def formation(players, team, pitch, actor, cfg):
    valid = np.isfinite(players).all(axis=1)
    if min(np.sum(valid & (team == side)) for side in (0, 1)) < cfg.players_per_team:
        return False
    valid[actor] = False
    home, away = (players[valid & (team == side), 0] < pitch[0]/2 for side in (0, 1))
    return max(min(home.mean(), (~away).mean()), min((~home).mean(), away.mean())) >= cfg.setup_separation


def penalty_setup(players, team, pitch, actor, goal_side, cfg):
    valid = np.isfinite(players).all(axis=1)
    if min(np.sum(valid & (team == side)) for side in (0, 1)) < cfg.players_per_team:
        return False
    x, y = players.T
    goal_dist = x if goal_side == 0 else pitch[0]-x
    center_dist = np.abs(y-pitch[1]/2)
    keeper = np.flatnonzero(valid & (team != team[actor]) & (np.abs(goal_dist) <= cfg.keeper_line_tolerance)
                           & (center_dist <= cfg.goal_half_width + cfg.mark_tolerance))
    if len(keeper) != 1:
        return False
    valid[[actor, keeper[0]]] = False
    behind = goal_dist >= cfg.penalty_depth-cfg.mark_tolerance
    outside = (goal_dist >= cfg.penalty_area_depth-cfg.mark_tolerance) | (
        center_dist >= cfg.goal_half_width+cfg.penalty_area_depth-cfg.mark_tolerance)
    beyond_arc = np.hypot(goal_dist-cfg.penalty_depth, center_dist) >= (
        cfg.penalty_player_distance-cfg.mark_tolerance)
    return (behind & outside & beyond_arc)[valid].all()


def deceleration(players, team, cfg):
    width = cfg.decel_frames
    previous = np.linalg.norm(players[width:-width]-players[:-2*width], axis=2)*cfg.fps/width
    after = np.linalg.norm(players[2*width:]-players[width:-width], axis=2)*cfg.fps/width
    continuous = minimum_filter1d(np.isfinite(players).all(axis=-1), 2*width+1,
                                  axis=0, mode="constant")[width:-width]
    valid = continuous & (previous < 12) & (after < 12)
    slowing = previous-after >= cfg.decel_delta
    supports = []
    for side in (0, 1):
        available = valid[:, team == side]
        count = available.sum(axis=1)
        share = (available & slowing[:, team == side]).sum(axis=1) / np.maximum(1, count)
        supports.append(np.where((count >= cfg.players_per_team) & (share >= cfg.decel_share), share, 0))
    return np.pad(np.minimum(*supports), width)


def build_rules(ball, players, team, pitch, base, live_signals, automatic, cfg):
    n = len(ball)
    seen = np.isfinite(ball).all(axis=1)
    inside = seen & (coordinates_on_lines(ball, pitch) >= 0).all(axis=1)
    pass_live = inside & live_signals["pass"] & live_signals["reliable"]
    carry_live = flags_at_least(inside & live_signals["carry"] & live_signals["reliable"], cfg.carry_frames)
    live = pass_live | carry_live
    marks, rejections, hints = [], [], []
    entered = crossings(ball, pitch, cfg)
    exits = [c for c in entered if not c.inward]
    paired = set()
    restarts = []
    soft = []
    slowing = deceleration(players, team, cfg) if n > 2*cfg.decel_frames else np.zeros(n)
    observed_players = np.isfinite(players).all(axis=2)
    speed = live_signals["speed"]

    def add(a, b, dead, priority, stage, rule, evidence):
        if 0 <= a < b <= n:
            marks.append(Lock(a, b, dead, priority, stage, rule, evidence))

    # Anchor confirmed restarts through the observed continuation. If the old mask
    # begins D late, the continuation also guards the delayed false stop.
    def release(t, stage, name, reason):
        end = min(n, t+cfg.restart_follow_frames)
        first_break = np.flatnonzero(~inside[t:end])
        if len(first_break):
            end = max(t+1, t+int(first_break[0]))
        add(t, end, False, 5, stage, name, reason)
        restarts.append(t)

    for entry in (c for c in entered if c.inward and c.side >= 2 and c.exact):
        t = entry.frame
        if t+cfg.confirmation_frames > n:
            continue
        future = inside[t:t+cfg.confirmation_frames]
        normal = coordinates_on_lines(ball[t:t+cfg.confirmation_frames], pitch)[:, entry.side]
        if not future.all() or not ((normal >= cfg.displacement).any() or pass_live[t:t+len(future)].any()):
            continue
        a = entry.previous
        distances = np.linalg.norm(players[a]-ball[a], axis=1)
        outside_players = coordinates_on_lines(players[a], pitch)[:, entry.side] <= 0
        eligible = outside_players & np.isfinite(distances) & (distances <= cfg.thrower_distance)
        if not eligible.any():
            continue
        candidates = [out for out in exits if out.side == entry.side and out.frame not in paired
                      and 0 < t-out.frame <= cfg.throw_seconds*cfg.fps
                      and abs(out.along-entry.along) <= cfg.throw_distance]
        if not candidates:
            continue
        out = candidates[-1]
        if live[out.frame:t].any():
            rejections.append((out.frame, t, "throw_in", "observed play between boundaries"))
            continue
        paired.add(out.frame)
        reason = f"touchline {entry.side}, exit {out.frame}, entry {t}, player outside"
        add(out.frame, t, True, 6, 1, "throw_in_pair", reason)
        release(t, 1, "throw_in_restart", reason)
        offered = np.unique(team[eligible])
        if len(offered) == 1:
            hints.append((t, int(offered[0])))

    for t, pos, actor, direction in observed_kicks(ball, players, cfg):
        left = int(pos[0] > pitch[0]/2)
        gd = pos[0] if left == 0 else pitch[0]-pos[0]
        lateral = abs(pos[1]-pitch[1]/2)
        candidates = [out for out in exits if 0 < t-out.frame <= cfg.restart_context_seconds*cfg.fps]
        latest = candidates[-1] if candidates else None
        arrived = (latest is not None and not live[latest.frame:t].any()
                   and not any(latest.frame < restart < t for restart in restarts))
        was_dead = bool(base[max(0,t-cfg.dead_context_frames):t].any())
        known_stop = arrived or was_dead
        goal_candidates = [out for out in exits if out.side < 2
                           and abs(out.along-pitch[1]/2) <= cfg.goal_half_width
                           and 0 < t-out.frame <= cfg.goal_kickoff_seconds*cfg.fps
                           and not live[out.frame:t].any()
                           and not any(out.frame < restart < t for restart in restarts)]
        at_center = np.linalg.norm(pos-np.asarray(pitch)/2) <= cfg.mark_tolerance
        at_penalty = np.hypot(gd-cfg.penalty_depth, lateral) <= cfg.mark_tolerance
        near_corner = np.hypot(gd, min(pos[1],pitch[1]-pos[1])) <= cfg.corner_radius
        goal_area = gd <= cfg.goal_area_depth and lateral <= cfg.goal_half_width+cfg.goal_area_depth
        if at_center and (goal_candidates or t <= cfg.kickoff_half_start_seconds*cfg.fps) and formation(
                players[t-1], team, pitch, actor, cfg):
            if goal_candidates:
                out = goal_candidates[-1]
                add(out.frame, t, True, 6, 2, "goal_to_kickoff", "goal-mouth crossing, center and team setup")
            release(t, 2, "kickoff", "center ball, team separation and first kick")
        elif known_stop and at_penalty and direction[0] * (1 if left else -1) > 0 and penalty_setup(
                players[t-1], team, pitch, actor, left, cfg):
            add(t-cfg.stationary_frames, t, True, 6, 2, "penalty_preparation", "mark, kicker, keeper and box geometry")
            release(t, 2, "penalty", "penalty formation and forward kick")
        elif arrived and latest.side == left and (near_corner or goal_area):
            name = "corner" if near_corner else "goal_kick"
            add(latest.frame, t, True, 6, 2, f"{name}_pair", "goal-line crossing and kick placement")
            release(t, 2, name, "stationary ball in restart area and observed kick")
        elif known_stop and not live[max(0,t-cfg.stationary_frames):t].any():
            if arrived:
                add(latest.frame, t, True, 6, 2, "restart_pair", "observed exit followed by stationary kick")
            name = "corner" if near_corner else "goal_kick" if goal_area else "free_kick"
            release(t, 2, name, "dead context and stationary ball followed by kick")
        else:
            continue

    for name, signal in (("pass_reception", pass_live), ("carry", carry_live)):
        for a, b in runs_of(signal):
            add(a, b, False, 3, 3, name, "continuous observed infield play")

    for out in exits:
        t, a = out.frame, out.previous
        if out.exact and a < t and live[max(0,a-cfg.decel_frames):a+1].any():
            add(a, a+1, False, 4, 4, "pre_crossing_play", "observed live before the line crossing")
        # Observed samples certify the ball is out. The existing decoder fills
        # any later, unobserved part without extending the hard mask across it.
        d = coordinates_on_lines(ball, pitch)[:, out.side]
        end = t+cfg.outside_frames
        while end < n and np.isfinite(d[end]) and d[end] <= -cfg.ball_radius-cfg.outside_margin:
            end += 1
        add(t, end, True, 5, 4, "line_out", f"outward crossing of line {out.side}, consecutive outside samples")
        missing = np.count_nonzero(~seen[a:t])
        if 0 < missing and t-a <= cfg.gap_frames+1:
            for i in range(a+1,t+1):
                weight = (i-a)/(t-a)
                soft.append((i, "enter", cfg.gap_bonus*(.5*weight+.5*slowing[i]), "short_occlusion_crossing"))

    fast = np.r_[0, np.cumsum(np.nan_to_num(speed) >= cfg.ball_approach_speed)]
    frames = np.arange(n)
    moving_before = fast[frames]-fast[np.maximum(0,frames-cfg.decel_frames)] > 0
    still = inside & np.isfinite(speed) & (speed <= cfg.ball_stationary_speed)
    settled = uniform_filter1d(still.astype(float), cfg.stationary_frames, mode="constant") >= .8
    stoppage = (slowing > 0) & settled & moving_before & ~live
    for a, b in runs_of(stoppage):
        point = a + int(slowing[a:b].argmax())
        soft.append((point, "enter", cfg.stop_bonus, "joint_deceleration"))
    for t in automatic.get("referee_stop", ()):
        lo, hi = max(0,t-cfg.decel_frames), min(n,t+cfg.decel_frames+1)
        if stoppage[lo:hi].any() and not live[t:hi].any():
            add(t, t+1, True, 5, 4, "referee_stop", "automatic referee signal and joint deceleration")
    for t in automatic.get("drop_ground_contact", ()):
        release(t, 4, "dropped_ball", "automatic ground contact")

    middle = np.abs(players[...,0]-pitch[0]/2) <= cfg.wait_middle_band
    touch = np.minimum(players[...,1], pitch[1]-players[...,1]) <= cfg.wait_touchline_band
    crowd = np.sum(observed_players & middle & touch, axis=1) >= cfg.wait_players
    pending = flags_at_least(base & still & (crowd | (slowing > 0)), cfg.wait_frames)
    pending &= ~live
    for t in np.flatnonzero(pending):
        soft.append((int(t), "leave", cfg.waiting_cost, "dead_waiting"))
    for t in automatic.get("waiting", ()):
        if base[t] and not live[t]:
            soft.append((t, "leave", cfg.waiting_cost, "automatic_waiting"))
    return marks, soft, rejections, hints


def constraints(marks, count, stage):
    """Resolve per-frame priorities. Equal conflicting evidence releases the lock."""
    state = np.full(count, -1, dtype=np.int8)
    priorities = np.zeros(count, dtype=np.int8)
    reason = np.full(count, -1, dtype=np.int32)
    blocked = np.zeros(count, dtype=bool)
    for index in sorted(range(len(marks)), key=lambda i: marks[i].priority, reverse=True):
        m = marks[index]
        if m.stage > stage:
            continue
        s = slice(m.start, m.end)
        conflict = (priorities[s] == m.priority) & (state[s] != int(m.dead))
        blocked[s] |= conflict
        take = (priorities[s] == 0) & ~blocked[s]
        state[s][take] = m.dead
        reason[s][take] = index
        priorities[s][take] = m.priority
    state[blocked] = -1
    reason[blocked] = -1
    return state, reason, blocked


def constrained_path(log_odds, enter, leave, state):
    """Use the original two-state recurrence with impossible emissions for locks."""
    n = len(state)
    alive, dead = (np.inf if state[0] == 1 else 0.,
                   np.inf if state[0] == 0 else -log_odds[0])
    back_a = np.zeros(n, dtype=bool)
    back_d = np.zeros(n, dtype=bool)
    for t in range(1, n):
        switch_a, switch_d = dead + leave[t], alive + enter[t]
        back_a[t], back_d[t] = switch_a < alive, switch_d < dead
        alive = np.inf if state[t] == 1 else min(alive, switch_a)
        dead = np.inf if state[t] == 0 else min(dead, switch_d)-log_odds[t]
    path = np.empty(n, dtype=bool)
    current = dead < alive
    for t in range(n-1, -1, -1):
        path[t] = current
        current ^= bool(back_d[t] if current else back_a[t])
    return path


def observed_line_out_locks(marks):
    return [mark for mark in marks if mark.rule in ("line_out", "pre_crossing_play")]


def apply_decoder(decode, p, game, settings, state, soft, rules=(1, 2, 3, 4, 5, 6)):
    def modify(log_odds, enter, leave):
        enter, leave = enter.copy(), leave.copy()
        for frame, kind, delta, _ in soft:
            if kind == "enter":
                enter[frame] = max(0, enter[frame]-delta)
            else:
                leave[frame] += delta
        return constrained_path(log_odds, enter, leave, state)

    namespace = dict(decode.__globals__, viterbi_cost=modify)
    decoder = FunctionType(decode.__code__, namespace, argdefs=decode.__defaults__)
    result, _ = decoder(p, game, "original", rules, settings)
    # The legacy line-out pass runs after Viterbi. Its result must still honor
    # verified restarts and all of the mutually consistent hard constraints.
    locked = state >= 0
    result[locked] = state[locked].astype(bool)
    return result
