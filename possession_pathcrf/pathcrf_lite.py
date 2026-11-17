"""PathCRF-lite: ball-free possession path inference (after Kim et al., KDD 2026, "PathCRF").

    python possession_pathcrf/pathcrf_lite.py            # DFL leave-one-match-out evaluation

Idea (paper): at every time step the possession state is one directed edge e = (u, v) between players:
a self-loop (u, u) = u controls the ball, (u, v) = the ball travels from u to v. Node scores say how likely
each player is the sender u and the receiver v; a masked CRF only allows
    stay      (u, v) -> (u, v)
    kick      (u, u) -> (u, v)          u releases the ball to v
    receive   (x, u) -> (u, w)          u controls it (w = u) or plays it on first time (w != u)
and Viterbi finds the best consistent path. No ball coordinates are used at inference.

Lite version: the neural backbone is replaced by two LightGBM node classifiers (sender, receiver) over
hand-made per-player features (own motion, pressure and convergence of opponents, team shape), at 5 fps like
the paper. Illegal transitions get a large finite penalty (not -inf) so a path always exists when tracks
appear or disappear. Team possession = team of the current sender, kept until the receiver touches the ball
("last team to apply force"). Dead-ball frames come from outside (truth or the dead-ball model).

Labels on DFL (training only): the ball is controlled by the nearest player when it is within CONTROL_M of
them; between two controls the edge is (previous controller, next controller).
"""

import os
import sys

import numpy as np
import pandas as pd

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(HERE)
sys.path[:0] = [os.path.join(ROOT, "src"), os.path.join(ROOT, "evaluation"), ROOT]

FPS = 25
STEP = 5                 # 25 fps -> 5 fps, as in the paper
CONTROL_M = 1.0          # ball within 1 m of a player = that player controls it (fixed before the run)
CONTROL_Z = 2.0          # ... and below head height
BIG = 30.0               # penalty of an illegal transition (soft mask)
CONTEXT = (-10, -5, 5, 10)   # context offsets in 5-fps steps (±1 s, ±2 s)


# ----------------------------------------------------------------------------------------------- labels
def control_labels(players, ball, ball_z, alive):
    """Per 25-fps frame: controller index (or -1) from the DFL ball."""
    d = np.linalg.norm(players - ball[:, None, :], axis=2)
    d = np.where(np.isfinite(d), d, np.inf)
    who = d.argmin(axis=1)
    near = d[np.arange(len(d)), who] < CONTROL_M
    low = ~(np.nan_to_num(ball_z, nan=0.0) > CONTROL_Z)
    return np.where(alive & near & low, who, -1)


def edge_labels(ctrl, alive):
    """(sender, receiver) per frame: control -> (u, u); between controls u then v -> (u, v)."""
    n = len(ctrl)
    s = np.full(n, -1)
    r = np.full(n, -1)
    seg_start = None
    for t in range(n + 1):
        if t < n and alive[t]:
            if seg_start is None:
                seg_start = t
            continue
        if seg_start is not None:
            idx = np.arange(seg_start, t)
            c = ctrl[idx]
            known = np.flatnonzero(c >= 0)
            if len(known):
                prev = np.maximum.accumulate(np.where(c >= 0, np.arange(len(c)), -1))
                nxt = np.minimum.accumulate(np.where(c >= 0, np.arange(len(c)), len(c))[::-1])[::-1]
                for k in range(len(c)):
                    if c[k] >= 0:
                        s[idx[k]] = r[idx[k]] = c[k]
                    else:
                        a = c[prev[k]] if prev[k] >= 0 else c[known[0]]
                        b = c[nxt[k]] if nxt[k] < len(c) else a
                        s[idx[k]], r[idx[k]] = a, b
            seg_start = None
    return s, r


# --------------------------------------------------------------------------------------------- features
def attack_sign(players, team, period):
    """+1 if the team attacks towards +x in that period (team with the lower mean x defends the left goal)."""
    sign = np.zeros((len(players), len(team)))
    for p in np.unique(period):
        if p <= 0:
            continue
        m = period == p
        mx = [np.nanmean(players[m][:, team == k, 0]) for k in (0, 1)]
        home_right = mx[0] < mx[1]
        for k in (0, 1):
            sign[np.ix_(m, team == k)] = 1.0 if (home_right == (k == 0)) else -1.0
    return sign


def node_features(players, team, period, pitch, idx=None):
    """Per (grid step, player) feature cube (T, N, F) and names. No ball.
    idx: a regular frame grid (default every STEP-th frame = 5 fps). Acceleration is always over 0.2 s and the
    context offsets are always ±1 s / ±2 s, so a 5-fps grid (training) and a 25-fps grid (prediction) give
    features with the same meaning."""
    length, width = pitch
    p = players.astype(np.float64)
    vel = np.full_like(p, np.nan)
    vel[2:-2] = (p[4:] - p[:-4]) / (4.0 / FPS)
    sign = attack_sign(p, team, period)
    idx = np.arange(0, len(p), STEP) if idx is None else np.asarray(idx)
    grid = int(idx[1] - idx[0]) if len(idx) > 1 else STEP
    lag = max(1, int(round(0.2 * FPS / grid)))                       # 0.2 s in grid steps
    p, vel, sign, per = p[idx], vel[idx], sign[idx], period[idx]
    speed = np.linalg.norm(vel, axis=2)
    acc = np.full_like(speed, np.nan)
    acc[lag:] = (speed[lag:] - speed[:-lag]) / 0.2
    x_att = np.where(sign > 0, p[..., 0], length - p[..., 0])
    y_c = np.abs(p[..., 1] - width / 2)
    vx_att = vel[..., 0] * sign
    same = team[:, None] == team[None, :]
    eye = np.eye(len(team), dtype=bool)
    T, N = speed.shape
    d_opp1, d_opp2, d_mate1 = (np.full((T, N), np.inf) for _ in range(3))
    n_opp5, n_mate10, conv_opp, conv_mate = (np.zeros((T, N)) for _ in range(4))
    for a in range(0, T, 1000):                                      # pairwise terms in chunks (memory)
        b = min(a + 1000, T)
        diff = p[a:b, :, None, :] - p[a:b, None, :, :]                # (t, N, N, 2): i - j
        dist = np.linalg.norm(diff, axis=3)
        dist_o = np.where(~same & np.isfinite(dist), dist, np.inf)
        dist_m = np.where(same & ~eye & np.isfinite(dist), dist, np.inf)
        srt = np.sort(dist_o, axis=2)
        d_opp1[a:b], d_opp2[a:b] = srt[..., 0], srt[..., 1]
        d_mate1[a:b] = dist_m.min(axis=2)
        n_opp5[a:b] = (dist_o < 5).sum(axis=2)
        n_mate10[a:b] = (dist_m < 10).sum(axis=2)
        # convergence: other players j running towards i (j's velocity points at i)
        with np.errstate(invalid="ignore", divide="ignore"):
            to_i = diff / dist[..., None]
            vj = vel[a:b, None, :, :]
            sj = np.linalg.norm(vj, axis=3)
            cos = np.nansum(to_i * vj, axis=3) / sj
        running_at = (cos > 0.8) & (sj > 1.5) & (dist < 15)
        conv_opp[a:b] = (running_at & ~same).sum(axis=2)
        conv_mate[a:b] = (running_at & same & ~eye).sum(axis=2)
    d_opp = np.stack([d_opp1, d_opp2], axis=2)
    team_mean_x = np.stack([np.nanmean(np.where(team == k, x_att, np.nan), axis=1) for k in (0, 1)], axis=1)
    rel_x = x_att - team_mean_x[:, team]
    rank_x = pd.DataFrame(x_att).rank(axis=1, pct=True).to_numpy()
    team_speed = np.stack([np.nanmean(np.where(team == k, speed, np.nan), axis=1) for k in (0, 1)], axis=1)
    own_speed, opp_speed = team_speed[:, team], team_speed[:, 1 - team]
    d_goal_own = np.hypot(x_att, y_c)
    d_goal_opp = np.hypot(length - x_att, y_c)
    base = {"x_att": x_att, "y_c": y_c, "vx_att": vx_att, "vy": vel[..., 1], "speed": speed, "acc": acc,
            "d_opp1": np.where(np.isfinite(d_opp[..., 0]), d_opp[..., 0], 60),
            "d_opp2": np.where(np.isfinite(d_opp[..., 1]), d_opp[..., 1], 60),
            "d_mate1": np.where(np.isfinite(d_mate1), d_mate1, 60), "n_opp5": n_opp5, "n_mate10": n_mate10,
            "conv_opp": conv_opp, "conv_mate": conv_mate, "rel_x": rel_x, "rank_x": rank_x,
            "own_speed": own_speed, "opp_speed": opp_speed, "d_goal_own": d_goal_own, "d_goal_opp": d_goal_opp}
    names, cubes = list(base), list(base.values())
    for key in ("speed", "conv_opp", "d_opp1", "conv_mate", "acc"):
        for off5 in CONTEXT:
            off = int(round(off5 * STEP / grid))                     # CONTEXT is in 5-fps steps
            shifted = np.full_like(base[key], np.nan, dtype=np.float64)
            if off > 0:
                shifted[:-off] = base[key][off:]
            else:
                shifted[-off:] = base[key][:off]
            names.append("{}_{:+d}".format(key, off5))
            cubes.append(shifted)
    present = np.isfinite(p).all(axis=2)
    return np.stack(cubes, axis=2).astype(np.float32), names, present, idx, per


# ---------------------------------------------------------------------------------------------- decode
try:
    import numba as _nb
except ImportError:      # numba is optional; the NumPy version below is the reference
    _nb = None


if _nb is not None:
    @_nb.njit(cache=True)
    def _viterbi_edges_nb(ls, lr, lam_kick, lam_recv, big):   # lam_kick, lam_recv, big: per-step arrays
        """Same recursion, tie-breaking and back-tracking as viterbi_edges_numpy, compiled."""
        T, N = ls.shape
        neg = -1e9
        score = np.empty((N, N))
        for u in range(N):
            for v in range(N):
                score[u, v] = ls[0, u] + lr[0, v]
        back_kind = np.zeros((T, N, N), dtype=np.int8)
        back_arg = np.zeros((T, N, N), dtype=np.int16)
        new = np.empty((N, N))
        recv_val = np.empty(N)
        recv_from = np.empty(N, dtype=np.int64)
        for t in range(1, T):
            for u in range(N):                       # best (x, u), x != u, first index on ties
                best, arg = -np.inf, 0
                for x in range(N):
                    val = neg if x == u else score[x, u]
                    if val > best:
                        best, arg = val, x
                recv_val[u] = best - lam_recv[t]
                recv_from[u] = arg
            best_flat, best_val = 0, -np.inf
            for u in range(N):
                for v in range(N):
                    if score[u, v] > best_val:
                        best_val, best_flat = score[u, v], u * N + v
            jump = best_val - big[t]
            for u in range(N):
                kick = score[u, u] - lam_kick[t]
                for v in range(N):
                    kind, val = 0, score[u, v]
                    if kick > val:
                        kind, val = 1, kick
                    if recv_val[u] > val:
                        kind, val = 2, recv_val[u]
                    if jump > val:
                        kind, val = 3, jump
                    if u == v and kind == 1:
                        kind = 0
                    new[u, v] = val
                    back_kind[t, u, v] = kind
                    back_arg[t, u, v] = recv_from[u] if kind == 2 else (best_flat if kind == 3 else 0)
            for u in range(N):
                for v in range(N):
                    score[u, v] = new[u, v] + ls[t, u] + lr[t, v]
        best_val, best_flat = -np.inf, 0
        for u in range(N):
            for v in range(N):
                if score[u, v] > best_val:
                    best_val, best_flat = score[u, v], u * N + v
        u, v = best_flat // N, best_flat % N
        su = np.empty(T, dtype=np.int64)
        sv = np.empty(T, dtype=np.int64)
        for t in range(T - 1, -1, -1):
            su[t], sv[t] = u, v
            if t == 0:
                break
            k, a = back_kind[t, u, v], back_arg[t, u, v]
            if k == 1:
                v = u
            elif k == 2:
                u, v = a, u
            elif k == 3:
                u, v = a // N, a % N
        return su, sv


def viterbi_edges(log_s, log_r, present, lam_kick, lam_recv, big=None):
    """Best edge path (numba-compiled when numba is installed; identical to viterbi_edges_numpy).
    lam_kick, lam_recv, big may be scalars or per-step arrays (the cost of a change INTO step t)."""
    T = len(log_s)
    arrays = [np.broadcast_to(np.asarray(x, dtype=np.float64), (T,)).copy()
              for x in (lam_kick, lam_recv, BIG if big is None else big)]
    if _nb is None:
        if all(np.all(a == a[0]) for a in arrays):
            return viterbi_edges_numpy(log_s, log_r, present, arrays[0][0], arrays[1][0], arrays[2][0])
        raise RuntimeError("per-step penalties need numba")
    neg = -1e9
    ls = np.ascontiguousarray(np.where(present, log_s, neg), dtype=np.float64)
    lr = np.ascontiguousarray(np.where(present, log_r, neg), dtype=np.float64)
    return _viterbi_edges_nb(ls, lr, arrays[0], arrays[1], arrays[2])


def viterbi_edges_numpy(log_s, log_r, present, lam_kick, lam_recv, big=None):
    """Best edge path. log_s, log_r: (T, N) node log-scores. Returns sender, receiver per step."""
    T, N = log_s.shape
    neg = -1e9
    ls = np.where(present, log_s, neg)
    lr = np.where(present, log_r, neg)
    eye = np.eye(N, dtype=bool)
    score = ls[0][:, None] + lr[0][None, :]
    back_kind = np.zeros((T, N, N), dtype=np.int8)   # 0 stay, 1 kick, 2 receive, 3 illegal jump
    back_arg = np.zeros((T, N, N), dtype=np.int16)
    for t in range(1, T):
        stay = score
        kick = np.diagonal(score)[:, None] - lam_kick                       # from (u, u) to (u, v)
        off = np.where(eye, neg, score)                                      # (x, u) with x != u
        recv_from = off.argmax(axis=0)                                       # best x for each u
        recv = (off.max(axis=0) - lam_recv)[:, None]                         # (x, u) -> (u, w)
        best_flat = score.argmax()
        jump = score.flat[best_flat] - (BIG if big is None else big)
        cand = np.stack([stay, np.broadcast_to(kick, (N, N)), np.broadcast_to(recv, (N, N)),
                         np.full((N, N), jump)])
        kind = cand.argmax(axis=0)
        new = cand.max(axis=0)
        # kick only makes sense for v != u; on the diagonal kick == stay
        kind = np.where(eye & (kind == 1), 0, kind)
        arg = np.where(kind == 2, recv_from[:, None], np.where(kind == 3, best_flat, 0))
        back_kind[t], back_arg[t] = kind, arg
        score = new + ls[t][:, None] + lr[t][None, :]
    u, v = np.unravel_index(score.argmax(), (N, N))
    su, sv = np.empty(T, dtype=np.int64), np.empty(T, dtype=np.int64)
    for t in range(T - 1, -1, -1):
        su[t], sv[t] = u, v
        if t == 0:
            break
        k, a = back_kind[t, u, v], back_arg[t, u, v]
        if k == 0:
            pass
        elif k == 1:
            v = u
        elif k == 2:
            u, v = a, u
        else:
            u, v = np.unravel_index(a, (N, N))
    return su, sv


def decode_match(prob_s, prob_r, present, alive_steps, lam_kick, lam_recv):
    """Run Viterbi on each in-play stretch; return sender per step (-1 when dead)."""
    T, N = prob_s.shape
    eps = 1e-6
    ls = np.log(np.clip(prob_s, eps, 1)) - np.log(np.clip(np.where(present, prob_s, 0).sum(1, keepdims=True), eps, None))
    lr = np.log(np.clip(prob_r, eps, 1)) - np.log(np.clip(np.where(present, prob_r, 0).sum(1, keepdims=True), eps, None))
    sender = np.full(T, -1)
    edges = np.diff(np.r_[False, alive_steps, False].astype(np.int8))
    for a, b in zip(np.flatnonzero(edges == 1), np.flatnonzero(edges == -1)):
        su, _ = viterbi_edges(ls[a:b], lr[a:b], present[a:b], lam_kick, lam_recv)
        sender[a:b] = su
    return sender
