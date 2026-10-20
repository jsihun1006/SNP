"""Ball-free team-shape features: who is in possession, judged only from where the players are.

Everything is expressed in each team's own attacking frame (u = distance travelled towards the
opponent's goal), so one model serves both halves and both teams. Attack direction comes from the
goalkeeper: the end of the pitch the team's deepest player stands nearest to is its own goal.
"""
import numpy as np

from common import HOME, AWAY, LENGTH, WIDTH, N_FRAMES, FPS, half_video_blocks, GAME


def attack_sign(pos, team, half):
    """+1 if the team attacks towards high x in this half, else -1. dict {(team, half): sign}."""
    out = {}
    for h in (1, 2):
        m = half == h
        for t in (HOME, AWAY):
            x = pos[m][:, team == t, 0]
            lo = np.nanmedian(np.nanmin(x, axis=1))
            hi = np.nanmedian(np.nanmax(x, axis=1))
            out[(t, h)] = 1 if lo < LENGTH - hi else -1   # own goal at x=0 -> attacks +x
    return out


def _rolling_mean(a, w):
    """Centred rolling mean over axis 0 ignoring NaN, window w frames."""
    valid = np.isfinite(a).astype(np.float64)
    a0 = np.nan_to_num(a, nan=0.0)
    cs = np.cumsum(np.vstack([np.zeros((1,) + a.shape[1:]), a0]), axis=0)
    cv = np.cumsum(np.vstack([np.zeros((1,) + a.shape[1:]), valid]), axis=0)
    idx = np.arange(len(a))
    lo = np.maximum(0, idx - w // 2)
    hi = np.minimum(len(a), idx + w // 2 + 1)
    with np.errstate(all="ignore"):
        return (cs[hi] - cs[lo]) / (cv[hi] - cv[lo])


def team_block(pos, spd, team, half, signs, t):
    """Per-frame shape features for one team in its attacking frame."""
    m = team == t
    x = pos[:, m, 0].copy()
    y = pos[:, m, 1]
    sgn = np.zeros(N_FRAMES)
    for h in (1, 2):
        sgn[half == h] = signs[(t, h)]
    u = np.where(sgn[:, None] > 0, x, LENGTH - x)
    with np.errstate(all="ignore"):
        cu = np.nanmean(u, axis=1)
        cy = np.nanmean(y, axis=1)
        su = np.nanstd(u, axis=1)
        sy = np.nanstd(y, axis=1)
        spd_m = np.nanmean(spd[:, m], axis=1)
    # advanced / deep line via nanpercentile on sorted columns (vectorised)
    with np.errstate(all="ignore"):
        adv = np.nanpercentile(u, 85, axis=1)
        deep = np.nanpercentile(u, 15, axis=1)
    cu_s = _rolling_mean(cu[:, None], 2 * FPS + 1)[:, 0]
    vel = np.full(N_FRAMES, np.nan)
    k = int(1.5 * FPS)
    vel[k:-k] = (cu_s[2 * k:] - cu_s[:-2 * k]) / 3.0       # m/s along attack axis
    return np.column_stack([cu, cy, su, sy, adv, deep, spd_m, vel]), u


def build(d):
    """(N_FRAMES, n_features) feature matrix; NaN-tolerant (use with HistGradientBoosting)."""
    pos, spd, team, half = d["pos"], d["spd"], d["team"], d["half"]
    signs = attack_sign(pos, team, half)
    fh, _ = team_block(pos, spd, team, half, signs, HOME)
    fa, _ = team_block(pos, spd, team, half, signs, AWAY)
    sm = [_rolling_mean(f, FPS * 4 + 1) for f in (fh, fa)]       # 4 s smoothed copies
    diff = fh - fa
    return np.column_stack([fh, fa, diff, sm[0], sm[1]]), signs
