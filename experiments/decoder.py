"""Segment decoder: Viterbi over {H, A, D} with switching costs, run per graded half.

Per-frame emission scores come from the caller; this module only does the temporal decoding,
so every experiment shares one decoder and differs only in its evidence.
"""
import numpy as np

from common import HOME, AWAY, LENGTH, WIDTH, N_FRAMES, half_video_blocks, GAME

H, A, D = 0, 1, 2
LABELS = np.array(["H", "A", "D"])


def nearest_team_distances(pos, ball, team):
    """(dH, dA): distance in metres from the ball to the nearest home / away player. NaN if ball missing."""
    diff = pos - ball[:, None, :]
    dist = np.sqrt(np.nansum(diff ** 2, axis=2))
    dist[np.isnan(pos[:, :, 0])] = np.inf
    dist[np.isnan(ball[:, 0])] = np.nan
    with np.errstate(all="ignore"):
        dh = np.min(np.where(team[None, :] == HOME, dist, np.inf), axis=1)
        da = np.min(np.where(team[None, :] == AWAY, dist, np.inf), axis=1)
    dh[~np.isfinite(dh)] = np.nan
    da[~np.isfinite(da)] = np.nan
    return dh, da


def nearest_evidence(dh, da, clip=2.0, soft=0.5):
    """Log-ratio of nearest-opponent distances: > 0 favours home. 0 where the ball is missing."""
    e = np.log((da + soft) / (dh + soft))
    e = np.clip(e, -clip, clip)
    return np.nan_to_num(e, nan=0.0)


def viterbi(scores, switch_ha, switch_d):
    """Best path through (n,3) scores (higher = better) with a cost per label switch."""
    n = len(scores)
    cost = np.array([[0, switch_ha, switch_d],
                     [switch_ha, 0, switch_d],
                     [switch_d, switch_d, 0]], dtype=np.float64)
    best = scores[0].astype(np.float64).copy()
    back = np.zeros((n, 3), np.int8)
    for t in range(1, n):
        cand = best[:, None] - cost          # cand[i, j]: come from i, go to j
        back[t] = np.argmax(cand, axis=0)
        best = cand[back[t], np.arange(3)] + scores[t]
    path = np.empty(n, np.int8)
    path[-1] = int(np.argmax(best))
    for t in range(n - 1, 0, -1):
        path[t - 1] = back[t, path[t]]
    return path


def decode(scores, switch_ha, switch_d):
    """Full-length label array ('X' outside the graded halves), decoded one half at a time."""
    labels = np.full(N_FRAMES, "X", dtype="<U1")
    for _, a, b in half_video_blocks(GAME):
        labels[a:b + 1] = LABELS[viterbi(scores[a:b + 1], switch_ha, switch_d)]
    return labels


def make_scores(e_team, d_bias=None, d_evidence=None):
    """Emission matrix. e_team > 0 favours home; D gets a constant bias and/or a per-frame evidence."""
    s = np.zeros((len(e_team), 3), np.float64)
    s[:, H] = e_team / 2
    s[:, A] = -e_team / 2
    if d_bias is not None:
        s[:, D] = d_bias
    if d_evidence is not None:
        s[:, D] += d_evidence
    elif d_bias is None:
        s[:, D] = -1e6    # D never chosen
    return s
