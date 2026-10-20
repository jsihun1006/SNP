"""Dead-ball features: windowed behaviour of ball and players that separates a stoppage from live play."""
import numpy as np

from common import LENGTH, WIDTH, FPS, N_FRAMES
from features import _rolling_mean


def _fill(a, v):
    return np.where(np.isfinite(a), a, v)


def build(d, dh, da):
    ball, spd, pos = d["ball"], d["spd"], d["pos"]
    vis = np.isfinite(ball[:, 0]).astype(float)
    bsp = np.hypot(*np.diff(ball, axis=0, prepend=np.nan).T) * FPS
    near = np.fmin(dh, da)
    to_edge = np.minimum.reduce([ball[:, 0], LENGTH - ball[:, 0], ball[:, 1], WIDTH - ball[:, 1]])
    oob = ((ball[:, 0] < 0) | (ball[:, 0] > LENGTH) | (ball[:, 1] < 0) | (ball[:, 1] > WIDTH)).astype(float)
    oob[vis == 0] = np.nan
    with np.errstate(all="ignore"):
        pspd = np.nanmean(spd, axis=1)
        slow = np.nanmean(spd < 1.0, axis=1)
    w2, w10 = 2 * FPS + 1, 10 * FPS + 1
    cols = [
        _rolling_mean(pspd[:, None], w2)[:, 0],
        _rolling_mean(pspd[:, None], w10)[:, 0],
        _rolling_mean(slow[:, None], w10)[:, 0],
        _rolling_mean(vis[:, None], w2)[:, 0],
        _rolling_mean(vis[:, None], w10)[:, 0],
        _fill(_rolling_mean(bsp[:, None], w2)[:, 0], -1.0),
        _fill(_rolling_mean(near[:, None], w2)[:, 0], -1.0),
        _fill(_rolling_mean(np.minimum(to_edge, 20)[:, None], w2)[:, 0], -1.0),
        _fill(_rolling_mean(oob[:, None], w2)[:, 0], -1.0),
    ]
    return np.column_stack(cols)
