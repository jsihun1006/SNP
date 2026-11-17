import numpy as np
from scipy.signal import find_peaks
from data_utils import contiguous_runs

def native_features(half):
    """Observed geometry only, with future context permitted for offline correction."""
    xy = half.xy
    ball = xy[:, -1]
    seen = np.isfinite(ball).all(axis=1)
    alive = half.coarse["base"] != 2
    before = np.full_like(ball, np.nan)
    after = np.full_like(ball, np.nan)
    for a, b in contiguous_runs(alive & seen, half.frame):
        if b - a >= 5:
            before[a + 2:b] = (ball[a + 2:b] - ball[a:b - 2]) / .08
            after[a:b - 2] = (ball[a + 2:b] - ball[a:b - 2]) / .08
    speed_before = np.linalg.norm(before, axis=1)
    speed_after = np.linalg.norm(after, axis=1)
    denom = speed_before * speed_after
    cosine = np.divide((before * after).sum(axis=1), denom,
                       out=np.ones(len(ball)), where=denom > 1e-6)
    angle = np.arccos(np.clip(cosine, -1, 1))
    deceleration = np.maximum(0, speed_before - speed_after)
    jerk = np.linalg.norm(after - before, axis=1)
    nearest, near_counts = [], []
    for team in (0, 1):
        nodes = xy[:, half.team == team]
        distances = np.linalg.norm(nodes - ball[:, None], axis=-1)
        distances = np.where(np.isfinite(distances), distances, np.inf)
        nearest.append(distances.min(axis=1))
        near_counts.append((distances < 3).sum(axis=1))
    contest = (nearest[0] < 3) & (nearest[1] < 3) & alive
    low_ball = np.minimum(nearest[0], nearest[1]) <= 1.5
    touch_score = np.nan_to_num(angle / np.pi + deceleration / 10)
    touch = low_ball & ((angle >= np.deg2rad(15)) | (deceleration >= 3)) & seen & alive
    touch &= np.isfinite(before).all(axis=1) & np.isfinite(after).all(axis=1)
    peaks = find_peaks(np.where(touch, touch_score, 0), distance=3)[0]
    is_peak = np.zeros(len(ball), bool)
    is_peak[peaks] = True
    base = half.coarse["base"]
    signed_margin = np.where(base == 1, half.signals[:, 0], -half.signals[:, 0])
    x = np.column_stack((ball, before, after, speed_before, speed_after, angle, deceleration, jerk,
                         seen, nearest[0], nearest[1], near_counts[0], near_counts[1], contest,
                         half.signals, signed_margin)).astype(np.float32)
    x[~np.isfinite(x)] = np.nan
    # Features from 0.2 seconds before and after the candidate.
    history = np.full_like(x, np.nan)
    future = np.full_like(x, np.nan)
    for a, b in contiguous_runs(alive, half.frame):
        if b - a > 5:
            history[a + 5:b] = x[a:b - 5]
            future[a:b - 5] = x[a + 5:b]
    return {"x": np.c_[x, history, future], "contest": contest, "touch": is_peak,
            "touch_score": touch_score, "nearest": np.column_stack(nearest)}


def snap(half, pred, info, radius):
    if not radius:
        return pred.copy()
    result = pred.copy()
    for a, b in contiguous_runs(pred != 2, half.frame):
        events = np.flatnonzero(np.diff(pred[a:b])) + a + 1
        candidates = np.flatnonzero(info["touch"][a:b]) + a
        previous = a
        for j, t in enumerate(events):
            next_t = events[j + 1] if j + 1 < len(events) else b
            available = candidates[(candidates > previous) & (candidates < next_t) & (abs(candidates - t) <= radius)]
            if len(available):
                new_team = pred[t]
                near = info["nearest"][available, new_team] <= 1.5
                available = available[near]
            if len(available):
                target = min(available, key=lambda row: (abs(int(row) - int(t)), -info["touch_score"][row]))
                if target < t:
                    result[target:t] = pred[t]
                else:
                    result[t:target] = pred[t - 1]
                previous = target
            else:
                previous = t
    return result
