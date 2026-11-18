"""Pure local team motion diagnostics; no tracking/GT files are read here.

Coordinates in metres, velocities in m/s. Fixed engineering parameters below
are not fitted to this match or public GT. Direction coherence is distinct from
ray convergence. Focus scores/confidence are descriptive, not probabilities.
"""

from dataclasses import dataclass
import math

import numpy as np


@dataclass(frozen=True)
class MotionConfig:
    window_frames: int = 10
    min_moving_speed_mps: float = .5
    max_step_speed_mps: float = 12.
    min_moving: int = 4
    min_geometry_ratio: float = .04
    irls_iterations: int = 4
    huber_distance_m: float = 3.
    support_distance_m: float = 5.
    support_angle_deg: float = 45.
    min_support_fraction: float = .6
    confidence_geometry_full: float = .2
    confidence_moving_full: int = 8
    pair_min_distance_m: float = .25


def divide_or_nan(numerator, denominator):
    out = np.full(np.broadcast_shapes(np.shape(numerator), np.shape(denominator)), np.nan)
    np.divide(numerator, denominator, out=out, where=np.asarray(denominator) > 0)
    return out


def filtered_velocities(positions, state, fps, pitch, config=MotionConfig()):
    """Trailing means of consecutive valid steps; every step must pass the mask.

    Returned arrays exclude the ball slot. A failed observation, X/half boundary,
    outside-pitch endpoint or excessive instantaneous speed resets continuity.
    Filtering is diagnostic only; original data are never changed.
    """
    count, slots = positions.shape[0], positions.shape[1]-1
    result = np.full((count, slots, 2), np.nan, dtype=np.float32)
    rejected = np.zeros((count, slots), dtype=bool)
    window = config.window_frames
    if window < 1:
        raise ValueError('Velocity window must be positive.')
    ends = np.arange(1, count+1)
    starts = np.maximum(0, ends-window)
    pitch = np.asarray(pitch, dtype=float)
    for slot in range(slots):
        p = np.asarray(positions[:, slot], dtype=float)
        observed = np.isfinite(p).all(axis=1) & ((p >= 0) & (p <= 1)).all(axis=1)
        v = np.full((count, 2), np.nan)
        v[1:] = (p[1:]-p[:-1])*pitch*fps
        speed = np.linalg.norm(v, axis=1)
        adjacent = np.zeros(count, dtype=bool)
        adjacent[1:] = observed[1:] & observed[:-1]
        rejected[:, slot] = adjacent & state['eligible'] & (speed > config.max_step_speed_mps)
        valid = adjacent & state['eligible'] & np.isfinite(speed) & (speed <= config.max_step_speed_mps)
        totals = np.vstack((np.zeros((1, 2)), np.cumsum(np.where(valid[:, None], v, 0), axis=0)))
        numbers = np.r_[0, np.cumsum(valid)]
        usable = numbers[ends]-numbers[starts] == window
        result[usable, slot] = ((totals[ends]-totals[starts])/window)[usable]
    return result, rejected


def robust_focus(points, velocities, pitch, config=MotionConfig()):
    """Batched IRLS line intersection, followed by FORWARD ray support gates."""
    points, velocities = np.asarray(points, dtype=float), np.asarray(velocities, dtype=float)
    observed = np.isfinite(points).all(axis=2) & ((points >= 0) & (points <= np.asarray(pitch))).all(axis=2)
    speed = np.linalg.norm(velocities, axis=2)
    moving = (observed & np.isfinite(velocities).all(axis=2)
              & (speed >= config.min_moving_speed_mps) & (speed <= config.max_step_speed_mps+1e-5))
    n = moving.sum(axis=1)
    u = np.zeros_like(velocities)
    np.divide(velocities, speed[:, :, None], out=u, where=moving[:, :, None])
    p = np.where(moving[:, :, None], points, 0)
    normal_p = p-u*np.sum(u*p, axis=2)[:, :, None]
    weights = moving.astype(float)
    q = np.full((len(points), 2), np.nan)
    ratio = np.full(len(points), np.nan)
    geometry = np.zeros(len(points), dtype=bool)
    for _ in range(config.irls_iterations):
        xx = np.sum(weights*(1-u[:, :, 0]**2), axis=1)
        yy = np.sum(weights*(1-u[:, :, 1]**2), axis=1)
        xy = np.sum(-weights*u[:, :, 0]*u[:, :, 1], axis=1)
        b = np.sum(weights[:, :, None]*normal_p, axis=1)
        trace, discriminant = xx+yy, np.sqrt((xx-yy)**2+4*xy**2)
        high, low = (trace+discriminant)/2, np.maximum((trace-discriminant)/2, 0)
        ratio = divide_or_nan(low, high)
        determinant = xx*yy-xy**2
        geometry = (n >= config.min_moving) & (ratio >= config.min_geometry_ratio) & (determinant > 1e-10)
        q[:] = np.nan
        q[geometry, 0] = ((b[:, 0]*yy-b[:, 1]*xy)/np.maximum(determinant, 1e-30))[geometry]
        q[geometry, 1] = ((b[:, 1]*xx-b[:, 0]*xy)/np.maximum(determinant, 1e-30))[geometry]
        delta = q[:, None, :]-p
        residual = np.abs(u[:, :, 0]*delta[:, :, 1]-u[:, :, 1]*delta[:, :, 0])
        scale = np.minimum(1, config.huber_distance_m/np.maximum(residual, 1e-10))
        weights = np.where(moving & geometry[:, None], scale, 0)
    delta = q[:, None, :]-p
    distance = np.linalg.norm(delta, axis=2)
    along = np.sum(u*delta, axis=2)
    cosine = np.zeros_like(distance)
    np.divide(along, distance, out=cosine, where=moving & (distance > 1e-8) & geometry[:, None])
    residual = np.abs(u[:, :, 0]*delta[:, :, 1]-u[:, :, 1]*delta[:, :, 0])
    support = (moving & geometry[:, None] & (along > 0)
               & (cosine >= math.cos(math.radians(config.support_angle_deg)))
               & (residual <= config.support_distance_m))
    ns = support.sum(axis=1)
    fraction = divide_or_nan(ns, n)
    rms = np.sqrt(divide_or_nan(np.where(support, residual**2, 0).sum(axis=1), ns))
    forward_fraction = divide_or_nan((moving & (along > 0)).sum(axis=1), n)
    alignment = divide_or_nan(np.where(moving & geometry[:, None], np.maximum(cosine, 0), 0).sum(axis=1), n)
    inside = np.isfinite(q).all(axis=1) & ((q >= 0) & (q <= np.asarray(pitch))).all(axis=1)
    accepted = geometry & inside & (ns >= config.min_moving) & (fraction >= config.min_support_fraction)
    score = fraction*alignment*np.exp(-rms/config.support_distance_m)
    confidence = score*np.minimum(1, ratio/config.confidence_geometry_full)*np.minimum(1, n/config.confidence_moving_full)
    status = np.full(len(points), 'insufficient_moving', dtype='<U24')
    status[(n >= config.min_moving) & ~geometry] = 'parallel_or_unstable'
    status[geometry & ~inside] = 'outside_pitch'
    status[geometry & inside & ~accepted] = 'low_forward_support'
    status[accepted] = 'ok'
    eta = np.full(len(points), np.nan)
    for index in np.flatnonzero(accepted):
        eta[index] = np.median(along[index, support[index]]/speed[index, support[index]])
    return {'focus_x_m': np.where(accepted, q[:, 0], np.nan), 'focus_y_m': np.where(accepted, q[:, 1], np.nan),
            'convergence_score': np.where(accepted, score, np.nan),
            'focus_confidence': np.where(accepted, confidence, np.nan),
            'focus_status': status, 'n_moving': n, 'focus_support_count': ns,
            'focus_support_fraction': fraction, 'focus_forward_fraction': forward_fraction,
            'focus_fit_rmse_m': rms, 'focus_geometry_ratio': ratio,
            'focus_median_line_eta_s': eta,
            'direction_coherence': np.linalg.norm(divide_or_nan(u.sum(axis=1), n[:, None]), axis=1)}


def hull_area_rate(points, velocities=None):
    """CCW convex hull shoelace area and instantaneous derivative on that hull.

    This differentiates the CURRENT hull vertices with available velocities; it
    does not subtract areas from changing observed player sets across frames.
    """
    if len(points) < 3:
        return np.nan, np.nan
    unique = {}
    for index, p in enumerate(points):
        unique.setdefault((float(p[0]), float(p[1])), index)
    ordered = sorted(unique.values(), key=lambda i: (points[i, 0], points[i, 1]))
    coordinates = [(float(p[0]), float(p[1])) for p in points]
    def cross(a, b, c):
        ax, ay = coordinates[a]
        bx, by = coordinates[b]
        cx, cy = coordinates[c]
        return (bx-ax)*(cy-ay)-(by-ay)*(cx-ax)
    lower, upper = [], []
    for index in ordered:
        while len(lower) >= 2 and cross(lower[-2], lower[-1], index) <= 0:
            lower.pop()
        lower.append(index)
    for index in reversed(ordered):
        while len(upper) >= 2 and cross(upper[-2], upper[-1], index) <= 0:
            upper.pop()
        upper.append(index)
    indices = lower[:-1]+upper[:-1]
    if len(indices) < 3:
        return 0., np.nan
    p = points[indices]
    next_p = np.roll(p, -1, axis=0)
    area = .5*np.sum(p[:, 0]*next_p[:, 1]-p[:, 1]*next_p[:, 0])
    if velocities is None or len(unique) != len(points):
        return float(abs(area)), np.nan
    v = velocities[indices]
    next_v = np.roll(v, -1, axis=0)
    rate = .5*np.sum(v[:, 0]*next_p[:, 1]-v[:, 1]*next_p[:, 0]
                      +p[:, 0]*next_v[:, 1]-p[:, 1]*next_v[:, 0])
    return float(abs(area)), float(rate if area >= 0 else -rate)


def measure_team(points, velocities, ball, pitch, rejected_steps=None, config=MotionConfig()):
    points, velocities = np.asarray(points, dtype=float), np.asarray(velocities, dtype=float)
    observed = np.isfinite(points).all(axis=2) & ((points >= 0) & (points <= np.asarray(pitch))).all(axis=2)
    speeds = np.linalg.norm(velocities, axis=2)
    valid = observed & np.isfinite(velocities).all(axis=2) & (speeds <= config.max_step_speed_mps+1e-5)
    n = valid.sum(axis=1)
    p = np.where(valid[:, :, None], points, 0)
    v = np.where(valid[:, :, None], velocities, 0)
    centroid, mean_v = divide_or_nan(p.sum(axis=1), n[:, None]), divide_or_nan(v.sum(axis=1), n[:, None])
    metrics = robust_focus(points, np.where(valid[:, :, None], velocities, np.nan), pitch, config)
    metrics.update(n_observed=observed.sum(axis=1), n_valid_vectors=n,
                   n_stationary=(valid & (speeds < config.min_moving_speed_mps)).sum(axis=1),
                   n_step_speed_rejected=rejected_steps.sum(axis=1) if rejected_steps is not None else np.zeros(len(points), dtype=int),
                   mean_vx_mps=mean_v[:, 0], mean_vy_mps=mean_v[:, 1], mean_vector_speed_mps=np.linalg.norm(mean_v, axis=1),
                   centroid_x_m=centroid[:, 0], centroid_y_m=centroid[:, 1])
    a, b = np.triu_indices(points.shape[1], 1)
    d, dv = p[:, a]-p[:, b], v[:, a]-v[:, b]
    distance = np.linalg.norm(d, axis=2)
    pair_valid = valid[:, a] & valid[:, b] & (distance >= config.pair_min_distance_m)
    rate = np.zeros_like(distance)
    np.divide(np.sum(d*dv, axis=2), distance, out=rate, where=pair_valid)
    npairs = pair_valid.sum(axis=1)
    pair_rate = divide_or_nan(np.where(pair_valid, rate, 0).sum(axis=1), npairs)
    pair_mean = divide_or_nan(np.where(pair_valid, distance, 0).sum(axis=1), npairs)
    pair_rate[n < 3] = np.nan
    pair_mean[n < 3] = np.nan
    close = (valid[:, a] & valid[:, b] & (distance < config.pair_min_distance_m)).sum(axis=1)
    metrics.update(pair_distance_rate_mps=pair_rate, mean_pair_distance_m=pair_mean,
                   close_pair_count=close, pair_count=npairs)
    metrics['focus_confidence'] /= 1+np.divide(close, np.maximum(n, 1))
    observed_area, vector_area, area_rate = [np.full(len(points), np.nan) for _ in range(3)]
    for index in range(len(points)):
        observed_area[index], _ = hull_area_rate(points[index, observed[index]])
        vector_area[index], area_rate[index] = hull_area_rate(points[index, valid[index]], velocities[index, valid[index]])
    metrics.update(observed_hull_area_m2=observed_area, valid_vector_hull_area_m2=vector_area, hull_area_rate_m2ps=area_rate)
    focus = np.column_stack([metrics['focus_x_m'], metrics['focus_y_m']])
    ball_valid = np.isfinite(ball).all(axis=1) & ((ball >= 0) & (ball <= np.asarray(pitch))).all(axis=1)
    ball_distance = np.linalg.norm(focus-ball, axis=1)
    ball_distance[~ball_valid] = np.nan
    metrics['focus_to_ball_distance_m'] = ball_distance
    return metrics
