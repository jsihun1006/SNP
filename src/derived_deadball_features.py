"""Additional tracking features used by the DFL dead-ball experiment."""

import warnings

import numpy as np
import pandas as pd

from config import CONFIG
from features import central_speed

FPS = CONFIG["fps"]


def divide(numerator, denominator):
    return np.divide(numerator, denominator, out=np.full(np.shape(numerator), np.nan), where=denominator > 0)


def median(values, axis):
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", RuntimeWarning)
        return np.nanmedian(values, axis=axis)


def lag(values, period, frames):
    shifted = np.full_like(values, np.nan)
    same_half = period[frames:] == period[:-frames]
    shifted[frames:][same_half] = values[:-frames][same_half]
    return shifted


def velocity(points, period):
    half_window = CONFIG["speed_half_window"]
    valid = np.isfinite(central_speed(points, period, half_window, CONFIG["max_player_speed"]))
    result = np.full_like(points, np.nan)
    result[half_window:-half_window] = (points[2 * half_window:] - points[:-2 * half_window]) * (
        FPS / (2 * half_window)
    )
    result[~valid] = np.nan
    return result


def geometry(points, previous, home):
    """Frame-aligned teammate and opponent distances, processed in chunks."""
    names = ("pair_rate", "opponent_median", "opponent_rate")
    result = {f"{side}_{name}": np.full(len(points), np.nan)
              for side in ("home", "away") for name in names}
    step_seconds = round(0.5 * FPS) / FPS
    for start in range(0, len(points), 1024):
        chunk = slice(start, start + 1024)
        current, past = points[chunk], previous[chunk]
        for side, mask in (("home", home), ("away", ~home)):
            own, other = current[:, mask], current[:, ~mask]
            old_own, old_other = past[:, mask], past[:, ~mask]
            i, j = np.triu_indices(own.shape[1], k=1)
            distances = np.linalg.norm(own[:, i] - own[:, j], axis=2)
            old_distances = np.linalg.norm(old_own[:, i] - old_own[:, j], axis=2)
            result[f"{side}_pair_rate"][chunk] = median((distances - old_distances) / step_seconds, axis=1)

            opposed = np.linalg.norm(own[:, :, None] - other[:, None], axis=3)
            old_opposed = np.linalg.norm(old_own[:, :, None] - old_other[:, None], axis=3)
            nearest = np.where(np.isfinite(opposed), opposed, np.inf).min(axis=2)
            nearest[~np.isfinite(nearest)] = np.nan
            result[f"{side}_opponent_median"][chunk] = median(nearest, axis=1)

            # Both observations of an opponent are needed to measure a change.
            common = np.isfinite(opposed) & np.isfinite(old_opposed)
            current_nearest = np.where(common, opposed, np.inf).min(axis=2)
            old_nearest = np.where(common, old_opposed, np.inf).min(axis=2)
            rate = np.full(current_nearest.shape, np.nan)
            np.subtract(current_nearest, old_nearest, out=rate, where=np.isfinite(current_nearest))
            result[f"{side}_opponent_rate"][chunk] = median(rate / step_seconds, axis=1)
    return result


def angle_cosine(first, second, min_speed=0.5):
    denominator = np.linalg.norm(first, axis=-1) * np.linalg.norm(second, axis=-1)
    moving = (np.linalg.norm(first, axis=-1) >= min_speed) & (np.linalg.norm(second, axis=-1) >= min_speed)
    return np.clip(divide((first * second).sum(axis=-1), np.where(moving, denominator, np.nan)), -1, 1)


def player_features(match):
    points = match.players.astype(np.float32)
    period = np.asarray(match.period)
    home = match.team == 0
    gap = round(0.5 * FPS)
    previous = lag(points, period, gap)
    vectors = velocity(points, period)
    speed = np.linalg.norm(vectors, axis=2)
    speed_change = speed - lag(speed, period, gap)
    result = geometry(points, previous, home)
    team_velocity, team_speed = [], []
    for side, mask in (("home", home), ("away", ~home)):
        changes = speed_change[:, mask]
        valid = np.isfinite(changes).sum(axis=1)
        result[f"{side}_decelerating_fraction"] = divide((changes < -0.5).sum(axis=1), valid)
        result[f"{side}_accelerating_fraction"] = divide((changes > 0.5).sum(axis=1), valid)
        sp = speed[:, mask]
        v = vectors[:, mask]
        moving = sp >= 0.5
        direction = np.where(moving[..., None], v / np.maximum(sp[..., None], 0.5), 0)
        result[f"{side}_direction_agreement"] = divide(
            np.linalg.norm(direction.sum(axis=1), axis=1), moving.sum(axis=1)
        )
        team_speed.append(median(sp, axis=1))
        team_velocity.append(divide(np.nansum(v, axis=1), np.isfinite(sp).sum(axis=1)[:, None]))

        own = points[:, mask]
        center = median(own, axis=1)
        result[f"{side}_center_distance_median"] = median(np.linalg.norm(own - center[:, None], axis=2), axis=1)
        x, y = own[..., 0], own[..., 1]
        observed = np.isfinite(own).all(axis=2)
        result[f"{side}_touchline_fraction"] = divide(
            (observed & (np.minimum(y, match.pitch[1] - y) <= 5)).sum(axis=1), observed.sum(axis=1)
        )
        result[f"{side}_goalline_fraction"] = divide(
            (observed & (np.minimum(x, match.pitch[0] - x) <= 5)).sum(axis=1), observed.sum(axis=1)
        )

    result["team_motion_cosine"] = angle_cosine(*team_velocity)
    valid_speeds = np.isfinite(team_speed).all(axis=0)
    low = np.where(valid_speeds, (team_speed[0] < 1.5) & (team_speed[1] < 1.5), np.nan)
    result["both_teams_slow_window"] = pd.Series(low).groupby(period, sort=False).transform(
        lambda values: values.rolling(2 * FPS, min_periods=FPS).mean()
    ).to_numpy()
    duration = np.zeros(len(low), dtype=np.float32)
    previous_period = None
    run = 0
    for frame, (half, state) in enumerate(zip(period, low)):
        if half != previous_period or state != 1:
            run = 0
        if state == 1:
            run += 1
        duration[frame] = run / FPS if np.isfinite(state) else np.nan
        previous_period = half
    result["both_teams_slow_duration"] = duration
    return pd.DataFrame(result, dtype=np.float32)
