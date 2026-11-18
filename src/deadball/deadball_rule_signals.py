"""Reusable, masked evidence for the two dead-ball boundaries."""

from pathlib import Path
import sys

_SRC = Path(__file__).resolve().parents[1]
sys.path[:0] = [str(_SRC), str(_SRC.parent)]

import numpy as np
import pandas as pd
from scipy.ndimage import maximum_filter1d

import final_dead_balls as fd
from config import CONFIG

FPS = CONFIG["fps"]
CONTEXT_SECONDS = 25
CORNER_RADIUS_M = 1.0
TOUCHLINE_BAND_M = 1.0
GOAL_AREA_DEPTH_M = 5.5
GOAL_AREA_HALF_WIDTH_M = (7.32 / 2) + GOAL_AREA_DEPTH_M
LOCATION_FLAGS = ("ball_in_corner_area", "ball_near_touchline", "ball_in_goal_area")


def ball_location_flags(ball, pitch):
    """Independent location flags from observed ball centres in metres."""
    x, y = ball.T
    length, width = pitch
    observed = np.isfinite(ball).all(axis=1)
    dx, dy = np.minimum(x, length - x), np.minimum(y, width - y)
    inside = (dx >= 0) & (dy >= 0)
    corner = inside & (np.hypot(dx, dy) <= CORNER_RADIUS_M)
    touchline = (dx >= 0) & (np.abs(dy) <= TOUCHLINE_BAND_M)
    goal_area = inside & (dx <= GOAL_AREA_DEPTH_M) & (np.abs(y - width / 2) <= GOAL_AREA_HALF_WIDTH_M)
    return {name: np.where(observed, flag, np.nan).astype(np.float32)
            for name, flag in zip(LOCATION_FLAGS, (corner, touchline, goal_area), strict=True)}


def shift(values, frames):
    """Positive offsets read the past. Missing samples at either end stay missing."""
    if frames == 0:
        return values
    out = np.full(values.shape, np.nan, dtype=np.float64)
    offset = abs(frames)
    if offset < len(values):
        if frames > 0:
            out[offset:] = values[:-offset]
        else:
            out[:-offset] = values[offset:]
    return out


def recent_exits(ball, seen, crossings, pitch, touch_team):
    length, width = pitch
    frame = np.arange(len(ball))
    marks = np.full(len(ball), -1, dtype=np.int64)
    marks[crossings] = crossings
    last = np.maximum.accumulate(marks)
    previous = np.r_[-1, last[:-1]]
    age = (frame - previous) / FPS
    has_exit = (previous >= 0) & (age <= CONTEXT_SECONDS)
    past = ball[previous.clip(0)]
    beyond_end = np.maximum(-past[:, 0], past[:, 0] - length)
    beyond_side = np.maximum(-past[:, 1], past[:, 1] - width)
    observed_count = np.r_[0, np.cumsum(seen)]
    lo = np.maximum(0, frame - CONTEXT_SECONDS * FPS)
    fraction_seen = (observed_count[frame] - observed_count[lo]) / np.maximum(1, frame - lo)
    old_touch = touch_team[previous.clip(0)]
    return {
        "previous_exit_age_s": np.where(has_exit, age, np.nan),
        "previous_exit_goal_line": np.where(has_exit, beyond_end >= beyond_side, np.nan),
        "previous_exit_left_half": np.where(has_exit, past[:, 0] < length / 2, np.nan),
        "previous_exit_last_touch_team": np.where(has_exit, old_touch, np.nan),
        "previous_ball_coverage": fraction_seen,
    }


def context_table(match, signal, geo, crossings):
    """Generic signals plus common geometry and chronological context.

    Uses a single tracking half. No ball status, event timestamps or class labels enter.
    """
    n = len(match.ball)
    frame = np.arange(n)
    similar_crossings = {}
    for margin in (0.0, 0.25, 0.50):
        hit = np.zeros(n, dtype=np.int8)
        hit[signal[f"cross_{margin}"]] = 1
        similar_crossings[f"cross_nearby_{margin}"] = maximum_filter1d(hit, size=9, mode="constant")

    # These columns let the reference heads see the generic notebook features.
    generic = {name: np.asarray(signal[name], dtype=np.float32) for name in (
        "decel", "local", "fallback", "restart", "kick", "throw", "pass", "carry", "stop", "prep", "reliable",
    )}
    generic.update(similar_crossings)

    ball = match.ball
    x, y = ball.T
    length, width = match.pitch
    seen = geo["seen"]
    dx, dy = np.minimum(x, length - x), np.minimum(y, width - y)
    speed = signal["speed"]
    before, after = shift(speed, 5), shift(speed, -5)
    velocity_observed = np.isfinite(before) & np.isfinite(after)
    # Localise the potential restart, without passing generic frozen coordinates
    # as per-frame location features during the entire DFL stoppage.
    moving_on_launch = velocity_observed & (after >= CONFIG["final"]["line_out_moving_mps"])
    launch = moving_on_launch & ((after - before) >= 1.3)
    geometry = launch | signal["throw"] | signal["restart"]
    corner_distance = np.hypot(dx, dy)
    goal_area_distance = np.hypot(np.maximum(dx - GOAL_AREA_DEPTH_M, 0),
                                  np.maximum(np.abs(y - width / 2) - GOAL_AREA_HALF_WIDTH_M, 0))
    goal_area_distance[~seen] = np.nan

    sides = []
    for team in (0, 1):
        distance = np.linalg.norm(match.players[:, match.team == team] - ball[:, None], axis=-1)
        closest = np.where(np.isfinite(distance), distance, np.inf).min(axis=1)
        sides.append(np.where(np.isfinite(closest), closest, np.nan))
    home, away = sides
    with np.errstate(invalid="ignore"):
        nearest_team = np.where(np.isfinite(home) & np.isfinite(away) & (np.abs(home-away) >= 0.3),
                                np.where(home < away, 0, 1), np.nan)
        possible_touch = signal["reliable"] & (np.fmin(home, away) <= 1.1)
    contact = np.where(possible_touch & np.isfinite(nearest_team), frame, -1)
    last_contact = np.maximum.accumulate(contact)
    recent_touch = (last_contact >= 0) & (frame - last_contact <= 2 * FPS)
    touch_team = np.where(recent_touch, nearest_team[last_contact.clip(0)], np.nan)

    home_defends_left = np.nanmedian(match.players[:, match.team == 0, 0]) < np.nanmedian(
        match.players[:, match.team == 1, 0])
    kicker_defends_this_end = np.where(np.isfinite(nearest_team), (nearest_team == 0) == (
        (x < length / 2) == home_defends_left), np.nan)
    history = recent_exits(ball, seen, crossings, match.pitch, touch_team)
    extra = {
        "launch_observed": geometry,
        "kick_speed_gain": np.where(geometry, after - before, np.nan),
        "kick_prior_stationarity": np.where(geometry & velocity_observed, before <= 1.0, np.nan),
        "kick_nearest_player_m": np.where(geometry, np.fmin(home, away), np.nan),
        "kick_team_margin_m": np.where(geometry, np.abs(home - away), np.nan),
        "kick_from_defending_team": np.where(geometry & seen, kicker_defends_this_end, np.nan),
        "kick_same_team_as_exit_touch": np.where(
            geometry & np.isfinite(nearest_team) & np.isfinite(history["previous_exit_last_touch_team"]),
            nearest_team == history["previous_exit_last_touch_team"], np.nan,
        ),
        "kick_corner_distance_m": np.where(geometry, corner_distance, np.nan),
        "kick_goal_area_distance_m": np.where(geometry, goal_area_distance, np.nan),
        "kick_touchline_distance_m": np.where(geometry, np.abs(dy), np.nan),
        "kick_center_distance_m": np.where(geometry, np.hypot(x - length / 2, y - width / 2), np.nan),
        "kick_penalty_mark_distance_m": np.where(geometry, np.hypot(dx - 11.0, y - width / 2), np.nan),
        "out_goal_line_at_cross": np.full(n, np.nan),
        "out_touchline_at_cross": np.full(n, np.nan),
        **history,
    }
    extra["out_goal_line_at_cross"][crossings] = dx[crossings] <= dy[crossings]
    extra["out_touchline_at_cross"][crossings] = dy[crossings] < dx[crossings]
    # A preceding analytic frame can capture a kick/exit even if the current
    # frame is one of the neighbouring positive boundary-target frames.
    context = {f"{name}_{offset:+d}f": shift(values, offset)
               for name, values in {
                   "restart_candidate": signal["restart"], "stop_candidate": signal["stop"],
                   "corner_geometry": extra["kick_corner_distance_m"],
                   "exit_goal_line": extra["out_goal_line_at_cross"],
               }.items() for offset in (-4, 4)}
    return pd.DataFrame({**generic, **extra, **context, **ball_location_flags(ball, match.pitch)},
                        dtype=np.float32), list(generic)


def tracking_condition(match, ball, movement):
    tracked = type(match)(players=match.players, team=match.team, ball=ball,
                          pitch=match.pitch, period=match.period, frame=match.frame)
    geo = fd.ball_signals(tracked)
    crossings = fd.line_crossings(geo)
    signal = movement(tracked.players, tracked.team, tracked.ball, tracked.pitch)
    table, generic = context_table(tracked, signal, geo, crossings)
    return table, generic, (geo, crossings, [signal])
