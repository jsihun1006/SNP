"""Turn tracking and smart tags into the model's per-frame features.

The design is deliberately small.  The HMM only needs to know how close the ball is to each team,
whether it is out of play, how long it has been missing, where it was last seen, and whether a
smart tag weakly supports a team.  Player identities are not available in the ATD, so the model
never pretends to know them.
"""

from __future__ import annotations

from dataclasses import dataclass
import warnings

import numpy as np

from .config import FPS, PITCH
from .data import TagTable, TrackingData


@dataclass(frozen=True)
class Features:
    start: int
    stop: int
    present: np.ndarray
    d_home: np.ndarray
    d_away: np.ndarray
    near_distance: np.ndarray
    near_team: np.ndarray
    near_player: np.ndarray
    second_distance: np.ndarray
    margin: np.ndarray
    near_speed: np.ndarray
    ball_speed: np.ndarray
    ball_acceleration: np.ndarray
    gap: np.ndarray
    edge: np.ndarray
    last_edge: np.ndarray
    out_of_play: np.ndarray
    last_touch: np.ndarray
    last_touch_age: np.ndarray
    d_home_last: np.ndarray
    d_away_last: np.ndarray
    tag_vote: np.ndarray
    tag_setpiece: np.ndarray
    tag_coverage: np.ndarray
    player_count: np.ndarray

    def __len__(self) -> int:
        return int(self.stop - self.start)


def _empty_tags(n_frames: int) -> TagTable:
    return TagTable.empty(n_frames)


def extract_features(
    tracking: TrackingData,
    tags: TagTable | None = None,
    observed_mask: np.ndarray | None = None,
    start: int = 0,
    stop: int | None = None,
) -> Features:
    """Extract features for ``[start, stop)``.

    ``observed_mask`` is how stress tests remove the ball without editing the tracking file.  A
    masked frame is treated exactly like a frame where the ATD has no ball: the gap grows, the last
    known position stays put, and the emission becomes the missing-ball emission.
    """
    n_total = tracking.n_frames
    start = int(start)
    stop = n_total if stop is None else int(stop)
    if start < 0 or stop > n_total or stop <= start:
        raise ValueError(f"invalid feature window [{start}, {stop}) for {n_total} frames")
    if tags is None:
        tags = _empty_tags(n_total)
    if tags.n_frames != n_total:
        raise ValueError(f"tag table has {tags.n_frames} frames; tracking has {n_total}")

    if observed_mask is None:
        observed = np.ones(n_total, dtype=bool)
    else:
        observed = np.asarray(observed_mask, dtype=bool)
        if observed.shape != (n_total,):
            raise ValueError("observed_mask must have one boolean per frame")
    present = tracking.ball_present[start:stop] & observed[start:stop]
    n = stop - start

    ball = tracking.ball[start:stop]
    players = tracking.players[start:stop]
    speeds = tracking.speeds[start:stop]
    teams = tracking.teams
    length, width = tracking.pitch

    # (n, P) distances from the ball to each player slot.
    dx = players[:, :, 0] - ball[:, None, 0]
    dy = players[:, :, 1] - ball[:, None, 1]
    distance = np.sqrt(dx * dx + dy * dy)

    safe = np.where(np.isfinite(distance), distance, np.inf)
    near_player = np.argmin(safe, axis=1).astype(np.int16)
    near_distance = safe[np.arange(n), near_player].astype(np.float32)
    no_player = ~np.isfinite(near_distance)
    near_distance[no_player] = np.nan
    near_player[no_player] = -1
    near_team = np.where(
        no_player,
        -1,
        teams[np.clip(near_player, 0, max(0, len(teams) - 1))],
    ).astype(np.int8)
    near_team[~present] = -1
    near_player[~present] = -1
    near_distance[~present] = np.nan

    near_speed = np.full(n, np.nan, dtype=np.float32)
    valid_near = present & (near_player >= 0)
    if valid_near.any():
        near_speed[valid_near] = speeds[
            np.flatnonzero(valid_near), near_player[valid_near]
        ]

    # Second-nearest player, useful for a "cluster around the ball" diagnostic.
    d2 = distance.copy()
    d2[np.arange(n), np.clip(near_player, 0, max(0, d2.shape[1] - 1))] = np.inf
    second_distance = np.nanmin(d2, axis=1)
    second_distance[~np.isfinite(second_distance)] = np.nan

    home_mask = teams == 0
    away_mask = teams == 1
    with warnings.catch_warnings(), np.errstate(all="ignore"):
        warnings.simplefilter("ignore", RuntimeWarning)
        d_home = np.nanmin(distance[:, home_mask], axis=1) if home_mask.any() else np.full(n, np.nan)
        d_away = np.nanmin(distance[:, away_mask], axis=1) if away_mask.any() else np.full(n, np.nan)
    d_home = np.asarray(d_home, dtype=np.float32)
    d_away = np.asarray(d_away, dtype=np.float32)
    d_home[~present] = np.nan
    d_away[~present] = np.nan
    margin = (second_distance - near_distance).astype(np.float32)

    # Ball speed and acceleration, in metres/second, only across consecutive observations.
    ball_speed = np.full(n, np.nan, dtype=np.float32)
    if n > 1:
        step = np.sqrt(np.diff(ball[:, 0]) ** 2 + np.diff(ball[:, 1]) ** 2) * FPS
        ok = present[1:] & present[:-1]
        ball_speed[1:] = np.where(ok, step, np.nan)
    ball_acceleration = np.full(n, np.nan, dtype=np.float32)
    if n > 1:
        ball_acceleration[1:] = np.diff(ball_speed)

    # Edge distance, last known edge, and gap since the last observation.  These loops are
    # intentionally simple: 199k iterations are cheap and the sequential state is the point.
    edge = np.minimum(
        np.minimum(ball[:, 0], length - ball[:, 0]),
        np.minimum(ball[:, 1], width - ball[:, 1]),
    ).astype(np.float32)
    edge[~present] = np.nan
    last_edge = np.full(n, np.nan, dtype=np.float32)
    gap = np.zeros(n, dtype=np.int32)
    last_seen = -1
    last_edge_value = np.nan
    for k in range(n):
        if present[k]:
            last_edge_value = edge[k]
            last_seen = k
            gap[k] = 0
        elif last_seen >= 0:
            gap[k] = k - last_seen
        else:
            gap[k] = k + 1
        last_edge[k] = last_edge_value

    out_of_play = present & (
        (ball[:, 0] < 0)
        | (ball[:, 0] > length)
        | (ball[:, 1] < 0)
        | (ball[:, 1] > width)
    )

    # Last touch: nearest player within 0.8 m.  This is not an identity, just a team hint.
    last_touch = np.full(n, -1, dtype=np.int8)
    last_touch_age = np.full(n, 9999, dtype=np.int32)
    touch = -1
    touch_frame = -1
    for k in range(n):
        if present[k] and near_team[k] >= 0 and np.isfinite(near_distance[k]) and near_distance[k] <= 0.8:
            touch = int(near_team[k])
            touch_frame = k
        last_touch[k] = touch
        last_touch_age[k] = k - touch_frame if touch_frame >= 0 else 9999

    # Per-team distance to the last known ball position.  This is the long-gap evidence: after a
    # tracking hole, the players have moved, but the team shape around the old ball is still a
    # useful prior.
    last_ball_x = np.nan
    last_ball_y = np.nan
    d_home_last = np.full(n, np.nan, dtype=np.float32)
    d_away_last = np.full(n, np.nan, dtype=np.float32)
    for k in range(n):
        if present[k]:
            last_ball_x = float(ball[k, 0])
            last_ball_y = float(ball[k, 1])
        if not (np.isfinite(last_ball_x) and np.isfinite(last_ball_y)):
            continue
        ddx = players[k, :, 0] - last_ball_x
        ddy = players[k, :, 1] - last_ball_y
        dist_last = np.sqrt(ddx * ddx + ddy * ddy)
        with warnings.catch_warnings(), np.errstate(all="ignore"):
            warnings.simplefilter("ignore", RuntimeWarning)
            if home_mask.any():
                d_home_last[k] = np.nanmin(dist_last[home_mask])
            if away_mask.any():
                d_away_last[k] = np.nanmin(dist_last[away_mask])

    return Features(
        start=start,
        stop=stop,
        present=present,
        d_home=d_home,
        d_away=d_away,
        near_distance=near_distance,
        near_team=near_team,
        near_player=near_player,
        second_distance=second_distance,
        margin=margin,
        near_speed=near_speed,
        ball_speed=ball_speed,
        ball_acceleration=ball_acceleration,
        gap=gap,
        edge=edge,
        last_edge=last_edge,
        out_of_play=out_of_play,
        last_touch=last_touch,
        last_touch_age=last_touch_age,
        d_home_last=d_home_last,
        d_away_last=d_away_last,
        tag_vote=tags.team_vote[start:stop].astype(np.int16),
        tag_setpiece=tags.setpiece[start:stop].astype(np.int8),
        tag_coverage=tags.coverage[start:stop].astype(bool),
        player_count=tracking.player_present[start:stop].sum(axis=1).astype(np.int16),
    )


def empty_features(n_frames: int) -> Features:
    """Small helper for smoke tests and visual placeholders."""
    tracking = TrackingData(
        ball=np.full((n_frames, 2), np.nan, np.float32),
        ball_present=np.zeros(n_frames, bool),
        players=np.full((n_frames, 1, 2), np.nan, np.float32),
        speeds=np.full((n_frames, 1), np.nan, np.float32),
        player_present=np.zeros((n_frames, 1), bool),
        teams=np.array([0], dtype=np.int8),
        pitch=PITCH,
        n_frames=n_frames,
    )
    return extract_features(tracking)
