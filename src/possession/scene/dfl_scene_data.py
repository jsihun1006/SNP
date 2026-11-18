"""Build 32 scene-level tracking features and align DFL training targets."""

from dataclasses import dataclass
from pathlib import Path
import warnings

import numpy as np
import pandas as pd

from features import match_from_dfl, nan_percentile
from transportmer_ha_data import contiguous_runs

STEP = 5
WINDOW = 50
STRIDE = 25
EVENTS = ("foul", "ball_out", "restart")
BOUNDARIES = ("home_to_away", "away_to_home", "dead_start", "restart")
TEAM = {"home": 0, "away": 1}
RESTARTS = {"throw_in", "goalkick", "corner_short", "corner_crossed", "freekick_short",
            "freekick_crossed", "shot_freekick", "shot_penalty"}
TEAM_FEATURES = ("center_x", "center_y", "width_x", "width_y", "mean_speed",
                 "observed_count", "nearest_to_ball", "within_3m_of_ball")
FEATURE_NAMES = (
    "ball_x", "ball_y", "ball_vx", "ball_vy", "ball_speed", "ball_acceleration",
    "ball_border_distance", "ball_observed", "ball_velocity_observed", "ball_acceleration_observed",
    *(f"{side}_{feature}" for side in ("home", "away") for feature in TEAM_FEATURES),
    "team_center_distance", "nearest_opponent_pair", "opponent_pairs_within_2m",
    "home_minus_away_ball_distance", "slow_player_share", "decelerating_player_share",
)


@dataclass
class Trajectories:
    xy: np.ndarray
    motion: np.ndarray
    observed: np.ndarray
    kinds: np.ndarray


@dataclass
class Half:
    match: str
    period: int
    frame: np.ndarray
    truth: np.ndarray
    features: np.ndarray
    events: dict
    trajectories: Trajectories | None = None

    @property
    def key(self):
        return f"{self.match}_H{self.period}"


@dataclass
class Scene:
    half: Half
    begin: int
    end: int
    samples: np.ndarray
    inputs: np.ndarray
    states: np.ndarray
    boundaries: np.ndarray
    events: np.ndarray


def derivative(values, frames, present, times):
    result = np.full_like(values, np.nan)
    if len(values) <= STEP:
        return result
    bad = np.cumsum(~present, axis=0)
    valid = present[:-STEP] & (bad[STEP:] == bad[:-STEP])
    valid &= (frames[STEP:] - frames[:-STEP] == STEP).reshape((-1,) + (1,) * (valid.ndim - 1))
    elapsed = (times[STEP:] - times[:-STEP]).reshape((-1,) + (1,) * (values.ndim - 1))
    with np.errstate(divide="ignore", invalid="ignore"):
        speed = (values[STEP:] - values[:-STEP]) / elapsed
    result[STEP:] = np.where(valid[..., None] & (elapsed > 0), speed, np.nan)
    return result


def scene_features(ball, players, team, pitch, frames, times):
    ball_seen = np.isfinite(ball).all(axis=-1)
    players_seen = np.isfinite(players).all(axis=-1)
    velocity = derivative(ball, frames, ball_seen, times)
    velocity_seen = np.isfinite(velocity).all(axis=-1)
    acceleration = derivative(velocity, frames, velocity_seen, times)
    speed = np.linalg.norm(velocity, axis=-1)
    player_v = derivative(players, frames, players_seen, times)
    player_speed = np.linalg.norm(player_v, axis=-1)
    player_speed[player_speed > 12] = np.nan
    player_delta = derivative(player_speed[..., None], frames, np.isfinite(player_speed), times)[..., 0]
    border = np.min(np.c_[ball, np.asarray(pitch) - ball], axis=1)
    columns = [ball[:, 0], ball[:, 1], velocity[:, 0], velocity[:, 1], speed,
               np.linalg.norm(acceleration, axis=-1), border, ball_seen, velocity_seen,
               np.isfinite(acceleration).all(axis=-1)]
    centers, nearest = [], []
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", RuntimeWarning)
        for code in (0, 1):
            member = team == code
            coords = players[:, member]
            center = np.nanmean(coords, axis=1)
            x, y = coords[..., 0], coords[..., 1]
            widths = [nan_percentile(points, 90) - nan_percentile(points, 10) for points in (x, y)]
            dist = np.linalg.norm(coords - ball[:, None], axis=-1)
            closest = np.nanmin(dist, axis=1)
            nearby = np.where(ball_seen, (dist <= 3).sum(axis=1), np.nan)
            centers.append(center)
            nearest.append(closest)
            columns.extend([*center.T, *widths, np.nanmean(player_speed[:, member], axis=1),
                            players_seen[:, member].sum(axis=1), closest, nearby])
        distance, pairs = [], []
        for i in range(0, len(ball), 4096):
            coords = players[i:i+4096]
            p, q = coords[:, team == 0], coords[:, team == 1]
            d = np.linalg.norm(p[:, :, None] - q[:, None, :], axis=-1)
            distance.append(np.nanmin(d, axis=(1, 2)))
            pairs.append((d <= 2).sum(axis=(1, 2)))
    valid_speed = np.isfinite(player_speed).sum(axis=1)
    valid_accel = np.isfinite(player_delta).sum(axis=1)
    columns.extend([np.linalg.norm(centers[0]-centers[1], axis=1), np.concatenate(distance),
                    np.concatenate(pairs), nearest[0]-nearest[1],
                    np.divide((player_speed < 1).sum(axis=1), np.maximum(1, valid_speed)),
                    np.divide((player_delta < -1).sum(axis=1), np.maximum(1, valid_accel))])
    assert len(columns) == len(FEATURE_NAMES) == 32
    return np.column_stack(columns).astype(np.float32)


def tracking_trajectories(ball, players, team, pitch, frames, times):
    xy = np.concatenate((players, ball[:, None]), axis=1)
    observed = np.isfinite(xy).all(axis=-1)
    velocity = derivative(xy, frames, observed, times)
    valid = np.isfinite(velocity).all(axis=-1)
    motion = np.concatenate((np.where(valid[..., None], velocity / 10, 0), valid[..., None]), axis=-1)
    xy = np.where(observed[..., None], xy / np.asarray(pitch, dtype=np.float32), 0)
    return Trajectories(xy, motion, observed, np.r_[team, 2].astype(np.int8))


def load_halves(cache, upstream, match, *, individual_tracks=False):
    file = Path(cache)
    track = pd.read_parquet(file / f"{match}_tracking.parquet")
    roster = pd.read_parquet(file / f"{match}_meta.parquet")
    raw = pd.read_parquet(file / f"{match}_events.parquet")
    synced = pd.read_parquet(upstream / "data/sportec_synced" / f"{match}.parquet")
    geometry = match_from_dfl(track, roster)
    halves = []
    for period in (1, 2):
        keep = geometry.period == period
        rows = track.loc[keep]
        frames = geometry.frame[keep]
        times = rows["timestamp_s"].to_numpy()
        if not len(frames) or np.any(np.diff(frames) <= 0) or np.any(np.diff(times) <= 0):
            raise ValueError(f"Invalid frame alignment for {match} H{period}")
        coordinates = scene_features(geometry.ball[keep], geometry.players[keep], geometry.team,
                                     geometry.pitch, frames, times)
        labels = rows["ball_owning_team"].map(TEAM).fillna(-1).to_numpy(dtype=np.int8)
        labels[~rows["ball_state"].eq(1).to_numpy()] = 2
        events = raw.loc[raw["period_id"].eq(period)]
        aligned = synced.loc[synced["period_id"].eq(period) & synced["frame_id"].notna()]

        def nearest_frame(event_times):
            positions = np.searchsorted(times, event_times).clip(1, len(times)-1)
            close = np.abs(times[positions-1]-event_times) <= np.abs(times[positions]-event_times)
            positions = np.where(close, positions-1, positions)
            return frames[positions[np.abs(times[positions]-event_times) <= 0.12]]

        foul = aligned.loc[aligned["spadl_type"].eq("foul"), "frame_id"].to_numpy(dtype=np.int64)
        out = nearest_frame(events.loc[events["event_type"].eq("BALL_OUT"), "timestamp_s"].dropna().to_numpy())
        restarts = aligned.loc[aligned["spadl_type"].isin(RESTARTS), "frame_id"].to_numpy(dtype=np.int64)
        kickoff = nearest_frame(events.loc[events["set_piece_type"].eq("KICK_OFF"), "timestamp_s"].dropna().to_numpy())
        events = {"foul": np.unique(foul), "ball_out": np.unique(out),
                  "restart": np.unique(np.r_[restarts, kickoff])}
        if not all(np.isin(event_frames, frames).all() for event_frames in events.values()):
            raise ValueError(f"Event frames do not align with {match} H{period}")
        trajectories = tracking_trajectories(geometry.ball[keep], geometry.players[keep], geometry.team,
                                             geometry.pitch, frames, times) if individual_tracks else None
        halves.append(Half(match, period, frames, labels, coordinates, events, trajectories))
    return halves


def make_scenes(halves, scope):
    scenes = []
    for half in halves:
        valid = np.ones(len(half.frame), dtype=bool)
        if scope == "inplay":
            valid = half.truth < 2
        for start, stop in contiguous_runs(valid, half.frame):
            rows = np.arange(start, stop, STEP)
            if rows[-1] != stop-1:
                rows = np.r_[rows, stop-1]
            truth = half.truth[start:stop]
            boundary = np.zeros((len(rows), 4 if scope == "full" else 2), dtype=np.float32)
            boundary[0] = -1
            known = np.r_[0, np.cumsum(truth < 0)]
            invalid = known[rows[1:]-start+1] != known[rows[:-1]-start]
            boundary[1:][invalid] = -1
            changed = np.flatnonzero((truth[1:] != truth[:-1]) & (truth[1:] >= 0) & (truth[:-1] >= 0)) + 1
            for k in changed:
                old, new = truth[k-1:k+1]
                channel = 2 if new == 2 else 3 if old == 2 else int(new == 0)
                i = np.searchsorted(rows, start+k)
                if boundary[i, channel] >= 0:
                    boundary[i, channel] = 1
            tags = np.zeros((len(rows), len(EVENTS)), dtype=np.float32)
            for j, event in enumerate(EVENTS):
                frame_ids = half.events[event]
                visible = frame_ids[(frame_ids >= half.frame[start]) & (frame_ids <= half.frame[stop-1])]
                where = np.searchsorted(half.frame[rows], visible)
                tags[where, j] = 1
            inputs = half.features[rows].copy()
            early = rows-start < STEP
            inputs[np.ix_(early, [2, 3, 4, 14, 22, 30])] = np.nan
            inputs[early, 8] = 0
            early_acceleration = rows-start < 2*STEP
            inputs[np.ix_(early_acceleration, [5, 31])] = np.nan
            inputs[early_acceleration, 9] = 0
            scenes.append(Scene(half, start, stop, rows, inputs, half.truth[rows], boundary, tags))
    return scenes
