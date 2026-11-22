"""Shared tracking I/O, frame masks and vector utilities for the committed workflow.

Only standard-library modules and numpy are imported. No data is read on import.
All source-derived results belong in the ignored local data/derived directory.
Functions retain the interfaces used by the existing diagnostic scripts.
"""

import csv
import math
from pathlib import Path
import xml.etree.ElementTree as ET

import numpy as np

FLIPS = ((1, 1), (-1, 1), (1, -1), (-1, -1))
HALVES = {"1st Half": 1, "2nd Half": 2}



def track_teams(meta_path):
    """Use the same home-first XML convention as the official baseline."""
    root = ET.parse(meta_path).getroot()
    ids = [team.get("id") for team in root.iter("Team")]
    if len(ids) != 2 or None in ids or len(set(ids)) != 2:
        raise ValueError("Expected exactly two distinct team IDs in metadata.")
    mapping = dict(zip(ids, ("H", "A")))
    teams = []
    for player in root.iter("Player"):
        team_id = player.get("teamId")
        if team_id not in mapping:
            raise ValueError("A track has an unknown team ID.")
        teams.append(mapping[team_id])
    if set(teams) != {"H", "A"}:
        raise ValueError("Expected track slots for both teams.")
    return teams


def read_tracking(path, teams, expected, start, count):
    """One streaming pass; append ball as the last coordinate slot."""
    positions = np.full((count, len(teams) + 1, 2), np.nan, dtype=np.float32)
    seen = 0
    with open(path, encoding="utf-8") as handle:
        for frame, line in enumerate(handle):
            parts = line.rstrip("\r\n").split(":")
            if len(parts) != 3 or frame >= expected or int(parts[0]) != frame + 1:
                raise ValueError("ATD indexing/format differs from its documented timeline.")
            if start <= frame < start + count:
                values = np.fromstring(parts[1].replace(";", ",") + "," + parts[2], sep=",")
                if values.size != (len(teams) + 1) * 3:
                    raise ValueError("ATD object count differs from metadata.")
                positions[frame - start] = values.reshape(-1, 3)[:, :2]
            seen += 1
    if seen != expected:
        raise ValueError("ATD frame count differs from documented length.")
    return positions


def reflected(ball, flip, pitch):
    return np.where(np.asarray(flip) > 0, ball, np.asarray(pitch, dtype=ball.dtype) - ball)


def load_label_mask(path, n_frames):
    active = np.zeros(n_frames, dtype=bool)
    halves = np.zeros(n_frames, dtype=np.int8)
    count = 0
    with Path(path).open(newline='', encoding='utf-8') as handle:
        reader = csv.DictReader(handle)
        if not {'frame', 'half_tag', 'label'}.issubset(reader.fieldnames or []):
            raise ValueError('Missing source label columns.')
        for frame, row in enumerate(reader):
            if frame >= n_frames or int(row['frame']) != frame:
                raise ValueError('Label frame indices must be complete and ordered.')
            tag, label = row['half_tag'], row['label']
            if tag not in ('', *HALVES) or label not in ('', 'H', 'A', 'D', 'X'):
                raise ValueError('Invalid half tag or possession label.')
            halves[frame] = HALVES.get(tag, 0)
            active[frame] = label != 'X'
            if active[frame] and not halves[frame]:
                raise ValueError('A non-X frame requires an explicit half tag.')
            count += 1
    if count != n_frames:
        raise ValueError('Incomplete source label timeline.')
    return active, halves


def read_mapping(path, n_tactical, n_broadcast):
    mapping = np.full(n_tactical, -1, dtype=np.int32)
    count, previous = 0, -1
    with Path(path).open(newline='', encoding='utf-8') as handle:
        for frame, row in enumerate(csv.DictReader(handle)):
            if frame >= n_tactical or int(row['tactical_frame']) != frame:
                raise ValueError('Invalid tactical mapping timeline.')
            if row['broadcast_atd_frame0']:
                value = int(row['broadcast_atd_frame0'])
                if not previous < value < n_broadcast:
                    raise ValueError('Invalid or reused broadcast mapping index.')
                mapping[frame], previous = value, value
            count += 1
    if count != n_tactical:
        raise ValueError('Incomplete frame mapping.')
    return mapping


def source_state(path, count):
    active, halves = load_label_mask(path, count)
    with Path(path).open(newline='', encoding='utf-8') as handle:
        labels = np.array([row['label'] for row in csv.DictReader(handle)], dtype='<U1')
    eligible = np.zeros(count, dtype=bool)
    eligible[1:] = active[1:] & active[:-1] & (halves[1:] == halves[:-1]) & (halves[1:] > 0)
    return {'active': active, 'halves': halves, 'labels': labels, 'eligible': eligible}


def trailing_mean(values, window):
    values = np.asarray(values, dtype=float)
    finite = np.isfinite(values)
    totals = np.r_[0., np.cumsum(np.where(finite, values, 0))]
    counts = np.r_[0, np.cumsum(finite)]
    end = np.arange(1, len(values)+1)
    start = np.maximum(0, end-window)
    n = counts[end]-counts[start]
    result = np.full(len(values), np.nan)
    usable = finite & (n >= math.ceil(window/2))
    np.divide(totals[end]-totals[start], n, out=result, where=usable)
    return result


def pack_source_frames(positions, frames, fps, pitch):
    """Float32 [x_norm,y_norm,vx_mps,vy_mps], unmapped=-1, last slot is ball.

    Derivatives use ORIGINAL adjacent source records, never adjacent displayed
    frames. A zero velocity is finite; an unavailable velocity is NaN.
    """
    frames = np.asarray(frames)
    positions = np.asarray(positions, dtype=np.float32)
    if (positions.ndim != 3 or positions.shape[2] != 2 or frames.ndim != 1
            or not np.issubdtype(frames.dtype, np.integer) or np.any(frames < -1)
            or np.any(frames >= len(positions)) or not np.isfinite(fps) or fps <= 0
            or len(pitch) != 2 or not np.isfinite(pitch).all() or np.any(np.asarray(pitch) <= 0)):
        raise ValueError('Invalid vector source positions, frames or units.')
    result = np.full((len(frames), positions.shape[1], 4), np.nan, dtype=np.float32)
    mapped = np.flatnonzero(frames >= 0)
    result[mapped, :, :2] = positions[frames[mapped]]
    adjacent = np.flatnonzero(frames > 0)
    current = positions[frames[adjacent], :-1]
    previous = positions[frames[adjacent]-1, :-1]
    valid = np.isfinite(current).all(axis=2) & np.isfinite(previous).all(axis=2)
    velocities = (current-previous)*np.asarray(pitch, dtype=np.float32)*fps
    velocities[~valid] = np.nan
    result[adjacent, :-1, 2:] = velocities
    return result
