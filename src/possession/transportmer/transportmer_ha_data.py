"""Match-grouped DFL frames and read-only MTAH ball-availability statistics."""

from collections import Counter
from dataclasses import dataclass
from pathlib import Path
from zlib import crc32

import numpy as np
import pandas as pd

import dfl_load
from features import match_from_dfl, runs_of

FPS, STEP, WINDOW = 25, 4, 60
TRAIN_STRIDE, INFER_STRIDE = 30, 15  # units of 6.25 Hz samples


def seed_for(seed, key):
    return np.random.SeedSequence([seed, crc32(key.encode("utf-8"))])


def split_matches(seed):
    matches = np.random.default_rng(seed).permutation(sorted(dfl_load.MATCHES)).tolist()
    holdout, *development = matches
    return {
        "seed": seed, "holdout": holdout, "development": development,
        "halves": [[match, half] for match in matches for half in (1, 2)],
        "folds": [{"oof": match, "train": [other for other in development if other != match]}
                  for match in development],
    }


def ball_profile(raw_path, dead_path):
    from games import GAME, half_video_blocks, source_frames

    n = source_frames("tactical")
    table = pd.read_csv(dead_path, usecols=["frame", "dead"])
    if not np.array_equal(table["frame"].to_numpy(), np.arange(n)) or not table["dead"].isin([0, 1]).all():
        raise ValueError("Existing dead-ball output must cover the tactical frame axis exactly.")
    seen = np.zeros(n, dtype=bool)
    present = np.zeros(n, dtype=bool)
    with Path(raw_path).open() as handle:
        for line in handle:
            fields = line.rstrip().split(":", 2)
            if len(fields) != 3:
                continue
            index = int(fields[0]) - 1
            if 0 <= index < n:
                if present[index]:
                    raise ValueError(f"Repeated tactical ATD frame {index}.")
                present[index] = True
                ball = fields[2].split(",")
                seen[index] = len(ball) >= 2 and np.isfinite(float(ball[0])) and np.isfinite(float(ball[1]))
    if not present.any():
        raise ValueError("Tactical ATD is empty or missing its tracking payload.")
    counts = Counter()
    in_play = missing = 0
    dead = table["dead"].to_numpy(dtype=bool)
    for _, first, last in half_video_blocks(GAME):
        alive = ~dead[first:last+1]
        gaps = alive & ~seen[first:last+1]
        in_play += int(alive.sum())
        missing += int(gaps.sum())
        counts.update(stop - start for start, stop in runs_of(gaps))
    if not in_play:
        raise ValueError("No predicted in-play frames to profile.")
    return {"atd": str(raw_path), "dead_source": str(dead_path), "fps": FPS,
            "in_play_frames": in_play, "missing_frames": missing,
            "missing_fraction": missing / in_play,
            "gap_lengths_frames": {str(k): v for k, v in sorted(counts.items())}}


@dataclass
class Half:
    match: str
    number: int
    frame: np.ndarray
    time: np.ndarray
    alive: np.ndarray
    labels: np.ndarray
    xy: np.ndarray
    kind: np.ndarray
    pitch: tuple
    episodes: list
    augmentation: dict

    @property
    def key(self):
        return f"{self.match}_H{self.number}"


def contiguous_runs(mask, frame):
    cuts = np.r_[0, np.flatnonzero(np.diff(frame) != 1) + 1, len(mask)]
    return [(int(first + a), int(first + b))
            for first, stop in zip(cuts[:-1], cuts[1:])
            for a, b in runs_of(mask[first:stop])]


def augment_ball(ball, episodes, profile, rng, interpolation_fraction):
    """Sample measured gap lengths on the original timeline, within supported observed runs."""
    result = ball.copy()
    if not episodes:
        return result, {"in_play": 0, "native_missing": 0, "masked": 0, "interpolated": 0,
                        "masked_runs": [], "interpolated_runs": []}
    play = np.zeros(len(ball), dtype=bool)
    for first, stop in episodes:
        play[first:stop] = True
    count = int(play.sum())
    known = np.isfinite(ball).all(axis=1)
    native = int((play & ~known).sum())
    extra = max(0, round(count * profile["missing_fraction"]) - native)
    interpolate = round(extra * interpolation_fraction / (1 - interpolation_fraction))
    available = [(first+a, first+b) for first, stop in episodes
                 for a, b in runs_of(known[first:stop])]
    histogram = profile["gap_lengths_frames"]
    lengths = np.array([int(n) for n in histogram], dtype=int)
    weights = np.array(list(histogram.values()), dtype=float)
    history = {"in_play": count, "native_missing": native, "masked": 0, "interpolated": 0,
               "target_added_mask": extra, "target_linear_replacement": interpolate}
    for mode, target in (("masked", extra), ("interpolated", interpolate)):
        history[mode + "_runs"] = []
        remaining = target
        while remaining and available and len(lengths):
            spans = np.array(available, dtype=int)
            capacity = spans[:, 1] - spans[:, 0] - 2  # leave both endpoint observations
            eligible = (lengths <= remaining) & (lengths <= capacity.max())
            if not eligible.any():
                break
            length = int(rng.choice(lengths[eligible], p=weights[eligible] / weights[eligible].sum()))
            choices = np.maximum(0, capacity - length + 1)
            slots = choices.cumsum()
            order = int(rng.integers(slots[-1]))
            index = int(np.searchsorted(slots, order, side="right"))
            origin, boundary = available.pop(index)
            first = origin + 1 + order - (int(slots[index-1]) if index else 0)
            stop = first + length
            available.extend([(origin, first), (stop, boundary)])
            if mode == "masked":
                result[first:stop] = np.nan
            else:
                ratio = np.arange(1, length+1)[:, None] / (length+1)
                result[first:stop] = ball[first-1] + ratio * (ball[stop] - ball[first-1])
            history[mode] += length
            history[mode + "_runs"].append(length)
            remaining -= length
    return result, history


def load_halves(ids, cache, profile, seed, interp_fraction):
    dfl_load.CONFIG["dfl_cache"] = str(cache)
    halves = []
    for match_id in ids:
        wanted = dfl_load.cache_paths(match_id)
        if not all(Path(path).is_file() for path in wanted):
            raise FileNotFoundError(f"Local DFL parquet cache required for {match_id}: {wanted}")
        tracking, _, meta = dfl_load.load_match(match_id)
        match = match_from_dfl(tracking, meta)
        for number in (1, 2):
            rows = match.period == number
            frame = match.frame[rows]
            time = tracking.loc[rows, "timestamp_s"].to_numpy()
            alive = tracking.loc[rows, "ball_state"].to_numpy() == 1
            if not len(frame) or np.any(np.diff(frame) <= 0) or np.any(np.diff(time) <= 0):
                raise ValueError(f"Invalid original frame axis for {match_id}, half {number}")
            labels = tracking.loc[rows, "ball_owning_team"].map({"home": 0, "away": 1}).fillna(-1).to_numpy(np.int8)
            labels[~alive] = 2
            episodes = contiguous_runs(alive, frame)
            if not np.isin(labels[alive], (0, 1)).any():
                raise ValueError(f"No H/A supervision for {match_id}, half {number}")
            pitch = np.asarray(match.pitch, dtype=np.float32)
            players = match.players[rows] / pitch
            active = np.isfinite(players[alive]).all(axis=-1).any(axis=0)
            ball, report = augment_ball(match.ball[rows] / pitch, episodes, profile,
                                        np.random.default_rng(seed_for(seed, f"{match_id}:H{number}")), interp_fraction)
            xy = np.concatenate([players[:, active], ball[:, None]], axis=1)
            # A partially missing coordinate pair carries neither position nor velocity.
            xy[~np.isfinite(xy).all(axis=-1)] = np.nan
            halves.append(Half(match_id, number, frame, time, alive, labels, xy,
                               np.r_[match.team[active], 2], match.pitch, episodes, report))
    return halves


def episode_starts(first, stop, stride):
    final = max(0, (stop-first-1)//STEP - WINDOW + 1)
    return first + STEP*np.unique(np.r_[np.arange(0, final, stride), final])


def windows(halves, training):
    result = []
    for i, half in enumerate(halves):
        for first, stop in half.episodes:
            for start in episode_starts(first, stop, TRAIN_STRIDE if training else INFER_STRIDE):
                if not training or np.isin(half.labels[start:min(stop, start + STEP*WINDOW):STEP], (0, 1)).any():
                    result.append((i, int(start), first, stop))
    return result


def collate(halves, batch):
    length = max(min(WINDOW, (stop-start-1)//STEP+1) for _, start, _, stop in batch)
    shape = (len(batch), length)
    index = np.stack([start + np.arange(length)*STEP for _, start, _, _ in batch])
    pad = index >= np.array([stop for _, _, _, stop in batch])[:, None]
    # Completely unobserved player slots are masked by the backbone at every
    # social block. Omit them before batching, retaining the ball token.
    active = []
    for i, (half_index, _, _, _) in enumerate(batch):
        keep = np.isfinite(halves[half_index].xy[index[i, ~pad[i]]]).all(axis=-1).any(axis=0)
        keep[-1] = True
        active.append(np.flatnonzero(keep))
    slots = max(map(len, active))
    xy = np.zeros((*shape, slots, 2), dtype=np.float32)
    motion = np.zeros((*shape, slots, 3), dtype=np.float32)
    seen = np.zeros((*shape, slots), dtype=bool)
    kinds = np.zeros((len(batch), slots), dtype=np.int64)
    restart = np.zeros(shape, dtype=np.int64)
    labels = np.full(shape, -1, dtype=np.int8)
    for i, (half_index, start, first, _) in enumerate(batch):
        half = halves[half_index]
        current = index[i, ~pad[i]]
        keep = active[i]
        n, k = len(current), len(keep)
        data = half.xy[current[:, None], keep]
        visible = np.isfinite(data).all(axis=-1)
        xy[i, :n, :k] = np.where(visible[..., None], data, 0)
        seen[i, :n, :k] = visible
        kinds[i, :k] = half.kind[keep]
        labels[i, :n] = half.labels[current]
        restart[i, 0] = start == first and first > 0 and not half.alive[first-1] and half.frame[first] == half.frame[first-1]+1
        if n > 1:
            src = current[1:, None] + np.arange(-STEP, 1)
            valid = np.isfinite(half.xy[src[:, :, None], keep]).all(axis=(1, 3))
            elapsed = np.diff(half.time[current]).astype(np.float32)
            motion[i, 1:n, :k, :2] = np.where(valid[..., None], np.diff(data, axis=0) / elapsed[:, None, None], 0)
            motion[i, 1:n, :k, 2] = valid
    return (xy, seen, kinds, motion, pad, restart, labels), index
