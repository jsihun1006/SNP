"""Read-only player observations indexed by original video frame, not identity."""
from __future__ import annotations

import hashlib
import json
import os
from pathlib import Path
import tempfile

import numpy as np

from .data_quality import PlayerFrame, iter_player_frames, read_metadata


def _integral(values, name):
    a = np.asarray(values)
    if a.ndim != 1 or not np.isfinite(a).all() or not np.equal(a, a.astype(np.int64)).all():
        raise ValueError(f"{name} must contain finite integers")
    return a.astype(np.int64, copy=True)


class FrameStore:
    """Raw source-slot arrays. A missing RECORD splits runs; missing players do not.

    Every array is read-only. Source-slot columns are never temporal person IDs.
    Gaps are supported for explicitly supplied stores; the ATD cache builder keeps
    the previous stage's strict full-record validation and rejects damaged files.
    """

    def __init__(self, frame_ids, periods, xy_m, team_codes, pitch, *, fps=25,
                 parse_missing=None, parse_partial=None, parse_invalid=None, fingerprint=None):
        ids = _integral(frame_ids, "frame_ids")
        period = _integral(periods, "periods")
        teams = _integral(team_codes, "team_codes")
        xy = np.asarray(xy_m)
        if (fps != 25 or not len(ids) or (ids < 0).any() or (np.diff(ids) <= 0).any()
                or period.shape != ids.shape or not np.isin(period, [0, 1, 2]).all()
                or xy.shape != (len(ids), len(teams), 2) or not np.isin(teams, [0, 1]).all()
                or not len(teams) or not np.issubdtype(xy.dtype, np.floating)):
            raise ValueError("Invalid player frame store contract")
        if len(pitch) != 2 or not np.isfinite(pitch).all() or min(pitch) <= 0:
            raise ValueError("Invalid pitch")
        self.frame_ids, self.periods = ids, period.astype(np.int8)
        self.xy_m, self.team_codes = xy, teams.astype(np.int8)
        self.pitch, self.fps = tuple(float(x) for x in pitch), 25
        self.parse_missing = np.asarray(np.isnan(xy).all(axis=-1) if parse_missing is None else parse_missing, dtype=bool)
        self.parse_partial = np.asarray((np.isnan(xy).any(axis=-1) & ~self.parse_missing)
                                        if parse_partial is None else parse_partial, dtype=bool)
        self.parse_invalid = np.asarray(np.isinf(xy).any(axis=-1) if parse_invalid is None else parse_invalid, dtype=bool)
        for array in (self.frame_ids, self.periods, self.xy_m, self.team_codes,
                      self.parse_missing, self.parse_partial, self.parse_invalid):
            if array.ndim == 2 and array.shape != xy.shape[:2]:
                raise ValueError("Invalid source-slot flag shape")
            array.setflags(write=False)
        if any(a.shape != xy.shape[:2] for a in (self.parse_missing, self.parse_partial, self.parse_invalid)):
            raise ValueError("Invalid source-slot flag shape")
        if fingerprint is None:
            h = hashlib.sha256()
            for a in (ids, period, teams, xy, self.parse_missing, self.parse_partial, self.parse_invalid):
                h.update(np.ascontiguousarray(a).tobytes())
            h.update(str(self.pitch).encode())
            fingerprint = h.hexdigest()
        self.fingerprint = fingerprint
        breaks = np.r_[True, (np.diff(ids) != 1) | (np.diff(period) != 0)]
        starts = np.flatnonzero(breaks)
        ends = np.r_[starts[1:]-1, len(ids)-1]
        self.runs = tuple((int(period[a]), int(a), int(b)) for a, b in zip(starts, ends) if period[a] > 0)

    def __len__(self):
        return len(self.frame_ids)

    def position(self, frame_id):
        row = int(np.searchsorted(self.frame_ids, frame_id))
        if row == len(self) or self.frame_ids[row] != frame_id:
            raise KeyError("Requested original frame is not recorded")
        return row

    def frame(self, row):
        if not 0 <= row < len(self):
            raise IndexError("Frame-store row outside source")
        return PlayerFrame(int(self.frame_ids[row]), int(self.periods[row]), self.xy_m[row],
                           self.team_codes, self.parse_missing[row], self.parse_partial[row], self.parse_invalid[row])


def _sha256(path):
    h = hashlib.sha256()
    with Path(path).open("rb") as source:
        for chunk in iter(lambda: source.read(1024*1024), b""):
            h.update(chunk)
    return h.hexdigest()


def build_frame_cache(*, meta_path, raw_path, output_dir, pitch, n_frames, blocks,
                      expected_slots=46, progress=False):
    """Strict raw ATD -> NEW local cache. No ball/GT/tag or derived CSV inputs."""
    output = Path(output_dir).resolve()
    if output.exists():
        raise FileExistsError("Frame cache already exists; refusing overwrite")
    meta = read_metadata(meta_path, expected_fps=25, expected_slots=expected_slots)
    before = [(Path(p).stat().st_size, Path(p).stat().st_mtime_ns) for p in (meta_path, raw_path)]
    hashes = [_sha256(p) for p in (meta_path, raw_path)]
    fingerprint = hashlib.sha256(json.dumps([hashes, list(pitch), n_frames]).encode()).hexdigest()
    output.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.TemporaryDirectory(prefix=".player_cache_", dir=output.parent) as temp:
        dest = Path(temp)
        shape = (n_frames, len(meta.team_codes))
        arrays = {
            "frame_ids": np.lib.format.open_memmap(dest/"frame_ids.npy", mode="w+", dtype="int64", shape=(n_frames,)),
            "periods": np.lib.format.open_memmap(dest/"periods.npy", mode="w+", dtype="int8", shape=(n_frames,)),
            "xy_m": np.lib.format.open_memmap(dest/"xy_m.npy", mode="w+", dtype="float64", shape=shape+(2,)),
        }
        for name in ("parse_missing", "parse_partial", "parse_invalid"):
            arrays[name] = np.lib.format.open_memmap(dest/f"{name}.npy", mode="w+", dtype="bool", shape=shape)
        for row, frame in enumerate(iter_player_frames(raw_path, meta, pitch, n_frames, blocks)):
            arrays["frame_ids"][row], arrays["periods"][row] = frame.frame, frame.period
            arrays["xy_m"][row] = frame.xy_m
            for name in ("parse_missing", "parse_partial", "parse_invalid"):
                arrays[name][row] = getattr(frame, name)
            if progress and (row+1) % 50000 == 0:
                print(f"Cached {row+1:,} player-only records", flush=True)
        after = [(Path(p).stat().st_size, Path(p).stat().st_mtime_ns) for p in (meta_path, raw_path)]
        if after != before:
            raise RuntimeError("ATD files changed during cache generation")
        for a in arrays.values():
            a.flush()
        del a
        arrays.clear()
        np.save(dest/"team_codes.npy", meta.team_codes, allow_pickle=False)
        info = dict(version="player-cache-v1", n_frames=n_frames, n_slots=len(meta.team_codes),
                    fps=25, pitch=list(pitch), source_fingerprint=fingerprint,
                    source_sha256=dict(metadata=hashes[0], raw=hashes[1]),
                    official_blocks=[list(b) for b in blocks], slots_are_person_ids=False,
                    inputs_excluded=["ball", "speed", "smart_tagging", "broadcast", "public_gt"])
        (dest/"cache.json").write_text(json.dumps(info, indent=2)+"\n")
        if output.exists():
            raise FileExistsError("Frame cache appeared during generation")
        os.rename(dest, output)
    return open_frame_cache(output)


def open_frame_cache(path):
    path = Path(path)
    info = json.loads((path/"cache.json").read_text())
    if info.get("version") != "player-cache-v1" or info.get("slots_are_person_ids") is not False:
        raise ValueError("Unrecognized player cache")
    load = lambda name: np.load(path/f"{name}.npy", mmap_mode="r", allow_pickle=False)
    store = FrameStore(load("frame_ids"), load("periods"), load("xy_m"), load("team_codes"), info["pitch"],
                       fps=info["fps"], parse_missing=load("parse_missing"), parse_partial=load("parse_partial"),
                       parse_invalid=load("parse_invalid"), fingerprint=info["source_fingerprint"])
    if len(store) != info["n_frames"] or len(store.team_codes) != info["n_slots"]:
        raise ValueError("Player cache shape disagrees with manifest")
    return store
