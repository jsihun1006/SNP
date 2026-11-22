"""Player-only ATD integrity checks. Slots are observations, never person identities."""
from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
import xml.etree.ElementTree as ET

import numpy as np


@dataclass(frozen=True)
class ObservationPolicy:
    bounds_margin_m: float = 2.0
    duplicate_radius_m: float = 0.2
    min_team_players: int = 8
    max_team_players: int = 11

    def __post_init__(self):
        if not np.isfinite(self.bounds_margin_m) or self.bounds_margin_m < 0:
            raise ValueError("bounds_margin_m must be finite and nonnegative")
        if not np.isfinite(self.duplicate_radius_m) or self.duplicate_radius_m < 0:
            raise ValueError("duplicate_radius_m must be finite and nonnegative")
        if not 1 <= self.min_team_players <= self.max_team_players:
            raise ValueError("Invalid team coverage policy")


@dataclass(frozen=True)
class ATDMetadata:
    team_codes: np.ndarray
    fps: float | None
    xml_pitch: tuple[float, float] | None
    mapping_source: str
    slot_order_source: str
    coordinate_units: tuple[str, ...]


@dataclass(frozen=True)
class PlayerFrame:
    frame: int
    period: int
    xy_m: np.ndarray  # source slots, NaN allowed; no ball field
    team_codes: np.ndarray
    parse_missing: np.ndarray
    parse_partial: np.ndarray
    parse_invalid: np.ndarray


def read_metadata(path, expected_fps=25.0, expected_slots=46,
                  require_explicit_teams=True) -> ATDMetadata:
    root = ET.parse(path).getroot()
    # Strip namespaces without depending on a particular EPTS namespace/version.
    for node in root.iter():
        node.tag = node.tag.rsplit("}", 1)[-1]
    teams = [n.get("id") for n in root.iter("Team")]
    if len(teams) != 2 or None in teams or len(set(teams)) != 2:
        raise ValueError("ATD must declare two distinct team IDs")
    score = root.find(".//Score")
    if score is None:
        if require_explicit_teams:
            raise ValueError("Missing explicit Local/Visiting team mapping")
        home, away = teams
        mapping_source = "team_order_fallback_unverified"
    else:
        home, away = score.get("idLocalTeam"), score.get("idVisitingTeam")
        if home == away or {home, away} != set(teams):
            raise ValueError("Invalid Local/Visiting team mapping")
        mapping_source = "Score.Local/Visiting"
    players = list(root.iter("Player"))
    ids = [p.get("id") for p in players]
    # Tiny legacy synthetic fixtures omit player IDs and channel references.
    refs = list(root.iter("PlayerChannelRef"))
    if refs:
        if None in ids or len(set(ids)) != len(ids):
            raise ValueError("Duplicate or missing source slot IDs")
        channels = {n.get("id"): n for n in root.iter("PlayerChannel")}
        if len(channels) != len(list(root.iter("PlayerChannel"))):
            raise ValueError("Duplicate player channel IDs")
        try:
            ordered = [channels[r.attrib["playerChannelId"]].attrib["playerId"] for r in refs]
        except KeyError:
            raise ValueError("Unresolved ATD player channel reference") from None
        if len(ordered) % 3 or any(len(set(ordered[i:i+3])) != 1
                                  for i in range(0, len(ordered), 3)):
            raise ValueError("Expected contiguous x/y/s channels per source slot")
        slot_ids = ordered[::3]
        if len(set(slot_ids)) != len(slot_ids) or set(slot_ids) != set(ids):
            raise ValueError("Source slot references must cover each declared player once")
        by_id = {p.get("id"): p for p in players}
        players = [by_id[i] for i in slot_ids]
        order_source = "PlayerChannelRef"
        channel_defs = {n.get("id"): n for n in root.iter("Channel")}
        for i, ref in enumerate(refs):
            if i % 3 == 2:  # speed is deliberately unused
                continue
            pc = channels[ref.get("playerChannelId")]
            channel = channel_defs.get(pc.get("channelId"))
            if channel is None or channel.findtext("Name") != ("position_x" if i % 3 == 0 else "position_y"):
                raise ValueError("ATD player x/y channel order is not supported")
            if channel.findtext("Unit") != "normalized":
                raise ValueError("ATD player positions must use normalized coordinates")
    else:
        if require_explicit_teams:
            raise ValueError("Missing source slot channel order")
        order_source = "Player_order_fallback_unverified"
    if not players or any(p.get("teamId") not in (home, away) for p in players):
        raise ValueError("Unexpected source slot team code")
    if expected_slots is not None and len(players) != expected_slots:
        raise ValueError("Unexpected ATD source slot count")
    text = root.findtext(".//FrameRate")
    fps = float(text) if text is not None else None
    if expected_fps is not None and (fps is None or not np.isfinite(fps) or fps != expected_fps):
        raise ValueError("ATD frame rate disagrees with the expected FPS")
    field = root.find(".//FieldSize")
    xml_pitch = None if field is None else (float(field.findtext("Width")), float(field.findtext("Height")))
    if xml_pitch is not None and (not np.isfinite(xml_pitch).all() or min(xml_pitch) <= 0):
        raise ValueError("Invalid metadata pitch dimensions")
    units = tuple(sorted({n.findtext("Unit") for n in root.iter("Channel")
                          if n.findtext("Name") in ("position_x", "position_y")}))
    return ATDMetadata(np.array([0 if p.get("teamId") == home else 1 for p in players], dtype=np.int8),
                       fps, xml_pitch, mapping_source, order_source, units)


def parse_players(groups: str, n_slots: int, pitch):
    """Parse x/y only. Speed and the separate ball field never enter the result."""
    entries = groups.split(";")
    if len(entries) != n_slots:
        raise ValueError("ATD source slot count mismatch")
    xy = np.full((n_slots, 2), np.nan, dtype=np.float64)
    missing = np.zeros(n_slots, dtype=bool)
    partial = missing.copy()
    invalid = missing.copy()
    for i, entry in enumerate(entries):
        fields = entry.split(",")
        if len(fields) != 3:
            invalid[i] = True
            continue
        try:
            x, y = float(fields[0]), float(fields[1])
        except ValueError:
            invalid[i] = True
            continue
        nan = np.isnan([x, y])
        missing[i], partial[i] = nan.all(), nan.any() and not nan.all()
        invalid[i] = bool(np.isinf([x, y]).any())
        if np.isfinite([x, y]).all():
            xy[i] = [x * pitch[0], y * pitch[1]]
    return xy, missing, partial, invalid


def validate_blocks(blocks, n_frames):
    previous_end = -1
    periods = set()
    for period, first, last in blocks:
        if period not in (1, 2) or period in periods or not 0 <= first <= last < n_frames or first <= previous_end:
            raise ValueError("Invalid or overlapping half blocks")
        previous_end = last
        periods.add(period)


def period_of(frame: int, blocks) -> int:
    return next((half for half, first, last in blocks if first <= frame <= last), 0)


def iter_player_frames(raw_path, meta: ATDMetadata, pitch, n_frames, blocks):
    """Strict entire-file stream: every ID must equal its 1-based line number."""
    if n_frames <= 0 or len(pitch) != 2 or not np.isfinite(pitch).all() or min(pitch) <= 0:
        raise ValueError("Invalid expected frame count or pitch")
    validate_blocks(blocks, n_frames)
    count = 0
    with Path(raw_path).open(encoding="utf-8") as source:
        for count, line in enumerate(source, 1):
            parts = line.rstrip("\r\n").split(":", 2)
            if len(parts) != 3:
                raise ValueError(f"Invalid ATD record structure at line {count}")
            try:
                frame_id = int(parts[0])
            except ValueError:
                raise ValueError(f"Invalid frame ID at line {count}") from None
            if frame_id != count or count > n_frames:
                raise ValueError(f"Noncontiguous, duplicate, or out-of-order frame ID at line {count}")
            xy, missing, partial, invalid = parse_players(parts[1], len(meta.team_codes), pitch)
            frame = frame_id - 1
            yield PlayerFrame(frame, period_of(frame, blocks), xy, meta.team_codes,
                              missing, partial, invalid)
    if count != n_frames:
        raise ValueError(f"Expected {n_frames} ATD records, got {count}")


def assess_observations(frame: PlayerFrame, pitch, policy=ObservationPolicy()):
    """Flag, but never merge, same-team near duplicates. Exclude only hard bounds."""
    xy = np.asarray(frame.xy_m, dtype=float).copy()
    teams = np.asarray(frame.team_codes)
    if xy.ndim != 2 or xy.shape[1] != 2 or teams.shape != (len(xy),) or not np.isin(teams, [0, 1]).all():
        raise ValueError("Invalid player-only observation arrays")
    finite = np.isfinite(xy).all(axis=1)
    strict_out = finite & ((xy < 0).any(axis=1) | (xy > pitch).any(axis=1))
    excluded = finite & ((xy < -policy.bounds_margin_m).any(axis=1) |
                         (xy > np.asarray(pitch) + policy.bounds_margin_m).any(axis=1))
    xy[~finite | excluded] = np.nan
    valid = np.isfinite(xy).all(axis=1)
    duplicates, participants = 0, 0
    counts = []
    for team in (0, 1):
        p = xy[valid & (teams == team)]
        counts.append(len(p))
        d = np.linalg.norm(p[:, None] - p[None, :], axis=-1)
        close = np.triu(d <= policy.duplicate_radius_m, k=1)
        duplicates += int(close.sum())
        participants += int((close.any(axis=0) | close.any(axis=1)).sum())
    n_h, n_a = counts
    coverage = policy.min_team_players <= min(counts) and max(counts) <= policy.max_team_players
    quality = {
        "raw_observed": int(finite.sum()), "missing_slots": int(frame.parse_missing.sum()),
        "partial_missing_slots": int(frame.parse_partial.sum()), "invalid_slots": int(frame.parse_invalid.sum()),
        "outside_pitch_slots": int(strict_out.sum()), "excluded_bounds_slots": int(excluded.sum()),
        "duplicate_suspect_pairs": duplicates, "duplicate_suspect_players": participants,
        "coverage_ok": bool(coverage),
        "quality_valid": bool(coverage and not duplicates and not strict_out.any()
                              and not frame.parse_invalid.any() and not frame.parse_partial.any()),
        "n_home_valid": n_h, "n_away_valid": n_a,
    }
    return xy, quality
