"""Load the gated BAR-ATH assets into small, explicit data containers.

The raw ATD is Metrica's FIFA/EPTS export.  ``video_frame = frame_id - 1``; the tactical file has
one record per submission row.  Smart tags are timed in seconds at 25 fps.  The homography is
already indexed by video frame and maps normalised pitch coordinates (the ATD's native 0-1
coordinates) to image pixels.
"""

from __future__ import annotations

import math
import xml.etree.ElementTree as ET
from dataclasses import dataclass, field
from pathlib import Path
import numpy as np
import pandas as pd

from .config import CODE_TO_LABEL, FPS, LABEL_TO_CODE, N_FRAMES, PITCH


@dataclass(frozen=True)
class TrackingData:
    """One row per tactical video frame, coordinates in metres."""

    ball: np.ndarray  # (N, 2), NaN when the ball is not tracked
    ball_present: np.ndarray  # (N,) bool
    players: np.ndarray  # (N, P, 2), NaN when a slot is empty
    speeds: np.ndarray  # (N, P), NaN when a slot is empty
    player_present: np.ndarray  # (N, P) bool
    teams: np.ndarray  # (P,) 0 home, 1 away, -1 unknown
    pitch: tuple[float, float]
    n_frames: int

    @property
    def n_players(self) -> int:
        return int(self.players.shape[1])

    @property
    def length(self) -> float:
        return float(self.pitch[0])

    @property
    def width(self) -> float:
        return float(self.pitch[1])


@dataclass(frozen=True)
class TagEvent:
    start_frame: int
    stop_frame: int
    code: str
    team: int
    type: str
    labels: dict[str, str] = field(default_factory=dict)


@dataclass(frozen=True)
class TagTable:
    """Frame-aligned smart-tag features.

    ``team_vote`` is positive for Barcelona and negative for Atletic Club, matching the direction
    used by the reference model.  It is a weak signal, not a label: the tags are only right about
    70% of the time and ``N/A`` covers every throw-in.
    """

    events: tuple[TagEvent, ...]
    code_names: tuple[str, ...]
    type_names: tuple[str, ...]
    code_matrix: np.ndarray
    type_matrix: np.ndarray
    team_vote: np.ndarray
    setpiece: np.ndarray
    coverage: np.ndarray
    n_frames: int

    @classmethod
    def empty(cls, n_frames: int) -> "TagTable":
        return cls(
            events=(),
            code_names=(),
            type_names=(),
            code_matrix=np.zeros((n_frames, 0), dtype=np.float32),
            type_matrix=np.zeros((n_frames, 0), dtype=np.float32),
            team_vote=np.zeros(n_frames, dtype=np.int16),
            setpiece=np.zeros(n_frames, dtype=np.int8),
            coverage=np.zeros(n_frames, dtype=bool),
            n_frames=int(n_frames),
        )


@dataclass(frozen=True)
class Homography:
    matrices: np.ndarray  # (N, 3, 3)
    reprojection_error: np.ndarray  # (N,)
    valid: np.ndarray  # (N,) bool

    @property
    def n_frames(self) -> int:
        return int(self.matrices.shape[0])

    def summary(self) -> dict[str, float | int]:
        usable = self.reprojection_error[self.valid]
        return {
            "n_frames": self.n_frames,
            "valid_frames": int(self.valid.sum()),
            "valid_pct": float(100.0 * self.valid.mean()) if self.n_frames else 0.0,
            "median_reprojection_error": float(np.median(usable)) if usable.size else float("nan"),
            "p99_reprojection_error": float(np.percentile(usable, 99)) if usable.size else float("nan"),
        }


def _team_id_map(root: ET.Element) -> tuple[dict[str, int], list[str]]:
    ids = [team.get("id", "") for team in root.iter("Team")]
    mapping = {team_id: index for index, team_id in enumerate(ids)}
    return mapping, ids


def load_tracking(
    raw_path: str | Path,
    meta_path: str | Path,
    pitch: tuple[float, float] = PITCH,
    n_frames: int | None = None,
) -> TrackingData:
    """Parse the raw ATD and its XML metadata.

    The parser is deliberately boring: the format is ``frame_id:players:ball``; the first frame id
    is 1 and submission row 0 is video frame 0.  Coordinate triples are ``x,y,speed`` and are
    scaled from the ATD's normalised 0-1 pitch to metres.
    """
    raw_path = Path(raw_path)
    meta_path = Path(meta_path)
    if not raw_path.is_file():
        raise FileNotFoundError(f"ATD raw file not found: {raw_path}")
    if not meta_path.is_file():
        raise FileNotFoundError(f"ATD metadata not found: {meta_path}")

    root = ET.parse(meta_path).getroot()
    team_map, _ = _team_id_map(root)
    xml_players = list(root.iter("Player"))
    teams = np.array(
        [team_map.get(player.get("teamId", ""), -1) for player in xml_players],
        dtype=np.int8,
    )

    with raw_path.open("r", encoding="utf-8") as handle:
        row_count = sum(1 for line in handle if line.strip())
    if n_frames is None:
        n_frames = row_count
    n_frames = int(n_frames)
    if row_count > n_frames:
        raise ValueError(f"{raw_path}: {row_count} rows exceeds n_frames={n_frames}")

    first_tokens: list[str] = []
    if not xml_players:
        with raw_path.open("r", encoding="utf-8") as handle:
            for line in handle:
                if line.strip():
                    first_tokens = line.rstrip("\n").split(":")[1].split(";")
                    break
    n_players = len(xml_players) or len(first_tokens)
    if n_players <= 0:
        raise ValueError(f"{raw_path}: cannot determine player-slot count")
    if len(teams) != n_players:
        # The XML should describe every slot.  If it does not, keep going with unknown teams
        # rather than inventing an alignment.
        teams = np.full(n_players, -1, dtype=np.int8)

    ball = np.full((n_frames, 2), np.nan, dtype=np.float32)
    ball_present = np.zeros(n_frames, dtype=bool)
    players = np.full((n_frames, n_players, 2), np.nan, dtype=np.float32)
    speeds = np.full((n_frames, n_players), np.nan, dtype=np.float32)
    player_present = np.zeros((n_frames, n_players), dtype=bool)

    length, width = float(pitch[0]), float(pitch[1])
    with raw_path.open("r", encoding="utf-8") as handle:
        for line in handle:
            if not line.strip():
                continue
            parts = line.rstrip("\n").split(":")
            if len(parts) != 3:
                raise ValueError(f"{raw_path}: malformed ATD row (expected 3 colon fields): {line[:80]!r}")
            frame_id = int(parts[0])
            frame = frame_id - 1
            if frame < 0 or frame >= n_frames:
                raise ValueError(f"{raw_path}: frame id {frame_id} outside 0..{n_frames - 1}")

            ball_token = parts[2].strip()
            if ball_token and not ball_token.upper().startswith("N"):
                values = ball_token.split(",")
                if len(values) >= 2:
                    ball[frame, 0] = float(values[0]) * length
                    ball[frame, 1] = float(values[1]) * width
                    ball_present[frame] = True

            player_tokens = parts[1].split(";") if parts[1] else []
            if len(player_tokens) > n_players:
                raise ValueError(
                    f"{raw_path}: row {frame_id} has {len(player_tokens)} player slots; expected <= {n_players}"
                )
            for slot, token in enumerate(player_tokens):
                token = token.strip()
                if not token or token.upper().startswith("N"):
                    continue
                values = token.split(",")
                if len(values) < 2:
                    continue
                players[frame, slot, 0] = float(values[0]) * length
                players[frame, slot, 1] = float(values[1]) * width
                player_present[frame, slot] = True
                if len(values) >= 3 and not values[2].upper().startswith("N"):
                    speeds[frame, slot] = float(values[2])

    return TrackingData(
        ball=ball,
        ball_present=ball_present,
        players=players,
        speeds=speeds,
        player_present=player_present,
        teams=teams,
        pitch=(length, width),
        n_frames=n_frames,
    )


def _team_code(text: str) -> int:
    text = (text or "").strip()
    if text == "Barcelona":
        return 0
    if text in {"Atletic Club", "Athletic Club"}:
        return 1
    return -1


ATTACKING_CODES = {
    "BUILD UP",
    "PROGRESSION",
    "LONG BALL",
    "ATTACKING TRANSITION",
    "CREATING CHANCES",
    "BALL IN FINAL THIRD",
    "BALL IN THE BOX",
    "PLAYERS IN THE BOX",
    "GOALS",
}
DEFENDING_CODES = {
    "DEFENDING IN MIDDLE THIRD",
    "DEFENDING IN DEFENSIVE THIRD",
    "DEFENDING IN ATTACKING THIRD",
    "DEFENSIVE TRANSITION",
}


def load_smart_tags(
    path: str | Path,
    n_frames: int = N_FRAMES,
    fps: int = FPS,
) -> TagTable:
    """Parse smart-tag intervals into frame-aligned arrays.

    The XML interval end is inclusive at the recorded second, so the half-open array interval is
    ``[round(start*fps), round(end*fps) + 1)``.  That is exactly the convention used by the
    reference experiments and keeps a tag from vanishing on its final frame.
    """
    path = Path(path)
    if not path.is_file():
        raise FileNotFoundError(f"smart-tag file not found: {path}")
    root = ET.parse(path).getroot()
    n_frames = int(n_frames)

    instances = list(root.iter("instance"))
    code_names = tuple(sorted({instance.findtext("code", default="") for instance in instances}))
    type_names = tuple(
        sorted(
            {
                label.findtext("text", default="")
                for instance in instances
                for label in instance.findall("label")
                if label.findtext("group") == "Type"
            }
        )
    )
    code_index = {name: i for i, name in enumerate(code_names)}
    type_index = {name: i for i, name in enumerate(type_names)}

    code_matrix = np.zeros((n_frames, len(code_names)), dtype=np.float32)
    type_matrix = np.zeros((n_frames, len(type_names)), dtype=np.float32)
    team_vote = np.zeros(n_frames, dtype=np.int16)
    setpiece = np.zeros(n_frames, dtype=np.int8)
    coverage = np.zeros(n_frames, dtype=bool)
    events: list[TagEvent] = []

    for instance in instances:
        try:
            start_seconds = float(instance.findtext("start", default="nan"))
            end_seconds = float(instance.findtext("end", default="nan"))
        except ValueError:
            continue
        if not math.isfinite(start_seconds) or not math.isfinite(end_seconds):
            continue
        start = max(0, int(round(start_seconds * fps)))
        stop = min(n_frames, int(round(end_seconds * fps)) + 1)
        if stop <= start:
            continue

        code = instance.findtext("code", default="")
        labels = {
            label.findtext("group", default=""): label.findtext("text", default="")
            for label in instance.findall("label")
        }
        team_text = labels.get("Team", "N/A")
        type_text = labels.get("Type", "")
        team = _team_code(team_text)

        sign = 0
        if team >= 0:
            if code in ATTACKING_CODES:
                sign = 1 if team == 0 else -1
            elif code in DEFENDING_CODES:
                sign = -1 if team == 0 else 1

        if code in code_index:
            code_matrix[start:stop, code_index[code]] += 1.0
        if type_text in type_index:
            type_matrix[start:stop, type_index[type_text]] += 1.0
        team_vote[start:stop] += sign
        coverage[start:stop] = True
        if code == "SET PIECES":
            setpiece[start:stop] = 1

        events.append(
            TagEvent(
                start_frame=start,
                stop_frame=stop,
                code=code,
                team=team,
                type=type_text,
                labels=labels,
            )
        )

    return TagTable(
        events=tuple(events),
        code_names=code_names,
        type_names=type_names,
        code_matrix=code_matrix,
        type_matrix=type_matrix,
        team_vote=team_vote,
        setpiece=setpiece,
        coverage=coverage,
        n_frames=n_frames,
    )


MATRIX_COLUMNS = [f"h{row}{col}" for row in range(3) for col in range(3)]


def load_homography(path: str | Path) -> Homography:
    """Load per-frame homographies.  The tactical file is valid on every frame."""
    path = Path(path)
    if not path.is_file():
        raise FileNotFoundError(f"homography file not found: {path}")
    table = pd.read_csv(path)
    missing = [name for name in ["frame", "reprojection_error", "valid", *MATRIX_COLUMNS] if name not in table]
    if missing:
        raise ValueError(f"{path}: missing columns {missing}")
    matrices = table[MATRIX_COLUMNS].to_numpy(dtype=np.float64).reshape(-1, 3, 3)
    error = table["reprojection_error"].to_numpy(dtype=np.float64)
    valid = table["valid"].to_numpy().astype(bool)
    if not np.array_equal(table["frame"].to_numpy(), np.arange(len(table))):
        raise ValueError(f"{path}: frame column is not contiguous from zero")
    return Homography(matrices=matrices, reprojection_error=error, valid=valid)


def load_public_gt(path: str | Path) -> np.ndarray:
    """Load the public 7,500-frame ground truth as a full-length code array."""
    path = Path(path)
    if not path.is_file():
        raise FileNotFoundError(f"public ground truth not found: {path}")
    table = pd.read_csv(path)
    if list(table.columns[:2]) != ["frame", "label"]:
        raise ValueError(f"{path}: expected frame,label columns, got {list(table.columns)}")
    if not np.array_equal(table["frame"].to_numpy(), np.arange(len(table))):
        raise ValueError(f"{path}: frame column is not contiguous from zero")
    unknown = sorted(set(table["label"]) - set(LABEL_TO_CODE))
    if unknown:
        raise ValueError(f"{path}: unknown labels {unknown}")
    return table["label"].map(LABEL_TO_CODE).to_numpy(dtype=np.int8)


def load_submission(path: str | Path) -> np.ndarray:
    """Read a full-match submission CSV and return its labels as codes."""
    path = Path(path)
    table = pd.read_csv(path)
    if list(table.columns) != ["frame", "label"]:
        raise ValueError(f"{path}: expected exactly frame,label; got {list(table.columns)}")
    if not np.array_equal(table["frame"].to_numpy(), np.arange(len(table))):
        raise ValueError(f"{path}: frame column is not 0..n-1")
    unknown = sorted(set(table["label"]) - set(LABEL_TO_CODE))
    if unknown:
        raise ValueError(f"{path}: unknown labels {unknown}")
    return table["label"].map(LABEL_TO_CODE).to_numpy(dtype=np.int8)


def codes_to_labels(codes: np.ndarray) -> list[str]:
    return [CODE_TO_LABEL[int(code)] for code in codes]
