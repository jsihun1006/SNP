"""Compute tactical team width/length locally, without video, homography or ground truth.

Run from the repository root:
    python starter/team_extents.py

The default CSV is data/derived/tactical_team_extents.csv (private and git-ignored).
Never upload it or print its match-derived rows into a cloud LLM conversation.
Only completion status is printed. Input rows/values are never included in errors.

frame is the zero-based tactical video frame; time_seconds is video elapsed time.
For each team H/A:
    width_m  = (max(y) - min(y)) * pitch width
    length_m = (second_largest(x) - second_smallest(x)) * pitch length
    x_min = second_smallest(x), x_max = second_largest(x); y bounds remain min/max(y)
    length_method records this convention, which also applies to video lines.
Bounds allow frame-specific projection onto video via tactical homography.
No goalkeeper identity is inferred; one observation is trimmed at each x extreme.
Order statistics count observations, so tied x values are not deduplicated.
Slots are not
stable player identities: n_tracks counts observations, not distinct players.
Missing/non-finite coordinate pairs are excluded and counted as n_missing.
Finite positions outside [0, 1] are included without clipping and counted in
n_outside_pitch. No deduplication, imputation or smoothing is applied.
Width/y bounds need two observations; length/x bounds need four, otherwise empty.
Metrics describe observed extent, not guaranteed full-team extent or possession.
"""

import argparse
import csv
import math
from pathlib import Path
import os
import sys
import tempfile
import xml.etree.ElementTree as ET

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from games import GAME, GAMES, require, source_frames  # noqa: E402

LENGTH_METHOD = "second_largest_x_minus_second_smallest_x"


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


def team_bounds(points):
    """Normalised x second-min/second-max and y min/max, with per-axis availability."""
    if len(points) < 2:
        return None, None, None, None
    xs, ys = zip(*points)
    x_min, x_max = None, None
    if len(points) >= 4:
        ordered_x = sorted(xs)
        x_min, x_max = ordered_x[1], ordered_x[-2]
    return x_min, x_max, min(ys), max(ys)


def team_extent(points, pitch):
    """(width, x length trimmed by one observation at each end) in metres, or missing."""
    xmin, xmax, ymin, ymax = team_bounds(points)
    length, width = pitch
    return ((ymax - ymin) * width if ymin is not None else None,
            (xmax - xmin) * length if xmin is not None else None)


def frame_metrics(groups, teams, pitch):
    """Parse one frame without accessing ball fields or inferring possession."""
    if len(groups) != len(teams):
        raise ValueError("Player group count differs from metadata slot count.")
    points = {"H": [], "A": []}
    missing = {"H": 0, "A": 0}
    outside = {"H": 0, "A": 0}
    for group, team in zip(groups, teams):
        fields = group.split(",")
        if len(fields) != 3:
            raise ValueError("Expected x,y,speed for each player slot.")
        try:
            x, y = float(fields[0]), float(fields[1])
        except ValueError:
            raise ValueError("Non-numeric player coordinates.") from None
        if not (math.isfinite(x) and math.isfinite(y)):
            missing[team] += 1
            continue
        points[team].append((x, y))
        outside[team] += int(not (0 <= x <= 1 and 0 <= y <= 1))
    result = {}
    for team in ("H", "A"):
        width, length = team_extent(points[team], pitch)
        bounds = team_bounds(points[team])
        result.update({
            team + "_n_tracks": len(points[team]),
            team + "_width_m": width,
            team + "_length_m": length,
            team + "_n_missing": missing[team],
            team + "_n_outside_pitch": outside[team],
            **{team + "_" + name: value for name, value in zip(
                ("x_min", "x_max", "y_min", "y_max"), bounds)},
        })
    return result


def write_extents(raw_path, teams, output, n_frames, fps, pitch):
    """Stream the ATD; publish output atomically only after alignment checks pass."""
    if not math.isfinite(fps) or fps <= 0:
        raise ValueError("FPS must be positive and finite.")
    if len(pitch) != 2 or any(not math.isfinite(v) or v <= 0 for v in pitch):
        raise ValueError("Pitch dimensions must be positive and finite.")
    output = Path(output)
    if output.resolve() == Path(raw_path).resolve():
        raise ValueError("Output must differ from the input ATD file.")
    output.parent.mkdir(parents=True, exist_ok=True)
    columns = ["frame", "time_seconds", "length_method"] + [
        team + "_" + name
        for team in ("H", "A")
        for name in ("n_tracks", "width_m", "length_m", "n_missing", "n_outside_pitch",
                     "x_min", "x_max", "y_min", "y_max")
    ]
    temporary = None
    try:
        with tempfile.NamedTemporaryFile(
            mode="w", newline="", encoding="utf-8", dir=output.parent,
            prefix=".team_extents_", suffix=".tmp", delete=False,
        ) as handle:
            temporary = Path(handle.name)
            writer = csv.DictWriter(handle, fieldnames=columns)
            writer.writeheader()
            count = 0
            with open(raw_path, encoding="utf-8") as source:
                for frame, line in enumerate(source):
                    parts = line.rstrip("\r\n").split(":")
                    if len(parts) != 3:
                        raise ValueError("Expected frame:players:ball ATD format.")
                    try:
                        frame_id = int(parts[0])
                    except ValueError:
                        raise ValueError("Invalid ATD frame ID.") from None
                    if frame_id != frame + 1 or frame >= n_frames:
                        raise ValueError("ATD frame IDs must be contiguous from 1 to video length.")
                    row = {"frame": frame, "time_seconds": format(frame / fps, ".6f"),
                           "length_method": LENGTH_METHOD}
                    row.update(frame_metrics(parts[1].split(";"), teams, pitch))
                    for key, value in row.items():
                        if key.endswith("_m") and value is not None:
                            row[key] = format(value, ".6f")
                        elif key.endswith(("_min", "_max")) and value is not None:
                            row[key] = format(value, ".9f")
                    writer.writerow(row)
                    count += 1
            if count != n_frames:
                raise ValueError("ATD record count differs from video length.")
        os.replace(temporary, output)
    finally:
        if temporary is not None:
            temporary.unlink(missing_ok=True)


def main():
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--data-dir", help="Local bundle root; otherwise use MTAH_DATA_DIR or data/.")
    args = parser.parse_args()
    try:
        meta, raw = require(["atd_meta", "atd_raw"], data_dir=args.data_dir, source="tactical")
        # Always place match-derived output in the repository's ignored private data folder.
        output = ROOT / "data" / "derived" / "tactical_team_extents.csv"
        write_extents(raw, track_teams(meta), output, source_frames("tactical"),
                      GAMES[GAME]["fps"], GAMES[GAME]["pitch"])
    except (OSError, ValueError, ET.ParseError):
        # Do not send raw data, metadata excerpts or parse errors to terminal/LLM logs.
        print("Team extents export failed. Check local input files and ATD format/alignment.",
              file=sys.stderr)
        return 1
    print("Saved private local CSV: data/derived/tactical_team_extents.csv")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
