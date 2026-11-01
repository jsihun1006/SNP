"""Find broadcast ATD frames resembling a requested tactical player layout, locally.

    python starter/find_layout_match.py --tactical-frame 11185

Slots are NOT linked across cameras. Compare each team's point set, testing four
coordinate reflections, then refine candidates with one-to-one point assignment
and a +/-2 s context window. Missing players and tracking errors mean coordinates
need not be identical. No possession GT, manual player corrections or video is used.

Output is a diagnostic query, not a verified synchronization anchor. Replays and
repeated formations can match. All frame indices are zero-based ATD records;
their frame_id is index+1. A broadcast MP4/ATD alignment discrepancy is NOT fixed
by this search. No resulting anchor is inserted into the possession pipeline.

Raw position arrays and candidate detail stay in data/derived/sync/. Terminal
output contains only a few frame candidates and aggregate error measures.
"""

import argparse
import csv
import json
import math
from pathlib import Path
import sys
import xml.etree.ElementTree as ET

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from games import GAME, GAMES, asset_paths, source_frames  # noqa: E402
from starter.tracking_utils import track_teams, FLIPS, reflected  # noqa: E402


def read_positions(raw_path, teams, expected, start=0, stop=None):
    stop = expected if stop is None else stop
    if not 0 <= start < stop <= expected:
        raise ValueError("Invalid tracking window.")
    positions = np.full((stop - start, len(teams), 2), np.nan, dtype=np.float32)
    count = 0
    with open(raw_path, encoding="utf-8") as handle:
        for frame, line in enumerate(handle):
            parts = line.rstrip("\r\n").split(":")
            if len(parts) != 3 or frame >= expected or int(parts[0]) != frame + 1:
                raise ValueError("ATD frame alignment/format check failed.")
            if start <= frame < stop:
                groups = parts[1].split(";")
                if len(groups) != len(teams):
                    raise ValueError("ATD slot count differs from metadata.")
                for slot, group in enumerate(groups):
                    fields = group.split(",")
                    if len(fields) != 3:
                        raise ValueError("Expected x,y,speed player groups.")
                    x, y = float(fields[0]), float(fields[1])
                    if np.isfinite(x) and np.isfinite(y):
                        positions[frame - start, slot] = x, y
            count += 1
    if count != expected:
        raise ValueError("ATD frame count differs from documented length.")
    positions *= np.asarray(GAMES[GAME]["pitch"], dtype=np.float32)
    return positions


def assignment_costs(cost):
    """Minimum-cost rectangular one-to-one assignment, independent of slot order.

    Hungarian augmenting-path algorithm; every row of the smaller side is matched.
    Test against brute-force assignments on small rectangular matrices.
    """
    cost = np.asarray(cost, dtype=float)
    if cost.ndim != 2 or not np.isfinite(cost).all():
        raise ValueError("Assignment needs a finite cost matrix.")
    if not cost.size:
        return np.array([], dtype=float)
    if cost.shape[0] > cost.shape[1]:
        cost = cost.T
    n, m = cost.shape
    u, v = np.zeros(n + 1), np.zeros(m + 1)
    p, way = np.zeros(m + 1, dtype=int), np.zeros(m + 1, dtype=int)
    for row in range(1, n + 1):
        p[0] = row
        j0 = 0
        distance = np.full(m + 1, np.inf)
        used = np.zeros(m + 1, dtype=bool)
        while True:
            used[j0] = True
            row0 = p[j0]
            reduced = cost[row0 - 1] - u[row0] - v[1:]
            improve = (~used[1:]) & (reduced < distance[1:])
            distance[1:][improve] = reduced[improve]
            way[1:][improve] = j0
            j1 = int(np.argmin(np.where(~used[1:], distance[1:], np.inf))) + 1
            delta = distance[j1]
            u[p[used]] += delta
            v[used] -= delta
            distance[~used] -= delta
            j0 = j1
            if p[j0] == 0:
                break
        while j0:
            j1 = way[j0]
            p[j0] = p[j1]
            j0 = j1
    return np.array([cost[p[column] - 1, column - 1]
                     for column in range(1, m + 1) if p[column]], dtype=float)


def team_points(positions, teams, side):
    selected = positions[np.asarray(teams) == side]
    return selected[np.isfinite(selected).all(axis=1)]


def layout_error(reference, candidate, ref_teams, candidate_teams, minimum=6):
    squared, trimmed, matched, ref_count, cand_count = [], [], 0, 0, 0
    for side in ("H", "A"):
        left, right = team_points(reference, ref_teams, side), team_points(candidate, candidate_teams, side)
        if min(len(left), len(right)) < minimum:
            return None
        costs = np.sum((left[:, None, :] - right[None, :, :]) ** 2, axis=2)
        assigned = assignment_costs(costs)
        squared.extend(assigned)
        # A small number of tracking/team assignment errors must not dominate ranking.
        trimmed.extend(np.sort(assigned)[:max(1, math.ceil(0.8 * len(assigned)))])
        matched += len(assigned)
        ref_count += len(left)
        cand_count += len(right)
    rmse = float(np.sqrt(np.mean(squared)))
    robust = float(np.sqrt(np.mean(trimmed)))
    unmatched_fraction = (ref_count + cand_count - 2 * matched) / (ref_count + cand_count)
    return {"rmse_m": rmse, "robust_rmse_m": robust,
            "score_m": robust + 5 * unmatched_fraction,
            "matched_players": matched, "reference_observations": ref_count,
            "broadcast_observations": cand_count}


def coarse_scores(reference, candidates, ref_teams, candidate_teams, minimum=6):
    """Batch symmetric nearest-point screening; final ranking uses assignments."""
    scores = np.zeros(len(candidates), dtype=float)
    for side in ("H", "A"):
        target = team_points(reference, ref_teams, side)
        if len(target) < minimum:
            raise ValueError("Reference has too few observations for this query.")
        mask = np.asarray(candidate_teams) == side
        for start in range(0, len(candidates), 256):
            points = candidates[start:start + 256, mask]
            valid = np.isfinite(points).all(axis=2)
            distances = np.sum((points[:, :, None, :] - target[None, None, :, :]) ** 2, axis=3)
            distances = np.where(valid[:, :, None], distances, np.inf)
            count = valid.sum(axis=1)
            forward = np.where(valid, np.min(distances, axis=2), 0).sum(axis=1) / np.maximum(count, 1)
            backward = np.min(distances, axis=1).mean(axis=1)
            value = (forward + backward) / 2
            value[count < minimum] = np.inf
            scores[start:start + len(points)] += value
    return scores / 2


def find_matches(reference_window, reference_center, broadcast, ref_teams, broadcast_teams,
                 fps=25, minimum=6, events_per_flip=8):
    offsets = np.array([-2 * fps, -fps, 0, fps, 2 * fps])
    radius = 2 * fps
    if reference_center < radius or reference_center + radius >= len(reference_window):
        raise ValueError("Reference needs a full +/-2 second context window.")
    reflections = [reflected(broadcast, flip, GAMES[GAME]["pitch"]) for flip in FLIPS]
    events = []
    for flip_index, candidate in enumerate(reflections):
        scores = coarse_scores(reference_window[reference_center], candidate,
                               ref_teams, broadcast_teams, minimum)
        scores[:radius] = np.inf
        scores[len(scores) - radius:] = np.inf
        for _ in range(events_per_flip):
            center = int(np.argmin(scores))
            if not np.isfinite(scores[center]):
                break
            events.append((center, flip_index))
            scores[max(0, center - 5 * fps):center + 5 * fps + 1] = np.inf
    matches = []
    for center, flip_index in events:
        candidate = reflections[flip_index]
        best = None
        for frame in range(max(radius, center - fps), min(len(broadcast) - radius, center + fps + 1)):
            context = []
            details = None
            for offset in offsets:
                error = layout_error(reference_window[reference_center + offset],
                                     candidate[frame + offset], ref_teams, broadcast_teams, minimum)
                if error is not None:
                    context.append(error["score_m"])
                    if offset == 0:
                        details = error
            if details is None or len(context) < 3:
                continue
            score = float(np.mean(context))
            entry = {"broadcast_atd_frame0": frame, "broadcast_atd_frame_id": frame + 1,
                     "xy_reflection_index": flip_index, "window_score_m": score,
                     "context_samples": len(context), **details}
            if best is None or score < best["window_score_m"]:
                best = entry
        if best is not None:
            matches.append(best)
    matches.sort(key=lambda item: item["window_score_m"])
    # Avoid reporting neighboring minima/reflections as separate independent events.
    selected = []
    for item in matches:
        if all(abs(item["broadcast_atd_frame0"] - other["broadcast_atd_frame0"]) > 5 * fps
               for other in selected):
            selected.append(item)
    return selected


def main():
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--tactical-frame", type=int, required=True, help="Zero-based tactical ATD/video frame.")
    parser.add_argument("--data-dir", help="Local data bundle root.")
    args = parser.parse_args()
    try:
        fps = int(GAMES[GAME]["fps"])
        radius = 2 * fps
        target = args.tactical_frame
        paths = {side: asset_paths(data_dir=args.data_dir, source=side) for side in ("tactical", "broadcast")}
        teams = {side: track_teams(paths[side]["atd_meta"]) for side in paths}
        reference = read_positions(paths["tactical"]["atd_raw"], teams["tactical"],
                                   source_frames("tactical"), target - radius, target + radius + 1)
        broadcast = read_positions(paths["broadcast"]["atd_raw"], teams["broadcast"], source_frames("broadcast"))
        matches = find_matches(reference, radius, broadcast, teams["tactical"], teams["broadcast"], fps)
        if not matches:
            raise ValueError("No sufficiently observed candidates.")
        report = {"tactical_frame0": target, "tactical_atd_frame_id": target + 1,
                  "method": "team_point_sets_one_to_one_assignment_with_temporal_context",
                  "context_seconds_each_side": 2, "minimum_observations_per_team": 6,
                  "verified_video_alignment": False, "live_replay_verified": False,
                  "candidates": matches}
        folder = ROOT / "data" / "derived" / "sync"
        folder.mkdir(parents=True, exist_ok=True)
        stem = "layout_query_{:06d}".format(target)
        (folder / (stem + ".json")).write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
        with (folder / (stem + ".csv")).open("w", newline="", encoding="utf-8") as handle:
            writer = csv.DictWriter(handle, fieldnames=list(matches[0]))
            writer.writeheader()
            writer.writerows(matches)
    except (OSError, ValueError, IndexError, ET.ParseError):
        print("Local player-layout search failed. No private coordinate rows logged.", file=sys.stderr)
        return 1
    summary = {"tactical_frame0": target, "video_alignment_verified": False,
               "top_candidates": matches[:3]}
    print(json.dumps(summary, indent=2))
    print("Saved private local player-layout query under data/derived/sync/.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
