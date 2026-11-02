"""Automatically select per-half ATD anchors from official starts and +/-50 frames.

    python model/영상추출/모델/tracking_alignment/align_half_starts.py

Reads official tactical half bounds from games.py, never possession GT. Search
broadcast team point sets globally, then compare every frame in the +/-50 window
near the strongest coarse events. Source-specific slots are matched one-to-one;
four coordinate reflections are considered. No manual broadcast anchor is read.

After each automatic start, map B = B_start + (T - T_start) through that half.
This is the requested per-half CONSTANT-OFFSET ASSUMPTION, not proof of live
status or synchronization throughout the half. Cuts/replays may still diverge.
Broadcast MP4/ATD indexing remains unverified. Never modify source tracking/GT.

Missing reference evidence, no match, chronological conflicts and broadcast end
are explicit unmatched cases. Halftime/outside the official blocks remain blank.
Output: private local model/영상추출/추출csv/tracking_alignment/sync/half_start_alignment/{mapping.csv,report.json}.
"""

import argparse
import csv
import json
from pathlib import Path
import sys
import xml.etree.ElementTree as ET

import numpy as np

ROOT = next(parent for parent in Path(__file__).resolve().parents
            if (parent / "games.py").is_file())
OUTPUT_ROOT = ROOT / "model/영상추출/추출csv/tracking_alignment"
sys.path.insert(0, str(ROOT))
from games import GAME, GAMES, asset_paths, half_video_blocks, source_frames  # noqa: E402
from model.영상추출.모델.tracking_alignment.find_layout_match import find_matches, layout_error, read_positions, team_points  # noqa: E402
from model.영상추출.모델.tracking_alignment.tracking_utils import track_teams, FLIPS, reflected  # noqa: E402

FOLDER = OUTPUT_ROOT / 'sync/half_start_alignment'


def acceptable_candidate(candidate, max_error_m=3.0):
    """Engineering distance gate, not a calibrated match probability or GT tuning."""
    return (candidate is not None and np.isfinite(candidate['window_score_m'])
            and candidate['window_score_m'] <= max_error_m)


def dense_refine(reference, broadcast, ref_teams, b_teams, candidates, minimum, radius=50):
    """Use all 101 relative frames; do not favor candidates with very sparse evidence."""
    best = None
    # Full assignment at every frame is expensive: coarse global screening first.
    for event in candidates[:3]:
        flip = event['xy_reflection_index']
        transformed = reflected(broadcast, FLIPS[flip], GAMES[GAME]['pitch'])
        center = event['broadcast_atd_frame0']
        for candidate in range(max(radius, center - radius),
                               min(len(broadcast) - radius, center + radius + 1)):
            center_error = layout_error(reference[radius], transformed[candidate],
                                        ref_teams, b_teams, minimum)
            if center_error is None:
                continue
            scores = []
            for offset in range(-radius, radius + 1):
                error = layout_error(reference[radius + offset], transformed[candidate + offset],
                                     ref_teams, b_teams, minimum)
                if error is not None:
                    scores.append(error['score_m'])
            if len(scores) < radius + 1:
                continue
            score = float(np.mean(scores))
            if best is None or score < best['window_score_m']:
                best = {'broadcast_atd_frame0': candidate, 'xy_reflection_index': flip,
                        'window_score_m': score, 'valid_context_frames': len(scores),
                        'context_total_frames': 2 * radius + 1, **center_error}
    return best


def build_mapping(blocks, anchors, tactical_count, broadcast_count):
    mapping = np.full(tactical_count, -1, dtype=np.int32)
    halves = np.zeros(tactical_count, dtype=np.int8)
    status = np.full(tactical_count, 'outside_evaluation', dtype='<U40')
    previous_end = -1
    for half, first, last in blocks:
        halves[first:last + 1] = half
        status[first:last + 1] = 'no_anchor'
        start = anchors.get(half)
        if start is None:
            continue
        if not 0 <= start < broadcast_count or start <= previous_end:
            status[first:last + 1] = 'invalid_or_overlapping_anchor'
            continue
        frames = np.arange(first, last + 1)
        targets = start + frames - first
        available = targets < broadcast_count
        mapping[frames[available]] = targets[available]
        status[frames[available]] = 'mapped_by_half_anchor_assumption'
        status[frames[~available]] = 'broadcast_exhausted'
        previous_end = int(targets[available][-1]) if available.any() else previous_end
    return mapping, halves, status


def main():
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument('--data-dir', help='Local bundle root.')
    parser.add_argument('--max-window-error-m', type=float, default=3.0,
                        help='Reject distant point-set windows (default: 3m; not GT-fitted).')
    args = parser.parse_args()
    try:
        if not np.isfinite(args.max_window_error_m) or args.max_window_error_m <= 0:
            raise ValueError('Maximum error must be positive and finite.')
        paths = {source: asset_paths(source=source, data_dir=args.data_dir) for source in ('tactical', 'broadcast')}
        teams = {source: track_teams(paths[source]['atd_meta']) for source in paths}
        broadcast = read_positions(paths['broadcast']['atd_raw'], teams['broadcast'], source_frames('broadcast'))
        blocks = half_video_blocks(GAME)
        anchors, reports = {}, []
        earliest = 0
        for half, first, last in blocks:
            reference = read_positions(paths['tactical']['atd_raw'], teams['tactical'],
                                       source_frames('tactical'), first - 50, first + 51)
            counts = {side: len(team_points(reference[50], teams['tactical'], side)) for side in ('H', 'A')}
            entry = {'half': half, 'tactical_start': first, 'tactical_end': last,
                     'requested_half_frames': last - first + 1, 'reference_center_observations': counts}
            if min(counts.values()) < 3 or len(broadcast) - earliest < 101:
                entry['status'] = 'insufficient_reference_or_broadcast'
                reports.append(entry)
                continue
            minimum = min(6, min(counts.values()))
            candidates = find_matches(reference, 50, broadcast[earliest:], teams['tactical'],
                                      teams['broadcast'], fps=25, minimum=minimum, events_per_flip=4)
            best = dense_refine(reference, broadcast[earliest:], teams['tactical'], teams['broadcast'],
                                candidates, minimum)
            if best is None:
                entry['status'] = 'no_matching_window'
            else:
                best['broadcast_atd_frame0'] += earliest
                start = best['broadcast_atd_frame0']
                entry.update(best)
                entry['minimum_observations_per_team'] = minimum
                entry['accepted_anchor'] = acceptable_candidate(best, args.max_window_error_m)
                if entry['accepted_anchor']:
                    anchors[half] = start
                    entry['status'] = 'automatic_anchor_candidate'
                    entry['quality_note'] = 'coordinate_similarity_only_not_live_verified'
                    entry['broadcast_end_if_complete'] = start + last - first
                    # Half two cannot reuse an earlier first-half broadcast interval.
                    earliest = min(len(broadcast), start + last - first + 1)
                else:
                    entry['status'] = 'rejected_low_similarity'
                    entry['quality_note'] = 'best_available_candidate_is_not_a_validated_correspondence'
            reports.append(entry)
            print(f'Processed half {half} automatic anchor search; no coordinate rows logged.', flush=True)
        mapping, halves, status = build_mapping(blocks, anchors, source_frames('tactical'), len(broadcast))
        for entry in reports:
            selected = halves == entry['half']
            entry['mapped_frames'] = int(((mapping >= 0) & selected).sum())
            entry['unmatched_evaluation_frames'] = int(((mapping < 0) & selected).sum())
        report = {'radius_frames': 50, 'index_basis': 'zero_based_ATD_not_verified_MP4',
                  'max_window_error_m': args.max_window_error_m,
                  'verified_synchronization': False, 'halves': reports,
                  'total_evaluation_frames': sum(last - first + 1 for _, first, last in blocks),
                  'mapped_frames': int((mapping >= 0).sum()),
                  'method': 'automatic_half_start_point_set_matching_then_constant_offset_each_half'}
        FOLDER.mkdir(parents=True, exist_ok=True)
        with (FOLDER / 'mapping.csv').open('w', newline='', encoding='utf-8') as handle:
            writer = csv.writer(handle)
            writer.writerow(['tactical_frame', 'broadcast_atd_frame0', 'half', 'status'])
            writer.writerows((index, int(target) if target >= 0 else '', int(half), str(state))
                             for index, (target, half, state) in enumerate(zip(mapping, halves, status)))
        (FOLDER / 'report.json').write_text(json.dumps(report, indent=2) + '\n', encoding='utf-8')
    except (OSError, ValueError, IndexError, ET.ParseError):
        print('Local half alignment failed. No coordinate rows logged.', file=sys.stderr)
        return 1
    print(json.dumps(report, indent=2), flush=True)
    return 0


if __name__ == '__main__':
    raise SystemExit(main())
