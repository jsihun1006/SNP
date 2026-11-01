"""Find a forward tactical/broadcast match using BOTH ball motion and team layouts.

    python starter/find_joint_tracking_match.py --tactical-frame 113660

Search all broadcast ATD start indices at one-frame resolution using masked ball
trajectory correlation over the next 30s. Missing balls never become (0,0), and
low common coverage is rejected. Consider four coordinate reflections. Rank
spatially separated events with independent team point-set assignments sampled
at 1Hz, refine nearby start indices, then check ball consistency in the NEXT 30s.

This queries tracking only, not possession GT or video. It does not change any
existing mapping, prediction, comparison-player default or source data. Missing
camera observations, tracking errors, replays and time edits can still invalidate
an offset. Results refer to zero-based ATD records, not verified MP4 positions.
All detailed outputs remain private in data/derived/sync/.
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
from starter.find_layout_match import layout_error  # noqa: E402
from starter.tracking_utils import track_teams, read_tracking, FLIPS, reflected  # noqa: E402


def correlate_valid(signal, kernel):
    """Sliding dot product; FFT convolution with reversed kernel, no scipy needed."""
    signal, kernel = np.asarray(signal, dtype=float), np.asarray(kernel, dtype=float)
    if not 0 < len(kernel) <= len(signal):
        raise ValueError('Correlation requires a nonempty template no longer than the signal.')
    size = 1 << (len(signal) + len(kernel) - 2).bit_length()
    convolution = np.fft.irfft(np.fft.rfft(signal, size) * np.fft.rfft(kernel[::-1], size), size)
    return convolution[len(kernel) - 1:len(signal)]


def ball_curve(reference, broadcast, minimum_common=20, minimum_fraction=0.25):
    """Every integer start's ball RMSE and observed overlap, with no imputation."""
    reference = np.asarray(reference, dtype=float)
    broadcast = np.asarray(broadcast, dtype=float)
    ref_valid = np.isfinite(reference).all(axis=1)
    b_valid = np.isfinite(broadcast).all(axis=1)
    ref = np.where(ref_valid[:, None], reference, 0)
    b = np.where(b_valid[:, None], broadcast, 0)
    ref_mask, b_mask = ref_valid.astype(float), b_valid.astype(float)
    common = np.rint(correlate_valid(b_mask, ref_mask)).astype(int)
    sum_squared = (correlate_valid(np.sum(b*b, axis=1), ref_mask)
                   + correlate_valid(b_mask, np.sum(ref*ref, axis=1)))
    for axis in (0, 1):
        sum_squared -= 2 * correlate_valid(b[:, axis], ref[:, axis])
    rmse = np.sqrt(np.maximum(sum_squared, 0) / np.maximum(common, 1))
    observed = int(ref_valid.sum())
    required = max(minimum_common, math.ceil(observed * minimum_fraction))
    rmse[common < required] = np.inf
    if observed == 0:
        rmse[:] = np.inf
    coverage = common / max(observed, 1)
    return {'rmse': rmse, 'common': common, 'reference_observed': observed,
            'required_common': required, 'score': rmse + 5 * (1 - coverage)}


def clean_ball(points):
    points = np.asarray(points, dtype=np.float32).copy()
    valid = np.isfinite(points).all(axis=1) & ((points >= 0) & (points <= 1)).all(axis=1)
    points[~valid] = np.nan
    return points * np.asarray(GAMES[GAME]['pitch'], dtype=np.float32)


def player_window_score(reference, broadcast, ref_teams, b_teams, start, flip, window, fps):
    pitch = np.asarray(GAMES[GAME]['pitch'], dtype=np.float32)
    scores, errors, matched = [], [], []
    for offset in range(0, window, fps):
        target = reference[offset, :-1] * pitch
        candidate = reflected(broadcast[start + offset, :-1] * pitch, FLIPS[flip], pitch)
        error = layout_error(target, candidate, ref_teams, b_teams, minimum=6)
        if error is not None:
            scores.append(error['score_m'])
            errors.append(error['rmse_m'])
            matched.append(error['matched_players'])
    return {'player_samples': len(scores),
            'player_score_m': float(np.mean(scores)) if scores else None,
            'player_raw_rmse_m': float(np.mean(errors)) if errors else None,
            'mean_matched_player_observations': float(np.mean(matched)) if matched else None}


def find_joint(reference, broadcast, ref_teams, b_teams, window=750, fps=25,
               previous_frame=None, events_per_flip=6):
    if len(reference) < 2 * window or len(broadcast) < 2 * window:
        raise ValueError('Both timelines need matching and validation windows.')
    ref_ball, b_ball = clean_ball(reference[:, -1]), clean_ball(broadcast[:, -1])
    observed = ref_ball[:window][np.isfinite(ref_ball[:window]).all(axis=1)]
    if len(observed) < 20 or np.linalg.norm(np.ptp(observed, axis=0)) < 5:
        raise ValueError('Reference ball motion is missing or too stationary for reliable sequence search.')
    limit = len(broadcast) - 2 * window + 1
    curves, candidates = [], []
    for flip, direction in enumerate(FLIPS):
        ball = reflected(b_ball, direction, GAMES[GAME]['pitch'])
        training = ball_curve(ref_ball[:window], ball)
        validation = ball_curve(ref_ball[window:2*window], ball)
        curves.append((training, validation))
        scores = training['score'][:limit].copy()
        for _ in range(events_per_flip):
            start = int(np.argmin(scores))
            if not np.isfinite(scores[start]):
                break
            candidates.append((start, flip))
            scores[max(0, start-5*fps):min(limit, start+5*fps+1)] = np.inf
    cache = {}

    def evaluate(start, flip):
        key = (start, flip)
        if key in cache:
            return cache[key]
        if not 0 <= start < limit:
            return None
        training, validation = curves[flip]
        if not np.isfinite(training['rmse'][start]):
            return None
        players = player_window_score(reference, broadcast, ref_teams, b_teams, start, flip, window, fps)
        if players['player_samples'] < 5:
            return None
        val_index = start + window
        val_error = float(validation['rmse'][val_index])
        entry = {'broadcast_atd_frame0': start, 'broadcast_atd_frame_id': start+1,
                 'xy_reflection_index': flip,
                 'ball_rmse_m': float(training['rmse'][start]),
                 'ball_common_frames': int(training['common'][start]),
                 'reference_ball_frames': training['reference_observed'],
                 'validation_ball_rmse_m': val_error if np.isfinite(val_error) else None,
                 'validation_common_frames': int(validation['common'][val_index]),
                 'validation_reference_ball_frames': validation['reference_observed'],
                 'validation_ball_consistent': bool(np.isfinite(val_error) and val_error <= 5.0),
                 **players,
                 'joint_score_m': float(training['score'][start]) + players['player_score_m']}
        cache[key] = entry
        return entry

    preliminary = [entry for start, flip in candidates if (entry := evaluate(start, flip)) is not None]
    preliminary.sort(key=lambda entry: entry['joint_score_m'])
    if not preliminary:
        raise ValueError('No candidates with sufficient common ball and player evidence.')
    # Select using the first window only. The subsequent ball window is a separate check.
    for entry in preliminary[:3]:
        center, flip = entry['broadcast_atd_frame0'], entry['xy_reflection_index']
        for start in range(max(0, center-fps), min(limit, center+fps+1)):
            evaluate(start, flip)
    ranked = sorted(cache.values(), key=lambda entry: entry['joint_score_m'])
    distinct = []
    for entry in ranked:
        if all(abs(entry['broadcast_atd_frame0'] - other['broadcast_atd_frame0']) > 5*fps for other in distinct):
            distinct.append(entry)
    best = distinct[0]
    profile = [entry for entry in ranked if entry['xy_reflection_index'] == best['xy_reflection_index']
               and abs(entry['broadcast_atd_frame0']-best['broadcast_atd_frame0']) <= fps]
    previous = evaluate(previous_frame, 0) if previous_frame is not None else None
    return {'best_candidate': best, 'distinct_candidates': distinct[:5],
            'previous_candidate_same_metric': previous, 'nearby_score_profile': profile,
            'matching_frames': window, 'validation_frames': window,
            'player_sampling_stride_frames': fps, 'validation_rmse_limit_m': 5.0,
            'live_replay_verified': False, 'video_alignment_verified': False}


def main():
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument('--tactical-frame', type=int, default=113660)
    parser.add_argument('--seconds', type=int, default=30)
    parser.add_argument('--previous-broadcast-frame', type=int, default=86788,
                        help='Audit the old candidate after search; it does not seed/rank the search.')
    parser.add_argument('--data-dir')
    args = parser.parse_args()
    try:
        fps = int(GAMES[GAME]['fps'])
        window = args.seconds * fps
        if window < 50 or not 0 <= args.tactical_frame < args.tactical_frame + 2*window <= source_frames('tactical'):
            raise ValueError('Invalid forward matching/validation windows.')
        paths = {source: asset_paths(source=source, data_dir=args.data_dir) for source in ('tactical', 'broadcast')}
        teams = {source: track_teams(paths[source]['atd_meta']) for source in paths}
        reference = read_tracking(paths['tactical']['atd_raw'], teams['tactical'], source_frames('tactical'),
                                  args.tactical_frame, 2*window)
        broadcast = read_tracking(paths['broadcast']['atd_raw'], teams['broadcast'], source_frames('broadcast'),
                                  0, source_frames('broadcast'))
        print('Loaded local forward tracking windows; no coordinate rows logged.', flush=True)
        result = find_joint(reference, broadcast, teams['tactical'], teams['broadcast'], window, fps,
                            args.previous_broadcast_frame)
        result['tactical_frame0'] = args.tactical_frame
        result['method'] = 'masked_forward_ball_trajectory_plus_team_point_assignment_then_later_ball_check'
        folder = ROOT / 'data/derived/sync'
        folder.mkdir(parents=True, exist_ok=True)
        stem = f'joint_query_{args.tactical_frame:06d}'
        (folder / (stem+'.json')).write_text(json.dumps(result, indent=2, allow_nan=False)+'\n', encoding='utf-8')
        with (folder / (stem+'_offset_profile.csv')).open('w', newline='', encoding='utf-8') as handle:
            writer = csv.DictWriter(handle, fieldnames=list(result['best_candidate']))
            writer.writeheader()
            writer.writerows(result['nearby_score_profile'])
    except (OSError, ValueError, IndexError, ET.ParseError):
        print('Local joint tracking search failed; no coordinate rows logged.', file=sys.stderr)
        return 1
    summary = {key: value for key, value in result.items() if key not in ('nearby_score_profile',)}
    print(json.dumps(summary, indent=2, allow_nan=False), flush=True)
    return 0


if __name__ == '__main__':
    raise SystemExit(main())
