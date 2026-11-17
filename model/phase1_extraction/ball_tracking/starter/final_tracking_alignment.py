"""Build the final per-half tactical/broadcast frame annotation table locally.

    python starter/final_tracking_alignment.py

Half 1 begins at the official tactical start from games.py. Half 2 is aligned from
the earliest Smart Tagging 'Half=2nd Half' interval, then its tags start at the
first subsequent broadcast ball observation (both coordinates finite).
Both stop at the official inclusive half ends. Search broadcast independently
using forward ball trajectories AND team-wise player assignments. The next ball
window is a separate consistency check. Half 2 cannot reuse half 1 broadcast rows.

Output: data/derived/sync/final_tracking_alignment.csv (one row per tactical frame)
and final_tracking_alignment_report.json. Tags annotate both source frame indices;
join on those indices to use the original positions/speeds. ATD and GT are never
modified. Missing coordinates remain missing. No manual broadcast anchors or GT.
Mapping after each anchor assumes constant timing; full-half sync is unverified.
The CSV also works with compare_tracking_player.py --half-map <CSV path>.
Initial label tables cover EVERY frame of each source: untagged frames are X;
tagged frames are blank pending H/A/D inference. These are preparation tables,
not completed submission files. No possession inference is performed here.
"""

import argparse
import csv
import json
import os
from pathlib import Path
import sys
import tempfile
import xml.etree.ElementTree as ET

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from games import GAME, GAMES, asset_paths, half_video_blocks, source_frames  # noqa: E402
from starter.tracking_utils import track_teams, read_tracking  # noqa: E402
from starter.load_smart_tagging import load as load_tags  # noqa: E402
from starter.find_joint_tracking_match import find_joint, clean_ball  # noqa: E402

OUTPUT = ROOT / 'data/derived/sync/final_tracking_alignment.csv'
COLUMNS = ['tactical_frame', 'tactical_atd_frame_id', 'tactical_half_tag',
           'broadcast_atd_frame0', 'broadcast_atd_frame_id', 'broadcast_half_tag',
           'half', 'status', 'half_frame', 'offset_frames', 'anchor_reference', 'xy_reflection_index',
           'tactical_label', 'broadcast_label']


def half_specs(blocks, events):
    """Separate user half annotations from the unchanged official evaluation mask."""
    if [row[0] for row in blocks] != [1, 2]:
        raise ValueError('Exactly two ordered official halves are required.')
    tagged = [event for event in events
              if ' '.join((event['tags'].get('Half') or '').split()).casefold() == '2nd half']
    if not tagged:
        raise ValueError('No Smart Tagging second-half interval.')
    second_start = min(int(event['start_frame']) for event in tagged)
    if not blocks[1][1] <= second_start <= blocks[1][2]:
        raise ValueError('Smart Tagging second-half start lies outside the official second-half range.')
    specs = []
    for half, first, last in blocks:
        specs.append({'half': half, 'tag': f'{half}st Half' if half == 1 else '2nd Half',
                      'first': first if half == 1 else second_start, 'last': last,
                      'reference_source': 'official_evaluation_start' if half == 1 else 'smart_tagging_first_2nd_half'})
    return specs


def require_supported(best, max_error_m=5.0):
    """Fail before publishing a final map when either ball or player evidence fails."""
    values = [best.get('ball_rmse_m'), best.get('player_score_m'), best.get('validation_ball_rmse_m')]
    if (not np.isfinite(max_error_m) or max_error_m <= 0
            or any(value is None or not np.isfinite(value) or value > max_error_m for value in values)
            or best.get('validation_ball_consistent') is not True or best.get('player_samples', 0) < 5):
        raise ValueError('Joint tracking candidate lacks sufficient independent support.')


def first_observed_pair(reference, broadcast, t_teams, b_teams, t_start, b_start, minimum=6):
    """First jointly available pair, chosen by availability, not by error minimization."""
    valid = np.isfinite(clean_ball(reference[:, -1])).all(axis=1)
    valid &= np.isfinite(clean_ball(broadcast[:, -1])).all(axis=1)
    for points, teams in ((reference, t_teams), (broadcast, b_teams)):
        finite = np.isfinite(points[:, :-1]).all(axis=2)
        for side in ('H', 'A'):
            valid &= finite[:, np.asarray(teams) == side].sum(axis=1) >= minimum
    indices = np.flatnonzero(valid)
    return ({'tactical_frame': int(t_start + indices[0]),
             'broadcast_atd_frame0': int(b_start + indices[0])} if len(indices) else None)


def start_second_half_at_broadcast_ball(spec, anchor, broadcast):
    """Trim only the tag range; preserve the automatically found temporal offset.

    Broadcast ball availability alone selects the first frame. Tactical ball and
    player availability are not additional requirements. Later ball gaps retain
    their half tag. Do not search beyond this half's mapped inclusive end.
    """
    if spec['half'] != 2:
        return spec.copy(), anchor.copy()
    start = anchor['broadcast_atd_frame0']
    stop = min(len(broadcast), start + spec['last'] - spec['first'] + 1)
    indices = np.flatnonzero(np.isfinite(broadcast[start:stop, -1]).all(axis=1))
    if not len(indices):
        raise ValueError('No broadcast ball observation within the mapped second half.')
    shift = int(indices[0])
    updated_spec = {**spec, 'first': spec['first']+shift,
                    'reference_source': 'smart_tagging_alignment_then_first_broadcast_ball'}
    updated_anchor = {**anchor, 'broadcast_atd_frame0': start+shift,
                      'broadcast_atd_frame_id': start+shift+1}
    return updated_spec, updated_anchor


def mapping_rows(blocks, specs, anchors, tactical_count, broadcast_count):
    """Preserve all tactical IDs; leave outside/unavailable broadcast IDs blank."""
    if set(anchors) != {spec['half'] for spec in specs}:
        raise ValueError('Both automatic anchors must exist.')
    previous_end = -1
    for spec in specs:
        b_start = anchors[spec['half']]['broadcast_atd_frame0']
        if not 0 <= spec['first'] <= spec['last'] < tactical_count or not 0 <= b_start < broadcast_count:
            raise ValueError('Invalid half bounds or broadcast anchor.')
        if b_start <= previous_end:
            raise ValueError('Broadcast halves overlap or reverse.')
        previous_end = min(broadcast_count-1, b_start+spec['last']-spec['first'])
    for frame in range(tactical_count):
        official_half = next((half for half, first, last in blocks if first <= frame <= last), 0)
        row = dict.fromkeys(COLUMNS, '')
        row.update(tactical_frame=frame, tactical_atd_frame_id=frame+1, half=official_half, tactical_label='X',
                   status='outside_evaluation' if official_half == 0 else 'before_smart_tagging_half_start')
        if official_half == 2 and frame < specs[1]['first'] and specs[1].get('reference_source') == 'smart_tagging_alignment_then_first_broadcast_ball':
            row['status'] = 'before_broadcast_ball_half_start'
        spec = next((spec for spec in specs if spec['first'] <= frame <= spec['last']), None)
        if spec is not None:
            best = anchors[spec['half']]
            target = best['broadcast_atd_frame0'] + frame - spec['first']
            row.update(tactical_half_tag=spec['tag'], tactical_label='', half_frame=frame-spec['first'],
                       anchor_reference=spec['reference_source'], xy_reflection_index=best['xy_reflection_index'],
                       status='broadcast_exhausted')
            if target < broadcast_count:
                row.update(broadcast_atd_frame0=target, broadcast_atd_frame_id=target+1,
                           broadcast_half_tag=spec['tag'], offset_frames=frame-target,
                           status='mapped_by_half_anchor_assumption')
        yield row


def write_mapping(output, blocks, specs, anchors, tactical_count, broadcast_count):
    """Atomic publication, including uniqueness/monotonicity checks."""
    output = Path(output)
    output.parent.mkdir(parents=True, exist_ok=True)
    temporary = None
    counts = {'1st Half': 0, '2nd Half': 0}
    previous_b = -1
    try:
        with tempfile.NamedTemporaryFile(mode='w', newline='', encoding='utf-8', dir=output.parent,
                                         prefix='.final_alignment_', suffix='.tmp', delete=False) as handle:
            temporary = Path(handle.name)
            writer = csv.DictWriter(handle, fieldnames=COLUMNS)
            writer.writeheader()
            for row in mapping_rows(blocks, specs, anchors, tactical_count, broadcast_count):
                target = row['broadcast_atd_frame0']
                if target != '':
                    if target <= previous_b:
                        raise ValueError('Mapped broadcast frames must be unique and increasing.')
                    previous_b = target
                    counts[row['broadcast_half_tag']] += 1
                writer.writerow(row)
        os.replace(temporary, output)
    finally:
        if temporary is not None:
            temporary.unlink(missing_ok=True)
    return counts


def write_source_labels(mapping_path, tactical_count, broadcast_count):
    """Project annotations to full source timelines, including unpaired broadcast.

    Only absence of an explicit half tag causes X. Ball/player missingness inside
    a tagged half does not assign X. Validate all input indices before writing.
    """
    mapping_path = Path(mapping_path)
    tags = {'tactical': np.full(tactical_count, '', dtype='<U8'),
            'broadcast': np.full(broadcast_count, '', dtype='<U8')}
    rows_seen, previous_b = 0, -1
    with mapping_path.open(newline='', encoding='utf-8') as handle:
        for frame, row in enumerate(csv.DictReader(handle)):
            if frame >= tactical_count or int(row['tactical_frame']) != frame:
                raise ValueError('Label preparation requires the complete ordered tactical timeline.')
            tag = row['tactical_half_tag']
            if tag not in ('', '1st Half', '2nd Half'):
                raise ValueError('Unknown half tag.')
            tags['tactical'][frame] = tag
            value = row['broadcast_atd_frame0']
            if value:
                target = int(value)
                if not previous_b < target < broadcast_count or row['broadcast_half_tag'] != tag or not tag:
                    raise ValueError('Invalid or reused broadcast annotation.')
                tags['broadcast'][target] = tag
                previous_b = target
            elif row['broadcast_half_tag']:
                raise ValueError('Broadcast half tag lacks a source frame.')
            rows_seen += 1
    if rows_seen != tactical_count:
        raise ValueError('Tactical label preparation timeline is incomplete.')
    summary = {}
    for source, source_tags in tags.items():
        output = mapping_path.parent / f'final_tracking_{source}_labels.csv'
        temporary = None
        try:
            with tempfile.NamedTemporaryFile(mode='w', newline='', encoding='utf-8', dir=output.parent,
                                             prefix='.frame_labels_', suffix='.tmp', delete=False) as handle:
                temporary = Path(handle.name)
                writer = csv.writer(handle)
                writer.writerow(['frame', 'atd_frame_id', 'half_tag', 'label'])
                writer.writerows((frame, frame+1, str(tag), '' if tag else 'X')
                                 for frame, tag in enumerate(source_tags))
            os.replace(temporary, output)
        finally:
            if temporary is not None:
                temporary.unlink(missing_ok=True)
        x_count = int((source_tags == '').sum())
        summary[source] = {'file': output.name, 'rows': len(source_tags), 'X_frames': x_count,
                           'pending_H_A_D_frames': len(source_tags)-x_count}
    return summary


def main():
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument('--data-dir')
    args = parser.parse_args()
    try:
        blocks = half_video_blocks(GAME)
        specs = half_specs(blocks, load_tags(data_dir=args.data_dir))
        fps = int(GAMES[GAME]['fps'])
        window = 30 * fps
        paths = {s: asset_paths(source=s, data_dir=args.data_dir) for s in ('tactical', 'broadcast')}
        teams = {s: track_teams(paths[s]['atd_meta']) for s in paths}
        broadcast = read_tracking(paths['broadcast']['atd_raw'], teams['broadcast'], source_frames('broadcast'),
                                  0, source_frames('broadcast'))
        anchors, reports, earliest = {}, [], 0
        for index, spec in enumerate(specs):
            search_first = spec['first']
            if spec['first'] + 2*window - 1 > spec['last']:
                raise ValueError('Insufficient matching and validation context within this half.')
            reference = read_tracking(paths['tactical']['atd_raw'], teams['tactical'], source_frames('tactical'),
                                      spec['first'], 2*window)
            result = find_joint(reference, broadcast[earliest:], teams['tactical'], teams['broadcast'], window, fps)
            best = result['best_candidate'].copy()
            require_supported(best)
            best['broadcast_atd_frame0'] += earliest
            best['broadcast_atd_frame_id'] += earliest
            start = best['broadcast_atd_frame0']
            observed = first_observed_pair(reference[:window], broadcast[start:start+window],
                                           teams['tactical'], teams['broadcast'], spec['first'], start)
            tag_spec, tag_anchor = start_second_half_at_broadcast_ball(spec, best, broadcast)
            specs[index] = tag_spec
            anchors[spec['half']] = tag_anchor
            reports.append({**tag_spec, 'anchor': tag_anchor,
                            'search_reference_tactical_frame': search_first, 'search_anchor': best,
                            'anchor_metrics_note': 'Scores describe the original search window, before trimming the tag start.',
                            'tag_start_trim_frames': tag_spec['first']-search_first,
                            'first_jointly_observed_pair': observed,
                            'broadcast_end_if_constant_offset': start+spec['last']-spec['first'],
                            'search_broadcast_first_index': earliest})
            earliest = min(len(broadcast), start+spec['last']-spec['first']+1)
            print(f"Selected automatic {spec['tag']} anchor; no coordinate rows logged.", flush=True)
        counts = write_mapping(OUTPUT, blocks, specs, anchors, source_frames('tactical'), len(broadcast))
        initial_labels = write_source_labels(OUTPUT, source_frames('tactical'), len(broadcast))
        report = {'index_basis': 'zero_based_ATD_records; atd_frame_id=frame+1',
                  'method': 'per_half_joint_ball_and_player_search_then_constant_offset',
                  'matching_seconds': 30, 'independent_ball_check_seconds': 30, 'maximum_support_error_m': 5,
                  'halves': reports, 'mapped_frames_by_tag': counts, 'mapped_frames': sum(counts.values()),
                  'tactical_rows': source_frames('tactical'),
                  'unpaired_tactical_frames': source_frames('tactical')-sum(counts.values()),
                  'unpaired_broadcast_frames': len(broadcast)-sum(counts.values()),
                  'official_half_blocks_unchanged': blocks, 'possession_gt_used': False,
                  'initial_frame_labels': initial_labels,
                  'initial_label_rule': 'no half tag => X; tagged half => blank pending H/A/D inference',
                  'completed_submission': False,
                  'second_half_tag_start_rule': 'first finite broadcast ball x/y at or after the automatic Smart Tagging alignment anchor',
                  'full_half_synchronization_verified': False, 'broadcast_mp4_alignment_verified': False,
                  'half_column_meaning': 'unchanged official evaluation mask; use the explicit half_tag columns for the requested annotations',
                  'unpaired_broadcast_note': 'Unpaired broadcast indices have no row/tag in this tactical-keyed CSV; source tracking is preserved.'}
        OUTPUT.with_name('final_tracking_alignment_report.json').write_text(
            json.dumps(report, indent=2, allow_nan=False)+'\n', encoding='utf-8')
    except (OSError, ValueError, IndexError, ET.ParseError):
        print('Final alignment failed; insufficient evidence or invalid local inputs. No coordinate rows logged.', file=sys.stderr)
        return 1
    print(json.dumps({'mapped_frames_by_tag': counts, 'initial_frame_labels': initial_labels, 'anchors': [
        {'tag': r['tag'], 'tactical_frame': r['first'],
         'broadcast_atd_frame0': r['anchor']['broadcast_atd_frame0'],
         'first_jointly_observed_pair': r['first_jointly_observed_pair']} for r in reports]}, indent=2), flush=True)
    print('Saved private local alignment, initial source label tables and report.', flush=True)
    return 0


if __name__ == '__main__':
    raise SystemExit(main())
