"""Final frame/half annotations tested using synthetic events and coordinates."""

import csv
from pathlib import Path
import tempfile
import unittest

import numpy as np

from model.영상추출.모델.tracking_alignment.final_tracking_alignment import (half_specs, require_supported, first_observed_pair,
                                              write_mapping, start_second_half_at_broadcast_ball, write_source_labels)
from model.영상추출.모델.tracking_alignment.compare_tracking_player import read_half_map


class FinalTrackingAlignmentTests(unittest.TestCase):
    def setUp(self):
        self.blocks = [(1, 2, 5), (2, 9, 17)]
        self.specs = half_specs(self.blocks, [
            {'start_frame': 14, 'tags': {'Half': '2nd Half'}},
            {'start_frame': 11, 'tags': {'Half': ' 2ND  Half '}},
            {'start_frame': 0, 'tags': {'Half': '1st Half'}},
        ])
        self.anchors = {1: {'broadcast_atd_frame0': 1, 'xy_reflection_index': 0},
                        2: {'broadcast_atd_frame0': 7, 'xy_reflection_index': 0}}

    def test_first_official_start_and_earliest_second_tag_are_independent(self):
        self.assertEqual([(s['first'], s['last'], s['tag']) for s in self.specs],
                         [(2, 5, '1st Half'), (11, 17, '2nd Half')])
        self.assertEqual(self.blocks[1][1], 9)
        for events in ([], [{'start_frame': 3, 'tags': {'Half': '2nd Half'}}]):
            with self.assertRaises(ValueError):
                half_specs(self.blocks, events)

    def test_final_csv_tags_both_sources_keeps_ids_and_loads_in_existing_player(self):
        with tempfile.TemporaryDirectory() as folder:
            output = Path(folder) / 'final.csv'
            counts = write_mapping(output, self.blocks, self.specs, self.anchors, 20, 15)
            with output.open() as handle:
                rows = list(csv.DictReader(handle))
            tf, bf, halves = read_half_map(output, 20, 15, self.blocks)
        self.assertEqual(len(rows), 20)
        self.assertEqual(counts, {'1st Half': 4, '2nd Half': 7})
        self.assertEqual(rows[2]['tactical_half_tag'], '1st Half')
        self.assertEqual(rows[2]['broadcast_half_tag'], '1st Half')
        self.assertEqual(rows[2]['broadcast_atd_frame_id'], '2')
        self.assertEqual(rows[2]['tactical_label'], '')
        self.assertEqual(rows[2]['broadcast_label'], '')
        self.assertEqual(rows[11]['tactical_half_tag'], '2nd Half')
        self.assertEqual(rows[17]['broadcast_atd_frame0'], '13')
        for frame in [0, 6, 9, 10, 18]:
            self.assertEqual(rows[frame]['tactical_half_tag'], '')
            self.assertEqual(rows[frame]['broadcast_half_tag'], '')
            self.assertEqual(rows[frame]['broadcast_atd_frame0'], '')
            self.assertEqual(rows[frame]['tactical_label'], 'X')
            self.assertEqual(rows[frame]['broadcast_label'], '')  # No corresponding broadcast row.
        self.assertEqual(rows[9]['status'], 'before_smart_tagging_half_start')
        self.assertEqual(bf[tf == 9].tolist(), [-1])
        self.assertEqual(bf[tf == 11].tolist(), [7])
        self.assertEqual(halves[tf == 9].tolist(), [2])

    def test_exhausted_broadcast_is_blank_and_does_not_repeat_last_frame(self):
        with tempfile.TemporaryDirectory() as folder:
            output = Path(folder) / 'final.csv'
            counts = write_mapping(output, self.blocks, self.specs, self.anchors, 20, 10)
            with output.open() as handle:
                rows = list(csv.DictReader(handle))
        self.assertEqual(counts['2nd Half'], 3)
        self.assertEqual(rows[13]['broadcast_atd_frame0'], '9')
        self.assertEqual(rows[14]['broadcast_atd_frame0'], '')
        self.assertEqual(rows[14]['broadcast_half_tag'], '')
        self.assertEqual(rows[14]['tactical_half_tag'], '2nd Half')
        self.assertEqual(rows[14]['status'], 'broadcast_exhausted')

    def test_overlapping_anchors_fail_without_replacing_previous_file(self):
        self.anchors[2]['broadcast_atd_frame0'] = 4
        with tempfile.TemporaryDirectory() as folder:
            output = Path(folder) / 'final.csv'
            output.write_text('previous result')
            with self.assertRaises(ValueError):
                write_mapping(output, self.blocks, self.specs, self.anchors, 20, 15)
            self.assertEqual(output.read_text(), 'previous result')

    def test_ball_and_players_and_later_ball_are_all_required(self):
        best = {'ball_rmse_m': 2.5, 'player_score_m': 2., 'validation_ball_rmse_m': 3.,
                'validation_ball_consistent': True, 'player_samples': 20}
        require_supported(best)
        for key, value in [('ball_rmse_m', 8.), ('player_score_m', 9.),
                           ('validation_ball_rmse_m', None), ('validation_ball_consistent', False),
                           ('player_samples', 0)]:
            with self.assertRaises(ValueError):
                require_supported({**best, key: value})

    def test_first_jointly_observed_pair_is_selected_by_availability(self):
        teams = ['H']*6+['A']*6
        tactical = np.full((3, 13, 2), .5, dtype=np.float32)
        broadcast = tactical.copy()
        broadcast[0, -1] = np.nan
        broadcast[1, 0] = np.nan
        self.assertEqual(first_observed_pair(tactical, broadcast, teams, teams, 100, 60),
                         {'tactical_frame': 102, 'broadcast_atd_frame0': 62})

    def test_second_tag_starts_at_broadcast_ball_only_and_preserves_offset_and_end(self):
        broadcast = np.full((15, 3, 2), np.nan, dtype=np.float32)
        broadcast[8, -1] = [.5, np.nan]  # Partial coordinate is still missing.
        broadcast[9, -1] = [.5, .4]     # No player observations are required.
        original_spec = self.specs[1].copy()
        original_anchor = self.anchors[2].copy()
        spec, anchor = start_second_half_at_broadcast_ball(original_spec, original_anchor, broadcast)
        self.assertEqual((spec['first'], anchor['broadcast_atd_frame0']), (13, 9))
        self.assertEqual(spec['first']-anchor['broadcast_atd_frame0'], 11-7)
        self.assertEqual(anchor['broadcast_atd_frame_id'], 10)
        self.assertEqual(spec['last'], 17)
        self.assertEqual(original_spec, self.specs[1])
        self.assertEqual(original_anchor, self.anchors[2])
        with tempfile.TemporaryDirectory() as folder:
            output = Path(folder) / 'final.csv'
            counts = write_mapping(output, self.blocks, [self.specs[0], spec],
                                   {1: self.anchors[1], 2: anchor}, 20, 15)
            with output.open() as handle:
                rows = list(csv.DictReader(handle))
        self.assertEqual(counts, {'1st Half': 4, '2nd Half': 5})
        self.assertEqual(rows[12]['tactical_half_tag'], '')
        self.assertEqual(rows[12]['status'], 'before_broadcast_ball_half_start')
        self.assertEqual(rows[13]['half_frame'], '0')
        self.assertEqual(rows[14]['broadcast_half_tag'], '2nd Half')  # Later missing balls keep tag.
        self.assertEqual(rows[17]['broadcast_atd_frame0'], '13')

    def test_ball_at_anchor_needs_no_shift_and_missing_whole_half_fails(self):
        broadcast = np.full((15, 3, 2), np.nan, dtype=np.float32)
        # An observation outside the mapped end must not extend this half.
        broadcast[14, -1] = [.5, .5]
        with self.assertRaises(ValueError):
            start_second_half_at_broadcast_ball(self.specs[1], self.anchors[2], broadcast)
        broadcast[7, -1] = [.5, .5]
        spec, anchor = start_second_half_at_broadcast_ball(self.specs[1], self.anchors[2], broadcast)
        self.assertEqual((spec['first'], anchor['broadcast_atd_frame0']), (11, 7))
        first_spec, first_anchor = start_second_half_at_broadcast_ball(self.specs[0], self.anchors[1], broadcast)
        self.assertEqual(first_spec, self.specs[0])
        self.assertEqual(first_anchor, self.anchors[1])

    def test_full_source_label_tables_assign_x_only_outside_half_tags(self):
        with tempfile.TemporaryDirectory() as folder:
            output = Path(folder) / 'final.csv'
            write_mapping(output, self.blocks, self.specs, self.anchors, 20, 15)
            summary = write_source_labels(output, 20, 15)
            for source, count, tagged in [('tactical', 20, set(range(2, 6)) | set(range(11, 18))),
                                           ('broadcast', 15, set(range(1, 5)) | set(range(7, 14)))]:
                with (Path(folder) / summary[source]['file']).open() as handle:
                    rows = list(csv.DictReader(handle))
                self.assertEqual(len(rows), count)
                for frame, row in enumerate(rows):
                    self.assertEqual(int(row['frame']), frame)
                    self.assertEqual(int(row['atd_frame_id']), frame+1)
                    self.assertEqual(row['label'], '' if frame in tagged else 'X')
                    self.assertEqual(bool(row['half_tag']), frame in tagged)
                self.assertEqual(summary[source]['X_frames'], count-len(tagged))
                self.assertEqual(summary[source]['pending_H_A_D_frames'], len(tagged))

    def test_label_preparation_rejects_missing_tactical_rows(self):
        with tempfile.TemporaryDirectory() as folder:
            output = Path(folder) / 'final.csv'
            write_mapping(output, self.blocks, self.specs, self.anchors, 20, 15)
            with self.assertRaises(ValueError):
                write_source_labels(output, 21, 15)
            self.assertFalse((Path(folder) / 'final_tracking_tactical_labels.csv').exists())


if __name__ == '__main__':
    unittest.main()
