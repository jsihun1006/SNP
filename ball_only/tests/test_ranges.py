"""Validate the full label table before deriving source-video half ranges."""
import csv
from decimal import Decimal
import json
from pathlib import Path
import subprocess
import sys
import tempfile
import unittest

from ball_only import ranges_from_alignment as ranges


class RangeTests(unittest.TestCase):
    def setUp(self):
        temp_root = Path(r'H:\SNP-GSR\.runtime\tmp').resolve()
        temp_root.mkdir(parents=True, exist_ok=True)
        self.assertTrue(temp_root.is_dir())
        self.temporary = tempfile.TemporaryDirectory(prefix='ball-only-ranges-test-', dir=temp_root)
        self.folder = Path(self.temporary.name).resolve()
        # Verify the scope before TemporaryDirectory performs recursive cleanup.
        self.assertEqual(self.folder.parent, temp_root)
        self.addCleanup(self.temporary.cleanup)

    @staticmethod
    def rows():
        tags = ['', '1st Half', '1st Half', '1st Half', '', '',
                '2nd Half', '2nd Half', '2nd Half', '', '']
        return [dict(frame=index, half_tag=tag, label='H' if tag else 'X')
                for index, tag in enumerate(tags)]

    def labels(self, rows=None, *, fieldnames=('frame', 'half_tag', 'label')):
        path = self.folder / 'labels.csv'
        with path.open('w', encoding='utf-8', newline='') as stream:
            writer = csv.DictWriter(stream, fieldnames=fieldnames, extrasaction='ignore')
            writer.writeheader()
            writer.writerows(self.rows() if rows is None else rows)
        return path

    @staticmethod
    def seconds(timestamp):
        hours, minutes, seconds = timestamp.split(':')
        return Decimal(hours) * 3600 + Decimal(minutes) * 60 + Decimal(seconds)

    def test_full_table_produces_two_exclusive_ranges_and_cautious_metadata(self):
        result = ranges.ranges_from_labels(self.labels())
        self.assertEqual(result['fps'], 25)
        self.assertEqual(result['input_label_rows'], 11)
        self.assertEqual(result['ignored_trailing_label_rows'], 0)
        self.assertIs(result['alignment_verified_full_half'], False)
        self.assertIs(result['broadcast_mp4_alignment_verified'], False)
        self.assertEqual(result['source'], 'main final_tracking_alignment')
        self.assertEqual(result['jobs'], [
            dict(part='first_half', start_frame=1, end_frame=4,
                 start='00:00:00.04', end='00:00:00.16'),
            dict(part='second_half', start_frame=6, end_frame=9,
                 start='00:00:00.24', end='00:00:00.36'),
        ])

    def test_extra_final_atd_row_is_ignored_only_at_known_video_end(self):
        result = ranges.ranges_from_labels(self.labels(), video_frames=10)
        self.assertEqual(result['video_frames'], 10)
        self.assertEqual(result['input_label_rows'], 11)
        self.assertEqual(result['ignored_trailing_label_rows'], 1)
        self.assertEqual([(job['start_frame'], job['end_frame']) for job in result['jobs']],
                         [(1, 4), (6, 9)])

    def test_known_video_end_clips_second_half_and_reports_the_clip(self):
        result = ranges.ranges_from_labels(self.labels(), video_frames=8)
        self.assertEqual(result['ignored_trailing_label_rows'], 3)
        self.assertEqual(result['clipped_part_ids'], ['second_half'])
        self.assertEqual(result['jobs'][1],
                         dict(part='second_half', start_frame=6, end_frame=8,
                              start='00:00:00.24', end='00:00:00.32'))

    def test_clipping_must_preserve_both_nonempty_halves(self):
        path = self.labels()
        for video_frames in (1, 3, 6):
            with self.subTest(video_frames=video_frames):
                with self.assertRaises(ValueError):
                    ranges.ranges_from_labels(path, video_frames=video_frames)
        one_frame_second_half = ranges.ranges_from_labels(path, video_frames=7)
        self.assertEqual(one_frame_second_half['jobs'][1]['end_frame'], 7)

    def test_table_must_cover_all_known_video_frames(self):
        with self.assertRaises(ValueError):
            ranges.ranges_from_labels(self.labels(), video_frames=12)

    def test_frame_timestamps_remain_exact_across_minute_boundaries(self):
        rows = [dict(frame=index, half_tag='', label='X') for index in range(1505)]
        for index, tag in ((1, '1st Half'), (2, '1st Half'),
                           (1501, '2nd Half'), (1502, '2nd Half')):
            rows[index].update(half_tag=tag, label='H')
        result = ranges.ranges_from_labels(self.labels(rows), fps=25)
        self.assertEqual(result['jobs'][1]['start'], '00:01:00.04')
        self.assertEqual(result['jobs'][1]['end'], '00:01:00.12')
        for job in result['jobs']:
            for edge in ('start', 'end'):
                self.assertEqual(self.seconds(job[edge]) * Decimal(25), job[edge + '_frame'])

    def test_fifty_fps_preserves_frame_boundaries(self):
        result = ranges.ranges_from_labels(self.labels(), fps=50)
        self.assertEqual(result['jobs'][0]['start'], '00:00:00.02')
        self.assertEqual(result['jobs'][0]['end'], '00:00:00.08')
        for job in result['jobs']:
            for edge in ('start', 'end'):
                self.assertEqual(self.seconds(job[edge]) * Decimal(50), job[edge + '_frame'])

    def test_duplicate_frame_cannot_assign_a_source_frame_to_two_halves(self):
        rows = self.rows()
        rows.insert(3, dict(frame=2, half_tag='2nd Half', label='H'))
        with self.assertRaises(ValueError):
            ranges.ranges_from_labels(self.labels(rows))

    def test_frames_must_be_a_zero_based_contiguous_full_table(self):
        cases = {}
        cases['missing_origin'] = self.rows()[1:]
        cases['internal_gap'] = [row for row in self.rows() if row['frame'] != 4]
        cases['negative_origin'] = self.rows()
        cases['negative_origin'][0]['frame'] = -1
        cases['noninteger_frame'] = self.rows()
        cases['noninteger_frame'][2]['frame'] = '2.5'
        cases['nonnumeric_frame'] = self.rows()
        cases['nonnumeric_frame'][2]['frame'] = 'unknown'
        for name, rows in cases.items():
            with self.subTest(case=name):
                with self.assertRaises(ValueError):
                    ranges.ranges_from_labels(self.labels(rows))

    def test_each_half_must_be_one_contiguous_block(self):
        for half, gap_frame in (('1st Half', 2), ('2nd Half', 7)):
            with self.subTest(half=half):
                rows = self.rows()
                rows[gap_frame].update(half_tag='', label='X')
                with self.assertRaises(ValueError):
                    ranges.ranges_from_labels(self.labels(rows))

    def test_first_half_must_precede_second_half(self):
        rows = self.rows()
        for row in rows:
            if row['half_tag']:
                row['half_tag'] = ('2nd Half' if row['half_tag'] == '1st Half' else '1st Half')
        with self.assertRaises(ValueError):
            ranges.ranges_from_labels(self.labels(rows))

    def test_exactly_two_recognized_nonempty_halves_are_required(self):
        for missing in ('1st Half', '2nd Half', 'both'):
            with self.subTest(missing=missing):
                rows = self.rows()
                for row in rows:
                    if row['half_tag'] == missing or missing == 'both':
                        row.update(half_tag='', label='X')
                with self.assertRaises(ValueError):
                    ranges.ranges_from_labels(self.labels(rows))
        rows = self.rows()
        rows[-1].update(half_tag='3rd Half', label='H')
        with self.assertRaises(ValueError):
            ranges.ranges_from_labels(self.labels(rows))

    def test_tagged_x_label_and_untagged_non_x_label_are_rejected(self):
        for frame, label in ((2, 'X'), (0, 'H'), (0, '')):
            with self.subTest(frame=frame, label=label):
                rows = self.rows()
                rows[frame]['label'] = label
                with self.assertRaises(ValueError):
                    ranges.ranges_from_labels(self.labels(rows))

    def test_invalid_trailing_rows_are_validated_before_video_clipping(self):
        rows = self.rows()
        rows[-1]['label'] = 'H'
        with self.assertRaises(ValueError):
            ranges.ranges_from_labels(self.labels(rows), video_frames=10)

    def test_required_columns_and_empty_tables_are_rejected(self):
        for missing in ('frame', 'half_tag', 'label'):
            with self.subTest(missing_column=missing):
                columns = [name for name in ('frame', 'half_tag', 'label') if name != missing]
                with self.assertRaises(ValueError):
                    ranges.ranges_from_labels(self.labels(fieldnames=columns))
        with self.assertRaises(ValueError):
            ranges.ranges_from_labels(self.labels([]))

    def test_invalid_fps_and_video_counts_are_rejected(self):
        path = self.labels()
        for fps in (0, -25, float('inf'), float('nan'), 'invalid'):
            with self.subTest(fps=fps):
                with self.assertRaises(ValueError):
                    ranges.ranges_from_labels(path, fps=fps)
        for video_frames in (0, -1, 7.5):
            with self.subTest(video_frames=video_frames):
                with self.assertRaises(ValueError):
                    ranges.ranges_from_labels(path, video_frames=video_frames)

    def test_cli_writes_the_same_configuration_and_never_rewrites_labels(self):
        labels = self.labels()
        before = labels.read_bytes()
        output = self.folder / 'ranges.json'
        package_root = Path(ranges.__file__).resolve().parent.parent
        completed = subprocess.run(
            [sys.executable, '-B', '-m', 'ball_only.ranges_from_alignment',
             '--labels', str(labels), '--fps', '25', '--video-frames', '10',
             '--out', str(output)],
            cwd=package_root, capture_output=True, text=True, timeout=30,
        )
        self.assertEqual(completed.returncode, 0, completed.stdout + completed.stderr)
        self.assertEqual(json.loads(output.read_text(encoding='utf-8')),
                         ranges.ranges_from_labels(labels, fps=25, video_frames=10))
        self.assertEqual(labels.read_bytes(), before)


if __name__ == '__main__':
    unittest.main()
