"""Coordinate player/export checks using synthetic arrays, no competition assets."""

from pathlib import Path
import csv
import tempfile
import unittest
from unittest.mock import Mock, patch

import cv2
import numpy as np

from starter.compare_tracking_player import (
    SIZE, comparison_length, export_video, make_template, read_tracking, render, view,
    read_half_map,
)
from starter.align_half_starts import build_mapping


class TrackingPlayerTests(unittest.TestCase):
    def test_half_map_preserves_original_indices_and_excludes_halftime(self):
        blocks = [(1, 2, 4), (2, 7, 10)]
        mapping, halves, status = build_mapping(blocks, {1: 1, 2: 6}, 12, 9)
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / 'mapping.csv'
            with path.open('w', newline='') as handle:
                writer = csv.writer(handle)
                writer.writerow(['tactical_frame', 'broadcast_atd_frame0', 'half', 'status'])
                writer.writerows((index, int(target) if target >= 0 else '', int(half), str(state))
                                 for index, (target, half, state) in enumerate(zip(mapping, halves, status)))
            tf, bf, half = read_half_map(path, 12, 9, blocks)
            self.assertEqual(tf.tolist(), [2, 3, 4, 7, 8, 9, 10])
            self.assertEqual(bf.tolist(), [1, 2, 3, 6, 7, 8, -1])
            self.assertEqual(half.tolist(), [1, 1, 1, 2, 2, 2, 2])
            # A manual one-frame edit must not silently break the constant offset.
            text = path.read_text().replace('3,2,1,mapped_', '3,1,1,mapped_')
            path.write_text(text)
            with self.assertRaises(ValueError):
                read_half_map(path, 12, 9, blocks)

    def test_range_and_raw_frame_slot_ball_alignment(self):
        self.assertEqual(comparison_length(199319, 164488, 11150, 7150), 157338)
        self.assertEqual(comparison_length(10, 20, 8, 5), 2)
        with self.assertRaises(ValueError):
            comparison_length(10, 20, -1, 5)
        with tempfile.TemporaryDirectory() as directory:
            raw = Path(directory) / 'synthetic.txt'
            raw.write_text('1:0.1,0.2,3;0.8,0.7,2:0.5,0.5,1\n'
                           '2:0.2,0.3,3;NaN,NaN,NaN:0.6,0.5,1\n'
                           '3:0.3,0.4,3;0.7,0.6,2:NaN,NaN,NaN\n')
            data = read_tracking(raw, ['H', 'A'], 3, 1, 2)
            self.assertEqual(data.shape, (2, 3, 2))
            np.testing.assert_allclose(data[0, 0], [0.2, 0.3])
            np.testing.assert_allclose(data[0, -1], [0.6, 0.5])
            self.assertTrue(np.isnan(data[0, 1]).all())
            self.assertTrue(np.isnan(data[1, -1]).all())
            raw.write_text('1:0.1,0.2,3:0.5,0.5,1\n')
            with self.assertRaises(ValueError):
                read_tracking(raw, ['H', 'A'], 3, 1, 2)

    def test_render_and_mp4_preserve_all_frames_and_fps(self):
        points = np.array([[0.1, 0.2], [0.8, 0.7], [0.5, 0.5]], dtype=np.float32)
        teams = ['H', 'A']
        template = make_template()
        frame = render(template, points, points, teams, teams, 11150, 7150)
        self.assertEqual(frame.shape, (SIZE[1], SIZE[0], 3))
        self.assertTrue(np.any(frame != template))
        without_labels = render(template, points, points, teams, teams, 11150, 7150, labels=False)
        self.assertTrue(np.any(frame != without_labels))
        unmatched = render(template, points, points, teams, teams, 110808, -1)
        np.testing.assert_array_equal(unmatched[104:532, 644:1276], template[104:532, 644:1276])
        data = np.repeat(points[None], 4, axis=0)
        data[:, 0, 0] += np.arange(4) * 0.1
        with tempfile.TemporaryDirectory() as directory:
            output = Path(directory) / 'synthetic.mp4'
            export_video(data, data, teams, teams, 11150, 7150, 25, output)
            cap = cv2.VideoCapture(str(output))
            try:
                self.assertTrue(cap.isOpened())
                self.assertEqual(int(cap.get(cv2.CAP_PROP_FRAME_COUNT)), 4)
                self.assertAlmostEqual(cap.get(cv2.CAP_PROP_FPS), 25)
                decoded = []
                while True:
                    ok, image = cap.read()
                    if not ok:
                        break
                    decoded.append(image)
                self.assertEqual(len(decoded), 4)
                self.assertTrue(np.any(decoded[0] != decoded[-1]))
            finally:
                cap.release()
            self.assertFalse(output.with_name('synthetic.partial.mp4').exists())
            mapped_output = Path(directory) / 'synthetic_halves.mp4'
            export_video(data, data, teams, teams, 0, 0, 25, mapped_output,
                         np.array([2, 3, 7, 8]), np.array([1, 2, 6, -1]), np.array([1, 1, 2, 2]))
            mapped = cv2.VideoCapture(str(mapped_output))
            try:
                self.assertEqual(int(mapped.get(cv2.CAP_PROP_FRAME_COUNT)), 4)
            finally:
                mapped.release()

    def test_scroll_playback_and_keyboard_keep_both_sources_aligned(self):
        module = 'starter.compare_tracking_player'
        data = np.arange(18, dtype=np.float32).reshape(3, 3, 2) / 30
        teams = ['H', 'A']
        slider = {'position': 7150}
        events = iter([(7151, -1), (None, ord(' ')), (None, -1),
                       (7150, -1), (None, ord('d')), (None, ord('q'))])
        displayed = []

        def event(delay):
            position, key = next(events)
            if position is not None:
                slider['position'] = position
            return key

        def fake_render(template, tactical, broadcast, tt, bt, tf, bf, fps, playing, labels):
            self.assertEqual(tf - bf, 4000)
            np.testing.assert_array_equal(tactical, data[bf - 7150])
            np.testing.assert_array_equal(broadcast, data[bf - 7150])
            displayed.append((bf, playing))
            return template

        with patch.multiple(
            module + '.cv2', namedWindow=Mock(), resizeWindow=Mock(), createTrackbar=Mock(),
            setTrackbarMin=Mock(), getTrackbarPos=Mock(side_effect=lambda *args: slider['position']),
            setTrackbarPos=Mock(side_effect=lambda name, window, position: slider.update(position=position)),
            waitKeyEx=Mock(side_effect=event), imshow=Mock(), getWindowProperty=Mock(return_value=1),
            destroyAllWindows=Mock(),
        ), patch(module + '.render', side_effect=fake_render), \
                patch(module + '.time.monotonic', side_effect=iter(range(100))):
            view(data, data, teams, teams, 11150, 7150, 25)
        self.assertEqual(displayed, [(7150, False), (7151, False), (7151, True),
                                     (7152, True), (7150, False), (7151, False)])

    def test_mapped_player_steps_across_halftime_and_displays_unmatched(self):
        module = 'starter.compare_tracking_player'
        tf = np.array([2, 3, 4, 7, 8, 9, 10])
        bf = np.array([1, 2, 3, 6, 7, 8, -1])
        halves = np.array([1, 1, 1, 2, 2, 2, 2])
        tactical = np.full((7, 3, 2), 0.5, dtype=np.float32)
        broadcast = tactical.copy()
        broadcast[-1] = np.nan
        teams = ['H', 'A']
        slider = {'position': 0}
        events = iter([(2, -1), (None, ord('d')), (6, -1), (None, ord('q'))])
        displayed = []

        def event(delay):
            position, key = next(events)
            if position is not None:
                slider['position'] = position
            return key

        def fake_render(template, tp, bp, tt, bt, t, b, fps, playing, labels, comparison_note=None):
            self.assertIn('HALF', comparison_note)
            if b < 0:
                self.assertTrue(np.isnan(bp).all())
            displayed.append((t, b))
            return template

        with patch.multiple(
            module + '.cv2', namedWindow=Mock(), resizeWindow=Mock(), createTrackbar=Mock(),
            setTrackbarMin=Mock(), getTrackbarPos=Mock(side_effect=lambda *args: slider['position']),
            setTrackbarPos=Mock(side_effect=lambda name, window, position: slider.update(position=position)),
            waitKeyEx=Mock(side_effect=event), imshow=Mock(), getWindowProperty=Mock(return_value=1),
            destroyAllWindows=Mock(),
        ), patch(module + '.render', side_effect=fake_render):
            view(tactical, broadcast, teams, teams, 0, 0, 25, tf, bf, halves)
        self.assertEqual(displayed, [(2, 1), (4, 3), (7, 6), (10, -1)])

    def test_initial_anchor_and_backward_extrapolation_label(self):
        module = 'starter.compare_tracking_player'
        data = np.full((3, 3, 2), 0.5, dtype=np.float32)
        teams = ['H', 'A']
        tf, bf = np.array([1102, 1103, 1104]), np.array([21, 22, 23])
        halves = np.full(3, 2)
        slider = {'position': 1}
        events = iter([ord('a'), ord('q')])
        shown = []

        def fake_render(template, tp, bp, tt, bt, t, b, fps, playing, labels, comparison_note=None):
            shown.append((t, b, comparison_note))
            return template

        with patch.multiple(
            module + '.cv2', namedWindow=Mock(), resizeWindow=Mock(), createTrackbar=Mock(),
            setTrackbarMin=Mock(), getTrackbarPos=Mock(side_effect=lambda *args: slider['position']),
            setTrackbarPos=Mock(side_effect=lambda name, window, position: slider.update(position=position)),
            waitKeyEx=Mock(side_effect=lambda delay: next(events)), imshow=Mock(),
            getWindowProperty=Mock(return_value=1), destroyAllWindows=Mock(),
        ), patch(module + '.render', side_effect=fake_render):
            view(data, data, teams, teams, 0, 0, 25, tf, bf, halves,
                 start_index=1, anchor_tactical_frame=1103)
        self.assertEqual([entry[:2] for entry in shown], [(1103, 22), (1102, 21)])
        self.assertIn('REFERENCE ANCHOR', shown[0][2])
        self.assertIn('BACKWARD EXTRAPOLATION', shown[1][2])


if __name__ == '__main__':
    unittest.main()
