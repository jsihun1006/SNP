"""Complete non-X timeline selection, boundary derivatives and original frame seeks."""

import json
from pathlib import Path
import tempfile
import unittest

import numpy as np

from starter.view_all_movement_vectors import select_timeline, timeline_segments, pack_non_x, write_player


def state(active, halves):
    active, halves = np.array(active, dtype=bool), np.array(halves, dtype=np.int8)
    eligible = np.zeros(len(active), dtype=bool)
    eligible[1:] = active[1:] & active[:-1] & (halves[1:] == halves[:-1]) & (halves[1:] > 0)
    return {'active': active, 'halves': halves, 'eligible': eligible}


class AllMovementTests(unittest.TestCase):
    def states(self):
        return {'tactical': state([False, True, True, False, True, True], [0, 1, 1, 0, 2, 2]),
                'broadcast': state([True, True, False, True, True], [1, 1, 0, 2, 2])}

    def test_every_non_x_source_frame_is_selected_once_in_original_order(self):
        states = self.states()
        tf, bf, halves = select_timeline(np.array([-1, 0, 1, -1, 3, 4]), states)
        self.assertEqual(tf.tolist(), [1, 2, 4, 5])
        self.assertEqual(bf.tolist(), [0, 1, 3, 4])
        self.assertEqual(halves.tolist(), [1, 1, 2, 2])
        states['broadcast']['active'][2] = True
        with self.assertRaises(ValueError):
            select_timeline(np.array([-1, 0, 1, -1, 3, 4]), states)

    def test_compact_segments_keep_original_indices_across_excluded_gap(self):
        tf, bf, halves = select_timeline(np.array([-1, 0, 1, -1, 3, 4]), self.states())
        segments = timeline_segments(tf, bf, halves)
        self.assertEqual(segments, [
            {'video_first': 0, 'count': 2, 'tactical_first': 1, 'broadcast_first': 0, 'half': '1st Half'},
            {'video_first': 2, 'count': 2, 'tactical_first': 4, 'broadcast_first': 3, 'half': '2nd Half'}])
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory)/'player.html'
            meta = write_player(path, tf, bf, halves, 25)
            html = path.read_text()
        self.assertEqual(meta['count'], 4)
        self.assertNotIn('__META__', html)
        self.assertIn('full_movement_vectors.mp4', html)
        self.assertNotIn('https://', html)

    def test_coordinate_derivative_is_removed_after_x_or_half_boundary(self):
        positions = np.full((6, 3, 2), .5, dtype=np.float32)
        positions[:, 0, 0] += np.arange(6)*.01
        tf = np.array([1, 2, 4, 5])
        packed = pack_non_x(positions, tf, self.states()['tactical'], 25, (105.3, 68))
        self.assertTrue(np.isnan(packed[[0, 2], :, 2:]).all())
        np.testing.assert_allclose(packed[[1, 3], 0, 2], [26.325, 26.325], atol=2e-4)
        np.testing.assert_equal(packed[[1, 3], 1, 2:], [[0, 0], [0, 0]])
        with self.assertRaises(ValueError):
            pack_non_x(positions, np.array([3]), self.states()['tactical'], 25, (105.3, 68))

    def test_unmapped_broadcast_does_not_drop_tactical_frames_or_repeat_frames(self):
        states = self.states()
        states['broadcast'] = state([False]*5, [0]*5)
        tf, bf, halves = select_timeline(np.full(6, -1), states)
        self.assertEqual(tf.tolist(), [1, 2, 4, 5])
        self.assertTrue((bf == -1).all())
        segments = timeline_segments(tf, bf, halves)
        self.assertEqual(len(segments), 2)
        self.assertTrue(all(s['broadcast_first'] is None for s in segments))


if __name__ == '__main__':
    unittest.main()
