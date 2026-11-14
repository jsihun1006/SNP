import contextlib
import hashlib
import io
import json
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

import cv2
import numpy as np

from ball_only import run as runner


class _Controller:
    def __init__(self, **kwargs):
        self.parameters = kwargs

    def snapshot(self, refresh=False):
        return {}


class _Detector:
    created = 0
    calls = []
    stop_after_first = None

    def __init__(self, weights, controller, device=0):
        type(self).created += 1

    def settings(self):
        return dict(test_double=True, precision='FP32')

    def run_chunk(self, frames, **kwargs):
        indices = [int(round(float(frame.mean()) / 12)) for frame in frames]
        type(self).calls.append(indices)
        if type(self).stop_after_first is not None:
            type(self).stop_after_first.write_text('test stop', encoding='utf-8')
            type(self).stop_after_first = None
        return [dict(candidates=[dict(point=[10., 20.], confidence=.7)],
                     decoded_test_frame=value)
                for value in indices]


class RunTests(unittest.TestCase):
    def setUp(self):
        temp_root = Path(tempfile.gettempdir()).resolve()
        self.temporary = tempfile.TemporaryDirectory(prefix='ball-only-run-test-', dir=temp_root)
        self.folder = Path(self.temporary.name).resolve()
        # Verify the scope before TemporaryDirectory performs recursive cleanup.
        self.assertEqual(self.folder.parent, temp_root)
        self.addCleanup(self.temporary.cleanup)
        self.base = self.folder / 'package'
        self.base.mkdir()
        package = Path(runner.__file__).parent
        for name in ('run.py', 'detector.py', 'runtime_control.py'):
            (self.base / name).write_bytes((package / name).read_bytes())
        self.env = patch.object(runner, 'environment', return_value=self.base)
        self.env.start()
        self.addCleanup(self.env.stop)
        self.video = self.folder / 'source.mp4'
        writer = cv2.VideoWriter(str(self.video), cv2.VideoWriter_fourcc(*'mp4v'), 25., (64, 48))
        self.assertTrue(writer.isOpened())
        for index in range(12):
            writer.write(np.full((48, 64, 3), index * 12, dtype=np.uint8))
        writer.release()
        self.weights = self.folder / 'fake-weights.pt'
        self.weights.write_bytes(b'test-only weights; no neural model is loaded')
        self.out = self.folder / 'result'
        _Detector.created = 0
        _Detector.calls = []
        _Detector.stop_after_first = None

    def execute(self, **changes):
        args = dict(video=self.video, weights=self.weights, out=self.out,
                    start='.16', end='.40', part_id='first-half', chunk_frames=4,
                    detector_factory=_Detector, controller_factory=_Controller)
        args.update(changes)
        with contextlib.redirect_stdout(io.StringIO()):
            return runner.run(**args)

    def output_rows(self):
        return [json.loads(line) for line in (self.out / 'candidates.jsonl').read_text(encoding='utf-8').splitlines()]

    def test_exact_source_range_checkpoint_boundaries_and_complete_resume(self):
        result = self.execute()
        self.assertEqual(result['frames'], 6)
        rows = self.output_rows()
        self.assertEqual([r['source_frame'] for r in rows], list(range(4, 10)))
        self.assertEqual([r['frame_idx'] for r in rows], list(range(4, 10)))
        self.assertEqual([r['part_frame'] for r in rows], list(range(6)))
        self.assertEqual([r['decoded_test_frame'] for r in rows], list(range(4, 10)))
        self.assertEqual({r['part_id'] for r in rows}, {'first-half'})
        self.assertTrue(all(not r['candidates_are_ground_truth'] for r in rows))
        self.assertEqual(_Detector.calls, [[4, 5, 6, 7], [8, 9]])
        manifests = [runner.read(p) for p in sorted((self.out / 'chunks').glob('*/manifest.json'))]
        self.assertEqual([(m['start_frame'], m['end_frame'], m['frames']) for m in manifests], [(4, 8, 4), (8, 10, 2)])
        digest = hashlib.sha256((self.out / 'candidates.jsonl').read_bytes()).hexdigest()
        self.assertEqual(result['file_sha256'], digest)

        def forbidden_factory(*args, **kwargs):
            raise AssertionError('A complete resume must not construct a detector')

        again = self.execute(detector_factory=forbidden_factory)
        self.assertEqual(again['file_sha256'], digest)
        self.assertEqual(_Detector.created, 1)
        self.assertEqual(len(self.output_rows()), 6)

    def test_config_change_rejected_before_new_inference(self):
        self.execute()
        with self.assertRaisesRegex(ValueError, 'different settings'):
            self.execute(part_id='second-half')
        self.assertEqual(_Detector.created, 1)

    def test_precomputed_ranges_reject_another_video_before_inference(self):
        with self.assertRaisesRegex(ValueError, 'does not match the source video'):
            self.execute(expected_video_sha256='0' * 64)
        self.assertEqual(_Detector.created, 0)
        self.assertFalse((self.out / 'job_complete.json').exists())
        result = self.execute(expected_video_sha256=runner.sha(self.video))
        self.assertEqual(result['frames'], 6)

    def test_corrupted_checkpoint_rejected_before_new_inference(self):
        self.execute()
        payload = self.out / 'chunks' / '000000004' / 'detections.jsonl'
        rows = [json.loads(line) for line in payload.read_text(encoding='utf-8').splitlines()]
        rows[0]['decoded_test_frame'] = 999
        payload.write_text('\n'.join(json.dumps(row) for row in rows) + '\n', encoding='utf-8')
        with self.assertRaisesRegex(ValueError, 'Checkpoint content changed'):
            self.execute()
        self.assertEqual(_Detector.created, 1)

    def test_stop_at_committed_checkpoint_then_resume_without_duplicates(self):
        self.out.mkdir()
        _Detector.stop_after_first = self.out / 'STOP'
        self.assertIs(self.execute(), False)
        self.assertTrue((self.out / 'chunks' / '000000004' / 'manifest.json').exists())
        self.assertFalse((self.out / 'chunks' / '000000008' / 'manifest.json').exists())
        self.assertFalse((self.out / 'job_complete.json').exists())
        self.assertEqual(runner.read(self.out / 'job_stopped.json')['next_source_frame'], 8)
        (self.out / 'STOP').unlink()
        result = self.execute()
        self.assertEqual(result['frames'], 6)
        self.assertEqual([r['source_frame'] for r in self.output_rows()], list(range(4, 10)))
        self.assertEqual(_Detector.calls, [[4, 5, 6, 7], [8, 9]])
        self.assertFalse((self.out / 'job_stopped.json').exists())

    def test_preexisting_stop_does_not_construct_detector(self):
        self.out.mkdir()
        (self.out / 'STOP').write_text('test stop', encoding='utf-8')
        self.assertIs(self.execute(), False)
        self.assertEqual(_Detector.created, 0)
        self.assertFalse((self.out / 'job_complete.json').exists())

    def test_fractional_frame_timestamp_rejected_and_eof_exact(self):
        with self.assertRaisesRegex(ValueError, 'frame boundary'):
            self.execute(start='.161')
        self.assertEqual(_Detector.created, 0)
        result = self.execute(start='.40', end='EOF')
        self.assertEqual(result['frames'], 2)
        self.assertEqual([r['source_frame'] for r in self.output_rows()], [10, 11])
        self.assertEqual([r['part_frame'] for r in self.output_rows()], [0, 1])
        with self.assertRaisesRegex(ValueError, 'EOF is allowed only'):
            runner.time_frame('EOF', 25., 12)


if __name__ == '__main__':
    unittest.main()
