import gc
import json
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parent
PROJECT = ROOT / 'source/SNP-ball-tracking'
sys.path.insert(0, str(PROJECT))
from ball_only.run import environment
environment()
import cv2
import numpy as np
import torch
from ball_only.detector import BallDetector
from ball_only.runtime_control import RuntimeController

REPORT = ROOT / 'batch-benchmark.json'
VIDEO = Path(r'H:\data\broadcast\BAR-ATH_broadcast.mp4')
frame_ids = [i + j for i in (7171, 30000, 50000, 80000) for j in range(8)]
frames = []
cap = cv2.VideoCapture(str(VIDEO))
for start in (7171, 30000, 50000, 80000):
    assert cap.set(cv2.CAP_PROP_POS_FRAMES, start)
    for _ in range(8):
        ok, frame = cap.read()
        assert ok
        frames.append(frame)
cap.release()

controller = RuntimeController(target_low=98., target_high=99.5)
detector = BallDetector(PROJECT / 'ball_only/weights/yolo26x.pt', controller)
report = dict(frame_ids=frame_ids, frames=len(frames), configurations=[], settings=detector.settings())

def compare(left, right):
    counts_equal = windows_equal = True
    max_box_delta = max_conf_delta = 0.
    for a, b in zip(left, right):
        windows_equal &= np.allclose(a['roi_centers'], b['roi_centers'], atol=.01, rtol=0) if np.asarray(a['roi_centers']).shape == np.asarray(b['roi_centers']).shape else False
        for mode in ('full', 'roi', 'tile', 'candidates'):
            aa, bb = a[mode], b[mode]
            if len(aa) != len(bb):
                counts_equal = False
                continue
            key = lambda x: (x.get('source', ''), x.get('window_index', 0), x['point'][0], x['point'][1])
            for x, y in zip(sorted(aa, key=key), sorted(bb, key=key)):
                windows_equal &= x.get('window_xyxy') == y.get('window_xyxy')
                max_box_delta = max(max_box_delta, float(np.max(np.abs(np.asarray(x['bbox_xyxy'])-np.asarray(y['bbox_xyxy'])))))
                max_conf_delta = max(max_conf_delta, abs(x['confidence'] - y['confidence']))
    return dict(counts_equal=bool(counts_equal), roi_windows_equal=bool(windows_equal),
                max_box_delta_pixels=max_box_delta, max_confidence_delta=max_conf_delta,
                passed=bool(counts_equal and windows_equal and max_box_delta <= .01 and max_conf_delta <= .0001))

baseline = None
for config in ((1,4,8), (4,8,32), (8,16,32), (8,16,64), (16,32,64), (4,8,16)):
    full, roi, tile = config
    print(json.dumps(dict(event='starting', batches=config)), flush=True)
    row = dict(batches=dict(full=full,roi=roi,tile=tile), status='running')
    report['configurations'].append(row)
    try:
        kwargs = dict(full_batch=full, roi_batch=roi, tile_batch=tile, max_rois=2)
        # Warm all three input shapes and batch sizes before measuring.
        detector.run_chunk(frames, **kwargs)
        seconds = []
        results = None
        torch.cuda.reset_peak_memory_stats()
        for repeat in range(2):
            torch.cuda.synchronize()
            started = time.perf_counter()
            results = detector.run_chunk(frames, **kwargs)
            torch.cuda.synchronize()
            seconds.append(time.perf_counter() - started)
        if baseline is None:
            baseline = results
        row.update(status='complete', seconds=seconds, fps=len(frames)/(sum(seconds)/len(seconds)),
                   peak_allocated_mib=torch.cuda.max_memory_allocated()/1048576,
                   peak_reserved_mib=torch.cuda.max_memory_reserved()/1048576,
                   comparison=compare(baseline,results), gpu=controller.snapshot(refresh=True),
                   phases=results[0]['timing']['chunk_phase_totals'])
    except RuntimeError as exc:
        row.update(status='failed', error=repr(exc))
        gc.collect()
        torch.cuda.empty_cache()
    REPORT.write_text(json.dumps(report, indent=2), encoding='utf-8')
    print(json.dumps(row), flush=True)

valid = [r for r in report['configurations'] if r.get('status') == 'complete' and r['comparison']['passed'] and r['peak_allocated_mib'] < 14000]
best = max(valid, key=lambda r: r['fps'])
report['selected'] = best['batches']
report['speedup'] = best['fps']/report['configurations'][0]['fps']
REPORT.write_text(json.dumps(report, indent=2), encoding='utf-8')
print(json.dumps(dict(event='selected', batches=report['selected'], speedup=report['speedup'])), flush=True)
