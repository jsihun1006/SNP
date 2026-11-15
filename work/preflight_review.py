"""Read-only preflight audit; writes review artifacts only under work/."""
import collections
import gzip
import json
import math
from pathlib import Path

import cv2
import numpy as np

BASE = Path(__file__).resolve().parent
SOURCE = Path(r'H:\data\ball-tracking-results\first_half\correction_validation_1000\chunks\18919\decisions.jsonl.gz')
VIDEO = Path(r'H:\data\broadcast\BAR-ATH_broadcast.mp4')
with gzip.open(SOURCE, 'rt', encoding='utf-8') as stream:
    rows = [json.loads(line) for line in stream]
stats = collections.Counter()
for i, row in enumerate(rows):
    assert row['source_frame'] == 18919 + i
    decision = row['selection']
    stats[decision['status']] += 1
    assert decision['source_frame'] == row['source_frame']
    assert not decision['visual_reviewed'] and not decision['manually_verified']
    if decision['status'] == 'selected_observation':
        candidate = row['candidates'][decision['candidate_index']]
        assert [decision['pixel_x'], decision['pixel_y']] == candidate['point']
        assert decision['candidate_confidence'] == candidate['confidence']
        stats['selected_' + candidate['origin']] += 1
        stats['selected_needs_review'] += int(decision['needs_review'])
    else:
        assert decision['pixel_x'] is None and decision['pixel_y'] is None
for frame in (19294, 19419, 19544):
    assert rows[frame - 18919]['selection']['status'] == 'review_candidate'

# Risk-biased checks: newly recovered or original alternatives to raw highest
# confidence, including two automatic selections already flagged for ATD conflict.
frames = [18972, 19222, 19233, 19574, 19064, 19353, 19488, 19383]
cap = cv2.VideoCapture(str(VIDEO))
panels, examples = [], []
for frame in frames:
    row = rows[frame - 18919]
    decision = row['selection']
    selected = row['candidates'][decision['candidate_index']]
    original = row['candidates'][:row['original_candidate_count']]
    highest = max(original, key=lambda value: value['confidence']) if original else None
    assert cap.set(cv2.CAP_PROP_POS_FRAMES, frame)
    ok, image = cap.read()
    assert ok
    panel = np.zeros((470, 640, 3), dtype=np.uint8)
    panel[30:390] = cv2.resize(image, (640, 360))
    marker = tuple(np.int32(np.asarray(selected['point']) / 3) + [0, 30])
    cv2.circle(panel, marker, 9, (0, 255, 0), 2, cv2.LINE_AA)
    if highest is not None:
        raw_marker = tuple(np.int32(np.asarray(highest['point']) / 3) + [0, 30])
        cv2.drawMarker(panel, raw_marker, (0, 120, 255), cv2.MARKER_TILTED_CROSS, 15, 2)
    def crop(point):
        x, y = np.int32(np.round(point))
        patch = image[max(0, y-18):min(1080, y+19), max(0, x-18):min(1920, x+19)]
        return cv2.resize(patch, (78, 78), interpolation=cv2.INTER_NEAREST)
    panel[391:469, 480:558] = crop(selected['point'])
    if highest is not None:
        panel[391:469, 561:639] = crop(highest['point'])
    title = f"f{frame} {selected['origin']} conf={selected['confidence']:.3f}"
    cv2.putText(panel, title, (7, 21), cv2.FONT_HERSHEY_SIMPLEX, .47, (255,255,255), 1, cv2.LINE_AA)
    text = f"selected; needs_review={decision['needs_review']}  ATDdist={decision['evidence']['atd_distance_px']:.1f}px"
    cv2.putText(panel, text, (7, 411), cv2.FONT_HERSHEY_SIMPLEX, .42, (230,230,230), 1, cv2.LINE_AA)
    cv2.putText(panel, "Green=chosen; orange X=raw highest; crops: chosen / raw", (7, 435),
                cv2.FONT_HERSHEY_SIMPLEX, .36, (200,200,200), 1, cv2.LINE_AA)
    panels.append(panel)
    examples.append(dict(frame=frame, selection=decision, candidate=selected,
                         raw_highest=highest, selected_raw_distance=math.dist(selected['point'], highest['point']) if highest else None))
cap.release()
for page in range(2):
    canvas = np.zeros((940, 1280, 3), dtype=np.uint8)
    for j, panel in enumerate(panels[page*4:(page+1)*4]):
        y, x = (j//2)*470, (j%2)*640
        canvas[y:y+470,x:x+640] = panel
    assert cv2.imwrite(str(BASE/f'preflight-review-{page+1}.jpg'), canvas, [cv2.IMWRITE_JPEG_QUALITY, 95])
report = dict(stats=dict(stats), checked_all_rows=len(rows), no_manual_overrides=True,
              accepted_points_exactly_match_candidates=True, known_ad_examples_review_only=True,
              sample_method='Eight risk-biased recovered or changed-highest selections; not an accuracy sample.',
              samples=examples)
(BASE/'preflight-review.json').write_text(json.dumps(report,indent=2),encoding='utf-8')
print(json.dumps(dict(stats=dict(stats), frames=frames, audit_checks_passed=True)))
