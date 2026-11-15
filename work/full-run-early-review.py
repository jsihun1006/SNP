"""Risk-biased, read-only snapshot QA of completed correction chunks."""
import collections
import gzip
import hashlib
import json
import math
from pathlib import Path

import cv2
import numpy as np

WORK = Path(__file__).resolve().parent
ROOT = Path(r'H:\data\ball-tracking-results\first_half\correction_recall_v1')
rows, chunk_info = [], []
for folder in sorted((ROOT/'chunks').iterdir(), key=lambda p: int(p.name)):
    marker = folder/'manifest.json'
    payload = folder/'decisions.jsonl.gz'
    if not marker.exists() or not payload.exists():
        continue
    manifest = json.loads(marker.read_text(encoding='utf-8'))
    assert hashlib.sha256(payload.read_bytes()).hexdigest() == manifest['sha256']
    with gzip.open(payload, 'rt', encoding='utf-8') as stream:
        local = [json.loads(line) for line in stream]
    assert len(local) == manifest['frames']
    assert [r['source_frame'] for r in local] == list(range(manifest['start_frame'], manifest['end_frame_exclusive']))
    for row in local:
        row['_chunk'] = folder.name
    rows.extend(local)
    chunk_info.append(dict(start=manifest['start_frame'], end=manifest['end_frame_exclusive'], frames=len(local)))

stats = collections.Counter()
pool = []
for row in rows:
    decision = row['selection']
    stats[decision['status']] += 1
    if decision['status'] != 'selected_observation':
        continue
    candidate = row['candidates'][decision['candidate_index']]
    assert [decision['pixel_x'], decision['pixel_y']] == candidate['point']
    assert not decision['visual_reviewed'] and not decision['manually_verified']
    original = row['candidates'][:row['original_candidate_count']]
    raw = max(original, key=lambda c: c['confidence']) if original else None
    raw_distance = math.dist(candidate['point'], raw['point']) if raw else 1920.
    evidence = decision['evidence']
    score = candidate['confidence']
    recovered = candidate.get('origin') != 'original_detector'
    conflict = bool(evidence.get('atd_conflict'))
    stats['selected_needs_review'] += int(decision['needs_review'])
    stats['selected_recovered'] += int(recovered)
    risk = (3*decision['needs_review'] + 4*conflict + 2*(1-score) + recovered +
            2*evidence.get('off_field_ground_projection', False) +
            evidence.get('camera_motion_uncertain', False))
    pool.append(dict(row=row, candidate=candidate, raw=raw, raw_distance=raw_distance,
                     score=score, conflict=conflict, recovered=recovered, risk=float(risk)))

chosen, per_chunk = [], collections.Counter()
categories = [
    ('prior_conflict', lambda item: item['conflict'], lambda item: item['risk']),
    ('low_confidence', lambda item: item['score'] < .4, lambda item: -item['score']),
    ('new_recovery', lambda item: item['recovered'], lambda item: item['raw_distance']+20*item['risk']),
    ('needs_review', lambda item: item['row']['selection']['needs_review'], lambda item: item['risk']),
]
for _ in range(3):
    for label, eligible, priority in categories:
        candidates = sorted((item for item in pool if eligible(item)), key=priority, reverse=True)
        for item in candidates:
            frame = item['row']['source_frame']
            if any(abs(frame-old['row']['source_frame']) < 50 for old in chosen):
                continue
            if per_chunk[item['row']['_chunk']] >= 3:
                continue
            item['sampling_reason'] = label
            chosen.append(item)
            per_chunk[item['row']['_chunk']] += 1
            break
for item in sorted(pool, key=lambda item: item['risk'], reverse=True):
    if len(chosen) >= 12:
        break
    if all(abs(item['row']['source_frame']-old['row']['source_frame']) >= 50 for old in chosen):
        item['sampling_reason'] = 'risk_backfill'
        chosen.append(item)
chosen.sort(key=lambda item: item['row']['source_frame'])
assert len(chosen) >= 8
assert all(chosen[i+1]['row']['source_frame']-chosen[i]['row']['source_frame'] >= 50
           for i in range(len(chosen)-1))

cap = cv2.VideoCapture(r'H:\data\broadcast\BAR-ATH_broadcast.mp4')
panels, samples = [], []
for item in chosen:
    row, candidate, raw = item['row'], item['candidate'], item['raw']
    decision = row['selection'];frame = row['source_frame']
    assert cap.set(cv2.CAP_PROP_POS_FRAMES, frame)
    ok, image = cap.read();assert ok
    panel = np.zeros((470,640,3), dtype=np.uint8)
    panel[30:390] = cv2.resize(image,(640,360))
    point = tuple(np.int32(np.array(candidate['point'])/3)+[0,30])
    cv2.circle(panel,point,9,(0,255,0),2,cv2.LINE_AA)
    if raw:
        raw_point=tuple(np.int32(np.array(raw['point'])/3)+[0,30])
        cv2.drawMarker(panel,raw_point,(0,120,255),cv2.MARKER_TILTED_CROSS,15,2)
    def crop(point):
        x,y=np.int32(np.round(point))
        patch=image[max(0,y-18):min(1080,y+19),max(0,x-18):min(1920,x+19)]
        return cv2.resize(patch,(78,78),interpolation=cv2.INTER_NEAREST)
    panel[391:469,480:558]=crop(candidate['point'])
    if raw:panel[391:469,561:639]=crop(raw['point'])
    origin='RECOVERY' if item['recovered'] else 'ORIGINAL'
    title=f'f{frame} {origin} conf={candidate["confidence"]:.3f} [{item["sampling_reason"]}]'
    cv2.putText(panel,title,(7,21),cv2.FONT_HERSHEY_SIMPLEX,.43,(255,255,255),1,cv2.LINE_AA)
    distance=decision['evidence'].get('atd_distance_px')
    distance_text='missing' if distance is None else f'{distance:.1f}px'
    text=f'needs_review={decision["needs_review"]} ATDdist={distance_text}'
    cv2.putText(panel,text,(7,411),cv2.FONT_HERSHEY_SIMPLEX,.4,(230,230,230),1,cv2.LINE_AA)
    cv2.putText(panel,'Green=chosen; orange X=raw highest; crops: chosen / raw',(7,435),
                cv2.FONT_HERSHEY_SIMPLEX,.35,(200,200,200),1,cv2.LINE_AA)
    panels.append(panel)
    samples.append(dict(source_frame=frame, sampling_reason=item['sampling_reason'],
                        decision=decision, selected_candidate=candidate, raw_highest=raw,
                        selected_raw_distance_px=item['raw_distance'], visual_assessment='pending'))
cap.release()
for page in range((len(panels)+3)//4):
    canvas=np.zeros((940,1280,3),np.uint8)
    for j,panel in enumerate(panels[page*4:(page+1)*4]):
        y,x=(j//2)*470,(j%2)*640;canvas[y:y+470,x:x+640]=panel
    assert cv2.imwrite(str(WORK/f'full-run-early-review-{page+1}.jpg'),canvas,[cv2.IMWRITE_JPEG_QUALITY,95])
report=dict(chunks_snapshot=chunk_info, frames_in_snapshot=len(rows), status_counts=dict(stats),
            accepted_coordinates_match_candidate=True,
            sampling='Twelve risk-biased selections: conflict, low confidence, recovery and review flags; at least50sourceframes apart.',
            samples=samples, limitations='Risk-biased sparse visual QA is not an accuracy estimate or all-frame ground truth.')
(WORK/'full-run-early-review.json').write_text(json.dumps(report,indent=2),encoding='utf-8')
print(json.dumps(dict(snapshot_frames=len(rows), stats=dict(stats), samples=[s['source_frame'] for s in samples])))
