"""Rank suspicion, not accuracy, of continuous 180-second ball-candidate clips.

Missing homography and outside-pitch detections are never scored as errors.
The indicators are soft: genuine balls can be stationary or on a penalty mark.
"""
import csv
import json
from collections import Counter, defaultdict
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parent
SOURCE = Path(r'H:\data\ball-tracking-results\first_half\homography\ball_candidates_pitch.csv')
START, END, FPS, WINDOW = 7171, 86638, 25, 4500
N = END - START
arrays = {name: np.zeros(N, dtype=float) for name in (
    'candidates', 'score_ge_03', 'score_ge_05', 'penalty_mark_candidates',
    'penalty_mark_score_ge_03', 'good_h_candidates', 'stationary_candidates')}
frame_points = defaultdict(list)
cell_frames = defaultdict(set)
known_h = np.zeros(N, dtype=bool)
with SOURCE.open(newline='', encoding='utf-8-sig') as stream:
    for row in csv.DictReader(stream):
        frame = int(row['source_frame'])
        index = frame - START
        assert 0 <= index < N
        confidence = float(row['confidence'])
        arrays['candidates'][index] += 1
        arrays['score_ge_03'][index] += confidence >= .3
        arrays['score_ge_05'][index] += confidence >= .5
        # Spatial indicators use the usable calibration rows, with error <=2.
        # This is a ranking feature gate, not candidate filtering/correction.
        h_ok = row['h_valid'] == '1' and row['h_quality_known'] == '1' and 0 <= float(row['reprojection_error']) <= 2
        if not h_ok or not row['pitch_x_m']:
            continue
        known_h[index] = True
        x, y = float(row['pitch_x_m']), float(row['pitch_y_m'])
        if not (0 <= x <= 105.3 and 0 <= y <= 68):
            continue
        arrays['good_h_candidates'][index] += 1
        # A location repeatedly close to a painted penalty mark is suspicious,
        # but not automatically false: the actual ball can be there too.
        mark_distance = min(np.hypot(x-11, y-34), np.hypot(x-94.3, y-34))
        is_mark = mark_distance <= 1.5
        arrays['penalty_mark_candidates'][index] += is_mark
        arrays['penalty_mark_score_ge_03'][index] += is_mark and confidence >= .3
        cell = (int(x // 2), int(y // 2))
        cell_frames[cell].add(index)
        frame_points[index].append((x, y, confidence, bool(is_mark), cell))

# Repeated observations in a 2m cell lasting >=3 seconds. Permit gaps of <=2
# frames for tiny jitter/temporary model misses, while counting actual frames.
for cell, observations in cell_frames.items():
    ordered = sorted(observations)
    if not ordered:
        continue
    groups = []
    begin = 0
    for j in range(1, len(ordered)+1):
        if j == len(ordered) or ordered[j] - ordered[j-1] > 3:
            group = ordered[begin:j]
            if group[-1]-group[0]+1 >= 75 and len(group) >= 50:
                arrays['stationary_candidates'][group] += 1
            begin = j

indicators = {
    'frames_with_candidates': (arrays['candidates'] > 0).astype(float),
    'multiple_candidate_frames': (arrays['candidates'] >= 3).astype(float),
    'multiple_ge_03_frames': (arrays['score_ge_03'] >= 2).astype(float),
    'multiple_ge_05_frames': (arrays['score_ge_05'] >= 2).astype(float),
    'penalty_mark_frames': (arrays['penalty_mark_candidates'] > 0).astype(float),
    'penalty_mark_ge_03_frames': (arrays['penalty_mark_score_ge_03'] > 0).astype(float),
    'stationary_cell_frames': (arrays['stationary_candidates'] > 0).astype(float),
    'calibrated_candidate_frames': known_h.astype(float),
}
rolling = {}
for name, values in {**arrays, **indicators}.items():
    c = np.concatenate(([0.], np.cumsum(values)))
    rolling[name] = c[WINDOW:] - c[:-WINDOW]

score = (2.0*rolling['penalty_mark_frames']
         + 1.5*rolling['penalty_mark_ge_03_frames']
         + .75*rolling['stationary_cell_frames']
         + rolling['multiple_ge_03_frames']
         + .5*rolling['multiple_ge_05_frames']
         + .15*rolling['multiple_candidate_frames']) / WINDOW

def clock(frame):
    t = frame/FPS
    return f'{int(t//3600):02d}:{int(t//60)%60:02d}:{t%60:05.2f}'

order = np.argsort(score)[::-1]
chosen = []
for index in order:
    index = int(index)
    if any(abs(index-c) < WINDOW for c in chosen):
        continue
    chosen.append(index)
    if len(chosen) == 3:
        break

results=[]
for index in chosen:
    cells=Counter()
    for j in range(index,index+WINDOW):
        for *_,cell in frame_points.get(j,[]):
            cells[cell]+=1
    results.append(dict(
        start_frame=START+index,end_frame_exclusive=START+index+WINDOW,
        start=clock(START+index),end_exclusive=clock(START+index+WINDOW),
        duration_seconds=180,frames=WINDOW,suspicion_score=float(score[index]),
        metrics={name:int(values[index]) for name,values in rolling.items()},
        repeated_2m_cells=[dict(x_range_m=[x*2,x*2+2],y_range_m=[y*2,y*2+2],observations=n)
                           for (x,y),n in cells.most_common(8)],
    ))
report=dict(
    selected=results[0],top_three_nonoverlapping=results,
    method='Rolling 4500-frame soft suspicion score: 2*penalty-mark frames +1.5*penalty-mark>=.3 frames +.75*stationary2m-cell frames +multi>=.3 frames +.5*multi>=.5 frames +.15*frames>=3 candidates; divided by4500',
    limitations=['This ranks suspected false positives, not measured false-positive rate.',
                 'A real ball can stop or lie on a penalty mark; none is hard-masked.',
                 'Missing H and outside-pitch points do not incur a false-positive penalty.',
                 'H-error<=2 is a spatial feature gate only, not a correction filter.',
                 'Input CSV has candidate rows only; calibrated_candidate_frames counts frames with candidates and usable H, not all usable-H frames.'],
    source=str(SOURCE),source_range=[START,END],window_frames=WINDOW,fps=FPS)
(ROOT/'ranked_windows.json').write_text(json.dumps(report,indent=2),encoding='utf-8')
print(json.dumps(report))
