"""Measure what existing review artifacts support, without inventing ground truth."""
import json
from pathlib import Path
import numpy as np

BASE=Path(__file__).resolve().parent.parent
pilot=BASE/'work/ball_pilot'
original=json.loads((pilot/'candidates.json').read_text(encoding='utf-8'))
final=json.loads((pilot/'final_track.json').read_text(encoding='utf-8'))
audit=json.loads((BASE/'outputs/ball-correction-3min/visual_review.json').read_text(encoding='utf-8'))
review_distances=[]
for item in audit:
    i=item['source_frame']-original['start']
    before=max(original['records'][i]['candidates'],key=lambda c:c['confidence'])
    after=final['records'][i]
    assert after['pixel_x'] is not None
    review_distances.append(float(np.linalg.norm(np.array(before['point'])-[after['pixel_x'],after['pixel_y']])))
accepted={'selected_observation','visually_reviewed_observation'}
counts=dict(accepted=0,original_max_exactly_same=0,original_max_different=0,no_original_candidate=0,
            targeted_selected=0,targeted_with_no_original_candidate=0,targeted_nearest_original_over_10px=0)
for before,after in zip(original['records'],final['records']):
    assert before['frame']==after['source_frame']
    if after['status'] not in accepted:continue
    counts['accepted']+=1
    point=np.array([after['pixel_x'],after['pixel_y']])
    if before['candidates']:
        best=max(before['candidates'],key=lambda c:c['confidence'])
        distance=float(np.linalg.norm(np.array(best['point'])-point))
        counts['original_max_exactly_same' if distance<1e-9 else 'original_max_different']+=1
    else:counts['no_original_candidate']+=1
    if after.get('origin')=='targeted_redetection':
        counts['targeted_selected']+=1
        if not before['candidates']:counts['targeted_with_no_original_candidate']+=1
        elif min(np.linalg.norm(np.asarray(c['point'])-point) for c in before['candidates'])>10:counts['targeted_nearest_original_over_10px']+=1
result=dict(reviewed_exact_frames=len(audit),reviewed_same_as_original_max=sum(d<1e-9 for d in review_distances),
            reviewed_max_coordinate_difference_px=max(review_distances),descriptive_counts=counts,
            precision_before=None,precision_after=None,recall_before=None,recall_after=None,accuracy_gain=None,
            reason='No independent localization/visibility ground truth. Reviewed samples do not cover changed/recovered choices.',
            warning='Coordinate changes, acceptance coverage, and use of recovered candidates are not accuracy measurements.')
(BASE/'outputs/ball-correction-3min/accuracy-evidence.json').write_text(json.dumps(result,indent=2),encoding='utf-8')
print(json.dumps(result))
