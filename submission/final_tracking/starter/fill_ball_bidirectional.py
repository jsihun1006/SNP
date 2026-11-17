"""Fill bounded ball gaps using two original observed endpoints, never extrapolation.

Only the previously accepted green detections are observations. Predictions never
become anchors. Camera/shot boundaries and ambiguous observations block a bridge.
"""
from pathlib import Path
import argparse
import hashlib
import json
import time

import cv2
import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
BASE = ROOT / 'data/derived/ball_yolo26x_pitch_20261116'
OUT = ROOT / 'data/derived/ball_bidirectional_fill_20261116'
HFILE = ROOT / 'data/broadcast/BAR-ATH_broadcast_homography.csv'
VIDEO = ROOT / 'data/broadcast/BAR-ATH_broadcast-001.mp4'
N = 164487
FPS = 25
PITCH = np.array([105.3, 68.])
MAX_GAP = 50
MAX_SPEED = 40.
MAX_DISAGREEMENT = 2.
SUPPORT_DISTANCE = 10
KEY = ['source_frame', 'candidate_index']


def read(path):
    return pd.read_csv(path, float_precision='round_trip', low_memory=False)


def sha(path):
    digest = hashlib.sha256()
    with Path(path).open('rb') as stream:
        for chunk in iter(lambda: stream.read(4 * 1024 * 1024), b''):
            digest.update(chunk)
    return digest.hexdigest()


def write(name, data):
    (OUT / name).write_text(json.dumps(data, ensure_ascii=False, indent=2) + '\n')


def point(row):
    return np.array([row['pitch_x_m'], row['pitch_y_m']], float)


def predict_gap(left, right, previous=None, following=None):
    """Require BOTH real endpoints. Return only their strict interior frames."""
    if left is None or right is None:
        return None, 'missing_observed_endpoint'
    a, b = int(left['source_frame']), int(right['source_frame'])
    if not (left.get('is_observation', True) and right.get('is_observation', True)):
        return None, 'predictions_cannot_be_anchors'
    gap = b - a - 1
    if gap < 1:
        return None, 'no_missing_interior'
    if gap > MAX_GAP:
        return None, 'gap_longer_than_50_frames'
    if left['half'] not in (1, 2) or left['half'] != right['half']:
        return None, 'half_boundary'
    p0, p1 = point(left), point(right)
    duration = (b - a) / FPS
    average = (p1 - p0) / duration
    if not np.isfinite([p0, p1]).all() or np.linalg.norm(average) > MAX_SPEED:
        return None, 'endpoint_displacement_inconsistent'
    offsets = np.arange(1, b - a)
    t = offsets / (b - a)
    have_previous = (previous is not None and previous.get('is_observation', True)
                     and previous['half'] == left['half']
                     and 0 < a - previous['source_frame'] <= SUPPORT_DISTANCE)
    have_following = (following is not None and following.get('is_observation', True)
                      and following['half'] == right['half']
                      and 0 < following['source_frame'] - b <= SUPPORT_DISTANCE)
    disagreement = np.nan
    if have_previous and have_following:
        v0 = (p0 - point(previous)) * FPS / (a - previous['source_frame'])
        v1 = (point(following) - p1) * FPS / (following['source_frame'] - b)
        if not np.isfinite([v0, v1]).all() or max(np.linalg.norm(v0), np.linalg.norm(v1)) > MAX_SPEED:
            return None, 'surrounding_observed_velocity_inconsistent'
        forward = p0 + t[:, None] * duration * v0
        backward = p1 - (1 - t[:, None]) * duration * v1
        disagreement = float(np.linalg.norm(forward - backward, axis=1).max())
        if disagreement > MAX_DISAGREEMENT:
            return None, 'forward_backward_predictions_disagree'
        # Cubic Hermite uses real velocity evidence at both ends and preserves
        # their exact positions. It is evaluated only inside the observed span.
        h00 = 2*t**3 - 3*t**2 + 1
        h10 = t**3 - 2*t**2 + t
        h01 = -2*t**3 + 3*t**2
        h11 = t**3 - t**2
        xy = h00[:, None]*p0 + h10[:, None]*duration*v0 + h01[:, None]*p1 + h11[:, None]*duration*v1
        method = 'two_sided_observed_velocity_hermite'
    else:
        # Two actual endpoint observations suffice only for very short holes.
        # Never replace a failed two-sided velocity test with this fallback.
        if gap > 5:
            return None, 'long_gap_without_two_sided_velocity_support'
        xy = (1-t[:, None])*p0 + t[:, None]*p1
        method = 'two_observed_endpoint_linear_max_5_frames'
    entire = np.vstack([p0, xy, p1])
    if np.linalg.norm(np.diff(entire, axis=0), axis=1).max()*FPS > MAX_SPEED:
        return None, 'predicted_speed_inconsistent'
    if not ((xy >= [-20, -20]) & (xy <= [125.3, 88])).all():
        return None, 'predicted_projection_outlier'
    return dict(source_frames=a+offsets, pitch_xy=xy, method=method,
                forward_backward_max_difference_m=disagreement,
                previous_support_frame=int(previous['source_frame']) if have_previous else None,
                following_support_frame=int(following['source_frame']) if have_following else None), 'candidate'


def homography_project(matrices, xy):
    normal = np.asarray(xy) / PITCH
    rays = np.einsum('nij,nj->ni', matrices, np.c_[normal, np.ones(len(normal))])
    if not np.isfinite(rays).all() or np.any(np.abs(rays[:, 2]) < 1e-10):
        return None
    return rays[:, :2] / rays[:, 2:]


def local_histogram(image):
    small = cv2.resize(image, (256, 144), interpolation=cv2.INTER_AREA)
    hsv = cv2.cvtColor(small, cv2.COLOR_BGR2HSV)
    hist = cv2.calcHist([hsv], [0, 1], None, [24, 16], [0, 180, 0, 256])
    cv2.normalize(hist, hist, norm_type=cv2.NORM_L1)
    return small, hist


def prepare():
    OUT.mkdir(exist_ok=True)
    source = read(BASE / 'reviewed_ball_pitch.csv')
    timeline = read(BASE / 'reviewed_all_frames.csv')
    h = read(HFILE)
    matrices = h[[f'h{i}{j}' for i in range(3) for j in range(3)]].to_numpy().reshape(-1, 3, 3)
    assert len(timeline) == N and source.verdict.eq('model_ball').all()
    counts = source.groupby('source_frame').size()
    usable = source.pitch_usable & ~source.visual_audit_label.isin(['uncertain', 'conflicting_review'])
    usable &= source.source_frame.map(counts).eq(1) & source.half.isin([1, 2])
    anchors = source[usable].sort_values('source_frame').copy()
    anchors['is_observation'] = True
    rows = anchors.to_dict('records')
    present = set(source.source_frame)
    with np.load(ROOT / 'data/derived/ball_fp_reduction_20261116/evidence.npz') as evidence:
        motion_frames = evidence['motion_frames']
        cut_distances = evidence['cut_color_distance']
    records, proposed = [], []
    for i, (left, right) in enumerate(zip(rows[:-1], rows[1:])):
        a, b = int(left['source_frame']), int(right['source_frame'])
        if b == a+1:
            continue
        rec = dict(gap_id=len(records), left_anchor_frame=a, right_anchor_frame=b,
                   left_anchor_candidate_index=int(left['candidate_index']),
                   right_anchor_candidate_index=int(right['candidate_index']),
                   missing_frames=b-a-1, half=int(left['half']))
        result, reason = predict_gap(left, right, rows[i-1] if i else None, rows[i+2] if i+2 < len(rows) else None)
        if result is not None:
            fs = np.arange(a, b+1)
            if any(f in present for f in fs[1:-1]):
                reason = 'intervening_green_observation_or_ambiguity'
            elif not timeline.half.iloc[fs].eq(left['half']).all():
                reason = 'half_boundary'
            elif not h.valid.iloc[fs].eq(1).all() or not np.isfinite(matrices[fs]).all():
                reason = 'homography_unavailable_inside_gap'
            else:
                sampled = (motion_frames > a) & (motion_frames <= b)
                max_cut = float(np.nanmax(cut_distances[sampled])) if sampled.any() else 0.
                rec['sampled_max_color_distance'] = max_cut
                if max_cut > .30:
                    reason = 'sampled_scene_change_cue'
                else:
                    # Static field probes must move smoothly under the existing H.
                    probes = np.array([[.25, .25], [.5, .25], [.75, .25], [.25, .5], [.5, .5], [.75, .5], [.25, .75], [.5, .75], [.75, .75]])
                    rays = np.einsum('nij,kj->nki', matrices[fs], np.c_[probes, np.ones(len(probes))])
                    if np.any(np.abs(rays[:, :, 2]) < 1e-10):
                        reason = 'homography_probe_at_infinity'
                    else:
                        projected = rays[:, :, :2] / rays[:, :, 2:]
                        movement = np.median(np.linalg.norm(np.diff(projected, axis=0), axis=2), axis=1)
                        rec['maximum_camera_probe_step_px'] = float(movement.max())
                        image_xy = homography_project(matrices[result['source_frames']], result['pitch_xy'])
                        if movement.max() > 160:
                            reason = 'homography_camera_discontinuity'
                        elif image_xy is None or not ((image_xy >= [4, 4]) & (image_xy < [1916, 1076])).all():
                            reason = 'prediction_outside_original_image'
                        else:
                            rec.update(method=result['method'], forward_backward_max_difference_m=result['forward_backward_max_difference_m'],
                                       previous_support_frame=result['previous_support_frame'], following_support_frame=result['following_support_frame'])
                            result['image_xy'] = image_xy
                            proposed.append((len(records), result))
        rec['decision'] = reason
        records.append(rec)
    # Read the actual images throughout each proposed short gap before accepting
    # it. Existing five-frame cut evidence alone could miss a brief camera cut.
    cap = cv2.VideoCapture(str(VIDEO))
    assert cap.isOpened()
    current = -1
    predictions = []
    for index, result in proposed:
        rec = records[index]
        a, b = rec['left_anchor_frame'], rec['right_anchor_frame']
        if current < 0 or a <= current or a-current > 75:
            assert cap.set(cv2.CAP_PROP_POS_FRAMES, a)
            current = a-1
        old_small = old_hist = None
        max_color = max_mad = 0.
        for f in range(a, b+1):
            while current < f:
                assert cap.grab(), f
                current += 1
            ok, image = cap.retrieve()
            assert ok
            small, hist = local_histogram(image)
            if old_hist is not None:
                max_color = max(max_color, float(cv2.compareHist(old_hist, hist, cv2.HISTCMP_BHATTACHARYYA)))
                max_mad = max(max_mad, float(np.abs(small.astype(float)-old_small).mean()/255))
            old_small, old_hist = small, hist
        rec.update(per_frame_max_color_distance=max_color, per_frame_max_image_difference=max_mad)
        if max_color > .35 or max_mad > .28:
            rec['decision'] = 'per_frame_scene_change_cue'
            continue
        rec['decision'] = 'filled_two_sided'
        for j, f in enumerate(result['source_frames']):
            x, y = result['pitch_xy'][j]
            ix, iy = result['image_xy'][j]
            predictions.append(dict(source_frame=int(f), half=rec['half'], video_seconds=float(f/FPS),
                candidate_index=np.nan, image_x=ix, image_y=iy, pitch_x_m=x, pitch_y_m=y,
                pitch_x_normalized=x/PITCH[0], pitch_y_normalized=y/PITCH[1],
                inside_pitch=bool(0<=x<=105.3 and 0<=y<=68), pitch_usable=True,
                coordinate_kind='predicted_two_sided', is_observation=False, is_prediction=True,
                gap_id=rec['gap_id'], left_anchor_frame=a, right_anchor_frame=b,
                left_anchor_candidate_index=rec['left_anchor_candidate_index'], right_anchor_candidate_index=rec['right_anchor_candidate_index'],
                previous_support_frame=result['previous_support_frame'], following_support_frame=result['following_support_frame'],
                prediction_method=result['method'], forward_backward_max_difference_m=result['forward_backward_max_difference_m']))
    cap.release()
    assert predictions
    predicted = pd.DataFrame(predictions).sort_values('source_frame').reset_index(drop=True)
    assert predicted.source_frame.is_unique and not set(predicted.source_frame) & present
    source['coordinate_kind'] = 'observed_green'
    source['is_observation'] = True
    source['is_prediction'] = False
    combined = pd.concat([source, predicted], ignore_index=True).sort_values(KEY, na_position='last')
    combined.to_csv(OUT/'ball_candidates_with_predictions.csv', index=False)
    combined[combined.pitch_usable].to_csv(OUT/'ball_pitch_coordinates.csv', index=False)
    predicted.to_csv(OUT/'predicted_frames.csv', index=False)
    anchors.to_csv(OUT/'observed_anchors.csv', index=False)
    pd.DataFrame(records).to_csv(OUT/'gap_decisions.csv', index=False)
    result_timeline = timeline.copy()
    result_timeline['original_frame_status'] = result_timeline.frame_status
    result_timeline['is_prediction'] = False
    result_timeline['observed_usable_pitch_count'] = result_timeline.usable_pitch_count
    result_timeline['image_candidate_count'] = result_timeline.reviewed_green_count
    result_timeline['coordinate_kind'] = np.where(result_timeline.reviewed_green_count.gt(0), 'observed_green', 'missing')
    for name in ['left_anchor_frame','right_anchor_frame','left_anchor_candidate_index','right_anchor_candidate_index','gap_id']:
        result_timeline[name] = np.nan
    for r in predicted.itertuples():
        f = int(r.source_frame)
        result_timeline.loc[f, 'frame_status'] = 'predicted_between_two_observed_anchors'
        result_timeline.loc[f, 'coordinate_kind'] = 'predicted_two_sided'
        result_timeline.loc[f, 'is_prediction'] = True
        result_timeline.loc[f, 'usable_pitch_count'] = 1
        result_timeline.loc[f, 'image_candidate_count'] = 1
        result_timeline.loc[f, ['image_x','image_y','pitch_x_m','pitch_y_m','pitch_x_normalized','pitch_y_normalized']] = [r.image_x,r.image_y,r.pitch_x_m,r.pitch_y_m,r.pitch_x_normalized,r.pitch_y_normalized]
        for name in ['left_anchor_frame','right_anchor_frame','left_anchor_candidate_index','right_anchor_candidate_index','gap_id']:
            result_timeline.loc[f, name] = getattr(r, name)
    result_timeline.to_csv(OUT/'all_frames.csv', index=False)
    decisions = pd.DataFrame(records)
    summary = dict(video_frames=N, fps=FPS, original_observed_candidates=len(source), original_usable_pitch_candidates=int(source.pitch_usable.sum()),
        original_observed_anchor_frames=len(anchors), predicted_frames=len(predicted), filled_gaps=int(decisions.decision.eq('filled_two_sided').sum()),
        final_usable_pitch_candidates=int(combined.pitch_usable.sum()), first_prediction_frame=int(predicted.source_frame.min()), last_prediction_frame=int(predicted.source_frame.max()),
        first_observed_anchor_frame=int(anchors.source_frame.min()), last_observed_anchor_frame=int(anchors.source_frame.max()),
        predictions_per_half={str(int(k)):int(v) for k,v in predicted.half.value_counts().sort_index().items()},
        methods=predicted.prediction_method.value_counts().to_dict(), gap_decisions=decisions.decision.value_counts().to_dict(),
        maximum_gap_missing_frames=MAX_GAP, maximum_short_linear_gap=5, maximum_support_distance_frames=SUPPORT_DISTANCE,
        maximum_speed_mps=MAX_SPEED, maximum_forward_backward_disagreement_m=MAX_DISAGREEMENT,
        both_endpoints_original_observations=True, predicted_points_used_as_anchors=False, leading_or_trailing_extrapolation=False,
        original_observed_coordinates_modified=False, homography_modified=False, uncertain_detections_promoted=False,
        full_match_detections_rerun=False, source_observations='only previously retained green candidates; no old SNP or ATD ball observations reinstated',
        input_sha256={str(path.relative_to(ROOT)):sha(path) for path in [BASE/'reviewed_ball_pitch.csv',BASE/'reviewed_all_frames.csv',HFILE]})
    write('summary.json',summary)
    print(json.dumps(summary,ensure_ascii=False,indent=2),flush=True)


def regression():
    def anchor(frame,x,half=1,observed=True):
        return dict(source_frame=frame,pitch_x_m=x,pitch_y_m=34.,half=half,is_observation=observed)
    a,b = anchor(10,10.),anchor(14,10.8)
    assert predict_gap(None,b)[1]=='missing_observed_endpoint'
    assert predict_gap(a,None)[1]=='missing_observed_endpoint'
    assert predict_gap(anchor(10,10.,observed=False),b)[1]=='predictions_cannot_be_anchors'
    assert predict_gap(a,anchor(14,10.8,half=2))[1]=='half_boundary'
    assert predict_gap(a,anchor(62,20.))[1]=='gap_longer_than_50_frames'
    pred,reason=predict_gap(a,b,anchor(9,9.8),anchor(15,11.))
    assert reason=='candidate' and pred['source_frames'].tolist()==[11,12,13]
    np.testing.assert_allclose(pred['pitch_xy'][:,0],[10.2,10.4,10.6],atol=1e-12)
    assert all(10 < f < 14 for f in pred['source_frames'])
    assert a['pitch_x_m']==10. and b['pitch_x_m']==10.8
    assert predict_gap(a,anchor(40,25.),anchor(9,10.),anchor(41,25.))[1]=='forward_backward_predictions_disagree'
    assert predict_gap(a,anchor(40,16.))[1]=='long_gap_without_two_sided_velocity_support'
    print(json.dumps(dict(regression_passed=True,one_sided_prediction_rejected=True,recursive_anchor_rejected=True,half_crossing_rejected=True,contradictory_directions_rejected=True)))


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('stage',choices=['prepare','regression'])
    globals()[p.parse_args().stage]()


if __name__=='__main__':
    main()
