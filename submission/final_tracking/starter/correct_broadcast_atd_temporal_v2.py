"""Conservative, two-observed-ended temporal corrections of the latest ATD.

Original/previous artifacts are immutable. Estimates are confined to short
closed excursions or missing runs with unchanged real observed endpoints.
"""
from pathlib import Path
from functools import lru_cache
import argparse
import itertools
import json
import shutil
import sys

import cv2
import numpy as np
import pandas as pd
import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from starter.refine_ball_temporal_ai import curves, model
from starter.fill_ball_bidirectional import local_histogram
from starter.apply_ball_precision import annotations, manual_label
from starter.render_broadcast_atd_from_csv import ball_text, clean, digest

PREV = ROOT / 'outputs/broadcast_atd_y_minus_0p1_original_v1'
OUT = ROOT / 'outputs/broadcast_atd_temporal_v2'
DATA = ROOT / 'data/derived/ball_temporal_ai_20261117'
RAW = ROOT / 'data/broadcast/BAR-ATH_broadcast_atd.txt'
BEFORE = PREV / RAW.name
AFTER = OUT / RAW.name
HFILE = RAW.with_name('BAR-ATH_broadcast_homography.csv')
VIDEO = RAW.with_name('BAR-ATH_broadcast-001.mp4')
EVIDENCE = ROOT / 'data/derived/ball_fp_reduction_20261116/evidence.npz'
N = 164487
PITCH = np.array([105., 68.])
LEGACY = np.array([105.3 / 105., 1.])


def read(path):
    return pd.read_csv(path, float_precision='round_trip', low_memory=False)


def save(name, value):
    (OUT / name).write_text(json.dumps(clean(value), ensure_ascii=False, indent=2,
                                     allow_nan=False) + '\n', encoding='utf-8')


class Correction:
    def __init__(self):
        torch.set_num_threads(2)
        self.original = ball_text(RAW)
        self.before = ball_text(BEFORE)
        self.xy = self.before[:N, :2] * PITCH
        self.norm = self.before[:N, :2]
        self.valid = np.isfinite(self.xy).all(axis=1)
        self.source = read(DATA / 'ball_homography_all_frames.csv')
        self.previous = read(PREV / 'corrected_ball_coordinates.csv')
        self.h = read(HFILE).iloc[:N]
        self.hm = self.h[[f'h{i}{j}' for i in range(3) for j in range(3)]].to_numpy().reshape(-1, 3, 3)
        self.hgood = self.h.valid.eq(1).to_numpy() & np.isfinite(self.hm).all(axis=(1, 2))
        self.hgood &= self.h.reprojection_error.le(5).to_numpy()
        self.half = self.source.half.to_numpy(int)
        self.source_norm = self.source[['pitch_x_normalized', 'pitch_y_normalized']].to_numpy(float)
        self.source_xy = self.source_norm * PITCH
        self.same = (self.norm == self.source_norm).all(axis=1) & self.valid
        self.pixels = self.source[['image_x', 'image_y']].to_numpy(float)
        self.observed = self.source.ball_detected.to_numpy(bool) & self.source.pitch_usable.to_numpy(bool)
        self.observed &= ~self.source.is_prediction.to_numpy(bool) & self.same
        self.observed &= np.isfinite(self.pixels).all(axis=1)
        self.observed &= ((self.pixels >= [0, 0]) & (self.pixels < [1920, 1080])).all(axis=1)
        self.frames = np.flatnonzero(self.observed)
        self.cut = np.zeros(N, bool)
        with np.load(EVIDENCE) as z:
            mf = z['motion_frames']; distance = z['cut_color_distance']
            for i in np.flatnonzero(distance > .3):
                self.cut[max(0, int(mf[max(i - 1, 0)]) + 1):int(mf[i]) + 1] = True
        self.bank = read(DATA / 'candidate_bank.csv')
        self.lookup = {(int(r.source_frame), int(r.candidate_index)): int(r.feature_index)
                       for r in self.bank.itertuples()}
        self.feature = np.load(DATA / 'appearance_features.npy', mmap_mode='r')
        self.bank_frames = self.bank.source_frame.to_numpy(int)
        assert np.all(np.diff(self.bank_frames) >= 0)
        self.refs = annotations()
        decisions = read(ROOT / 'data/derived/ball_precision_review_20261117/final/observation_decisions.csv')
        self.blocked = {int(r.source_frame): r for r in decisions[decisions.reject_nonball].itertuples()}
        self.fixed = self.source.manual_object_label.eq('ball').to_numpy() & self.same
        checkpoint = torch.load(DATA / 'bidirectional_blend_model.pt', map_location='cpu', weights_only=True)
        self.net = model(); self.net.load_state_dict(checkpoint['state_dict']); self.net.eval()
        self.mean = np.asarray(checkpoint['mean'], np.float32)
        self.std = np.asarray(checkpoint['std'], np.float32)
        evaluation = json.loads((OUT / 'model_validation.json').read_text())
        self.method = evaluation.get('selected_method', evaluation.get('selected'))
        assert self.method in ['learned', 'hermite', 'linear']
        self.decisions = []

    @lru_cache(maxsize=8192)
    def vector(self, index):
        x = self.feature[index].astype(np.float32)
        return x / max(np.linalg.norm(x), 1e-9)

    def feature_id(self, f):
        cid = self.source.candidate_index.iloc[f]
        return None if not np.isfinite(cid) else self.lookup.get((int(f), int(cid)))

    def similarity(self, a, b):
        ia, ib = self.feature_id(a), self.feature_id(b)
        return None if ia is None or ib is None else float(self.vector(ia) @ self.vector(ib))

    def project(self, frames, xy):
        q = np.einsum('nij,nj->ni', self.hm[frames], np.c_[xy / PITCH, np.ones(len(frames))])
        with np.errstate(divide='ignore', invalid='ignore'):
            return q[:, :2] / q[:, 2:]

    def propose(self, a, b, kind):
        rec = dict(left_anchor_frame=int(a), right_anchor_frame=int(b),
                   interior_frames=int(b - a - 1), kind=kind, decision='REJECTED')
        self.decisions.append(rec)
        def reject(reason):
            rec['reason'] = reason
            return None
        if a < 0 or b >= N or not self.observed[a] or not self.observed[b]:
            return reject('BOTH_REAL_IMAGE_OBSERVATIONS_REQUIRED')
        if not 1 <= b - a - 1 <= (12 if kind == 'OUTLIER' else 50):
            return reject('SPAN_TOO_LONG')
        if self.half[a] not in [1, 2] or not (self.half[a:b + 1] == self.half[a]).all():
            return reject('HALF_OR_EXPORT_BOUNDARY')
        if not self.hgood[a:b + 1].all() or self.cut[a + 1:b + 1].any():
            return reject('H_INVALID_OR_CAMERA_CUT')
        c = curves(self.xy * LEGACY, a, b, self.frames, self.observed)
        if c is None:
            return reject('TWO_SIDED_VELOCITY_SUPPORT_REQUIRED')
        context = c['context']
        rec.update(context_frames=context.tolist(), max_context_speed_mps=c['max_speed'],
                   context_fit_residual_m=c['residual'], forward_backward_disagreement_m=c['disagreement'])
        if not (self.half[context] == self.half[a]).all() or self.cut[context.min() + 1:context.max() + 1].any():
            return reject('CONTEXT_HALF_OR_CUT')
        if not self.hgood[context.min():context.max() + 1].all():
            return reject('CONTEXT_H_INVALID')
        if c['max_speed'] > 40 or c['disagreement'] > 2 or c['residual'] > (.5 if kind == 'OUTLIER' else .4):
            return reject('FORWARD_BACKWARD_MOTION_DISAGREES')
        similarity = self.similarity(a, b)
        rec['endpoint_appearance_cosine'] = similarity
        if similarity is None or similarity < .8:
            return reject('ENDPOINT_APPEARANCE_DISAGREES')
        fs = np.arange(a + 1, b)
        if self.method == 'learned':
            features = np.clip((c['features'] - self.mean) / self.std, -6, 6).astype(np.float32)
            with torch.no_grad():
                alpha = self.net(torch.from_numpy(features)).numpy().ravel()
        else:
            alpha = np.full(len(fs), 1. if self.method == 'hermite' else 0.)
        world = (alpha[:, None] * c['hermite'] + (1 - alpha[:, None]) * c['line']) / LEGACY
        path = np.vstack([self.xy[a], world, self.xy[b]])
        if np.linalg.norm(np.diff(path, axis=0), axis=1).max() * 25 > 40:
            return reject('PREDICTED_SPEED_INCONSISTENT')
        if not ((world >= [-20, -20]) & (world <= [125, 88])).all():
            return reject('PREDICTED_PITCH_PROJECTION_OUTLIER')
        probes = np.array([[.25, .25, 1], [.5, .5, 1], [.75, .75, 1]])
        ray = np.einsum('nij,kj->nki', self.hm[a:b + 1], probes)
        with np.errstate(divide='ignore', invalid='ignore'):
            pp = ray[:, :, :2] / ray[:, :, 2:]
        jump = np.median(np.linalg.norm(np.diff(pp, axis=0), axis=2), axis=1).max()
        rec['max_camera_probe_step_px'] = jump
        if not np.isfinite(pp).all() or jump > 160:
            return reject('H_CAMERA_DISCONTINUITY')
        image = self.project(fs, world)
        if not np.isfinite(image).all() or not ((image >= [4, 4]) & (image < [1916, 1076])).all():
            return reject('PREDICTED_IMAGE_POSITION_UNAVAILABLE')
        if kind == 'OUTLIER':
            residual = np.linalg.norm(world - self.xy[fs], axis=1)
            rec.update(min_old_path_residual_m=float(residual.min()), max_old_path_residual_m=float(residual.max()))
            if residual.min() < 1.5 or residual.max() < 2.5:
                return reject('NOT_A_CLOSED_LOCAL_EXCURSION')
            if self.fixed[fs].any():
                return reject('EXACT_PREVIOUS_BALL_REVIEW_PRESERVED')
        for f, pixel in zip(fs, image):
            if manual_label(dict(source_frame=int(f), image_x=pixel[0], image_y=pixel[1]), self.refs) in ['nonball', 'uncertain', 'annotation_conflict']:
                return reject('PREDICTED_POSITION_CONFLICTS_WITH_PREVIOUS_OBJECT_REVIEW')
        rec.update(decision='PROPOSED', reason='TWO_REAL_ENDPOINTS_AND_BIDIRECTIONAL_CONTEXT_AGREE')
        return dict(a=a, b=b, fs=fs, world=world, image=image, alpha=alpha, context=context, record=rec)

    def recover(self, f, predicted, pixel, a, b):
        ia, ib = self.feature_id(a), self.feature_id(b)
        left, right = self.vector(ia), self.vector(ib)
        found = []
        lo, hi = np.searchsorted(self.bank_frames, [f, f + 1])
        for obj in self.bank.iloc[lo:hi].to_dict('records'):
            label = manual_label(obj, self.refs)
            if label in ['nonball', 'uncertain', 'annotation_conflict'] or bool(obj['strong_nonball_agreement']):
                continue
            blocked = self.blocked.get(f)
            if blocked is not None and label != 'ball':
                if blocked.bbox_x1 - 2 <= obj['image_x'] <= blocked.bbox_x2 + 2 and blocked.bbox_y1 - 2 <= obj['image_y'] <= blocked.bbox_y2 + 2:
                    continue
            if label != 'ball' and (min(obj['score_v2'], obj['score_v3']) < .03 or max(obj['score_v2'], obj['score_v3']) < .4):
                continue
            if np.isfinite(obj['confidence']) and obj['confidence'] < .08:
                continue
            size = max(obj['bbox_x2'] - obj['bbox_x1'], obj['bbox_y2'] - obj['bbox_y1'])
            image_distance = np.linalg.norm(pixel - [obj['image_x'], obj['image_y']])
            if image_distance > max(9., min(24., 1.3 * size)):
                continue
            v = self.vector(int(obj['feature_index']))
            sim_l, sim_r = float(v @ left), float(v @ right)
            if min(sim_l, sim_r) < .92:
                continue
            try:
                q = np.linalg.solve(self.hm[f], np.array([obj['image_x'], obj['image_y'], 1.]))
                world = q[:2] / q[2] * PITCH
            except np.linalg.LinAlgError:
                continue
            residual = float(np.linalg.norm(world - predicted))
            if not np.isfinite(world).all() or residual > 1.:
                continue
            obj.update(world=world, appearance_left=sim_l, appearance_right=sim_r,
                       prediction_distance_m=residual, prediction_distance_px=float(image_distance),
                       association_score=min(sim_l, sim_r) - .005 * residual - .001 * image_distance)
            found.append(obj)
        found.sort(key=lambda x: x['association_score'], reverse=True)
        unique = []
        for obj in found:
            if any(np.hypot(obj['image_x'] - q['image_x'], obj['image_y'] - q['image_y']) < 4 for q in unique):
                continue
            unique.append(obj)
        if not unique or (len(unique) > 1 and unique[0]['association_score'] - unique[1]['association_score'] < .025):
            return None
        return unique[0]

    def candidates(self):
        step = np.linalg.norm(np.diff(self.xy, axis=0), axis=1)
        edges = np.flatnonzero((step > 2.5) & self.valid[:-1] & self.valid[1:])
        proposals = []
        for a in edges:
            if not self.observed[a]:
                continue
            for length in range(1, 13):
                b = int(a + length + 1)
                if b >= N or not self.observed[b] or step[b - 1] <= 2.5 or not self.valid[a:b + 1].all():
                    continue
                item = self.propose(int(a), b, 'OUTLIER')
                if item is not None:
                    proposals.append(item)
        missing = ~self.valid
        starts = np.flatnonzero(missing & ~np.r_[False, missing[:-1]])
        ends = np.flatnonzero(missing & ~np.r_[missing[1:], False])
        for start, end in zip(starts, ends):
            item = self.propose(int(start) - 1, int(end) + 1, 'MISSING')
            if item is not None:
                proposals.append(item)
        return proposals

    def select(self, proposals):
        used = np.zeros(N, bool); protected = self.fixed.copy(); chosen = []
        for item in sorted(proposals, key=lambda v: (len(v['fs']), -v['record'].get('min_old_path_residual_m', 0))):
            fs = item['fs']; context = item['context']; rec = item['record']
            if used[context].any() or protected[fs].any() or used[fs].any():
                rec.update(decision='REJECTED', reason='OVERLAP_OR_IMMUTABLE_CONTEXT')
                continue
            used[fs] = True; protected[context] = True
            chosen.append(item)
        cap = cv2.VideoCapture(str(VIDEO)); assert cap.isOpened()
        accepted = []
        for item in sorted(chosen, key=lambda v: v['a']):
            a, b, rec = item['a'], item['b'], item['record']
            cap.set(cv2.CAP_PROP_POS_FRAMES, a)
            prior_small = prior_hist = None; max_color = max_mad = 0.
            for f in range(a, b + 1):
                ok, image = cap.read(); assert ok, f
                small, hist = local_histogram(image)
                if prior_hist is not None:
                    max_color = max(max_color, float(cv2.compareHist(prior_hist, hist, cv2.HISTCMP_BHATTACHARYYA)))
                    max_mad = max(max_mad, float(np.abs(small.astype(float) - prior_small).mean() / 255.))
                prior_small, prior_hist = small, hist
            rec.update(actual_video_frames_checked=b - a + 1, per_frame_max_color_distance=max_color,
                       per_frame_max_image_difference=max_mad)
            if max_color > .35 or max_mad > .28:
                rec.update(decision='REJECTED', reason='ACTUAL_PER_FRAME_CAMERA_TRANSITION')
                continue
            rec.update(decision='ACCEPTED', reason='TWO_SIDED_MOTION_APPEARANCE_AND_VIDEO_CUT_GATES_PASSED')
            accepted.append(item)
        cap.release()
        return accepted


def temporal_stats(xy, half, cut):
    valid = np.isfinite(xy).all(axis=1)
    mask = valid[:-1] & valid[1:] & (half[:-1] == half[1:]) & np.isin(half[1:], [1, 2]) & ~cut[1:]
    step = np.linalg.norm(np.diff(xy, axis=0), axis=1)
    v = step[mask]
    residual = np.linalg.norm(xy[1:-1] - (xy[:-2] + xy[2:]) / 2, axis=1)
    spikes = valid[1:-1] & valid[:-2] & valid[2:] & (step[:-1] > 2) & (step[1:] > 2)
    spikes &= (residual > 2) & (np.linalg.norm(xy[2:] - xy[:-2], axis=1) < 1.5)
    spikes &= (half[:-2] == half[1:-1]) & (half[1:-1] == half[2:]) & np.isin(half[1:-1], [1, 2]) & ~cut[1:-1] & ~cut[2:]
    return dict(valid_frames=int(valid.sum()), comparable_adjacent_steps=int(mask.sum()),
        step_median_m=float(np.median(v)), step_p95_m=float(np.quantile(v, .95)),
        step_p99_m=float(np.quantile(v, .99)), step_maximum_m=float(v.max()),
        steps_over_3m=int((v > 3).sum()), steps_over_5m=int((v > 5).sum()),
        steps_over_10m=int((v > 10).sum()), isolated_return_spike_candidates=int(spikes.sum()))


def run():
    OUT.mkdir(parents=True, exist_ok=True)
    if AFTER.exists():
        raise FileExistsError(f'Refusing existing export {AFTER}')
    required = [RAW, BEFORE, HFILE, DATA / 'ball_homography_all_frames.csv',
        DATA / 'candidate_bank.csv', DATA / 'appearance_features.npy', DATA / 'bidirectional_blend_model.pt',
        PREV / 'corrected_ball_coordinates.csv', EVIDENCE,
        ROOT / 'data/derived/ball_precision_review_20261117/final/observation_decisions.csv']
    hashes = {str(p.relative_to(ROOT)): digest(p) for p in required}
    video_stat = dict(bytes=VIDEO.stat().st_size, mtime_ns=VIDEO.stat().st_mtime_ns)
    work = Correction()
    proposals = work.candidates()
    accepted = work.select(proposals)
    after = work.before.copy(); changes = []; contexts = set()
    for segment, item in enumerate(accepted):
        a, b, fs = item['a'], item['b'], item['fs']
        contexts.update(item['context'].tolist())
        candidate_rows = [work.recover(int(f), world, pixel, a, b)
                          for f, world, pixel in zip(fs, item['world'], item['image'])]
        alternative = np.array([obj['world'] if obj is not None else predicted
                                for obj, predicted in zip(candidate_rows, item['world'])])
        path = np.vstack([work.xy[a], alternative, work.xy[b]])
        if np.linalg.norm(np.diff(path, axis=0), axis=1).max() * 25 > 40:
            candidate_rows = [None] * len(fs)
            alternative = item['world']
            item['record']['candidate_recovery_withheld_due_to_path_speed'] = True
        for j, f in enumerate(fs):
            obj = candidate_rows[j]
            norm = alternative[j] / PITCH
            # Exact reuse of the existing measured normalized coordinate, if available.
            if obj is not None and work.source.ball_detected.iloc[f] and work.source.pitch_usable.iloc[f]:
                if int(obj['candidate_index']) == int(work.source.candidate_index.iloc[f]):
                    norm = work.source_norm[f].copy()
            after[f, :2] = norm
            predicted = obj is None
            action = ('AI_OUTLIER_CORRECTION' if item['record']['kind'] == 'OUTLIER' else 'AI_GAP_FILL') if predicted else 'RECOVERED_OBSERVATION'
            changes.append(dict(frame=int(f), atd_frame_id=int(f) + 1, segment=segment,
                temporal_action=action, left_anchor_frame=a, right_anchor_frame=b,
                context_frames=json.dumps(item['context'].tolist()),
                left_anchor_source='EXACT_UNCHANGED_ACTUAL_CSV_OBSERVATION',
                right_anchor_source='EXACT_UNCHANGED_ACTUAL_CSV_OBSERVATION',
                final_source='AI_TWO_SIDED_' + work.method.upper() if predicted else 'RECOVERED_' + str(obj['origin']).upper(),
                final_source_is_prediction=predicted, candidate_index=None if predicted else int(obj['candidate_index']),
                detection_origin='' if predicted else obj['origin'],
                confidence=None if predicted else obj['confidence'],
                model_alpha=float(item['alpha'][j]) if predicted else None,
                prediction_method=work.method if predicted else 'EXISTING_MEASURED_CANDIDATE_TWO_SIDED_ASSOCIATION',
                appearance_left=None if predicted else obj['appearance_left'],
                appearance_right=None if predicted else obj['appearance_right'],
                forward_backward_disagreement_m=item['record']['forward_backward_disagreement_m'],
                before_x=work.before[f, 0], before_y=work.before[f, 1], final_x=norm[0], final_y=norm[1],
                image_x=float(item['image'][j, 0]) if predicted else obj['image_x'],
                image_y=float(item['image'][j, 1]) if predicted else obj['image_y']))
    log = pd.DataFrame(changes)
    assert len(log) and log.frame.is_unique
    log = log.sort_values('frame').reset_index(drop=True)
    changed = np.zeros(N + 1, bool); changed[log.frame.to_numpy(int)] = True
    assert not (set(log.frame) & contexts)
    assert np.array_equal(after[list(contexts), :2], work.before[list(contexts), :2], equal_nan=True)
    with BEFORE.open('rb') as stream, AFTER.open('wb') as output:
        for frame, line in enumerate(stream):
            if changed[frame]:
                body = line.rstrip(b'\r\n'); ending = line[len(body):]
                prefix, tokens = body.rsplit(b':', 1); values = tokens.split(b',')
                values[:2] = [format(float(v), '.17g').encode('ascii') for v in after[frame, :2]]
                output.write(prefix + b':' + b','.join(values) + ending)
            else:
                output.write(line)
    shutil.copyfile(RAW.with_suffix('.xml'), AFTER.with_suffix('.xml'))
    actual = ball_text(AFTER)
    assert np.array_equal(actual, after, equal_nan=True)
    result = pd.DataFrame(dict(frame=np.arange(N + 1), atd_frame_id=np.arange(1, N + 2),
        source_frame=np.r_[np.arange(N), np.nan], video_seconds=np.arange(N + 1) / 25,
        before_x=work.before[:, 0], before_y=work.before[:, 1], final_x=after[:, 0], final_y=after[:, 1],
        before_x_m=work.before[:, 0] * 105, before_y_m=work.before[:, 1] * 68,
        final_x_m=after[:, 0] * 105, final_y_m=after[:, 1] * 68,
        temporal_action='UNCHANGED', final_source=work.previous.final_source,
        final_source_is_prediction=work.previous.final_source_is_prediction,
        left_anchor_frame=np.nan, right_anchor_frame=np.nan,
        original_atd_x=work.original[:, 0], original_atd_y=work.original[:, 1],
        source_detection_origin=list(work.source.detection_origin.fillna('')) + [''],
        candidate_index=np.r_[work.source.candidate_index.to_numpy(), np.nan],
        image_x=np.nan, image_y=np.nan, image_coordinate_kind='H_REPROJECTED_PITCH_COORDINATE',
        input_actual_anchor_eligible=np.r_[work.observed, False],
        prior_original_xy_restored=work.previous.original_xy_restored))
    common = [key for key in log.columns if key in result.columns and key not in ['frame', 'atd_frame_id', 'before_x', 'before_y']]
    ids = log.frame.to_numpy(int)
    result.loc[ids, common] = log[common].to_numpy()
    measured = np.flatnonzero(np.isfinite(after[:N, :2]).all(axis=1) & work.hgood)
    image = work.project(measured, after[measured, :2] * PITCH)
    result.loc[measured, ['image_x', 'image_y']] = image
    recovered_ids = log.loc[log.temporal_action.eq('RECOVERED_OBSERVATION'), 'frame'].to_numpy(int)
    result.loc[recovered_ids, ['image_x', 'image_y']] = log.set_index('frame').loc[recovered_ids, ['image_x', 'image_y']].to_numpy()
    result.loc[recovered_ids, 'image_coordinate_kind'] = 'ACTUAL_REUSED_DETECTION'
    original_actual = np.r_[work.observed, False] & ~changed
    original_ids = np.flatnonzero(original_actual)
    result.loc[original_ids, ['image_x', 'image_y']] = work.pixels[original_ids]
    result.loc[original_ids, 'image_coordinate_kind'] = 'ACTUAL_EXISTING_DETECTION'
    result['final_actual_observation'] = original_actual
    result.loc[recovered_ids, 'final_actual_observation'] = True
    result.loc[recovered_ids, 'source_detection_origin'] = log.set_index('frame').loc[recovered_ids, 'detection_origin'].to_numpy()
    result.to_csv(OUT / 'corrected_ball_coordinates.csv', index=False)
    timeline = result.iloc[:N].copy()
    timeline['half'] = work.half
    timeline['pitch_x_normalized'] = timeline.final_x
    timeline['pitch_y_normalized'] = timeline.final_y
    timeline['pitch_x_m'] = timeline.final_x_m
    timeline['pitch_y_m'] = timeline.final_y_m
    timeline['is_prediction'] = timeline.final_source_is_prediction
    timeline['new_temporal_estimate'] = timeline.temporal_action.isin(['AI_OUTLIER_CORRECTION', 'AI_GAP_FILL'])
    timeline['actual_recovered_detection'] = timeline.temporal_action.eq('RECOVERED_OBSERVATION')
    timeline['ball_observed'] = timeline.final_actual_observation
    timeline['ball_detected'] = timeline.final_actual_observation
    timeline['detection_origin'] = np.where(timeline.final_actual_observation, timeline.source_detection_origin, '')
    timeline['homography_valid'] = work.hgood
    timeline['ball_position_available'] = np.isfinite(after[:N, :2]).all(axis=1)
    timeline.to_csv(OUT / 'ball_homography_all_frames.csv', index=False)
    log.to_csv(OUT / 'correction_log.csv', index=False)
    decisions = pd.DataFrame(work.decisions)
    decisions.to_csv(OUT / 'segment_decisions.csv', index=False)
    decisions[decisions.decision.ne('ACCEPTED')].to_csv(OUT / 'unresolved_or_rejected_candidates.csv', index=False)
    before_stats = temporal_stats(work.xy, work.half, work.cut)
    after_stats = temporal_stats(after[:N, :2] * PITCH, work.half, work.cut)
    # Every new estimate is strictly inside two frozen actual observed endpoints.
    assert all(work.observed[r.left_anchor_frame] and work.observed[r.right_anchor_frame]
               and r.left_anchor_frame < r.frame < r.right_anchor_frame for r in log.itertuples())
    assert not changed[-1] and np.array_equal(work.before[:, 2], after[:, 2], equal_nan=True)
    assert np.isfinite(after[:N, :2]).all(axis=1)[work.valid].all()
    byte_contract = dict(rows=0, all_unmodified_rows_byte_identical=True, player_prefix_speed_newlines_preserved=True)
    with BEFORE.open('rb') as a, AFTER.open('rb') as b:
        for f, (old, new) in enumerate(itertools.zip_longest(a, b)):
            assert old is not None and new is not None
            old_body = old.rstrip(b'\r\n'); new_body = new.rstrip(b'\r\n')
            if not changed[f]: assert old == new
            assert old_body.rsplit(b':', 1)[0] == new_body.rsplit(b':', 1)[0]
            assert old_body.rsplit(b',', 1)[1] == new_body.rsplit(b',', 1)[1]
            assert old[len(old_body):] == new[len(new_body):]
            byte_contract['rows'] += 1
    assert byte_contract['rows'] == N + 1
    assert all(digest(ROOT / p) == signature for p, signature in hashes.items())
    assert dict(bytes=VIDEO.stat().st_size, mtime_ns=VIDEO.stat().st_mtime_ns) == video_stat
    summary = dict(status='COORDINATES_COMPLETE', input_ATD=str(BEFORE.relative_to(ROOT)),
        video_frames=N, ATD_rows=N + 1, input_valid_frames=int(work.valid.sum()),
        strict_unchanged_actual_observation_anchors=int(work.observed.sum()),
        corrected_frames=len(log), corrected_existing_frames=int((changed[:N] & work.valid).sum()),
        filled_missing_frames=int((changed[:N] & ~work.valid).sum()),
        recovered_actual_detection_frames=int(log.temporal_action.eq('RECOVERED_OBSERVATION').sum()),
        AI_estimated_frames=int(log.final_source_is_prediction.sum()),
        accepted_segments=len(accepted), selected_model=work.method,
        actions=log.temporal_action.value_counts().to_dict(), recovered_detection_origins=log.detection_origin.value_counts().to_dict(),
        before_temporal_metrics=before_stats, after_temporal_metrics=after_stats,
        auto_anomaly_candidates_are_not_confirmed_nonballs=True,
        actual_position_accuracy_improvement_claimed=False, no_one_sided_or_recursive_prediction=True,
        no_new_whole_video_ball_detection=True, original_and_previous_files_preserved=True,
        homography_matrices_modified=False, input_hashes=hashes, source_video_stat=video_stat)
    save('correction_summary.json', summary)
    save('test_results.json', dict(passed=True, exact_ATD_and_CSV_match=True,
        byte_contract=byte_contract, player_and_speed_channels_preserved=True,
        all_new_estimates_have_both_real_unchanged_endpoints=True,
        all_support_context_frames_unchanged=True, predictions_never_used_as_anchors=True,
        every_accepted_span_has_full_actual_video_cut_check=True,
        original_missing_frames_not_one_sided_filled=True,
        no_current_finite_coordinate_lost=True, all_source_hashes_unchanged=True,
        source_SNP_not_relabelled_as_YOLO=True, extra_ATD_row_preserved=True))
    (OUT / 'README.md').write_text(f'''# 앞뒤 실제 관측 기반 공 좌표 시간 일관성 보정

시작점은 `broadcast_atd_y_minus_0p1_original_v1`의 ATD입니다. 전체 {N:,}프레임을 검사해
짧은 왕복 튐과 짧은 결측 가운데 **양쪽 실제 공 관측과 주변 속도가 일치하는 구간만** 보정했습니다.
총 {len(log):,}프레임, 실제 검출 후보 재사용 {summary['recovered_actual_detection_frames']:,}프레임,
양쪽 관측 기반 추정 {summary['AI_estimated_frames']:,}프레임입니다.

새 추정은 기존 CSV의 실제 이미지 관측과 현재 좌표가 정확히 일치하는 {int(work.observed.sum()):,}개 앵커만 사용합니다.
원본 ATD 참조·기존 예측·보관된 미확정 검출을 실제 YOLO 관측으로 간주하지 않습니다.
양끝 및 양쪽 속도지원 관측은 수정하지 않고 고정하며, 새 예측을 다시 앵커로 사용하지 않습니다.
앞/뒤 한쪽만 있는 구간, 긴 결측, 서로 다른 전후반, 컷, H 단절, 외형/운동 불일치 구간은 유지합니다.
속도만으로 이상을 확정하지 않습니다. 양쪽 궤적 일치·외형 임베딩 유사도·실제 영상 모든 프레임의 컷 검사를 함께 사용합니다.

기존 17→32→16→1 신경망과 외형 임베딩을 재사용했습니다. 새 라벨이나 전체 영상 재탐지는 만들지 않았습니다.
현재 관측을 인위적으로 가린 대리평가에서 선택한 궤적 방법은 `{work.method}`입니다.
이 평가는 실제 위치 정답 검증이 아니며, 수치가 매끄러워져도 실제 공 위치 정확도 개선을 주장하지 않습니다.
SNP·YOLO 검출 후보는 원래 출처를 유지하고, 추정 좌표는 실제 관측과 별도로 표시합니다.

[새 ATD](BAR-ATH_broadcast_atd.txt) · [전체 프레임 좌표](ball_homography_all_frames.csv) · [보정 기록](correction_log.csv) · [보류 기록](unresolved_or_rejected_candidates.csv)

`final_x/y`, `pitch_x/y_normalized`는 정규화 좌표, `_m`는 105×68m 미터 좌표입니다.
ATD는164,488행, 영상용 CSV는164,487프레임입니다. 마지막 추가 행을 보존했습니다.
선수·속도(s)·원본 H·원본/기존 ATD·기존 YOLO/SNP 결과는 덮어쓰지 않았습니다.
화면의 영상 점은 원본 H로 경기장 좌표를 재투영한 점이며 실제 이미지 검출과 다를 수 있습니다.
''', encoding='utf-8')
    return summary


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('stage', choices=['correct'])
    parser.parse_args()
    run()
