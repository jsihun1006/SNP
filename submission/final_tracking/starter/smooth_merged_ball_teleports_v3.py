"""Local, two-ended smoothing of impossible excursions in the current merged V2.

Preserve the valid/missing mask and all unmodified records. Recorded Tactical
and actual broadcast positions are frozen anchors; predictions never are.
"""
from pathlib import Path
import itertools
import json
import shutil
import sys

import cv2
import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from starter.refine_ball_temporal_ai import curves
from starter.fill_ball_bidirectional import local_histogram
from starter.render_broadcast_atd_from_csv import ball_text, clean, digest

BASE = ROOT / 'outputs/broadcast_tactical_ball_merge_v2'
OUT = ROOT / 'outputs/broadcast_tactical_ball_smoothed_v3'
BEFORE = BASE / 'BAR-ATH_broadcast_atd.txt'
TABLE = BASE / 'ball_homography_all_frames.csv'
AFTER = OUT / BEFORE.name
HFILE = ROOT / 'data/broadcast/BAR-ATH_broadcast_homography.csv'
BVIDEO = ROOT / 'data/broadcast/BAR-ATH_broadcast-001.mp4'
TVIDEO = ROOT / 'data/tactical/BAR-ATH_tactical-001.mp4'
EVIDENCE = ROOT / 'data/derived/ball_fp_reduction_20261116/evidence.npz'
N = 164487
PITCH = np.array([105., 68.])


def read(path): return pd.read_csv(path, float_precision='round_trip', low_memory=False)


def save(name, data):
    (OUT / name).write_text(json.dumps(clean(data), ensure_ascii=False, indent=2,
                                     allow_nan=False) + '\n', encoding='utf-8')


def metrics(xy, half, reliable):
    valid = np.isfinite(xy).all(axis=1)
    pair = valid[:-1] & valid[1:] & (half[:-1] == half[1:]) & (half[1:] > 0)
    step = np.linalg.norm(np.diff(xy, axis=0), axis=1)
    def stats(mask):
        s = step[mask]
        return dict(pairs=int(mask.sum()), median_step_m=float(np.median(s)),
            p99_step_m=float(np.quantile(s, .99)), maximum_step_m=float(s.max()),
            steps_over_3m=int((s > 3).sum()), steps_over_5m=int((s > 5).sum()), steps_over_10m=int((s > 10).sum()))
    return dict(valid_frames=int(valid.sum()), all_aligned_live_halves=stats(pair),
                frozen_reliable_live_mask=stats(pair & reliable[:-1] & reliable[1:]))


class Smoother:
    def __init__(self):
        self.d = read(TABLE)
        self.ball = ball_text(BEFORE)
        self.xy = self.ball[:N, :2] * PITCH
        assert np.array_equal(self.d[['final_x', 'final_y']], self.ball[:N, :2], equal_nan=True)
        self.valid = np.isfinite(self.xy).all(axis=1)
        self.tac = self.d.tactical_used.to_numpy(bool)
        self.hl = self.d.highlight_inferred.to_numpy(bool)
        self.half = self.d.alignment_half.to_numpy(int)
        self.chron = (self.half > 0) & ~(self.hl & ~self.tac)
        self.anchor = self.valid & self.chron & ~self.d.is_prediction.to_numpy(bool)
        self.anchor &= self.d.final_actual_observation.to_numpy(bool) | self.tac
        self.frames = np.flatnonzero(self.anchor)
        h = read(HFILE).iloc[:N]
        self.hm = h[[f'h{i}{j}' for i in range(3) for j in range(3)]].to_numpy().reshape(-1, 3, 3)
        self.hgood = h.valid.eq(1).to_numpy() & np.isfinite(self.hm).all(axis=(1, 2)) & h.reprojection_error.le(5).to_numpy()
        self.cut = np.zeros(N, bool)
        with np.load(EVIDENCE) as z:
            mf, distance = z['motion_frames'], z['cut_color_distance']
            for i in np.flatnonzero(distance > .3):
                self.cut[max(0, int(mf[max(i - 1, 0)]) + 1):int(mf[i]) + 1] = True
        self.step = np.linalg.norm(np.diff(self.xy, axis=0), axis=1)
        self.decisions = []

    def proposal(self, a, b, kind, seam=None):
        record = dict(kind=kind, left_anchor_frame=a, right_anchor_frame=b,
                      frames=b - a - 1, decision='REJECTED')
        if seam is not None: record['seam_frame'] = seam
        self.decisions.append(record)
        def reject(reason):
            record['reason'] = reason
            return None
        if not (0 <= a < b < N) or not self.anchor[a] or not self.anchor[b]: return reject('TWO_RECORDED_NONPREDICTED_ENDPOINTS_REQUIRED')
        if b - a - 1 < 1 or b - a - 1 > 25: return reject('SPAN_TOO_LONG')
        if not self.valid[a:b + 1].all(): return reject('DO_NOT_FILL_MISSING_FRAMES')
        if self.half[a] == 0 or not (self.half[a:b + 1] == self.half[a]).all(): return reject('HALF_OR_TIME_MAPPING_BOUNDARY')
        c = curves(self.xy, a, b, self.frames, self.anchor)
        if c is None: return reject('INSUFFICIENT_TWO_SIDED_MOTION_SUPPORT')
        context = c['context']; lo, hi = int(context.min()), int(context.max())
        pure_tactical = bool(self.tac[context].all())
        if not self.chron[context].all() or not (self.half[context] == self.half[a]).all(): return reject('CONTEXT_CHRONOLOGY')
        if not pure_tactical:
            if not self.chron[lo:hi + 1].all(): return reject('UNCERTAIN_BROADCAST_REPLAY_TIME')
            if self.cut[lo + 1:hi + 1].any(): return reject('BROADCAST_CONTEXT_CAMERA_CUT')
            broadcast_context = context[~self.tac[context]]
            if not self.hgood[broadcast_context].all(): return reject('BROADCAST_ENDPOINT_H_UNRELIABLE')
        record.update(context_frames=context.tolist(), all_tactical_context=pure_tactical,
            context_residual_m=c['residual'], forward_backward_difference_m=c['disagreement'],
            context_max_speed_mps=c['max_speed'], left_source=self.d.final_source.iloc[a], right_source=self.d.final_source.iloc[b])
        if c['max_speed'] > 40 or c['residual'] > (.5 if kind == 'CLOSED_EXCURSION' else .4): return reject('SIDE_MOTION_INCONSISTENT')
        if c['disagreement'] > (2 if kind == 'CLOSED_EXCURSION' else 8): return reject('FORWARD_BACKWARD_DISAGREEMENT')
        fs = np.arange(a + 1, b); world = c['hermite']
        if not ((world >= [-20, -20]) & (world <= [125, 88])).all(): return reject('SMOOTHED_PITCH_BOUNDS')
        new_path = np.vstack([self.xy[a], world, self.xy[b]])
        new_steps = np.linalg.norm(np.diff(new_path, axis=0), axis=1)
        if new_steps.max() * 25 > 40: return reject('NEW_PATH_SPEED_INCONSISTENT')
        residual = np.linalg.norm(world - self.xy[fs], axis=1)
        record.update(max_correction_m=float(residual.max()), min_correction_m=float(residual.min()),
                      new_max_step_m=float(new_steps.max()))
        if kind == 'CLOSED_EXCURSION':
            if residual.min() < 1.5 or residual.max() < 2.5: return reject('NOT_A_WHOLE_INTERIOR_EXCURSION')
        else:
            left, right = context[context <= a], context[context >= b]
            if len(left) < 4 or len(right) < 4: return reject('SEAM_REQUIRES_THREE_EXTRA_SUPPORTS_EACH_SIDE')
            if not (self.tac[left] == self.tac[a]).all() or not (self.tac[right] == self.tac[b]).all(): return reject('SEAM_SUPPORT_SOURCE_CHANGE')
            if not (self.tac[a:seam] == self.tac[a]).all() or not (self.tac[seam:b + 1] == self.tac[b]).all(): return reject('SEAM_NOT_SINGLE_SOURCE_TRANSITION')
            other_steps = self.step[a:b].copy(); other_steps[seam - a - 1] = 0
            if other_steps.max() * 25 > 40: return reject('SEAM_CONTAINS_ADDITIONAL_JUMP')
            vl = (self.xy[a] - self.xy[left[0]]) / ((a - left[0]) / 25)
            vr = (self.xy[right[-1]] - self.xy[b]) / ((right[-1] - b) / 25)
            left_fit = np.linalg.norm(self.xy[a:seam] - self.xy[a] - ((np.arange(a, seam) - a) / 25)[:, None] * vl, axis=1)
            right_fit = np.linalg.norm(self.xy[seam:b + 1] - self.xy[b] - ((np.arange(seam, b + 1) - b) / 25)[:, None] * vr, axis=1)
            if max(left_fit.max(), right_fit.max()) > .75: return reject('SEAM_SIDES_NOT_STABLE')
            if residual.max() > 4: return reject('SEAM_CORRECTION_EXCEEDS_4M_TUBE')
            if new_steps.max() >= self.step[seam - 1] * .75: return reject('SEAM_PEAK_NOT_MATERIALLY_REDUCED')
            record.update(old_seam_step_m=float(self.step[seam - 1]),
                          left_internal_residual_m=float(left_fit.max()), right_internal_residual_m=float(right_fit.max()))
        record.update(decision='PROPOSED', reason='TWO_ENDED_BOUNDED_WORLD_SMOOTHING')
        return dict(a=a, b=b, fs=fs, context=context, world=world, record=record)

    def propose_all(self):
        closed = []
        edges = np.flatnonzero((self.step > 2.5) & self.valid[:-1] & self.valid[1:])
        for a in edges:
            if not self.anchor[a]: continue
            for n in range(1, 26):
                b = int(a + n + 1)
                if b >= N or not self.anchor[b] or self.step[b - 1] <= 2.5 or not self.valid[a:b + 1].all(): continue
                item = self.proposal(int(a), b, 'CLOSED_EXCURSION')
                if item is not None: closed.append(item)
        used = np.zeros(N, bool); protected = np.zeros(N, bool)
        selected = []
        def accept(item):
            fs, context = item['fs'], item['context']
            if used[fs].any() or protected[fs].any() or used[context].any():
                item['record'].update(decision='REJECTED', reason='OVERLAP_OR_FROZEN_SUPPORT')
                return False
            used[fs] = True; protected[context] = True; selected.append(item)
            return True
        for item in sorted(closed, key=lambda r: (len(r['fs']), -r['record']['min_correction_m'])): accept(item)
        seams = np.flatnonzero((self.step > 3) & self.valid[:-1] & self.valid[1:] & (self.tac[:-1] != self.tac[1:])) + 1
        seam_proposals = []
        for seam in seams:
            for left in range(3, 13):
                a = int(seam - left)
                if a < 0 or not self.anchor[a] or not (self.tac[a:seam] == self.tac[a]).all(): continue
                for right in range(3, 13):
                    b = int(seam + right - 1)
                    if b >= N or not self.anchor[b] or not (self.tac[seam:b + 1] == self.tac[b]).all(): continue
                    fs = np.arange(a + 1, b)
                    if used[fs].any() or protected[fs].any(): continue
                    item = self.proposal(a, b, 'SOURCE_SEAM', int(seam))
                    if item is not None and not used[item['context']].any(): seam_proposals.append(item)
        chosen_seams = set()
        for item in sorted(seam_proposals, key=lambda r: (r['record']['max_correction_m'] + len(r['fs']) * .03, len(r['fs']))):
            seam = item['record']['seam_frame']
            if seam in chosen_seams:
                item['record'].update(decision='REJECTED', reason='ALTERNATE_WINDOW_OF_SELECTED_SEAM'); continue
            if accept(item): chosen_seams.add(seam)
        return sorted(selected, key=lambda r: r['a'])

    def video_check(self, selected):
        caps = {False: cv2.VideoCapture(str(BVIDEO)), True: cv2.VideoCapture(str(TVIDEO))}
        assert all(cap.isOpened() for cap in caps.values())
        accepted = []
        for item in selected:
            r = item['record']; tactical = bool(r['all_tactical_context'])
            lo, hi = int(item['context'].min()), int(item['context'].max())
            if tactical:
                first, last = int(self.d.tactical_frame.iloc[lo]), int(self.d.tactical_frame.iloc[hi])
                assert last - first == hi - lo
            else: first, last = lo, hi
            cap = caps[tactical]; cap.set(cv2.CAP_PROP_POS_FRAMES, first)
            prior = prior_hist = None; max_color = max_mad = 0.
            for f in range(first, last + 1):
                ok, image = cap.read(); assert ok, (tactical, f)
                small, hist = local_histogram(image)
                if prior_hist is not None:
                    max_color = max(max_color, float(cv2.compareHist(prior_hist, hist, cv2.HISTCMP_BHATTACHARYYA)))
                    max_mad = max(max_mad, float(np.abs(small.astype(float) - prior).mean() / 255.))
                prior, prior_hist = small, hist
            r.update(checked_video='TACTICAL' if tactical else 'BROADCAST', actual_checked_video_frames=last - first + 1,
                max_video_color_distance=max_color, max_video_pixel_difference=max_mad)
            if max_color > .35 or max_mad > .28:
                r.update(decision='REJECTED', reason='ACTUAL_SOURCE_VIDEO_SCENE_TRANSITION'); continue
            r.update(decision='ACCEPTED', reason='FROZEN_TWO_ENDED_CONTEXT_AND_SOURCE_VIDEO_VERIFIED')
            accepted.append(item)
        for cap in caps.values(): cap.release()
        return accepted

    def replacements(self, item):
        fs, a, b = item['fs'], item['a'], item['b']
        path = item['world'].copy(); kinds = ['ESTIMATE'] * len(fs)
        if item['record']['kind'] == 'SOURCE_SEAM': return path, kinds
        tac_xy = self.d[['tactical_x', 'tactical_y']].to_numpy(float) * PITCH
        old_b = self.d[['target_final_x', 'target_final_y']].to_numpy(float) * PITCH
        for j, f in enumerate(fs):
            # First reuse an existing recorded Tactical position, never a synthetic observation.
            if self.d.tactical_ball_available.iloc[f] and np.isfinite(tac_xy[f]).all():
                if np.linalg.norm(tac_xy[f] - path[j]) <= 1.:
                    path[j] = tac_xy[f]; kinds[j] = 'TACTICAL'; continue
            if self.d.target_final_actual_observation.iloc[f] and not self.d.target_is_prediction.iloc[f]:
                if np.isfinite(old_b[f]).all() and np.linalg.norm(old_b[f] - path[j]) <= 1.:
                    path[j] = old_b[f]; kinds[j] = 'BROADCAST'
        whole = np.vstack([self.xy[a], path, self.xy[b]])
        if np.linalg.norm(np.diff(whole, axis=0), axis=1).max() * 25 > 40:
            # Keep only recoveries that satisfy the complete, two-ended speed bound.
            path = item['world'].copy(); kinds = ['ESTIMATE'] * len(fs)
            item['record']['recorded_reselection_withheld_due_to_speed'] = True
        return path, kinds


def main():
    OUT.mkdir(parents=True, exist_ok=True)
    if AFTER.exists(): raise FileExistsError(f'Refusing existing output {AFTER}')
    required = [TABLE, BEFORE, BEFORE.with_suffix('.xml'), HFILE, EVIDENCE,
        ROOT / 'outputs/broadcast_atd_temporal_v2/ball_homography_all_frames.csv',
        ROOT / 'data/tactical/BAR-ATH_tactical_atd.txt', BASE / 'frame_alignment.csv']
    hashes = {str(p.relative_to(ROOT)): digest(p) for p in required}
    video_stats = {str(p.relative_to(ROOT)): dict(bytes=p.stat().st_size, mtime_ns=p.stat().st_mtime_ns) for p in [BVIDEO, TVIDEO]}
    work = Smoother(); selected = work.video_check(work.propose_all())
    after = work.ball.copy(); changes = []; protected = set()
    for segment, item in enumerate(selected):
        new_path, kinds = work.replacements(item)
        a, b = item['a'], item['b']; protected.update(item['context'].tolist())
        for j, f in enumerate(item['fs']):
            kind = kinds[j]
            if kind == 'TACTICAL': norm = work.d.loc[f, ['tactical_x', 'tactical_y']].to_numpy(float)
            elif kind == 'BROADCAST': norm = work.d.loc[f, ['target_final_x', 'target_final_y']].to_numpy(float)
            else: norm = new_path[j] / PITCH
            after[f, :2] = norm
            action = 'TACTICAL_REFERENCE_RESELECTED' if kind == 'TACTICAL' else ('BROADCAST_OBSERVATION_RESELECTED' if kind == 'BROADCAST' else
                ('BIDIRECTIONAL_SEAM_SMOOTHING' if item['record']['kind'] == 'SOURCE_SEAM' else 'BIDIRECTIONAL_OUTLIER_ESTIMATE'))
            changes.append(dict(frame=int(f), atd_frame_id=int(f) + 1, segment=segment, smoothing_action=action,
                smoothing_left_anchor_frame=a, smoothing_right_anchor_frame=b,
                smoothing_context_frames=json.dumps(item['context'].tolist()),
                smoothing_reference_domain='TACTICAL_LIVE' if item['record']['all_tactical_context'] else 'BROADCAST_OR_MIXED_LIVE',
                source_before=work.d.final_source.iloc[f], source_kind_after=kind,
                before_x=work.ball[f, 0], before_y=work.ball[f, 1], final_x=norm[0], final_y=norm[1],
                displacement_m=float(np.linalg.norm((norm - work.ball[f, :2]) * PITCH)),
                forward_backward_difference_m=item['record']['forward_backward_difference_m'],
                two_sided_fit_residual_m=item['record']['context_residual_m'],
                interpretation='CONDITIONAL_SOURCE_SEAM_TRANSITION' if item['record']['kind'] == 'SOURCE_SEAM' else 'CLOSED_IMPOSSIBLE_EXCURSION'))
    log = pd.DataFrame(changes).sort_values('frame').reset_index(drop=True)
    assert len(log) and log.frame.is_unique
    changed = np.zeros(N, bool); changed[log.frame.to_numpy(int)] = True
    assert not (set(log.frame) & protected)
    assert np.array_equal(after[list(protected), :2], work.ball[list(protected), :2], equal_nan=True)
    assert np.array_equal(np.isfinite(after[:N, :2]).all(axis=1), work.valid)
    d = work.d.copy(deep=True)
    for field in ['final_x', 'final_y', 'final_source', 'is_prediction', 'final_actual_observation',
                  'tactical_used', 'image_x', 'image_y', 'image_coordinate_kind', 'candidate_index', 'temporal_action']:
        d['unsmoothed_' + field] = work.d[field]
    d['smoothing_action'] = 'UNCHANGED'
    d['smoothing_left_anchor_frame'] = np.nan; d['smoothing_right_anchor_frame'] = np.nan
    d['smoothing_context_frames'] = ''; d['smoothing_reference_domain'] = ''
    d['world_live_reference'] = work.chron
    for r in log.itertuples():
        f = r.frame; is_tac = r.source_kind_after == 'TACTICAL'; is_b = r.source_kind_after == 'BROADCAST'; estimate = not (is_tac or is_b)
        for col, value in [('final_x', after[f, 0]), ('final_y', after[f, 1]),
            ('pitch_x_normalized', after[f, 0]), ('pitch_y_normalized', after[f, 1]),
            ('final_x_m', after[f, 0] * 105), ('final_y_m', after[f, 1] * 68),
            ('pitch_x_m', after[f, 0] * 105), ('pitch_y_m', after[f, 1] * 68)]: d.at[f, col] = value
        d.at[f, 'smoothing_action'] = r.smoothing_action
        d.at[f, 'temporal_action'] = r.smoothing_action
        for field in ['smoothing_left_anchor_frame', 'smoothing_right_anchor_frame', 'smoothing_context_frames', 'smoothing_reference_domain']:
            d.at[f, field] = getattr(r, field)
        d.at[f, 'final_source'] = 'TACTICAL_ATD_TEMPORAL_RESELECTED' if is_tac else ('BROADCAST_OBSERVATION_TEMPORAL_RESELECTED' if is_b else
            ('BIDIRECTIONAL_SOURCE_SEAM_ESTIMATE' if r.smoothing_action == 'BIDIRECTIONAL_SEAM_SMOOTHING' else 'BIDIRECTIONAL_SMOOTHED_WORLD_ESTIMATE'))
        for col in ['is_prediction', 'source_is_prediction', 'final_source_is_prediction', 'new_temporal_estimate']: d.at[f, col] = estimate
        for col in ['final_actual_observation', 'ball_observed', 'ball_detected']: d.at[f, col] = is_b
        d.at[f, 'actual_recovered_detection'] = is_b
        d.at[f, 'tactical_used'] = is_tac
        d.at[f, 'world_live_reference'] = True
        d.at[f, 'left_anchor_frame'] = r.smoothing_left_anchor_frame if estimate else np.nan
        d.at[f, 'right_anchor_frame'] = r.smoothing_right_anchor_frame if estimate else np.nan
        d.at[f, 'candidate_index'] = work.d.target_candidate_index.iloc[f] if is_b else np.nan
        for col in ['detection_origin', 'source_detection_origin']:
            d.at[f, col] = work.d.target_detection_origin.iloc[f] if is_b else ''
        d.loc[f, ['image_x', 'image_y']] = np.nan
        d.at[f, 'image_coordinate_kind'] = 'TACTICAL_RECORDED_GROUND_REFERENCE' if is_tac else 'TWO_SIDED_SMOOTHED_GROUND_ESTIMATE'
        if is_b:
            d.loc[f, ['image_x', 'image_y']] = work.d.loc[f, ['target_image_x', 'target_image_y']].to_numpy()
            d.at[f, 'image_coordinate_kind'] = 'ACTUAL_REUSED_BROADCAST_DETECTION'
        elif work.hgood[f] and not work.hl[f]:
            q = work.hm[f] @ np.r_[after[f, :2], 1.]
            if np.isfinite(q).all() and abs(q[2]) > 1e-10: d.loc[f, ['image_x', 'image_y']] = q[:2] / q[2]
    assert d.loc[~changed, work.d.columns].equals(work.d.loc[~changed])
    # Original scalar speed(s) remains a recorded channel; XY-derived speed is separate.
    d['smoothed_position_speed_mps'] = np.nan
    for item in selected:
        path = after[item['a']:item['b'] + 1, :2] * PITCH
        for j, f in enumerate(item['fs'], 1): d.at[f, 'smoothed_position_speed_mps'] = np.linalg.norm(path[j + 1] - path[j - 1]) * 12.5
    d.to_csv(OUT / 'ball_homography_all_frames.csv', index=False)
    log.to_csv(OUT / 'correction_log.csv', index=False)
    pd.DataFrame(work.decisions).to_csv(OUT / 'segment_decisions.csv', index=False)
    pd.DataFrame(work.decisions).query('decision != "ACCEPTED"').to_csv(OUT / 'rejected_or_unresolved_candidates.csv', index=False)
    with BEFORE.open('rb') as source, AFTER.open('wb') as destination:
        for f, line in enumerate(source):
            if f < N and changed[f]:
                body = line.rstrip(b'\r\n'); prefix, tokens = body.rsplit(b':', 1); values = tokens.split(b',')
                values[:2] = [format(float(v), '.17g').encode('ascii') for v in after[f, :2]]
                new = prefix + b':' + b','.join(values) + line[len(body):]
            else: new = line
            destination.write(new)
    shutil.copyfile(BEFORE.with_suffix('.xml'), AFTER.with_suffix('.xml'))
    assert np.array_equal(ball_text(AFTER), after, equal_nan=True)
    with BEFORE.open('rb') as a, AFTER.open('rb') as b:
        rows = 0
        for f, (aa, bb) in enumerate(itertools.zip_longest(a, b)):
            assert aa is not None and bb is not None
            x, y = aa.rstrip(b'\r\n'), bb.rstrip(b'\r\n')
            assert x.rsplit(b':', 1)[0] == y.rsplit(b':', 1)[0]
            assert x.rsplit(b',', 1)[1] == y.rsplit(b',', 1)[1]
            assert aa[len(x):] == bb[len(y):]
            if f == N or not changed[f]: assert aa == bb
            rows += 1
    assert rows == N + 1
    for item in selected:
        p = after[item['a']:item['b'] + 1, :2] * PITCH
        assert np.linalg.norm(np.diff(p, axis=0), axis=1).max() * 25 <= 40 + 1e-9
        assert work.anchor[item['a']] and work.anchor[item['b']]
    assert all(digest(ROOT / rel) == signature for rel, signature in hashes.items())
    for relative, stat in video_stats.items():
        p = ROOT / relative; assert dict(bytes=p.stat().st_size, mtime_ns=p.stat().st_mtime_ns) == stat
    for name, target in [('merged_v3_masked_validation.json', 'model_validation.json'),
                         ('merged_v3_masked_validation.csv', 'masked_observation_evaluation.csv')]:
        shutil.copyfile((ROOT / 'outputs/ball_tracking_validation') / name, OUT / target)
    before_metrics = metrics(work.xy, work.half, work.chron)
    after_metrics = metrics(after[:N, :2] * PITCH, work.half, work.chron)
    summary = dict(status='COORDINATES_COMPLETE', input_current_merged_CSV=str(TABLE.relative_to(ROOT)),
        coordinate_base_is_latest_merged_V2=True, frames=N, ATD_rows=N + 1,
        corrected_frames=len(log), accepted_segments=len(selected),
        closed_excursion_segments=sum(r['record']['kind'] == 'CLOSED_EXCURSION' for r in selected),
        source_seam_segments=sum(r['record']['kind'] == 'SOURCE_SEAM' for r in selected),
        action_counts=log.smoothing_action.value_counts().to_dict(),
        anchor_frames=int(work.anchor.sum()), all_anchor_and_support_frames_immutable=True,
        no_prediction_used_as_anchor=True, no_missing_frame_filled=True,
        valid_coverage_unchanged=int(work.valid.sum()), before_temporal_metrics=before_metrics, after_temporal_metrics=after_metrics,
        source_aware_video_checks=True, actual_source_video_frames_checked=sum(r['record']['actual_checked_video_frames'] for r in selected),
        per_frame_max_model_speed_mps=40, max_closed_span_frames=25, max_seam_correction_m=4,
        seam_results_are_bounded_conditional_transition_estimates=True, physical_accuracy_improvement_claimed=False,
        motion_model='Two-ended endpoint-velocity-preserving cubic Hermite; validation selected over linear',
        both_real_broadcast_observations_and_recorded_tactical_refs_have_distinct_provenance=True,
        speed_s_channel_preserved_not_recomputed=True, original_and_previous_outputs_modified=False,
        source_hashes=hashes, source_video_stat=video_stats)
    save('smoothing_summary.json', summary)
    save('test_results.json', dict(passed=True, all_unmodified_rows_byte_identical=True,
        all_unmodified_CSV_fields_equal=True, actual_ATD_matches_CSV=True,
        player_frame_speed_and_newlines_preserved=True, all_endpoints_and_support_coordinates_unchanged=True,
        all_new_estimates_strictly_between_recorded_nonpredicted_endpoints=True,
        no_half_boundary_or_uncertain_broadcast_replay_bridged=True,
        source_video_scene_checks_passed=True, all_modified_paths_under40mps=True,
        no_added_or_lost_coordinate_frames=True, no_one_sided_extrapolation=True, source_hashes_unchanged=True,
        extra_ATD_row_preserved=True, new_estimates_not_mislabelled_as_actual_observations=True,
        original_observation_records_archived=True, Tactical_references_not_relabelled_as_YOLO=True))
    (OUT / 'README.md').write_text(f'''# 통합 V2 순간이동 좌표의 양방향 평활화

기준은 `{TABLE.relative_to(ROOT)}`입니다. 원래 결측 마스크 CSV나 Tactical 통합 전 데이터로 되돌리지 않았습니다.
현재 좌표가 있는 프레임 중 확인 가능한 짧은 왕복 이탈과 출처 전환 경계만 수정했습니다.
총 {len(log):,}프레임 / {len(selected):,}구간이며 위치 유효성 {int(work.valid.sum()):,}프레임은 그대로입니다.

실제 Broadcast 이미지 관측과 실제로 기록된 Tactical ATD 참조를 구분해 고정 앵커로 사용했습니다.
기존 AI/보간 좌표를 앵커로 사용하지 않았고, 양끝 및 양쪽 지원 관측은 수정하지 않았습니다.
양끝이 없는 구간·긴 불확실 구간·전후반 경계는 연결하지 않았으며 결측을 새로 채우지 않았습니다.
측정된 기존 Tactical/Broadcast 좌표가 양방향 궤적1m 이내이고 경로 속도 조건을 통과하면 합성보다 그 기록을 우선했습니다.
나머지는 양끝 위치와 주변 속도를 보존하는 Hermite 궤적으로 추정했습니다.

왕복 이탈은1~25프레임, 내부 전체 이탈1.5m 이상/최대2.5m 이상, 앞뒤 예측 차이2m 이내입니다.
단일 출처 경계는 양쪽3개 추가 관측과 안정적인 양측 궤적이 있는 경우에만 최대4m 범위에서 완화했습니다.
출처 경계 평활화는 조건부 전환 추정이며 어느 출처가 실제 위치 정답이라는 판정이 아닙니다.
수정 경로는40m/s 이하로 제한했습니다. 이 값은 이 작업의 보수적인 설정이며 보편적인 축구공 최고속도 주장이 아닙니다.

Tactical 앵커/지원만 쓰는 구간에서는 Tactical 원본 영상으로 컷을 검사했습니다. 관계없는 Broadcast 컷 때문에 실시간 Tactical 경기장 궤적을 막지 않았습니다.
Broadcast가 포함된 앵커는 Broadcast 영상과 H 신뢰도를 확인했고 불확실한 리플레이 시간 구간은 연결하지 않았습니다.
중간 H가 나쁜 프레임도 양끝 근거가 충분하면 경기장 좌표만 추정할 수 있습니다. 영상 직접 검출로 표시하지 않습니다.

[새 ATD](BAR-ATH_broadcast_atd.txt) · [전체 CSV](ball_homography_all_frames.csv) · [보정 기록](correction_log.csv) · [구간 판정](segment_decisions.csv)

이전값은 `unsmoothed_*`와 원래 V2 파일에 보존했습니다. 실제 검출/ATD 참조 재선택과 새 추정을 구분했습니다.
기존 선수·공 속도(s,kph)·프레임·줄끝·원본 H를 유지했습니다. 좌표에서 계산한 속도는 별도 `smoothed_position_speed_mps`이며 s를 덮어쓰지 않습니다.
관측을 인위적으로 가린 검증은 Hermite가 선형보다 낮은 대리 복원 오차를 보였지만 실제 공 위치 정답 정확도 검증이 아닙니다.
모든 순간이동이 완전히 제거되었다거나 실제 위치 정확도가 향상되었다고 주장하지 않습니다. 근거가 부족한 후보는 보류 기록에 남겼습니다.
''', encoding='utf-8')


if __name__ == '__main__': main()
