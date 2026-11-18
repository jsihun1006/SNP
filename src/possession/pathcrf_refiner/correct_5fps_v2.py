"""Frozen 5fps PathCRF: add boundaries, calibrate direction, Viterbi, then touch snap."""
from __future__ import annotations

import argparse
from itertools import product
import json
from pathlib import Path
import pickle
import sys
from time import perf_counter

import numpy as np
import pandas as pd
import torch  # load the available OpenMP runtime before LightGBM
from scipy.signal import find_peaks

from correct_5fps import (TRAIN, VALIDATION, TEST, SEED, Episode, native_features, report, snap,
                          reference_prediction)
from run import ROOT, digest, write_json
from transportmer_ha_data import contiguous_runs
from evaluation import possession_metrics as scores


def read_bundle(cache, match):
    with (cache / 'scores' / f'{match}.pkl').open('rb') as stream:
        halves, episodes = pickle.load(stream)
    reference_prediction(halves, episodes)  # verify saved native-frame baseline identity
    return halves


def feature_map(half):
    observed = native_features(half)
    base = half.coarse['base']
    # Only saved predictions, confidence, and observed trajectories. Never DFL truth/events.
    x = np.c_[observed['x'], (base == 0), (base == 1)]
    for shift in (-25, 25):
        context = np.full((len(base), 4), np.nan, np.float32)
        basic = np.c_[base, half.signals[:, 0], observed['nearest']]
        for a, b in contiguous_runs(base != 2, half.frame):
            if b - a <= abs(shift):
                continue
            if shift < 0:
                context[a - shift:b] = basic[a:b + shift]
            else:
                context[a:b - shift] = basic[a + shift:b]
        x = np.c_[x, context]
    observed['model_x'] = x.astype(np.float32)
    return observed


def fit(training, info, output):
    from lightgbm import LGBMClassifier
    xs, states, boundaries = [], [], []
    for half in training:
        # Regular 5fps samples: no ground-truth-dependent hard-example sampling.
        rows = []
        target = np.full(len(half.frame), -1, np.int8)
        for a, b in contiguous_runs(np.isin(half.labels, (0, 1)), half.frame):
            y = half.labels[a:b]
            changes = np.flatnonzero(np.diff(y)) + a + 1
            target[a:b] = 0
            hit_count = np.zeros(b - a, np.int8)
            for t in changes:
                left, right = max(a, t - 2), min(b, t + 3)
                target[left:right] = 2 - half.labels[t]
                hit_count[left - a:right - a] += 1
            target[a:b][hit_count > 1] = -1
        for a, b in contiguous_runs(half.coarse['base'] != 2, half.frame):
            rows.extend(range(a, b, 5))
        rows = np.asarray(rows)
        valid = np.isin(half.labels[rows], (0, 1)) & (target[rows] >= 0)
        rows = rows[valid]
        xs.append(info[half.key]['model_x'][rows])
        states.append(half.labels[rows])
        boundaries.append(target[rows])
    x, ys, yt = np.concatenate(xs), np.concatenate(states), np.concatenate(boundaries)
    params = dict(n_estimators=180, learning_rate=.04, num_leaves=15, max_depth=4,
                  min_child_samples=40, reg_lambda=2, random_state=SEED, n_jobs=4, verbosity=-1)
    state = LGBMClassifier(**params)
    state.fit(x, ys)
    counts = np.bincount(yt, minlength=3)
    weight = np.sqrt(counts.max() / np.maximum(1, counts)).clip(1, 16)
    direction = LGBMClassifier(**params)
    direction.fit(x, yt, sample_weight=weight[yt])
    state.booster_.save_model(str(output / 'state_calibration.txt'))
    direction.booster_.save_model(str(output / 'direction_probability.txt'))
    write_json(output / 'small_model_training.json', {'samples': len(x), 'state_counts': np.bincount(ys).tolist(),
               'direction_counts': counts.tolist(), 'direction_weights': weight.tolist(), 'parameters': params,
               'boundary_target': 'same-direction ground-truth boundary within 2 native frames, conflicts masked',
               'feature_count': x.shape[1], 'truth_features': False, 'base_model_retrained': False})
    return state, direction, weight


def probabilities(halves, info, state, direction, weight):
    for half in halves:
        scene = info[half.key]
        x = scene['model_x']
        ps = np.clip(state.predict_proba(x), 1e-6, 1)
        pt = direction.predict_proba(x) / weight
        pt /= pt.sum(axis=1, keepdims=True)
        scene['state_log'] = np.log(ps)
        scene['direction_logodds'] = np.log(np.clip(pt[:, 1:], 1e-6, 1)) - np.log(np.clip(pt[:, :1], 1e-6, 1))
        scene['direction_probability'] = pt
        margin = np.clip(half.signals[:, 0], 0, .98)
        confidence = .5 + .49 * margin
        base = half.coarse['base']
        p = np.where(base == 1, confidence, 1 - confidence)
        scene['base_log'] = np.log(np.clip(np.c_[1 - p, p], 1e-6, 1))


def decode_grid(halves, info, settings):
    """Two-state global DP on an observed candidate grid, allowing entry/exit pairs."""
    outputs = [[h.coarse['base'].copy() for h in halves] for _ in settings]
    alpha = np.array([s['state_blend'] for s in settings], np.float32)
    beta = np.array([s['direction_strength'] for s in settings], np.float32)
    cost = np.array([s['switch_cost'] for s in settings], np.float32)
    contest = np.array([s['contest_bonus'] for s in settings], np.float32)
    variants = len(settings)
    variant_rows = np.arange(variants)
    for index, half in enumerate(halves):
        scene = info[half.key]
        base = half.coarse['base']
        for a, b in contiguous_runs(base != 2, half.frame):
            original = np.flatnonzero(np.diff(base[a:b])) + a + 1
            touches = np.flatnonzero(scene['touch'][a:b]) + a
            peaks = []
            for direction in (1, 2):
                peaks.extend(find_peaks(scene['direction_probability'][a:b, direction], distance=3)[0] + a)
            # Every 5fps position is eligible even outside geometric contest candidates.
            edges = np.unique(np.r_[a, np.arange(a + 5, b, 5), original, touches, peaks, b]).astype(int)
            base_prefix = np.vstack([np.zeros(2), np.cumsum(scene['base_log'][a:b], axis=0)])
            state_prefix = np.vstack([np.zeros(2), np.cumsum(scene['state_log'][a:b], axis=0)])
            lo, hi = edges[:-1] - a, edges[1:] - a
            unary0 = (base_prefix[hi] - base_prefix[lo]) / 5
            unary1 = (state_prefix[hi] - state_prefix[lo]) / 5
            # Retain at least 25% of the original model's unary evidence.
            dp = (1 - alpha[:, None]) * unary0[0] + alpha[:, None] * unary1[0]
            back = np.zeros((len(lo), variants, 2), np.int8)
            for step in range(1, len(lo)):
                t = edges[step]
                directional = scene['direction_logodds'][t]
                adjustment = contest * (.5 if scene['contest'][t] else -.5)
                ha = beta * directional[0] - cost + adjustment
                ah = beta * directional[1] - cost + adjustment
                to_h = dp[:, 1] + ah
                to_a = dp[:, 0] + ha
                back[step, :, 0] = to_h > dp[:, 0]
                back[step, :, 1] = np.where(to_a > dp[:, 1], 0, 1)
                next_h = np.maximum(dp[:, 0], to_h)
                next_a = np.maximum(dp[:, 1], to_a)
                unary = (1 - alpha[:, None]) * unary0[step] + alpha[:, None] * unary1[step]
                dp = np.c_[next_h, next_a] + unary
            path = np.empty((len(lo), variants), np.int8)
            path[-1] = dp.argmax(axis=1)
            for step in range(len(lo) - 1, 0, -1):
                path[step - 1] = back[step, variant_rows, path[step]]
            durations = np.diff(edges)
            for k in range(variants):
                outputs[k][index][a:b] = np.repeat(path[:, k], durations)
    return outputs


def quality_key(score, index):
    return score['spellF1'], score['short_recall_iou_0_5'] or 0, -index


def error_summary(half, pred):
    truth = half.labels
    known = truth >= 0
    wrong = known & (truth != pred)
    result = {'wrong_runs': len(contiguous_runs(wrong, half.frame)), 'wrong_frames': int(wrong.sum())}
    actual = scores.possession_spells(truth, contiguous_runs(known, half.frame))
    estimated = scores.possession_spells(pred, contiguous_runs(known, half.frame))
    flags = scores.matched_gt_flags(scores.spell_overlaps(actual, estimated), len(actual), .5)
    result.update(missed_spells=int((~flags).sum()), fully_missed_spells=0, overextended_spells=0,
                  fragmented_spells=0, boundary_missed_spells=0, short_missed=0)
    for k, (a, b, team) in enumerate(actual):
        if flags[k]:
            continue
        result['short_missed'] += int(b - a < 50)
        pieces = [(x, y) for x, y, t in estimated if t == team and x < b and y > a]
        best = max(pieces, key=lambda span: (min(span[1], b) - max(span[0], a)) / (max(span[1], b) - min(span[0], a)), default=None)
        if not (pred[a:b] == team).any():
            result['fully_missed_spells'] += 1
        elif len(pieces) > 1:
            result['fragmented_spells'] += 1
        elif best and best[1] - best[0] > b - a:
            result['overextended_spells'] += 1
        else:
            result['boundary_missed_spells'] += 1
    return result


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--cache', type=Path, default=ROOT / 'data/transition_final/correct_5fps_v1')
    parser.add_argument('--output', type=Path, default=ROOT / 'artifacts/possession/pathcrf/correct_5fps_v2')
    parser.add_argument('--reference-run', type=Path, default=ROOT / 'artifacts/possession/pathcrf/dfl_5train_1val_1test')
    args = parser.parse_args()
    args.output.mkdir(parents=True, exist_ok=True)
    started = perf_counter()
    torch.set_num_threads(4)
    checkpoint = Path('/private/tmp/pathcrf-official-dfl/saved/140/model/state_dict_best_acc.pt')
    sha_before = digest(checkpoint)
    write_json(args.output / 'manifest.json', {'split': {'train':TRAIN,'validation':VALIDATION,'test':TEST},
               'seed':SEED, 'base_model':'fixed 5fps PathCRF', 'checkpoint_sha256':sha_before,
               'runner_sha256':digest(Path(__file__)), 'dead_mask':'unchanged saved predicted D',
               'selection':'validation Spell F1, short recall tie break, simpler setting tie break',
               'holdout_status':'previously inspected, retrospective diagnostic',
               'learned_models':'small state calibration and directional boundary LightGBM, train five matches only',
               'correction_resolution':'native observed trajectories; 5fps grid plus observed peaks',
               'D_correction':False, 'base_neural_retraining':False})
    print('Loading existing frozen score caches for training and validation', flush=True)
    training = [h for match in TRAIN for h in read_bundle(args.cache, match)]
    validation = read_bundle(args.cache, VALIDATION)
    info = {h.key:feature_map(h) for h in training + validation}
    prep_seconds = perf_counter() - started
    print(f'Prepared {len(training)} training halves + {len(validation)} validation halves in {prep_seconds:.1f}s', flush=True)
    tick = perf_counter()
    state, direction, weight = fit(training, info, args.output)
    fit_seconds = perf_counter() - tick
    probabilities(validation, info, state, direction, weight)
    settings = [dict(zip(('state_blend','direction_strength','switch_cost','contest_bonus'), values))
                for values in product((0., .25, .5, .75), (0., .5, 1.), (-2., 0., 2., 4.), (0., 1.))]
    tick = perf_counter()
    print(f'Validation: {len(settings)} predeclared Viterbi configurations', flush=True)
    decoded = decode_grid(validation, info, settings)
    val_metrics = [report(validation,p) for p in decoded]
    best = max(range(len(settings)), key=lambda i:quality_key(val_metrics[i],i))
    chosen = settings[best]
    base = [h.coarse['base'] for h in validation]
    post = []
    # No duration prior, minimum spell length, or automatic pruning of short spells.
    for source_name, source in (('base',base),('candidate_viterbi',decoded[best])):
        for radius in (0,5,10):
            pred = [snap(h,p,info[h.key],radius) for h,p in zip(validation,source,strict=True)]
            post.append({'source':source_name,'snap_radius_frames':radius,'metrics':report(validation,pred)})
    post_best = max(range(len(post)),key=lambda i:quality_key(post[i]['metrics'],i))
    final = post[post_best]
    selected = {'viterbi':chosen,'final':{k:v for k,v in final.items() if k!='metrics'},
                'validation_metrics':final['metrics']}
    write_json(args.output/'selected.json',selected)  # freeze before opening test cache
    write_json(args.output/'validation.json',{'grid':[{'parameters':s,'metrics':m} for s,m in zip(settings,val_metrics,strict=True)],
               'post_grid':post,'selected':selected})
    validation_seconds = perf_counter() - tick
    print(f'Frozen validation selection: {selected}',flush=True)
    del decoded, training

    tick = perf_counter()
    test = read_bundle(args.cache,TEST)
    for h in test:
        saved = pd.read_parquet(args.reference_run/f'estimated_D_primary_{h.key}.parquet')
        np.testing.assert_array_equal(saved.frame,h.frame)
        np.testing.assert_array_equal(saved.truth,h.labels)
        np.testing.assert_array_equal(saved.base,h.coarse['base'])
        info[h.key]=feature_map(h)
    probabilities(test,info,state,direction,weight)
    base=[h.coarse['base'].copy() for h in test]
    candidate=decode_grid(test,info,[chosen])[0]
    snapped=[snap(h,p,info[h.key],final['snap_radius_frames']) for h,p in zip(test,candidate,strict=True)]
    predictions={'base':base,
                 'snap_only':[snap(h,p,info[h.key],10) for h,p in zip(test,base,strict=True)],
                 'candidate_viterbi':candidate,'candidate_viterbi_snap':snapped}
    selected_source=base if final['source']=='base' else candidate
    predictions['validation_selected']=[snap(h,p,info[h.key],final['snap_radius_frames'])
                                        for h,p in zip(test,selected_source,strict=True)]
    rows=[]
    for name,pred in predictions.items():
        totals={}
        for h,p in zip(test,pred,strict=True):
            for key,value in error_summary(h,p).items():totals[key]=totals.get(key,0)+value
        row={'condition':name,**report(test,pred),**totals,'validation_selected':name=='validation_selected'}
        rows.append(row)
    # Preserve the previous six-correction combination as a labelled reference, not a new selected setting.
    old=pd.read_csv(args.cache/'holdout_comparison.csv')
    old_row=old.loc[old.condition.eq('combined')].iloc[0].to_dict()
    old_row.update(condition='previous_combined_reference',validation_selected=False)
    rows.append(old_row)
    pd.DataFrame(rows).to_csv(args.output/'holdout_comparison.csv',index=False,encoding='utf-8-sig')
    changes=[]
    for k,h in enumerate(test):
        data={'match':h.match,'half':h.number,'frame':h.frame,'seconds':h.time,'truth':h.labels,
              'dead':h.coarse['base']==2,'touch_candidate':info[h.key]['touch'],'contest_candidate':info[h.key]['contest']}
        for name,pred in predictions.items():data[name]=pred[k]
        table=pd.DataFrame(data)
        table.to_parquet(args.output/f'{h.key}.parquet',index=False)
        table.to_csv(args.output/f'{h.key}.csv',index=False,encoding='utf-8-sig')
        new=predictions['validation_selected'][k]
        for kind,mask in (
            ('fixed', (h.labels>=0)&(base[k]!=h.labels)&(new==h.labels)),
            ('damaged',(h.labels>=0)&(base[k]==h.labels)&(new!=h.labels))):
            for a,b in contiguous_runs(mask,h.frame):
                changes.append({'kind':kind,'match':h.match,'half':h.number,'start_frame':int(h.frame[a]),
                                'end_frame':int(h.frame[b-1]),'start_seconds':float(h.time[a]),
                                'end_seconds':float(h.time[b-1]),'frames':b-a,'seconds':(b-a)/25})
    pd.DataFrame(changes).to_csv(args.output/'fixed_damaged_intervals.csv',index=False,encoding='utf-8-sig')
    assert digest(checkpoint)==sha_before
    timing={'cache_and_features_seconds':prep_seconds,'small_model_fit_seconds':fit_seconds,
            'validation_selection_seconds':validation_seconds,'holdout_seconds':perf_counter()-tick,
            'total_seconds':perf_counter()-started}
    write_json(args.output/'results.json',{'selected':selected,'metrics':rows,'timing':timing,
               'base_checkpoint_unchanged':True,'frame_axis_and_saved_baseline_exact':True,'D_masks_identical':True})
    print(pd.DataFrame(rows).to_string(index=False),flush=True)
    print(f'Saved: {args.output}; timing={timing}',flush=True)


if __name__=='__main__':main()
