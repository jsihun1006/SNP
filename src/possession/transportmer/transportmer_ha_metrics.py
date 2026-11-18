"""In-play smoothing, official spell metrics, and frame-aligned error intervals."""

import numpy as np
import pandas as pd

from evaluation import possession_metrics as scores
from starter.baseline import smooth
from transportmer_ha_data import FPS, contiguous_runs

METHODS = ("raw", "majority50", "prob15", "prob25", "prob35")


def postprocess(half, probability, method):
    if method not in METHODS:
        raise ValueError(f"Unknown smoothing method {method}")
    pred = np.full(len(half.frame), scores.D, dtype=np.int8)
    for first, stop in half.episodes:
        values = probability[first:stop]
        if not np.isfinite(values).all():
            raise ValueError(f"In-play frame without probability in {half.key}")
        hard = (values >= 0.5).astype(np.int8)
        if method == "majority50":
            pred[first:stop] = (smooth(hard, 50) == "A").astype(np.int8)
        elif method.startswith("prob"):
            radius = int(method[4:]) // 2
            index = np.arange(len(values))
            begin = np.maximum(0, index-radius)
            end = np.minimum(index+radius+1, len(values))
            cumulative = np.r_[0., np.cumsum(values, dtype=np.float64)]
            pred[first:stop] = (cumulative[end] - cumulative[begin]) >= (end-begin) * 0.5
        else:
            pred[first:stop] = hard
    return pred


def paired_transitions(half, pred):
    runs = contiguous_runs(half.alive & (half.labels >= 0), half.frame)
    pairs, missed, spurious = [], [], []
    for run in runs:
        gt = scores.transitions(half.labels, [run])
        observed = scores.transitions(pred, [run])
        paired, absent, extra = scores.match_transitions(gt, observed, 25, typed="pair")
        pairs.extend(paired)
        missed.extend(absent)
        spurious.extend(extra)
    return pairs, missed, spurious


def metrics(halves, predictions):
    ground, predicted, runs = [], [], []
    latency = []
    missed = spurious = offset = 0
    for half, pred in zip(halves, predictions, strict=True):
        ground.append(half.labels)
        predicted.append(pred)
        runs.extend((offset+a, offset+b) for a, b in contiguous_runs(half.labels >= 0, half.frame))
        offset += len(pred)
        pairs, absent, extra = paired_transitions(half, pred)
        latency.extend(40 * abs(left[0]-right[0]) for left, right in pairs)
        missed += len(absent)
        spurious += len(extra)
    ground, predicted = np.concatenate(ground), np.concatenate(predicted)
    actual = scores.possession_spells(ground, runs)
    estimated = scores.possession_spells(predicted, runs)
    overlaps = scores.spell_overlaps(actual, estimated)
    spell = [scores.match_spells(overlaps, len(actual), len(estimated), iou)
             for iou in scores.SPELL_IOUS]
    short = np.array([end-start < 2*FPS for start, end, _ in actual], dtype=bool)
    matched = scores.matched_gt_flags(overlaps, len(actual), scores.SPELL_IOU_HEADLINE)
    return {"spellF1": float(np.mean([2*tp/max(1, 2*tp+fp+fn) for tp, fp, fn in spell])),
            "spurious_transitions": spurious, "missed_transitions": missed,
            "matched_transitions": len(latency), "transition_mae_ms": float(np.mean(latency)) if latency else None,
            "transition_tolerance_frames": 25, "short_spells": int(short.sum()),
            "short_matched": int((short & matched).sum()),
            "short_recall_iou_0_5": float(matched[short].mean()) if short.any() else None}


def mismatches(half, raw, final):
    valid = half.alive & (half.labels >= 0)
    wrong_raw = valid & (half.labels != raw)
    wrong_final = valid & (half.labels != final)
    pairs, missed, spurious = paired_transitions(half, final)
    missing_ball = ~np.isfinite(half.xy[:, -1]).all(axis=1)
    for stage, wrong in (("persistent", wrong_raw & wrong_final),
                         ("resolved", wrong_raw & ~wrong_final),
                         ("introduced", ~wrong_raw & wrong_final)):
        use_pred = raw if stage == "resolved" else final
        pair_set, miss_set, extra_set = paired_transitions(half, raw) if stage == "resolved" else (pairs, missed, spurious)
        for first, stop in contiguous_runs(wrong, half.frame):
            episode_first, episode_stop = next((a, b) for a, b in half.episodes if a <= first < b)
            def touches_error(events):
                return any(episode_first <= event[0] < episode_stop and first-1 <= event[0] <= stop
                           for event in events)
            tags = []
            if touches_error(miss_set):
                tags.append("missed_transition")
            if touches_error(extra_set):
                tags.append("spurious_transition")
            if any(left[0] != right[0] and min(left[0], right[0]) < stop
                   and max(left[0], right[0]) > first for left, right in pair_set
                   if episode_first <= left[0] < episode_stop):
                tags.append("timing_error")
            context_first = max(episode_first, first-FPS)
            context_stop = min(episode_stop, stop+FPS)
            contexts = []
            if missing_ball[context_first:context_stop].any():
                contexts.append("ball_missing")
            if episode_first > 0 and not half.alive[episode_first-1] and first-episode_first < FPS:
                contexts.append("after_restart")
            if not missing_ball[context_first:context_stop].all():
                xy = half.xy[context_first:context_stop]
                distance = np.linalg.norm((xy[:, :-1] - xy[:, -1:]) * np.asarray(half.pitch), axis=-1)
                near = distance <= 2.0
                if (near[:, half.kind[:-1] == 0].any(axis=1) & near[:, half.kind[:-1] == 1].any(axis=1)).any():
                    contexts.append("both_teams_within_2m_of_ball")
            yield {
                "match": half.match, "half": half.number, "stage": stage,
                "error_type": tags[0] if tags else "unclassified", "error_tags": "|".join(tags),
                "context": "|".join(contexts) or "undetermined",
                "start_row": first, "stop_row_exclusive": stop,
                "start_frame": int(half.frame[first]), "end_frame_inclusive": int(half.frame[stop-1]),
                "context_start_frame": int(half.frame[context_first]), "context_end_frame": int(half.frame[context_stop-1]),
                "truth_at_start": scores.LABELS[half.labels[first]],
                "raw_at_start": scores.LABELS[raw[first]],
                "smoothed_at_start": scores.LABELS[final[first]],
                "prediction_at_start": scores.LABELS[use_pred[first]],
            }


def frame_table(half, probability, predictions):
    result = pd.DataFrame({"match": half.match, "half": half.number, "frame": half.frame,
                           "seconds": half.time, "ground_truth": np.array(scores.LABELS)[np.where(half.labels < 0, scores.X, half.labels)],
                           "p_away": probability})
    for method, labels in predictions.items():
        result[method] = np.array(scores.LABELS)[labels]
    return result
