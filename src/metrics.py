"""프레임 지표와 데드볼 구간 지표. 라벨과 예측은 1 데드 / 0 인플레이."""

import os
import sys

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
from config import CONFIG  # noqa: E402
from features import runs_of  # noqa: E402

FPS = CONFIG["fps"]


def auc(label, score):
    """순위 기반 ROC AUC(동점은 평균 순위)."""
    label = np.asarray(label).astype(bool)
    order = np.argsort(score, kind="mergesort")
    ranks = np.empty(len(score))
    sorted_score = np.asarray(score)[order]
    edges = np.flatnonzero(np.r_[True, np.diff(sorted_score) != 0, True])
    for a, b in zip(edges[:-1], edges[1:]):
        ranks[order[a:b]] = (a + b + 1) / 2.0
    pos, neg = label.sum(), (~label).sum()
    if not pos or not neg:
        return np.nan
    return (ranks[label].sum() - pos * (pos + 1) / 2.0) / (pos * neg)


def frame_metrics(label, pred, score=None):
    label, pred = np.asarray(label).astype(bool), np.asarray(pred).astype(bool)
    tp, fp, fn = (label & pred).sum(), (~label & pred).sum(), (label & ~pred).sum()
    precision = tp / max(tp + fp, 1)
    recall = tp / max(tp + fn, 1)
    out = {"frame_acc": (label == pred).mean(), "dead_precision": precision, "dead_recall": recall,
           "dead_f1": 2 * precision * recall / max(precision + recall, 1e-9), "pred_dead_share": pred.mean(),
           "true_dead_share": label.mean()}
    if score is not None:
        out["auc"] = auc(label, score)
    return out


def segment_metrics(spells, pred, period):
    """spells: dfl_labels.dead_spells 결과. pred (n,) 0/1.

    - 정답 구간 재현율: 정답 데드 구간 프레임의 절반 이상을 데드로 예측하면 찾은 것으로 본다.
    - 예측 구간 정밀도: 하프 안에서 min_segment_s 이상 이어진 예측 데드 구간 중 정답 데드와 절반 이상 겹치는 비율.
    - 경계 오차: 찾은 정답 구간마다, 가장 많이 겹치는 예측 구간과의 시작·끝 차이(초, 예측 − 정답).
    """
    pred = np.asarray(pred).astype(bool)
    label = np.zeros(len(pred), dtype=bool)
    for a, b in zip(spells["start"], spells["stop"]):
        label[a:b] = True
    minimum = int(CONFIG["min_segment_s"] * FPS)
    segments = []
    for p in np.unique(period):
        rows = np.flatnonzero(period == p)
        segments += [(rows[a], rows[b - 1] + 1) for a, b in runs_of(pred[rows]) if b - a >= minimum]
    found, start_err, end_err = [], [], []
    seg_start = np.array([s for s, _ in segments])
    seg_stop = np.array([e for _, e in segments])
    for a, b in zip(spells["start"], spells["stop"]):
        hit = pred[a:b].mean() >= 0.5
        found.append(hit)
        if hit and len(segments):
            overlap = np.minimum(seg_stop, b) - np.maximum(seg_start, a)
            k = int(np.argmax(overlap))
            if overlap[k] > 0:
                start_err.append((seg_start[k] - a) / FPS)
                end_err.append((seg_stop[k] - b) / FPS)
    true_segments = [label[s:e].mean() >= 0.5 for s, e in segments]
    found = np.array(found)
    return {"spell_recall": found.mean() if len(found) else np.nan, "spells": len(found),
            "pred_segments": len(segments),
            "segment_precision": np.mean(true_segments) if segments else np.nan,
            "start_err_med": np.median(start_err) if start_err else np.nan,
            "start_abs_med": np.median(np.abs(start_err)) if start_err else np.nan,
            "end_err_med": np.median(end_err) if end_err else np.nan,
            "end_abs_med": np.median(np.abs(end_err)) if end_err else np.nan,
            "found": found}
