"""Scoring, stress tests, and full-match structure.

All ranking metrics come from the challenge's own ``evaluation/possession_metrics.py``.  The lab
does not reimplement the scorer, so a number printed here means the same thing as a number printed
by the official evaluation script.
"""

from __future__ import annotations

import sys
from typing import Any

import numpy as np

from .config import (
    A,
    D,
    GRADED_HALVES,
    H,
    N_FRAMES,
    PUBLIC_GT_PATH,
    PUBLIC_START,
    PUBLIC_STOP,
    ROOT,
    X,
)
from .data import TagTable, TrackingData, load_public_gt
from .features import Features, extract_features
from .model import DEFAULT_PARAMS, ModelParams, decode

# The official scorer is a namespace module in evaluation/, not an installed package.
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))
from evaluation import possession_metrics as pm  # noqa: E402


def evaluate_codes(gt: np.ndarray, pred: np.ndarray, bootstrap_n: int = 0) -> dict[str, Any]:
    """Run the official scorer.  ``bootstrap_n=0`` disables the slow bootstrap intervals."""
    gt = np.asarray(gt, dtype=np.int8)
    pred = np.asarray(pred, dtype=np.int8)
    if gt.shape != pred.shape:
        raise ValueError(f"gt and pred lengths differ: {gt.shape} vs {pred.shape}")
    return pm.evaluate(gt, pred, bootstrap_n=bootstrap_n)


def public_metrics(
    pred: np.ndarray,
    gt: np.ndarray | None = None,
    bootstrap_n: int = 2000,
) -> dict[str, Any]:
    """Score the public window.

    The lab/ATD arrays are full-match length (199,319) while the official public window is
    ``[11,185, 18,685)``.  Slice both sides before scoring so the confusion matrix has the same
    shape and semantics as the reference metrics; frames outside the window are X, not evidence.
    """
    if gt is None:
        gt = load_public_gt(PUBLIC_GT_PATH)
    pred = np.asarray(pred, dtype=np.int8)
    gt = np.asarray(gt, dtype=np.int8)
    if pred.size == N_FRAMES:
        pred = pred[PUBLIC_START:PUBLIC_STOP]
    if gt.size == N_FRAMES:
        gt = gt[PUBLIC_START:PUBLIC_STOP]
    return evaluate_codes(gt, pred, bootstrap_n=bootstrap_n)


def to_jsonable(value: Any) -> Any:
    """Convert numpy scalars/arrays into JSON-safe values."""
    if isinstance(value, dict):
        return {str(key): to_jsonable(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [to_jsonable(item) for item in value]
    if isinstance(value, np.ndarray):
        return value.tolist()
    if isinstance(value, (np.integer,)):
        return int(value)
    if isinstance(value, (np.floating,)):
        return float(value)
    if isinstance(value, (np.bool_,)):
        return bool(value)
    return value


def smooth_baseline(codes: np.ndarray, window: int) -> np.ndarray:
    """Centred H/A vote with D wherever neither team appears in the window.

    This is the honest simple baseline used in the report: it has no gap model and no transition
    prior, so it shows what the HMM actually adds.
    """
    codes = np.asarray(codes, dtype=np.int8)
    n = len(codes)
    if n == 0:
        return codes.copy()
    h_cum = np.r_[0, np.cumsum(codes == H)]
    a_cum = np.r_[0, np.cumsum(codes == A)]
    index = np.arange(n)
    lo = np.maximum(0, index - window)
    hi = np.minimum(n, index + window + 1)
    home = h_cum[hi] - h_cum[lo]
    away = a_cum[hi] - a_cum[lo]
    out = np.full(n, D, dtype=np.int8)
    seen = (home + away) > 0
    out[seen] = np.where(home[seen] >= away[seen], H, A)
    return out


def stress_masks(
    n_frames: int = N_FRAMES,
    public: tuple[int, int] = (PUBLIC_START, PUBLIC_STOP),
    seed: int = 1,
) -> dict[str, np.ndarray]:
    """Three reproducible missing-ball scenarios inside the public window.

    * clean: nothing removed
    * random: about 40.7% of the public window removed in short runs
    * long: two long holes totalling about 40.7% of the window

    The real BR-ATH masks use the exact positions from the reference experiment.  The generic
    branches keep the function testable on tiny synthetic arrays.
    """
    start, stop = int(public[0]), int(public[1])
    if not (0 <= start < stop <= n_frames):
        raise ValueError(f"invalid public window [{start}, {stop}) for {n_frames} frames")
    clean = np.ones(n_frames, dtype=bool)

    random_mask = np.ones(n_frames, dtype=bool)
    rng = np.random.default_rng(seed)
    target = int(0.407 * (stop - start))
    filled = 0
    attempts = 0
    while filled < target and attempts < 100_000:
        attempts += 1
        if stop - start >= 160:
            length = int(rng.integers(20, 160))
        else:
            length = int(rng.integers(1, max(2, (stop - start) // 3)))
        length = min(length, stop - start)
        if length <= 0:
            break
        begin_high = stop - 1 - length
        begin = start if begin_high <= start else int(rng.integers(start, begin_high))
        if random_mask[begin : begin + length].all():
            random_mask[begin : begin + length] = False
            filled += length
    if filled < target:
        # Fill the remainder at the first available unobserved frame.
        for index in range(start, stop):
            if filled >= target:
                break
            if random_mask[index]:
                random_mask[index] = False
                filled += 1

    long_mask = np.ones(n_frames, dtype=bool)
    if n_frames == N_FRAMES and (start, stop) == (PUBLIC_START, PUBLIC_STOP):
        long_mask[12000:13800] = False
        long_mask[15000:16253] = False
    else:
        window = stop - start
        first_length = max(1, int(round(0.24 * window)))
        second_length = max(1, int(round(0.17 * window)))
        first_start = start + int(round(0.109 * window))
        second_start = start + int(round(0.509 * window))
        long_mask[first_start : min(stop, first_start + first_length)] = False
        long_mask[second_start : min(stop, second_start + second_length)] = False

    return {"clean": clean, "random": random_mask, "long": long_mask}


def evaluate_stress(
    tracking: TrackingData,
    tags: TagTable,
    params: ModelParams = DEFAULT_PARAMS,
    masks: dict[str, np.ndarray] | None = None,
    context: int = 500,
) -> list[dict[str, Any]]:
    """Run the clean/random/long missing-ball suite and return one row per method."""
    if masks is None:
        masks = stress_masks(tracking.n_frames)
    lo = max(0, PUBLIC_START - context)
    hi = min(tracking.n_frames, PUBLIC_STOP + context)
    rows: list[dict[str, Any]] = []
    for name, observed in masks.items():
        features = extract_features(tracking, tags, observed_mask=observed, start=lo, stop=hi)
        base = np.where(features.present, features.near_team, D).astype(np.int8)
        for window in (50, 100):
            smoothed = smooth_baseline(base, window)
            pred = np.full(tracking.n_frames, X, dtype=np.int8)
            pred[PUBLIC_START:PUBLIC_STOP] = smoothed[PUBLIC_START - lo : PUBLIC_STOP - lo]
            result = evaluate_codes(load_public_gt(PUBLIC_GT_PATH), pred, bootstrap_n=0)
            rows.append(
                {
                    "mask": name,
                    "method": f"baseline_w{window}",
                    "spellF1": float(result["spell_f1_mean"]),
                    "macroF1": float(result["macro_f1"]),
                    "n_pred": int(result["n_pred_spells"]),
                    "Df1": float(result["per_class"]["D"]["f1"]),
                }
            )
        for mode, label in (("persist", "HMM persistence"), ("hybrid", "HMM gap-aware")):
            decode_params = ModelParams.from_mapping(
                {**params.__dict__, "miss_mode": mode, "gap_thresh": params.gap_thresh}
            )
            result = decode(features, decode_params, with_confidence=False)
            pred = np.full(tracking.n_frames, X, dtype=np.int8)
            pred[PUBLIC_START:PUBLIC_STOP] = result.codes[PUBLIC_START - lo : PUBLIC_STOP - lo]
            scored = evaluate_codes(load_public_gt(PUBLIC_GT_PATH), pred, bootstrap_n=0)
            rows.append(
                {
                    "mask": name,
                    "method": label,
                    "spellF1": float(scored["spell_f1_mean"]),
                    "macroF1": float(scored["macro_f1"]),
                    "n_pred": int(scored["n_pred_spells"]),
                    "Df1": float(scored["per_class"]["D"]["f1"]),
                }
            )
    return rows


def full_match_stats(
    pred: np.ndarray,
    halves: tuple[tuple[int, int], ...] = GRADED_HALVES,
) -> dict[str, Any]:
    """Possession-spell structure over the two graded halves."""
    pred = np.asarray(pred, dtype=np.int8)
    n = len(pred)
    graded = np.zeros(n, dtype=bool)
    for start, stop in halves:
        graded[start:stop] = True

    dummy = np.full(n, X, dtype=np.int8)
    for start, stop in halves:
        dummy[start:stop] = H
    runs = pm.valid_runs(dummy)
    spells = pm.possession_spells(pred, runs)
    durations = np.array([(stop - start) / 25.0 for start, stop, _ in spells], dtype=np.float64)
    shares = {label: float(np.mean(pred[graded] == code)) for label, code in (("H", H), ("A", A), ("D", D))}
    per_half = []
    for start, stop in halves:
        half_spells = [spell for spell in spells if start <= spell[0] < stop]
        per_half.append(
            {
                "start": int(start),
                "stop": int(stop),
                "n_spells": len(half_spells),
                "n_frames": int(stop - start),
            }
        )
    by_label = {label: int(sum(1 for _, _, code in spells if code == value)) for label, value in (("H", H), ("A", A), ("D", D))}
    return {
        "n_spells": len(spells),
        "n_graded_frames": int(graded.sum()),
        "mean_duration_s": float(durations.mean()) if durations.size else 0.0,
        "median_duration_s": float(np.median(durations)) if durations.size else 0.0,
        "min_duration_s": float(durations.min()) if durations.size else 0.0,
        "max_duration_s": float(durations.max()) if durations.size else 0.0,
        "durations_s": durations.tolist(),
        "spell_labels": [str(code) for _, _, code in spells],
        "shares": shares,
        "n_spells_by_label": by_label,
        "per_half": per_half,
    }


def features_for_window(
    tracking: TrackingData,
    tags: TagTable,
    public: tuple[int, int] = (PUBLIC_START, PUBLIC_STOP),
    context: int = 500,
    observed_mask: np.ndarray | None = None,
) -> Features:
    lo = max(0, public[0] - context)
    hi = min(tracking.n_frames, public[1] + context)
    return extract_features(tracking, tags, observed_mask=observed_mask, start=lo, stop=hi)


def public_prediction_window(pred: np.ndarray, public: tuple[int, int] = (PUBLIC_START, PUBLIC_STOP)) -> np.ndarray:
    return np.asarray(pred)[public[0] : public[1]]
