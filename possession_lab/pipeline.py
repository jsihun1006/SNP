"""Inference and submission-writing helpers."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from .config import (
    CODE_TO_LABEL,
    CONTEXT,
    GRADED_HALVES,
    N_FRAMES,
    PITCH,
    PUBLIC_GT_PATH,
    PUBLIC_START,
    PUBLIC_STOP,
    X,
    asset_paths,
)
from .data import Homography, TagTable, TrackingData, load_homography, load_public_gt, load_smart_tags, load_tracking
from .evaluate import features_for_window
from .model import DEFAULT_PARAMS, DecodeResult, ModelParams, decode


@dataclass(frozen=True)
class LabData:
    tracking: TrackingData
    tags: TagTable
    homography: Homography
    gt: np.ndarray | None


def _require(paths: dict[str, Path]) -> None:
    missing = [str(path) for path in paths.values() if not path.is_file()]
    if missing:
        joined = "\n  ".join(missing)
        raise FileNotFoundError(
            "the gated match bundle is not available:\n  "
            f"{joined}\n\nUnpack it under data/ or set MTAH_DATA_DIR. "
            "See DATA.md; do not commit the data."
        )


def load_lab(data_dir: str | Path | None = None, load_gt: bool = True) -> LabData:
    """Load the tactical ATD, smart tags, homography, and (optional) public GT."""
    paths = asset_paths(data_dir)
    _require(paths)
    tracking = load_tracking(paths["atd_raw"], paths["atd_meta"], pitch=PITCH, n_frames=N_FRAMES)
    tags = load_smart_tags(paths["smart_tagging"], n_frames=tracking.n_frames)
    homography = load_homography(paths["homography"])
    gt = load_public_gt(PUBLIC_GT_PATH) if load_gt and Path(PUBLIC_GT_PATH).is_file() else None
    return LabData(tracking=tracking, tags=tags, homography=homography, gt=gt)


def predict_full(
    tracking: TrackingData,
    tags: TagTable,
    params: ModelParams = DEFAULT_PARAMS,
    halves: tuple[tuple[int, int], ...] = GRADED_HALVES,
    context: int = CONTEXT,
) -> np.ndarray:
    """Decode each graded half with a context pad and leave everything else as X."""
    n = tracking.n_frames
    prediction = np.full(n, X, dtype=np.int8)
    for start, stop in halves:
        start, stop = int(start), int(stop)
        if not (0 <= start < stop <= n):
            raise ValueError(f"invalid half [{start}, {stop}) for {n} frames")
        lo = max(0, start - context)
        hi = min(n, stop + context)
        features = features_for_window(tracking, tags, public=(lo, hi), context=0)
        result = decode(features, params, with_confidence=False)
        prediction[start:stop] = result.codes[start - lo : stop - lo]
    return prediction


def decode_public(
    tracking: TrackingData,
    tags: TagTable,
    params: ModelParams = DEFAULT_PARAMS,
    public: tuple[int, int] = (PUBLIC_START, PUBLIC_STOP),
    context: int = CONTEXT,
) -> tuple[Any, DecodeResult, int]:
    """Return public-window features, the decode result, and the context left offset."""
    lo = max(0, public[0] - context)
    hi = min(tracking.n_frames, public[1] + context)
    features = features_for_window(tracking, tags, public=public, context=context)
    return features, decode(features, params), lo


def write_submission(pred: np.ndarray, path: str | Path) -> Path:
    """Write the exact ``frame,label`` CSV shape the challenge expects."""
    pred = np.asarray(pred, dtype=np.int8)
    labels = [CODE_TO_LABEL[int(code)] for code in pred]
    frame = np.arange(len(pred), dtype=np.int64)
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    pd.DataFrame({"frame": frame, "label": labels}).to_csv(path, index=False)
    return path


def validate_codes(pred: np.ndarray) -> dict[str, Any]:
    """A local preflight; the official validator remains the final word."""
    pred = np.asarray(pred)
    if pred.ndim != 1:
        raise ValueError("prediction must be one-dimensional")
    valid = {0, 1, 2, 3}
    unique = {int(value) for value in np.unique(pred)}
    unknown = sorted(unique - valid)
    counts = {
        label: int(np.sum(pred == code))
        for label, code in (("H", 0), ("A", 1), ("D", 2), ("X", 3))
    }
    runs = 0
    previous = None
    for value in pred:
        if value != previous:
            runs += 1
            previous = value
    return {
        "ok": not unknown and pred.size > 0,
        "n_rows": int(pred.size),
        "unknown": unknown,
        "counts": counts,
        "n_runs": runs,
    }


def validate_submission_file(path: str | Path) -> dict[str, Any]:
    from .data import load_submission

    codes = load_submission(path)
    report = validate_codes(codes)
    report["path"] = str(path)
    return report


def public_prediction(pred: np.ndarray, public: tuple[int, int] = (PUBLIC_START, PUBLIC_STOP)) -> np.ndarray:
    return np.asarray(pred)[public[0] : public[1]]
