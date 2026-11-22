"""Offline reconstruction baselines. Never use a clean hidden target as context."""
import numpy as np


def reconstruct_baselines(corrupted_features, padding_mask):
    x = np.asarray(corrupted_features)
    if x.ndim != 3 or x.shape[-1] != 53 or padding_mask.shape != x.shape[:2]:
        raise ValueError("Baselines require [B,T,53] corrupted inputs")
    b, t, _ = x.shape
    persistence = np.zeros((b, t, 13), dtype=np.float32)
    interpolation = persistence.copy()
    fallback = np.zeros((b, t, 13), dtype=bool)
    times = np.arange(t)
    for i in range(b):
        hidden = x[i, :, 52] > .5
        for k in range(13):
            valid = (x[i, :, 24+k] > .5) & ~hidden & ~padding_mask[i] & np.isfinite(x[i, :, k])
            anchors = np.flatnonzero(valid)
            if not len(anchors):
                # Zero is the frozen TRAIN median in standardized feature space.
                fallback[i, :, k] = True
                continue
            values = x[i, anchors, k]
            previous = np.clip(np.searchsorted(anchors, times, side="right")-1, 0, len(anchors)-1)
            persistence[i, :, k] = values[previous]  # next anchor only if no predecessor
            interpolation[i, :, k] = np.interp(times, anchors, values)
    return dict(persistence=persistence, interpolation=interpolation), fallback
