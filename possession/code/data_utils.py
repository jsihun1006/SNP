import numpy as np

def runs_of(mask):
    """[(start, stop)] half-open runs where a boolean array is True."""
    edges = np.diff(np.concatenate(([0], np.asarray(mask, dtype=np.int8), [0])))
    return list(zip(np.flatnonzero(edges == 1).tolist(), np.flatnonzero(edges == -1).tolist()))

def contiguous_runs(mask, frame):
    cuts = np.r_[0, np.flatnonzero(np.diff(frame) != 1) + 1, len(mask)]
    return [(int(first + a), int(first + b))
            for first, stop in zip(cuts[:-1], cuts[1:])
            for a, b in runs_of(mask[first:stop])]
