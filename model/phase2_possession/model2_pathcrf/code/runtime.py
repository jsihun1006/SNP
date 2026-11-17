from dataclasses import dataclass
import numpy as np
import torch
from inputs import N_FEATURES
from data_utils import contiguous_runs

@dataclass(frozen=True, order=True)
class Window:
    half: int
    start: int
    stop: int


def inference_windows(halves, masks, config):
    width, step = config["refine_window_frames"], config["infer_stride_frames"]
    refs, patches = set(), []
    for index, (half, mask) in enumerate(zip(halves, masks, strict=True)):
        for begin, end in contiguous_runs(half.coarse["base"] != 2, half.frame):
            for local_a, local_b in contiguous_runs(mask[begin:end], half.frame[begin:end]):
                a, b = local_a + begin, local_b + begin
                limit = max(begin, end - width)
                patches.append((index, a, b, begin, end))
                target_end = min(end, b + 1)
                start = max(begin, min((a + target_end - width) // 2, limit))
                if target_end - a > width:
                    start = max(begin, min(a - width // 2, limit))
                refs.add(Window(index, start, min(end, start + width)))
                while start + width < target_end:
                    start = min(start + step, limit)
                    refs.add(Window(index, start, min(end, start + width)))
    return sorted(refs), patches


def collate(halves, refs, config, rng=None):
    spans, members = [], []
    for win in refs:
        half = halves[win.half]
        values = half.features[win.start:win.stop]
        seen = values[..., 6].any(axis=0)
        spans.append(values)
        members.append([np.flatnonzero(seen & (half.team == code)) for code in (0, 1)])
    side = max(1, max(len(s) for sides in members for s in sides))
    length = max(map(len, spans))
    values = np.zeros((len(refs), length, 2 * side + 1, N_FEATURES), dtype=np.float32)
    seen = np.zeros(values.shape[:-1], dtype=bool)
    labels = np.full((len(refs), length), -1, dtype=np.int64)
    trans = np.full_like(labels, -1)
    for i, (win, span, teams) in enumerate(zip(refs, spans, members, strict=True)):
        last = len(span)
        for code, player_ids in enumerate(teams):
            values[i, :last, code * side:code * side + len(player_ids)] = span[:, player_ids]
        values[i, :last, -1] = span[:, -1]
        seen[i, :last] = values[i, :last, :, 6] > 0
        seen[i, :last, -1] = True  # retain the missing-ball indicator
        if rng is not None:
            if config["pitch_flip"]:
                for axis, size in enumerate((config["pitch_x"], config["pitch_y"])):
                    if rng.random() < .5:
                        present = values[i, :last, :, 6] > 0
                        values[i, :last, :, axis][present] = size - values[i, :last, :, axis][present]
                        values[i, :last, :, axis + 2] *= -1
            label = halves[win.half].labels[win.start:win.stop].astype(np.int64)
            label[~np.isin(label, (0, 1))] = -1
            labels[i, :last] = label
            left, right = label[:-1], label[1:]
            known = (left >= 0) & (right >= 0)
            trans[i, 1:last][known] = np.where(left[known] == right[known], 0, 2 - right[known])
    return tuple(torch.from_numpy(x) for x in (values, seen, labels, trans))
