"""BAAT: monotonic expansion, bounded by the source image, no hidden area cap."""
from dataclasses import dataclass
import math
from typing import Iterable


@dataclass(frozen=True)
class Tile:
    x1: int
    y1: int
    x2: int
    y2: int
    iterations: int = 0
    converged: bool = True

    @property
    def bbox(self):
        return self.x1, self.y1, self.x2, self.y2


def _starts(length, size, overlap):
    if length <= size:
        return [0]
    stride = max(1, size - int(size * overlap))
    return sorted(set([*range(0, length - size + 1, stride), length - size]))


def regular_grid(width: int, height: int, size: int = 1280, overlap: float = .2):
    if min(width, height, size) <= 0 or not 0 <= overlap < 1:
        raise ValueError("Positive dimensions and overlap in [0,1) required")
    return [Tile(x, y, min(width, x + size), min(height, y + size))
            for y in _starts(height, size, overlap)
            for x in _starts(width, size, overlap)]


def _intersects(a, b):
    return max(a[0], b[0]) < min(a[2], b[2]) and max(a[1], b[1]) < min(a[3], b[3])


def _contains(a, b):
    return a[0] <= b[0] and a[1] <= b[1] and a[2] >= b[2] and a[3] >= b[3]


def expand_tile(tile: Tile, boxes: Iterable, width: int, height: int,
                margin: int = 5, max_iterations: int = 100):
    if margin < 0 or max_iterations < 1:
        raise ValueError("Invalid expansion limits")
    # A coarse box outside the image can only be contained in its visible portion.
    clipped = []
    for b in boxes:
        if len(b) != 4 or not all(math.isfinite(float(v)) for v in b):
            raise ValueError("Invalid coarse box")
        if b[2] <= b[0] or b[3] <= b[1]:
            raise ValueError("Inverted coarse box")
        c = (max(0, b[0]), max(0, b[1]), min(width, b[2]), min(height, b[3]))
        if c[2] > c[0] and c[3] > c[1]:
            clipped.append(c)
    w = tile.bbox
    for iteration in range(max_iterations):
        partial = [b for b in clipped if _intersects(w, b) and not _contains(w, b)]
        if not partial:
            return Tile(*w, iterations=iteration, converged=True)
        # Include current edges: literal paper equations could shrink the window.
        w = (max(0, math.floor(min(w[0], min(b[0] for b in partial) - margin))),
             max(0, math.floor(min(w[1], min(b[1] for b in partial) - margin))),
             min(width, math.ceil(max(w[2], max(b[2] for b in partial) + margin))),
             min(height, math.ceil(max(w[3], max(b[3] for b in partial) + margin))))
    complete = not any(_intersects(w, b) and not _contains(w, b) for b in clipped)
    return Tile(*w, iterations=max_iterations, converged=complete)


def adaptive_tiles(width, height, boxes, size=1280, overlap=.2, margin=5, max_iterations=100):
    boxes = list(boxes)
    # Preserve grid order and duplicate windows for a controlled SAHI/BAAT comparison.
    return [expand_tile(t, boxes, width, height, margin, max_iterations)
            for t in regular_grid(width, height, size, overlap)]
