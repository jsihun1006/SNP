import time
import warnings
import numpy as np

from ..types import Detection
from ..tiling import adaptive_tiles, regular_grid


def iou(a, b):
    ix = max(0., min(a[2], b[2]) - max(a[0], b[0]))
    iy = max(0., min(a[3], b[3]) - max(a[1], b[1]))
    intersection = ix * iy
    union = (a[2]-a[0])*(a[3]-a[1]) + (b[2]-b[0])*(b[3]-b[1]) - intersection
    return intersection / union if union > 0 else 0.


def nms(detections, threshold=.5):
    if not 0 <= threshold <= 1:
        raise ValueError("NMS threshold must be in [0,1]")
    pending = sorted(detections, key=lambda d: d.confidence, reverse=True)
    keep = []
    while pending:
        best = pending.pop(0)
        keep.append(best)
        pending = [d for d in pending if d.class_id != best.class_id or iou(d.bbox, best.bbox) <= threshold]
    return keep


class DetectionEngine:
    def __init__(self, detector, mode="baat", size=1280, overlap=.2,
                 confidence=.25, coarse_confidence=.001, nms_iou=.5):
        if mode not in {"full", "sahi", "baat"}:
            raise ValueError("mode must be full, sahi, or baat")
        self.detector, self.mode = detector, mode
        self.size, self.overlap = size, overlap
        self.confidence, self.coarse_confidence, self.nms_iou = confidence, coarse_confidence, nms_iou
        self.last_tiles, self.last_coarse, self.last_timing = [], [], {}

    def predict(self, image):
        import cv2
        if image is None or image.ndim != 3 or min(image.shape[:2]) < 1:
            raise ValueError("Expected nonempty HWC image")
        started = time.perf_counter()
        h, w = image.shape[:2]
        self.last_coarse, self.last_tiles = [], []
        if self.mode == "full":
            result = nms(self.detector.predict(image, self.confidence), self.nms_iou)
        else:
            if self.mode == "baat":
                coarse_image = cv2.resize(image, (self.size, self.size))
                self.last_coarse = [Detection((d.bbox[0]*w/self.size, d.bbox[1]*h/self.size,
                                               d.bbox[2]*w/self.size, d.bbox[3]*h/self.size),
                                              d.confidence, d.class_id)
                                    for d in self.detector.predict(coarse_image, self.coarse_confidence)]
                self.last_tiles = adaptive_tiles(w, h, [d.bbox for d in self.last_coarse], self.size, self.overlap)
                if any(not t.converged for t in self.last_tiles):
                    warnings.warn("BAAT reached its iteration limit; containment is incomplete")
                if any((t.x2-t.x1)*(t.y2-t.y1) > 4*self.size**2 for t in self.last_tiles):
                    warnings.warn("BAAT expanded beyond 4x nominal area; detector will resize this crop")
            else:
                self.last_tiles = regular_grid(w, h, self.size, self.overlap)
            all_detections = []
            for t in self.last_tiles:
                crop = image[t.y1:t.y2, t.x1:t.x2]
                for d in self.detector.predict(crop, self.confidence):
                    x1, y1, x2, y2 = d.bbox
                    box = (max(0, x1+t.x1), max(0, y1+t.y1), min(w, x2+t.x1), min(h, y2+t.y1))
                    if box[2] > box[0] and box[3] > box[1]:
                        all_detections.append(Detection(box, d.confidence, d.class_id))
            result = nms(all_detections, self.nms_iou)
        self.last_timing = {"detection_seconds": time.perf_counter()-started,
                            "tiles": len(self.last_tiles), "coarse_detections": len(self.last_coarse)}
        return result
