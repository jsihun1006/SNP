"""FP32 YOLO26x ball candidates at full-image, crop, and native-tile scales.

The detector supplies image evidence, not accepted tracks or pitch positions.
All scales belong to ONE detector family. Merging retains a real observation's
box and center; it never averages positions or treats scales as independent
votes. CUDA and an unchanged inference resolution are required.
"""
from __future__ import annotations

import math
import time
from pathlib import Path

import numpy as np


VERSION = "portable_ball_full_roi_tile_batched_v1"
BALL_CLASS_ID = 32


def _axis_starts(length, size, overlap):
    if length <= size:
        return [0]
    step = max(1, int(round(size * (1.0 - overlap))))
    starts = list(range(0, length - size + 1, step))
    if starts[-1] != length - size:
        starts.append(length - size)
    return starts


def tile_windows(width, height, tile_size=640, overlap=.20):
    """Return integer xyxy windows covering every original pixel, including edges."""
    width, height, tile_size = int(width), int(height), int(tile_size)
    if min(width, height, tile_size) < 1 or not 0 <= float(overlap) < 1:
        raise ValueError("Positive dimensions and 0 <= overlap < 1 are required")
    return [(x, y, min(width, x + tile_size), min(height, y + tile_size))
            for y in _axis_starts(height, tile_size, float(overlap))
            for x in _axis_starts(width, tile_size, float(overlap))]


def crop_window(width, height, center, crop_size=640):
    """Place a fixed native-pixel crop around an on-image center; never pad it."""
    width, height, crop_size = int(width), int(height), int(crop_size)
    point = np.asarray(center, dtype=float)
    if point.shape != (2,) or not np.isfinite(point).all():
        raise ValueError("ROI center must contain two finite original-image pixels")
    if min(width, height, crop_size) < 1:
        raise ValueError("Positive crop/image dimensions are required")
    if not (0 <= point[0] < width and 0 <= point[1] < height):
        raise ValueError("ROI center is outside the source image")
    cw, ch = min(width, crop_size), min(height, crop_size)
    x = max(0, min(width - cw, int(round(point[0] - cw / 2))))
    y = max(0, min(height - ch, int(round(point[1] - ch / 2))))
    return (x, y, x + cw, y + ch)


def _box_iou(left, right):
    ax, ay, bx, by = map(float, left)
    cx, cy, dx, dy = map(float, right)
    intersection = max(0., min(bx, dx) - max(ax, cx)) * max(0., min(by, dy) - max(ay, cy))
    union = (bx - ax) * (by - ay) + (dx - cx) * (dy - cy) - intersection
    return intersection / union if union > 0 else 0.


def _same_observation(left, right, iou_threshold):
    a, b = left["bbox_xyxy"], right["bbox_xyxy"]
    iou = _box_iou(a, b)
    if iou >= iou_threshold:
        return True
    # Only compensate subpixel rounding for nearly coincident, similarly sized
    # boxes. A broad distance radius would collapse neighboring real objects.
    area_a = (a[2] - a[0]) * (a[3] - a[1])
    area_b = (b[2] - b[0]) * (b[3] - b[1])
    ratio = max(area_a, area_b) / max(1e-12, min(area_a, area_b))
    distance = math.dist(left["point"], right["point"])
    return iou >= .35 and ratio <= 1.6 and distance <= 1.5


def merge_candidates(candidates, iou_threshold=.55):
    """Suppress overlapping scale duplicates, preserving an actual observed box.

    A complete observation is preferred to one truncated by a crop/tile edge.
    Other observations remain in ``source_observations`` for audit. Confidence
    is the chosen observation's score, without an ensemble-vote bonus.
    """
    if not 0 < iou_threshold <= 1:
        raise ValueError("iou_threshold must be in (0, 1]")
    groups = []
    for original in sorted(candidates, key=lambda row: (bool(row.get("truncated")), -float(row["confidence"]))):
        row = dict(original)
        box = list(map(float, row["bbox_xyxy"]))
        if (len(box) != 4 or not all(map(math.isfinite, box))
                or box[2] <= box[0] or box[3] <= box[1]):
            raise ValueError("Cannot merge an invalid candidate box")
        row["bbox_xyxy"] = box
        row["point"] = [(box[0] + box[2]) / 2, (box[1] + box[3]) / 2]
        row["image_x"], row["image_y"] = row["point"]
        row["detector_family"] = "YOLO26x"
        matched = next((group for group in groups
                        if _same_observation(group[0], row, iou_threshold)), None)
        if matched is None:
            groups.append([row])
        else:
            matched.append(row)
    result = []
    for group in groups:
        representative = dict(group[0])
        sources = {}
        observations = []
        for member in group:
            # A per-mode result may already have merged overlapping windows.
            # Preserve its original observations when merging modes again.
            evidence = member.get("source_observations") or [member]
            for observed in evidence:
                name = str(observed.get("source", "YOLO26x"))
                sources[name] = max(float(observed["confidence"]), sources.get(name, 0.))
                record = {key: observed[key] for key in
                          ("source", "source_kind", "bbox_xyxy", "point", "confidence", "window_xyxy",
                           "window_index", "truncated", "truncated_at_window_edge", "truncated_at_image_edge")
                          if key in observed}
                if record not in observations:
                    observations.append(record)
        representative.update(sources=sources, source_observations=observations,
                              same_detector_observation_count=len(observations), independent_detector_families=1,
                              max_source_confidence=max(sources.values()),
                              coordinate_provenance="unchanged_real_detection_bbox_center")
        result.append(representative)
    return result


def candidate_from_box(box, window, image_size, source_kind, window_index=0):
    """Map one six-value YOLO crop box to original pixels; return None if empty."""
    values = np.asarray(box, dtype=float).reshape(-1)
    if len(values) != 6 or not np.isfinite(values).all():
        raise ValueError("YOLO returned a non-finite or malformed detection")
    lx, ly, rx, ry, confidence, class_id = values.tolist()
    if class_id != BALL_CLASS_ID:
        raise ValueError("Ball-only inference returned another detection class")
    if not 0 <= confidence <= 1 or rx < lx or ry < ly:
        raise ValueError("YOLO returned invalid box geometry/confidence")
    x0, y0, x1, y1 = map(int, window)
    width, height = map(int, image_size)
    if not (0 <= x0 < x1 <= width and 0 <= y0 < y1 <= height):
        raise ValueError("Prediction window must be inside the source image")
    if min(lx, ly) < -.1 or rx > x1 - x0 + .1 or ry > y1 - y0 + .1:
        raise ValueError("YOLO box is outside its original crop coordinates")
    # Ultralytics boxes are clipped to their source dimensions. Tolerate only
    # floating-point edge noise, retaining the exact in-bounds observed center.
    lx, ly = max(0., lx), max(0., ly)
    rx, ry = min(float(x1 - x0), rx), min(float(y1 - y0), ry)
    if rx <= lx or ry <= ly:
        return None
    epsilon = 2.
    left, top = lx <= epsilon, ly <= epsilon
    right, bottom = rx >= x1 - x0 - epsilon, ry >= y1 - y0 - epsilon
    internal = (left and x0 > 0) or (top and y0 > 0) or (right and x1 < width) or (bottom and y1 < height)
    external = (left and x0 == 0) or (top and y0 == 0) or (right and x1 == width) or (bottom and y1 == height)
    original = [lx + x0, ly + y0, rx + x0, ry + y0]
    point = [(original[0] + original[2]) / 2, (original[1] + original[3]) / 2]
    source = "YOLO26x_" + str(source_kind).upper()
    return dict(bbox_xyxy=original, point=point, image_x=point[0], image_y=point[1],
                confidence=float(confidence), class_id=BALL_CLASS_ID, class_name="sports ball",
                geometry_valid=True, source=source, source_kind=source_kind, detector_family="YOLO26x",
                sources={source: float(confidence)}, window_xyxy=list(window), window_index=int(window_index),
                truncated=bool(internal or external), truncated_at_window_edge=bool(internal),
                truncated_at_image_edge=bool(external),
                coordinate_provenance="unchanged_real_detection_bbox_center")


class BallDetector:
    """Batched FP32 ball candidate generation without player or pitch inputs.

    Every frame receives full-image and complete overlapping-tile inference.
    ROI windows are selected from that frame's full-image candidates; when
    there are none, a central crop is used. Candidates are not confirmed match
    balls. The model retains known microphone/logo false positives.
    """

    def __init__(self, weights, controller=None, *, full_imgsz=1536,
                 roi_imgsz=960, tile_size=640, tile_overlap=.20,
                 confidence=.10, device=0):
        import torch
        import ultralytics
        from ultralytics import YOLO, settings
        from .runtime_control import RuntimeController

        self.weights = Path(weights).resolve(strict=True)
        self.full_imgsz, self.roi_imgsz = int(full_imgsz), int(roi_imgsz)
        self.tile_size, self.tile_overlap = int(tile_size), float(tile_overlap)
        self.confidence, self.device = float(confidence), int(device)
        if min(self.full_imgsz, self.roi_imgsz, self.tile_size) < 1:
            raise ValueError("Inference sizes must be positive")
        if not 0 <= self.tile_overlap < 1 or not 0 < self.confidence < 1:
            raise ValueError("Invalid tile overlap or confidence")
        if not torch.cuda.is_available():
            raise RuntimeError("CUDA is required; this ball-only job has no CPU fallback")
        if not 0 <= self.device < torch.cuda.device_count():
            raise ValueError("Requested CUDA device does not exist")
        self.torch = torch
        self.ultralytics_version = ultralytics.__version__
        self.controller = controller if controller is not None else RuntimeController(gpu_index=self.device)
        settings.update({"sync": False})
        self.model = YOLO(str(self.weights))
        names = self.model.names
        name = names.get(BALL_CLASS_ID) if isinstance(names, dict) else names[BALL_CLASS_ID]
        if name != "sports ball":
            raise ValueError("Expected COCO YOLO26x class 32 = sports ball")

    def settings(self):
        return dict(version=VERSION, model="YOLO26x", weights=str(self.weights),
                    precision="FP32", device=self.device,
                    full_imgsz=self.full_imgsz, roi_imgsz=self.roi_imgsz,
                    roi_crop_size=640, tile_size=self.tile_size,
                    tile_overlap=self.tile_overlap, confidence=self.confidence,
                    classes=[BALL_CLASS_ID], rect=True, detector_families=1,
                    full_and_tile_on_every_frame=True,
                    roi_policy="full-image candidate centers; image-center fallback",
                    silent_quality_fallback=False,
                    ultralytics_version=self.ultralytics_version,
                    tf32=dict(matmul=self.torch.backends.cuda.matmul.allow_tf32,
                              cudnn=self.torch.backends.cudnn.allow_tf32))

    @staticmethod
    def _frame(frame):
        if not isinstance(frame, np.ndarray) or frame.ndim != 3 or frame.shape[2] != 3 or frame.dtype != np.uint8:
            raise ValueError("Expected an original BGR uint8 frame")
        if min(frame.shape[:2]) < 1:
            raise ValueError("Cannot infer an empty frame")
        return frame.shape[1], frame.shape[0]

    def _jobs(self, frames, jobs, *, mode, imgsz, batch):
        """Explicitly split windows before predict; its batch arg does not split lists."""
        batches = 0
        elapsed = 0.
        result = [[] for _ in frames]
        per_frame = [dict(input_images=0, wall_seconds=0., invalid_geometry=0)
                     for _ in frames]
        for first in range(0, len(jobs), batch):
            group = jobs[first:first + batch]
            images = [np.ascontiguousarray(frames[f][y0:y1, x0:x1])
                      for f, (x0, y0, x1, y1), _ in group]
            # All windows in a batch have one shape, preserving rectangular
            # inference rather than silently adding square-image padding.
            if len({image.shape for image in images}) != 1:
                raise ValueError("A prediction batch contains different crop shapes")
            started = time.perf_counter()
            try:
                with self.controller.gpu_work():
                    predictions = self.model.predict(
                        images, device=self.device, imgsz=imgsz, quantize=32,
                        batch=len(images), conf=self.confidence,
                        classes=[BALL_CLASS_ID], tile=False,
                        rect=True, verbose=False, save=False, stream=False)
                    if len(predictions) != len(group):
                        raise RuntimeError("YOLO output count differs from the actual input batch")
                    boxes = [prediction.boxes.data.float().cpu().numpy()
                             for prediction in predictions]
            except self.torch.cuda.OutOfMemoryError as exc:
                raise RuntimeError(
                    f"CUDA memory exhausted for {mode}: imgsz={imgsz}, batch={len(images)}. "
                    "Resolution and FP32 were preserved; choose a smaller batch explicitly.") from exc
            seconds = time.perf_counter() - started
            elapsed += seconds
            batches += 1
            if bool(self.model.predictor.model.fp16):
                raise RuntimeError("YOLO unexpectedly enabled FP16 instead of FP32")
            for (frame_index, window, window_index), detection_boxes in zip(group, boxes):
                timing = per_frame[frame_index]
                timing["input_images"] += 1
                timing["wall_seconds"] += seconds / len(group)
                width, height = self._frame(frames[frame_index])
                for box in detection_boxes:
                    candidate = candidate_from_box(box, window, (width, height), mode, window_index)
                    if candidate is None:
                        timing["invalid_geometry"] += 1
                    else:
                        result[frame_index].append(candidate)
        total = dict(mode=mode, wall_seconds=elapsed, input_images=len(jobs),
                     forward_batches=batches, maximum_batch=batch, imgsz=imgsz)
        return [merge_candidates(values) for values in result], per_frame, total

    def run_chunk(self, frames, *, full_batch=1, roi_batch=4, tile_batch=8, max_rois=2):
        """Return per-frame full/ROI/tile evidence, grouped by inference scale.

        max_rois=0 means all distinct full-candidate crop windows. At least one
        central ROI is processed when no full-image candidate exists. Batches
        change throughput, never input resolution or model precision.
        """
        frames = list(frames)
        if not frames:
            return []
        batches = tuple(int(value) for value in (full_batch, roi_batch, tile_batch))
        if min(batches) < 1 or int(max_rois) < 0:
            raise ValueError("Batch sizes must be positive and max_rois nonnegative")
        max_rois = int(max_rois)
        sizes = [self._frame(frame) for frame in frames]
        if len(set(sizes)) != 1:
            raise ValueError("One video chunk must contain frames of identical dimensions")
        width, height = sizes[0]
        started = time.perf_counter()
        full_jobs = [(f, (0, 0, width, height), 0) for f in range(len(frames))]
        full, full_timing, full_total = self._jobs(
            frames, full_jobs, mode="full", imgsz=self.full_imgsz, batch=batches[0])
        roi_jobs = []
        roi_centers = []
        for f, candidates in enumerate(full):
            centers = []
            windows = set()
            for candidate in sorted(candidates, key=lambda c: -float(c["confidence"])):
                center = list(map(float, candidate["point"]))
                window = crop_window(width, height, center, 640)
                if window in windows:
                    continue
                centers.append(center)
                windows.add(window)
                roi_jobs.append((f, window, len(centers) - 1))
                if max_rois and len(centers) >= max_rois:
                    break
            if not centers:
                center = [width / 2., height / 2.]
                centers.append(center)
                roi_jobs.append((f, crop_window(width, height, center, 640), 0))
            roi_centers.append(centers)
        roi, roi_timing, roi_total = self._jobs(
            frames, roi_jobs, mode="roi", imgsz=self.roi_imgsz, batch=batches[1])
        windows = tile_windows(width, height, self.tile_size, self.tile_overlap)
        tile_jobs = [(f, window, i) for f in range(len(frames))
                     for i, window in enumerate(windows)]
        tile, tile_timing, tile_total = self._jobs(
            frames, tile_jobs, mode="tile", imgsz=self.tile_size, batch=batches[2])
        seconds = time.perf_counter() - started
        result = []
        for f in range(len(frames)):
            modes = dict(full=full_timing[f], roi=roi_timing[f], tile=tile_timing[f])
            result.append(dict(full=full[f], roi=roi[f], tile=tile[f],
                candidates=merge_candidates(full[f] + roi[f] + tile[f]),
                roi_centers=roi_centers[f],
                timing=dict(wall_seconds=seconds / len(frames),
                    wall_seconds_is_chunk_average=True, chunk_wall_seconds=seconds,
                    chunk_frame_count=len(frames), modes=modes,
                    chunk_phase_totals=dict(full=full_total, roi=roi_total, tile=tile_total))))
        return result
