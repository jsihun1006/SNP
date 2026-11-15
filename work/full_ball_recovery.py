"""Bounded, observation-only recovery for full-run ball correction.

Search centers are hints, never output coordinates.  This module appends only
new YOLO observations and does not decide which observation is the match ball.
The frame provider is called once per requested local frame, in increasing
order, so the caller can decode sequentially without retaining a raw video.
"""
from __future__ import annotations

from contextlib import nullcontext
import math
import time

import cv2
import numpy as np


ACCEPTED_STATUSES = {"selected_observation", "visually_reviewed_observation"}


def _finite_point(value):
    if value is None:
        return None
    point = np.asarray(value, dtype=float)
    return point if point.shape == (2,) and np.isfinite(point).all() else None


def _inside(point, width, height):
    return point is not None and 0 <= point[0] < width and 0 <= point[1] < height


def _propagate(point, anchor, target, context):
    """Map an anchor image point through measured camera motion only.

    Cuts and unmeasured motion break the hint; ball motion is deliberately not
    fabricated.  The resulting broad crop can still find a nearby moving ball.
    """
    position = np.asarray(point, dtype=float).copy()
    cuts = context["cuts"]
    motion_valid = context.get("motion_valid")
    if target > anchor:
        transitions = range(anchor + 1, target + 1)
    else:
        transitions = range(anchor, target, -1)
    for index in transitions:
        if bool(cuts[index]) or (motion_valid is not None and not bool(motion_valid[index])):
            return None
        warp = np.asarray(context["warps"][index], dtype=float)
        if warp.shape != (2, 3) or not np.isfinite(warp).all():
            return None
        matrix = np.vstack((warp, (0.0, 0.0, 1.0)))
        if target < anchor:
            try:
                matrix = np.linalg.inv(matrix)
            except np.linalg.LinAlgError:
                return None
        q = matrix @ np.r_[position, 1.0]
        if not np.isfinite(q).all() or abs(q[2]) <= 1e-12:
            return None
        position = q[:2] / q[2]
    return position


def _anchors(initial):
    good = []
    for index, row in enumerate(initial):
        point = _finite_point((row.get("pixel_x"), row.get("pixel_y")))
        confidence = row.get("candidate_confidence")
        if (row.get("status") in ACCEPTED_STATUSES and point is not None
                and not row.get("needs_review", False)
                and (row.get("visual_reviewed", False)
                     or (confidence is not None and float(confidence) >= .5))):
            good.append(index)
    return good


def _offset(center, crop_size, width, height):
    crop_width, crop_height = min(crop_size, width), min(crop_size, height)
    x = max(0, min(width - crop_width, int(round(float(center[0]) - crop_width / 2))))
    y = max(0, min(height - crop_height, int(round(float(center[1]) - crop_height / 2))))
    return x, y, crop_width, crop_height


def _make_plans(records, context, initial, crop_size, max_jobs_per_frame,
                temporal_gap_frames, full_frame_fallback):
    n = len(records)
    width, height = int(context.get("width", 1920)), int(context.get("height", 1080))
    anchors = _anchors(initial)
    previous = np.full(n, -1, dtype=int)
    following = np.full(n, -1, dtype=int)
    anchor_set = set(anchors)
    last = -1
    for i in range(n):
        previous[i] = last
        if i in anchor_set:
            last = i
    last = -1
    for i in range(n - 1, -1, -1):
        following[i] = last
        if i in anchor_set:
            last = i
    plans = []
    for i, record in enumerate(records):
        hints = []
        prior = _finite_point(context["atd_pixel"][i])
        error = float(context["h_error"][i])
        if _inside(prior, width, height) and np.isfinite(error) and 0 <= error <= 2:
            hints.append(dict(kind="atd", point=prior, anchor_frame=None))
        temporal = []
        for anchor in (int(previous[i]), int(following[i])):
            if anchor < 0 or abs(anchor - i) > temporal_gap_frames:
                continue
            point = (initial[anchor]["pixel_x"], initial[anchor]["pixel_y"])
            predicted = _propagate(point, anchor, i, context)
            if _inside(predicted, width, height):
                temporal.append(dict(kind="temporal", point=predicted,
                                     anchor_frame=int(records[anchor]["frame"]),
                                     gap_frames=abs(anchor - i)))
        temporal.sort(key=lambda item: item["gap_frames"])
        hints.extend(temporal)
        # An uncertain existing observation is a search seed, not a label.
        if not hints and record["candidates"]:
            best = max(record["candidates"], key=lambda candidate: candidate["confidence"])
            seed = _finite_point(best["point"])
            if _inside(seed, width, height) and float(best["confidence"]) >= .15:
                hints.append(dict(kind="existing_candidate", point=seed, anchor_frame=None))
        uncertain = (initial[i].get("status") not in ACCEPTED_STATUSES
                     or initial[i].get("needs_review", False)
                     or float(initial[i].get("candidate_confidence") or 0) < .45)
        selected_hints = []
        windows = set()
        for hint in hints:
            nearest = min((np.linalg.norm(np.asarray(c["point"]) - hint["point"])
                           for c in record["candidates"]), default=float("inf"))
            if not uncertain and nearest <= 40:
                continue
            window = _offset(hint["point"], crop_size, width, height)
            if window in windows:
                continue
            if any(np.linalg.norm(hint["point"] - other["point"]) < 64 for other in selected_hints):
                continue
            windows.add(window)
            selected_hints.append(hint)
            if len(selected_hints) >= max_jobs_per_frame:
                break
        if selected_hints:
            plans.append((i, selected_hints))
        elif full_frame_fallback and uncertain and not record["candidates"] and not hints:
            plans.append((i, [dict(kind="full_frame", point=None, anchor_frame=None)]))
    return plans


def recover_chunk(records, context, frames, initial, model, controller, *,
                  batch_size=4, crop_size=512, imgsz=1280, confidence=.08,
                  max_jobs_per_frame=2, temporal_gap_frames=12,
                  progress_callback=None, full_frame_fallback=False,
                  full_imgsz=1536):
    """Append crop detections in-place and return JSON-serializable counters.

    ``frames`` can be a sequence of original BGR uint8 frames or a callable
    ``frame_getter(local_index)``.  Callables receive strictly increasing unique
    indices. ``context`` supplies H, h_error, atd_pixel, warps, cuts and optional
    motion_valid, width and height.  H is normalized-pitch-to-image.

    Optional whole-frame fallback uses the original image at input 1536 and a
    batch of one. It remains disabled by default pending the caller's benchmark.
    """
    started = time.perf_counter()
    if len(records) != len(initial):
        raise ValueError("Records and initial selections must have identical lengths")
    if min(batch_size, crop_size, imgsz, max_jobs_per_frame, full_imgsz) < 1 or temporal_gap_frames < 0:
        raise ValueError("Invalid recovery batch/crop/input/job/gap setting")
    if not 0 < confidence < 1:
        raise ValueError("Recovery confidence must be between zero and one")
    for key in ("H", "h_error", "atd_pixel", "warps", "cuts"):
        if len(context[key]) != len(records):
            raise ValueError("Context length mismatch: " + key)
    if context.get("motion_valid") is not None and len(context["motion_valid"]) != len(records):
        raise ValueError("Context length mismatch: motion_valid")
    plans = _make_plans(records, context, initial, crop_size, max_jobs_per_frame,
                        temporal_gap_frames, full_frame_fallback)
    counts_before = [len(record["candidates"]) for record in records]
    stats = dict(target_frames=len(plans), planned_jobs=sum(len(hints) for _, hints in plans),
                 processed_jobs=0, processed_frames=0, new_candidates=0,
                 frames_with_new_candidates=0, frames_previously_without_candidates_now_with_candidates=0,
                 duplicate_lower_confidence_skipped=0, crop_jobs=0, full_frame_jobs=0,
                 hint_counts={}, model_precision="FP32", crop_size=crop_size, input_size=imgsz,
                 confidence_threshold=confidence, batch_size=batch_size,
                 full_frame_fallback=bool(full_frame_fallback), full_input_size=full_imgsz,
                 elapsed_seconds=0.0)
    queue = []
    queue_key = None
    changed_frames = set()
    completed_frames = set()
    last_progress_jobs = 0

    def publish(force=False):
        nonlocal last_progress_jobs
        stats["elapsed_seconds"] = time.perf_counter() - started
        stats["processed_frames"] = len(completed_frames)
        stats["frames_with_new_candidates"] = len(changed_frames)
        stats["frames_previously_without_candidates_now_with_candidates"] = sum(
            counts_before[index] == 0 and bool(records[index]["candidates"]) for index in changed_frames)
        if progress_callback is not None and (force or stats["processed_jobs"] - last_progress_jobs >= 32):
            progress_callback(dict(stats))
            last_progress_jobs = stats["processed_jobs"]

    def flush():
        nonlocal queue, queue_key
        if not queue:
            return
        work_context = controller.gpu_work() if controller is not None else nullcontext()
        with work_context:
            predictions = list(model.predict(
                [job["image"] for job in queue], device=0,
                imgsz=queue[0]["imgsz"], quantize=32, conf=confidence,
                classes=[32], tile=False, rect=True, save=False, verbose=False,
                stream=False, batch=len(queue)))
            if len(predictions) != len(queue):
                raise RuntimeError("Recovery model returned a different image count")
            all_boxes = []
            for prediction in predictions:
                if prediction.boxes is None:
                    all_boxes.append(np.empty((0, 6)))
                else:
                    all_boxes.append(prediction.boxes.data.float().cpu().numpy())
        predictor = getattr(model, "predictor", None)
        backend = getattr(predictor, "model", None)
        if bool(getattr(backend, "fp16", False)):
            raise RuntimeError("Recovery unexpectedly enabled FP16")
        for job, rows in zip(queue, all_boxes):
            index = job["index"]
            candidates = records[index]["candidates"]
            offset_x, offset_y = job["offset"]
            crop_height, crop_width = job["image"].shape[:2]
            for box in rows:
                if len(box) != 6 or not np.isfinite(box).all():
                    raise ValueError("Recovery model returned an invalid box")
                x1, y1, x2, y2, score, class_id = map(float, box)
                if class_id != 32:
                    raise ValueError("Ball recovery returned an unexpected class")
                if not 0 <= score <= 1 or min(x2 - x1, y2 - y1) < 1:
                    continue
                if x1 < -.1 or y1 < -.1 or x2 > crop_width + .1 or y2 > crop_height + .1:
                    raise ValueError("Recovery box is outside its original input crop")
                global_box = [x1 + offset_x, y1 + offset_y, x2 + offset_x, y2 + offset_y]
                point = [(global_box[0] + global_box[2]) / 2,
                         (global_box[1] + global_box[3]) / 2]
                if any(np.linalg.norm(np.asarray(c["point"]) - point) < 2
                       and float(c["confidence"]) >= score for c in candidates):
                    stats["duplicate_lower_confidence_skipped"] += 1
                    continue
                kind = job["hint"]["kind"]
                source = (f"YOLO26x_RECOVERY_FULL_{job['imgsz']}" if kind == "full_frame"
                          else f"YOLO26x_RECOVERY_{crop_size}_{job['imgsz']}")
                observed = dict(point=point, bbox_xyxy=global_box, confidence=score,
                                source=source, sources={source: score}, source_observations=[],
                                origin=("full_frame_redetection" if kind == "full_frame" else "targeted_redetection"),
                                detector_family="YOLO26x",
                                coordinate_provenance="unchanged_real_detection_bbox_center",
                                recovery_hint=dict(kind=kind,
                                    center=None if job["hint"]["point"] is None else job["hint"]["point"].tolist(),
                                    anchor_frame=job["hint"].get("anchor_frame")),
                                recovery_window_xyxy=[offset_x, offset_y, offset_x + crop_width, offset_y + crop_height],
                                recovery_input_size=job["imgsz"], recovery_confidence_threshold=confidence,
                                truncated_at_window_edge=bool(min(x1, y1, crop_width - x2, crop_height - y2) <= 1.))
                prior = _finite_point(context["atd_pixel"][index])
                observed["atd_distance_px"] = float(np.linalg.norm(np.asarray(point) - prior)) if prior is not None else None
                h = np.asarray(context["H"][index], dtype=float)
                if h.shape == (3, 3) and np.isfinite(h).all():
                    try:
                        ray = np.linalg.solve(h, np.r_[point, 1.])
                    except np.linalg.LinAlgError:
                        ray = None
                    if ray is not None and np.isfinite(ray).all() and abs(ray[2]) > 1e-12:
                        pitch = ray[:2] / ray[2]
                        observed["pitch"] = pitch.tolist()
                        observed["penalty_spot_distance_m"] = float(min(
                            np.linalg.norm((pitch - [11 / 105.3, .5]) * [105.3, 68]),
                            np.linalg.norm((pitch - [1 - 11 / 105.3, .5]) * [105.3, 68])))
                px, py = int(round(point[0] - offset_x)), int(round(point[1] - offset_y))
                patch = job["image"][max(0, py - 14):min(crop_height, py + 15),
                                     max(0, px - 14):min(crop_width, px + 15)]
                if patch.size:
                    hsv = cv2.cvtColor(patch, cv2.COLOR_BGR2HSV)
                    observed["green_fraction"] = float(np.mean(
                        (hsv[:, :, 0] >= 30) & (hsv[:, :, 0] <= 95)
                        & (hsv[:, :, 1] > 45) & (hsv[:, :, 2] > 35)))
                else:
                    observed["green_fraction"] = 0.0
                candidates.append(observed)
                changed_frames.add(index)
                stats["new_candidates"] += 1
            completed_frames.add(index)
            stats["processed_jobs"] += 1
        queue = []
        queue_key = None
        publish()

    publish(force=True)
    for index, hints in plans:
        frame = frames(index) if callable(frames) else frames[index]
        if not isinstance(frame, np.ndarray) or frame.ndim != 3 or frame.shape[2] != 3 or frame.dtype != np.uint8:
            raise ValueError("Frame provider must return original BGR uint8 images")
        height, width = frame.shape[:2]
        for hint in hints:
            kind = hint["kind"]
            if kind == "full_frame":
                image = np.ascontiguousarray(frame)
                offset = (0, 0)
                job_imgsz = full_imgsz
                limit = 1
                stats["full_frame_jobs"] += 1
            else:
                if not _inside(hint["point"], width, height):
                    raise ValueError("Planned search hint lies outside the actual video frame")
                x, y, cw, ch = _offset(hint["point"], crop_size, width, height)
                image = np.ascontiguousarray(frame[y:y + ch, x:x + cw])
                offset = (x, y)
                job_imgsz = imgsz
                limit = batch_size
                stats["crop_jobs"] += 1
            key = (job_imgsz, image.shape)
            if queue and key != queue_key:
                flush()
            queue_key = key
            stats["hint_counts"][kind] = stats["hint_counts"].get(kind, 0) + 1
            queue.append(dict(index=index, image=image, offset=offset, hint=hint, imgsz=job_imgsz))
            if len(queue) >= limit:
                flush()
    flush()
    publish(force=True)
    return stats
