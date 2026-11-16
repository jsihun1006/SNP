"""Geometry for detector boxes on the ORIGINAL tactical image.

The supplied homography maps normalized pitch coordinates to video pixels.
This module is a projection utility, not a calibration or accuracy estimator.
"""
import numpy as np


def box_anchors(boxes_xyxy, class_ids):
    """Person (COCO 0): bottom centre; ball (COCO 32): box centre.

    Boxes must already be in original video pixels (Ultralytics boxes.xyxy are).
    Person bottom centre approximates a ground contact point. A cropped bbox or
    an airborne ball violates this approximation; do not call the result truth.
    """
    boxes = np.asarray(boxes_xyxy, dtype=np.float64).reshape(-1, 4)
    classes = np.asarray(class_ids).reshape(-1)
    if len(boxes) != len(classes):
        raise ValueError("boxes and class_ids must have equal length")
    if not np.isin(classes, [0, 32]).all():
        raise ValueError("Only COCO person=0 and sports ball=32 are supported")
    if not np.isfinite(boxes).all() or np.any(boxes[:, 2:] < boxes[:, :2]):
        raise ValueError("Boxes must be finite x1,y1,x2,y2 with nonnegative size")
    anchors = (boxes[:, :2] + boxes[:, 2:]) / 2
    anchors[classes == 0, 1] = boxes[classes == 0, 3]
    return anchors


def image_to_pitch(matrix, points_px, valid=True, pitch=(105.3, 68.0), eps=1e-10):
    """Return normalized/metre coordinates and validity flags without clamping.

    In-frame or matrix validity alone does not prove calibration accuracy. Points
    outside 0..1 remain outside; the in_pitch flag allows a caller to filter them
    while preserving the original result for diagnosis.
    """
    points = np.asarray(points_px, dtype=np.float64).reshape(-1, 2)
    homography = np.asarray(matrix, dtype=np.float64)
    xy = np.full_like(points, np.nan)
    usable = np.zeros(len(points), dtype=bool)
    if homography.shape != (3, 3):
        raise ValueError("homography must be 3x3")
    if valid and np.isfinite(homography).all():
        try:
            inverse = np.linalg.inv(homography)
        except np.linalg.LinAlgError:
            inverse = None
        if inverse is not None:
            homogeneous = np.column_stack((points, np.ones(len(points)))) @ inverse.T
            usable = np.isfinite(homogeneous).all(axis=1) & (np.abs(homogeneous[:, 2]) > eps)
            xy[usable] = homogeneous[usable, :2] / homogeneous[usable, 2, None]
    in_pitch = usable & (xy[:, 0] >= 0) & (xy[:, 0] <= 1) & (xy[:, 1] >= 0) & (xy[:, 1] <= 1)
    return {"xy_normalized": xy, "xy_metres": xy * np.asarray(pitch),
            "projection_valid": usable, "in_pitch": in_pitch}
