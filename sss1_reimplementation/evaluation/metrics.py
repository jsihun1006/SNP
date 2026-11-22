import math
import numpy as np
from ..detection.engine import iou


def locsim(distance, tau=1.):
    """Official sign: exp(log(.05) * distance**2 / tau**2), range [0,1]."""
    if tau <= 0 or not math.isfinite(tau):
        raise ValueError("tau must be finite and positive")
    d = np.asarray(distance, dtype=float)
    if not np.isfinite(d).all() or (d < 0).any():
        raise ValueError("Distances must be finite and nonnegative")
    return np.exp(np.log(.05) * d**2 / tau**2)


def localization_metrics(gt, predictions, bbox_iou=.5, success_m=1.):
    """Diagnostics with confidence-ordered, one-to-one BBOX association.

    This is deliberately separate from official confidence-ranked mAP-LocSim.
    Missing detections and invalid worlds remain in the success-rate denominator.
    """
    annotations = [a for a in gt["annotations"] if not a.get("iscrowd", 0)]
    if not annotations:
        raise ValueError("Nonempty ground truth required; no estimated scores")
    errors, deltas = [], []
    valid_count = sum(p.get("world_valid", True) and "position_on_pitch" in p for p in predictions)
    matched = 0
    for image in gt["images"]:
        targets = [a for a in annotations if a["image_id"] == image["id"]]
        used = set()
        detections = sorted([p for p in predictions if p["image_id"] == image["id"]],
                            key=lambda p: p["score"], reverse=True)
        for p in detections:
            def xyxy(b):
                x, y, w, h = b
                return (x, y, x+w, y+h)
            candidates = [(iou(xyxy(p["bbox"]), xyxy(a["bbox"])), j)
                          for j, a in enumerate(targets) if j not in used and a["category_id"] == p["category_id"]]
            best, j = max(candidates, default=(0., -1))
            if j < 0 or best < bbox_iou:
                continue
            used.add(j)
            matched += 1
            if "position_on_pitch" not in p or not p.get("world_valid", True):
                continue
            if "position_on_pitch" not in targets[j]:
                raise ValueError("World-coordinate ground truth is missing")
            delta = np.asarray(p["position_on_pitch"][:2]) - np.asarray(targets[j]["position_on_pitch"][:2])
            if not np.isfinite(delta).all():
                raise ValueError("Nonfinite world coordinates")
            deltas.append(delta)
            errors.append(float(np.linalg.norm(delta)))
    e = np.asarray(errors)
    return dict(association="confidence-greedy bbox IoU >= " + str(bbox_iou),
                ground_truth_count=len(annotations), prediction_count=len(predictions),
                bbox_matched_count=matched, world_matched_count=len(errors),
                detection_recall_at_bbox_iou=matched/len(annotations),
                world_valid_rate=valid_count/len(predictions) if predictions else 0.,
                localization_success_rate=float((e <= success_m).sum()/len(annotations)),
                success_radius_m=success_m,
                world_mae_m=float(e.mean()) if e.size else None,
                world_median_error_m=float(np.median(e)) if e.size else None,
                world_rmse_m=float(np.sqrt(np.mean(e**2))) if e.size else None,
                world_axis_mae_m=np.abs(deltas).mean(0).tolist() if e.size else None,
                matched_mean_locsim_tau1=float(locsim(e).mean()) if e.size else None)
