import time
import numpy as np
from .geometry import project_ground


FIELDS = ["image_id", "frame", "detection_id", "bbox_x1", "bbox_y1", "bbox_x2", "bbox_y2", "confidence",
          "pelvis_u", "pelvis_v", "ground_u", "ground_v", "world_x", "world_y", "world_valid", "world_reason"]


class Pipeline:
    def __init__(self, detection_engine, pose_estimator=None):
        self.detection, self.pose = detection_engine, pose_estimator

    def infer(self, image, image_id, frame=0, camera=None):
        if camera is not None and image.shape[:2] != (camera.height, camera.width):
            raise ValueError("Frame dimensions differ from calibration; rescale calibration explicitly")
        start = time.perf_counter()
        boxes = self.detection.predict(image)
        points = self.pose.predict(image, boxes) if self.pose is not None else np.full((len(boxes), 2, 2), np.nan)
        points = np.asarray(points)
        if points.shape != (len(boxes), 2, 2):
            raise ValueError("Pose must return (detections, 2, 2) original-image pixels")
        rows = []
        for index, (d, p) in enumerate(zip(boxes, points)):
            world = project_ground(camera, p[1])
            row = dict(image_id=image_id, frame=frame, detection_id=index,
                       **dict(zip(FIELDS[3:7], d.bbox)), confidence=d.confidence,
                       pelvis_u=float(p[0, 0]), pelvis_v=float(p[0, 1]), ground_u=float(p[1, 0]), ground_v=float(p[1, 1]),
                       world_x=world.x, world_y=world.y, world_valid=world.valid, world_reason=world.reason)
            if self.pose is None:
                row["world_reason"] = "missing_pose_checkpoint"
            # Empty CSV fields / JSON null, never non-standard JSON NaN.
            rows.append({k: None if isinstance(v, float) and not np.isfinite(v) else v for k, v in row.items()})
        seconds = time.perf_counter()-start
        return rows, dict(seconds=seconds, fps=1/seconds if seconds else None, **self.detection.last_timing)


def coco_predictions(rows, category_id=1):
    output = []
    for r in rows:
        item = dict(image_id=r["image_id"], category_id=category_id, score=r["confidence"],
                    bbox=[r["bbox_x1"], r["bbox_y1"], r["bbox_x2"]-r["bbox_x1"], r["bbox_y2"]-r["bbox_y1"]],
                    world_valid=r["world_valid"])
        if r["world_valid"]:
            item["position_on_pitch"] = [r["world_x"], r["world_y"], 0.]
        if all(r[k] is not None for k in ("pelvis_u", "pelvis_v", "ground_u", "ground_v")):
            item["keypoints"] = [r["pelvis_u"], r["pelvis_v"], 1, r["ground_u"], r["ground_v"], 1]
        output.append(item)
    return output
