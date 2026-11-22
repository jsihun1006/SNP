from pathlib import Path
import os
import numpy as np
from ..types import Detection


class YOLODetector:
    def __init__(self, checkpoint="yolo26l.pt", device="cpu", imgsz=1280, person_class=0):
        os.environ.setdefault("YOLO_CONFIG_DIR", str(Path(__file__).resolve().parents[1]/"outputs/.ultralytics"))
        from ultralytics import YOLO
        self.model = YOLO(str(checkpoint))
        self.device, self.imgsz, self.person_class = device, imgsz, person_class
        names = self.model.names
        if person_class not in names:
            raise ValueError(f"Class {person_class} missing from model names: {names}")

    def predict(self, image, confidence=.25):
        result = self.model.predict(source=image, imgsz=self.imgsz, conf=confidence,
                                    classes=[self.person_class], device=self.device,
                                    verbose=False, rect=False, max_det=1000)[0]
        boxes = result.boxes
        if boxes is None:
            return []
        return [Detection(tuple(map(float, b)), float(s), int(c))
                for b, s, c in zip(boxes.xyxy.cpu().numpy(), boxes.conf.cpu().numpy(), boxes.cls.cpu().numpy())]


def official_sahi(image, checkpoint, device="cpu", confidence=.25, size=1280, overlap=.2, nms_iou=.5):
    """Optional SAHI package reference; merge settings are explicit for comparison."""
    os.environ.setdefault("YOLO_CONFIG_DIR", str(Path(__file__).resolve().parents[1]/"outputs/.ultralytics"))
    from sahi import AutoDetectionModel
    from sahi.predict import get_sliced_prediction
    model = AutoDetectionModel.from_pretrained(model_type="ultralytics", model_path=str(checkpoint),
                                               confidence_threshold=confidence, device=device, image_size=size)
    # SAHI takes RGB, Ultralytics takes BGR.
    result = get_sliced_prediction(image[..., ::-1].copy(), model,
                                  slice_height=size, slice_width=size,
                                  overlap_height_ratio=overlap, overlap_width_ratio=overlap,
                                  perform_standard_pred=False, postprocess_type="NMS",
                                  postprocess_match_metric="IOU", postprocess_match_threshold=nms_iou,
                                  postprocess_class_agnostic=False, verbose=0)
    return [Detection(tuple(p.bbox.to_xyxy()), p.score.value, p.category.id)
            for p in result.object_prediction_list if p.category.id == 0]
