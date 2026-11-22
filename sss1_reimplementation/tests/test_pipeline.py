import json
from pathlib import Path
import numpy as np
import pytest
import cv2
from ..types import Detection
from ..detection import DetectionEngine
from ..pipeline import Pipeline, coco_predictions
from ..scripts.prepare_data import convert_pose, prepare_detection
from .test_geometry import down_camera


class FixtureDetector:
    """Contract fixture, NOT learned detection."""
    def predict(self,image,confidence): return [Detection((800,400,1200,800),.9)]


class FixturePose:
    """Contract fixture, NOT RTMPose inference."""
    def predict(self,image,boxes): return np.array([[[1000,500],[1000,600]]] * len(boxes))


def test_contract_integration_and_missing_calibration():
    image=np.zeros((1200,2000,3),np.uint8)
    pipe=Pipeline(DetectionEngine(FixtureDetector(),mode="full"),FixturePose())
    rows,timing=pipe.infer(image,12,0,down_camera())
    assert rows[0]["world_valid"] and rows[0]["world_x"]==pytest.approx(0.)
    assert timing["seconds"]>0
    assert coco_predictions(rows)[0]["image_id"]==12
    rows,_=pipe.infer(image,12)
    assert rows[0]["world_x"] is None and not rows[0]["world_valid"]
    json.dumps(rows,allow_nan=False)
    rows2,_=pipe.infer(image,12,1)
    assert rows2[0]["detection_id"]==rows[0]["detection_id"]==0
    with pytest.raises(ValueError,match="dimensions"): pipe.infer(image[:100],12,camera=down_camera())


def test_missing_pose_does_not_fabricate_keypoints():
    pipe=Pipeline(DetectionEngine(FixtureDetector(),mode="full"))
    rows,_=pipe.infer(np.zeros((1200,2000,3),np.uint8),1,camera=down_camera())
    assert rows[0]["ground_u"] is None and not rows[0]["world_valid"]
    assert "position_on_pitch" not in coco_predictions(rows)[0]


def test_data_conversion_retains_calibration_and_prepares_yolo(tmp_path):
    doc=dict(images=[dict(id=1,file_name="frame.jpg",width=160,height=80,camera_matrix=[[1,0,0,0]]*3)],
             categories=[dict(id=1,name="person")],annotations=[dict(id=1,image_id=1,category_id=1,
             bbox=[30,10,20,50],area=1000,keypoints=[[40,30,1],[40,60,1]])])
    source=tmp_path/"source.json";source.write_text(json.dumps(doc),encoding="utf-8")
    output=tmp_path/"pose.json"
    convert_pose(source,output)
    converted=json.loads(output.read_text(encoding="utf-8"))
    assert len(converted["annotations"][0]["keypoints"])==6
    assert "camera_matrix" in converted["images"][0]
    cv2.imwrite(str(tmp_path/"frame.jpg"),np.zeros((80,160,3),np.uint8))
    prepare_detection(source,tmp_path,tmp_path/"yolo","train","sahi")
    labels=(tmp_path/"yolo/labels/train/1_0.txt").read_text().split()
    assert list(map(float,labels))==pytest.approx([0,.25,.4375,.125,.625])
    with pytest.raises(FileExistsError): convert_pose(source,output)
