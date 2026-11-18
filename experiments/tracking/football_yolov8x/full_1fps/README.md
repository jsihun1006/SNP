# 축구 특화 YOLOv8x 전체 영상 검출

전술 영상의 원본 프레임 0부터 199318까지 25프레임 간격으로 처리했다. 샘플은 7,973개이며, 기존 manifest에 기록된 전체 경과 시간은 675.63초다. MPS FP16, 입력 크기 640, confidence 0.10, IoU 0.70, 배치 4를 사용했다.

기존 `detections.csv`, `frame_status.csv`, `run_manifest.json`은 `artifacts/tracking/football_yolov8x/full_1fps/`에 있다. 가중치 출처와 해시는 [모델 설명](../../../../src/tracking/football_yolov8x/README.md)에 기록했다.

```sh
python -m src.tracking.detection.extract --model football_yolov8x --imgsz 640 --conf .10 --iou .70 --batch 4 --device mps --half --stride 25 --out artifacts/tracking/football_yolov8x/full_1fps_reproduction
```
