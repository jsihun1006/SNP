# 축구 특화 YOLOv8x 25fps 진단

전술 영상의 원본 프레임 11185부터 18684까지 모두 처리했다. 샘플은 7,500개이며, 기존 manifest에 기록된 전체 경과 시간은 520.97초다. MPS FP16, 입력 크기 640, confidence 0.10, IoU 0.70, 배치 4를 사용했다.

기존 `detections.csv`, `frame_status.csv`, `run_manifest.json`은 `artifacts/tracking/football_yolov8x/dense_5min/`에 있다. 가중치 출처와 해시는 [모델 설명](../../../../src/tracking/football_yolov8x/README.md)에 기록했다.

```sh
python -m src.tracking.detection.extract --model football_yolov8x --imgsz 640 --conf .10 --iou .70 --batch 4 --device mps --half --stride 1 --start-frame 11185 --end-frame 18685 --out artifacts/tracking/football_yolov8x/dense_5min_reproduction
```
