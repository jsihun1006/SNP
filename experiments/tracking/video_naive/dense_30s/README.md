# YOLO11n 25fps 30초 진단

전술 영상의 원본 프레임 11250부터 11999까지 모두 처리했다. 샘플은 750개이며, 기존 manifest에 기록된 전체 경과 시간은 44.78초다. MPS, 입력 크기 1280, confidence 0.15, IoU 0.70, 배치 8을 사용했다.

원시 좌표와 미리보기는 `artifacts/tracking/video_naive/dense_30s/`에 있다. 검출 범위와 공 후보 커버리지는 [첫 전체 실험의 보고서](../full_1fps/RESULTS.md#추가-원본-25fps-실행)에 함께 기록돼 있다.

```sh
python -m src.tracking.detection.extract --model video_naive --imgsz 1280 --conf .15 --iou .70 --batch 8 --device mps --stride 1 --start-frame 11250 --end-frame 12000 --out artifacts/tracking/video_naive/dense_30s_reproduction
```
