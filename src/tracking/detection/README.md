# 공용 영상 검출 파이프라인

YOLO11n과 축구 특화 YOLOv8x는 검출 좌표를 저장하고, homography를 역변환하고, 검증·ATD 비교·미리보기를 만드는 코드를 공유한다. 이전 두 모델 폴더의 `extract.py`, `geometry.py`, `validate.py`, `assess.py`, `preview.py`를 이 폴더에 통합했고, 중복된 좌표 테스트는 `tests/test_detector_geometry.py`에 모았다.

| 모델 선택 | 공 클래스 | 사람 후보 | 기본 confidence | 부가 옵션 |
|---|---:|---|---:|---|
| `video_naive` | 32 | COCO person 0 | 0.15 | 기본 FP32 |
| `football_yolov8x` | 0 | 골키퍼 1·선수 2·심판 3 | 0.10 | `--half` |

클래스 ID와 가중치, 모델별 기본 설정은 `profile.py`에 둔다. 두 설정의 가중치와 원시 예측은 각각 `artifacts/tracking/<모델>`에서 찾을 수 있다. 이전 결과와 manifest는 당시 실행 기록이므로 그대로 보관했다.

```sh
python -m src.tracking.detection.extract --model football_yolov8x --out artifacts/tracking/football_yolov8x/new_run
python -m src.tracking.detection.validate --model football_yolov8x --out artifacts/tracking/football_yolov8x/new_run
```

나머지 실행 예시는 각 모델의 README에 있다. YOLO11n 평가 문서는 실행별 출력의 `RESULTS.md`로 저장한다. 기존 실험 결과 문서는 `experiments/tracking/video_naive/full_1fps/RESULTS.md`에 보관한다.
