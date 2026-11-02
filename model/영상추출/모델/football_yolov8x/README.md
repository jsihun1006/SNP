# 축구 특화 영상 검출 모델

공·골키퍼·선수·심판을 검출하는 `gianpaj/football-players-detection-1` YOLOv8x 공개 가중치를 로컬에서 사용한다.
픽셀 바운딩 박스에서 공 중심과 사람 하단 중앙을 잡고, 제공된 전술 homography 역변환으로 피치 좌표를 계산한다.
학습이나 추론에 대회 공개 정답을 사용하지 않는다.

- 모델: https://huggingface.co/gianpaj/football-players-detection-1 (AGPL-3.0)
- 학습 데이터: https://universe.roboflow.com/roboflow-jvuqo/football-players-detection-3zvbc/dataset/1 (CC BY 4.0, 2022-12-05 버전)
- 공개 가중치: https://huggingface.co/gianpaj/football-players-detection-1/resolve/main/weights/best.pt
- 가중치 SHA-256: `a35ca40ea9e728288b86b37f728afbe601dfd7ec58f30d4661900c2d9b308932`
- 클래스: ball=0, goalkeeper=1, player=2, referee=3. 팀 분류는 하지 않는다.
- 기존 video_naive의 `.venv`를 사용하며 `weights/football_yolov8x.pt` 가중치는 Git에서 제외한다.

## 실행 조건과 산출물

전체 원본 199,319프레임 중 25프레임마다 검출해 원본 시간대의 7,973개 프레임을 분석한다.
학습 입력 크기 640, 모델 카드의 권장 confidence 0.1, IoU 0.7, MPS FP16을 고정했다.
공개 평가 구간 11185–18684는 원본 25fps로 별도 추출해 진단한다. 점유 제출 파일과 평가는
공개 구간만 우대하지 않도록 전체 영상에서 동일한 1fps 결과만 사용한다.

`outputs/full_1fps/detections.csv`는 네 클래스의 픽셀·경기장 좌표를 담고,
`outputs/full_1fps/frame_status.csv`는 검출이 없어도 분석한 프레임을 기록한다.
`outputs/dense_5min/`은 별도 25fps 진단이며 영상 및 파생 좌표는 로컬 전용이다.
두 파일 모두 원본 0기반 프레임 번호를 보존한다. 경기장 밖 좌표는 강제로 잘라내지 않고 플래그를 기록한다.

```bash
PY=model/영상추출/모델/video_naive/.venv/bin/python
DETECT=model/영상추출/모델/football_yolov8x/extract.py
$PY $DETECT --imgsz 640 --batch 4 --half --stride 25 --out model/영상추출/추출csv/football_yolov8x/새전체결과
$PY $DETECT --imgsz 640 --batch 4 --half --stride 1 --start-frame 11185 --end-frame 18685 --out model/영상추출/추출csv/football_yolov8x/새5분결과
```

기존 결과를 덮어쓰지 않도록 새 출력 폴더를 지정해야 한다. 비-MPS 환경에서는 `--device cpu`와 `--half` 제거가 필요할 수 있다.
공 검출 confidence는 정답 확률이 아니며 작은 공, 가림, 높이 있는 공의 지면 투영, 오검출에 취약하다.
