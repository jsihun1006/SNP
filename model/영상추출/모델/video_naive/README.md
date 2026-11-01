# 영상 → 좌표 naive baseline: YOLO11n

이 모델은 **실제 전술 MP4를 읽어** 사람과 공을 검출한다. ATD 좌표나 공개 baseline의
H/A/D를 입력으로 사용하지 않는다. 학습은 하지 않으며 COCO 사전학습 YOLO11n을 사용한다.

```text
원본 영상 → YOLO11n(person, sports ball) → 원본 픽셀의 bbox
        → 사람: bbox 하단 중앙 / 공: bbox 중심
        → 제공 homography 역변환 → 정규화 경기장 좌표 → 미터 좌표
```

제공 homography는 이미 계산된 카메라 보정값이다. 이 첫 모델은 카메라 보정 자체를 새로 추정하지 않는다.
공중에 뜬 공을 지면으로 투영하면 실제 위치와 차이가 날 수 있다.

## 첫 실행 범위

전술 원본은 1920×1080, 25fps, 199,319프레임이다. 전체 영상을 **25프레임 간격(1fps)**으로
처리하고 원본 프레임 번호와 시간을 보존한다. 샘플링하지 않은 프레임은 좌표를 만들거나
보간해 검출한 것처럼 기록하지 않는다. 별도 짧은 구간은 25fps로 실행한다.
기본 입력 크기 1280, confidence 0.15, NMS IoU 0.7, batch 8을 사용한다.
설정은 공개 점유 정답에 맞춰 고르지 않았으며 검출 과정에서는 해당 정답을 읽지 않는다.

## 실행

저장소 루트에서:

```bash
model/영상추출/모델/video_naive/.venv/bin/python model/영상추출/모델/video_naive/extract.py
model/영상추출/모델/video_naive/.venv/bin/python model/영상추출/모델/video_naive/validate.py
model/영상추출/모델/video_naive/.venv/bin/python model/영상추출/모델/video_naive/assess.py
model/영상추출/모델/video_naive/.venv/bin/python model/영상추출/모델/video_naive/preview.py
```

기존 결과를 보존하려면 새 `--out`을 지정한다. 전체 25fps 실행은 처리 시간이 길어진다.

```bash
model/영상추출/모델/video_naive/.venv/bin/python model/영상추출/모델/video_naive/extract.py \
  --stride 1 --out model/영상추출/모델/video_naive/outputs_full25
```

환경 재구성은 Python3.12 가상환경에서 `requirements.lock.txt`를 설치하고 공식 가중치를 받는다.
가중치 URL: https://github.com/ultralytics/assets/releases/download/v8.3.0/yolo11n.pt
첫 실행의 파일 해시는 `outputs/run_manifest.json`에 기록한다.
영상 처리는 로컬에서 수행한다. Ultralytics HUB와 동기화는 비활성화했다.

`run.py --out 새폴더`로 검출 → 무결성 검사 → 참고 데이터 비교 → 미리보기를 한 번에 실행할 수도 있다.

## 결과 파일

- `outputs/detections.csv`: 프레임·클래스·confidence·bbox·픽셀 기준점·경기장 좌표·변환 유효성.
- `outputs/frame_status.csv`: 샘플링한 모든 프레임. 검출이 0개인 프레임도 포함한다.
- `outputs/run_manifest.json`: 영상·가중치·보정값 해시, 실제 실행 설정·버전·처리시간.
- `outputs/reference_comparison.csv`, `outputs/summary.json`: 기존 ATD와의 일치도와 커버리지.
- `outputs/preview.mp4`, `outputs/contact_sheet.jpg`: 사용자 확인용 로컬 미리보기.
- `RESULTS.md`: 실제 실행 결과와 한계.

`x_norm`, `y_norm`은 경기장 0–1 좌표, `x_m`, `y_m`은 105.3×68m 기준이다.
범위 밖 좌표를 강제로 잘라내지 않고 `in_pitch=0`으로 표시한다.
`detection_index`는 그 프레임 내 번호이며 선수 신원이나 지속적인 트랙 ID가 아니다.

## 해석 범위

- `person`에는 선수뿐 아니라 심판·코치·관중이 포함될 수 있다. 팀 구분과 선수 신원 추적은 하지 않는다.
- 공 검출 프레임 비율은 **커버리지**다. 검출 정답이 없어 정확도·recall·mAP로 부를 수 없다.
- 비교하는 ATD도 보정되지 않은 자동 추적이다. 좌표 차이는 **기존 자동 데이터와의 일치도**다.
- 좌표 추출 모델이며 H/A/D 점유 예측이나 대회 spellF1 평가를 완료한 모델은 아니다.
- 생성 좌표·미리보기·가중치·가상환경은 Git에서 제외한다. 원본 영상과 ATD는 수정하지 않는다.

## 출처와 라이선스

- [YOLO11 공식 설명](https://docs.ultralytics.com/models/yolo11): COCO 사전학습 검출기, `yolo11n.pt`.
- [공식 예측 API](https://docs.ultralytics.com/modes/predict): 원본 이미지 좌표와 배치 추론.
- [COCO 클래스 정의](https://raw.githubusercontent.com/ultralytics/ultralytics/main/ultralytics/cfg/datasets/coco.yaml): person=0, sports ball=32.
- Ultralytics 코드·가중치 라이선스: AGPL-3.0/Enterprise. 프로젝트의 원본 경기 데이터에는 별도 이용약관이 적용된다.

## 실제 저장된 추가 구간

`outputs/dense_30s/`에는 00:07:30–00:08:00의 원본 25fps 750프레임 좌표와 30초 미리보기가 있다.
기본 `outputs/`의 1fps 전체 시간대 결과와 구분한다. 자세한 수치와 검증 범위는 `RESULTS.md`에 기록했다.
