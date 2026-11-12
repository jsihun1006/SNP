# Tryolabs를 broadcast에 적용한 로컬 실험

실행일: 2026-11-10. 영상으로부터 검출·추적을 직접 실행하는 별도 실험이다.
기존 `competition_judgement`, tactical 모델, ATD, homography, 정답, 제출 파일은 입력으로 사용하지 않는다.
원본 영상은 읽기만 하고, 외부 코드·모델과 미리보기·프레임별 로그는 `data/derived/tryolabs_broadcast/`에 저장한다.

## 코드가 하는 일

분석한 원본은 [tryolabs/soccer-video-analytics](https://github.com/tryolabs/soccer-video-analytics),
commit `b00c7c75a1d52bd5ca183521a35fc76fbd89cc0e`이다.

1. `inference/yolov5.py`: COCO YOLOv5x로 사람을 찾고, 별도 `ball.pt`로 공을 찾는다. 입력 크기는640.
2. `run_utils.py`: 사람 confidence >0.35, 공 >0.3을 유지한다. 사람에는 심판·스태프도 포함될 수 있다.
3. `run.py`: Norfair 선수/공 tracker와 카메라 운동 추정을 실행한다. 선수 초기화3, 공 초기화20;
   `hit_counter_max`는 각각90/2000이다. 25fps에서2000은80초 분량의 설정값이지만 실제 잔존 시간은 tracker 상태에 따라 달라진다.
4. `inference/hsv_classifier.py`: 박스 위쪽 몸통을 잘라 HSV 색상별 픽셀 수로 팀을 선택한다.
   `InertiaClassifier`가 같은 track ID의 최근20회 분류를 다수결로 결합한다.
5. `soccer/match.py`: 공 중심과 선수 박스 아래 양 모서리 중 가까운 쪽의 거리가45px 이하이면
   해당 팀의 근접 카운터를 올리고,20 이상이면 점유팀을 바꾼다. 실제 발 keypoint 검출이 아니다.
6. `Match.update_possession`: 공 검출 전에 기존 점유팀 시간을 올린다. dead ball, 리플레이, 경기 전 행사 판별은 없다.

원본 설명에는 연속 프레임이라고 쓰여 있지만 실제로는 **공 누락/거리 초과 시 근접 카운터를 초기화하지 않는다.**
따라서20회가 실제로 연속이라는 보장은 없다. 또한 첫 점유팀이 Man City로 고정돼 있다.

공 모델은 제작자가 [release v0](https://github.com/tryolabs/soccer-video-analytics/releases/tag/v0)에서
특정 영상에 과적합된 데모용이며 일반적인 축구 영상용 모델이 아니라고 명시했다.

## 이번 적용에서 보존하거나 바꾼 부분

별도 실행기는 [starter/run_tryolabs_broadcast.py](../starter/run_tryolabs_broadcast.py)다.
upstream 소스는 수정하지 않고 로컬에서 import한다. 원본 선수/공 모델,640 입력, BGR 전달,
confidence/거리/시간 기준, tracker 설정, `Match.update`의 점유 규칙을 보존했다.
YOLOv5 코드는 무고정 최신 branch 대신 같은 시기의 **v6.2**, commit
`d3ea0df8b9f923685ce5f2555c303b8eddbf83fd`로 고정했다.

- Chelsea/Man City 이름·색상 대신 원본에 있는 일반 HSV 범위로 `BLUE`, `LIGHT`, `OTHER`를 구성했다.
  BLUE=blue, LIGHT=white, OTHER=black/green/yellow이며 장면별 보정은 하지 않았다.
  이 명칭은 **색상 휴리스틱의 추정값**이다. 실제 팀 ID나 H/A 매핑을 확정하지 않는다.
  초기 away 점유라는 원본 동작을 보존하므로 초기 표시값은 LIGHT다.
- 원본 `HSVClassifier`는 모든 색상 득점이0이어도 첫 범주를 선택할 수 있다. `OTHER`도
  보장된 미분류 판정기가 아니다. 조명·유니폼 줄무늬·작은 박스·골키퍼·심판에 오분류가 생길 수 있다.
- 원본/Pillow의 호환되지 않는 그리기 대신 OpenCV 진단 화면과 H.264 인코더를 사용한다.
  좌측은 원본, 우측은 검출·추적 결과다. 표시만960×540으로 축소하고 **거리 계산은1920×1080에서 한다.**
- 원본 점유 카운터와 함께 별도의 `Fresh proximity evidence`를 표시한다. 공 후보가 정확히1개이고,
  해당 프레임에서 관측된 선수 track이45px 이내이며 BLUE/LIGHT일 때만20프레임 연속 근접 근거를 인정한다.
  하나라도 끊기면0으로 초기화한다. **이 값도 controlled possession 정답이나 신뢰확률이 아니다.**
- 공 후보는 초록 원으로 강조한다. 모델이 검출한 물체라는 뜻이지 실제 공임을 확인했다는 뜻은 아니다.
  과거 검출을 재사용한 tracker 박스는 주황 점선과 age로 드러낸다. 원본 `Converter`는
  현재 보정된 `estimate` 대신 `last_detection`을 반환하므로 화면 이동 시 과거 위치가 남을 수 있다.
- 공은 있는데 선수가0명인 경우 원본의 `min([])` 예외를 막고 해당 frame은 공 근거 없이 진행한다.
  운동 추정 예외 시 상태를 초기화한다. 두 경우 모두 `events.json`에 기록한다.
- 고정 구간 사이에는 모든 상태를 초기화한다. 편집된 구간을 하나의 연속 경기로 취급하지 않는다.

## 실행과 산출물

설치된 로컬 환경에서 재실행:

```bash
data/derived/tryolabs_broadcast/.venv/bin/python starter/run_tryolabs_broadcast.py --self-check
data/derived/tryolabs_broadcast/.venv/bin/python starter/run_tryolabs_broadcast.py \
  --starts 600 1800 3600 --seconds 10 \
  --output data/derived/tryolabs_broadcast/preview_new
open -a 'QuickTime Player' data/derived/tryolabs_broadcast/preview/broadcast_tryolabs.mp4
```

기존 출력 폴더가 있으면 덮어쓰기를 거부한다. `--starts`는 경기 시계가 아닌 **broadcast 파일 기준 초**다.
이번 원본은1920×1080,25fps,164,487프레임,6,579.48초다.
파일 앞부분에는 경기 전 행사도 포함되므로 파일10분과 경기10분은 다르다.
ATD나 tactical 시간축과의 정렬은 이 실험에 필요하지 않으며 검증하지도 않는다.

| 파일 | 내용 |
|---|---|
| `preview/broadcast_tryolabs.mp4` | 좌우 비교 미리보기, H.264,25fps,1920×800 |
| `preview/summary.json` | 구간·프레임 수·관측률·실행시간·버전·모델 SHA256·원본 불변성 확인 |
| `preview/frames.csv` | 원본/출력 frame 대응, 검출 수, stale age, 원본 카운터, 근접 근거·보류 사유 |
| `preview/detections.jsonl` | 모델의 실제 bounding box/confidence; 사람이 수정하지 않은 후보 |
| `preview/events.json` | 운동 추정 실패·빈 선수 예외 방어 발생 기록 |
| `preview/segment_*.jpg` | 각 구간의 처음/중간/마지막 화면 |
| `environment_diagnostics.json` | 실제 재현한 원본 라이브러리 호환성 오류 |
| `preview.log` | 모델 실행 및 진행 로그 |

새 환경 설치에는 [requirements-tryolabs.txt](../requirements-tryolabs.txt)를 별도 venv에 사용한다.
이번 `.venv`는 기존 `mtah2027`의 torch를 읽는 `--system-site-packages` 환경이며,
numpy/OpenCV/Norfair/torchvision 등 추가 설치는 venv 안에 했다. 기존 환경 패키지를 변경하지 않았다.
원본 저장소와 모델을 다시 받으려면 다음 공개 자료를 사용한다.

```bash
BASE=data/derived/tryolabs_broadcast
git clone https://github.com/tryolabs/soccer-video-analytics.git "$BASE/upstream"
git -C "$BASE/upstream" checkout b00c7c75a1d52bd5ca183521a35fc76fbd89cc0e
git clone --branch v6.2 --depth 1 https://github.com/ultralytics/yolov5.git "$BASE/yolov5"
curl -L --fail https://github.com/tryolabs/soccer-video-analytics/releases/download/v0/ball.pt.zip -o "$BASE/ball.pt.zip"
unzip "$BASE/ball.pt.zip" ball.pt -d "$BASE"
curl -L --fail https://github.com/ultralytics/yolov5/releases/download/v6.2/yolov5x.pt -o "$BASE/yolov5x.pt"
python3.11 -m venv "$BASE/.venv"
"$BASE/.venv/bin/python" -m pip install -r requirements-tryolabs.txt
```

다운로드의 경로는 이미 존재하면 재사용하고 clone/설치를 반복할 필요가 없다.
`ball.pt` SHA256은 `a2425cd079d0ade7a4d9d4689554abc65c7d675d4abe23503939e757dabaf8e9`,
`yolov5x.pt`는 `9f27a794fa0308e2606f90565164571d6f0a0ba18a3ce2e5e5d323b71c157859`이다.

## 오류와 한계

실제 환경 오류와 입력 영상에서의 검출 실패를 구분한다.

| 항목 | 확인 방식·원인 | 처리 |
|---|---|---|
| 기본 Python에 cv2/torch 없음 | 시스템 Python3.9.6 패키지 검사 | 기존 CPython3.11 기반 별도 venv |
| torchvision/Norfair 누락 | 실제 import에서 `ModuleNotFoundError` | 별도 venv에 설치 |
| `torch.load`의 `UnpicklingError` | legacy 모델 전체 객체를 담은 checkpoint와 현재 weights-only 기본값의 불일치 | 위 공개 모델 로딩 범위에서만 `weights_only=False`, 즉시 원복 |
| Pillow `ImageDraw.textsize` 제거 | 원본 `Draw.text_in_middle_rectangle` 직접 호출 | 원본 Match 계산을 유지하고 그리기만 OpenCV로 교체 |
| 공 누락 시 점유 누적 | synthetic self-check 및 실제 영상의 frame 로그 | 원본 카운터를 남기되 관측 근거를 별도 표시 |
| 공 있는데 선수0명에서 `min([])` | synthetic self-check로 원본 오류 재현 | guard와 이벤트 로그 |
| 카메라 운동과 과거 박스 불일치 | synthetic self-check로 converter가 estimate 대신 last_detection임을 확인 | stale age와 점선 표시; 정확한 현재 위치로 간주하지 않음 |

카메라 컷·리플레이에 대한 자동 구간 분리, 강건한 공 재식별, 골키퍼/심판 분류, 원근에 따른 거리 보정,
공중볼·실제 제어 여부·dead ball 판별은 구현돼 있지 않다.45px은 미터 단위 거리가 아니며
카메라 줌과 선수의 영상 내 위치에 따라 의미가 달라진다.
CPU에서 처리하며 실시간 속도나 전체 경기 정확도를 주장하지 않는다.

이 실험은 코드 학습과 파이프라인 디버깅이다. 영상에서 사람이 정답을 만들거나 후보를 교정하거나
장면별 임계값을 조정하지 않았으며, 관측률을 정확도/재현율/F1로 해석하지 않는다.
대회 원본·클립·프레임별 파생물은 기존 `.gitignore`가 제외하는 `data/derived/` 아래에 두었다.

## 실제 실행 결과

고정된 파일 시각10분/30분/60분부터 각각10초, 총30초/750프레임을 처리했다.
각 구간은250프레임이며, H.264 출력750프레임 전체 디코딩 및25fps 검증을 통과했다.
CPU 처리·인코딩·출력 검사에 총243.26초가 걸렸다. 전체109분 영상을 처리한 결과는 아니다.

| 파일 시간 | source frame0 범위 | 평균 사람 검출 수 | 공 후보가 나온 프레임 | 공 후보2개 이상 | 원본 main ball이 과거 검출인 프레임 |
|---|---|---:|---:|---:|---:|
| 10:00–10:10 | 15000–15249 | 11.54 | 87/250 (34.8%) | 7 | 12 |
| 30:00–30:10 | 45000–45249 | 1.62 | 10/250 (4.0%) | 0 | 0 |
| 60:00–60:10 | 90000–90249 | 19.35 | 221/250 (88.4%) | 88 | 15 |

이는 **공 후보 관측 빈도**다. 수작업 정답을 만들지 않았으므로 검출 정확도·재현율이 아니다.
사람 검출 수에는 심판·주변 스태프가 섞일 수 있다.

- 첫 구간: 공 후보 누락이163프레임이며, 근접 조건을 연속 만족한 최대 길이는16프레임이다.
  BLUE/LIGHT 분류가 불확실한 경우도 있다. 원본 점유값은10초 내내 초기값 LIGHT였다.
- 두 번째 구간: 심판 클로즈업이 포함된다. 저장된 첫 화면에서 모델은 **심판 손 주변의 작은 물체를
  confidence 약0.68의 공 후보로 표시**한다. 이는 파이프라인의 오검출을 설명하기 위한 시각적 진단이며,
  프레임별 수동 정답이나 보정값으로 저장·사용하지 않았다. 원본 점유 카운터는 이 구간에서도 LIGHT10초를 누적했다.
- 세 번째 구간: 공 후보는 더 자주 나오지만88프레임에서 복수 후보가 생겼다.
  공 후보가1개여도 근접 선수가45px보다 먼 경우가90프레임이다. 원본 점유는55프레임 LIGHT 후195프레임 BLUE였지만,
  별도 fresh 근접 근거는 최대7프레임 연속에 그쳤다.
- 세 구간 모두 엄격한20연속 fresh 근접 근거를 만족한 프레임은0개다.
  **이는 실제 점유가 없다는 뜻이 아니라, 이번 검출·분류 결과로 그 조건을 충족하지 못했다는 뜻**이다.
  모델의 일반화 한계, 작은 공·가림, 복수 후보와 색상 분류를 먼저 개선해야 안정적인 점유 근거를 얻을 수 있다.
  1920폭 영상을640으로 줄여 추론하므로 가로 방향 물체 크기는 약1/3이 된다. 공처럼 작은 물체에는 불리하다.
- 실제30초 처리 중 운동 추정 예외나 빈 선수 guard 발생은0건이다. 빈 선수/과거 박스/누락 카운터 문제는
  별도 synthetic self-check로 재현했다. `environment_diagnostics.json`에는 실제 PyTorch/Pillow 오류를 남겼다.
- 원본 영상의 size/mtime가 실행 전후 동일함을 확인했다. 파생 데이터가 Git에서 제외되는 것도 확인했다.
  QuickTime 자동 실행은 `Unable to find application named 'QuickTime Player'`로 실패했다.
  이 실행에서 GUI 재생을 확인했다고 주장하지 않는다. 파일 링크나 위 `open` 명령으로 사용자의 앱에서 확인할 수 있다.

현재 결과는 **선수·공 추적 파이프라인의 시연과 실패 원인 확인에는 유용하지만,
이 경기의 점유율이나 H/A/D 정답으로 신뢰할 단계는 아니다.**
