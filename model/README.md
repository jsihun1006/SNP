# 모델 작업 폴더

**Phase 1: 점유 계산에 필요한 데이터 생성 → Phase 2: 프레임별 점유 예측 → 평가**

- **Phase 1**: 영상·제공 ATD에서 좌표, 팀 정보, 프레임 대응, 특징값을 만든다.
- **Phase 2**: Phase 1 결과 또는 제공 ATD를 받아 `H/A/D/X`를 예측한다. 팀별 점유율은 이 예측에서 집계한다.
- 이 문서의 Phase 1·2는 **팀 내부 작업 단계**다. 아래 파일 규칙은 **앞으로 추가하는 모델·실행**에 적용한다.

## 1. 폴더 구조

```text
model/
├── 영상추출/                              # Phase 1
│   ├── 모델/<추출모델명>/                  # 코드·설정·환경·모델 설명
│   └── 추출csv/<추출모델명>/<실행ID>/      # 좌표·특징·정렬 결과
├── 점유율계산/                            # Phase 2
│   ├── 계산알고리즘/<점유모델명>/          # 점유 판단 코드·설정·평가 코드
│   └── 성능지표/
│       ├── <점유모델명>/<실행ID>/          # 해당 실행의 평가 결과
│       └── 모델_성능_비교.csv              # 실행별 주요 점수 집계
└── 모델별결과csv/
    ├── <점유모델명>/<실행ID>/              # 프레임별 점유 예측·진단
    └── <추출모델명>/                      # 필요한 경우 Phase 1 결과로 연결
```

- **모델명**: 코드·알고리즘 구분. 영문 소문자와 `_` 사용. 예: `football_yolov8x`.
- **실행ID**: 데이터·설정·실행 구분. 예: `full_1fps_20261102`. 설정을 바꿔 실행하면 새 ID를 만든다.
- **원본 입력**: 저장소의 `data/`, `mp4/`에서 읽는다. 원본을 모델 폴더에 복사하지 않는다.
- **결과는 한 곳에 저장**: 코드 폴더의 `outputs/` 등은 결과 폴더로 연결하는 상대 심볼릭 링크로 사용한다.

## 2. 코드 폴더에 넣을 파일

| 파일 | Phase 1 `영상추출/모델/<모델명>/` | Phase 2 `점유율계산/계산알고리즘/<모델명>/` |
|---|---|---|
| `README.md` | 입력·추출 방식·출력 컬럼·실행 명령 | 입력 데이터·점유 판단 방식·출력·실행 명령 |
| `extract.py` | **필수**: 좌표·특징 추출 진입점 | — |
| `possession.py` | — | **필수**: 프레임별 점유 예측 진입점 |
| `validate.py` | **필수**: 추출 CSV의 컬럼·프레임·좌표 검사 | **필수**: 예측 CSV의 프레임·라벨 검사 |
| `evaluation.py` | — | **필수**: 공식 평가 코드 호출·점수 저장 |
| `config.json` | **필수**: 모델·임계값·샘플링 등 설정 | **필수**: 입력 선택·거리·평활화·상태 전환 등 설정 |
| `requirements.txt` | **필수**: 실행 의존성. 공유 환경이면 설치 파일·경로 명시 | 동일 |
| `weights/` | 가중치를 사용하는 경우 로컬 보관 | 동일 |
| 기타 `.py` | 전처리·트래킹·좌표 변환 등 보조 코드 | 계산·후처리·점유율 집계 등 보조 코드 |

- 각 모델 README에는 **입력 경로 → 실행 명령 → 출력 경로 → 컬럼 의미**를 적는다.
- 가중치는 **출처·버전·다운로드 방법·해시**를 모델 README에 적는다.
- 코드 폴더에 대용량 CSV·영상·평가 결과를 직접 쌓지 않는다.

## 3. Phase 1 — 추출 데이터 저장 규칙

**저장 위치:** `영상추출/추출csv/<추출모델명>/<실행ID>/`

| 파일 | 생성 조건 | 저장 내용 |
|---|---|---|
| `detections.csv` | 영상 검출 모델 | **검출 1개당 1행**. 클래스·confidence·bbox·경기장 좌표 |
| `tracking.csv` | 트래킹·ATD 변환 모델 | **프레임 × 객체당 1행**. 객체/슬롯 ID·팀·좌표·관측 여부 |
| `features.csv` | 특징 추출 모델 | **프레임당 1행**. `frame,time_seconds`와 이름·단위가 정의된 특징 컬럼 |
| `frame_status.csv` | **필수** | **처리한 프레임당 1행**. 검출 0개인 프레임도 기록 |
| `run_manifest.json` | **필수** | 입력 파일·모델/가중치·설정·버전·프레임 범위·FPS·좌표 기준·해시 |
| `validation.json` | **필수** | 형식 검사 결과·처리 프레임 수·누락/중복 여부 |
| `summary.json`, `preview.mp4` 등 | 선택 | 검출 커버리지·요약·확인용 미리보기 |

**모델 종류에 맞는 데이터 파일을 생성한다.** 검출만 한 모델에 트랙 ID나 팀 정보를 임의로 붙이지 않는다.

### CSV 컬럼

| 파일 | 지정 컬럼 |
|---|---|
| `detections.csv` | `frame,time_seconds,detection_index,class_id,class_name,confidence,x1_px,y1_px,x2_px,y2_px,anchor_x_px,anchor_y_px,x_norm,y_norm,x_m,y_m,projection_valid,in_pitch,homography_error` |
| `tracking.csv` | `frame,time_seconds,track_id,id_kind,object_type,team,x_norm,y_norm,x_m,y_m,confidence,observed` |
| `frame_status.csv` | 공통 필수: `frame,time_seconds`. 영상 검출은 `n_person,n_ball,n_person_in_pitch,n_ball_in_pitch` 추가 |

- **프레임**: 원본 소스의 **0-based 번호** 유지. ATD 원본 ID는 `frame + 1`이다. 샘플링 후 재번호를 매기지 않는다.
- **시간·좌표**: `time_seconds = frame / fps`. `_px`는 원본 영상 픽셀, `_norm`은 경기장 정규화 좌표, `_m`은 미터다. FPS·피치 크기는 manifest에 기록한다.
- **ID·팀**: `detection_index`는 해당 프레임 안의 번호다. `id_kind`는 `tracker`/`atd_slot`로 구분한다. ATD 슬롯을 영구 선수 ID로 취급하지 않는다. `team`은 `H/A`, 미확인·비대상은 빈칸이다.
- **결측·추정**: 결측은 빈칸으로 저장한다. 기존 CSV의 `NaN`도 결측으로 읽는다. 보간·유지·예측 좌표는 `observed=0`으로 표시하고 생성 방식을 기록한다.
- **처리 범위**: 처리하지 않은 프레임은 추출 결과에 만들지 않는다. 검출 실패와 미처리를 `frame_status.csv`로 구분한다.
- 클래스 번호·객체 종류·추가 특징의 의미는 **모델 README에 정의**한다. 서로 다른 모델의 `class_id`를 같다고 가정하지 않는다.

## 4. Phase 2 — 점유 예측 저장 규칙

**저장 위치:** `모델별결과csv/<점유모델명>/<실행ID>/`

| 파일 | 필수 내용 |
|---|---|
| `prediction.csv` | **정확히 `frame,label` 두 컬럼**. 현재 전술 영상 전체 `0..199318`의 199,319행. 라벨은 `H/A/D/X` |
| `frame_diagnostics.csv` | 최소 `frame,predicted_label,input_source`. 거리·선택 객체·결측·보간 여부 등 판단 근거 추가 |
| `run_manifest.json` | 점유 모델·설정·코드/예측 해시, **입력 Phase 1 모델명·실행ID·파일 경로·해시**. 제공 ATD 사용 시 그 출처도 기록 |
| `validation.json` | 전체 프레임 수·순서·중복·라벨·제출 형식 검사 결과 |
| `possession_share.csv` | 점유율 집계 시 생성. `scope,team,frames,seconds,share` |

- `H`: 홈팀, `A`: 원정팀, `D`: 데드볼, `X`: 판단 유보. **공 미검출을 곧바로 D로 확정하지 않는다.**
- 입력 좌표가 1fps여도 최종 예측은 **전체 프레임**에 대해 만든다. 그 사이의 유지·보간·판단 규칙을 명시한다.
- `share`는 **H/(H+A), A/(H+A)**이며 0~1 값으로 저장한다. D/X는 분모에서 제외하고, 분모가 0이면 빈칸이다. `scope`에 집계 구간을 적는다.
- **추론 진단에는 GT를 섞지 않는다.** 정답과 비교한 프레임별 진단은 평가 결과의 `reports/`에 둔다.

## 5. 최종 성능 평가

**저장 위치:** `점유율계산/성능지표/<점유모델명>/<실행ID>/`

| 파일 | 내용 |
|---|---|
| `scoreboard.csv` | 공식 평가 출력. `spell_f1_mean`, `macro_f1`, 클래스별 지표 등 |
| `reports/` | 상세 평가 JSON·전환 분석·정답 비교 진단 |
| `RESULTS.md` | 입력 조합·평가 구간·주요 점수·한계 |
| `evaluation_manifest.json` | 평가한 예측 경로·해시·GT 구분·평가 설정·코드 버전 |

- **흐름**: `prediction.csv` 생성 → 형식 검증 → 저장소 `evaluation/`의 공식 평가 → 위 폴더에 저장 → `모델_성능_비교.csv`에 집계.
- **주요 지표**: `spellF1`과 `macroF1`. 검출 비율·좌표 일치도는 Phase 1 진단이며 점유 예측 점수와 구분한다.
- **비교표 컬럼**: `submission,spell_f1_mean,macro_f1,accuracy,f1_H,f1_A,f1_D,trans_f1_25,poss,n_graded`.
- 신규 `submission` 값은 `<점유모델명>__<실행ID>`로 적는다. 점유 예측·평가 폴더의 실행ID는 같게 유지한다. 실행 전 점수는 채우지 않는다.
- 공개 5분 GT는 **파이프라인 확인용**이다. 학습·튜닝·모델 선택에 사용하지 않는다. 최종 대회 평가는 비공개 전체 경기 GT 기준이다.

## 6. 새 작업 시작 순서

1. **추출 모델 추가**: `영상추출/모델/<모델명>/`에 코드·설정·설명을 넣는다.
2. **Phase 1 실행·검증**: 새 실행ID로 좌표/특징 CSV와 manifest를 저장한다.
3. **점유 모델 추가**: `점유율계산/계산알고리즘/<모델명>/`에 사용할 입력 파일·컬럼·처리 규칙을 정한다.
4. **Phase 2 실행·검증**: 같은 실행의 `prediction.csv`·진단·manifest를 모델별 결과 폴더에 저장한다.
5. **평가·기록**: 공식 평가 결과와 비교표를 저장한다. 다른 추출 모델과 조합하면 새 실행ID로 구분한다.

## 7. 현재 모델과 기존 파일

| 단계 | 모델/도구 | 입력 → 역할 |
|---|---|---|
| Phase 1 | `video_naive` | 영상 → YOLO11n 사람·공 검출. 팀 분류·ID 트래킹 없음 |
| Phase 1 | `football_yolov8x` | 영상 → 축구 전용 공·선수 검출. 팀 분류·ID 트래킹 없음 |
| Phase 1 | `tracking_alignment` | 제공 ATD → 두 소스 정렬·초기 X/빈칸 라벨·팀 이동 진단 |
| Phase 2 | `naive` | 제공 ATD → 최근접 선수 팀과 평활화로 점유 예측 |
| Phase 2 | `model1` | ATD의 최근접 팀/누락 관측 → 상태 전환 비용으로 점유 예측 |
| Phase 2 | `video_ball_hybrid` | 영상 공 검출 + ATD 선수/팀 → 최근접 팀 관측 + model1 디코더 |

- **기존 호환**: `model.py`, `evaluate.py`, `predictions/<모델명>.csv`, 실행ID 없는 결과 등은 기존 이름을 유지한다. 이번 README 수정으로 파일을 일괄 변경하지 않는다.
- **기존 연결**: `outputs`, `predictions`, `diagnostics`, `reports`, `scoreboard.csv` 등 일부 경로는 분류된 결과 폴더로 이어지는 상대 심볼릭 링크다.
- **이전 manifest**: 절대경로·코드 해시는 실행 당시 기록이다. 신규 실행은 새 경로·설정·해시로 기록한다.
- **Git**: 코드·설명·설정·집계 지표를 관리한다. 원본/파생 좌표·프레임별 예측·영상·가중치·가상환경은 로컬에 보관하고 `.gitignore`로 제외한다.

## KKJ branch 정리

- `starter/`의 정렬·시각화 Python 9개, 문서 2개, 루트 `requirements-tracking.txt` → `영상추출/모델/tracking_alignment/`.
- 공식 starter의 baseline·로더·노트북은 유지. 합성 테스트는 루트 `tests/`에서 실행한다.
- 초기 X/빈칸 라벨·이동 진단은 점유 예측이나 정답이 아니다. 별도의 대회 점수를 부여하지 않는다.
- 상세 실행법: [tracking_alignment/README.md](영상추출/모델/tracking_alignment/README.md).
