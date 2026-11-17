# possession

채택 모델: **5fps PathCRF + 29에폭 25fps 모델 + 고정 보정**. 기존 DFL J03WR9의 고정 D 조건에서 Spell F1 **76.22%**인 설정이다. 추가 MLP는 사용하지 않는다.

이 폴더에는 **추론·DFL 전처리 코드, 파라미터, H/A 가중치 2개, 필수 DFL NPZ, README**를 둔다. 학습 코드·로그·오답노트는 포함하지 않는다. 바로 사용할 DFL 입력 NPZ와 기존 5fps 출력 NPZ는 inputs/에 포함한다. 기존 데드볼 모델 결과를 입력으로 받아 H/A/D를 완성한다.

## 파일

- `run.py`: 동봉 NPZ를 읽는 기본 실행 진입점
- `prepare_dfl.py`: 로컬 DFL 전처리 → NPZ
- `inputs/`: 준비된 DFL 입력 및 고정 5fps NPZ
- `predict.py`: 최종 추론 → `frame,label` CSV
- `code/network_25fps.py`: 25fps attention 모델·Viterbi
- `code/inputs.py`, `runtime.py`, `data_utils.py`: 좌표·속도·관측 마스크·창·5fps 연결
- `code/postprocess.py`, `signals.py`: 채택 보정·터치 후보·경계 이동
- `vendor/pathcrf/`: 5fps 추론에 필요한 공개 코드와 원래 LICENSE
- `weights/pathcrf_trial140.pt`: 고정 5fps 가중치
- `weights/specialist_epoch29.pt`: 채택한 29에폭 25fps 가중치
- `parameters.json`, `parameters.csv`: 모델·보정·분할·가중치 SHA256
- `requirements.txt`: 추론 의존성

## 바로 실행

DFL 평가 경기 J03WR9의 **전처리된 NPZ가 포함**되어 있다. 사용자가 NPZ를 따로 만들 필요 없이 아래 한 명령으로 25fps 추론과 최종 보정까지 연결된다. 기존 5fps 출력은 동봉한 고정 캐시를 재사용한다.

```sh
python model/possession/run.py
```

입력: `inputs/J03WR9.npz`, `inputs/J03WR9_base5.npz` → 기존 모델·보정 → `outputs/possession.csv`.
기본 NPZ에는 선수·볼·팀·하프·원본 프레임·시간·관측 결측·기존 예측 D가 들어 있으며, 점유 정답은 들어 있지 않다. 이 DFL 진단 출력은 대회 영상용 제출 파일과 프레임 수가 다르다.

다른 입력을 처리할 때는 `run.py --input ... --dead ... --output ...` 또는 아래 `predict.py`를 사용한다. CSV는 `frame,label` 형식이다.

## DFL 전처리 코드

```sh
# 기존 로컬 DFL XML/Parquet에서 같은 입력 규격으로 변환. 모델 실행은 하지 않음.
python model/possession/prepare_dfl.py J03WR9
# 필요할 때 7경기 모두 변환
python model/possession/prepare_dfl.py all
```

현재 저장소의 `data/external/dfl`과 `data/transition_final/source`를 기본 입력으로 사용한다. `--data-dir`, `--raw-dir`, `--native-dir`, `--base-dir`, `--dead`, `--output-dir`로 경로를 바꿀 수 있다. 원본 XML만 있을 때는 로컬 XML을 읽어 Parquet로 변환하고, 하프/프레임 정렬·105×68m 좌표·팀/골키퍼·관측 결측·기존 D 마스크를 NPZ로 저장한다. 외부 다운로드는 하지 않는다. DFL 정답 D를 입력 마스크로 몰래 사용하지 않는다.

이번 변경에서는 준비된 NPZ 복사와 코드 연결만 했으며, 전처리·추론·학습을 새로 실행하지 않았다.

## 직접 추론

```sh
python -m pip install -r model/possession/requirements.txt
python model/possession/predict.py \
  --input /path/to/tracking.npz \
  --dead /path/to/dead_mask.csv \
  --output /path/to/possession.csv \
  --device auto
```

출력은 **`frame,label` 두 열**, **0부터 연속하는 영상 프레임**, 라벨 **H/A/D**다. 입력에 영상 전체 프레임을 넣어야 한다. 정답 점유 라벨은 입력하지 않는다. 이번 정리 작업에서 실제 대회 경기 추론이나 제출은 실행하지 않았다.

## 입력 NPZ

| 키 | 형태 | 내용 |
|---|---|---|
| `players` | N×P×2 | 선수 트랙의 미터 좌표. 결측 NaN |
| `ball` | N×2 | 볼의 미터 좌표. 결측 NaN |
| `team` | P | 홈 0 / 원정 1 |
| `frame` | N | 제출 영상 프레임 0~N−1 |
| `half` | N | 전반 1 / 후반 2 / 하프 밖 0 |
| `seconds` | N | 각 하프 안에서 증가하는 원본 시간(초) |
| `dead` | N, 선택 | 기존 D 모델의 0/1 마스크. `--dead` CSV 우선 |
| `pitch` | 2, 선택 | 좌표의 피치 길이·너비. 기본 105×68m |
| `source_frame` | N, 선택 | 중복 없는 원본 ID. 기본 `frame` |
| `player_names` | P, 선택 | 고유 트랙명. 홈 `home_`, 원정 `away_` 접두사 |
| `goalkeeper` | P, 선택 | 골키퍼 1 / 다른 역할 0 / 미확인 −1. 기본 −1 |
| `fallback_ha` | N, 선택 | 5fps 추론 불가 구간의 명시적 H/A 대안 0/1 |

좌표는 105×68m로 환산한다. 서로 다른 트랙을 임의로 연결하지 않는다. `half=0` 프레임은 D여야 한다. D CSV는 `frame,dead`(0/1) 또는 `frame,label`(H/A/D)이며 영상 축과 같아야 한다. 볼 결측 자체를 D로 처리하지 않는다.

같은 D 조건의 기존 5fps 캐시가 있으면 `--base-cache match_pred.npz`로 재사용할 수 있다. 캐시 키는 `h1_frame/base/margin/carrier`, `h2_frame/base/margin/carrier`다. 5fps가 새 입력의 인플레이를 처리하지 못하면 명시적 `fallback_ha` 또는 정렬된 캐시를 제공해야 한다.

## 채택 파라미터

| 항목 | 값 |
|---|---|
| 5fps | 공개 PathCRF trial 140, 10초 창, 고정 가중치 |
| 공개 코드 revision | `1b7c6f85a6b78c92e4543fde72ca2b86a933f997` |
| 25fps | 128차원 / 4헤드 / 블록 2 / 블록당 양방향 시간 attention 2층 |
| 입력 창 | 4초·100프레임 / 추론 간격 25프레임 |
| 채택 가중치 | 29에폭. 학습 seed 100, 배치 64, 최대 40/얼리스탑 5 |
| 후보 | 불확실도 0.5 / 상대 거리 2.5m / 0.2초 접근량 0.65m / 전후 15프레임 |
| 점수 온도 | 5fps 2.0 / 25fps 0.5 |
| 중앙 혼합 | 5fps:25fps = 1:1 |
| 가장자리 완화 | 10프레임(0.4초) |
| 교체 최소 이득 | 변경 프레임당 0.1 |
| 짧은 점유 쌍 병합 | 병합 대안 대비 프레임당 이득 0.15 미만 |
| 마지막 경계 스냅 | 관측 터치 후보 ±10프레임(0.4초) |

후보 밖은 5fps 예측을 유지하고 D 마스크는 고정한다. 추론 중 학습이나 파라미터 탐색은 하지 않는다. 상세 설정과 해시는 `parameters.json`이 기준이다.

## 기록된 결과 범위

학습 5경기, 검증 J03WQQ, 평가 J03WR9다. 보정은 검증에서 선택했다. J03WR9는 이미 확인한 홀드아웃의 추가 진단이며 **76.22%는 대회 제출 점수가 아니다.** D 모델을 바꾸면 평가 조건도 바뀐다.

| 지표 | 값 |
|---|---:|
| Spell F1 | 76.22% |
| 가짜 / 누락 H/A 전환 | 48 / 112 |
| 전환 시점 MAE | 140.0ms |
| 2초 미만 점유 검출 | 28/94 |

가중치는 로컬에 포함되어 있다. `.gitignore`는 가중치와 향후 출력·캐시의 Git 추적을 제외한다.
