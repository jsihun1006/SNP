# 2단계: 시계열 Window Dataset·분할·자기지도 마스킹

작성일: 2026-11-09. 선수 전용 데이터 계층을 구현했다. **Graph Encoder, Feature Encoder, Bi-LSTM, fusion, neural loss 및 학습 코드는 구현하거나 실행하지 않았다.** 원시/파생 경기 데이터는 로컬에만 보관했고 공개 GT·Smart Tagging·볼·속도·broadcast를 입력으로 사용하지 않았다. GitHub push도 하지 않았다.

## 1. 이전 단계 검토와 범위

[연구 결과](CODEX_RESEARCH_OUTPUT.md), [구현 계획](MODEL_IMPLEMENTATION_PLAN.md), [1단계 실행 기록](PHASE1_DATA_RELATIONS.md) 및 현재 Python 코드를 검토했다. 개발 시작 시 기존 합성 pytest **66개를 재실행해 모두 통과**했다. 이전 단계의 실패 테스트는 없었다.

이전 단계에서 남은 데이터 문제는 물리적으로 해결했다고 주장하지 않는다. 팀 오분류·ID 교체·실제 중복 여부는 확인할 수 없고, 기존 strict quality 기준에서는 평가 구간 약29.7%의 경합 후보 지수가 보류된다. 이번에는 이 관측들을 시간축에서 지우지 않고 observation-valid/quality/loss mask로 처리한다. 수작업 수정이나 GT 기반 기준 조정은 하지 않았다.

기존 Python 파일은 수정하지 않았다. 기존 경합 후보 지표·공식 baseline·`starter/`·`evaluation/`·`games.py`와 원본 ATD, 1단계 산출물을 보존했다. 새 코드와 데이터는 독립된 경로에 추가했다.

## 2. 신규 Python 파일

모듈은 모두 `MTAH_Relational_Starter/mtah_relational/`에 있다.

| 파일 | 주요 기능 |
|---|---|
| [frame_store.py](../MTAH_Relational_Starter/mtah_relational/frame_store.py) | read-only `FrameStore`, strict ATD→로컬 raw cache, 원본 프레임 ID/period/자연 결측 사유 보존 |
| [splits.py](../MTAH_Relational_Starter/mtah_relational/splits.py) | 반별 연속 Train/Validation/Test 블록, manifest fingerprint, 원시 support 중복 감사 |
| [corruption.py](../MTAH_Relational_Starter/mtah_relational/corruption.py) | seed/epoch/routing만 사용하는 연속8–13프레임 `SpanMasker` |
| [normalization.py](../MTAH_Relational_Starter/mtah_relational/normalization.py) | 고유 Train 관측만으로 median imputer/IQR scaler fit, 저장/재사용 |
| [windows.py](../MTAH_Relational_Starter/mtah_relational/windows.py) | 151프레임 index·Dataset, 학습/추론 구분, 별도 loss targets, batch collator, index 저장·검사 |
| [temporal_cli.py](../MTAH_Relational_Starter/mtah_relational/temporal_cli.py) | 로컬 준비·감사 CLI, `open_temporal_dataset` 재로딩 API |

신규 합성 테스트 파일:

* [test_temporal_windows.py](../MTAH_Relational_Starter/tests/test_temporal_windows.py)
* [test_temporal_cache_and_pipeline.py](../MTAH_Relational_Starter/tests/test_temporal_cache_and_pipeline.py)

기존 Python 수정은 **0개**다. 기존 단계에서 구현한 `preprocessing.spatial_input`, geometry, schema, quality policy를 호출한다. CLI도 기존 `cli.py`와 분리했다.

## 3. Window와 원본 시간축

기본은25fps, 길이151, 중심 index75, 중심 전후75프레임이다. source slot은 현재 관측 컬럼일 뿐이며 시간적으로 동일한 인물로 연결하지 않는다. 각 시점 raw stream은 가변 `N_t` 선수 집합이다.

### Reconstruction용

`mode='reconstruction'`는 Train/Validation/Test 모두에서 복원 평가용 **완전한151프레임 context**를 생성한다. validation/test도 clean input 추론이 아니라 고정된 마스크의 reconstruction 검증 데이터다.

* 중심 c의 실제 raw support는 `[c−75,c+75]`다.
* 모든 support가 한 반·한 split block·한 contiguous record run 안에 있어야 한다.
* quality가 낮거나 선수 모두가 결측인 record도 기록 자체가 존재하면 시간축에 남긴다.
* record ID가 건너뛰면 run을 나눠 그 gap을 가로지르는 window는 생성하지 않는다.
* raw cache 생성은 이전 단계의 엄격한 전체 ATD 검증을 사용하므로 누락/중복 record가 있는 파일은 기본적으로 거부한다. 별도 `FrameStore`로 제공된 불연속 기록은 Dataset에서 run별 처리하며 합성 테스트로 확인했다. 원본 record를 임의 보간/복구하지 않는다.

### 실제 추론용

`mode='inference'`에는 artificial mask와 clean target가 없다. 해당 반/기록 run 안의 context만 사용하고 양끝 부족분을 패딩한다. 중심 위치는 항상75이며 패딩의 `frame_ids=-1`, `padding_mask=True`다. 이상적인 요청 번호는 별도 `requested_frame_ids`에 남기며 둘 모두 **routing 정보이고 model feature가 아니다**.

* split을 지정한 inference는 그 split 안에서만 context를 읽는다.
* `split=None`인 deployment inference는 전후반 전체를 사용한다. 반 내부 Train/Validation 경계는 넘어갈 수 있으므로 **holdout 성능 평가에 사용하면 안 된다.**
* CLI의 deployment index는 training stride와 별개로 stride1이다.
* non-graded period0에는 inference center를 만들지 않는다. 이후 최종 제출 단계에서 전체199,319행과 X 처리 계약을 따로 적용해야 한다.

## 4. 시간 블록 분할과 누출 감사

각 반을 원시 프레임 기준 **70%/15%/나머지15%**의 연속 블록으로 나눈다. 정수 길이는 앞 두 비율의 floor, 나머지는 Test다. 이 비율은 GT나 경기 장면을 보지 않고 정한 초기 설정이다. window를 만든 뒤 무작위로 나누지 않는다.

실제 manifest:

| 반 | split | raw 시작 | raw 끝 |
|---|---|---:|---:|
| 1 | Train | 11185 | 66810 |
| 1 | Validation | 66811 | 78730 |
| 1 | Test | 78731 | 90651 |
| 2 | Train | 110808 | 166380 |
| 2 | Validation | 166381 | 178288 |
| 2 | Test | 178289 | 190197 |

window 중심은 각 블록 가장자리75프레임 안쪽에서 시작/끝난다. 따라서 서로 다른 split에서 가장 가까운 중심은 최소151프레임 떨어지며, input과 target가 참조하는 **실제 raw support의 교집합이0**이다. 같은 split 안의 겹치는 window는 허용한다.

`assert_disjoint_support`는 중심 번호만 비교하지 않고 각 window의 support·block role·반·길이를 검증하고 union을 비교한다. `audit_window_index`는 모든 index의 row→원본 frame 대응과 contiguous run을 검사한다. 잘못된 Train index를 Validation 파일/role로 섞으면 로딩 단계에서 거부한다.

현재 전처리는 프레임 독립이며 temporal dependency 반경 R=0이다. 향후 차분·스무딩을 추가하면 support를 R만큼 넓혀야 한다. 지금의75 경계 규칙을 그대로 두고 미래 파생 특징을 추가해서는 안 된다.

## 5. Raw-first joint masking

`SpanMasker`는 원시 좌표·관측 품질·타깃 가용성을 보지 않고 window 안의 연속8–13프레임을 선택한다. seed2027이 기본이며 NumPy SeedSequence로 center/period/split/epoch를 조합한다. worker 접근 순서나 Python hash에 의존하지 않는다.

* Train은 epoch에 따라 마스크를 바꿀 수 있다.
* Validation/Test는 epoch와 무관하게 고정 마스크를 쓴다.
* hidden 시점은 `source.frame()` 호출 전에 차단한다.
* hidden raw node/team 집합은 빈 배열, engineered 값·validity·quality·ordered H/A 구조는0이다.
* 따라서 clean 선수 수, 원래 결측 양상, duplicate/OOB flags, 팀별 구조가 숨긴 시점 입력으로 남지 않는다.
* 실제 Graph edge는 아직 만들지 않았다. 후속 graph builder는 이 corrupted raw node 집합만 받아야 한다. clean topology cache를 추가하면 안 된다.
* 과거/미래 delta, 스무딩, 보간, clean pooled embedding은 입력으로 받지 않는다.

### 세 가지 상태 구분

| 상태 | artificial | padding | 입력 feature-valid | loss target |
|---|---:|---:|---|---|
| 자연 결측 record | False | False | 관측 가능한 항목만 True | 이 시점이 숨겨졌을 때만 관측 가능한 항목 사용 |
| 인위적 hidden frame | True | False | 모두 False | clean observed/quality가 허용할 때만 사용 |
| 경계 패딩 | False | True | 모두 False | 없음 |

hidden frame의 **원래 자연 결측 여부를 모델에 알려주지 않는다.** 그 원래 관측 가능성은 loss-only target의 `observed`에만 존재한다. 인위적으로 숨긴 사실과 자연 결측은 입력의 artificial bit로 구분한다.

## 6. Clean Target 분리와 Shape

`dataset[i]`는 `InputWindow(inputs, routing)`만 반환한다. 타깃은 `dataset.get_targets(i, window=window)`에서 별도로 얻는다. target dict를 input object에 넣거나 model forward에 전달하지 않는다. 이미 만들어진 window의 epoch snapshot을 사용하므로 이후 `set_epoch`이 호출돼도 해당 sample의 마스크를 정확히 재생할 수 있다.

| 항목 | 단일 window | batch |
|---|---|---|
| `inputs.values` | `[151,24]` | `[B,151,24]` |
| `inputs.feature_valid` | bool `[151,24]` | `[B,151,24]` |
| `inputs.quality` | `[151,4]` | `[B,151,4]` |
| `inputs.quality_valid` | bool `[151]`, visible-only | `[B,151]` |
| `inputs.team_structure` | `[151,2,10]`, H/A 순서 | `[B,151,2,10]` |
| `inputs.team_structure_valid` | bool `[151,2,10]` | `[B,151,2,10]` |
| `inputs.raw_xy_m`, `raw_team` | 길이151의 tuple: `[N_t,2]`, `[N_t]` | B개의 가변 길이 tuple |
| `artificial_mask`, `padding_mask` | bool `[151]` | `[B,151]` |
| `inputs.dense_features()` | `[151,53]` | `[B,151,53]` |
| `routing.frame_ids` | int64 `[151]`, padded=-1 | 모델과 분리된 routing tuple |
| `targets.values`, `observed`, `quality_weight`, `loss_mask` | `[151,13]` | `[B,151,13]` |

53 channels는 `scaled values24 + feature_valid24 + quality4 + artificial bit1`이다. strict quality-valid와 padding은 별도 보조 마스크이며 53 channels에 자동 추가하지 않는다. H/A 개별 구조는 대칭 feature로만 축약하지 않고 별도 경로로 보존한다. team structure는 이 단계에서 학습 embedding이 아니라 원래 물리 단위의 통계량이다.

복원 target은 schema의 첫13개 항목이다. hidden 시점의 clean summary만 계산하며, visible 시점은 target=0/observed=False로 둔다. `loss_mask = artificial & clean_observed & quality_weight>0`이며 quality_weight는 현재 strict policy의 0/1이다. 자연 결측을 보간해 진짜 정답으로 취급하지 않는다. 모델 손실이나 optimizer는 아직 없다.

## 7. Train-only scaler/imputer

`fit_train_normalizer`는 호출자가 지정한 임의 fold를 받지 않고 manifest의 Train raw frames만 선택한다. 겹치는 window마다 같은 frame을 반복 집계하지 않고 고유 관측으로 fit한다. strict quality-valid이며 feature-valid인 값에서 median과 IQR을 계산한다.

* 결측 imputer는 Train median이다. 표준화 후 결측값은0이며 valid mask는 False를 유지한다.
* IQR=0이면 scale1을 쓴다. Train에서 관측0인 feature는 median0/scale1 기본값을 명시하고 loss 대상에서 제외한다.
* validation/test/deployment는 저장된 동일 파라미터로 transform만 하며 다시 fit하지 않는다.
* 노드 좌표와 ordered team structure는 물리 단위를 보존한다. 품질 비율에도 추가 학습 scaler를 쓰지 않는다.
* artifact의 split manifest·policy·pitch·feature schema 정합성을 검사한다.

Train 전체 분포 통계를 먼저 고정하는 것과 각 hidden sample을 이용해 window 내부 평균/분산을 다시 계산하는 것은 다르다. 이번 hidden-sentinel 검증은 **학습된 scaler를 고정**하고 입력 관측만 바꾼다. sample 조회 중 fit/refit하는 경로는 없다. Holdout 좌표를 바꿔도 Train median/scale/count가 변하지 않는 것도 별도로 테스트했다.

## 8. 실행과 재사용

프로젝트 루트의 `mtah2027` 환경에서 실행한다:

```bash
export PYTHONPATH="$PWD/MTAH_Relational_Starter:$PWD"
python -m mtah_relational.temporal_cli build \
  --out-dir data/derived/relational_temporal_v1
```

출력은 새 디렉터리에만 저장한다. 기존 1단계 CSV를 덮어쓰지 않는다. raw cache·index·scaler도 실제 경기 파생물이므로 로컬 전용이다.

이번 실제 데이터 결과는 `data/derived/relational_temporal_v1_20261109/`에 있다. fit 없이 다시 사용할 수 있다:

```python
from mtah_relational.temporal_cli import open_temporal_dataset
from mtah_relational.windows import collate_inputs, collate_targets

root = "data/derived/relational_temporal_v1_20261109"
train = open_temporal_dataset(root, split="train")
train.set_epoch(0)
windows = [train[0], train[1]]
inputs, routing = collate_inputs(windows)
targets = collate_targets([train.get_targets(i, window=w)
                           for i, w in enumerate(windows)])
# inputs만 후속 모델에 전달한다. targets는 loss 쪽에서만 쓴다.
assert inputs.dense_features().shape == (2, 151, 53)

validation = open_temporal_dataset(root, split="validation")
deployment = open_temporal_dataset(root, mode="inference", split=None)
# deployment.get_targets(...)는 허용하지 않는다. Holdout 평가에 deployment를 쓰지 않는다.
```

로컬 출력 파일:

* `raw_cache/`: 물리 단위 xy·원본 frame IDs·period·parse flags·team metadata. read-only memmap으로 연다.
* `split_manifest.json`: 반별 시계열 분할과 fingerprint.
* `train/validation/test_windows.npy`와 각각의 `.json`: 원본 frame support가 포함된 index와 role/spec metadata.
* `deployment_windows.npy/.json`: 공식 frame마다 대응하는 padding 가능한 index.
* `train_normalizer.json`: Train 전용 median/IQR/imputer와 fit provenance.
* `dataset_config.json`, `temporal_report.json`, `LOCAL_ONLY.txt`。

모든 window를 복제해 저장하지 않고 raw cache와 index를 분리한다. window에 접근할 때 corrupted 관측에서 입력을 다시 계산한다.

## 9. 테스트 결과

합성 pytest **98개 통과**: 기존66개 + 신규32개. Conda 환경에 pytest가 설치되지 않은 상태여서 이전처럼 기존 base의 pure-Python pytest9.0.3을 해당 프로세스에서 불러왔다. 실제 실행 interpreter는 mtah2027의 Python3.11.17이며 환경에 패키지를 설치하지 않았다.

검증한 항목:

* Window151/중심75/원본 frame IDs, half·period0·record gap 경계, 자연 결측 record 보존.
* raw 참조 범위에 기반한 Train/Validation/Test 분리, 다른 split의 index 혼입 거부.
* 동일 seed의 연속8–13 mask, Train epoch 갱신, Validation/Test 고정 mask.
* hidden 좌표·관측 수·품질을 바꿔도 모든 model input이 동일함. hidden raw access 자체를 금지한 감사.
* 자연 결측/artificial/padding 구분, hidden 시점 H/A 구조·quality 차단.
* target의 input/batch 혼입 방지, 별도 collator, epoch snapshot의 target 재생.
* Holdout 전체 좌표를 바꿔도 Train normalizer 불변, 저장/로드 시 split 불일치 거부.
* read-only raw cache, 손상된 strict record 거부, partial 출력 정리와 artifact 재로딩.

일반적인 로컬 테스트 명령(pytest 설치 후):

```bash
PYTHONDONTWRITEBYTECODE=1 PYTHONPATH="$PWD/MTAH_Relational_Starter:$PWD" \
python -m pytest -q -p no:cacheprovider MTAH_Relational_Starter/tests
```

이 Mac에서 사용한 임시 runner는 [1단계 환경 설명](PHASE1_DATA_RELATIONS.md)과 같다. 필요하면 사용자 로컬 환경에서 `python -m pip install "pytest>=8,<10"`을 실행한다. 이번 단계에는 PyTorch/PyG가 필요하지 않다.

## 10. 실제 경기 데이터 로컬 검증

원시199,319개 record를 player-only cache에 저장하고 모든 index의 원본 frame 대응·support·반/run 경계를 검사했다.

| index | window 수 | padding window |
|---|---:|---:|
| Train reconstruction | 110,899 | 0 |
| Validation reconstruction | 23,528 | 0 |
| Test reconstruction | 23,530 | 0 |
| 전체 반 deployment inference | 158,857 | 300 |

Train/Validation/Test input+target raw support 교집합은 **0프레임**이다. scaler는 Train111,199개 고유 raw frame 중 strict quality가 유효한 **80,538개**에서 fit했다. Holdout을 fit에 사용하지 않았다.

모든 window의 feature tensor를 생성한 것은 아니다. **전체157,957개 reconstruction index와158,857개 inference index를 감사**했고, 각 split/반의 first·middle·last에서 총18개 reconstruction window를 실제로 구성했다. 별도로4개의 inference 반 경계 sample을 확인했다. 각 split에서 실제 raw를 메모리상의 sentinel로 바꾼 테스트와 hidden raw-read guard도 통과했다. 원본/cache를 수작업으로 수정한 것이 아니다.

| 실제 materialized sample | 수 | 유효target cell | 유효target 없는sample |
|---|---:|---:|---:|
| Train | 6 | 455 | 3 |
| Validation | 6 | 546 | 2 |
| Test | 6 | 572 | 2 |

**18개 중7개에서 loss 가중치가 모두0**이다. 자연 결측/품질 정책에 따라 사용 가능한 hidden target가 없는 경우이며 정답을 만들거나 마스크를 다시 고르지 않았다. 이 first/middle/last 표본은 전체 window의 무작위 표본이 아니므로7/18을 전체 비율로 해석하지 않는다. 다음 A모델은 zero-loss batch를 건너뛰고 count를 기록해야 한다.

현재 실패한 테스트나 index 검사 항목은 없다. 실제 경합 정확도·확률 보정·H/A/D spellF1은 검증하지 않았다. 단일 경기만으로 다른 경기에 대한 일반화도 검증할 수 없다.

## 11. 다음 A모델 준비 상태

데이터 계층은 A모델의 `[B,151,53]` 입력과 별도 `[B,151,13]` 복원 target/loss mask를 제공한다. H/A 개별 구조와 raw node stream도 보존해 B/C모델에 같은 split·mask·target를 적용할 수 있다.

다음 구현은 Spatial Feature Encoder + 소형 Bi-LSTM + masked loss다. 보간/persistence 대조군·zero-loss batch 처리·seed 기록이 필요하다. 추론 시 양쪽 padding을 RNN에서 어떻게 처리하는지는 A단계에서 구현·시험해야 한다. 현재는 데이터 수준의 padding mask만 제공한다.

아직 미구현/미검증: Graph edge builder와 GNN, Bi-LSTM forward/backward, fusion, optimizer, 학습 수렴, 전체 window materialization, 멀티프로세스 DataLoader 동작, MPS/GPU 호환성. 시간에 의존하는 전처리 특징을 추가하면 support와 hidden-sentinel 검사를 다시 확장해야 한다. 경합 후보를 실제 경합 확률이라고 부르지 않는다.
