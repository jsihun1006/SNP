# 4단계: Spatial Feature Encoder + 자기지도 Bi-LSTM (A 모델)

작성일: 2026-11-09. **A 모델 구현, 합성 검증, 실제 경기의 로컬 MPS 학습·복원 평가·임베딩 추출을 완료했다. 이번 실험에서 선형 보간의 복원 오차가 A 모델보다 낮았다.** 복원 성능은 경합 감지나 H/A/D 점유 판단 성능을 검증하는 지표가 아니다.

후속 [A 모델 단회 진단](A_MODEL_DIAGNOSTIC.md)에서 token·추가 학습·혼합 mask·작은 Temporal과 표현을 Validation만으로 비교했다. 기존 A12는 동결 유지하고 실험 체크포인트를 별도로 보관했다. 아래 수치와116개 테스트 기록은 최초 A 실행 당시 결과다. 후속 테스트는125개 통과했다.

## 1. 이전 단계 검토와 명세 차이

[연구 결과](CODEX_RESEARCH_OUTPUT.md), [구현 계획](MODEL_IMPLEMENTATION_PLAN.md), [기존 시계열 구현 기록](PHASE2_TEMPORAL_DATA.md), 실제 Python 인터페이스를 검토했다. 이전 98개 합성 테스트를 현재 `mtah2027` 환경에서 재실행하여 모두 통과했다. 기존 코드의 실패 테스트는 발견하지 않았다. 과거 문서의 단계 번호와 사용자의 단계 번호는 다르다. Window Dataset은 기존 `PHASE2_TEMPORAL_DATA.md`에 기록되어 있고, 이번 단계는 사용자의 번호에 따라 4단계라고 부른다.

* 실제 입력은 명세와 동일한 **53차원**이다. 차원을 바꾸지 않았다.
* 이전 계획의 A 모델 추정치 **38,029개**에는 별도 학습 가능한 feature mask token 64개가 포함된다. 이번 요청의 정확한 레이어에는 이 token이 없어 **37,965개**다. hidden input은 기존 Dataset이 제공하는 `values/validity/quality=0, artificial_bit=1`이며 같은 Spatial Encoder를 통과한다.
* 초기 환경에는 PyTorch와 pytest가 없었다. 기존 Conda 환경에 `torch 2.15.0`, `pytest 9.2.0`을 설치하고 검증했다. 격리된 실행에서는 MPS가 감지되지 않았지만 GPU 접근이 가능한 로컬 실행에서는 MPS를 사용할 수 있었다.
* 기존 데이터의 팀 오분류, slot 교체, 결측, 중복 의심, 공식 후반 시작 구간의 관측 품질 문제를 물리적으로 해결했다고 주장하지 않는다. 원시 데이터를 수정하거나 Smart Tagging으로 후반 경계를 바꾸지 않았다. 동일한 공식 구간·quality 정책과 loss eligibility를 유지했다.

## 2. 신규 파일 및 보존 범위

Python 파일은 모두 `MTAH_Relational_Starter/` 아래에 추가했다.

| 파일 | 역할 |
|---|---|
| [model_a.py](../MTAH_Relational_Starter/mtah_relational/model_a.py) | `SpatialBiLSTM`, 로컬 체크포인트 저장·안전한 재로드 |
| [model_a_data.py](../MTAH_Relational_Starter/mtah_relational/model_a_data.py) | frame-local 특징 캐시, `FeatureWindowDataset`, 분리된 타깃, 배치 구성 |
| [reconstruction.py](../MTAH_Relational_Starter/mtah_relational/reconstruction.py) | 유효성·품질·feature 가중 Huber와 특징별 MAE 집계 |
| [reconstruction_baselines.py](../MTAH_Relational_Starter/mtah_relational/reconstruction_baselines.py) | corrupted input만 읽는 persistence·선형 보간 |
| [train_model_a.py](../MTAH_Relational_Starter/mtah_relational/train_model_a.py) | seed, CPU/MPS 검사, 학습·early stopping·동일 조건 평가·임베딩 추출 |
| [plot_model_a.py](../MTAH_Relational_Starter/mtah_relational/plot_model_a.py) | 집계 결과만 읽는 서버 없는 PNG 그래프 |
| [test_model_a.py](../MTAH_Relational_Starter/tests/test_model_a.py) | 신규 합성 테스트 및 전체 학습 파이프라인 검증 |

별도 설정은 [model_a_small.json](../MTAH_Relational_Starter/configs/model_a_small.json), 의존성은 [requirements-model-a.txt](../MTAH_Relational_Starter/requirements-model-a.txt)다. 이 실행 기록과 패키지 README·현재 구현 계획·연구 문서의 후속 링크도 갱신했다.

기존 `starter/`, `evaluation/`, `games.py`, 전처리·Rule-based 지수·Window Dataset Python은 변경하지 않았다. 시작 시 보존 목록 98개 파일의 SHA256과 기존 원시/파생 데이터 30개 파일의 size·mtime을 마지막에 대조해 동일함을 확인했다. 이전 notebook 변경 사항도 그대로 보존한다. GitHub push/commit, 외부 업로드, 로컬 웹 서버 실행은 하지 않았다. Smart Tagging, 공개 GT, 볼, speed, broadcast는 모델 입력·타깃·선택에 사용하지 않는다.

## 3. 입력·출력 및 모델

| 경로 | Shape / 내용 |
|---|---|
| `dense[..., :24]` | `[B,151,24]`: Train median/IQR 정규화된 공간 특징 |
| `dense[..., 24:48]` | `[B,151,24]`: 해당 특징의 관측/정의 가능성 |
| `dense[..., 48:52]` | `[B,151,4]`: 이전 단계가 정의한 관측 품질 특징 |
| `dense[..., 52]` | `[B,151]`: artificial mask indicator |
| `padding_mask` | bool `[B,151]`: 경계 패딩, 모델 forward의 별도 인자 |
| Spatial Encoder | Linear53→64, LayerNorm, ELU, Linear64→64, LayerNorm, ELU |
| Temporal Encoder | Bi-LSTM64→32×2, 1층, `[B,151,64]` |
| Reconstruction Head | Linear64→64, ELU, Linear64→13 |
| 모델 반환값 | `(prediction[B,151,13], embedding[B,151,64])` |
| Loss-only 타깃 | `values/observed/quality_weight/loss_mask`: `[B,151,13]`, artificial `[B,151]` |
| 배포 중심 임베딩 | `[158857,64]`, 원본 frame 번호 배열 `[158857]`, 중심 관측 quality `[158857]` |

공간 특징 24개는 기존 `spatial_schema.FEATURE_NAMES` 순서를 그대로 사용한다. **처음 13개가 복원 타깃**이다.

```text
cross_density_3m, cross_density_5m, cross_density_10m,
cross_min, cross_nearest_mean, cross_cluster_fraction, centroid_gap,
width_sum, width_abs_diff, length_sum, length_abs_diff, radius_sum, radius_abs_diff,
team_mix, cross_pairs_3m, cross_pairs_5m, cross_pairs_10m, cross_cluster_count,
centroid_mid_x, centroid_mid_y, bbox_area_sum, bbox_area_abs_diff,
within_nearest_sum, within_nearest_abs_diff
```

53개 입력의 공간 특징은 H/A 교환에 대칭이다. **A 모델의 embedding만으로 어느 팀이 점유했는지 알 수 없다.** 이전 단계의 ordered H/A 구조 `[151,2,10]`, 원시 선수 집합과 quality는 기존 `WindowDataset`에서 계속 얻을 수 있고 삭제하지 않았다. A 모델에는 요청한 53개만 전달한다. Track slot을 개인 ID로 연결하지 않으며 선수별 LSTM은 사용하지 않는다.

자연 결측은 timestep padding이 아니다. artificial hidden frame도 시퀀스에 남는다. 실제 경계 padding은 recurrent context에서 제외한다. 패딩을 포함한 배치는 각 샘플의 연속된 실제 구간에만 LSTM을 실행하고 padding 출력은0으로 둔다. 내부 기록 gap을 padding으로 연결하려 하면 거부한다. 미래75프레임을 사용하는 **오프라인 모델**이다.

## 4. 마스킹, 캐시와 누출 방지

기존 `open_temporal_dataset()`과 `WindowDataset`을 기준 구현으로 유지한다. 이를 매 학습 window마다 그대로 호출하면 151개 프레임의 동일한 공간 기하를 반복 계산한다. A 전용 adapter는 **프레임 독립인 현재 전처리만** 캐시한다. fit이나 시계열 차분·스무딩·보간은 이 캐시에 없다.

1. 기존 routing/seed/epoch로 숨길 원시 시점을 먼저 정한다.
2. 입력을0으로 초기화하고 **visible row만** input value·validity·quality 캐시에서 읽는다. hidden row는 읽지 않으며 artificial bit만1이다.
3. 원래 quality나 자연 결측 양상도 hidden input에 남기지 않는다.
4. clean target 캐시는 별도 `get_targets(item, window=...)`에서만 접근한다. `AInputWindow`와 모델 forward에는 타깃 필드가 없다.
5. 캐시 source/scaler/policy fingerprint를 검증한다. 실제 실행 전 split별 first/middle/last 9개 window의 입력과 타깃 전체 배열이 원래 Dataset과 정확히 같은지 검사했다. 입력 float32 반올림과 loss-only clean target의 원래 정밀도 차이까지 맞췄다.
6. 숨긴 원시 좌표를 sentinel로 바꾸고 캐시까지 다시 만든 합성 테스트에서 corrupted input은 정확히 동일했다. 기존 hidden raw read guard와 Train-only fitting 테스트도 그대로 통과한다.

이 최적화는 **현재의 frame-local 함수에만** 성립한다. 이후 속도·가속도·시간 미분·스무딩을 넣으면 이 캐시 계약을 다시 설계해야 한다. frozen Train 통계는 모든 split에서 재사용하되 새로 fitting하지 않는다.

Train은 매 epoch 마스크를 바꾸고, Validation/Test 마스크는 고정한다. 세 비교 방법은 동일한 중심 프레임·마스크·loss eligibility를 공유한다. 25·50프레임은 최종 checkpoint에서 평가만 하며, 해당 길이를 학습하거나 선택 기준으로 사용하지 않는다.

## 5. Loss와 비교 방법

```text
eligible = artificial & clean_observed & finite(target) & finite(quality_weight)
           & (quality_weight > 0) & (feature_weight > 0)
weighted_huber = Σ eligible * quality_weight * feature_weight * Huber(pred,target)
                 / Σ eligible * quality_weight * feature_weight
```

기본 Huber delta=1, feature별 weight=1이다. 현재 quality weight는 기존 strict policy의0/1이다. undefined target·자연 결측을 정답0으로 만들지 않고 제외한다. NaN/Inf는 마스킹 곱셈 **전에** sanitize한다. eligible target이 없으면 loss=None, backward/optimizer update를 건너뛴다. epoch 전체가 비어도 train loss를 `null`로 기록할 수 있다. 유효한 Validation 타깃이 전혀 없으면 선택 가능한 objective가 없으므로 명확한 오류로 중단한다.

* **Persistence:** 각 feature에서 mask 이전의 마지막 관측을 유지한다. 이전 anchor가 없으면 이후 첫 관측을 사용한다.
* **Interpolation:** mask 이전·이후의 관측 anchor 사이를 선형 보간한다. 한쪽만 있으면 그 값을 유지한다.
* **두 기준 모두 anchor가 없으면:** frozen Train median에 해당하는 표준화 값0을 사용한다. 타깃을 버리지 않고 fallback 수를 센다. 이번 실제 평가에서 이 fallback에 해당한 유효 target cell은0개였다.
* **A 모델:** corrupted 53차원 입력만 받아 복원한다. head의 출력은 제약 없는 연속값이며 확률이 아니다.

전체 MAE/Huber는 모든 유효 cell의 품질·feature weight로 가중 집계한다. feature별 MAE/Huber도 별도로 저장한다. `mae_standardized`는 median/IQR 단위, `mae_schema_units`는 Train scale을 다시 곱한 기존 기하 스키마 단위다. **후자는 미터 단위가 아니다.** 기존 `symmetric_summary()`는 거리를 pitch diagonal, 폭/길이를 각 축, 팀 합을 평균 비율 등으로 정규화한다. 임의의 다른 물리 단위 MAE를 합산하지 않았다.

Huber와 MPS backend, checkpoint 로딩 API는 PyTorch 공식 문서를 확인했다: [Huber](https://docs.pytorch.org/docs/2.15/generated/torch.nn.functional.huber_loss.html), [MPS](https://docs.pytorch.org/docs/2.15/notes/mps.html), [torch.load](https://docs.pytorch.org/docs/2.15/generated/torch.load.html). 체크포인트는 tensor와 primitive만 저장하고 `weights_only=True`로 로컬 재로드한다.

## 6. 실제 데이터 분할과 학습 설정

원시 ATD199,319개 record,25fps와 공식 전후반을 유지했다. **아래 frame은 0-based**이며 ATD의1-based 번호에서1을 뺀 값이다. 151프레임 window는 각 블록 안에 완전히 들어가는 중심만 사용한다.

| Split | 전반 raw support (양끝 포함) | 후반 raw support (양끝 포함) | 전체 가능 window | 이번 사용 window |
|---|---|---|---:|---:|
| Train | 11185–66810 | 110808–166380 | 110,899 | 22,181 |
| Validation | 66811–78730 | 166381–178288 | 23,528 | 942 |
| Test | 78731–90651 | 178289–190197 | 23,530 | 942 |

split 간 원시 input+target support 교집합은 **0프레임**이다. scaler/imputer는 이전 단계에서 Train 고유111,199프레임 중 strict quality 유효80,538개로 fit한 것을 그대로 재사용했다. GT, 볼, 태그를 사용한 선택은 없다.

첫 소형 실험으로 결과 확인 전에 Train 중심5프레임 간격, Validation/Test25프레임 간격을 고정했다. **전체 window 학습/평가가 아니다.** 각 window 내부는25fps의 연속151프레임이다. 표본 수를 줄였으며, 다른 split 사이의 정보를 섞거나 window 내부를1fps로 줄인 것이 아니다. 같은 split 안의 window는 중첩되므로 개별 window를 독립 표본으로 해석하지 않는다.

seed2027, Adam lr0.001, batch64, 최대12 epoch, gradient clipping1, patience3, min_delta0.0001, Huber delta1을 사용한다. 선택 기준은 **고정 8–13프레임 마스크의 Validation Huber만**이다. Test와 긴 마스크 결과는 best checkpoint 선택 뒤 계산한다. seed와 deterministic-algorithms를 설정하되 `warn_only=True`이므로 다른 장비/버전 간 bitwise 재현성을 보장하지 않는다.

MPS는 실제 모델 forward/backward/Adam update와 padded inference를 합성 preflight로 검사한다. MPS가 없거나 preflight가 지원되지 않으면 CPU로 전환하고 이유를 기록한다. `PYTORCH_ENABLE_MPS_FALLBACK=1`도 설정하므로 일부 연산의 CPU fallback이 가능하다. 전체 연산이 모두 GPU에서 실행됐다고 주장하지 않는다.

## 7. 실제 학습·평가 결과

실제 환경: `macOS-26.6.2-arm64-arm-64bit`, Conda `mtah2027`, Python3.11.17, PyTorch2.15.0, NumPy2.4.6, **MPS**. MPS preflight 통과, fallback 사유 없음(연산별 fallback 허용). CPU thread4. parameter **37,965개**.

총12 epoch, best epoch **12**, 마지막 Train Huber **0.124708**, best Validation Huber **0.142863**. 첫 epoch Train/Validation은 0.407101/0.260382였다. 학습 루프+epoch Validation+checkpoint 저장 시간은 **91.41초**이며 캐시 생성·최종 평가·전체 임베딩 추출 시간은 제외한다. epoch별 시간은 7.57–7.69초다.

매 epoch optimizer update347회, 전체 4,164회. 이번 실제 실행의 전체 batch 단위 skip은 0회다. 이는 개별 zero-target window가 없다는 뜻은 아니다. early stopping을 구현했지만 이번에는 patience에 도달하기 전에 최대12 epoch까지 Validation이 개선되어 종료했다. 따라서 학습 수렴을 확인한 결과는 아니다.

정규화 MAE/Huber는 낮을수록 좋다. 세 방법은 같은 유효 타깃 cell을 평가했다.

### Validation 비교

| Mask 프레임 | Persistence MAE | Interpolation MAE | A MAE | Persistence Huber | Interpolation Huber | A Huber | 유효 cell 수 |
|---|---:|---:|---:|---:|---:|---:|---:|
| 8-13 | 0.241091 | 0.151281 | 0.273364 | 0.179774 | 0.113039 | 0.142863 | 94,289 |
| 25 | 0.427465 | 0.276590 | 0.663803 | 0.323682 | 0.208054 | 0.474236 | 218,439 |
| 50 | 0.636680 | 0.432237 | 1.061802 | 0.486924 | 0.329550 | 0.786624 | 439,790 |

cell 수는 mask 내 유효 feature×시점의 반복 출현 수이며, 서로 다른 원시 프레임 수가 아니다.

### Test 비교

| Mask 프레임 | Persistence MAE | Interpolation MAE | A MAE | Persistence Huber | Interpolation Huber | A Huber | 유효 cell 수 |
|---|---:|---:|---:|---:|---:|---:|---:|
| 8-13 | 0.177431 | 0.108497 | 0.215590 | 0.122562 | 0.076329 | 0.095623 | 73,125 |
| 25 | 0.361481 | 0.241635 | 0.559143 | 0.263868 | 0.179583 | 0.378713 | 180,570 |
| 50 | 0.533958 | 0.360054 | 0.896528 | 0.388737 | 0.265145 | 0.628898 | 361,166 |

cell 수는 mask 내 유효 feature×시점의 반복 출현 수이며, 서로 다른 원시 프레임 수가 아니다.

### Test 특징별 MAE: 기본 8–13프레임 mask

| Feature | Persistence | Interpolation | A | A 스키마 단위 MAE |
|---|---:|---:|---:|---:|
| `cross_density_3m` | 0.152003 | 0.104119 | 0.165967 | 0.002743 |
| `cross_density_5m` | 0.113930 | 0.075838 | 0.154103 | 0.004712 |
| `cross_density_10m` | 0.077112 | 0.048176 | 0.126918 | 0.008286 |
| `cross_min` | 0.134410 | 0.044773 | 0.119333 | 0.001444 |
| `cross_nearest_mean` | 0.046112 | 0.013419 | 0.113049 | 0.001710 |
| `cross_cluster_fraction` | 1.533885 | 1.058854 | 1.377666 | 0.005964 |
| `centroid_gap` | 0.034271 | 0.010569 | 0.118453 | 0.004076 |
| `width_sum` | 0.024739 | 0.005147 | 0.121473 | 0.020357 |
| `width_abs_diff` | 0.038795 | 0.010846 | 0.081330 | 0.011833 |
| `length_sum` | 0.042856 | 0.009458 | 0.121005 | 0.009344 |
| `length_abs_diff` | 0.057010 | 0.017612 | 0.096566 | 0.007446 |
| `radius_sum` | 0.025379 | 0.005151 | 0.123709 | 0.003672 |
| `radius_abs_diff` | 0.026093 | 0.006497 | 0.083090 | 0.003258 |

25/50프레임 및 Validation 특징별 MAE·Huber도 `feature_metrics.csv`와 `report.json`에 모두 저장했다. 기본 Test에서13개 특징 모두 선형 보간의 MAE가 A보다 낮았다. 반면 A의 전체 Huber는 Persistence보다 낮아도 MAE는 더 높다. 두 지표는 큰 오차에 부여하는 가중이 달라 순위가 같을 필요가 없다.

### 임베딩 분산과 collapse 진단

| 대상 | 중심 프레임 수 | 평균 분산 | 최소 분산 | 분산<1e-6 차원 | effective rank /64 |
|---|---:|---:|---:|---:|---:|
| all | 158,857 | 0.01933290 | 0.00091175 | 0 | 5.743 |
| quality_valid | 111,733 | 0.00811232 | 0.00002110 | 0 | 3.635 |

전체158,857개 시점의64차원 임베딩과 frame mapping을 저장했다. 중심 quality 유효 프레임은111,733개다. 상수 벡터로의 완전 collapse 징후는 없지만 품질 유효 subset의 effective rank≈3.64로 낮다. **64개 독립적인 유용한 요인을 학습했다고 주장할 수 없고, 낮은 유효 차원의 원인은 추가 검증이 필요하다.** 품질이 나쁜 프레임과 결측 패턴이 표현에 영향을 줄 수 있다.

### 사용한 window의 실제 원시 support

위 분할 표는 전체 블록 범위다. 일정 간격으로 고른 이번 window의 실제 input+target support 합집합은 다음과 같다. Train scaler는 표본 window와 무관하게 기존 Train 전체 고유 프레임에서 fit된 상태로 고정했다.

| Split | 이번 사용 raw frame 수 | 전반 범위 | 후반 범위 |
|---|---:|---|---|
| train | 111,197 | 11185–66810 | 110808–166378 |
| validation | 23,802 | 66811–78711 | 166381–178281 |
| test | 23,802 | 78731–90631 | 178289–190189 |

최종 코드 SHA256은 실행 당시 `environment.json`과 정확히 일치했다. 이전98개 파일 SHA256 및 기존 데이터30개 파일의 size/mtime이 동일했고, 실제 best checkpoint를 CPU에서 저장·재로드한 합성 입력 출력도 bitwise 일치했다. 모든 기록 metric·전체 embedding에 NaN/Inf가 없으며 frame mapping은 유일하고 증가한다. 결과는 `verification.json`에 기록했다.

그래프: [학습 곡선](../data/derived/model_a_run_v2_20261109/figures/learning_curve.png), [마스크 길이 비교](../data/derived/model_a_run_v2_20261109/figures/mask_comparison.png), [특징별 MAE](../data/derived/model_a_run_v2_20261109/figures/feature_mae.png), [임베딩 분산](../data/derived/model_a_run_v2_20261109/figures/embedding_variance.png). 모두 로컬 PNG이며 서버가 필요 없다.

## 8. 합성 테스트 및 검증 범위

```bash
PYTHONDONTWRITEBYTECODE=1 \
PYTHONPATH="$PWD/MTAH_Relational_Starter:$PWD" \
python -m pytest -q -p no:cacheprovider MTAH_Relational_Starter/tests
```

실행 결과: **116 passed in 15.45s**. 기존98개 + 신규18개이며 실패0개다. 다음을 검증했다.

* 요청한 shape·37,965개 parameter·모든 parameter의 finite gradient.
* 수식에 맞는 Huber, NaN/자연 결측/quality 제외, all-invalid batch에서 parameter와 optimizer state 불변.
* 학습·Validation 선택·체크포인트 재로드·세 마스크/세 방법 평가·2,400개 합성 임베딩까지 전체 실행. 별도로 학습 batch가 모두 skip되는 epoch도 검증.
* padding의 실제 recurrent context 제외 및 padding NaN 불변성, all-padding 유한한0 출력, visible NaN/eligible prediction Inf 명확한 거부.
* CPU 체크포인트 저장·재로드 후 동일 입력의 출력이 bitwise 일치.
* 선형 궤적 합성 특징의8/13/25/50프레임 보간 정답, persistence, anchor 없음 처리, NumPy metric와 Torch loss 일치.
* 기존 Dataset과 cache adapter의 입력/타깃 일치, hidden 좌표 변경 시 입력 불변, 원시 split support 중복0.
* 기존 전반/후반·기록 gap·원본 frame·자연 결측/artificial 구별·holdout perturbation 시 Train 통계 불변 테스트 유지.

실제 데이터에서는 전체 원시 캐시199,319프레임의 공간 특징을 계산하고, 위 표의 window로 학습·평가했다. 배포용 전체158,857개 중심 임베딩을 추출했다. Validation/Test 복원 평가는 split 내부 full-context window만 사용했다. **배포 임베딩은 학습 이후 split=None inference라 반 내부 split 경계를 넘는 문맥을 포함할 수 있다. 이것을 holdout 성능 평가용 embedding으로 사용하면 안 된다.**

## 9. 실행 및 결과 사용

프로젝트 루트에서 실행한다. 기존 결과를 덮어쓰지 않으므로 새 out-dir를 지정한다.

```bash
conda activate mtah2027
export PYTHONPATH="$PWD/MTAH_Relational_Starter:$PWD"
# 새 환경에서 의존성이 없을 때만 실행
# python -m pip install -r MTAH_Relational_Starter/requirements-model-a.txt
export PYTORCH_ENABLE_MPS_FALLBACK=1

python -m mtah_relational.train_model_a \
  --data data/derived/relational_temporal_v1_20261109 \
  --feature-cache data/derived/model_a_feature_cache_v1 \
  --out-dir data/derived/model_a_reproduction \
  --config MTAH_Relational_Starter/configs/model_a_small.json

python -m mtah_relational.plot_model_a \
  --run-dir data/derived/model_a_run_v2_20261109
```

실제 최종 실행 폴더는 `data/derived/model_a_run_v2_20261109/`다. v1은 로깅 보완 전 첫 실행 기록으로 보존했다. v2는 all-invalid epoch 로깅까지 검증한 최종 코드로 동일 설정을 실행했으며 하이퍼파라미터를 Test에 맞춰 변경하지 않았다.

| 로컬 결과 | 내용 |
|---|---|
| `best.pt` | Validation objective 기준 best model tensor·설정·source/split/code fingerprint |
| `config.json`, `environment.json` | 설정·Python/Torch/NumPy·장비·seed·git HEAD·Python별 SHA256 |
| `history.csv` | epoch별 Train/Validation loss, Validation MAE, 업데이트/skip 수·시간 |
| `verification.json` | 이전 파일 보존·실행 코드 SHA 일치·체크포인트 roundtrip·실제 표본 support·finite 검사 |
| `report.json` | raw support 감사, 사용 window 수, 모든 복원 평가·임베딩 진단 |
| `feature_metrics.csv` | split×mask×method×13 feature의 표준화/스키마 MAE·Huber·가중치·cell 수 |
| `embeddings.npy` | float32 `[158857,64]`, 마스킹하지 않은 배포 중심 임베딩 |
| `embedding_frame_ids.npy` | 해당 row의 원본 0-based tactical frame 번호 |
| `embedding_quality_valid.npy` | 중심 프레임의 strict 관측 품질. semantic confidence가 아님 |
| `figures/*.png` | 학습 곡선·마스크 길이 비교·특징별 MAE·64차원 분산 |

캐시·체크포인트·임베딩과 실제 경기 파생 결과는 모두 로컬 `data/derived/`에 보관하며 Git ignore 대상이다. 다른 기기에서 사용하려면 대회 데이터 접근 권한이 있는 로컬 환경에서 재생성한다.

```python
import torch
from mtah_relational.model_a import load_checkpoint

model, metadata = load_checkpoint("data/derived/model_a_run_v2_20261109/best.pt", device="cpu")
model.eval()
# corrupted_x: torch.float32 [B,151,53], padding: torch.bool [B,151]
with torch.inference_mode():
    prediction, temporal_embedding = model(corrupted_x, padding)
    center_embedding = temporal_embedding[:, 75]  # [B,64]
# Clean target은 이 forward에 전달하지 않는다.
```

## 10. 남은 문제와 다음 단계의 조건

1. **이번 A 모델은 선형 보간을 이기지 못했다.** 짧은 mask에서도 보간보다 오차가 크며25/50프레임에서는 더 크게 악화된다. 긴 마스크를 학습하지 않은 분포 차이와 단일 소형 실험이라는 제한이 있다. Test 결과를 보고 epoch/weight/threshold를 다시 선택하지 않았다. 최적화가 수렴했다고도 주장하지 않는다.
2. `cross_cluster_fraction`은 연결 성분이 바뀔 때 불연속적이며 Train IQR이 작다. 표준화 전체 오차에서 비중이 크다. 그러나 특징별 결과에서도 선형 보간이 우수하므로 이 특징 하나만 원인이라고 결론내릴 수 없다. scaler와 feature weight는 이번 실행에서 유지했다.
3. 64차원 모두 분산이0은 아니지만 effective rank가 낮아 정보가 강하게 상관되어 있다. 분산만으로 의미 있는 경합 표현이 학습되었다고 해석할 수 없다. collapse 진단 임계값1e-6은 semantic calibration 기준이 아니다.
4. 단일 경기·단일 seed·제한된 시간 블록 표본이다. 다른 경기나 실제 경합으로의 일반화, 통계적 유의성, 점유 정확도는 검증하지 못했다. strict quality 밖의 프레임은 loss에서 제외되므로 결과는 전체 관측 품질에 대한 성능이 아니다.
5. 원시 데이터의 팀 오분류·ID 불연속·결측과 GK 포함 문제는 남아 있다. A 모델은 순열에 불변인 집계 특징을 쓰지만 잘못된 관측의 사실관계를 복구하지 않는다. H/A 구조는 별도 보존하며 향후 점유 decoder에 필요하다.
6. 다음 B/C 실험은 동일 split·frozen scaler·고정 mask·13개 target·baseline·metric을 유지해야 비교할 수 있다. 먼저 Validation에서만 mask 길이 혼합·학습량·표준화 민감도를 사전 계획해 검증할 수 있다. 이번에는 GNN, fusion, 군집, 경합 분류 head, H/A/D decoder를 구현하지 않았다.

**자기지도 복원 성능의 개선과 실제 경합 감지·공식 점유 평가 개선은 별도의 검증 문제다.** 현재 산출물은 비교 가능한 A 모델 및 임베딩 인터페이스다.
