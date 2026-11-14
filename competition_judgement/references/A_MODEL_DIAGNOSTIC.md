# A 모델 단회 진단 및 Validation 전용 비교

작성일: 2026-11-09. 기존 A 모델과 제출 baseline은 보존하고, 별도 모듈·출력 폴더에서 한 차례의 제한된 진단을 수행했다. B 모델, Graph Fusion, 군집화, H/A/D decoder는 구현하지 않았다. 실제 경기 원본·파생물·체크포인트는 모두 로컬에 보관한다. Smart Tagging, 공개 GT, 볼 좌표는 이번 분석·학습에 사용하지 않았다. 새 Test 평가와 Test 기반 튜닝도 하지 않았다.

## 1. 코드 경로 진단

| 질문 | 확인 결과와 근거 |
|---|---|
| 입력53/타깃13 일치 | `spatial_schema.FEATURE_NAMES`24개, validity24개, quality4개, artificial1개. 타깃은 `TARGET_NAMES=FEATURE_NAMES[:13]`. `model_a.SpatialBiLSTM`의 Linear53→64 및 출력13과 일치 |
| 동일 scaler | `model_a_data.build_feature_cache`는 동일 `base.normalizer`로 input24개와 target13개를 transform. 타깃에는 동일 median/scale의 처음13개 적용. Validation도 같은 Train normalizer를 로드하며 source/split/policy fingerprint를 검사 |
| 입력과 target 정밀도 | input은 원래 Dataset의 float32 반올림을 유지하고, target은 원래 float64 summary를 변환 후 float32 저장. 서로 다른 scaler가 아니다. adapter와 기존 Dataset의 입력·타깃 배열 일치 테스트 유지 |
| Artificial mask | `FeatureWindowDataset.__getitem__`은 mask를 먼저 정하고 visible row만 cache에서 읽음. hidden의 값·validity·quality=0, artificial bit=1. 8–13/25/50 모두 한 개의 연속 구간 |
| hidden 누출 | target은 `get_targets`의 loss-only 경로. 실제 hidden 원시 좌표를 바꾸고 cache를 다시 만든 합성 테스트에서 corrupted input 불변. frame-local cache 외의 delta·스무딩·보간·clean count 경로 없음 |
| Loss 범위 | `reconstruction.masked_huber`: artificial & clean_observed & finite target & 유효 quality & feature weight>0인 cell만 사용. NaN 연산 전에 sanitize, all-invalid batch는 update 없음 |
| 양방향 시간 문맥 | LSTM의 `bidirectional=True`, hidden32×2. 합성 미분 테스트에서 forward state는 과거, reverse state는 미래에만 의존. 실제 hidden target의 예측 gradient도 과거·미래 visible 값 양쪽에 반응 |
| Train/Validation/Test 전처리 | 기존 `open_temporal_dataset`/frozen normalizer/quality 정책 공통. 역할별로 달라야 할 mask epoch만 Train에서 변경. 이번 신규 학습은 Train/Validation만 열었고 Test window는 생성하지 않음 |
| 보간과 신경망 평가 조건 | `train_model_a.evaluate`의 같은 batch에서 입력1개·target1개·artificial1개를 만든 뒤 세 방법에 공통 사용. 동일 `ReconstructionMetrics`, quality/feature weight, 실제 target cell 수. 이번 original Validation 재평가도 이전 결과와 일치 |

기존 코드에서 복원 부진을 설명할 **스키마·scaler·마스킹·loss·단방향 실행 버그는 발견하지 않았다.** 이 검증은 실제 선수의 팀 오분류나 추적 누락 자체가 없음을 증명하지 않는다.

### Mask token과 64개 차이

기존 A에는 별도의 `nn.Parameter(64)` token이 없다. 대신 `[0,…,0,1]`인 hidden 입력이 두 Linear/LayerNorm/ELU를 통과해 모든 hidden 시점에 같은64차원 인코딩을 만든다. MLP의 weight/bias/LayerNorm parameter는 학습되므로 이 표현도 학습 과정에서 변한다. **고정된 zero embedding을 LSTM에 넣는 구현은 아니다.** hidden bit가0인 자연 결측과 구별된다.

원래 예상38,029개와 실제37,965개의 차이는 정확히 별도 token64개다. 신규 실험용 `TokenSpatialBiLSTM`은 MLP 이후 artificial 시점만 학습 가능한64벡터로 대체한다. 기존 A의 hidden 인코딩으로 token을 초기화하므로 변경 직후 함수는 동일하다. 합성 테스트에서 출력 일치·추가64개·token gradient·hidden 값/quality 변화 불변을 확인했다. 이 실험을 위해 기존 A 클래스나 체크포인트 로더는 변경하지 않았다.

## 2. 한 번으로 제한한 실험 설계

설정은 [a_diagnostic_once.json](../MTAH_Relational_Starter/configs/a_diagnostic_once.json)이며 새 결과를 계산하기 전에 로컬 `preregistered_config.json`에 저장했다. 결과를 보고 추가 trial이나 설정 변경을 반복하지 않았다.

| 이름 | 시작 상태 | 학습 mask | 학습 상한 | 목적 |
|---|---|---|---|---|
| original | 기존 A best epoch12 | 기존8–13 | 추가 없음 | 기준점 |
| continued_short | 동일 A epoch12 | 8–13 | 추가8 epoch | 학습량 증가 효과 분리 |
| token_short | 동일 A + 함수 보존 초기화 token | 8–13 | 추가8 epoch | control과 비교해 token만 추가 |
| token_mixed | 동일 A + 같은 초기 token | 8–13 /25 /50 | 추가8 epoch | 학습 mask 분포 확대 |
| small_temporal_mixed | 새로 초기화한 작은 Temporal Conv | 같은 혼합 | 최대20 epoch | 신경망 복원의 작은 대조군 |

공통 seed2027, batch64, Adam lr0.001, gradient clip1, Huber delta1, feature weight13개 모두1, patience3, min_delta0.0001이다. warm-start 실험은 동일 weight에서 **optimizer를 새로 시작**했다. continued/token_short는 동일한 Train window·shuffle·epoch12–19의 mask를 공유한다. 기존 A의 optimizer state 복구 실험은 아니다.

혼합 mask는 관측·quality와 무관한 routing 기반 난수로 각 범주를1/3 확률로 고른다. 기존과 같은 유효 cell 가중 목적함수이므로 긴 span이 더 많은 target cell을 제공한다. 범주별 loss 기여가1/3이 되도록 다시 가중한 실험은 아니다.

모든 모델은 동일 Train22,181개, Validation942개 window, 동일 frozen scaler 및 같은 세 고정 Validation mask를 쓴다. early stopping/epoch 선택 objective는 **세 길이 Validation Huber의 동일 가중 평균**이다. 기존 original의 checkpoint는 과거 짧은 mask로 선택된 것을 그대로 사용한다. 비교 모델이 서로 다른 목적으로 학습된 점을 감추지 않는다.

사전에 고정한 채택 규칙은 최저 composite의 challenger가 original 대비 composite Huber를2% 이상 개선하고, 짧은 mask Huber가5% 넘게 나빠지지 않는 경우다. 통과하지 않으면 original을 유지한다. 이2%/5%는 실무용 제한이며 통계적 유의성 기준이나 GT로 정한 값이 아니다. 개별 feature/표현 분석 결과를 보고 이 규칙을 바꾸지 않는다.

작은 Temporal 모델은 공간 MLP53→64→64 뒤에 dilation1/4/16, kernel5의 symmetric depthwise Conv + pointwise Conv를 residual로3회 적용한다. receptive field85프레임, embedding64, 같은 reconstruction head13이다. Graph나 Fusion은 없다. A의151프레임 문맥과 다르고 warm-start 이력·학습 mask도 달라 순수한 아키텍처 우열을 확정하는 실험은 아니다.

**기존 A의25·50프레임 평가는 학습 mask 분포 밖의 평가였다.** 8–13의 성능만으로 긴 결측 복원을 보장할 수 없다. 이번 mixed 실험은 이 분포 차이를 직접 확인하는 제한된 실험이다.

## 3. 실제 Validation 결과

실제 실행 환경은 `mtah2027` / PyTorch2.15.0 / **MPS**, seed2027이다. 다음 수치는 모두 같은942개 Validation 중심과 고정 mask에서 계산한 표준화 오차다. Test는 신규 평가하지 않았다.

| 모델 | MAE 8–13 | MAE25 | MAE50 | Huber8–13 | Huber25 | Huber50 | 평균 Huber |
|---|---:|---:|---:|---:|---:|---:|---:|
| original | 0.273364 | 0.663803 | 1.061802 | 0.142863 | 0.474236 | 0.786624 | 0.467908 |
| continued_short | 0.251037 | 0.616169 | 0.990583 | 0.134285 | 0.436691 | 0.733946 | 0.434974 |
| token_short | 0.251013 | 0.612899 | 0.981734 | 0.133241 | 0.434313 | 0.727552 | 0.431702 |
| token_mixed | 0.346090 | 0.408956 | 0.530287 | 0.188328 | 0.246382 | 0.347077 | 0.260596 |
| small_temporal_mixed | 0.382229 | 0.474351 | 0.598530 | 0.194721 | 0.274727 | 0.382690 | 0.284046 |
| Linear interpolation | 0.151281 | 0.276590 | 0.432237 | 0.113039 | 0.208054 | 0.329550 | 0.216881 |

| 실험 | 파라미터 | 완료 epoch | best 추가 epoch | 시간(초) | token 변화 L2 |
|---|---:|---:|---:|---:|---:|
| continued_short | 37,965 | 8 | 8 | 64.45 | 해당 없음 |
| token_short | 38,029 | 8 | 8 | 66.25 | 0.215118 |
| token_mixed | 38,029 | 7 | 4 | 66.30 | 0.173951 |
| small_temporal_mixed | 26,509 | 14 | 11 | 91.41 | 해당 없음 |

4개 추가 실험 총 실행 시간은 약288.4초다(표현 분석·최초 검증 제외). token_mixed와 small_temporal은 patience에 도달해 상한 전에 중단했다. continued/token_short는 추가8 epoch 상한에 도달했다. 수렴을 보장하지 않는다.

* 추가 학습만으로 짧은 Huber가 6.00% 감소했다. 최초12 epoch가 충분한 학습량이었다고 볼 수 없다.
* token_short는 같은 추가 학습 대조군보다 짧은 Huber가 0.78% 낮았지만 MAE 차이는 0.000025다. token은 실제로 학습되지만, 부재가 주된 성능 병목이라는 근거는 약하다.
* mixed의 평균 Huber는 original 대비 44.31% 개선됐지만 짧은 Huber는 31.82% 악화됐다. 긴 mask를 학습하는 이점과 짧은 복원 손실 간 상충이다.
* 모든 신규 모델은 세 길이 모두에서 선형 보간보다 MAE와 Huber가 높았다. 작은 Temporal 모델에서도 개선 우위를 확인하지 못했다. 이 결과를 보고 다음 trial을 추가하지 않았다.

## 4. 13개 공간 특징의 해석

폭·길이는 기존 P90–P10, 반경은 팀 중심까지 평균 거리이며 입력은 기하 스키마 비율 + Train IQR 정규화 단위다. 짧은 구간에서 이런 집합 요약이 매끄럽게 변하면 endpoint 보간이 매우 강한 기준이다.

| 특징 | 관찰 가능한 변화와 보간 특성 | 관측 인원 민감성 |
|---|---|---|
| cross_density_3m/5m/10m | 거리가 threshold를 통과할 때 쌍 수가 계단식으로 변함. 좁은3m가 짧은 접근·이탈에 민감하며 큰 radius는 더 매끄러울 수 있음 | 분자 쌍 수와 분모 nH×nA가 모두 관측 집합에 의존 |
| cross_min | 최근접 H-A 쌍의 거리. 가까운 선수의 접근·이탈 또는 최소 쌍 교체 때 비선형 변화 가능 | 가까운 선수가 사라지거나 새로 관측되면 급변 가능 |
| cross_nearest_mean | 양 팀 각 선수의 최근접 상대거리 평균. minimum보다 집계가 안정적일 수 있음 | 평균에 포함되는 선수 및 각 최근접 상대가 변경됨 |
| cross_cluster_fraction | H-A3m 연결 성분의 최대 크기/관측 수. 한 edge가 연결·분리되면 큰 불연속 발생 | 노드/edge 출현·소실과 분모 변화에 민감. 연결 사슬은 직접적인 한 곳의 경합이 아님 |
| centroid_gap | 두 팀 중심 거리. 단기 집합 이동은 대개 매끄러워 보간이 강함 | 누락·추가된 관측이 평균 중심을 이동시킴 |
| width_sum/width_abs_diff | 양 팀 횡적 구조. 합은 특히 매끄럽고 절댓차는 교차점에서 꺾임 | 관측 구성과 percentile 순위가 바뀌면 값이 변함 |
| length_sum/length_abs_diff | 양 팀 종적 구조. 합과 차의 비선형성이 다르며 GK 제거를 보장하지 않음 | GK/전방 선수 관측 변화 및 percentile 순위에 민감 |
| radius_sum/radius_abs_diff | 팀 중심 주변의 응집도. 짧은 문맥에서 보간이 강한 편 | 중심과 평균에 포함되는 집합이 동시에 변함 |

모든 특징은 일정 정도 관측 인원/추적 교체에 민감하다. 인원 정규화는 완전한 누락 보정이 아니다. count가 같아도 slot 교체나 다른 선수의 누락은 남는다. 따라서 급변을 곧바로 실제 선수 접근·경합이라고 확정하지 않는다.

다음은 기본8–13프레임 Validation MAE다. 마지막 열은 **양 끝 quality-valid인 연속 관측 중 인원 변화가 있는 경우/없는 경우의 평균 절대 feature 차이 비율**이다. 정답 라벨이 없는 기술 통계이며 인원 변화의 인과효과로 해석하지 않는다.

| Feature | Interpolation MAE | original MAE | token_mixed MAE | count 변화 시 평균 차이 배수 |
|---|---:|---:|---:|---:|
| `cross_density_3m` | 0.124671 | 0.191442 | 0.240325 | 13.62× |
| `cross_density_5m` | 0.080301 | 0.165748 | 0.206166 | 13.82× |
| `cross_density_10m` | 0.053536 | 0.149059 | 0.179151 | 14.76× |
| `cross_min` | 0.052259 | 0.157714 | 0.198070 | 9.71× |
| `cross_nearest_mean` | 0.027583 | 0.137851 | 0.171424 | 44.25× |
| `cross_cluster_fraction` | 1.524689 | 1.930740 | 2.452944 | 16.97× |
| `centroid_gap` | 0.018447 | 0.127729 | 0.153548 | 69.85× |
| `width_sum` | 0.006472 | 0.113753 | 0.143501 | 19.15× |
| `width_abs_diff` | 0.014751 | 0.096781 | 0.142253 | 22.87× |
| `length_sum` | 0.014775 | 0.119023 | 0.163916 | 43.89× |
| `length_abs_diff` | 0.026571 | 0.122396 | 0.156320 | 58.84× |
| `radius_sum` | 0.009656 | 0.128563 | 0.139962 | 34.10× |
| `radius_abs_diff` | 0.012941 | 0.112935 | 0.151590 | 48.28× |

**매끄러운 구조 요약:** width_sum·radius_sum·radius_abs_diff·width_abs_diff·length_sum은 보간 MAE가 약0.006–0.015로 작다. centroid_gap도0.018 정도다. 이 특징들에서 신경망의 오차 바닥이 보간보다 높다.

**불연속·희소 변화:** cross_density_3m/5m/10m는 유효 연속 관측에서 각각 약93.5%/91.4%/88.2%가 정확히 같은 값을 유지한다. threshold 통과 시 계단식으로 바뀐다. cluster fraction은96.9%에서 변하지 않지만 드문 큰 점프가 있다. 따라서 높은 zero-change 비율과 높은 복원 오차가 동시에 가능하다. minimum distance는 최근접 쌍 교체, 평균 최근접 거리는 관측 집합 변경의 영향을 받는다.

**스케일 진단:** Train quality-valid 80,538개에서 최대 혼합 component가2명인 비율은 62.25%다. cluster fraction의 Q25=0.090909091, Q75=0.095238095, IQR=0.004329004다. 이는 `2/22`와 `2/21`의 차이와 같다. 작은 IQR은 관측 인원21/22 차이와 연결되고, component가 실제로 커지는 변화는 수십 IQR로 증폭될 수 있다. scaler 구현 오류가 아닌 **이산 특징과 robust scaling의 조합 문제**다.

기본 Validation Huber 중 cluster fraction의 기여는 original 86.61%, token_mixed 85.32%, interpolation 95.02%다. feature weight를 모두1로 두어도 실제 loss 중요도가 균등하지 않다. **낮은 rank와 접촉 특징 집중을 설명하는 강한 단서**이지만 원인의 인과적 분해 실험은 아니다. 이번 결과를 보고 loss weight나 scale을 바꿔 추가 튜닝하지 않았다.

관측 인원이 안정적인 mask window는803개, 변화가 있는 window는139개였고, interpolation MAE는 각각0.118394/0.365320이었다. 각각 유효 cell81,731/12,558개다. window 안의 count 변화는 clean quality를 진단용으로만 읽어 분류했으며 모델 입력이나 학습 mask 선택에 사용하지 않았다. count가 같아도 선수 교체·추적 불연속을 배제할 수 없다.

## 5. Representation 진단 방법과 결과

배포용 `split=None` embedding을 holdout 분석에 그대로 쓰지 않았다. **Train과 Validation의 각 블록 안에151프레임이 모두 들어가는 window에서 artificial mask 없이 다시 추론**했다. Train 중심은25프레임 간격, Validation 중심은 매 프레임이다. original과 최저 Validation composite challenger를 같은 중심에서 비교했다. Test는 분석 대상이 아니다.

PCA의 mean/basis는 **Train quality-valid 중심만으로 fit**하고 Validation에 투영했다. 별도로 Validation 공분산의 eigenvalue/EVR/effective rank를 기술 통계로 계산했다. 이 기술 통계는 PCA를 Validation에 맞춰 학습 입력을 변환한 것이 아니다. 차원별 분산64개, EVR64개, 누적 EVR64개와 PC1–5/특징 상관은 로컬 JSON/CSV에 저장했다.

시간 변화는 같은 반·같은 블록의 연속 frame 중심 embedding 간 L2 차이로 계산했다. 서로 다른 sliding window의 중심 hidden state를 비교하므로 **하나의 무한 시퀀스에 state를 전달한 결과가 아니다.** 예측이나 loss에는 미래/과거 문맥을 정상적으로 쓰되 반·split 밖으로 recurrent state를 전달하지 않는다.

현재 공간 기하와의 관계는 quality-valid subset에서 PC1–5와13개 feature의 Pearson 상관, 연속 embedding 차이와 공간 특징 차이의 Spearman 상관으로 확인했다. bad/good 품질 전환별 hidden 변화와 quality4개 입력을0으로 바꾼 진단용 ablation도 계산했다. 이 상관/ablation은 경합 정답·인과관계·semantic confidence가 아니다.

Train 4,437개 중심 중 quality-valid subset으로 PCA를 fit하고, Validation 23,528개 중심(quality-valid17,057개, bad6,471개)을 분석했다.

| 모델 / Validation subset | effective rank | PC1 EVR | PC1–2 누적 | 90% 설명 PC 수 | 95% 설명 PC 수 | 최소 차원 분산 | 최대 차원 분산 |
|---|---:|---:|---:|---:|---:|---:|---:|
| original / all | 4.752 | 53.38% | 71.64% | 4 | 8 | 0.00020773 | 0.05650495 |
| original / quality_valid | 2.952 | 72.30% | 88.09% | 3 | 6 | 0.00000262 | 0.04119229 |
| original / quality_bad | 4.517 | 48.41% | 75.46% | 4 | 7 | 0.00070502 | 0.09224567 |
| token_mixed / all | 7.469 | 38.85% | 58.77% | 7 | 12 | 0.00151175 | 0.07950471 |
| token_mixed / quality_valid | 5.563 | 52.41% | 69.62% | 6 | 9 | 0.00000288 | 0.05022520 |
| token_mixed / quality_bad | 6.201 | 39.95% | 63.48% | 6 | 10 | 0.00502772 | 0.12510095 |

기존 보고의3.635는 배포 전체의 quality-valid111,733개에서 계산한 수치다. 이번 original2.952는 **Validation 내부 full-context17,057개**라 모집단이 다르다. 저하로 비교하면 안 된다. 같은 Validation에서 token_mixed는5.563으로 증가했다. 두 모델 모두64개 차원의 분산이0은 아니며 rank 증가 자체가 짧은 복원 성능 개선으로 이어지지는 않았다.

**original:** Train PCA fit 중심3,208개. Train basis를 적용한 Validation 누적 설명 비율은 첫2개 86.45%, 첫5개 91.73%다. embedding 변화량과13개 표준화 기하 변화량 norm의 Spearman은 0.807이다. 인원 불변/변화 시 평균 embedding 차이는 0.029734/0.249401 (각16,820/172쌍)다. quality4개를0으로 바꾼942개 고정 window의 중심 embedding 이동 평균은0.084246다.

**token_mixed:** Train PCA fit 중심3,208개. Train basis를 적용한 Validation 누적 설명 비율은 첫2개 63.50%, 첫5개 81.84%다. embedding 변화량과13개 표준화 기하 변화량 norm의 Spearman은 0.766이다. 인원 불변/변화 시 평균 embedding 차이는 0.040824/0.295177 (각16,820/172쌍)다. quality4개를0으로 바꾼942개 고정 window의 중심 embedding 이동 평균은0.123493다.

original Train-PC1의 Validation 상관은 cluster fraction +0.839, cross_min −0.752, density3m +0.672였다. token_mixed도 이 접촉 관련 축과 강하게 관련된다. PCA 부호는 임의이므로 두 모델의 부호 자체를 비교하지 않는다. 상대 거리 변화와의 연관은 있지만 경합 판별력은 검증한 것이 아니다.

| 인접 frame의 품질 변화 | 관측 쌍 수 | original 평균 hidden 차이 | token_mixed 평균 hidden 차이 |
|---|---:|---:|---:|
| good_to_good | 16,992 | 0.031958 | 0.043399 |
| bad_to_bad | 6,407 | 0.044599 | 0.056223 |
| good_to_bad | 63 | 0.107757 | 0.133823 |
| bad_to_good | 64 | 0.111890 | 0.131872 |

품질 전환에서 hidden 차이가 더 크다. 임베딩 변화량을 그대로 경합 신호로 쓰면 tracking/관측 구성 변화에 반응할 수 있다. quality4개 ablation에서도 변화가 있지만 이는 out-of-distribution 입력이며 원인을 완전히 분리하지 않는다. feature validity와 기하 자체에도 관측 구성 정보가 남는다.

공간 특징과 높은 상관, 작은 IQR 특징의 loss 집중, 서로 관련된13개 target은 낮은 유효 차원을 설명할 수 있다. 낮은 rank를 단순 실패로 단정할 수 없고, 동시에 현재 표현이 축구 의미를 잘 분리한다고 주장할 수도 없다.

로컬 그래프: [Validation 비교](../data/derived/a_diagnostic_once_20261109/figures/validation_comparison.png), [PCA/차원별 분산](../data/derived/a_diagnostic_once_20261109/figures/pca_variance.png). 서버 없이 PNG로 연다.

낮은 effective rank만으로 실패라고 단정하지 않는다. target은13개이며 상호 관련된 거리·폭·반경을 반복 집계한다. 64차원 encoder에64개의 독립 요인을 강제한 목적함수도 아니다. 상수 collapse, 낮은 내재 차원, feature/quality에 치우친 표현은 구분해야 한다. 반대로 분산이 있고 상관이 관찰된다는 이유만으로 점유 판단에 유용하다고 주장할 수도 없다.

## 6. 테스트, 제출 파이프라인과 보존

전체 합성 테스트 **125 passed in 16.35s**, 실패0개다(기존116개 + 신규9개). token 함수 보존 초기화·64개 추가 parameter·gradient·hidden 누출 차단, forward/reverse 방향 미분, mixed mask의 연속성·Validation 고정성, 작은 Temporal forward/backward, 실험 checkpoint bitwise roundtrip, PCA rank1/상수 처리와 사전 채택 규칙을 검증했다. 기존 scaler·mask·자연 결측·all-invalid·split 누출 테스트도 통과했다.

실제 hidden target의 past/future visible feature gradient norm은 **1.040928/1.450899**였다. 단일 입력에서의 국소 민감도이며 시간 문맥의 모든 범위를 유효하게 학습했다는 증명은 아니다.

기존 baseline_raw/smooth25/smooth50/smooth125 **4개 모두199,319행, frame0–199318**이 유효했다. 기존 라벨을 수정 없는 `starter.baseline.write_submission`으로 새 로컬 파일에 생성하고, 공식 `evaluation/validate_submission.py --rows199319`를 실행해 모두 통과했다. 원본과 재생성 라벨은 동일하다. GT 없이 **기존 baseline 결과→writer→형식 validator**를 확인한 것이며, 볼 기반 원시 추론이나 정확도 채점은 재실행하지 않았다.

보존 목록118개 기존 파일의 SHA256과43개 이전 로컬 결과의 size/mtime이 그대로였다. 새 모델들의 Validation window·mask별 유효 target cell 수가 동일하고, 기록 metric와 저장 embedding이 모두 finite임을 확인했다. 학습 시작 당시 저장한 Python SHA256도 그대로 일치한다. 이후 추가한 별도 표현/scaling 분석 파일까지 포함한 최종 코드 SHA는 로컬 `verification.json`에 기록했다. 실패 테스트는 없고, 단일 seed·단일 경기·제한된 epoch 및 원시 추적 오류의 원인 미확정이 남은 제한이다.


신규 파일:

* [a_diagnostic_models.py](../MTAH_Relational_Starter/mtah_relational/a_diagnostic_models.py): 별도 token A, 작은 Temporal, 혼합 mask, gradient/covariance 진단.
* [diagnose_a_training.py](../MTAH_Relational_Starter/mtah_relational/diagnose_a_training.py): 제한된 Validation-only 비교, experimental checkpoint, 기존 제출 writer/validator 감사.
* [diagnose_a_representation.py](../MTAH_Relational_Starter/mtah_relational/diagnose_a_representation.py): split 내부 unmasked 추론, PCA/상관/품질·count 변화 분석, 오프라인 PNG.
* [diagnose_a_scaling.py](../MTAH_Relational_Starter/mtah_relational/diagnose_a_scaling.py): Train-only component 분포·IQR 및 특징별 Validation loss 기여 진단. scaler/weight 재적합 없음.
* [test_a_diagnostics.py](../MTAH_Relational_Starter/tests/test_a_diagnostics.py), [a_diagnostic_once.json](../MTAH_Relational_Starter/configs/a_diagnostic_once.json).

기존 모델·전처리·마스킹·loss·scaler·Rule-based Python은 수정하지 않았다. `starter/`, `evaluation/`, `games.py` 및 이전 로컬 결과도 그대로 보존한다. 연구 문서에는 이 후속 진단의 링크만 추가한다.

프로젝트 루트에서 재현한다. 기존 출력을 덮어쓰지 않으므로 새 out-dir를 사용한다.

```bash
conda activate mtah2027
export PYTHONPATH="$PWD/MTAH_Relational_Starter:$PWD"
export PYTORCH_ENABLE_MPS_FALLBACK=1
python -m pytest -q -p no:cacheprovider MTAH_Relational_Starter/tests
python -m mtah_relational.diagnose_a_training \
  --out-dir data/derived/a_diagnostic_reproduction
python -m mtah_relational.diagnose_a_representation \
  --diagnostic-dir data/derived/a_diagnostic_reproduction
python -m mtah_relational.diagnose_a_scaling \
  --diagnostic-dir data/derived/a_diagnostic_reproduction
```

실제 실행은 `data/derived/a_diagnostic_once_20261109/`에 저장했다. `training_report.json`, variant별 `history.csv/best.pt/validation.json`, `representation/report.json`, `scaling_diagnostic.json`, `verification.json`, PCA 상관 CSV, embedding NPY 및 `figures/`는 모두 로컬 전용이다. 서버나 외부 보고 서비스를 사용하지 않는다. experimental checkpoint는 원래 A용 loader가 아닌 `diagnose_a_training.restore_variant()`로 연다.

## 7. 채택 판단과 B 모델 준비 상태

**기존 A checkpoint를 동결 기준으로 유지한다.** 최저 composite challenger는 token_mixed였지만 짧은 mask 악화31.82%로 사전 허용5%를 넘었다. original보다 다양한 변화에 반응하는 표현과 긴 복원 개선은 관찰했으나, 전체적으로 선형 보간을 이기지 못했고 기존 짧은 복원 기준도 유지하지 못했다. 따라서 token/mixed를 기본 A로 자동 교체하지 않았다.

continued_short는 짧은 손실을 약6% 개선했고 token_short도 소폭 추가 개선했으므로 학습 자체는 더 진행될 여지가 있다. 다만 비교 기준을 매번 바꾸는 대신 기존 A12를 동결하고, **20 epoch 상당의 continued_short를 학습량을 맞춘 보조 대조군으로 보관**한다. 향후 B를20 epoch 학습한다면 A12만과 비교하지 말고 이 A20 대조군도 포함해야 한다. 이 대조군의 개선을 숨기거나 token 효과로 돌리지 않는다.

이번 진단에서 추가로 scale/weight/mask 확률을 재탐색하지 않는다. A 모델에 더 긴 튜닝을 쓰기보다, 작은 IQR로 인한 손실 편중과 품질 영향을 다음 A/B 공통 실험 계획에 명시하는 것이 우선이다. 손실 균형을 바꾸려면 Train에서만 정하고 A/B 모두에 적용한 새로운 실험으로 기록해야 한다.


B 구현에 필요한 frame-local 선수 집합·팀 코드·품질·raw-first masking, 원본 frame routing, 전후반/기록 gap 경계, Train-only normalizer, 별도 clean target은 준비되어 있다. A 진단용 token과 mixed masker는 기존 Dataset 계약과 호환된다. 원본 slot을 지속적인 선수 ID로 연결하지 않고 매 프레임 graph→집합 pooling→Bi-LSTM 순서로 구현할 수 있다.

B와의 공정한 비교에는 A와 같은 split·53차원 feature 기준·13개 target·quality·고정 Validation mask를 유지해야 한다. mixed-mask 학습을 채택할지는 다음 실험 전에 정하고 A/B에 동일하게 적용한다. B만 긴 mask 학습을 추가하고 기존 short-only A와 비교해 Graph 효과라고 주장해서는 안 된다. H/A 개별 구조 sidecar는 계속 보존하고, B의 팀별 pooling에서도 이를 유지해야 한다.

현재 준비 완료는 **입력·학습·평가 계약**에 대한 것이다. Graph builder/encoder와 graph 순열·팀 교환·mask 누출 검증은 다음 단계에서 구현할 작업이며 이번에는 진행하지 않았다. 단일 경기에서의 재구성 개선이 최종 H/A/D 점유 성능 개선을 의미하지 않는다. 기존199,319행 제출 baseline 경로와 독립적으로 연구 모듈을 유지한다.
