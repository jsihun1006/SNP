# 5단계: Dynamic Graph Encoder + Bi-LSTM B 모델

작성일: 2026-11-09. B 모델 구현·합성 검증·실제 로컬 MPS 학습·Validation 비교·합성 추적 노이즈 실험을 완료했다. **이번 설정에서 B는 A12/A20 및 선형 보간보다 복원 성능이 낮았다.** 선수 순열/H-A 교환 대칭성은 검증했지만 실제 경합·점유·추적 정확도 개선은 검증하지 않았다. Test는 잠근 상태로 유지했다.

## 1. 기존 구현 검토와 보존

요청한 [A 실행 기록](PHASE4_MODEL_A.md), [1단계 관계 특징](PHASE1_DATA_RELATIONS.md), [시계열 데이터](PHASE2_TEMPORAL_DATA.md), [A 진단](A_MODEL_DIAGNOSTIC.md), [전체 계획](MODEL_IMPLEMENTATION_PLAN.md), [외부 연구](CODEX_RESEARCH_OUTPUT.md)를 검토했다. 기존 `open_temporal_dataset`, frozen Train normalizer, SpanMasker, 관측 policy, loss-only target와 Huber/metric 집계를 재사용했다.

A12, A20, 기존 baseline, 전처리·마스킹·loss 코드 및 이전 원시/파생 결과를 보존했다. `starter/`, `evaluation/`, `games.py`와 H/A/D 생성 로직을 바꾸지 않았다. Smart Tagging, 볼, speed, broadcast, 공개 GT를 B 입력·타깃·모델 선택에 사용하지 않았다. 경기 데이터·임베딩·checkpoint는 로컬 전용이며 외부 업로드·GitHub push·웹 서버 실행을 하지 않았다.

## 2. 실제 Graph Builder

구현은 [dynamic_graph.py](../MTAH_Relational_Starter/mtah_relational/dynamic_graph.py)의 `build_graph()`다. 각 프레임의 유효 관측 집합만 사용하며 슬롯을 인물 ID로 연결하지 않는다.

* 기존 `assess_observations`의 finite/2m bounds-margin 정책을 적용하고 남은 N개의 관측을 노드로 만든다. 같은 팀 duplicate 의심을 수작업 삭제하거나 팀을 교정하지 않는다.
* Node feature는 `[x/105.3,y/68]` 2개다. 팀 코드는 edge 관계와 pooling에만 사용하며 홈/원정 전용 학습 가중치는 없다.
* 같은 팀의 자기 자신을 제외한 kNN(k=3)을 연결한다. k번째 거리와 동률인 상대를 모두 포함하므로 일부 노드의 이웃은3개보다 많을 수 있다. 배열 index로 동률을 자르지 않는다.
* 상대 팀은10m 이내 모두 연결한다. 가까운 상대가10m 밖에 있더라도 최근접 상대의 모든 동률을 연결한다.
* 방향을 양방향화하고 bool adjacency로 중복을 제거한다. self-loop는 노드마다 정확히1개를 직접 생성한다.
* 48구역은 edge 생성에 사용하지 않는다. 같은 구역 여부와 무관하게 연속 좌표에서 계산한다.

```text
edge_index: [2,E], source → destination
edge_attr:  [E,8]
[dx/L, dy/W, d/10, same_team, opposite_team,
 opposite & d<=3, opposite & d<=5, opposite & d<=10]
dx,dy = destination − source
self-loop = [0,0,0,1,0,0,0,0]
```

반대 방향에서는 dx/dy만 부호가 바뀐다. d/10은 clipping이 아니어서 먼 최근접 상대 edge에서는1을 넘을 수 있다. kNN/radius 동률은 미터 거리 tolerance1e-8, float32 edge flag의 threshold 비교에는1e-5m tolerance를 사용한다. 추적 정밀도보다 작은 수치 경계 처리다.

## 3. GAT·pooling·Bi-LSTM의 구현

[model_b.py](../MTAH_Relational_Starter/mtah_relational/model_b.py)에 구현했다.

| 단계 | 입력 → 출력 |
|---|---|
| GAT1 | node2 → heads4×32, head 평균으로32; edge8 |
| LayerNorm/ELU | node별32 |
| GAT2 | node32 → heads4×32, head 평균으로32; edge8 |
| LayerNorm/ELU | node별32 |
| Team pooling | H mean32, A mean32, 전체 valid node max32 |
| Symmetric readout | `[z_H+z_A, abs(z_H−z_A), max_all]`96 → Linear64 → LayerNorm |
| Temporal | Bi-LSTM input64, hidden32/방향,1층 → `[B,151,64]` |
| Head | Linear64→64, ELU, Linear64→13 |

attention은 같은 팀과 상대 팀에 공유하는 edge-aware GAT 수식이다. `concat=False`, dropout0, residual projection 없음이다. self-loop의 물리적 속성은 builder가 만들기 때문에 PyG의 자동 loop/평균 edge-attribute 채우기를 추가하지 않는다. 빈 팀의 평균과 전체 빈 집합의 max는0이며 별도 `team_observed`로 관측 유무를 보존한다. temporal padding은 다른 반/기록 gap으로 채우지 않고 recurrent context에서 제외한다.

`GraphBiLSTM.forward(GraphBatch)`는 `BOutput`을 반환한다.

```text
prediction        [B,151,13]
embedding         [B,151,64]      temporal
graph_embedding   [B,151,64]      pre-temporal
team_embedding    [B,151,2,32]    H, A 순서의 z_H,z_A
team_observed     [B,151,2]       bool, semantic confidence가 아님
```

H/A를 바꾸면 `z_H,z_A`가 교환되고 scene readout은 같다. 이 대칭 표현만으로 H/A 소유자를 구별할 수 있다고 주장하지 않는다. 선수별 recurrent state는 없고, **프레임 안 GAT→pooling을 끝낸 뒤** 고정 길이 표현에 LSTM을 적용한다.

### PyG/MPS 확인과 실행 방식의 차이

PyG2.8.1을 `mtah2027`에 설치하고 **실제 MPS GATConv forward/backward가 가능함을 확인**했다. 호환성 실패로 CPU로 바꾼 것은 아니다. 현재 학습 backend는 작은 그래프를 한꺼번에 처리하기 위한 **동일 수식의 dense-masked PyTorch GAT**다. sparse neighbor 집합은 그대로 유지하며, dense 실행이라고 완전 연결 graph를 학습하는 것이 아니다.

노드 수가 작은 graph들을 배치 처리하고 큰 edge projection 중간 tensor를 줄이기 위해 `edge_attr @ W_edge`와 edge attention vector의 선형 축약 순서를 바꿨다. edge parameter를 제거하거나 고정하지 않았다. 설치된 PyG GATConv와 같은 parameter를 넣고 forward 및 node/edge projection parameter gradient가 일치하는 합성 테스트를 통과했다. node/edge shape·head 수·readout·파라미터 수는 원래 설계와 같다.

확인한 공개 API: [PyG GATConv 공식 문서](https://pytorch-geometric.readthedocs.io/en/latest/generated/torch_geometric.nn.conv.GATConv.html). 웹 latest 문서와 별개로 이번 parity 검증/실행 버전은 로컬2.8.1이다. 근거 source는 설치 환경의 `torch_geometric/nn/conv/gat_conv.py` 및 신규 합성 테스트다.

## 4. 마스킹·캐시·데이터 인터페이스

[graph_windows.py](../MTAH_Relational_Starter/mtah_relational/graph_windows.py)의 `GraphWindowDataset`은 기존 routing과 SpanMasker로 **먼저 hidden 시점을 정한다**. 그 뒤 visible 시점만 `VisibleGraphCache.get_visible()`에 질의한다. hidden/padding의 graph는 `None`, mapping은−1이다. hidden row의 node 수·edge 수·좌표·quality·topology를 생성하거나 조회하지 않는다. 이전 window에서 그 프레임이 visible하여 캐시에 있더라도 hidden 입력에 전달하지 않는다.

캐시는 **visible 질의로 생성된 원시 관측/불변 adjacency만** 메모리에 보관하는 LRU다. 학습된 GAT·pooling·temporal embedding을 epoch/optimizer update 사이에 캐시하지 않는다. 변경된 입력을 평가하는 강건성 실험은 별도 source wrapper와 별도 cache를 사용한다.

같은 batch 안에서 반복되는 visible frame은 한 번 인코딩하고 각 window 위치에 gather한다. 중복 위치의 loss gradient는 같은 encoder 계산으로 합산되어 GAT까지 전달된다. 반복 인코딩과 재사용 인코딩의 출력·gradient 일치 테스트를 통과했다. Graph Encoder gradient를 끊는 최적화가 아니다.

```text
GraphBatch.x          [G,Nmax,2]       G=unique visible frames in this batch
GraphBatch.team       [G,Nmax]         -1=padding, 0=H,1=A
GraphBatch.adjacency  [G,Nmax,Nmax]     source,destination
GraphBatch.mapping    [B,151]          local G index only; hidden/pad=-1
artificial,padding    [B,151] bool
```

Nmax는 visible 관측만으로 정한다. 미관측 슬롯을 가짜 노드로 채우지 않으며 padding 노드는 attention과 pooling에서 제외한다. 모델은 절대 frame 번호나 track slot을 feature로 받지 않는다. 현재 모델의 pitch normalization은 공식105.3×68m 고정이며 다른 경기장은 입력과 edge geometry를 함께 확장해야 한다.

Hidden 시점은 별도 **학습 가능한64차원 graph mask token**으로 대체한다. `z_H,z_A`와 team-observed는 hidden에서0/False다. 자연 결측 frame은 artificial=False인 empty/partial graph로 남고, padding은 별도 bool로 recurrent context에서 제외한다. 같은 시점의 clean target은 기존 `FeatureFrameCache`의 분리된 loss accessor만 읽는다. 모델 forward에는 target가 없다.

Loss는 A와 같은 Huber delta1, 동일13개 target, quality0/1, feature weight 전부1과 frozen Train scaler다. 원래 결측을 정답0으로 학습하지 않고 all-invalid batch는 update를 건너뛴다. 기존 작은 cluster IQR을 임의로 바꾸지 않았다.

## 5. 실제 학습 예산과 공정성의 범위

Train22,181개/Validation942개 중심은 A와 동일하다. window 내부는25fps151프레임이며, 각 반/원시 split/gap 경계를 넘지 않는다. Train mask는8–13만, epoch0–19와 seed2027의 기존 SpanMasker를 쓴다.25/50은 checkpoint 선택 뒤 평가한 **학습 mask 분포 밖**의 조건이다. Test window를 새로 평가하지 않았다.

기본 batch64, Adam lr0.001, gradient clip1이다. B12/B20 예산을 사전에 고정해20 epoch 한 번을 학습했다. 이 고정 예산 실험에는 early stopping을 적용하지 않았고, 예산별 best checkpoint는 **짧은 Validation Huber만**으로 선택했다. terminal12/20 checkpoint도 저장했다. A20의 과거 절차와 맞춰 epoch12 이후 Adam state를 초기화했다. 추가 hyperparameter 탐색은 하지 않았다.

중복 frame 계산을 줄이기 위해 **같은 블록 내16개 이웃 window를 묶고, 묶음을 shuffle하여64개 batch**로 만든다. 같은 epoch에서 모든 Train window를 정확히 한 번 사용하고 mask는 바뀌지 않는다. A의 완전한 개별 window shuffle과는 Adam minibatch 구성이 다르다. **데이터·목적함수·노출 예산은 맞췄지만, 최적화 순서까지 통제한 순수 architecture ablation은 아니다.**

A12는 기존 epoch12 best, A20은 기존 continued_short의 추가8 epoch checkpoint를 그대로 사용한다. A20의 당시 선택 기준은 세 mask 평균 Huber였고, B는 짧은 mask 기준이므로 이 차이도 한계다. 각 budget의 학습량과 실제 선택 epoch/노출량을 구분한다.

실행한 학습 예산과 checkpoint 선택을 구분한다.

| 모델 | 예산 epoch | 선택된 총 epoch | 예산 내 update 수 | 선택 checkpoint update 수 | 예산 내 유효 target cell | 선택 checkpoint 유효 cell |
|---|---:|---:|---:|---:|---:|---:|
| A12 | 12 | 12 | 4,164 | 4,164 | 26,340,535 | 26,340,535 |
| A20 | 20 | 20 | 6,940 | 6,940 | 43,910,685 | 43,910,685 |
| B12 | 12 | 3 | 4,162 | 1,041 | 26,340,535 | 6,595,173 |
| B20 | 20 | 18 | 6,938 | 6,244 | 43,910,685 | 39,520,442 |

유효 target cell은 중복 window 안의 feature×시점 출현 수이며 서로 다른 raw frame 수가 아니다.

B의 첫12 epoch target 노출량은 기존 A history의 Train target weight와 **매 epoch 정확히 일치**했다. A20 후반8 epoch의 target cell 수는 당시 로그에 없어서 동일 window·seed·epoch mask·원시 관측으로 재구성한 수치다. B와 같은 합계를 사용한 것은 이 데이터 계약이 같기 때문이며 A20에서 새로 측정한 로그처럼 제시하지 않는다.

B는 epoch5에서 유효 target가 없는 batch2개를 건너뛰어20 epoch 6,938회 update를 수행했다. 마스크를 다시 고르거나 품질 기준으로 표본을 바꾸지 않았다. 총 학습 루프 시간은 **429.75초**(첫 geometry cache 생성·epoch Validation 포함, 최종 비교·robustness·표현 export 제외)였다. 첫 epoch52.54초, 이후 epoch 범위19.26–21.22초였다.

**환경/규모:** macOS26.6.2 arm64, Python3.11.17, torch2.15.0, PyG2.8.1, NumPy2.4.6, MPS. 전체43,853개 = GAT2층+LN7,360 + readout6,336 + Bi-LSTM25,088 + head5,005 + graph mask token64.

MPS 공유-frame gradient 누적에서 `index_put_with_accumulate_mps` 비결정성 경고가 발생했다. seed와 deterministic-algorithms를 설정했지만 warn_only이므로 bitwise 재현성은 보장하지 않는다. NaN/Inf나 unsupported 연산 중단은 없었다.

## 6. 복원 성능 및 특징별 loss

모든 방법에 같은 Validation 중심·mask·loss target·quality·scaler를 사용했다. 선형 보간은 A의 corrupted feature 관측으로 수행하고, B는 corrupted raw graph만 받는다. 긴 마스크 역시 같은13개 clean target cell에서 평가했다. MAE/Huber는 Train median/IQR 단위이며 미터나 경합 확률이 아니다.

아래 B12/B20은 각 budget 안에서 짧은 Validation Huber가 가장 낮은 checkpoint다. B12는 epoch3, B20은 epoch18이며 이 선택 epoch를 실제12/20 epoch terminal로 오해하면 안 된다.

| Mask | 모델 | 전체 MAE | 전체 Huber | cluster 제외 MAE | cluster 제외 Huber | cluster Huber 기여율 |
|---|---|---:|---:|---:|---:|---:|
| 8-13 | interpolation | 0.151281 | 0.113039 | 0.036830 | 0.006100 | 95.02% |
| 8-13 | A12 | 0.273364 | 0.142863 | 0.135249 | 0.020729 | 86.61% |
| 8-13 | A20 | 0.251037 | 0.134285 | 0.121161 | 0.017825 | 87.75% |
| 8-13 | B12 | 1.212521 | 0.883110 | 0.588711 | 0.264904 | 72.31% |
| 8-13 | B20 | 1.152329 | 0.850255 | 0.485811 | 0.191483 | 79.21% |
| 25 | interpolation | 0.276590 | 0.208054 | 0.069839 | 0.012526 | 94.44% |
| 25 | A12 | 0.663803 | 0.474236 | 0.208413 | 0.042789 | 91.67% |
| 25 | A20 | 0.616169 | 0.436691 | 0.198514 | 0.039676 | 91.61% |
| 25 | B12 | 1.289126 | 0.946847 | 0.621633 | 0.287478 | 71.97% |
| 25 | B20 | 1.221578 | 0.909918 | 0.515060 | 0.210548 | 78.64% |
| 50 | interpolation | 0.432237 | 0.329550 | 0.112636 | 0.023793 | 93.34% |
| 50 | A12 | 1.061802 | 0.786624 | 0.376717 | 0.118938 | 86.04% |
| 50 | A20 | 0.990583 | 0.733946 | 0.339184 | 0.099680 | 87.46% |
| 50 | B12 | 1.305390 | 0.955879 | 0.641442 | 0.301779 | 70.86% |
| 50 | B20 | 1.296377 | 0.966813 | 0.584950 | 0.261890 | 75.00% |

엄격한 quality 정책 아래 각 feature의 유효 cell 수가 동일해 전체 가중 MAE와 feature macro 평균이 일치한다. baseline 포함 모든 모델의 유효 cell 수는8–13:94,289,25:218,439,50:439,790으로 동일했다.

### 기본8–13프레임 특징별 MAE

| Feature | Interpolation | A12 | A20 | B12 | B20 |
|---|---:|---:|---:|---:|---:|
| `cross_density_3m` | 0.124671 | 0.191442 | 0.182485 | 0.762280 | 0.668549 |
| `cross_density_5m` | 0.080301 | 0.165748 | 0.162737 | 0.705640 | 0.577526 |
| `cross_density_10m` | 0.053536 | 0.149059 | 0.133553 | 0.718301 | 0.535787 |
| `cross_min` | 0.052259 | 0.157714 | 0.114680 | 0.551491 | 0.502363 |
| `cross_nearest_mean` | 0.027583 | 0.137851 | 0.126284 | 0.610855 | 0.546013 |
| `cross_cluster_fraction` | 1.524689 | 1.930740 | 1.809554 | 8.698245 | 9.150543 |
| `centroid_gap` | 0.018447 | 0.127729 | 0.105830 | 0.572745 | 0.401399 |
| `width_sum` | 0.006472 | 0.113753 | 0.096339 | 0.496554 | 0.337756 |
| `width_abs_diff` | 0.014751 | 0.096781 | 0.084804 | 0.517003 | 0.464979 |
| `length_sum` | 0.014775 | 0.119023 | 0.112095 | 0.617810 | 0.488828 |
| `length_abs_diff` | 0.026571 | 0.122396 | 0.127780 | 0.543816 | 0.548317 |
| `radius_sum` | 0.009656 | 0.128563 | 0.107732 | 0.486383 | 0.305287 |
| `radius_abs_diff` | 0.012941 | 0.112935 | 0.099613 | 0.481655 | 0.452935 |

### 기본8–13프레임 특징별 Huber 기여율

| Feature | Interpolation | A12 | A20 | B12 | B20 |
|---|---:|---:|---:|---:|---:|
| `cross_density_3m` | 1.90% | 2.17% | 2.13% | 3.48% | 2.92% |
| `cross_density_5m` | 0.70% | 1.51% | 1.57% | 3.29% | 2.49% |
| `cross_density_10m` | 0.28% | 1.25% | 1.17% | 3.36% | 2.17% |
| `cross_min` | 0.57% | 1.38% | 0.94% | 1.95% | 1.83% |
| `cross_nearest_mean` | 0.44% | 1.32% | 1.11% | 2.44% | 2.06% |
| `cross_cluster_fraction` | 95.02% | 86.61% | 87.75% | 72.31% | 79.21% |
| `centroid_gap` | 0.20% | 1.00% | 0.82% | 2.33% | 1.25% |
| `width_sum` | 0.04% | 0.66% | 0.49% | 1.60% | 0.77% |
| `width_abs_diff` | 0.16% | 0.54% | 0.45% | 1.76% | 1.56% |
| `length_sum` | 0.10% | 0.89% | 0.85% | 2.58% | 1.68% |
| `length_abs_diff` | 0.31% | 1.07% | 1.33% | 1.86% | 2.02% |
| `radius_sum` | 0.11% | 0.83% | 0.67% | 1.59% | 0.70% |
| `radius_abs_diff` | 0.18% | 0.77% | 0.72% | 1.45% | 1.34% |

13개 각각의 Huber·MAE·스키마 단위 MAE·target 수와25/50프레임 결과도 `feature_metrics.csv`에 모두 저장했다. 여기서 schema 단위는 기존 기하 비율이며 미터가 아니다.

B20의 cluster 기여는 약79.2%로 여전히 집중되어 있다. 비중이 A보다 낮아진 것은 다른12개 특징의 오류도 크기 때문이지 loss 균형이 좋아졌다는 근거가 아니다. cluster를 빼도 B20 MAE0.485811은 A20의0.121161보다 약4.01배 높다. 기본13개 특징 모두 B20의 MAE가 A20보다 높았다.

| exact terminal | 8–13 MAE | 8–13 Huber |
|---|---:|---:|
| B12 epoch12 | 1.253694 | 0.935180 |
| B20 epoch20 | 1.161984 | 0.859660 |

terminal checkpoint의 긴 mask 결과도 `terminal_comparison.json`에 보관했다. terminal 비교로도 결론은 바뀌지 않는다. 기대보다 낮은 결과를 보고 추가 trial이나 loss/scaler 변경을 반복하지 않았다.

## 7. Tracking 노이즈·관측 구성 변화

강도와 seed는 실제 평가 전 [model_b_robustness.json](../MTAH_Relational_Starter/configs/model_b_robustness.json)에 고정했다. 같은 Validation942개 중16개 간격으로 고른59개 window만 사용했다. 전체 Validation 강건성 성능으로 일반화하지 않는다. source wrapper는 원본 좌표/flags/team arrays를 복사해 가상 변경하고, 파일은 수정하지 않는다.

* 좌표 noise: 유효 관측 xy에 각각 Gaussian SD0.1/0.5/1.0m. 강도별 공통 난수를 스케일한다.
* Player dropout: 관측마다0.1/0.2/0.3 확률로 제거. 공통 uniform 난수로 큰 dropout에 작은 dropout이 포함되게 한다.
* Permutation: 매 프레임 좌표·team·parse flag를 함께 순열 변경. 고유 인물 ID는 만들지 않는다.
* Team swap: H/A 코드를 뒤집고 대칭 scene과 ordered team 표현을 함께 검사한다.
* Composition: 각 window index65–85에서20% random 관측 제거. 프레임별 집합 변화이며 동일 인물 궤적 연결이 아니다.
* Physical shift 대조: 같은65–85 시점에 관측 Home 전체를 Away 중심 방향으로1m 이동. 그 크기에서의 공간 변화 반응과 composition 반응을 비교한다. 실제 선수의 정답 움직임이나 경합 label이 아니다.

모든 perturbation 뒤 A/B에 같은 기존 bounds policy를 적용한다(2m 초과 제외, clipping 없음). A의53개 입력은 perturb한 visible 원시 관측으로 다시 계산하며, B도 그 관측으로 graph를 다시 만든다. 학습된 표현을 재사용하지 않는다. artificial hidden은 source 접근 guard로 막고 hidden target는 원본 관측의 loss-only 값으로 고정한다.

표현 변화는 artificial mask가 없는 같은 window에서 측정하고, 복원 robustness는 기존 고정8–13 mask의 visible 입력만 perturb하여 측정한다. Physical shift는 실제 geometry를 바꾸므로 원본 target에 대한 불변 복원 오차를 평가하지 않고 **표현 변화만** 보고한다. 일부 노이즈가 원본 오차를 줄여도 tracking 정확도가 좋아졌다는 뜻은 아니다.

실제 robustness 실행 시간은93.58초였다. 아래는 **동일59개 window**의 clean 관측을 기준으로 한 표현 변화다. `L2/std`는 원래 표현의 차원별 분산 합 제곱근으로 나눈 평균 L2로, 모델별 표현 스케일 차이를 일부 보정한 민감도다. 실제 확률이나 오차율은 아니다.

| 변형 | B12 graph L2/std | B20 graph L2/std | A20 temporal L2/std | B20 temporal L2/std | A20 masked Huber | B20 masked Huber |
|---|---:|---:|---:|---:|---:|---:|
| clean_0 | 0.000000 | 0.000000 | 0.000000 | 0.000000 | 0.115083 | 0.765144 |
| noise_0.1 | 0.014411 | 0.031256 | 0.075577 | 0.037412 | 0.092287 | 0.766993 |
| noise_0.5 | 0.050252 | 0.109738 | 0.351846 | 0.122362 | 0.257045 | 0.771032 |
| noise_1 | 0.081901 | 0.170993 | 0.482675 | 0.212731 | 0.392609 | 0.774911 |
| dropout_0.1 | 0.146227 | 0.265075 | 0.233701 | 0.281848 | 0.430507 | 0.774700 |
| dropout_0.2 | 0.264426 | 0.459533 | 0.562686 | 0.535861 | 0.800835 | 0.804145 |
| dropout_0.3 | 0.382853 | 0.658573 | 0.791305 | 0.799043 | 1.164022 | 0.847519 |
| composition_0.2 | 0.035905 | 0.063031 | 0.572337 | 0.512896 | 0.202540 | 0.766542 |
| physical_shift_1 | 0.005844 | 0.019430 | 0.349099 | 0.202330 | 대상 아님 | 대상 아님 |

clean은 제공된 원래 tracking 관측이며 정확한 선수 위치 정답이라는 뜻이 아니다. A12/B12까지의 full 수치·정규화 전 L2·p95·cosine 변화는 `robustness/report.json`에 저장했다.59개 표본의 clean Huber를 전체942개 Validation 값과 섞어 비교하지 않는다.

**Permutation / Team swap:** 실제 Validation 모두 scene·temporal·ordered-team 불변/교환 검증을 통과했다. permutation에 따른 graph 성분 최대 차이는 B12 약1.19e-6, B20 약1.91e-6이었고, team swap의 대칭 scene 차이는0이었다. 사용 tolerance는 atol5e-5/rtol1e-4다.

**민감도:** B20은 좌표 jitter1m에서 temporal L2/std 약0.213, dropout30%에서 약0.799였다. 이 설정에서 좌표 jitter보다 누락에 민감했다. dropout30%의 masked Huber는0.765141→0.847522로 증가했다. A보다 상대적인 증가가 작아도 B의 clean 오차 바닥이 훨씬 크므로 tracking 성능이 우수하다고 해석하지 않는다.

**관측 변화와 공간 변화:** 같은21프레임 span에서20% composition 제거의 B20 중심 temporal 변화는 L2/std 약0.513, Home1m shift는 약0.202였다. 사전 고정한 이 서로 다른 강도에서 관측 구성 변화에도 크게 반응한다는 단서다. 두 perturbation의 물리적 강도가 같다는 뜻도, 실제 경합을 구별했다는 뜻도 아니다.

일부 A/B 조건에서 작은 noise나 dropout이 오차를 낮추는 비단조 결과도 있었다. 원래 noisy tracking,59개 표본, 모델 bias와 target 구조가 함께 작용할 수 있다. 이를 유리한 강도 선택이나 정확도 개선의 근거로 쓰지 않았다.

## 8. 임베딩·추론 시간·메모리

Validation 블록 안에151프레임이 모두 들어가는23,528개 중심에서 artificial mask 없이 다시 추론했다. 반·split을 넘는 배포 embedding은 사용하지 않았다. 각 모델의 temporal64, B의 graph64·ordered H/A32×2·team-observed·원본 center frame mapping을 로컬에 저장했다. 품질 유효17,057개와 bad6,471개를 구분했다.

같은 Validation 중심의 temporal embedding 통계다. PC 기반 effective rank는 공분산 eigenvalue의 entropy 지수이며, 차원 수64와 동일한 의미가 아니다.

| 모델 | 유효 품질 rank | 유효 품질 평균 분산 | 유효 품질 최소 분산 | 전체 rank | 품질 불량 rank |
|---|---:|---:|---:|---:|---:|
| A12 | 2.952 | 0.00720685 | 0.00000262 | 4.752 | 4.517 |
| A20 | 3.417 | 0.00754160 | 0.00000078 | 5.355 | 5.000 |
| B12 | 3.806 | 0.04544083 | 0.00005294 | 3.580 | 2.997 |
| B20 | 9.298 | 0.03465214 | 0.00010473 | 10.248 | 8.854 |

B20은 A20보다 분산/rank가 크고 상수 collapse 징후는 없다. 그러나 복원은 더 나쁘므로 **큰 rank를 더 좋은 관계 표현이라는 증거로 쓰지 않는다.** 기존 A 보고의 전체 배포 rank3.64와 이번 Validation rank를 직접 비교하지 않는다. 차원별 분산64개·EVR·누적 EVR은 로컬 JSON에 저장했다.

| 모델 | 23,528개 중심 전체 추론(초) | warm64-window forward 평균(ms) |
|---|---:|---:|
| A12 | 2.078 | 5.695 |
| A20 | 1.923 | 4.909 |
| B12 | 12.252 | 10.292 |
| B20 | 5.477 | 10.333 |

warm 측정은 동일64개 연속 중심의 입력을 미리 pack한 뒤 첫1회 제외5회 MPS 동기화 평균이다. 실제 data loading/graph 생성 시간은 제외한다. 전체 추론은 공유 geometry cache가 순차적으로 warm되어 B12/B20 wall time 차이를 architecture나 학습 개선으로 해석할 수 없다. batch 안 frame 중복 제거를 사용한 window throughput이며 독립9664개 graph를 처리하는 시간과도 다르다.

학습 프로세스 OS peak RSS는1087.08MB, step 종료 시 관측한 MPS driver allocation 최대는1357.00MB였다. 후자는 정확한 GPU peak가 아니며, Apple unified memory에서 두 값을 단순 합산하지 않는다.

합성 MPS benchmark에서 약9천 unique graph를 처리한 최초64-window optimizer step은1.99초였다. 이웃 window 묶음으로904개 graph를 공유한 동일 형태의 warm step은약0.038–0.041초였다. 서로 다른 unique graph 수와 compile/cache 상태의 측정이므로 PyG 대비 일정 배속이나 일반화된 성능 우위를 주장하지 않는다.

로컬 PNG: [복원 비교](../data/derived/model_b_v1_20261109/figures/reconstruction_comparison.png), [학습 곡선](../data/derived/model_b_v1_20261109/figures/learning_curve.png), [분산/rank](../data/derived/model_b_v1_20261109/figures/embedding_variance.png), [추적 노이즈 민감도](../data/derived/model_b_v1_20261109/figures/tracking_noise_sensitivity.png). 웹 서버 없이 파일로 연다.

## 9. 테스트·보존·파일 및 실행

전체 **143 passed in 22.35s**(기존125 + 신규18), 실패0개다. CPU 합성 suite와 별도로 PyG/MPS 및 B 전체 batch forward/backward를 검증했고 실제 학습도 MPS에서 완료했다.

검증 범위: node/edge shape,3/5/10m와 방향 부호, kNN 동률/먼 상대 fallback, 중복/self-loop,0/1/가변 N, 순열·팀 교환과 ordered pooling, GAT/mask-token gradient, hidden raw-read guard와 target 일치, 중복 frame 인코딩의 gradient 일치, half/gap/split 경계의 기존 회귀 테스트, padding, finite 출력/gradient, checkpoint bitwise 재로드, 원본 보존 perturbation·공통 난수, 합성 end-to-end 학습/비교다.

기존 A12/A20 Validation 값을 동일 조건으로 다시 얻었고 기존 source·checkpoint를 수정하지 않았다. 제출 baseline 4종을 기존 writer로 새 로컬 폴더에 재생성한 뒤 **공식 validator에서199,319행/frame0–199318 모두 통과**, 원래 라벨과 동일함을 확인했다. GT 채점이나 볼 기반 원시 추론은 재실행하지 않았다.

모든 저장 metric와 embedding은 finite이며, 실제 mask token과 두 GAT 층 parameter가 초기값에서 변했음을 검증했다. 신규 소스·설정·실행 환경·선택 epoch·update/target 수는 로컬 report와 verification에 기록한다. 제공 경기의 실제 추적 오류 자체를 교정하거나 경합 정답으로 평가한 테스트는 없다.

시작 시 보호한 기존124개 파일 중 README의 문서 갱신1개를 제외한123개는 SHA256이 동일했다. 기존79개 로컬 산출물의 size/mtime도 그대로였고, 학습 시 snapshot에 포함된 Python source hash도 일치했다. 실제 B20 best의 graph mask token은 초기값 대비 L2 약0.2264만큼 변했다. `verification.json`에 보존·동등성·finite·회귀 검사와 현재 source SHA를 기록했다.

신규 코드 위치는 기존 nested package 아래다.

| 파일 | 역할 |
|---|---|
| [dynamic_graph.py](../MTAH_Relational_Starter/mtah_relational/dynamic_graph.py) | frame-local tie-safe kNN/radius/fallback·self-loop·edge8 |
| [graph_windows.py](../MTAH_Relational_Starter/mtah_relational/graph_windows.py) | raw-first 그래프 window, visible-only cache, 가변 노드 pack, 그룹 배치 |
| [model_b.py](../MTAH_Relational_Starter/mtah_relational/model_b.py) | 수학적으로 동일한 dense GAT, pooling, mask token, Bi-LSTM, checkpoint |
| [train_model_b.py](../MTAH_Relational_Starter/mtah_relational/train_model_b.py) | B12/B20 고정 예산 학습·Validation 비교·특징별 오차 |
| [graph_robustness.py](../MTAH_Relational_Starter/mtah_relational/graph_robustness.py) | 원본 보존 noise/dropout/permutation/team-swap/composition |
| [analyze_model_b.py](../MTAH_Relational_Starter/mtah_relational/analyze_model_b.py) | split 내부 표현·팀 pooling export·rank·timing·terminal 평가·PNG |
| [test_model_b.py](../MTAH_Relational_Starter/tests/test_model_b.py) | 신규 합성 회귀 테스트 |
| [model_b_small.json](../MTAH_Relational_Starter/configs/model_b_small.json), [model_b_robustness.json](../MTAH_Relational_Starter/configs/model_b_robustness.json) | 학습/robustness 사전 설정 |
| [requirements-model-b.txt](../MTAH_Relational_Starter/requirements-model-b.txt) | 기존 환경+PyG 의존성 |

프로젝트 루트에서 재현한다. 기존 출력은 덮어쓰지 않는다.

```bash
conda activate mtah2027
export PYTHONPATH="$PWD/MTAH_Relational_Starter:$PWD"
export PYTORCH_ENABLE_MPS_FALLBACK=1
# 의존성이 없는 환경에서만:
# python -m pip install -r MTAH_Relational_Starter/requirements-model-b.txt
python -m pytest -q -p no:cacheprovider MTAH_Relational_Starter/tests
python -m mtah_relational.train_model_b --out-dir data/derived/model_b_reproduction
python -m mtah_relational.graph_robustness --run-dir data/derived/model_b_reproduction
python -m mtah_relational.analyze_model_b --run-dir data/derived/model_b_reproduction
```

이번 실제 결과는 `data/derived/model_b_v1_20261109/`, 합성 MPS/PyG preflight는 `data/derived/model_b_preflight_20261109/report.json`이다. 로컬 `report.json`, `history.csv`, `feature_metrics.csv`, `b12/b20_best.pt`, terminal checkpoints, `robustness/report.json`, `representations/report.json`, NPY와 PNG가 산출물이다. checkpoint 로드는 `load_b_checkpoint()`를 사용한다.

```python
import torch
from mtah_relational.model_b import load_b_checkpoint

model, metadata = load_b_checkpoint("data/derived/model_b_v1_20261109/b20_best.pt", device="cpu")
model.eval()
# graph_batch는 target가 없는 GraphBatch다.
with torch.inference_mode():
    out = model(graph_batch)
    temporal = out.embedding[:, 75]        # [B,64]
    home = out.team_embedding[:, 75, 0]    # [B,32]
    away = out.team_embedding[:, 75, 1]    # [B,32]
```

## 10. 결론과 다음 Fusion 실험을 위한 준비

**Q1. B가 A보다 선수 간 공간 관계를 효과적으로 표현하는가?**

현재13개 공간 특징의 복원 기준으로는 **아니다.** B20은 A20보다 모든 기본 특징의 MAE가 높고, cluster 제외 평균도 나빴다. rank는 더 크지만 유용한 경합 관계를 더 잘 표현한다는 근거는 아니다. 올바른 불변성과 팀별 표현 인터페이스를 구현했다는 점과 실제 표현 효용을 구분한다. batch grouping·checkpoint 선택 차이도 있어 architecture의 보편적인 우열로 확장하지 않는다.

**Q2. Tracking 노이즈·관측 결측이 있을 때 얼마나 안정적인가?**

순열/팀 교환에는 수치 tolerance 안에서 안정적이었다. 좌표 jitter에는 비교적 작은 표현 변화가 있었지만 dropout과 단기 관측 구성 변화에는 민감했다. 특히 composition 변화가 중심 embedding을 크게 움직였다. 작은 성능 저하만으로 강건성을 단정하면 이미 높은 clean 복원 오차와 표현의 낮은 반응성을 놓칠 수 있다. 이 결과는 고정한 합성 noise에 대한 민감도이며 실제 tracking 정확도·경합 recall·점유 개선을 측정한 것이 아니다.

현재 비교는 단일 경기·단일 seed·사전 고정한 소형 구조·20 epoch 예산이다. 유효 관측 target만 학습하므로 tracking이 어려운 구간의 경합 recall을 측정한 것이 아니다. 실제 선수 ID 오류·팀 오분류의 원인을 교정하지 못했고, mean/softmax pooling의 관측 수 정보 손실·두 층의 제한된 관계 전파·작은 cluster IQR의 loss 집중이 병목일 수 있다. 이는 가능한 원인이지 이번 실험으로 인과적으로 분해한 결론은 아니다. B를 유리하게 만들기 위한 추가 hyperparameter 탐색이나 loss 재가중은 하지 않았다.

다음 실험은 A/B와 interpolation을 독립된 기준으로 보존하고, 같은 split·mask·target·예산에서 **A+B와 B+interpolation을 별도 사전 등록 ablation**으로 비교하는 구성이 적절하다. 어느 신호를 Fusion에 채택할지는 이번에 결정하지 않았다. 낮은 복원 손실이나 큰 rank만으로 branch를 선택하지 말고, 품질·dropout에 대한 gate 민감도와 ordered `z_H,z_A` 보존도 함께 확인해야 한다. loss 균형을 바꾸려면 Train 통계로 정의한 동일한 새 손실로 A/B를 공통 재학습하는 별도 실험이어야 한다.

Graph/Feature Fusion, 비지도 군집화, 실제 경합 확률, H/A/D decoder는 구현하지 않았다. 현재 B는 비교 가능한 독립 branch와 입력/출력 인터페이스이며 점유 개선이 확인된 완성 모델이 아니다.
