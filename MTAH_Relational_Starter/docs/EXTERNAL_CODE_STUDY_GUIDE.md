# Codex 외부 코드·논문 심화 학습 노트

> 목적: 이전 대화에서 검토한 GitHub 코드 6개 항목 및 2023년 논문을 **실제 소스 파일 수준**에서 비교하고, MTAH Phase 1 선수좌표 기반 경합 감지/점유 추론으로 옮길 수 있는 개념을 식별한다.
>
> 확인 기준일: **2026-11-09**. GitHub 파일 경로와 동작은 공개 코드·README를 직접 대조했다. 아래 링크는 검토한 커밋으로 고정했다. **우리 데이터에서의 성능을 검증했다는 뜻이 아니다.**
>
> 코드 전체를 복제하지 않았다. 라이선스와 출처를 확인하고 개념만 독립 구현한다. 실제 외부 코드를 읽고 싶으면 `scripts/fetch_reference_sources.sh`를 사용한다.

## A. 먼저 읽을 우선순위

| 순서 | 출처 | 직접 읽을 코드/자료 | 이번 프로젝트에 도입할 핵심 | 판단 |
|---|---|---|---|---|
| 1 | **PathCRF (2026)** | `dataset.py`, `models/gat_lstm.py`, `models/set_lstm.py`, `models/dynamic_sparse_crf.py`, `inference.py` | 선수 관계 그래프, set attention, 시계열 일관성 | **핵심 참조** |
| 2 | **Ball Radar (2023)** | `set_transformer/model.py`, `models/player_ball.py`, `train.py`, `postprocessor.py` | permutation-invariant/equivariant, Bi-LSTM, 계층적 추론 | **핵심 참조** (학습 방식은 교체) |
| 3 | **Borghesi et al. (2023)** | 논문 원문, abstract / methods | 선수 궤적 인코딩, attention aggregation, dead/alive 분리 | **핵심 참조** (TCN 대안) |
| 4 | **football-cv-analytics** | `src/analytics/team_tactical_metrics_v2.py`, `src/analytics/inter_team_spatial_metrics.py`, `src/analytics/identity_qa.py`, `src/possession/final.py` | 팀 구조 지표, 결측 QA, `controller` vs `owner` 구분 | **특징 공학 참조** |
| 5 | **Tryolabs** | `soccer/match.py`, `run.py` | 상태 지속시간/히스테리시스 개념 | **중계영상 후순위** |
| 6 | **AAU footballtracking** | `docs/implementation_plan.md`, `src/exporters/data_exporter.py` | 영상 calibration/볼 근접 후처리 설계 | **현재 미사용** |
| 7 | **Roboflow sports PR #36** | pull request 설명/변경 내역 | 거리 기반 possession smoothing 제안 | **미병합·현재 미사용** |

**중요한 원칙:** 참고 원본 대부분은 라벨이나 볼 관측을 이용한다. **이들 코드를 그대로 실행하는 것이 비지도학습은 아니다.** 우리가 구현해야 할 것은 `self-supervised representation learning + unsupervised pattern discovery`이며, 공식 H/A/D 점유 정답을 직접 학습하는 것과 다르다.

---

## B. PathCRF — player graph + temporal constraints

- 저장소: https://github.com/hyunsungkim-ds/pathcrf
- 논문: https://arxiv.org/abs/2602.12080
- 검토 커밋: `1b7c6f85a6b78c92e4543fde72ca2b86a933f997`
- 주제: **공 트래킹 없이 선수 궤적으로 점유 경로를 추론**하고 on-ball event를 식별.
- 원래 학습 데이터: Sportec DFL 추적·이벤트/점유 정보를 전처리한 **라벨 포함** 데이터. MTAH 단일 무라벨 경기와 다르다.

### B1. 코드 읽기 경로 및 실제 역할

1. [`dataset.py`](https://github.com/hyunsungkim-ds/pathcrf/blob/1b7c6f85a6b78c92e4543fde72ca2b86a933f997/dataset.py) — `SoccerWindowDataset`: 경기 구간, `window_seconds`, `sample_freq`, 입력 특징, 팀 순서, 별도 외부(outside) 노드, 정답 보유 경로를 윈도우화한다. `SoccerWindowGraphs.__getitem__`은 각 시점의 PyTorch Geometric `Data(x, edge_index, edge_attr)` 그래프를 리스트로 내보낸다. `edge_attr`는 팀 관계와 위치 거리 등으로 구성된다. **정답 `poss`를 함께 반환하므로 무라벨 프로젝트에 그대로 사용하면 안 된다.**
2. [`models/gat_lstm.py`](https://github.com/hyunsungkim-ds/pathcrf/blob/1b7c6f85a6b78c92e4543fde72ca2b86a933f997/models/gat_lstm.py) — `GATEncoder`: PyG `GATConv` → `LayerNorm` → `ELU`, 필요 시 `global_mean_pool`. `GATLSTM`은 노드/그래프 임베딩에 RNN(옵션: Bi-LSTM/Transformer)을 적용하고, macro(점유 관련 보조 판별)와 micro(볼/점유 edge) 출력을 만든다.
3. [`models/set_lstm.py`](https://github.com/hyunsungkim-ds/pathcrf/blob/1b7c6f85a6b78c92e4543fde72ca2b86a933f997/models/set_lstm.py) — `SetLSTM`: 팀별 `SetTransformer`/전체 세트 임베딩 + 시계열 `nn.LSTM(bidirectional=True)`와 micro head. `micro_type="poss_edge"`인 경우 CRF 분기를 결합한다. **팀별 인코더가 있는 구조**를 참조한다.
4. [`models/dynamic_sparse_crf.py`](https://github.com/hyunsungkim-ds/pathcrf/blob/1b7c6f85a6b78c92e4543fde72ca2b86a933f997/models/dynamic_sparse_crf.py) — `DynamicSparseCRF`: 가능한 edge-state 전환을 `_build_allowed_next`로 한정한다. `neg_log_likelihood`는 **정답 tags**를 필요로 하며 `decode`는 Viterbi류 전역 경로 탐색에 사용한다. `models/dynamic_dense_crf.py`도 대조.
5. [`inference.py`](https://github.com/hyunsungkim-ds/pathcrf/blob/1b7c6f85a6b78c92e4543fde72ca2b86a933f997/inference.py) — 겹치는 시퀀스 예측 병합, 후보 edge 전이 제약, greedy/Viterbi decoding, 이벤트 생성 과정 참조.
6. [`train.py`](https://github.com/hyunsungkim-ds/pathcrf/blob/1b7c6f85a6b78c92e4543fde72ca2b86a933f997/train.py) — **지도학습의 경계가 어디인지** 확인할 파일. 기존 target/loss를 자기지도 목적함수로 대체해야 한다.

### B2. 우리가 가져올 부분

- **순서 불변 관계 표현:** 노드의 배열 위치를 실제 선수 신원으로 믿지 않는다. `edge_index` 재매핑 후 노드 순서를 무작위 섞어도 프레임 pooled embedding이 거의 같아야 한다.
- **팀별 집계:** 같은 팀 내부 연결(intra-team)과 상대 팀 연결(inter-team)을 별도 구분한다.
- **구조와 시간의 분리:** 매 프레임 임베딩을 생성한 뒤 Bi-LSTM에 연결하면 개별 선수 ID 연속성이 없어도 팀 수준 시간 패턴을 모델링할 수 있다.
- **결과 연속성:** 점유 최종 단계에서 급격한 H↔A 전환을 억제하는 동적 프로그래밍/HSMM/CRF 개념을 별도 적용한다.

### B3. 그대로 쓰면 안 되는 부분

- 고정 22명+outside 노드와 연속적인 개인 identity, 선수별 이벤트·패스 edge target을 전제로 한 로직.
- 유효한 정답 edge 시퀀스에 의해 훈련되는 CRF loss.
- MTAH의 `D=Dead Ball`을 논문 내 외부 edge로 직접 대응시키는 해석. **의미가 다르다.**

### B4. Codex가 대답할 연구 질문

1. `GATEncoder`에서 어떻게 가변 개수 노드를 하나의 고정 길이 team/graph embedding으로 바꾸는가?
2. `edge_attr`에 거리·같은 팀 여부 외에 상대 접근 속도를 넣을 경우, 선수 ID 단절을 어떻게 회피할 것인가?
3. `DynamicSparseCRF`의 edge-to-edge 제약을 **H/A/D 팀 상태**로 바꾸면 어떤 전이가 허용되며 어떤 전이가 단지 낮은 우선순위인가?
4. 유효 node가 18명인 프레임과 22명인 프레임의 representation을 어떻게 품질 마스크와 함께 사용해야 하는가?

---

## C. Ball Radar — set representation + hierarchical Bi-LSTM

- 저장소: https://github.com/hyunsungkim-ds/ballradar
- 논문: https://arxiv.org/abs/2306.08206
- 검토 커밋: `2486896cfb933ecd304489883241a00e3645ca8c`
- 연구 목적: 선수 궤적으로 공을 소유한 선수 확률을 추론하고 이를 조건으로 공 궤적을 예측한다.

### C1. 코드 읽기 경로와 역할

1. [`set_transformer/model.py`](https://github.com/hyunsungkim-ds/ballradar/blob/2486896cfb933ecd304489883241a00e3645ca8c/set_transformer/model.py) — `SetTransformer`: `ISAB` encoder; `embed_type`이 `i`로 시작하면 PMA 기반 invariant readout `(B,D)`, 그렇지 않으면 원소별 equivariant output `(B,N,D)`. **어느 연산이 순서 불변인지 정확히 이해할 것.**
2. [`models/player_ball.py`](https://github.com/hyunsungkim-ds/ballradar/blob/2486896cfb933ecd304489883241a00e3645ca8c/models/player_ball.py) — `PlayerBall`: macro player-possession 모듈과 micro ball-regression 모듈; 팀별 set 인코딩과 `nn.LSTM(..., bidirectional=...)` 구성. 시퀀스 텐서를 팀별로 분할하는 로직과 mask 사용을 읽는다.
3. [`train.py`](https://github.com/hyunsungkim-ds/ballradar/blob/2486896cfb933ecd304489883241a00e3645ca8c/train.py) — macro target에 `CrossEntropyLoss`, 볼 좌표 micro target에 `MSELoss`, 실제성 제약(`real_loss`) 등을 사용. **훈련 데이터가 있으므로 supervised 방식임을 확인**한다.
4. [`postprocessor.py`](https://github.com/hyunsungkim-ds/ballradar/blob/2486896cfb933ecd304489883241a00e3645ca8c/postprocessor.py) — 공 속도/가속도·소유 가능성 점수·움직임 조건을 활용한 물리적 후처리. 당장 경합 입력에 볼을 넣지 않으므로 후순위.
5. [`dataset.py`](https://github.com/hyunsungkim-ds/ballradar/blob/2486896cfb933ecd304489883241a00e3645ca8c/dataset.py) — 윈도우·트래킹 전처리 및 학습 데이터 제약을 이해하는 용도.

### C2. 우리가 가져올 부분

- team-aware Set Transformer 또는 더 단순한 masked pooling으로 **선수 순서에 무관한 표현**을 형성.
- `Bi-LSTM`의 양방향 시점 컨텍스트: 앞뒤 ±3초 (25fps에서 총 151프레임)는 초기 탐색 설정이지 성능 검증 결과가 아니다.
- 다음 단계 볼 위치 auxiliary 학습 시 mask-out-heldout-ball 좌표 실험. 경합 모델에는 볼 좌표를 전달하지 않음.

### C3. 그대로 쓰면 안 되는 부분

- 공 소유 선수 정답(`macro_target`) 또는 실제 볼 좌표(`micro_target`)로 학습하는 loss를 **비지도 경합 학습**이라고 부르는 것.
- mask에 전달되는 보유자/볼 target을 hidden input으로 활용하는 학습을 우리가 설계한 `raw + engineered` 동시 마스킹 복원과 동일시하는 것.
- 선수 슬롯의 위치가 매 시점 같은 물리적 선수를 뜻한다고 가정하는 것.

### C4. Codex가 대답할 연구 질문

1. player-level equivariant representation과 frame-level invariant representation 중 우리의 Bi-LSTM에 어느 것이 안전한가?
2. 학습용 마스크로 원시 좌표를 가렸다면 동일 시점의 파생 거리·폭·길이를 왜 함께 가려야 하는가?
3. 앞뒤 3초의 맥락이 경합에 유리한지 보려면 어떤 unlabeled holdout 테스트를 할 수 있는가?

---

## D. Borghesi et al. (2023) — TCN + set aggregation + dead/live 분리

- 논문: https://www.sciencedirect.com/science/article/pii/S0957417423002816
- DOI: https://doi.org/10.1016/j.eswa.2023.119780
- 소속/원문 기록: https://iris.polito.it/handle/11583/2976470
- 제목: *Using Temporal Convolutional Networks to estimate ball possession in soccer games*.
- 모델: 객체별 시계열 TCN 임베딩 → permutation-invariant 집계(합, self-attention 등 비교) → 점유 상태 분류. 단일 3분류와 `dead/alive` 우선+`home/away` 후속의 이단계 설계를 비교.
- 원 논문의 보고 수치(86.2% possession accuracy / 89.2% dead-alive accuracy)는 **해당 연구 데이터의 지도학습 평가**이며 MTAH 적용 성능을 의미하지 않는다. 원 데이터는 공개되지 않음.

**MTAH로 옮길 개념:** Graph-Bi-LSTM 외에 `team feature → temporal TCN`을 **경량 대안(ablations)**으로 검토한다. Dead-ball은 경합 자체와 분리해서 학습/추정한다. 논문의 supervised criterion은 가져오지 않는다. 현재 이 논문에 대응하는 검증된 공식 코드 저장소는 확인하지 못했으므로 구현을 가정하지 말 것.

---

## E. football-cv-analytics — engineered spatial features & QA

- 저장소: https://github.com/samuehuang/football-cv-analytics
- 검토 커밋: `dea6fb82841120aff7150872f4c9fd47c04fd9a1`
- 목적: 비디오 검출 → 추적/정제 → 공 및 점유 → 이벤트 → 팀 전술 지표의 단계별 파이프라인.

### E1. 코드 읽기 경로 및 구현 지표

1. [`src/analytics/team_tactical_metrics_v2.py`](https://github.com/samuehuang/football-cv-analytics/blob/dea6fb82841120aff7150872f4c9fd47c04fd9a1/src/analytics/team_tactical_metrics_v2.py) — 팀 centroid `(mean x, mean y)`, length/width (좌표 분위수 차이), compactness (팀 중심에서 선수 평균 거리), shape-area proxy `(length × width)` 계산. **Convex Hull area와 다른 개념**임에 주의. 이 코드에는 각 팀 정확히 10명 추적된 프레임만 처리하는 필터가 있어 MTAH에 그대로 적용하면 큰 손실이 난다.
2. [`src/analytics/inter_team_spatial_metrics.py`](https://github.com/samuehuang/football-cv-analytics/blob/dea6fb82841120aff7150872f4c9fd47c04fd9a1/src/analytics/inter_team_spatial_metrics.py) — `cdist`로 H-A distance matrix, 최소·평균/중앙 최근접 상대거리, centroid gap, convex hull 면적 등을 구한다. 역시 완전 10v10 조건을 MTAH에 맞게 바꿔야 한다.
3. [`src/analytics/identity_qa.py`](https://github.com/samuehuang/football-cv-analytics/blob/dea6fb82841120aff7150872f4c9fd47c04fd9a1/src/analytics/identity_qa.py) — track/team segment 및 switch-event QA. ID 변경 신호의 품질 진단 아이디어는 유용하지만 물리적 선수 고유 ID를 복구하는 보증이 아니다.
4. [`src/possession/final.py`](https://github.com/samuehuang/football-cv-analytics/blob/dea6fb82841120aff7150872f4c9fd47c04fd9a1/src/possession/final.py) — possession episode, transit, aerial contest, control 등으로 상태를 구분. **경합 동안 last controlled team을 유지**해야 하는 MTAH 규칙을 설계할 때 참고하되 ball-detection 전제가 강하다.

### E2. 우리가 옮길 지표 정의

- H-A 교차 거리행렬 `D[i,j] = sqrt((xH_i-xA_j)^2 +(yH_i-yA_j)^2)` (좌표는 미터로 변환).
- `cross_pairs_r = count(D <= r)` : r=3/5/10m.
- `nearest_opponent_dist_{median,mean,min}` : 관측한 선수 기준의 거리 분포. 결측/중복에 민감하므로 `n_home,n_away` 동반 기록.
- `centroid_team = mean(pos)`; `compactness = mean(||player-centroid||)`.
- `width,length`: x/y 범위를 원시 max-min과 분위수 범위 둘 다 계산하면 결측 강건성을 비교할 수 있다.
- `hull_area`: 3명 이상 비공선 점에서만 유효; `width*length`와 혼동하지 말 것.
- **추적 완전성 플래그 필수:** 구역별 밀집·team width 축소가 사실은 선수 누락 때문에 생긴 현상일 수 있다.

### E3. 코드 그대로 채택하지 않는 이유

MTAH는 전체 경기 동안 유효 선수 수가 가변이며, 추적 슬롯은 연속 인물이 아니다. 외부 코드의 sample 전용 완전 10v10 조건·tracked player ID 기반 지표를 재검토해야 한다. 예제 샘플에서의 coverage/regression consistency는 공식 MTAH 경합 감지의 benchmark accuracy가 아니다.

---

## F. Tryolabs — video/ball-dependent possession demo (후순위)

- 저장소: https://github.com/tryolabs/soccer-video-analytics
- 검토 커밋: `b00c7c75a1d52bd5ca183521a35fc76fbd89cc0e`
- [`soccer/match.py`](https://github.com/tryolabs/soccer-video-analytics/blob/b00c7c75a1d52bd5ca183521a35fc76fbd89cc0e/soccer/match.py) `Match.update`: 볼과 가장 가까운 선수를 찾고 픽셀 거리 임계값(코드 기본 `45 px`)과 연속 프레임 기준(기본 `20`)으로 현재 팀을 바꾼다. 픽셀 거리이므로 pitch 3m와 다른 척도다.
- [`run.py`](https://github.com/tryolabs/soccer-video-analytics/blob/b00c7c75a1d52bd5ca183521a35fc76fbd89cc0e/run.py) — object detection, team classifier, Norfair tracker, possession overlay 실행.
- README는 공 detector가 작은 데모 영상에 과적합된 toy model일 수 있음을 명시한다.
- **우리에게 의미:** 향후 broadcast 영상을 활용할 때 검토. 현재 tactical ATD 선수 좌표가 이미 제공되고 broadcast는 tactical과 프레임 동기화되지 않으므로 메인 Graph 학습에는 사용하지 않음.

---

## G. AAU footballtracking — implementation_plan과 실제 코드 구분

- 저장소: https://github.com/ML-interns-aau/footballtracking
- 검토 커밋: `ff14035dc9e359b9170c92463816372cbd73da87`
- [`docs/implementation_plan.md`](https://github.com/ML-interns-aau/footballtracking/blob/ff14035dc9e359b9170c92463816372cbd73da87/docs/implementation_plan.md)는 하드코딩 카메라 매핑 대체, 소유권 히스테리시스(예시 거리 1.5m·5프레임), 볼 속도 기반 패스 처리를 **계획**한다.
- 해당 계획에 등장하는 `src/pipeline/data_exporter.py` 경로와 달리, 검토한 현재 저장소의 exporter는 [`src/exporters/data_exporter.py`](https://github.com/ML-interns-aau/footballtracking/blob/ff14035dc9e359b9170c92463816372cbd73da87/src/exporters/data_exporter.py)이며, 이 파일에는 계획에 제시한 `update_passes` 함수가 확인되지 않았다.
- **결론:** 문서 내용을 구현 완료된 검증 모델로 주장하면 안 된다. camera calibration은 ATD pitch 좌표가 이미 있는 MTAH에서 불필요. 경합 감지는 ball-free로 유지.

---

## H. Roboflow sports PR #36 — 미병합 제안 (후순위)

- 링크: https://github.com/roboflow/sports/pull/36
- 확인일 **2026-11-09** 기준 `open`, `merged=false`.
- PR 설명에 볼 거리, temporal smoothing, possession percentage 및 패스 카운트/화면 오버레이 기능을 제안한다.
- **결론:** 검증된 upstream stable API처럼 사용하지 말 것. 필요하다면 diff를 읽고 아이디어만 참조할 수 있으나 현재 ball-free 경합 모델에는 사용하지 않음.

---

## I. 공식 MTAH 코드와 데이터 계약 (모든 자료보다 우선)

- 공식: https://github.com/metrica-sports/more-than-a-hack-2027
- 검토 커밋: `f8f2629ff1e0985f81d7e59d8413678514ee1530`
- [`DATA.md`](https://github.com/metrica-sports/more-than-a-hack-2027/blob/f8f2629ff1e0985f81d7e59d8413678514ee1530/DATA.md) — tactical 199,319frames/25fps, 정규화 0~1 좌표, 46슬롯, 관측 평균 20.5명, 선수 ID 불연속, 결측·중복, 볼 결측.
- [`games.py`](https://github.com/metrica-sports/more-than-a-hack-2027/blob/f8f2629ff1e0985f81d7e59d8413678514ee1530/games.py) — `GAMES['BAR-ATH']['pitch'] = (105.3,68.0)` 및 전/후반 graded frame 구간.
- [`starter/load_atd.py`](https://github.com/metrica-sports/more-than-a-hack-2027/blob/f8f2629ff1e0985f81d7e59d8413678514ee1530/starter/load_atd.py) — FIFA/EPTS XML/TXT를 `kloppy`로 읽는 정렬 확인; `video_frame=frame_id - 1`.
- [`starter/baseline.py`](https://github.com/metrica-sports/more-than-a-hack-2027/blob/f8f2629ff1e0985f81d7e59d8413678514ee1530/starter/baseline.py) — 공 관측 시 최근접 선수 팀 + centered majority smoothing. **볼 `NaN`을 `D`로 대체하는 것은 좋은 Dead Ball 판정 방식이 아님**(실제 live 상황에서도 공이 소실됨).
- [`CHALLENGE.md`](https://github.com/metrica-sports/more-than-a-hack-2027/blob/f8f2629ff1e0985f81d7e59d8413678514ee1530/CHALLENGE.md) — live 경합/루즈볼 동안 마지막 제어 팀 유지, 새로운 팀이 controlled possession을 획득할 때만 전환. `D`는 dead ball, 경합 상태가 아니다.
- [`RULES.md`](https://github.com/metrica-sports/more-than-a-hack-2027/blob/f8f2629ff1e0985f81d7e59d8413678514ee1530/RULES.md) — 제공 경기 수작업 라벨링·수동 수정·공개 GT 튜닝 금지. 자동/비지도/자기지도/알고리즘 pseudo-label은 가능. 경기 파일/재구성 가능한 파생 데이터 재배포 금지.
- [`SUBMISSION.md`](https://github.com/metrica-sports/more-than-a-hack-2027/blob/f8f2629ff1e0985f81d7e59d8413678514ee1530/SUBMISSION.md) — `frame,label` 헤더, `0..199318` 순서, `H/A/D/X`, 총 199,319행, 공식 validator 통과. 주요 순위 metric은 `spellF1`.

**MTAH phase 1과 연구 모듈의 구분:** 경합 score는 `H/A/D` 레이블이 아니다. **선수만 사용하는 경합/구조 추정**과 **볼을 활용할 수 있는 최종 possession evidence**를 구분할 것. `D` 판별은 별도 과제다.

---

## J. Codex 학습 결과 확인 체크리스트

공부를 마치면 Codex는 아래 질문을 자체 코드·문서에 근거해 답해야 한다.

- [ ] 원본 7개 항목 각각의 핵심 입력/출력/정답 의존성/리포지토리 파일 경로를 표시했는가?
- [ ] `SetTransformer`의 invariant와 equivariant를 구분했는가?
- [ ] PathCRF graph pooling 뒤 Bi-LSTM을 적용하는 이유를 ID 단절과 연결해 설명했는가?
- [ ] 어떤 target/loss가 지도학습이며 무엇으로 대체해야 하는지 설명했는가?
- [ ] 48개 구역과 3m 거리 기반 그래프 계산의 차이를 이해했는가?
- [ ] 같은 프레임의 raw·derived 특징을 동시 마스킹하지 않으면 정보 누출이 가능한 이유를 설명했는가?
- [ ] tracked count 결측/중복과 팀 compactness의 거짓 변화 문제를 고려했는가?
- [ ] 경합 score, 점유팀, `D`를 혼동하지 않았는가?
- [ ] 원 저자의 성능 수치를 MTAH 성능이라고 기록하지 않았는가?
- [ ] 현재 구현된 `mtah_relational/features.py`, `cli.py`와 새 모듈의 접점을 밝혔는가?

**평가 기준:** 새 알고리즘은 논문 코드와 비슷하게 보이는 것보다, 라벨 없이 학습 가능하고 원래 경기의 조건에서 견고하게 실행되며 결과가 재현 가능해야 한다.
