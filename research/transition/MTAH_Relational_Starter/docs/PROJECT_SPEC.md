# MTAH 2027 Project Specification — Ball-Free Contest Candidate & Possession Context

**작성 기준:** 2026-11-09, 공식 MTAH GitHub 문서와 대화에서 결정된 설계를 구분해 기록. 실험 데이터는 아직 실행 검증되지 않았다.

## 0. 한 문장 목표

**Metrica ATD의 선수 위치와 시간에 따른 선수 간 관계·팀 구조 변화만으로 경합 가능성이 높은 구간을 찾아, 그 전후 시계열을 통해 점유 유지/전환을 추론하는 비지도·자기지도 학습 파이프라인을 개발한다.**

경합 검출은 최종 목적의 한 모듈일 뿐. 대회 최종 산출물은 매 프레임의 `H`, `A`, `D`, `X`이다.

## 1. 문제 정의와 공식 데이터

### 대회 목적

- Barcelona vs Athletic Club, 한 경기 tactical source, 총 **199,319 frames**, **25fps**.
- 정답 대상 구간: 1H **11185–90651**, 2H **110808–190197**; 나머지는 비평가 구간.
- 제출: `frame,label` 헤더, `0..199318` 프레임 순서, 정확히 199,319 rows.
- `H`=홈팀 통제 점유 중 live ball, `A`=원정팀 통제 점유 중 live ball, `D`=dead ball, `X`=abstain.
- **Contested는 공식 네 가지 출력 상태 중 하나가 아니다.** 경합/루즈볼이라도 볼이 in play라면 마지막 통제 팀의 점유가 유지된다. 상대 팀의 실제 통제 확보에 따라 H↔A 전환.
- 1차 ranking metric은 `spellF1`, `macroF1`은 tie-break. 짧은 스위치 오검출은 큰 감점 요소가 될 수 있다.

### 현재 입력

- 사용: `data/tactical/BAR-ATH_tactical_atd.xml`, `..._atd.txt`, `games.py`.
- 선수 좌표는 0..1 정규화된 피치 좌표. 공식 코드에서 105.3×68.0m로 환산.
- 선수 팀 메타데이터 존재, **물리적 선수 신원 정보는 없음**. 46개 슬롯은 분절 tracklet.
- 평가 프레임에 평균 20.5명 검출; 22명 이상 중복 검출도 발생할 수 있음. NaN 및 선수 ID 바뀜.
- 볼 좌표는 전체 56.9%, 평가 구간 59.3% 프레임에서 관측. **현재 경합 감지에서는 쓰지 않는다.**
- Smart Tagging은 이번 설계에서 제외한다(사용자 결정, 신뢰도 우려). 경기의 broadcast 영상은 별도 비동기 타임라인이므로 1차 범위에서 제외한다.

## 2. 연구 질문

Q1. 볼 없이 단순한 거리·군집·팀 간 관계 변화로 경합 후보를 포착할 수 있는가?

Q2. 현재 프레임의 상대 선수 간 거리뿐 아니라 전후 ±3초 동안 **접근 → 밀집 → 이탈/구조 변화** 패턴이 시계열 특징을 개선하는가?

Q3. raw player set graph 표현과 engineered spatial feature가 자기지도 학습에서 상호 보완적인가?

Q4. 발견된 경합 후보가 점유 스위치 모듈에 유의미한 **전환 신뢰도/억제 단서**를 제공하는가? (정답 없이는 확정 불가)

## 3. 단계별 아키텍처

```text
MTAH tactical ATD XML/TXT
     |
     v
Preflight & QC
(frame_id, team slot metadata, NaN, duplicate flags, half boundary)
     |
     +--------------------+---------------------------+
     |                    |                           |
     v                    v                           v
Raw observed nodes       Engineered spatial          Optional ball-data branch
(variable N, team)       features & graph stats      (not used for contest)
     |                    |
     v                    v
Permutation-invariant    Feature MLP/Encoder
Dynamic Graph Encoder    (distances/structure)
     |                    |
     v                    v
Bi-LSTM A                Bi-LSTM B
     +--------------------+
                  |
       Gated/Attention Fusion
                  |
       Self-supervised representation
       (joint masked-span reconstruction)
                  |
       Unsupervised grouping
       (GMM / HDBSCAN as experiments)
                  |
       Contest CANDIDATE score
                  |
     Later: possession-state model
  (separate D detector + H/A temporal decoder)
                  |
         submission frame,label
```

**Phase 1 현재 구현**은 공간 통계 계산 및 시각화만 한다. 나머지 블록은 목표 설계이며 개발·평가가 필요하다.

## 4. 공간적 지표 사양

### Relative player distances

서로 다른 팀 선수 i,j 사이 거리를 `d_ij = sqrt((x_i-x_j)^2 + (y_i-y_j)^2)`로 계산. 이미 미터 단위 변환된 좌표를 사용할 것.

- `cross_pairs_3m = count(d_ij <= 3)` 초기 핵심 지표.
- `cross_pairs_5m` 및 `cross_pairs_10m`: 사전 접근·압박 맥락.
- `cross_density_3m = cross_pairs_3m / (N_H*N_A)` 관측 선수 수 변동 보정.
- `cross_min_m`: 가장 가까운 상대 선수 간 거리.
- `max_cross_cluster_3m`: H-A 간 근접 관계를 bipartite graph로 만들었을 때 최대 연결 요소에 포함된 양 팀 선수 합계. **경합의 직접 정답이 아님.**

### Intra-team / Collective shape

- 팀 중심(center x,y), 팀 폭(y range), 팀 길이(x range), 팀 bounding-box area, 중심까지 평균 거리(radius), 양 팀 중심 간 거리.
- 단순 폭/길이는 한 선수가 누락되면 변하므로 `n_home`, `n_away`, `low_coverage`, `overcount`를 함께 기록.
- 향후 핵심: 최근접 동료 거리, k-NN 팀 관계, edge-weighted cohesion, quantile-based width/length, 48구역 점유 분포, 중심 이동 벡터.

### Temporal relation evolution

- `cross_pairs_3m`의 ±1초 차이 및 ±3초 window.
- 교차 그래프 연결 밀도 및 연결 변화.
- 팀 중심 이동 방향, 팀 구조의 수축·확장, 상대 간 거리 감소 속도.
- track slot ID 기준 edge turnover는 **ID switch**에 민감하므로 기본 특징으로 바로 쓰지 않고, 두 시점의 **그래프 통계량** 차이로 먼저 대체한다.

**중요:** 48구역 분할은 카운트 및 시각화용. 3m 거리 계산은 항상 연속 좌표를 사용해 구역 경계의 불연속성을 피한다.

## 5. 모델 상세

### A. Raw Dynamic Graph stream

- 프레임 t마다 관측 선수들을 노드로 구성: `node = [x, y, team, observed]` (+ 선택적으로 전후 위치의 유사도 또는 speed).
- 같은 팀 노드 간 intra-team edge, 다른 팀 간 inter-team edge를 구분.
- sparse kNN + radius weighted edge 또는 attention: node 수가 작으므로 O(N²) 관계 행렬도 가능.
- Graph Encoder (GNN / GAT / Set Attention)는 입력 선수 **순서 불변(permutation-invariant)** pooling 후 **고정 길이 시점 임베딩**을 생성.
- 46개 슬롯 인덱스를 46명의 물리적 선수 신원으로 인식하지 않는다. 장시간 1인 궤적을 암묵 가정하지 않는다.

### B. Engineered Feature stream

- 4절 지표를 robust scaling 및 missingness mask와 함께 인코딩.
- 상대 근접 지표, 팀 폭/길이, 클러스터 크기, 전체 이동 방향 등의 신호.

### C. Dual Bi-LSTM

- 실험 초깃값: 컨텍스트 현재 ±3초 = **151 frames** at 25fps.
- raw graph embedding 시퀀스 / feature stream 시퀀스 두 인코더 분리.
- 작은 hidden dim 예시 (방향당 32~64), 1~2 layers부터 시작해 한 경기 과적합 제한.
- 전체 경기 오프라인 추론에 적합; 실시간 예측은 미래 프레임을 쓸 수 없어 별도 모델 필요.
- halftime / pregame / postgame 경계와 추적 불연속 구간을 관통하지 말 것.

### D. Self-supervised training

경합/점유 정답 없이 player relationship temporal representation을 학습한다.

- 랜덤 span masking: 같은 시점의 원시 노드 위치와 파생 특징을 **동시에 마스킹**.
- reconstruction loss는 관측 가능한 실제 입력의 가려진 부분에서만 계산. missing observation은 target에서 제외.
- optional: 다음 구간 움직임 예측, agreement between stream representations (가공 특징에서 정답 누출 금지).
- 레이블 없이 학습 가능하지만 **복원 성능 != 경합 검출 정확도**.
- train/holdout은 시간적으로 이격하여 분리하고, 겹치는 시퀀스 사이 embargo(최소 window radius)를 둔다. 한 경기 특성상 독립 경기 일반화는 검증할 수 없다.

### E. Unsupervised pattern discovery

- GMM / HDBSCAN / KMeans 등을 latent embeddings에서 실험.
- 발견된 cluster는 단지 **움직임 패턴**일 뿐 `contest=True`나 `H/A/D`를 자동 의미하지 않음.
- 자동 생성 proximity-based 후보와 군집 위치를 비교하여 contested *candidate* score를 구축. 이것은 수학적으로 정의된 proxy이며 검증된 경합 확률이 아님.
- 규정상 수동 프레임 라벨을 이용해 군집에 정답을 할당하거나 튜닝하면 안 됨.

### F. Later possession state decoder

- 라이브 경기라면 `H/A`는 **마지막으로 통제한 팀**을 유지하다가 상대가 통제했다는 증거가 충분할 때 전환.
- `D`는 **볼이 경기 중단인지**의 별도 탐지. `no ball detected => D`라는 단순 규칙은 오류.
- HMM/HSMM/CRF 같은 temporally constrained decoder를 고려할 수 있으나, unlabeled inference와 초기 state 정의의 식별 가능성을 설계해야 한다.
- 향후 **볼 관측 프레임만** 보조 신호로 이용하는 별도 소스 비교 실험 고려. 현재 contest stream은 ball-free를 유지한다.

## 6. A/B/C 비교 실험

| 실험 | 입력 | 확인할 질문 |
|---|---|---|
| A | engineered feature + temporal model | 수작업 관계 특징만으로 충분한가? |
| B | raw dynamic graph + temporal model | 관계 임베딩이 추가 구조를 잡는가? |
| C | dual-stream fusion | A/B 대비 재구성·안정성 향상이 있는가? |

**비지도 평가:** 마스킹 복원 MAE/MSE, holdout 시퀀스 reconstruction, seed별 군집 안정성, permutation invariance, dropout robustness, 리듀스된 불연속 시퀀스, 모델 사이 상대적 시간 복잡도. 추론한 contested score는 **정답 없이 정량 accuracy/F1로 평가할 수 없음**.

**최종 대회 확인:** full 199,319 row CSV를 생성해야 하고, 공식 `validate_submission.py`가 통과해야 함. 공개 5분 GT는 pipeline sanity/alignment 검사 목적에만 사용; 이를 최적화 목표로 쓰지 않는다. spellF1은 비공개 full GT 제출 뒤 평가된다.

## 7. 현재 스타터 코드 상태

`mtah_relational.cli`는 다음만 구현:

- ATD XML에서 각 track slot의 홈/원정 정보 파싱.
- raw ATD text에서 NaN을 고려하여 선수 좌표 추출. 볼 좌표는 **사용하지 않음**.
- 미터 변환, 상대 선수 쌍(3/5/10m), 최대 bipartite 근접 군집, 팀 중심/폭/길이/밀집, 관측 상태 플래그.
- 프레임 ±1초 내 변화량(같은 전후반 구간에서만) 및 진단 플롯.
- 선택적으로 raw node tensor NPZ 출력. `npz`의 슬롯 축은 **개인 ID가 아닌 관측 슬롯 순서**.

미구현: 개별 선수 재식별, Graph Neural Network 학습, Two-Stream Bi-LSTM, 자동 latent clustering, 진정한 contested confidence calibration, 상태 decoder, 최종 제출 파일 생성.

## 8. 예상 실패와 관리

- tracking dropout → 선수 수가 줄면 compactness와 연결 수 왜곡. 관측 count, dropout masks 필수.
- 고강도 pressing과 contested → 공간적으로 비슷할 수 있어 군집의 의미 해석 불확실.
- 관중/리플레이? tactical ATD는 halftime에도 프레임 있으므로 period mask 필수.
- 같은 팀이 전진하고 상대가 후퇴 → 실제 점유 전환이 아닌 경우도 존재; structure change만으로 단정 금지.
- 팀 반전 방향: 홈/원정의 x 진행 방향은 전후반 교대; 점유 방향성을 활용하려면 orientation 자동 추정과 period-aware normalization 필요.
- 비지도 학습은 미리 정한 패턴이 아닌 축구적 의미를 자동 부여하지 않는다.

## 9. 규정 / 보안

- 원본 경기의 수동 possession/contest label 작성, 수동 correction, 공개 GT 기반 학습·검증·하이퍼파라미터 튜닝 금지.
- 비지도/자기지도/algorithmic pseudo-label은 허용됨.
- 해당 경기의 인간 제작 이벤트/추가 데이터는 사용할 수 없음.
- 원본/파생 tracking, video clips를 GitHub public repo, cloud LLM에 업로드하지 않음. Codex/Cloud 모델에 보내는 것은 **코드와 설명**으로만 제한; 실제 경기 파일은 로컬 VS Code terminal에서 실행.
- 최종 CSV는 GitHub 코드로 재현 가능해야 하며 기술 문서는 입력·제외 데이터·임계값·제약 설명 포함.

## 10. 공식 참고

- https://github.com/metrica-sports/more-than-a-hack-2027/blob/main/CHALLENGE.md
- https://github.com/metrica-sports/more-than-a-hack-2027/blob/main/DATA.md
- https://github.com/metrica-sports/more-than-a-hack-2027/blob/main/EVALUATION.md
- https://github.com/metrica-sports/more-than-a-hack-2027/blob/main/RULES.md
- https://github.com/metrica-sports/more-than-a-hack-2027/blob/main/SUBMISSION.md
