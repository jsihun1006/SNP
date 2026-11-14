# 외부 코드 → MTAH 구현 설계 전환표

**읽기 순서:** `EXTERNAL_CODE_STUDY_GUIDE.md` → 이 문서 → `PROJECT_SPEC.md` → `IMPLEMENTATION_PLAN.md`.

이 파일은 Codex가 논문과 코드의 세부 기능을 **기존 MTAH 프로젝트 내 새 파일로 어떻게 옮길지** 정리한 기술 설계다. 아래 파일·클래스는 구현 제안이며 **아직 존재하지 않을 수 있다**. 기존 패키지의 현재 구현은 `mtah_relational/cli.py`와 `features.py`에 있는 공간 특성 추출뿐이다.

## 1. 무엇을 채택할지 결정

| 외부 소스 | 추출한 알고리즘 개념 | 우리 쪽 설계 | 우선순위 | 하지 않을 것 |
|---|---|---|---|---|
| PathCRF `models/gat_lstm.py` | `GATConv`, edge_attr, `global_mean_pool` | 팀/전 프레임 그래프 인코더 `graph_encoder.py` | 높음 | 개인 ID 고정한 temporal node-RNN |
| PathCRF `dataset.py` | per-frame graph window 및 edge features | `dataset.py`의 half-aware masked windows | 높음 | `poss`라는 감독 라벨 입력 |
| PathCRF CRF/inference | 전이 제약·Viterbi | 후속 `possession_decoder.py` / HSMM | 낮음 (후속) | supervised CRF loss 그대로 사용 |
| Ball Radar `set_transformer/model.py` | team-set invariant/equivariant 표현 | GAT과 비교할 `set_encoder.py` baseline | 중간 | supervised player-possession head |
| Ball Radar `models/player_ball.py` | macro/micro + Bi-LSTM | raw/engineered stream temporal encoder | 높음 | ground-truth ball regressor를 contest 학습으로 오용 |
| Borghesi 논문 | TCN과 permutation-invariant pooling | `temporal_tcn.py` 대조 실험 | 중간 | 지도학습 결과 수치 재사용 |
| football-cv `team_tactical_metrics_v2.py` | centroid/width/length/compactness | `features.py` 지표·강건성 개선 | 즉시 | 완전 10v10 프레임만 남기기 |
| football-cv `inter_team_spatial_metrics.py` | H-A cdist/nearest/hull | 교차 그래프 거리/최근접 지표 | 즉시 | 누락/중복을 결측 없는 표본처럼 취급 |
| Tryolabs | possession smoothing/hysteresis | 최종 상태 전환 최소지속시간 후보 | 후속 | 45px 볼 거리/20프레임 값을 3m 경합에 사용 |
| AAU/Roboflow | ball-dependent possession heuristic | 당장 채택 없음 | 제외 | 계획 문서나 미병합 PR을 검증된 모델로 사용 |

## 2. 입력 파이프라인과 데이터 계약

**실제 원본:** `data/tactical/BAR-ATH_tactical_atd.xml` + `.txt`, `games.py`의 metadata. `Smart Tagging`은 학습·추론에서 사용하지 않음. Contest Branch는 공 좌표도 읽지 않음.

```text
ATD 199,319 frames  (frame_id 1..199319)
  └── parse & frame alignment (video_frame=frame_id-1)
       └── observation frame {team, x_m, y_m, speed_mps?, track_slot_qa_only}
            ├── player-quality masks: valid, missing, duplicate-suspect, n_home/n_away
            ├── per-frame graph: nodes + same-team / opposite-team edges
            └── engineered features: pair counts, nearest distances, team geometry, changes
                 └── temporal windows: centered 151 frames, never cross half or large gaps
```

### 제안 입력 자료형 (인터페이스, 구현 전)

```python
# conceptual schemas; do not assume functions/classes already exist
FrameObservations = {
    "frame": int,  # zero-based tactical video frame
    "xy_m": float32_array,  # [N_t, 2], finite observed players only
    "team": int_array,  # [N_t], 0=H, 1=A
    "speed_mps": optional_float32_array,  # [N_t] (ID/observability caveats)
    "tracking_quality": dict,  # nH,nA,missing,duplicate_suspect
}
GraphFrame = {
    "node_features": float32_array, # [N_t, node_dim]
    "edge_index": int64_array, # [2, E_t]
    "edge_attr": float32_array, # [E_t, edge_dim]
    "frame": int,
}
WindowSample = {
    "frame_ids": int64_array, # [T] with T=151 for +/-3 seconds at 25fps
    "graphs": list_of_GraphFrame, # length T; one per frame, variable N_t
    "engineered_features": float32_array, # [T,F]
    "valid_mask": bool_array, # [T] valid time positions
    "quality_mask": bool_array, # [T] reliable enough for training loss
    "self_supervision_mask": bool_array, # [T] artificially hidden span
}
```

**좌표**: MTAH는 0~1 정규화. `(x_m,y_m)=(105.3*x,68*y)`. `frame_id-1`이 제출 `frame`. 원본 `Track 1..46`은 임시 tracklet이지 인물 ID가 아님. 관측 선수 수는 고정 22가 아니다. `speed` 필드는 있지만 스칼라 속도만으로 개별 선수 이동 방향을 확정하지 않는다. Slot ID를 QA 목적으로 보유해도 model feature로 ID embedding을 전달하지 않는다.

## 3. Graph 만들기: 팀 구분 + 다중 거리 척도

1. 현재 frame에서 관측된 선수에 대해 `N_t`개 node 생성.
2. 노드 입력 제안: `x/L`, `y/W`, team indicator, 중앙선/팀 중심까지 상대 위치, availability/quality mask. 팀 공격 방향은 half/side metadata와 검증 후 사용; 추정 안 된 팀 방향을 임의 고정하지 말 것.
3. 엣지(`edge_attr`): 동일 팀 플래그, 상대 팀 플래그, 실제 미터 거리, 상대 좌표차(방향성은 좌표 기준으로 명시), 3/5/10m 인접 여부.
4. same-team은 kNN(예: 3명) + 필요시 전원 연결 비교, cross-team은 3/5/10m 다중 근접 연결 + 가장 가까운 상대 top-k를 고려. **3m를 완전한 경합 정답 임계값으로 쓰지 않는다.**
5. 고정 48구역은 추가 보조 특징/시각화일 뿐, 거리 측정에서는 연속 좌표를 사용한다.
6. 각 시점 graph에서 team pooling을 각각 수행해 `z_H[t]`, `z_A[t]`; 전체/교차관계 `z_cross[t]`도 추출. pooling은 노드 순서 불변. Missing/duplicate quality를 별도 encoding.
7. `edge turnover` 같은 시간 차분은 추적 슬롯 기반 엣지 ID를 직접 비교하면 오류가 난다. 가능하면 팀/거리 히스토그램·최근접 상대거리 분포·그래프 통계의 변화로 비교하거나 신뢰도 높은 짧은 tracklet 구간에서만 계산.

### 단순 기준 모델부터

`proximity_count_3m`, `max_bipartite_component`, `near_opponent_distance`, `team_compactness`, `centroid_gap` + 1/3초 변화량으로 unlabeled candidate score를 만드는 결정적(deterministic) baseline을 반드시 보존한다. 복잡한 모델의 이점을 판단할 비교 기준이 된다.

## 4. Two-Stream Self-Supervised Encoder (제안 구조)

```text
[GraphFrame_t] → permutation-invariant GNN pooling → z_graph[t] ──┐
                                                                  ├→ gated fusion → z[t]
[engineered features_t] → MLP/LayerNorm → z_feature[t] ───────────┘
                           (각각 별도의 Bi-LSTM 가능; 또는 fusion 후 공통 Bi-LSTM 비교)

centered 151-frame windows → masked span reconstruction → unsupervised embeddings
```

- **Model A:** engineered features → Bi-LSTM. 파라미터가 적은 기준 모델.
- **Model B:** graph → permutation-invariant pooling → Bi-LSTM. 관계 학습 효과.
- **Model C:** 두 stream → gated fusion. 두 가지 특징이 중복될 수 있어 정규화/게이트/ablations 필요.
- Bi-LSTM 전후 3초는 하이퍼파라미터 초깃값. 반박 가능한 실험 계획이지 알려진 최적 윈도우 아님.
- 단일 경기, 25fps의 인접 프레임은 상관도가 높다. shuffle split이 아닌 **연속된 구간 단위 split**을 사용하고 최소 151프레임 이상의 purge gap으로 직접 윈도우 겹침을 방지한다. 다만 한 경기만으로 경기 간 일반화는 평가 불가.

## 5. 자기지도 학습 목표: 감독 라벨 불필요

### Masked time-span reconstruction

1. 각 training window에서 연속된 시간 span (예: 0.3~0.5초)을 선택.
2. 그 시간대 원시 좌표 graph의 node/edge input과 engineered feature를 **함께 마스킹**. 모르는 정보의 자리에는 learnable mask embedding/explicit visibility mask를 둔다.
3. visible context(앞+뒤)로 masked 구간의 **frame-level spatial summary**를 재구성하도록 학습. 선수 ID가 불연속하므로 처음에는 각 선수별 정확한 좌표 재구성을 강제하지 않는다.
4. 손실은 마스킹된 타임스텝만 계산. 예: `L_mask = weighted_Huber(predicted_spatial_features, original_spatial_features)`; train mask와 자연 결측 mask를 구분하고, 품질 저하 관측은 loss 가중치를 낮춘다.
5. `raw nodes`에서 파생된 target features는 **loss 계산에만 사용**한다. 같은 시점 원래 수치를 다른 stream input으로 주지 않는다.
6. 동작을 검증한 후 움직임 방향 변화/접근 전후와 같은 auxiliary proxy target을 추가할 수 있으나, ground-truth contest label이라고 부르지 않는다.

### 평가와 해석

- 허용되는 지표: holdout masked-reconstruction MAE/Huber, 예측할 features의 유효 관측률, embedding stability, 인위적 노이즈/선수 누락에 대한 민감도, deterministic baseline 대비 후보 안정성.
- 알 수 없는 지표: **실제 경합 Precision/Recall/F1** (정답 경합 라벨이 없음); **점유 spellF1 개선** (별도 공식 비공개 평가 필요).
- 순수 군집 클러스터의 0/1/2 ID는 경합/비경합의 의미가 아니다. 선택된 후보를 관계지표 분포로 **사전 정의된 투명한 규칙에 따라 해석**하되 특정 경기 프레임을 보고 결과를 수작업 수정하지 않는다.
- `contest_score`는 일단 **score**이고, 확률 보정 없이는 `p(contest)`라고 표기하지 않는다.

## 6. 최종 점유 추론과 인터페이스를 분리

```text
[player-only contest/structure encoder] → contest_score[t], structure_change[t]
                                               |
[future ball evidence / ball masking model] ----+→ possession evidence H/A
                                               |
[dead-ball model (separate)] -------------------+→ H/A/D decoder
                                                      ↓
                                           frame,label CSV (199,319 rows)
```

- **contest:** 선수들이 충돌·압박·루즈볼에 반응할 수 있는 불확실 구간. 공이 살아 있다면 `D` 아님.
- **possession:** 반대 팀이 controlled possession을 확보할 때만 H/A 전환. 경합 중에는 기존 소유 팀 유지하는 규칙을 모델링.
- **dead-ball:** 경기 중단. 선수 근접만으로 잘 구분되지 않아 별도 근거를 사용해야 한다.
- 공식 기준 `X`는 answer abstention; ground truth가 없는 전/후반 외 기간에 사용할 수 있다.
- 전후반 범위는 `games.half_video_blocks()`로 가져오고, halftime을 가로지르는 시계열 학습/스무딩은 금지한다.
- 제출할 때 `evaluation/validate_submission.py` 사용. 공개 5분 GT에서 수치 비교하여 모델 구조/threshold 튜닝하지 않는다.

## 7. 구현 파일 배치 및 최소 수용 테스트

| 제안 파일(신규) | 목적 | 완료 판단 |
|---|---|---|
| `mtah_relational/data_quality.py` | 결측, 중복 후보, 선수 수, 구간별 validity | 결측이 팀 응집도에 끼치는 영향 표시 |
| `mtah_relational/graph_data.py` | variable-N graph, 팀/거리 엣지 | 노드 순열 시 그래프 불변 통계 테스트 |
| `mtah_relational/windows.py` | half-aware 151-frame windows | 후반·하프타임 경계 교차 0건; split purge |
| `mtah_relational/graph_encoder.py` | Graph GNN/Set Pooling | 관측 선수 순열에 output 변화 미미 |
| `mtah_relational/temporal.py` | Bi-LSTM encoders | 그래프/feature 두 stream shape 및 mask 테스트 |
| `mtah_relational/self_supervised.py` | joint masking + masked-summary loss | masked span 누출 테스트, NaN loss 없음 |
| `mtah_relational/cluster.py` | GMM/HDBSCAN 또는 baseline anomaly score | 재현성(seed), 군집 품질 진단, 미보정 score |
| `mtah_relational/visualize.py` | pitch graph, score, quality overlay | 영상 라벨을 만들지 않고 디버그 그래프 확인 |
| `mtah_relational/possession_decoder.py` | 추후 점유 상태 모델(별도 단계) | 팀 controlled possession/last owner/`D` 구분 |

**주의:** 이 표는 구현 예정 명세다. 현재 실제 파일은 `features.py`, `cli.py`, tests 등만 있다. Codex는 가짜로 실행 완료를 보고하면 안 된다.

### 반드시 추가할 synthetic/unit tests

1. 22개 선수를 섞어 입력해도 pooled output이 동일(허용 오차 내).
2. 홈/원정 색/코드만 맞바꾸면 contest score는 같고 팀별 성분은 서로 맞바뀜.
3. 3m/5m/10m 동심 경계에서 pair count monotonic (`count3<=count5<=count10`).
4. 자연 결측 및 duplicate-suspected frame은 학습 loss weight에 반영.
5. 두 stream 동일 시점 joint mask: 숨긴 구간의 실측 engineered 값이 모델 입력으로 유입되지 않음.
6. 전/후반 경계를 가로지르는 training window 0건.
7. 영상 index와 ATD index 정렬 `frame=frame_id-1`.
8. 다시 실행했을 때 동일 seed/config로 동일 feature 파일 (수치 허용 오차 및 deterministic 세팅 내).

### 추천 실행 순서

1. 기존 CLI: `python -m mtah_relational.cli check`.
2. 기존 CLI: `python -m mtah_relational.cli extract --start 11185 --count 5000 --save-nodes`.
3. `python -m mtah_relational.cli plot`.
4. `python -m pytest -q tests/test_features.py tests/test_atd_parser.py tests/test_cli_integration.py`.
5. 위 1~4가 로컬에서 확인된 후에만 Graph Dataset/Encoder 및 자기지도학습 추가.

## 8. Codex가 먼저 제출할 기술 리뷰

구현 전에 `docs/CODEX_RESEARCH_OUTPUT.md`에 다음 내용으로 **외부 코드 학습 결과**를 정리한다.

- 각 출처별 입력 tensor shape, 출력 shape, 사용하는 supervised target, 자료 누락 조건.
- `models/gat_lstm.py`와 `models/player_ball.py`의 분기·pooling·Bi-LSTM 차이.
- 우리 그래프가 permutation-invariance를 유지하는 설계.
- **실제 데이터 없는 상태에서 할 수 있는 synthetic 검증**과 로컬 데이터가 있어야 가능한 진단.
- 현재 `features.py`/`cli.py` 재활용 또는 수정 목록(변경 이유 포함).
- 최소 GNN/Set 대안의 파라미터 수와 예상 계산 비용은 **코드 계산으로** 보고하고, 근거 없는 추정치는 구분.

마지막으로 **'원본 코드에서 직접 포팅한 부분 / 아이디어만 참고해 새로 구현한 부분 / 미사용 부분'**을 구분해 라이선스/재현성 노트를 남긴다.
