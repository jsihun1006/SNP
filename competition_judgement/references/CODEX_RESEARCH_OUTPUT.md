# MTAH 2027 외부 코드 연구 결과

작성일: 2026-11-09. 범위: 선행 연구 단계의 공개 문서·공개 소스 정적 분석과 당시 합성 검증. 실제 경기 파일, 공개 GT, 기존 경기 추출물은 이 선행 연구에서 열거나 실행 입력으로 사용하지 않았다. 이 단계에서는 이 문서와 [구현 계획](MODEL_IMPLEMENTATION_PLAN.md)만 작성했다.

이후 사용자가 승인한 1단계 구현과 실제 로컬 ATD 실행은 [PHASE1_DATA_RELATIONS.md](PHASE1_DATA_RELATIONS.md)에 별도 기록했다. 아래 기존 코드 감사·의존성/6개 검증 기록은 **수정 전 연구 당시 상태**이며, 현재 구현 완료 상태는 후속 문서를 따른다. 신경망 학습·외부 업로드는 진행하지 않았다.

**후속 실행 기록:** [시계열 Dataset·마스킹](PHASE2_TEMPORAL_DATA.md), [4단계 A 모델 구현·로컬 학습](PHASE4_MODEL_A.md). 현재는 A 모델만 학습 완료했다. 선형 보간보다 복원 성능이 낮았으며, 경합이나 점유 성능 개선을 주장하지 않는다. 아래 “미구현/미설치” 기록은 선행 연구 당시 기준으로 보존한다.

## 1. 결론

1. **PathCRF와 Ball Radar는 선수 좌표만으로 추론할 수 있지만, 원래 학습은 지도학습이다.** 점유 선수·패스 경로·볼 위치 정답 없이 원본 학습 코드를 실행할 수 없다. 볼을 입력하지 않는 것과 정답 없이 학습하는 것은 다르다.
2. **선수 순서 불변인 공간 인코더만 붙여서는 ID 불연속 문제가 해결되지 않는다.** 원본 모델의 선수별 Bi-LSTM은 시점 간 동일 노드가 동일 선수라는 전제를 유지한다. 우리는 매 프레임 관계를 집계한 고정 길이 표현을 만든 후 Bi-LSTM에 전달해야 한다.
3. 두 스트림은 같은 좌표의 다른 표현이다. 독립된 센서가 아니며, 같은 시점만 동시 마스킹해도 전후 차분·스무딩·그래프 토폴로지를 통해 누출될 수 있다. **원시 관측부터 가리고 모든 파생 입력의 의존 범위를 추적**해야 한다.
4. 비지도 군집은 움직임 패턴이다. 실제 경합, 압박, 대기 장면을 정답 없이 식별·보정했다고 주장할 수 없다. 첫 산출물은 **경합 후보 지수와 관측 품질**이며 `P(contest)`가 아니다.
5. 현재 첫 개발 단계는 모델보다 **프레임·팀·단위·가변 관측·누출 계약을 검증하는 데이터 계층**이다. 이번 요청에서는 그 구현도 진행하지 않는다.

## 2. 실제 자료 위치와 접근 기록

### 2.1 프로젝트에서 읽은 자료

요청한 설계 패키지는 루트가 아닌 [MTAH_Relational_Starter](../MTAH_Relational_Starter/)에 있다. 시작 문서와 학습 프롬프트를 확인하고 다음 문서를 읽었다.

| 자료 | 확인 내용 |
|---|---|
| [CODEX_START_HERE.md](../MTAH_Relational_Starter/CODEX_START_HERE.md), [CODEX_REFERENCE_STUDY_PROMPT.md](../MTAH_Relational_Starter/CODEX_REFERENCE_STUDY_PROMPT.md) | 연구 우선, ID·보안·자기지도 제약 |
| [EXTERNAL_CODE_STUDY_GUIDE.md](../MTAH_Relational_Starter/docs/EXTERNAL_CODE_STUDY_GUIDE.md) | 참조 커밋·주요 파일 |
| [REFERENCE_TO_MODEL_BLUEPRINT.md](../MTAH_Relational_Starter/docs/REFERENCE_TO_MODEL_BLUEPRINT.md) | 인터페이스·A/B/C 실험·수용 기준 |
| [REFERENCE_REVIEW.md](../MTAH_Relational_Starter/docs/REFERENCE_REVIEW.md) | 기존 연구 요약과 제외 범위 |
| [PROJECT_SPEC.md](../MTAH_Relational_Starter/docs/PROJECT_SPEC.md), [IMPLEMENTATION_PLAN.md](../MTAH_Relational_Starter/docs/IMPLEMENTATION_PLAN.md) | 현재 구현과 향후 설계 구분 |
| [DATA.md](../DATA.md), [CHALLENGE.md](../CHALLENGE.md), [RULES.md](../RULES.md), [EVALUATION.md](../EVALUATION.md), [SUBMISSION.md](../SUBMISSION.md) | 데이터 계약·규정·평가지표 |
| [games.py](../games.py), [load_atd.py](../starter/load_atd.py), [baseline.py](../starter/baseline.py) | 공식 구간·정렬·기존 거리 기반 접근 |
| [features.py](../MTAH_Relational_Starter/mtah_relational/features.py), [cli.py](../MTAH_Relational_Starter/mtah_relational/cli.py), [tests](../MTAH_Relational_Starter/tests/) | 현재 구현 전체와 합성 테스트 3개 파일 |

패키지의 README·의존성 목록·참조 다운로드 스크립트도 확인했다. 시작 문서의 “연구 후 Phase 1 구현” 지시는 이번 사용자의 **문서만 작성** 요청에 따라 실행하지 않는다. 설계 문서에 적힌 결측률·검출 인원 등 기존 경기 통계는 이번에 재측정하지 않았으며 검증 결과로 인용하지 않는다.

### 2.2 다운로드된 참조 저장소

로컬 기준 경로는 프로젝트의 형제 폴더 `../mtah_external_references/`이다. 각 저장소의 `HEAD`를 확인했으며 아래 고정 커밋과 일치했다. 외부 저장소의 경기 샘플·가중치는 읽거나 실행하지 않았다.

| 저장소 | 고정 커밋 | 라이선스 파일 확인 |
|---|---|---|
| [PathCRF](https://github.com/hyunsungkim-ds/pathcrf/tree/1b7c6f85a6b78c92e4543fde72ca2b86a933f997) | `1b7c6f85a6b78c92e4543fde72ca2b86a933f997` | 루트 `LICENSE`: MPL-2.0 |
| [Ball Radar](https://github.com/hyunsungkim-ds/ballradar/tree/2486896cfb933ecd304489883241a00e3645ca8c) | `2486896cfb933ecd304489883241a00e3645ca8c` | 루트 LICENSE 미확인; `set_transformer/LICENSE`는 MIT, 전체 프로젝트에 확대 적용 불가 |
| [football-cv-analytics](https://github.com/samuehuang/football-cv-analytics/tree/dea6fb82841120aff7150872f4c9fd47c04fd9a1) | `dea6fb82841120aff7150872f4c9fd47c04fd9a1` | 루트 MIT |
| [Tryolabs](https://github.com/tryolabs/soccer-video-analytics/tree/b00c7c75a1d52bd5ca183521a35fc76fbd89cc0e) | `b00c7c75a1d52bd5ca183521a35fc76fbd89cc0e` | 루트 `LICENSE.md`: MIT |
| [AAU footballtracking](https://github.com/ML-interns-aau/footballtracking/tree/ff14035dc9e359b9170c92463816372cbd73da87) | `ff14035dc9e359b9170c92463816372cbd73da87` | 루트 LICENSE 미확인 |

이번 직접 포팅은 **0개**다. 이후에도 아이디어에 근거한 독립 구현을 기본으로 한다. 코드 복사 시 파일별 고지와 의무를 따로 확인한다. 라이선스 파일 미확인은 자유로운 복제 허가를 의미하지 않는다.

### 2.3 확인 범위와 미확인 자료

| 대상 | 실제 읽은 범위 | 남은 제한 |
|---|---|---|
| PathCRF | `dataset.py`, `models/gat_lstm.py`, `models/set_lstm.py`, `models/dynamic_sparse_crf.py` 본문; `train.py:94–335, 500–580`; `inference.py`의 모델 호출·윈도우 병합·제약 Viterbi·episode 입력/출력; `models/utils.py:197–283`; `dynamic_dense_crf.py:1–135`; `datatools/preprocess.py`의 생성자·RDP·정렬 진입·touch/label 병합 | 전 저장소 전수 감사 아님. 정적 분석이며 checkpoint 추론·학습·이벤트 추출 전체 실행 미실시 |
| Ball Radar | `models/player_ball.py`, `set_transformer/{model,blocks,attention}.py`, `dataset.py` 본문; `train.py:140–217`; `models/utils.py:103–142`; `postprocessor.py`의 거리·점수·carry·보간·run 주요 분기; `datatools/{trace_helper,metrica_helper}.py`의 실행 관련 함수; 전처리 notebook의 해당 코드 셀 | notebook 저장 출력은 읽지 않음. 후처리 전 분기 실행·모든 보조 모델 검증 미실시 |
| football-cv | `team_tactical_metrics_v2.py`, `inter_team_spatial_metrics.py` 본문; `identity_qa.py:141–543`; identity-aware builder의 팀/segment/품질 함수와 main 전처리; identity-aware metrics wrapper; `possession/final.py`의 정규화·상태·hard invariant 부분 | 비디오 검출/ID 복구 전체 시스템 검증 아님. `final.py` 전체 분기 전수 검증 아님 |
| Tryolabs | `soccer/match.py:1–114`, `soccer/player.py:73–94`, `run.py` 주요 파이프라인 | draw 함수·검출기 학습 코드 전수 분석 미실시; 경기 입력 실행 없음 |
| AAU | `docs/implementation_plan.md`, `src/exporters/data_exporter.py` 본문; `src/pipeline_runner.py`의 거리·집계 연결; `src/analytics/events.py`의 상수·초기화·nearest·상태 전이; possession summary builder | 신경망 학습 모델이 아닌 휴리스틱 파이프라인. 전체 비디오/이벤트 실행 미검증 |
| [PathCRF 논문](https://arxiv.org/html/2602.12080v1) | HTML의 그래프·공간/시간·CRF·학습·데이터 준비·결론 관련 절 열람 | 참고문헌 전수 추적·수치 재현 미실시; 분석 버전은 arXiv v1 |
| [Ball Radar 논문](https://arxiv.org/html/2306.08206v1) | HTML의 PPC/BTR·손실·후처리·데이터 준비 및 GK 부록 열람 | 전체 부록 실험 재현 미실시; 코드와 논문 차이는 아래 별도 표시 |
| [Borghesi 논문](https://www.sciencedirect.com/science/article/pii/S0957417423002816) | 출판사 검색에 노출된 초록·highlights·data availability 확인 | **출판사 원문과 기관 PDF 직접 열기 403**. 전체 methods·커널/층수·정확한 학습 손실·공식 구현을 실제 읽었다고 주장하지 않음 |
| [기관 PDF](https://iris.polito.it/retrieve/d9aa6bfd-b26e-44cd-bf9e-6004e676211a/1-s2.0-S0957417423002816-main.pdf) | 검색 색인의 제목/초록만 확인 | PDF 본문 미열람 |
| [Roboflow sports PR #36](https://github.com/roboflow/sports/pull/36) | 패키지 내 기존 설명만 읽음 | **외부 PR/diff 직접 미열람, 현재 병합 상태 미확인**. 설계 근거·검증된 모델로 사용하지 않음 |

## 3. 공식 문제와 이번 연구의 경계

공식 사실은 로컬 [DATA](../DATA.md), [CHALLENGE](../CHALLENGE.md), [EVALUATION](../EVALUATION.md), [games.py](../games.py)에 근거한다.

| 항목 | 계약 |
|---|---|
| Tactical | 25fps, 199,319프레임; 제출 `frame=0..199318` |
| 원시 ATD 번호 | `frame_id=1..199319`, **`frame=frame_id-1`** |
| 선수 | 46개 소스 슬롯; 46명의 지속적 개인 ID가 아님 |
| 좌표 | 정규화 xy를 공식 pitch `105.3×68.0m`로 환산; XML의 105m와 혼용 금지 |
| 평가 전반 | `11185..90651` 양끝 포함, 79,467프레임 |
| 평가 후반 | `110808..190197` 양끝 포함, 79,390프레임 |
| 현재 입력 | 선수 xy·팀 정보·공식 구간 메타데이터. Smart Tagging·볼 좌표·broadcast 제외 |
| 출력 의미 | H/A=라이브 볼의 마지막 통제 점유 팀; D=경기 중단; X=기권 |
| 평가 | spellF1 우선, macroF1 동률 해소. 실제 경합은 공식 출력 클래스가 아님 |

기존 영상 동기화에서 Smart Tagging으로 정한 후반 시작·X 마스크를 이번 학습에 재사용하면 “Smart Tagging 미사용” 조건을 간접 위반한다. 새 파이프라인은 **`games.half_video_blocks()`**만 구간 출처로 사용한다. 후반 공식 평가 시작과 실제 킥오프 추정 시점을 혼동해 평가 구간을 잘라내지 않는다. 공식 구간 안의 모든 프레임이 live라는 뜻도 아니다.

RULES가 허용하는 입력 범위와 이번 사용자가 선택한 더 좁은 입력 범위를 구분한다. 사전학습 가중치를 대회가 일괄 금지한다고 단정하지 않는다. 다만 이번 순수 단일 경기 자기지도 실험에는 외부 라벨로 학습한 checkpoint를 가져오지 않는다. 공개 GT는 정렬·형식 검사 전용이며 모델 선택·threshold 조정·군집 이름 부여에 사용하지 않는다. 이번 연구에서 GT 파일을 열지 않았다.

공식 spellF1은 최대 연속 H/A 구간을 하나의 possession으로 보고, 정답과 예측을 temporal IoU로 일대일 대응한 F1을 **0.3/0.4/0.5/0.6/0.7 threshold에서 평균**한다. D는 possession 객체가 아니지만 구간을 나누는 데 영향을 준다. macroF1은 H/A/D의 프레임별 F1 비가중 평균이다. 평가 프레임의 X는 macroF1에서 miss이며, temporal 평가 전에는 같은 반 안에서 앞 label로 채우고 반 시작은 뒤에서 채운다. 따라서 높은 프레임 정확도와 낮은 spellF1이 함께 나올 수 있고, X가 항상 possession 단절을 만드는 것도 아니다. 이는 [EVALUATION.md](../EVALUATION.md)의 정의이며 이번에 평가기를 실행한 결과가 아니다.

기존 공식 baseline 4개는 서로 다른 모델이 아니라 **볼에 가장 가까운 선수 팀을 택하고 다수결 window를 달리한 같은 알고리즘**이다. 결측 볼·잘못된 팀/slot·공중 패스·근접과 실제 통제의 차이에 취약하며, smoothing은 짧은 오검출을 줄이는 동시에 실제 짧은 점유를 지우거나 전환을 늦춘다. 팀 폭 기반 방식은 더 긴 전술 구조의 단서지만 순간 control을 확인하지 못한다. 새 관계 모델 역시 이 한계를 자동으로 해결했다고 볼 수 없다.

## 4. PathCRF: 그래프와 경로 일관성

### 4.1 실제 입력·전처리

근거: [dataset.py](https://github.com/hyunsungkim-ds/pathcrf/blob/1b7c6f85a6b78c92e4543fde72ca2b86a933f997/dataset.py), [preprocess.py](https://github.com/hyunsungkim-ds/pathcrf/blob/1b7c6f85a6b78c92e4543fde72ca2b86a933f997/datatools/preprocess.py).

`SoccerWindowDataset`은 라벨이 포함된 Parquet에서 phase/episode를 만들고 선수 순서를 고정한다. 기본 25fps에서 `sample_freq=5`이면 5Hz, 10초는 50step이다. `poss`는 이전/다음 점유 선수와 팀, `ball`은 볼 xy 정답이다. 기본 node 8차원은 팀측/keeper 지표 2개와 xy/vx/vy/speed/accel 6개다. 별도 `ball_feature` 옵션을 켜면 실제 볼 xy를 노드에 복제하므로 현재 모델에 사용할 수 없다.

`SoccerWindowGraphs.__getitem__`은 프레임별 PyG `Data` 리스트와 `poss[T,3]`, `ball[T,2]`, meta를 반환한다. `_complete_edge_index`를 노드 수별로 캐시하며 **완전 연결 directed graph**를 만든다. `edge_in_dim>0` 분기의 실제 edge 속성은 `[same_team, Euclidean_distance]` 두 개다. “dynamic”을 반경에 따라 엣지를 생성·삭제한다는 의미로 이해하면 틀린다. 노드/관계 값이 시간에 따라 변하며 토폴로지는 대체로 완전 연결이다.

`SoccerWindowTensors`는 `[T,N,F]`를 팀당 11명과 outside 4개 위치로 패딩하고 player mask를 반환한다. 자연 결측을 있는 그대로 학습하는 구조는 아니다. feature NaN·불충분한 episode·매핑 불가 점유 선수 등을 제외한다. 선수별 속도와 keeper 추정 또한 지속적인 인물 궤적을 전제한다.

전처리 `Preprocessor`는 이벤트와 볼 궤적을 사용한다. `simplify_trajectory`는 RDP 기반 방향 변화, `align_event_rdp`는 이벤트와 후보 touch의 DP 정렬, `merge_tracking_poss`는 선수별 점유 정답을 만든다. dataset은 A→A 다음 B→B 같은 정답 경로를 A→B로 보정하기도 한다. **선수 좌표만 이용한 비지도 라벨 생성기가 아니다.** 우리에게 해당 이벤트 정답이 없으며 이 경로를 복제하지 않는다.

### 4.2 핵심 클래스와 shape

근거: [gat_lstm.py](https://github.com/hyunsungkim-ds/pathcrf/blob/1b7c6f85a6b78c92e4543fde72ca2b86a933f997/models/gat_lstm.py), [set_lstm.py](https://github.com/hyunsungkim-ds/pathcrf/blob/1b7c6f85a6b78c92e4543fde72ca2b86a933f997/models/set_lstm.py).

| 클래스/함수 | 입력 | 실제 출력·의미 |
|---|---|---|
| `GATEncoder.forward` (`gat_lstm.py:59`) | `x[ΣN,F]`, `edge_index[2,E]`, `batch[ΣN]`, optional `edge_attr[E,De]` | GATConv→LayerNorm→ELU; pooling off면 `[ΣN,D]`, on이면 `global_mean_pool`로 `[B,D]` |
| `GATLSTM.forward` (`:191`) | 길이 T의 PyG Batch 리스트 | macro 노드 logits 또는 None; ball 분기 micro `[T,B,2]`, edge 분기 `[T,B,26²]` |
| `SetLSTM.forward` (`set_lstm.py:186`) | `[B,T,N,F]`, optional 정답·random_mask | macro `[B,T,N]` 또는 sender/receiver별 logits; micro ball `[B,T,2]` 또는 edge `[B,T,N²]`; dynamic CRF용 edge embedding |
| `build_edge_compression_maps` (`models/utils.py:197`) | 선수 22, outside 4 | 원래 676 states에서 player 출발 572개+outside self-loop 4개=576 states로 압축 |
| `DynamicSparseCRF.neg_log_likelihood` (`:252`) | emission `[B,T,K]`, edge embedding `[B,T,K,D]`, **정답 tags `[B,T]`** | 평균 `logZ−gold_score` |
| `DynamicSparseCRF.decode` (`:271`) | emission·embedding·optional sequence mask | Viterbi 최적 edge IDs `[B,T]` |

GAT은 multi-head attention으로 이웃별 정보를 가중 결합하며 `concat=False`이므로 heads 수를 곱한 차원이 아니라 지정한 `out_dim`을 출력한다. **프레임 내부** 순열에 대해 node 출력은 equivariant, pool 출력은 invariant다.

하지만 `GATLSTM`의 macro 및 edge 분기는 노드 embedding과 broadcast된 graph embedding을 다시 노드별 시퀀스로 묶어 LSTM에 넣는다. `SetLSTM` 역시 `[T,B×N,D]`로 묶는 선수별 temporal 분기가 있다. 따라서 전체 시퀀스에 같은 순열을 적용하는 것과, 매 프레임 독립적으로 순열이 달라지는 것은 다르다. 프레임마다 새 노드 집합인 MTAH에는 원본 temporal reshape를 유지할 수 없다. ball graph-pooling 분기가 있다고 전체 모델이 identity-free가 되는 것도 아니다.

### 4.3 CRF·Viterbi의 정확한 역할

근거: [dynamic_sparse_crf.py](https://github.com/hyunsungkim-ds/pathcrf/blob/1b7c6f85a6b78c92e4543fde72ca2b86a933f997/models/dynamic_sparse_crf.py), [dynamic_dense_crf.py](https://github.com/hyunsungkim-ds/pathcrf/blob/1b7c6f85a6b78c92e4543fde72ca2b86a933f997/models/dynamic_dense_crf.py), [train.py](https://github.com/hyunsungkim-ds/pathcrf/blob/1b7c6f85a6b78c92e4543fde72ca2b86a933f997/train.py).

상태 `(u,u)`는 선수 통제, `(u,v)`는 송신→수신 경로, `(u,o)`는 outside 이동이다. `_build_allowed_next`는 통제 선수에서 출발하는 kick, 경로 유지, 수신자에게서 시작하는 다음 경로를 허용한다. outside self-loop는 **흡수 상태**다. 원본은 in-play episode 단위로 실행하기 때문에 재시작 전이가 없는 것이며, 이를 전 경기 decoder에 복사하면 첫 중단 뒤 돌아오지 못한다.

`_compute_logZ`는 incoming predecessor 목록에서 log-sum-exp, `decode`는 max와 backpointer를 사용한다. 학습은 정답 경로 NLL과 CE 보조 손실이다. `train.run_epoch`에는 ball 분기의 MSE도 있다. 정답 없는 이번 상황에서 CRF NLL을 그대로 사용할 근거가 없다.

논문은 동적 transition을 MLP로 설명하지만, 읽은 sparse/dense 구현은 **`nn.Linear(2D,1)`**다. 따라서 score는 이전·현재 embedding의 가산 분해이며 두 embedding의 비선형 상호작용을 직접 만드는 깊은 MLP는 아니다. sparse 버전의 효용에는 허용 전이 그래프 자체도 포함된다. dense 버전은 모든 state 전이를 허용한다. 논문 개념과 고정 커밋의 실제 표현력을 구분해야 한다.

추론 [inference.py](https://github.com/hyunsungkim-ds/pathcrf/blob/1b7c6f85a6b78c92e4543fde72ca2b86a933f997/inference.py)의 `_window_weights`는 overlap ramp로 윈도우 예측을 합친다. `_viterbi_constrained_decode`는 학습 CRF 없이도 emission과 허용 전이만으로 DP를 한다. impossible path면 독립 argmax fallback이 있어 무조건 제약을 만족한다고 단정할 수 없다. `inference_episode`는 NaN이 있으면 episode를 반환하지 않는 분기, keeper 기반 팀 정렬, 다운샘플링·업샘플링을 포함한다. MTAH는 결측 프레임까지 보존해야 하므로 어댑터 없이 채택할 수 없다.

채택할 것은 **관계 공간 인코딩, 시간 문맥, 전역 경로 제약이라는 분리 원리**다. 576 player-edge CRF, 이벤트 라벨 생성, 선수별 RNN, outside=D 대응은 채택하지 않는다. PathCRF 원 논문은 Set Attention backbone을 중심으로 제시하며, GAT은 저장소의 대안 구조다. [원 논문 방법 및 학습 절](https://arxiv.org/html/2602.12080v1).

## 5. Ball Radar: Set Transformer와 계층적 Bi-LSTM

### 5.1 Set 인코딩의 정확한 불변성

근거: [model.py](https://github.com/hyunsungkim-ds/ballradar/blob/2486896cfb933ecd304489883241a00e3645ca8c/set_transformer/model.py), [blocks.py](https://github.com/hyunsungkim-ds/ballradar/blob/2486896cfb933ecd304489883241a00e3645ca8c/set_transformer/blocks.py), [attention.py](https://github.com/hyunsungkim-ds/ballradar/blob/2486896cfb933ecd304489883241a00e3645ca8c/set_transformer/attention.py).

`Attention`은 scaled dot-product attention이다. `MAB(x,y)`는 query x 순열에 equivariant, key/value y의 동시 순열에 invariant다. `ISAB`는 학습 inducing point가 선수 집합을 읽고 그 요약을 선수들이 다시 읽는다. `PMA`는 seed query가 집합을 읽어 고정 크기로 요약한다.

`SetTransformer`는 d=128, inducing m=16, heads=4, seeds k=4를 고정한다. `embed_type='e'`면 `[B,N,Din]→[B,N,Dout]`, `'i'`면 PMA/SAB 뒤 `[B,Dout]`이다. equivariance는 선수별 출력을 재정렬한다는 뜻이지 선수 축을 없앤다는 뜻이 아니다. 이 구현의 attention 인터페이스에는 padding mask가 없으므로 zero-padding만 추가하면 가짜 선수도 attention에 참여한다. 가변 N과 all-empty set 처리부터 새로 구현해야 한다.

### 5.2 PPC→BTR의 실제 데이터 흐름

근거: [PlayerBall](https://github.com/hyunsungkim-ds/ballradar/blob/2486896cfb933ecd304489883241a00e3645ca8c/models/player_ball.py), [dataset.py](https://github.com/hyunsungkim-ds/ballradar/blob/2486896cfb933ecd304489883241a00e3645ca8c/dataset.py).

| 함수·모듈 | 입력→출력 | 감독 정보/전제 |
|---|---|---|
| `SoccerDataset.__init__` | CSV→window input·macro·micro target | phase별 선수 feature가 완전한 열 선택, keeper 기반 좌우 정렬, episode>0, 안정적 player 이름 |
| `PlayerBall.forward` | 보통 `[B,T,26×6]`→`[B,T,28]` | 22선수+4outside; 앞26개는 possession logits, 뒤2개는 볼 xy |
| macro/PPC | 팀별 PPE, 전체 FPE/FPI+선수별 Bi-LSTM→선수 logits | 개인 시퀀스 연속성, 팀별 인코더는 별도 가중치 |
| micro/BTR | 선수 feature+macro hidden+macro logit→팀별 invariant ST→융합→Bi-LSTM→xy | macro의 단서를 이용한 ball regression |
| `TraceHelper.predict_episode` | 단일 episode input→출력 tensor | 전체/overlap-window 추론 및 경계 가중 병합 |

PPC의 macro output은 코드상 **softmax 전 logit**이며 BTR에도 그 값이 전달된다. 논문 설명의 “확률”을 모든 구현 위치에 그대로 대입하면 안 된다. micro 출력은 `[108,72]`로 스케일하므로 MTAH pitch로 바꿔야 한다. preprocessing은 10fps를 가정하고 속도 계산에 0.1초를 쓰며, 이를 25fps에 그대로 쓰면 단위가 틀린다.

PPE와 FPE의 선수별 shared temporal RNN도 ID 연속성이 필요하다. 반면 **팀별 invariant pooling→하나의 temporal RNN** 원리는 우리 팀 수준 학습에 맞는다. 단, 별도 H/A 인코더와 ordered concat은 팀 교환 불변성을 자동 보장하지 않는다.

### 5.3 학습과 마스킹·후처리

근거: [train.py:140](https://github.com/hyunsungkim-ds/ballradar/blob/2486896cfb933ecd304489883241a00e3645ca8c/train.py#L140), [models/utils.py:112](https://github.com/hyunsungkim-ds/ballradar/blob/2486896cfb933ecd304489883241a00e3645ca8c/models/utils.py#L112), [postprocessor.py](https://github.com/hyunsungkim-ds/ballradar/blob/2486896cfb933ecd304489883241a00e3645ca8c/postprocessor.py).

원본 손실은 `λCE·CE(player_poss) + MSE(ball_xy) + λreal·real_loss`다. `calc_real_loss`는 예측 볼의 연속 이동 벡터 간 각도 θ와 최근접 선수거리 d로 `mean(tanh(θ)·d)`를 계산한다. 선수에서 멀리 떨어진 급격한 방향 변화를 벌점화하지만, 단독 사용하면 멈춘 볼·직선 같은 퇴화 해법도 가능하므로 볼 정답 없는 식별 문제를 해결하지 않는다. HTML 논문 수식의 부호와 달리 코드의 MSE·real loss는 양의 벌점이다.

`PlayerBall.forward`의 `random_mask`는 좌표 집합을 숨기는 mask가 아니라, 전달된 macro one-hot 정답·micro 볼 정답의 일부를 입력 hint로 남긴다. **우리의 raw+engineered 동시 마스킹 복원과 다른 문제다.** 추가로 읽은 `train.py`의 masking 조건은 `model_type == 'player_ball'`인데 이 클래스는 일반 ball 설정에서 `model_type='macro_regressor'`를 설정한다. 이 분기는 해당 조합에서 자동으로 실행된다고 볼 수 없다. 이번에 실제 학습 실행으로 검증하지 않았다.

`Postprocessor.calc_poss_scores`의 실제 점수는 `poss_probs / (sqrt(ball_distance)+eps)`다. 논문 설명의 거리 나눗셈과 구현이 다르다. Savitzky–Golay smoothing, touch/carry threshold, 추정 carrier 위치에 볼을 붙인 후 보간하는 후처리를 사용한다. 선수별 좌표와 추정 볼이 모두 필요하므로 현재 경합 모듈에는 사용하지 않는다.

전처리 `MetricaHelper.find_gt_player_poss`는 이벤트의 from/to와 OUT 정보를 이용하고, `TraceHelper.find_gt_team_poss`는 다음 점유자를 bfill한다. 즉 패스 비행 구간에 **미래 수신자**를 줄 수 있어 MTAH의 마지막 통제 팀 유지와 동일한 label semantics가 아니다. `calc_single_player_running_features`의 dropna→차분/보간도 단절·ID swap에 민감하다. notebook은 이 전처리 함수를 호출한다. 코드만 읽었으며 경기 출력은 읽지 않았다.

논문의 계층적 공간·시간 결합은 참고하되 원 학습은 다경기·점유/볼 감독에 의존한다. 현재 단일 경기 무라벨 조건에서 해당 성능을 기대할 근거는 없다. [원 논문 PPC/BTR·손실·데이터 절](https://arxiv.org/html/2306.08206v1).

## 6. football-cv-analytics: 구조 특징과 품질 관리

### 6.1 정확히 재사용할 수 있는 수학

근거: [team_tactical_metrics_v2.py](https://github.com/samuehuang/football-cv-analytics/blob/dea6fb82841120aff7150872f4c9fd47c04fd9a1/src/analytics/team_tactical_metrics_v2.py), [inter_team_spatial_metrics.py](https://github.com/samuehuang/football-cv-analytics/blob/dea6fb82841120aff7150872f4c9fd47c04fd9a1/src/analytics/inter_team_spatial_metrics.py).

이 두 파일은 주로 **top-level 실행 스크립트**이며 재사용 가능한 encoder 클래스가 아니다. import만으로 CSV 읽기·디렉터리 생성·출력을 수행하므로 모델 내부에서 import해 실행하지 않는다.

| 지표 | 구현 정의 | MTAH 적용 시 주의 |
|---|---|---|
| centroid | 선수 xy 평균 | 측면 선수 누락만으로 중심이 이동할 수 있음 |
| length/width | x/y 각각 P90−P10 | min/max와 다른 정의. 이상치 완화이지 GK 제거 보증 아님 |
| compactness | centroid까지 유클리드 거리 평균 | 작을수록 응집; density와 같은 수치 아님 |
| shape area | length×width | bounding-box proxy, convex hull 면적 아님 |
| mean nearest opponent | H 각 row min, A 각 column min을 합쳐 평균 | 전체 H×A 평균거리와 다름 |
| centroid distance | 양 팀 중심 거리 | 최소 선수 간 거리와 다름 |
| hull area | `ConvexHull(points).volume` (2D) | 3명 미만/공선이면 invalid, 0으로 채우지 않음 |

두 스크립트 모두 팀당 **정확히 10개 unique track ID** 조건을 적용한다. tactical metrics는 팀별로 검사하므로 양 팀 동시 10v10까지 보장하는 코드도 아니다. 같은 ID의 중복 행은 unique count를 통과하면서 통계에 중복 반영될 수 있다. MTAH의 goalkeeper 포함·결측·중복·가변 N에 그대로 가져올 수 없다.

### 6.2 identity QA와 possession semantics

근거: [identity_qa.py](https://github.com/samuehuang/football-cv-analytics/blob/dea6fb82841120aff7150872f4c9fd47c04fd9a1/src/analytics/identity_qa.py), [identity-aware builder](https://github.com/samuehuang/football-cv-analytics/blob/dea6fb82841120aff7150872f4c9fd47c04fd9a1/src/analytics/build_tactical_identity_aware_dataset.py), [final.py](https://github.com/samuehuang/football-cv-analytics/blob/dea6fb82841120aff7150872f4c9fd47c04fd9a1/src/possession/final.py).

`build_team_segments`는 track별 팀 표기 연속 구간을 만들고, `detect_switch_events`는 전후 지속시간·시간간격·이동속도로 팀 변경 후보를 고른다. `pair_switch_events`는 서로 반대 방향 팀 전환과 시간·근접 조건을 보고 `confidence='HIGH'`를 붙인다. 이는 휴리스틱 이름이며 검증된 확률이나 인물 신원 복구 보증이 아니다. 긴 결측인데 같은 팀인 segment를 연속으로 묶을 수 있는 점도 주의한다.

identity-aware builder는 안정 팀 매핑과 switch 시점에 따라 팀을 다시 할당하며 quality table을 만든다. metrics wrapper는 원본 스크립트의 경로만 치환해 실행한다. 즉 wrapper 자체에 새로운 재식별 알고리즘이 있는 것은 아니다. 우리의 기본 모델에서는 소스 팀 메타데이터를 유지하고 오류 의심을 품질로 기록한다. 실제 경기의 특정 슬롯을 사람이 보고 홈/원정으로 바꾸지 않는다.

`possession/final.py`는 controller와 last touch를 구분하고, `AerialContest`에서는 possession을 `Contested`, 일부 transit에서는 `Unknown`으로 강제한다. 이 출력은 MTAH H/A/D/X와 다르다. **컨트롤이 없다는 사실과 점유 소유권이 없다는 사실을 분리하는 개념만 참고**하고 상태 이름을 직접 매핑하지 않는다. last touch 또한 last controlled possession과 동일하지 않다.

## 7. Tryolabs와 AAU: 거리·스무딩의 장점과 한계

### 7.1 Tryolabs 실제 코드

근거: [Match.update](https://github.com/tryolabs/soccer-video-analytics/blob/b00c7c75a1d52bd5ca183521a35fc76fbd89cc0e/soccer/match.py#L45), [Player.distance_to_ball](https://github.com/tryolabs/soccer-video-analytics/blob/b00c7c75a1d52bd5ca183521a35fc76fbd89cc0e/soccer/player.py#L73), [run.py](https://github.com/tryolabs/soccer-video-analytics/blob/b00c7c75a1d52bd5ca183521a35fc76fbd89cc0e/run.py).

`Match.update(players,ball)`에서 ball과 가까운 발을 기준으로 nearest player를 고른다. 45픽셀 이내 같은 팀 후보가 counter 20에 도달하면 `change_team`한다. FPS에 따라 실시간 지속시간이 달라지고 원근·줌에 따라 픽셀의 물리적 거리가 달라진다.

“연속 20프레임”이라는 주석과 달리, ball missing 또는 거리 초과로 early return할 때 counter가 초기화되지 않는다. 같은 팀 후보가 재등장하면 counter가 이어질 수 있다. 빈 players 목록에 `min`을 호출하는 경우도 고려해야 한다. 초기 home possession, update 초반 시간 누적, run의 수동 초기 팀 설정도 현재 자동 판별에 가져오지 않는다.

향후 참고할 것은 최소 지속시간·히스테리시스이며, 초 단위 지속시간과 결측 시 reset/hold 계약을 새로 정의한다. YOLO/Norfair/HSV 팀 분류·카메라 보정·볼 detector는 현재 ATD 입력에 불필요하다. 순간 근접은 공중 패스·우연한 통과·오검출을 실제 통제와 구별하지 못하며 smoothing은 짧은 진짜 점유를 지울 수 있다.

### 7.2 AAU 계획과 구현의 차이

근거: [implementation_plan.md](https://github.com/ML-interns-aau/footballtracking/blob/ff14035dc9e359b9170c92463816372cbd73da87/docs/implementation_plan.md), [data_exporter.py](https://github.com/ML-interns-aau/footballtracking/blob/ff14035dc9e359b9170c92463816372cbd73da87/src/exporters/data_exporter.py), [events.py](https://github.com/ML-interns-aau/footballtracking/blob/ff14035dc9e359b9170c92463816372cbd73da87/src/analytics/events.py), [pipeline_runner.py](https://github.com/ML-interns-aau/footballtracking/blob/ff14035dc9e359b9170c92463816372cbd73da87/src/pipeline_runner.py).

계획 문서의 1.5m/5frames와 `DataExporter.update_passes`는 구현 완료 근거가 아니다. 읽은 exporter에 그 함수는 없다. 실제 `EventsDetector._detect_pass`는 radius 3.5m와 2 contact frame 등으로 `LOOSE_BALL/CONTROLLED/PASS_IN_PROGRESS`를 관리하고, 패스 중 다른 선수 touch로 수신/가로채기를 처리한다. 매 순간 nearest 후보와 persistent player ID 및 볼 속도가 필요하다. “최소 프레임”이 모든 수신 확정 분기에 일관되게 적용된 것도 아니다.

또 다른 경로인 `pipeline_runner`의 possession flag는 각 선수의 볼거리 `<2m` 기본값으로 독립 계산된다. `PossessionSummaryCSVBuilder.add_frame`는 팀별 any flag를 집계하므로 양 팀이 동시에 possession=True일 수 있으며 합계 100%를 보장하지 않는다. 이벤트 state machine과 이 percentage는 같은 판정기가 아니다.

exporter는 볼 결측 시 `(0,0)` 기본값을 저장하는 부분도 있다. 우리는 자연 결측과 실제 원점 관측을 분리해야 한다. 저장소에 Gemini/Groq 기반 `ai/` 경로도 있지만 **어떤 모델·분석 데이터 연결도 실행하지 않았다**. 해당 클라우드 보고 기능은 도입 대상에서 제외한다.

## 8. 관련 논문을 적용할 때의 한계

[Borghesi et al., Using Temporal Convolutional Networks to estimate ball possession in soccer games](https://www.sciencedirect.com/science/article/pii/S0957417423002816)의 확인 가능한 초록은 객체별 TCN 임베딩, sum/self-attention/2D-convolution 집계, dead/live 후 H/A의 두 단계 분류를 설명한다. professional tracking의 **지도학습** 결과이며 데이터는 confidential이다. 전체 원문은 접근 차단되어 커널·dilation·attention 수식·ID 단절 처리법을 검증하지 못했다. 특히 일반적인 2D convolution 자체가 집합 순서 불변이라는 결론은 내릴 수 없다.

이를 근거로 “원 논문 재현”이라 부르지 않고 **우리 pooled feature 뒤의 소형 temporal TCN 대조군**을 설계한다. 이는 객체별 TCN을 먼저 적용하는 논문 요약 구조와도 다르다. ID를 연결하지 않아도 되는 temporal aggregator 비교가 목적이다.

선수 궤적과 전체 구조를 함께 쓰는 원리는 PathCRF의 local/global embedding과 Ball Radar의 PPE/FPE/FPI에서 코드로 확인했다. 하지만 두 사례가 **ID가 불완전한 무라벨 단일 경기에서도 성공한다는 증거는 아니다.** 현재 안정적인 개인 궤적을 만들지 못하므로 개별 가속도·개별 approaching velocity 대신 팀 중심·거리 분포·응집도 변화부터 사용한다. 자동 short-tracklet 연결이나 optimal transport는 별도 연구 대안이며 정체성의 정답으로 취급하지 않는다.

## 9. 현재 features/CLI 감사

근거는 프로젝트 안의 [features.py](../MTAH_Relational_Starter/mtah_relational/features.py)와 [cli.py](../MTAH_Relational_Starter/mtah_relational/cli.py)다.

| 현재 동작 | 올바른 점 | 보강할 점 |
|---|---|---|
| `geometry_features` | finite xy, 순서 불변, inclusive 3/5/10m pair count | 미터 입력을 가정할 뿐 단위/bounds/duplicate 검사 없음 |
| `_team_stats` | x length, y width, centroid/radius/bbox 명확 | min/max가 GK·이상치·누락에 민감; 1명 radius=0은 응집 증거 아님 |
| `cross_mean_m` | 전체 H×A 거리 평균 | 최근접 상대거리 평균으로 이름·해석 혼동 금지; 새 nearest 통계 필요 |
| bipartite BFS | H-A edge만 연결, 최대 component, 고립 노드 제외 | 연결 사슬은 큰 면적에 퍼질 수 있음. component 크기≠실제 경합 강도 |
| 빈 팀 처리 | 거리 NaN, pair/cluster=0 | 관측 불가인 0을 확실한 비경합으로 학습하지 않도록 target-valid 필요 |
| `n_home/n_away/low_coverage/overcount` | 관측 상태 표시 | >22만으로 duplicate 검출 불가; 누락과 중복이 상쇄될 수 있음 |
| `add_temporal_features` | ±25 row 차분에 같은 period와 frame 차25 확인 | 두 단방향 1초 변화량이지 선수 가속도 아님; `period=0`끼리도 계산됨; strict unique/order 검사 없음 |
| `_parse_player_groups` | slot count·finite xy·pitch 환산, ball 값 미사용 | xy bounds, malformed field policy, speed schema/신뢰도, provenance 필요 |
| `_meta_team_codes` | XML 첫 두 Team 순서 사용 | 서로 다른 유효 Team ID인지 검증, 공식 팀 mapping 교차 확인 필요 |
| `run_extract` | `frame_id−1`, 공식 blocks, CSV/NPZ 함께 생성 | 행 수만으로 strict 연속성 증명 불가; check가 자동 선행되지 않음 |
| 공식 `check_alignment_raw` | 빠른 count/끝점 검사 | `last−first+1==count`만으로 내부 누락+중복 조합을 검출할 수 없음 |

현재 정적 지표는 **25개**, temporal delta 14개 추가 시 **39개**다(`frame/period` 제외). robust quantile, convex hull, nearest opponent 요약, GNN, Bi-LSTM, 자기지도 loss, clustering, H/A/D decoder는 구현되지 않았다.

노드 NPZ slot 축은 개인 ID가 아니다. 결측 전후 같은 slot의 위치를 차분해 개인 속도/가속도를 학습시키면 잘못된 이동 벡터를 만들 수 있다. 팀 중심의 차분 또한 **선수 교체·관측 구성 변화**를 함께 반영하므로 quality 조건을 붙인다.

## 10. 재사용·새 구현·제외 결정

| 구성 | 참고/재사용 범위 | 새 구현 필요 | 이번 제외 |
|---|---|---|---|
| 공식 데이터 계약 | games의 pitch/half, frame 규칙 | strict record validator·데이터 출처 계약 | SmartTag 기반 sync/X mask |
| 기존 geometry | 검증된 거리·BFS 수학 및 현재 함수 | bounds/quality·nearest·quantile·의존성 mask | 수동 GK/팀 수정 |
| PathCRF GAT | edge-aware attention·LayerNorm·pooling 아이디어 | variable-N, 팀 공유 가중치, pool 후 temporal | node-wise RNN·outside/576 edge target |
| Ball Radar ST | ISAB/PMA와 invariant/equivariant 구분 | mask-aware 가변 N, small-width 대안 | player-possession CE·ball MSE·정답 hint |
| temporal | 양방향 문맥·overlap 병합 원리 | 두 stream Bi-LSTM·gated fusion | 원본 고정 선수축 reshape |
| SSL | 숨긴 관측을 문맥으로 복원하는 목표 | joint raw corruption·dependency mask·loss-valid | 자연 결측을 정답으로 보간한 label |
| pattern discovery | GMM 등 일반 방법 | train-only fit·seed/coverage 안정성 | cluster에 수작업 contest/H/A 정답 부여 |
| CRF/Viterbi | 경로 일관성·DP | 팀 owner·live/dead 분리 decoder, 추후 | 라벨 없는 supervised CRF NLL |
| Tryolabs/AAU | 지속시간·상태 분리의 개념 | 시간 단위·결측 정책 명시 | 픽셀 threshold·클라우드 AI·영상 detector |

세부 shape·loss·누출 검사·파라미터 계산과 단계별 수용 기준은 [MODEL_IMPLEMENTATION_PLAN.md](MODEL_IMPLEMENTATION_PLAN.md)에 명세한다. 참고 개념을 분석했을 뿐 **어떤 외부 코드를 포팅하거나 모델을 새로 구현한 것은 아니다.**

## 11. 이번 실제 검증 결과

사용한 interpreter: `/opt/anaconda3/envs/mtah2027/bin/python`.

```bash
PYTHONDONTWRITEBYTECODE=1 PYTHONPATH="$PWD/MTAH_Relational_Starter:$PWD" \
  /opt/anaconda3/envs/mtah2027/bin/python -m pytest -q -p no:cacheprovider \
  MTAH_Relational_Starter/tests
```

이 명령은 **`No module named pytest`로 실행 실패**했다. 의존성을 설치하지 않았다. PyTorch와 `torch_geometric`도 이 환경에서 찾을 수 없었다.

대신 표준 라이브러리 `importlib`, `unittest.mock.patch.object`, 임시 디렉터리로 기존 테스트 함수를 직접 호출했다. pytest의 `tmp_path/monkeypatch`가 필요한 합성 integration 함수에는 동등한 최소 어댑터를 제공했다. **6개 기존 검증 함수가 assertion 실패 없이 완료**했다. 이는 pytest suite 통과 보고나 모델 실행 결과가 아니다.

검증 대상은 합성 pair 수, 순열, NaN, bipartite component, period 경계 차분, 합성 ATD→CSV/NPZ다. 추가 합성 확인에서 정적25/변화량포함39 fields, period0 내부 delta가 계산되는 현 동작, 동일 관측 중복에 따른 pair count 증가를 확인했다. 외부 모델 파라미터 수와 제안 모델 규모는 레이어 수식을 산술 계산했으며 다음 문서에 조건과 함께 기록했다.

실제 ATD 결측·중복률, GPU/MPS 호환성, 학습 수렴, 경합 정확도, 공식 점수는 **이번에 측정하지 않았다**. 공개 GT로 비교·튜닝하지 않았다. 원본/파생 경기 데이터를 출력·전송하거나 서버를 실행하지 않았다.
