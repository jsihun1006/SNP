# Codex에 먼저 전달할 외부 코드·논문 학습 요청

너는 내가 VS Code에서 개발 중인 **MTAH 2027 Phase 1 축구 점유 팀 추정 프로젝트의 연구·개발 파트너**다. 지금은 코드를 일괄 생성하지 말고, 내가 검토한 기존 연구들의 **실제 코드 경로와 데이터 의존성**을 먼저 분석해 우리 모델 설계에 반영해라.

## 우선 읽기 (순서 중요)

1. `CODEX_START_HERE.md` (기존 목표/규칙)
2. `docs/EXTERNAL_CODE_STUDY_GUIDE.md` (**핵심 학습 자료**)
3. `docs/REFERENCE_TO_MODEL_BLUEPRINT.md` (**코드 전환 설계**)
4. `docs/REFERENCE_REVIEW.md` (기존 요약)
5. `docs/PROJECT_SPEC.md`, `docs/IMPLEMENTATION_PLAN.md`
6. 공식 `DATA.md`, `CHALLENGE.md`, `RULES.md`, `SUBMISSION.md`, `starter/load_atd.py`, `starter/baseline.py`, `games.py`
7. 현재 우리 코드 `mtah_relational/features.py`, `mtah_relational/cli.py`, `tests/` 전체

## 외부 코드 학습 대상

자료의 링크·검토 커밋·세부 함수는 `docs/EXTERNAL_CODE_STUDY_GUIDE.md`에 정리되어 있다.

- `hyunsungkim-ds/pathcrf`: dynamic graph/GAT, SetLSTM, CRF, inference
- `hyunsungkim-ds/ballradar`: SetTransformer, hierarchical Bi-LSTM, supervised macro/micro 학습
- Borghesi et al. (2023): TCN + set aggregation + two-stage dead/live & home/away 논문
- `samuehuang/football-cv-analytics`: 팀 geometry, H-A distance, identity QA, ball/possession 분리
- `tryolabs/soccer-video-analytics`: 향후 broadcast 데모; 현재 사용 안 함
- `ML-interns-aau/footballtracking` 구현 계획: 현재 사용 안 함
- `roboflow/sports` PR #36: 미병합 제안, 현재 사용 안 함

실제 원본 소스 접근이 가능하면 위 저장소의 고정 커밋을 확인하고 링크된 주요 파일들을 읽어라. 사용자의 로컬 터미널에서 `bash scripts/fetch_reference_sources.sh`를 실행하면 **현재 프로젝트의 형제 폴더**에 공개 참조 저장소를 복제할 수 있다. 인터넷 접근이 없다면 문서만 사용하고, 원본을 실제로 읽은 것처럼 주장하지 마라.

## 이번 분석에서 반드시 답할 질문

1. Ball Radar와 PathCRF의 **원본 입력·출력·supervised target**이 각각 무엇인가? 무엇을 그냥 쓰면 규정 또는 데이터 구조에 어긋나는가?
2. MTAH는 46개의 track slot이 있지만 선수 고유 ID는 없다. graph와 set pooling을 사용해 **node permutation invariance**를 유지하려면 어떻게 설계해야 하는가?
3. Ball Radar `SetTransformer`의 invariant/equivariant 차이를 설명하고, GAT pooling과의 트레이드오프를 비교해라.
4. 기존 `features.py`에서 거리·compactness·상대팀 관계를 정확히 추출하는가? 결측·중복으로 인한 거짓 구조 변화가 발생하는가?
5. Raw graph stream + engineered stream을 동시에 학습할 때 **masked-window leakage**를 어떻게 방지하는가?
6. 단일 경기 무라벨 데이터에서 **masked-span self-supervised loss**를 정확히 정의해라. 그룹/선수 ID 불연속을 고려한 복원 대상을 제안해라.
7. 군집/경합 score가 `P(contest)`라는 보정 확률이라고 주장할 수 있는가? 없다면 어떻게 표현하는가?
8. 실제 경합 vs 압박, 점유팀 H/A, Dead Ball D를 구분해라. 마지막 controlled possessor 규칙을 후속 점유 decoder에 어떻게 반영할 것인가?
9. Borghesi TCN을 소형 baseline으로 비교한다면 어떤 ablation을 할 것인가?
10. 공개 5분 GT를 학습/튜닝에 사용하지 않으면서 검증 가능한 테스트는 무엇인가?

## 다음 산출물

`docs/CODEX_RESEARCH_OUTPUT.md`를 새로 작성하되 다음 항목 포함:
- 출처별 코드/논문 핵심 설명 (파일·함수 경로 근거)
- 재사용 vs 새 구현 vs 제외 표
- 우리의 새로운 모듈 인터페이스와 텐서 shape 제안
- 실험 A/B/C (engineered-only, graph-only, fusion)
- masked self-supervision loss와 leakage QA
- 구현 순서, 실제 MTAH 데이터에서 확인할 진단 항목
- 현재 코드를 무단 변경하지 않고 진행할 작업 계획
- 불확실하거나 코드 접근이 안 된 내용 명시

**이번 첫 실행에서 기존 코드 또는 실제 데이터 파일을 수정하지 말고 연구 결과 문서부터 작성해라.** 그다음 작은 Phase 1 구현 단위에 대해 설명하고 테스트 기준을 제시해라.

## 준수 조건

- `Smart Tagging`은 학습과 추론에서 사용하지 않는다.
- 경합 모듈은 선수 좌표만 사용한다(볼 좌표 입력 금지). 최종 possession pipeline은 이후 별도 설계.
- 지도학습된 기존 모델을 **비지도학습이라고 부르지 않는다.**
- 제공 경기의 사람 손으로 만든 라벨/공개 GT 튜닝/수동 tracking 정정 금지.
- 수동 설정한 특정 프레임 답을 hard-code 하지 않는다.
- 경기 파일·복원 가능한 파생 tracking을 공개 GitHub나 클라우드 LLM에 올리지 않는다. 코드는 로컬에서 실행한다.
- 외부 코드 복사 시 라이선스와 저작권 확인. 가능하면 설계를 참고한 독립 구현.
- `frame_id`는 1부터, 제출 `frame`은 0부터 시작하며 `frame=frame_id-1`.
- 실제 실행하지 않은 시험 결과나 정확도 수치를 만들어내지 않는다.
