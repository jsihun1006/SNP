# 04 — 단계별 Codex 구현 로드맵

## Stage 0. 실제 로컬 저장소 감사
- `git status --short`, 현재 소스/결과 위치, `RULES.md`, `games.py`, 실제 기존 함수 시그니처와 보호파일 SHA/mtime을 먼저 확인.
- `references/` 문서는 스냅샷; 현행 구현 사실과 다르면 코드 우선 + 차이 문서화.
- 현재 baseline 199,319행 validate 되는지 확인(기존 값 유지).
- `competition_judgement/`만 새 코드 폴더로 사용, `data/derived/competition_judgement/`에 로컬 산출물.

## Stage 1. 규칙 기반 후보 검출 v1 — **이번 Codex 실행 범위**
- 원시 파싱 재구현 대신 `mtah_relational`의 phase1 features/candidate index 재사용, 없다면 기존 CLI를 안전하게 실행.
- `src/competition_judgement/` 밑에 quality adapter, candidate evaluator, temporal grouping, CLI, artifact writer 구현.
- 기존 `CausalContestedIndex` strict 기준선을 정확히 재현하고 사전 지정한 Train quantile로 후보 상태 분리.
- `insufficient_observation`과 `not_candidate`를 엄격히 구분.
- 출력 `candidate_frames.csv` 199,319행, `candidate_segments.csv`, 집계 JSON, `policy_resolved.json`(Train에서 구한 실제 임계값·설정 fingerprint).
- 합성 단위 테스트 및 기존 회귀 테스트 통과.
- 가용한 로컬 자료로 **실제 경기 프레임 전체 처리**, 집계만 보고, 원본 좌표는 대화에 복사하지 않는다.
- 완료 보고서 `competition_judgement/docs/STAGE1_EXECUTION_REPORT.md`.

## Stage 2. 관측 품질 보완 및 시간 안정성 (Stage 1 후 별도 승인)
- strict vs coverage-aware 후보 분류의 차이 분석(초기 strict valid70.3% 한계).
- 시간 변화·연결성 변화·후보 구간 분할의 경계 민감도와 노이즈 실험.
- 기준/임계값을 바꾸면 버전 분리 및 새 사전 설정. 실제 영상에 맞춘 상수 튜닝 금지.

## Stage 3. 상황별 점유 판단 연결 (별도 작업)
- `general`, `possible_contest`, `insufficient_observation` 신호를 기존 볼 기반 baseline과 결합할 인터페이스 설계.
- Ball observation validity, last-controlled owner, dead-ball evidence는 **독립적인 근거**로 처리. candidate만으로 H/A/D 결정하지 않음.
- D와 경합 혼동 금지; 볼 missing/오검출 시 소유권 유지·보류 정책. 실제 규정의 H/A/D/X 의미 준수.
- 공식 199,319행 제출 writer는 기존 코드만 사용하고 형식/재현성 검증.

## Stage 4. 선택적 신경망 비교 (지금 수행 안 함)
- 동결 A12/A20/B12/B20을 특징 extractor로 사용할 가치가 **물리적·정답 무관 진단**에서 보일 때만 비교.
- C의 기본 역할은 '무조건 A+B를 학습'이 아니라 candidate signal 및 downstream possession routing을 보강하는 것.
- 비교에서 실패하면 더 단순한 규칙 기반 모델로 유지. 복원 Huber/임베딩 effective rank는 점유 정확도가 아님.

## 제약 / 중단 조건
- 기존 모델을 수정/재학습하려는 경우, 제출 baseline/평가 코드를 바꾸려는 경우, 공개 GT/제공 경기 수작업 라벨로 threshold 튜닝하려는 경우 **중단하고 사용자에게 설명**한다.
- 실제 데이터의 외부 전송·GitHub push 수행 금지.
- 실험 결과를 '실제 경합이 검증됐다'고 주장하지 않는다.
