# 05 — 기존 프로젝트의 재사용 자산 및 폐기하지 않을 결과

## 기존 코드 (로컬 저장소에서 우선 확인)
| 위치 | 기능 | 새 프로젝트 활용 |
|---|---|---|
| `games.py` | 공식 frame/pitch/half blocks | **읽기 전용** |
| `MTAH_Relational_Starter/mtah_relational/data_quality.py` | 원시 플레이어 reader 및 strict QA | 재사용 |
| `.../features.py` | H/A pair·거리·최대 연결 component·폭/길이 | 재사용 |
| `.../spatial_schema.py` | 대칭24·ordered `[2,10]` 팀 구조 | 정의 재사용 |
| `.../candidate_baseline.py` | CausalContestedIndex(현재 접촉+과거1초) | 변경 없는 기준선 |
| `.../zones.py` | 8×6 zone 분석 | 보조 진단만 |
| `.../preprocessing.py` | frame-local 입력·관측 마스크 | 재사용 |
| `.../phase1.py`, `.../cli.py` | 전체 프레임 집계·CSV 출력 | 데이터 생성 경로 |
| `.../splits.py`, `.../normalization.py`, `.../frame_store.py` | 시간 블록/Train-only 통계/원시 cache | 유효성 확인 후 필요 부분 활용 |
| `.../model_a.py`, `.../model_b.py` | MLP/Bi-LSTM 및 GAT/Bi-LSTM | **동결**, 이번 Phase 입력 아님 |
| `starter/baseline.py`, `evaluation/validate_submission.py` | 기존 제출 기준선·공식 형식 검증 | 보존; 필요 시 읽기 전용 실행 |

## 수치로 확인한 이전 교훈
- 원본199,319프레임, 공식 평가158,857프레임, strict candidate index valid111,733프레임(70.3%). 나머지29.7%를 비경합0으로 채우면 큰 편향 가능.
- A 8–13 Mask Validation MAE 0.273364 (A12), 0.251037 (A20), Interpolation 0.151281.
- B 8–13 Mask Validation MAE 1.212521 (B12), 1.152329 (B20); 인원 구성 변화에 민감. `cross_cluster_fraction` 매우 작은 Train IQR로 손실 집중.
- 위는 **기하 특징 재구성 오차**이며 실제 경합·점유 정확도를 의미하지 않는다.
- 기존 GAT/MLP 실험을 폐기할 필요는 없지만 새로운 작업의 주요 objective가 아니다.

## 역사적 문서 주의
`references/MODEL_IMPLEMENTATION_PLAN.md`는 원래 A/B/C masked reconstruction을 중심으로 계획한 내용이다. 현재 사용자가 변경한 목적은 이 폴더의 `docs/00_OBJECTIVE_AND_ROUTING.md`가 우선한다. B 모델까지 구현·실행된 최신 상태는 `references/PHASE5_MODEL_B.md`를 따른다. 보고서의 상대 경로는 원래 저장소 경로이므로 복사된 references 내부에 실제 소스 파일이 존재한다고 오해하지 않는다.
