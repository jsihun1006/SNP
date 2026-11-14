# 개발 로드맵: VS Code / Codex용 작업 목록

## 완료(이번 스타터)

- [x] MTAH 공식 자료 스키마 정리: 199,319 frames, 25fps, 46 slots, team mapping, 0-based frame alignment.
- [x] 선수 좌표 기반 상대 근접 그래프 통계, 팀 공간 특징 함수 및 결측 마스크.
- [x] ±1초 차이 features (전후반 경계 mask).
- [x] ATD raw parser CLI, local-only CSV/NPZ, 진단 플롯 CLI.
- [x] 인공 데이터 기반 기본 단위 테스트.

## Phase 1 — 데이터 품질 및 특징 보강 (우선순위 최상)

- [ ] `python -m mtah_relational.cli check`를 실제 local ATD에서 실행.
- [ ] 첫 5,000 frames/half, 전체 199,319 frames의 관측률 및 중복률·비정상 좌표 분포 통계. **수동 라벨 판정 없음**.
- [ ] 빠른 속도 변화·ID switch를 고려한 노이즈/불연속 처리. 원본 보존, 자동 조정만.
- [ ] `speed` 필드 사용한 구간/팀 속도 요약 및 frame-level centroid vector.
- [ ] 팀 내부 kNN·near-neighbor graph 통계, robust width(10–90% 분위수) 추가.
- [ ] `+/-3sec` time-window 만들어 변화량 추출. QC/half 경계를 무조건 존중.
- [ ] `features.csv`와 `raw_nodes.npz`의 프레임 ID/관측 mask 완전 일치 테스트.
- **완료 기준**: synthetic test와 local 경기 모두 pass, 지정한 모든 출력 수치 단위와 결측 처리가 문서화됨.

## Phase 2 — 단순 비학습 baseline

- [ ] 거리·군집·접근 변화와 팀 구조 점수로 contest *candidate* score 생성.
- [ ] 3m/5m/10m 관련 임계값을 사용자/논문 근거의 전역 parameter로 두고, 해당 경기 특정 장면에 과적합하는 수동 tuning 금지.
- [ ] 분포/시간 안정성/경계 민감도 측정. 실제 duel/contest F1이라고 부르지 않는다.
- **완료 기준**: 전체 프레임 candidate_score reproducible, 비정상 경계 누출 없음.

## Phase 3 — Self-supervised model

- [ ] Graph encoder: raw nodes + team attribute + within/between relations, permutation-invariant pooling.
- [ ] Raw graph stream Bi-LSTM A / engineered-feature Bi-LSTM B.
- [ ] 두 branch 같은 시각 범위를 묶어 span masking (leakage 방지).
- [ ] Dual-stream attention/gate fusion.
- [ ] train/validation block split, overlap embargo, half split.
- [ ] `torch.manual_seed`, NumPy seed, deterministic config, `model.pt` 저장.
- [ ] Mac `mps`를 우선 사용하되 지원하지 않는 연산은 CPU fallback. 실험 config/log 저장.
- **완료 기준**: sample batch forward/backward, masked loss finite, permutation stability, holdout reconstruction, 단순 모델과 비교 보고.

## Phase 4 — Unsupervised segmentation

- [ ] Latent embeddings에 GMM/HDBSCAN/변화점 탐지 도입.
- [ ] 유사한 패턴의 지속시간·이전/이후 구조 변화·군집 안정성 분석.
- [ ] 후보 점수와 분석 결과를 동영상 overlay/프레임별 plot으로 점검(수동 레이블 추가 금지).
- **완료 기준**: 패턴 탐지의 결과가 일관되게 재현 가능; 근거 없는 `contest probability` 표기 금지.

## Phase 5 — H/A/D 완성 (대회 필수)

- [ ] 별도 ball-live/dead detector. 결측 ball != dead.
- [ ] H/A possession evidence + last controlled team 유지 규칙 + state decoder.
- [ ] 안전하게 `H/A/D/X` 출력. `D`를 duel로 해석하지 말 것.
- [ ] 모든 프레임 정확히 199,319 rows, `frame,label`로 저장.
- [ ] 공식 `python evaluation/validate_submission.py <file>.csv` 통과.
- [ ] **공개 5분 GT는 alignment sanity에만 사용. 학습·튜닝 불가.**
- **완료 기준**: 클린 설치 후 제공 경기 데이터를 넣으면 동일한 CSV가 생성됨.

## 실험 비교 (반드시 같은 데이터 분할)

| Model | 설명 | 목적 |
|---|---|---|
| Rule | 규칙 기반 근접/구조 변화 | 빠르고 해석 가능한 대조군 |
| A | Feature Encoder + Bi-LSTM | 설계 특징의 유용성 |
| B | Graph Encoder + Bi-LSTM | 배운 관계 표현 유용성 |
| C | Dual-Stream + Fusion | 가공/원시 입력 보완성 |

※ 어느 모델이 나을지 아직 모른다. 라벨이 없으므로 `contest accuracy`는 계산할 수 없다.

## 공개 저장소 체크

- [ ] `git status --short`에서 경기 원본/파생 CSV·NPZ·PNG 비공개 데이터 포함 여부 확인.
- [ ] 경기 데이터/추출물이 타사 클라우드나 LLM으로 업로드되지 않도록 유지.
- [ ] 고정된 버전 requirements + 재현 명령 + 실제 시험 로그 + 기술 문서.
- [ ] 개인/조직 소유 저장소 권한과 브랜치 `KKJ` 확인 후 커밋/푸시.
