# VERSION_MANIFEST

현재 원본 코드 보존 커밋: [`d19600b462fdae2530deb65bf3c9ab54c733a723`](https://github.com/jsihun1006/SNP/commit/d19600b462fdae2530deb65bf3c9ab54c733a723) (2026-11-15). 모든 단계의 Git Commit은 이 **신규 보존 커밋**이며, 개발 당시 커밋/태그는 발견하지 못했다.

아래 상태는 현재 소스와 당시 기록의 연결 가능 범위다. 단계별 완전한 과거 작업 트리 snapshot은 없으므로 버전별 복제본이나 과거 날짜 커밋을 만들지 않았다. src/config/tests는 기존 competition_judgement 경로에 한 번만 보존한다. 과거 테스트 수는 누적 suite 범위이며 현재 해당 파일의 test case 수와 구분한다.

| 단계 | 실제 entry module (`competition_judgement.`) | 설정 (`configs/`) / 테스트 (`tests/`) | 원래 결과 폴더 | 복원 상태 | 당시 테스트 / 해당 파일 현재 | hash 기록 일치 | 주요 변경 / 최종본 |
|---|---|---|---|---|---|---|---|
| [v1](versions/v1/README.md) | `cli` | `initial_policy.json` / `test_stage1.py` | `stage1_v1_final_20261109` | 최종 수정본 모듈 확보; 초기 detector byte 미복원 | 187 / 31 | 6/6 | 후보 점수·Train threshold·연속 5 frame; 선행 단계 |
| [v2_1](versions/v2_1/README.md) | `analyze_local_clusters` | `local_clusters_v2_1.json` / `test_local_clusters.py` | `local_v2_1_20261109` | 모듈/설정/테스트 확보 | 213 / 26 | 21/21 | 3m/5m mixed components·다중 군집; 선행 단계 |
| [v2_2](versions/v2_2/README.md) | `analyze_temporal_relations` | `temporal_relations_v2_2.json` / `test_temporal_relations.py` | `temporal_v2_2_final_20261109` | 최종 모듈 확보; 초기 temporal_features byte 미복원 | 233 / 20 | 16/16 | ±50 Anchor·kernel·기하 대응; 선행 단계 |
| [stage_a](versions/stage_a/README.md) | `run_missingness_stage_a` | `missingness_stage_a.json` / `test_missingness_stage_a.py` | `missingness_stage_a_final_20261109` | 모듈/설정/테스트 확보 | 377 / 25 | 12/12 | 결측 원인·관측 지원·mask experiments; 선행 단계 |
| [stage_b](versions/stage_b/README.md) | `run_evidence_stage_b` | `evidence_stage_b.json` / `test_evidence_stage_b.py` | `evidence_stage_b_final_20261109` | 핵심 계산 확보; 초기 진단 코드 일부 미복원 | 404 / 27 | 10/14 | Full/Partial P1–N3·중복 처리; 선행 단계 |
| [stage_b_revised](versions/stage_b_revised/README.md) | `run_pattern_b1` | `pattern_b1.json` / `test_pattern_b1.py` | `evidence_pattern_b1_final_20261110` | 모듈/설정/테스트 확보 | 440 / 36 | 7/7 | Evidence 지원과 A–D 순서/패턴 분리; 선행 단계 |
| [stage_b_2](versions/stage_b_2/README.md) | `run_local_temporal_b2` | `local_temporal_b2.json` / `test_local_temporal_b2.py` | `local_temporal_b2_final_20261110` | 모듈/설정/테스트 확보 | 477 / 37 | 6/6 | 같은 Anchor·Short/Near Rise→Fall; 선행 단계 |
| [v2_3](versions/v2_3/README.md) | `run_candidate_v23` | `candidate_v23.json` / `test_candidate_v23.py` | `candidate_v23_final_20261110` | 실제 최종 판정 모듈 확보 | 522 / 45 | 5/5 | Spatial Gate·5상태·Frame 출처 보존; **최종 판정** |

모든 코드 경로의 prefix는 `competition_judgement/`다. 결과 경로의 prefix는 `data/derived/competition_judgement/`다. hash 기록은 중복 기록을 포함하며 전체 당시 작업 tree가 완전히 복원됐다는 뜻은 아니다.

실제 초기 v1, 초기 v2-2 및 Stage B 초기 진단 byte는 복원하지 못했다. Stage B의 당시 verification은 진단 모듈 하나가 불일치하고, 초기 summary는 resume/verifier까지 다른 hash를 기록한다. 현재 resume/verifier는 후속 verification과 일치한다. 현재 코드를 초기 원본으로 표시하지 않는다.

기존 영상 디버거 `view_candidate_video.py` / `test_candidate_video.py`는 v2-3 후속 도구이며 후보 판정의 대안 구현이 아니다. `starter/zone_contest_metrics*.py`, 다른 possession 파이프라인은 최종 runner 호출 경로에 없다.

원래 입력/산출물 hash: results/artifact_manifest.json. 과거 테스트 원본 기록 경로/hash: results/historical_test_evidence.json. 현재 소스별 byte hash: results/source_snapshot.json. 현재 테스트의 실제 성공/실패 범위: VALIDATION.md.
