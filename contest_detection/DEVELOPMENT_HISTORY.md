# 개발 이력: 확인된 구현과 복원 한계

조사일 2026-11-15. 날짜는 기존 문서/산출물의 기록이며 Git commit 날짜를 추정한 것이 아니다. 파일명만으로 최종본을 선택하지 않고 호출 관계, 고정 설정, 테스트 기록, 결과 축, 당시 source hash를 대조했다. 각 수치는 한 경기의 반복 관측에 대한 기술 통계다.

## 이력 조사 근거

원본 브랜치 KKJ는 `17f527eb31c4bc2a8f9dc61d0c08e93cd1173808`이었다. origin은 `https://github.com/jsihun1006/SNP.git`, upstream은 Metrica 공식 저장소다. `git fetch origin` 후 origin/transition은 없었다. `git log --all`, `git branch -avv`, `git reflog --all`, `git stash list`, `git ls-files`, untracked/ignored 목록을 조사했다. stash는 비어 있었고, 이 경합 패키지·관계 패키지·초기 zone contest 코드의 Git log는 비어 있었다. 전체 Git 조사 요약은 results/git_inventory.json에 있다.

원본 작업 트리에는 notebook 3개 수정과 다수 미추적 파일이 있었다. 별도 worktree에서 현재 로컬 파일을 명시적으로 복사했으며 원본을 편집하지 않았다. 최초/최종 working status와 보존 소스 해시를 대조한다. 기존 main/KKJ 이력을 바꾸지 않는다.

`starter/zone_contest_metrics.py` 및 `_48.py`는 구역 단위 근접·밀집의 별도 선행 탐색이다. v2-3 호출 그래프에서 사용되지 않으며, 요청된 9,936/446 결과의 v1 진입점은 `competition_judgement.cli`다. 선행 탐색을 그 v1과 동일 구현이라고 단정하지 않는다. 대체 possession 패키지들도 최종 후보 runner의 호출 경로에 없다.

## v1: 최초 후보 / 2026-11-09

`detector.py`, `quality.py`, `cli.py`, `artifacts.py`와 `initial_policy.json`. player-only 기하·양 팀 거리·접촉 수·혼합·밀집 및 과거 25프레임 변화를 사용한다. Train의 고유 strict-valid frame에서 threshold를 한 번 적합한다. 현재까지 연속 5프레임을 통과해야 candidate를 내며, 앞의 4프레임을 소급 수정하지 않는다. half/split/gap/unknown에서 이력을 끊는다. Candidate / Not-candidate / Insufficient를 분리하고 미관측을 음성으로 채우지 않는다.

공식 평가 158,857프레임에서 candidate 9,936 / not-candidate 101,797 / insufficient 47,124, candidate segment 446개다. 전체 199,319축에서는 공식 범위 밖을 포함해 insufficient 87,586이다. 187개 통과 기록을 output_validation.json에서 확인했다. 현재 기존 합성 검사의 v1 파일은 31개다.

초기 `stage1_v1_20261109`의 detector hash는 현재와 다르다. `stage1_v1_final_20261109`의 6개 source hash는 현재와 모두 일치한다. 초기 byte 소스는 찾지 못했으며 최종 수정본을 최초 원형으로 가장하지 않는다. 보고서에는 표현/CSV formatting 수정이 후보 결과를 바꾸지 않았다는 검증이 남아 있다. 전역 집계만으로 국소·다중 관계를 설명하기 어렵다는 한계에서 v2-1로 진행했다.

## v2-1: Local Cluster / 2026-11-09

`local_clusters.py`는 양 팀 간 거리 ≤3m/≤5m의 bipartite graph와 혼합 연결 성분을 만든다. `local_features.py`는 군집별 팀 수, cross-edge density, 평균 반경·직경, contact strength를 계산한다. 같은 팀 연결만으로 군집을 만들지 않는다. 서로 독립적인 다중 군집을 유지하고 같은 frame의 3m→5m family를 보존한다. family/cluster id는 영구 시간 ID가 아니다.

전체 199,319 frame / 789,005 cluster row, 기존 187+26=213개 테스트 기록. 현재 source hash는 verification/output_validation의 21개 기록과 모두 일치한다. 경계 가까운 좌표에서는 작은 jitter가 연결 성분과 후보 기하를 바꿀 수 있다. v1 판정은 그대로 보존했고 새 경합 정답을 생성하지 않았다. 정적인 근접이 지속 마킹인지 일시적 관계인지 구분하기 위해 시간 관계를 추가했다.

## v2-2: Temporal Relations / 2026-11-09

`TemporalSource`, `summarize_anchor_series`, `analyze_anchor_batch`, `reciprocal_unique`가 현재 Anchor의 ±50프레임(현재 포함 101개, 25fps)을 읽는다. soft proximity kernel, approach/maintain/separate 및 재접근·지속시간 하한·기하 matching을 계산한다. record/observed/strict/reliable을 분리하고 half/split/gap을 넘지 않는다. geometric reciprocal-unique 대응은 선수 신원을 증명하지 않는다.

199,319 frame / 789,005 cluster row와 233개 테스트 통과 기록. 최종 source 16개 hash 모두 일치한다. 초기 temporal_features hash는 다른데, 보고서의 known-empty probe/nullable flag 수정 기록과 구분한다. 초기 소스 byte는 미복원이다. 공식 군집 745,891개 중 101개 모두 reliable인 표본은 191,868개다. 결측이 패턴 지원과 지속시간을 크게 제한하므로 Stage A에서 이를 별도 분석했다.

## Stage A: Missingness / 2026-11-09

`missingness_*`, `controlled_temporal_masking.py`, `partial_window_diagnostics.py`. record 부재, 양 팀 관측 누락, strict QC 실패, 팀 인원/국소 구성 변화, 중심 결측·긴 gap을 분리한다. 결측과 실제 지역 이탈을 확정 구분할 수 없으며 unknown을 분리로 처리하지 않는다. 지속시간은 관측된 하한과 검열 사유를 남긴다.

기존 233 + root tests 119 + 신규 25 = 377 통과 기록이다. 따라서 앞 단계 수와의 차이 144개가 모두 신규 Missingness 검사는 아니다. 고정 Validation 1,778 window / 30 support block과 Train donor를 사용한다. 최종 source hash 기록 12개 일치. Full/Partial coverage는 정답의 recall이 아니며 관측 confidence도 보정된 확률이 아니다. 다음 단계에서는 지원 수준을 보존한 개별 Evidence를 도입했다.

## Stage B: 최초 Evidence / 2026-11-09

`evidence_features.compute_evidence`는 Full/Partial namespace에서 P1 접근, P2 집중, P3 관계 변화, P4 이후 분리·확장과 N1 지속 근접, N2 기하 동반 이동 단서, N3 안정 근접을 계산한다. `evidence_frames.py`의 동일 family N1/N3 중복 처리는 단순 합산을 피한다. 서로 다른 Anchor의 Positive/Negative 최댓값을 한 사건의 증명으로 합치지 않는다.

377+27=404개 테스트 기록. 핵심 계산·source·frame aggregation·runner는 당시 hash와 일치하지만 `evidence_diagnostics.py`는 verification/summary와 다르고, resume/verifier는 초기 summary와 다른 최종 verification hash가 일치한다. 따라서 초기 진단 byte 전체의 완전 복원이라고 부를 수 없다. 초기 불완료 폴더 `evidence_stage_b_20261109`와 최종 폴더를 구분한다.

Evidence 크기/평균만으로 사건 순서를 인증할 수 없고, 상관된 Positive·Negative를 종합 점수로 더하는 해석에 한계가 있어 B.1 개편으로 진행했다.

## 개편 Stage B = B.1: Evidence / Pattern 분리 / 2026-11-10

`pattern_values.feature_profiles`는 개별 P1–N3 계산과 각 기능에 실제 필요한 범위의 COMPLETE / EXPERIMENTAL_PARTIAL / UNKNOWN 지원을 보존한다. `pattern_profiles.profiles`는 관측 지원(obs), 순서(seq), outcome을 별도로 둔다. `without_sequence`는 값만 있을 때 SUPPORTED를 만들지 않는다.

A는 접근/집중, B는 일시적 Rise→Peak→Fall 및 구조 변화, C는 안정 근접, D는 기하 이동/형태 안정이다. A/B는 이후 후보 근거, C/D는 Alternative다. D는 선수 신원 기반 동반 이동의 확인이 아니다. Partially observed Evidence는 complete temporal pattern으로 승격하지 않는다.

404+36=440개 테스트 기록, 7개 source hash 일치. 전체 101개가 부족하더라도 필요한 일부 범위의 근거는 완전할 수 있지만, Full Pattern 전체를 확인하지 못한 국소 변화가 남아 B.2를 추가했다.

## B.2: 같은 Anchor의 Local Rise→Fall / 2026-11-10

`local_temporal_features.local_features`는 past/future Short(-10..0/0..10), Near(-25..0/0..25)를 분리한다. scope 내부 median5와 lag5 변화, soft deadband .02, 거리 deadband .1m를 고정한다. 동일 row·동일 scope의 완전한 Rise와 Fall만 Local RF다. 서로 다른 scope/Anchor의 두 변화를 조합하지 않는다. 결측·구성 변화 시 UNKNOWN을 보존한다.

440+37=477개 테스트 기록. 기존 B.1 값의 재계산 보존과 6개 source hash 일치. 새로운 로컬 관측을 Full Pattern으로 승격하지 않았다. 이 단계까지는 최종 후보 통합기가 아니며 v2-3에서 출처를 유지한 규칙으로 결합한다.

## v2-3: 최종 후보 통합 / 2026-11-10

실제 진입점 `run_candidate_v23.main`, 설정 `candidate_v23.json`, 핵심 `decide_clusters`, Frame 집계 `CandidateFrames`. B.1/B.2의 `(frame,radius_m,cluster_id)` key와 순서를 정확히 join한다. 현재 Spatial Gate 실패는 INSUFFICIENT. A/B SUPPORTED는 PATTERN, 완전한 동일 scope RF는 LOCAL, 충분히 관측된 동적 부재는 NO_DETECTED_SIGNAL, 나머지는 REVIEW다. N1–N3/C/D는 veto하지 않는다. 관측된 empty mixed graph는 시간 평가 NOT_APPLICABLE인 별도 Frame 처리다.

전체 199,319 frame / 789,005 cluster view. Validation half1 66811..78730 및 half2 166381..178288의 23,828 frame/112,286 view에서 PATTERN 8,850 / LOCAL 3,503 / REVIEW 74,313 / NO 1,926 / INSUFFICIENT 23,694. 후보 frame은 6,139(25.76%). 이는 test set 결과나 실제 경합 precision이 아니다.

고정 clean 3,191 probe의 jitter .1m에서 신규 213/소실 48. seed 2027, train donor와 고정 Validation support 사용. `.1m` 민감도가 남으므로 강건성이 확보됐다고 주장하지 않는다. offline 미래 의존성 최대 A0/B50/Short10/Near25프레임이다.

477+45=522개 통과 기록. 핵심 5개 source hash 전부 일치하고 frozen input 5개 hash도 현재 일치한다. 당시 전체 재계산 검증은 보존 기록이며, 이번에는 저장 결과 검증과 합성 테스트만 수행했다. 초기 radius join 오류 폴더 `candidate_v23_incomplete_radius_join_20261110`는 미완료이며 최종본에서 제외한다. 영상 디버거는 후속 도구(기존 522+12=534 기록)로 판정 규칙을 바꾸지 않는다.

## 이번 호환성 준비

원본 소스/설정을 수정하지 않고 import 재사용 adapter, 공통 launcher, 의존성 파일, source byte 보존 gitattributes와 문서를 추가했다. 기존 README의 Bash 명령·로컬 전용 과거 지시는 역사 자료로 보존하며 이번 사용법은 이 디렉터리를 따른다. 기존의 no-upload 기록은 실제 경기 산출물에는 계속 적용하며 이번 업로드는 코드·설정·테스트·경량 집계/해시만 포함한다. 모델 라벨/성능·실제 선수 신원·episode stitching·DFL 통합은 앞으로의 별도 개발 과제다.
