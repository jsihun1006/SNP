# 업로드한 검증 자료

실제 경기의 frame/cluster table·시계열·reference keys·동영상·raw cache·모델은 포함하지 않는다. 이 디렉터리는 소스/산출물 해시와 경량 집계만 담는다.

| 파일 | 의미 |
|---|---|
| source_snapshot.json | 원본에서 복사/확인한 185개 소스·설정·문서·테스트의 byte SHA256 및 크기. 기존 tracked root test도 포함 |
| historical_source_checks.json | 단계별 당시 source hash 기록과 현재 소스 비교. 초기/최종/불일치 구분 |
| historical_test_evidence.json | 187/213/233/377/404/440/477/522 누적 테스트 기록과 원래 JSON 경로/hash |
| git_inventory.json | fetch 직후 Git branches/log/reflog/stash/remote 및 tracked/untracked/ignored 개수 |
| artifact_manifest.json | 기존 결과·원본 ATD·필요 입력/캐시 208개, 총 5,105,717,477 byte의 상대경로·크기·SHA256. 실제 데이터는 없음 |
| artifact_validation.json | 기존 199,319/789,005 축·key·후보 gate/basis·Validation 분포·v1 수치·jitter 수치 재확인 |
| default_251_tests.txt | 기본 경합 247 + 새 검사 4개 통과 stdout |
| original_worktree_preservation.json | 원본 KKJ HEAD/status byte 동일 및 185개 원본 파일 해시 보존 |
| core_403_tests.txt | 기존 관계 계산+경합 검사 403개 통과의 원래 stdout |
| legacy_419_tests.txt | 기존 관계/경합/영상 415 + 새 adapter/launcher 4개 통과 stdout |
| repository_test_failures.json | root 방송 관련 테스트를 포함한 광역 실행: 604 passed / 14 failed, 실패 node와 원인 |

경로는 원본 repository root 기준이다. hash는 파일 내용의 존재/무결성 증거이며 실제 데이터 제공이나 과거 소스 byte 복원의 대체물이 아니다. 큰 파일을 재생성하면 gzip metadata/serialization 차이 때문에 hash 대신 key/schema/수치도 비교해야 한다.

원래 최종 산출물은 `data/derived/competition_judgement/candidate_v23_final_20261110/`다. 원본 접근 권한을 가진 로컬 환경에서 REPRODUCTION.md대로 새 output에 재현한다. 기존 산출물을 덮어쓰지 않는다.
