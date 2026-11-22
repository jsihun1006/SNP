# v1: 복원 경로와 증거

- 현재 원본 보존 commit: `d19600b462fdae2530deb65bf3c9ab54c733a723`. 개발 당시 Git commit/tag는 없음.
- 실행 코드: [원래 cli.py](../../../competition_judgement/src/competition_judgement/cli.py).
- 설정: [원래 initial_policy.json](../../../competition_judgement/configs/initial_policy.json).
- 테스트: [원래 test_stage1.py](../../../competition_judgement/tests/test_stage1.py): 현재 31 case. 당시 누적 187 passed 기록과 범위가 다름.
- 산출물: `data/derived/competition_judgement/stage1_v1_final_20261109/`, 공개 업로드 없음. [경로·SHA256](../../results/artifact_manifest.json).
- 상태: 최종 수정본 모듈 확보; 초기 detector byte 미복원. 당시 source hash 기록 중 6/6 일치; [상세](../../results/historical_source_checks.json).
- 변경: 후보 점수·Train threshold·연속 5 frame.

이는 현재 보존 tree의 실행 경로 안내다. 동일 파일을 과거 버전인 것처럼 복제하지 않았다. 전체 과거 tree의 byte 복원과 현재 확보한 단계별 모듈을 구분한다. [전체 개발 이력](../../DEVELOPMENT_HISTORY.md) 및 [검증](../../VALIDATION.md)을 함께 읽는다.
