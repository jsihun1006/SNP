# stage_b_2: 복원 경로와 증거

- 현재 원본 보존 commit: `d19600b462fdae2530deb65bf3c9ab54c733a723`. 개발 당시 Git commit/tag는 없음.
- 실행 코드: [원래 run_local_temporal_b2.py](../../../competition_judgement/src/competition_judgement/run_local_temporal_b2.py).
- 설정: [원래 local_temporal_b2.json](../../../competition_judgement/configs/local_temporal_b2.json).
- 테스트: [원래 test_local_temporal_b2.py](../../../competition_judgement/tests/test_local_temporal_b2.py): 현재 37 case. 당시 누적 477 passed 기록과 범위가 다름.
- 산출물: `data/derived/competition_judgement/local_temporal_b2_final_20261110/`, 공개 업로드 없음. [경로·SHA256](../../results/artifact_manifest.json).
- 상태: 모듈/설정/테스트 확보. 당시 source hash 기록 중 6/6 일치; [상세](../../results/historical_source_checks.json).
- 변경: 같은 Anchor·Short/Near Rise→Fall.

이는 현재 보존 tree의 실행 경로 안내다. 동일 파일을 과거 버전인 것처럼 복제하지 않았다. 전체 과거 tree의 byte 복원과 현재 확보한 단계별 모듈을 구분한다. [전체 개발 이력](../../DEVELOPMENT_HISTORY.md) 및 [검증](../../VALIDATION.md)을 함께 읽는다.
