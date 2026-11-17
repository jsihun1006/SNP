# 최종 반영 트래킹 코드 — 확정본

현재 기준은 **최종 통합 데이터의 실제 입력 해시**입니다. 이전의 포괄적인 연구 파일 목록과
별도 원본 ATD baseline을 제출 대상 목록으로 쓰지 않습니다.

- [처음부터 트래킹 마무리까지 확정 경로](FINAL_TRACKING_PIPELINE.md)
- [실제 코드49파일과 사용 안내](../../submission/final_tracking/README.md)
- [코드 파일 목록](../../submission/final_tracking/CODE_FILES.txt)
- [원문·함수 역할·의존성·입출력 해시](../../submission/final_tracking/CODE_MANIFEST.json)
- [이번 읽기 검증 결과](../../submission/final_tracking/VERIFICATION.json)
- [제출 기술문서](SUBMISSION_DRAFT.md)
- [재실행 전 입력 조건](GAPS.md)

확정 결과는 `outputs/unified_tracking_v1/`의 **V3 공 + Tactical 선수46슬롯**입니다.
V3 입력은 최종 manifest와 현재 파일 해시가 일치합니다. 선/H 보정, V4, 별도 ATD baseline,
옛 부분 보간 결과는 제외했습니다. 필요한 공통 함수만 들어 있는 모듈은 실행 단계와 구분했습니다.

새 알고리즘·보정·추론을 수행하지 않았습니다. 사용이 확인된 최초 변환기 원문과 현재
체크아웃에 없는 정렬4파일은 별도 코드 폴더에만 복사했습니다. 기존 작업 소스와 최종 데이터는
변경하지 않았습니다. 현재 원본 ATD2개는 Git LFS 포인터이며 재실행 전에 실데이터 준비가 필요합니다.

기존 원본 ATD baseline 소스와 `outputs/submission_public_code_v1/`는 별도 과거 준비 작업입니다.
이번 확정 경로·코드 모음에 포함하지 않았습니다. 이전 문서는 `archive_before_confirmed_tracking_scope/`에
보존했습니다. 원본 `SUBMISSION.md`와 Git staging/commit/push 상태는 변경하지 않았습니다.
