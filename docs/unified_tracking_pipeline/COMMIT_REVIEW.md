# 최종 반영 트래킹 코드 — 커밋 전 검토

현재 검토 대상은 **`submission/final_tracking/`의 확정 소스49파일과 설명/검증 기록**이다.
기존의132개 연구 보관 목록이나 별도 원본 ATD baseline을 현재 커밋 대상으로 쓰지 않는다.

- [최종 입력에서 역추적한 실제 경로](../submission_pipeline/FINAL_TRACKING_PIPELINE.md)
- [실제 코드 폴더](../../submission/final_tracking/README.md)
- [파일 목록](../../submission/final_tracking/CODE_FILES.txt)
- [입출력/역할/원문 해시](../../submission/final_tracking/CODE_MANIFEST.json)
- [이번 검증](../../submission/final_tracking/VERIFICATION.json)
- [기술문서](../submission_pipeline/SUBMISSION_DRAFT.md)

V3는 최종 manifest/실파일 입력 해시로 사용이 확정됐다. 폐기된 선/H 보정·V4·원본 ATD baseline·
옛 부분 보간 결과는 제외했다. 공통 함수를 제공하는 모듈은 해당 함수 역할로만 유지하며,
중간 비교 영상이나 다른 모델의 좌표를 실행 단계에 포함하지 않는다.

현재 작업 폴더의 소스·원본/결과 데이터는 변경하지 않았다. 필요한 정렬4파일과 최초변환기
원문은 별도 코드 폴더에만 복사했다. 실제 데이터/영상/모델/특징/검수 자산을 복사하지 않았다.
현재 ATD2개는 LFS 포인터 상태라 재실행 전에 실데이터를 준비해야 한다.

`commit_candidates.txt`는 이번 소스 모음·문서만 가리키도록 갱신했다. 코드 안의 기존 고정
객체 판단과 라벨 출처 때문에 이 로컬 검토 목록을 곧바로 공개 승인으로 해석하지 않는다.
Git add/commit/push는 실행하지 않았으며, 다른 작업/노트북 변경은 대상에 넣지 않았다.
이전 검토 문서는 `docs/submission_pipeline/archive_before_confirmed_tracking_scope/`에 보존했다.
