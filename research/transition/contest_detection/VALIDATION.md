# 이번 검증 / 2026-11-15

## 테스트 범위

| 검사 | 결과 | 의미 |
|---|---|---|
| 기존 MTAH_Relational_Starter + 경합 (영상 제외) | 403 passed, 23.33s | 기존 모델의 소규모 합성 검사 포함; 실제 경기 학습 아님 |
| `python contest_detection/run.py test --legacy` | 419 passed, 23.84s | 위 403 + 후속 영상 디버거 12 + 새 보존/adapter/launcher 4 |
| root tests도 포함한 광역 suite | 604 passed / 14 failed, 25.63s, 2 subtests passed | 전체 618개; 경합 실패 없음. 아래 원인 기록 |
| 기본 `run.py test` | 251 passed, 2.63s | 경합 247 + 새 검사 4, 영상/Torch 불필요 |
| 당시 v2-3 verification 기록 | 522 passed, 32.92s | 과거 기록. 이번 실행 수로 표기하지 않음 |

403은 관계 package 156 + 경합 본체 247이다. 과거 522는 이 403에 당시 root 119를 더한 suite다. 현재 base의 root test는 199개이고 후속 영상 12 및 새 검사 4가 더 있어 총 618개다. 과거 119개 root case의 정확한 node 목록/그 당시 전체 source tree는 확보하지 못했다. 전체 618개를 실행해도 과거 522개를 동일 환경으로 완전 재현했다고 표현할 수 없다.

광역 실패의 원인은 `norfair` 미설치 1개, 외부 upstream `inference` module 부재 3개, `imageio_ffmpeg` 미설치 9개, 기존 `tryolabs_v4_context.goal_geometry_hint`의 OpenCV HoughLinesP 결과 unpack TypeError 1개다. 경합 범위를 벗어나므로 의존성을 새로 설치하거나 해당 알고리즘/테스트를 변경하지 않았다. 결과를 숨기거나 skip으로 통과 처리하지 않았다. 상세 node는 results/repository_test_failures.json에 있다.

기존 Python 모듈 51개를 모두 import했고, adapter는 Torch/OpenCV 없이 import됨을 확인했다. 합성 strict-valid 양 팀 8명 입력으로 기존 공간 함수가 16개의 radius view를 반환함도 확인했다.

## 원본/산출물 확인

- 원본 소스 185개 byte를 보존했고 staged blob SHA256까지 대조했다. v2-3 5개, B.1 7개, B.2 6개 source hash는 당시 기록과 모두 일치한다. 일부 초기 hash 불일치는 VERSION_MANIFEST.md 참조.
- 기존 입력/산출물 208개(약5.11GB)를 읽기 전용으로 hash했다. 검사 전후 size/mtime도 같고 최종 runner의 frozen input 5개 hash가 기존 기록과 일치했다.
- 저장된 frame 199,319개가 0..199318 순서와 일치하고, cluster 789,005개 key가 고유·정렬됨을 확인했다. Cluster 16열 / Frame 39열을 기록했다.
- 후보 row의 spatial gate/basis가 유효하고, noncandidate basis가 0이며, 충분한 동적 부재 mask가 63임을 확인했다. 이 검사는 저장 결과의 정합성 검사이며 전체 789,005개 결정을 다시 추론한 것은 아니다.
- split_manifest 기준 Validation 23,828 frame / 112,286 cluster에서 PATTERN 8,850, LOCAL 3,503, REVIEW 74,313, NO 1,926, INSUFFICIENT 23,694. 후보 frame 6,139개를 확인했다.
- v1 공식 158,857 frame에서 9,936 / 101,797 / 47,124 및 후보 segment 446개를 확인했다. jitter .1m의 3,191 고정 probe에서 신규213/소실48을 집계했다.

## 의미·재현성 검사

기존 합성 suite는 현재/공식/QC gate, NaN/unknown, 관계/군집 key 정렬, frame axis, half/split/gap, 전체지원/부분지원, 증거 중복/Anchor 혼합 금지, 고정 seed masking/jitter, permutation/team swap, source 보존을 검사한다. 새 adapter 테스트는 원래 함수 객체를 그대로 내보내는지 확인하므로 계산 코드를 복제하지 않는다. launcher는 repo root·공백 포함 인자 전달·운영체제 경로 구분자·UTF-8를 검증한다.

현재 환경은 Python3.11.17 / NumPy2.4.6 / pandas3.1.0 / pytest9.2.0 / torch2.15.0 / macOS arm64 CPU다. Windows/CUDA/RTX5080, Windows wheel 설치, 전체 경기 재추론, 실제 경합 정확도, DFL 학습·Ablation은 실행하지 않았다. 원래 `verify_candidate_v23`의 전체 재계산 결과는 역사 자료이며 이번 검증과 구분한다.

기존 소스의 빈 EOF 줄 때문에 `git diff --check`가 `MTAH_Relational_Starter/tests/test_features.py:41`을 지적했다. 원본 byte 보존을 위해 그대로 두었다. 새 코드/문서는 whitespace 검사를 통과한다.
