> Historical collection record from `final` commit `acf854c`. Paths and input-materialization states below describe that original workspace. Current module usage/status is in [README](../README.md); this submission checkout contains no private match data.

# 최종 트래킹 코드의 입력·실행 조건

현재 기준은 [확정 코드 경로](PIPELINE.md)다. 폐기한 분기와 상류의 별도 작업을
현재 로컬 코드 누락처럼 표시하지 않는다.

| 구분 | 현재 확인된 상태 | 처리 |
|---|---|---|
| 최종 사용 코드 | 실제 입력/출력 관계와 원문을 확인한49파일 | `model/phase1_extraction/ball_tracking/`에 모음 |
| 최초 H 변환·병합 | 실제 사용한 원문 확인 | 해당 Git 원문을 별도 코드 폴더에만 포함. 새 구현 없음 |
| 정렬 공통4파일 | 현재 checkout에 없고 동일해시 보존본 존재 | 별도 코드 폴더에 포함. 기존 작업 폴더는 수정하지 않음 |
| 폐기한 선/H 보정 및 V4 | 최종 입력이 아님 | 포함·복구·실행하지 않음 |
| 별도 ATD baseline | 최종 입력이 아님 | 현재 제출 대상 코드 목록에서 제외 |
| SNP 상류 object_audit | 이미 보정된 CSV를 지정 입력으로 수집 | 상류 내부 생성기를 로컬 필수 실행 단계로 추가하지 않음 |
| Broadcast 원본 ATD | 현재134bytes Git LFS 포인터 | OID `a71c1c9391e564032446be91ffbd7564205b47966060181532d80a338685e458`, 실제141,966,465bytes 자산 준비 필요 |
| Tactical 원본 ATD | 현재134bytes Git LFS 포인터 | OID `f130e350d2bfdf734b7e32f072931fccc6d015c1e837d082064875fe0a700431`, 실제195,714,285bytes 자산 준비 필요 |
| 원본 H의 줄끝 | 현재 LF, 당시 CRLF. 그 외 바이트 내용 동일 | 과거 원문의 해시 assert용으로 원본 형식 입력 복사본 필요. 원본을 수정하지 않음 |
| 검출/모델/특징/승인 자산 | 기존 실행에서 실제 참조한 private 입력 | 코드와 함께 공개하거나 새로 라벨 생성하지 않음. 실제 실행에는 manifest의 입력 필요 |
| V3 소스 경로 차이 | 좌표식 동일, 검증 보고서 복사 경로1표현만 변경 | 과거 실행 SHA를 재구성해 일치 확인, manifest에 양쪽 SHA 기록 |
| 점유 CSV | 이 작업은 트래킹 마무리 범위 | H/A/D/X 추론을 새로 구현하거나 연결 완료로 표시하지 않음 |

**코드 선택/정리 완료와 현재 환경에서 재실행 가능한지는 별개다.** 코드49파일을 정리했고
최종 데이터에 반영된 경로를 확인했지만, 위 ATD 실데이터가 없는 상태에서 실행됐다고 표시하지 않는다.
새 보정·추론·코드 재구현은 하지 않았다.
