# BAR–ATH 공 보정 및 Tactical 선수 통합 트래킹

범위: **현재 최종 데이터에 실제 반영된 트래킹 경로**. 이 문서는 해당 코드·입력·산출물만
설명한다. 점유 H/A/D/X 추론은 이 트래킹 범위에 포함하지 않는다.

## 1. 접근과 입력

Broadcast 공 관측을 제공 Homography로 경기장 좌표에 연결하고, 원래 관측과 실제 재탐지,
Tactical 참조, 양방향 추정의 출처를 구분한다. 마지막에는 채택된 V3 공 좌표를 그대로 유지하면서
시간 대응이 있는 원본 Tactical 선수46슬롯을 결합한다. 정규화 좌표를 기본으로 보존한다.

로컬 처리의 시작 입력은 사용자가 지정한 SNP의 **보정 완료 전·후반 CSV**다. 전반은
`codex/ball-tracking-correction-20261115`의 `843c33078559fe095b96e5fd7ecd5c5a15c2a29e`,
후반은 `ball-tracking`의 `39f6e1ddc8f7a2081f058a48f3adc384666bdeea`다.
두 입력은 저장된 Git 원본과 대조했다. 이미 완성된 CSV를 가져오는 단계와 그 상류 내부 생성기를
다시 실행하는 작업을 혼동하지 않는다. SNP 원검출 소스 보관본은 `vendor/snp_ball_tracking/`에
있으며 이 로컬 확정 실행 목록은 고정 보정 CSV부터 시작한다.

제공 Broadcast/Tactical ATD·XML·H·영상, 원래 검출/bbox bank, 기존 검수/승인 기록과 모델/특징
자산을 사용한다. 이번 정리에서 새 검수 라벨·모델·관측을 만들지 않았다.

## 2. 프레임 정렬

기존 joint matcher는 30초 공 궤적, 팀별 선수 배치 대응, 네 가지 반사 가설을 검색하고 다음30초의
공 궤적을 별도로 확인한다. 공식 half 범위와 제공 자동 Half 메타데이터를 사용한다.
최종 데이터가 사용한 대응은 전반 T=B+4014, 후반 T=B+26919이며 각각 유효한 half 범위에만 적용된다.
반별 일정 오프셋을 사용하므로 모든 리플레이의 실제 촬영 시점까지 맞춘다는 주장은 하지 않는다.

## 3. 실제 적용 순서

| 순서 | 처리 | 결과/핵심 코드 |
|---|---|---|
| 1 | 정렬과 고정 SNP 입력 확인 | `final_tracking_alignment.py`, 원본 전·후반CSV |
| 2 | 후반 원본 H 역변환·전후반 최초 병합 | `audit_corrected_ball_homography.py:prepare/main` |
| 3 | 기존 관측/bbox·외형 특징·추가 FP 적용 | FP prepare/evidence/score/apply |
| 4 | 몸체/유니폼 문맥과 실제 요청 큐 | context → `export_ball_uniform_context.py` → redetection queue |
| 5 | 실제 X 검출·후보 점수화 | 완료된 X11,092프레임/34,784상자, X-only finalizer |
| 6 | 원본 H로 후보 투영·검수 입력 반영 | `export_yolo26x_green_pitch.py` |
| 7 | 원래 SNP 관측 전체 복구 | FP 원본과 검수된 X 후보에서 restored CSV 생성 |
| 8 | 정밀 모델/후보 준비·좌표 선택 | harden/candidate finalizer → `apply_ball_precision.py` |
| 9 | 실제 고정밀 CSV 선별 | `export_ball_precision_view.py`가 temporal 입력 CSV 생산 |
| 10 | 양끝 관측 사이의 시간 보정 | `refine_ball_temporal_ai.py` |
| 11 | 새 ATD에 적용·Y 조건의 원본 XY 복원 | fromCSV → 차이 검사 CSV → Y export |
| 12 | 시간 문맥 보정 V2 | `correct_broadcast_atd_temporal_v2.py` |
| 13 | 보정본에 Tactical 공 대체 | 원래 temporal 결측 마스크와 보정 V2 좌표를 구분 |
| 14 | 채택된 V3 국소 보정 | merge V2 → 실제 참조 재선택/양방향 Hermite |
| 15 | Tactical 선수 결합과 최종 영상 | unified builder → unified renderer |

파일별 함수와 실제 입출력은 [확정 코드 경로](FINAL_TRACKING_PIPELINE.md)와
[기계 명세](../../submission/final_tracking/CODE_MANIFEST.json)에 있다.

초기 H는 정규화 경기장→영상 방향이다. 후반 영상 관측에 같은 source_frame의 역H를 적용하고
전반은 기존 정규화 좌표를 유지한다. 최초 변환/비교는105.3×68m, 최종 ATD/XML과 metre 열은105×68m다.
정규화 XY를 전달하므로 초기 metre와 최종 metre를 단위 확인 없이 직접 비교하지 않는다.

Y 조건은 `(Y 복원 직전 보정본 y_m − 원본 ATD y_m)/68 ≤ −0.1`이며 **XY 쌍**을 원본으로
복원했다. H 교체나 Y만 수정하는 처리로 설명하지 않는다. Tactical 결합은 원래 temporal CSV의
경기장 결측을 마스크로 쓰되, 실제 좌표 대상은 시간/Y 보정 후 V2다. H 무효만으로 하이라이트를
판단하지 않고 선수 가용성도 확인한다.

## 4. V3와 최종 결과

V3는 실제로 최종 입력이다. 최종 manifest와 현재 V3 파일 SHA256이 일치한다. V3는 짧은
이상/출처 경계에서1,346프레임/292구간을 수정했으며,657프레임은 Tactical 기록 재선택,
689프레임은 비관측 양방향 추정이다. 예측을 새 앵커로 사용하지 않고 기존 결측을 새로 채우지 않았다.
Hermite 선택은 기록 관측을 가린 대리 복원 평가에 근거한다. 실제 공 위치 GT 정확도 평가는 아니다.
최대 약102.26m 점프가 남아 있어 전부 해결했다는 주장은 하지 않는다.

| 최종 결과 | 값 |
|---|---:|
| Broadcast 전체 프레임 / FPS | 164,487 / 25 |
| Tactical 선수 슬롯 | 46(Barcelona24/Athletic22) |
| 공 유효 / 결측 | 107,517 / 56,970 |
| 선수 대응 / 미대응 | 155,923 / 8,564 |
| 기존 예측 유지 | 5,203(V3 신규689 포함, 합산하지 않음) |
| 최종 metre 좌표 | 105×68m |

Builder는 V3 공과 정렬된 Tactical 선수 ATD payload를 그대로 유지하며 새 평활화나 선수 예측을
추가하지 않았다. `s`는 kph 속도이고 높이가 아니다. 위치 변경 후 재계산한 속도로 설명하지 않는다.
슬롯 번호는 확인된 선수ID·유니폼 번호가 아니다.

## 5. 제외한 것

결과가 폐기된 **선/H 보정 분기**, 더 강한 **V4**, 별도로 만든 원본 ATD baseline, 옛 부분 보간
결과,3D·롱패스·골대 카메라·미채택 AIRBORNE 대체는 포함하지 않는다. N/S/M/L 비교 좌표나
S+new-X 혼합 verdict를 최종 좌표에 섞지 않았다. 필요한 공통 함수와 실제 임계값 입력만 유지했다.
중간 비교 영상/HTML와 완료·패키징 전용 코드를 좌표 생산 단계로 나열하지 않았다.

## 6. 코드 및 검증

[별도 코드 폴더](../../submission/final_tracking/README.md)에 원문49파일만 모았다.
실제 코드가 없는 정렬4파일은 동일 해시 보존본에서 복사했고 최초 변환기는 확인된 Git 원문을
이 폴더에만 복사했다. 다른 원본 코드는 수정하지 않았다. V3의 현재 소스는 실행 원문과 검증
보고서 복사 경로1표현만 다르며, 그 표현을 메모리에서 되돌리면 당시 실행 SHA와 정확히 일치한다.

이번 검증은 소스49파일의 바이트/구문/공통 의존성과 실제 저장된 산출물·입력 관계를 확인했다.
최초 병합→FP 배열, restored→precision 이전 좌표 열, high-precision→temporal 이전 좌표 열도
전수 비교했다. 최종 CSV/TXT/XML/roster4개는 기존 integrity 해시와 일치한다. 새로운 추론이나
end-to-end 재생성은 하지 않았다. 결과는 [VERIFICATION.json](../../submission/final_tracking/VERIFICATION.json)에 있다.

현재 체크아웃의 Broadcast/Tactical ATD는 실데이터가 아닌 Git LFS 포인터다. 포인터 OID는
당시 입력 해시와 일치하지만 현재 실내용은 로컬 LFS 캐시에 없어 재실행 전에 준비해야 한다.
현재 H는 원본과 CRLF/LF 줄끝만 다르다. 기존 원문의 바이트 해시 assert를 실행하려면 원본
형식의 입력 복사본이 필요하다. 이번에 원본 경로를 수정하지 않았다.

## 7. 제출 연결과 한계

원본 `SUBMISSION.md`는 보존했다. 이 트래킹 CSV는 대회가 요구하는 Tactical199,319행의
`frame,label` 점유 CSV가 아니다. 점유 모델·loose-ball/경합/재시작 규칙을 여기에 새로 적용하지 않았다.
실제 위치 정답 정확도 향상,100% 공 탐지, 모든 리플레이 동기화를 주장하지 않는다.

기존 경로에는 객체 검수·고정 판단·경기 내 분류 head가 있다. 이를 전부 무라벨 알고리즘이라고
표시하지 않는다. 공개 저장소/대회 규칙에 맞는 라벨 출처·가중치/자산 배포 범위 검토는 별도이며,
이번 코드 모음은 로컬 검토본이다. Git stage/commit/push·외부 제출은 하지 않았다.
