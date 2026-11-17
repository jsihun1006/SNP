> Historical collection record from `final` commit `acf854c`. Paths and input-materialization states below describe that original workspace. Current module usage/status is in [README](../README.md); this submission checkout contains no private match data.

# 최종 데이터에 실제 반영된 트래킹 코드

이 문서는 **현재 저장된 최종 데이터의 입력 해시에서 역추적해 확인한 경로**만 정리한다.
파일명이나 과거 설명을 근거로 적용 여부를 추정하지 않았다. 기존 최종 데이터·영상은
유지하고, 확인한 소스만 별도 `model/phase1_extraction/ball_tracking/`에 모았다. 새 보정 알고리즘이나
일괄 실행기를 만들지 않았으며 트래킹을 재실행하지 않았다.

## 1. 확정한 최종 대상

| 항목 | 값 |
|---|---|
| 데이터 | `outputs/unified_tracking_v1/` |
| 공 입력 | `outputs/broadcast_tactical_ball_smoothed_v3/`의 CSV·ATD |
| 선수 입력 | 원본 Tactical ATD/XML의 전체46슬롯 |
| 정렬 입력 | `data/derived/sync/final_tracking_alignment.csv` |
| 전체 시간축 | Broadcast164,487프레임,25fps |
| 공 유효 / 결측 | 107,517 / 56,970프레임 |
| 선수 대응 / 미대응 | 155,923 / 8,564프레임 |
| 최종 단위 | 정규화 XY,105×68m,속도 s는 kph |

**V3는 실제 반영돼 있다.** 최종 manifest가 V3 CSV의 SHA256
`32c406da92142da3d64e9383941a4d60817d933031125089476a674ef461aa8e`를 입력으로 기록하고,
현재 파일도 그 값과 일치한다. 폐기한 것은 더 강한 V4 평활화와 별도 선/H 보정 분기다.

최종 CSV SHA256은 `bdd017ba3f8335028a951631768780af570e71bf064c7bb850572af5419b0608`이다.

## 2. 확정 소스 위치

- [실제 소스 모음과 안내](../README.md)
- [포함한 코드49파일 목록](../CODE_FILES.txt)
- [파일 역할·의존성·입출력·SHA256](../CODE_MANIFEST.json)

기존 작업 폴더의 코드는 수정하지 않았다. 현재 체크아웃에서 빠진 정렬 공통4파일은
동일 해시 보존본에서, 최초 H 변환·병합기는 Git 원문 blob에서 **이 별도 폴더에만** 복사했다.
폐기한 보정기와 검사기는 복구하지 않았다.

## 3. 처음부터 마무리까지의 실제 경로

아래 파일은 별도 소스 모음 안의 상대경로이며, `.py`는 `starter/` 아래에 있다.

| 순서 | 실제 사용 단계 | 코드·함수 | 실제로 이어지는 출력/용도 |
|---|---|---|---|
| 0 | Broadcast↔Tactical 정렬 | `final_tracking_alignment.py` 및 joint/layout matcher | 최종에서 읽는 `final_tracking_alignment.csv` |
| 1 | 지정한 SNP 보정 원본 확보 | 고정 Git 커밋의 CSV 수집 | 전반·후반 `sources/*.csv`가 로컬 처리의 시작 입력 |
| 2 | 원본 H로 후반 변환·최초 병합 | `audit_corrected_ball_homography.py:prepare/main` | 최초 `combined_corrected_ball_pitch.csv` |
| 3 | 기존 관측/bbox 준비 | `download_ball_fp_reference.py`, `prepare_ball_fp_inputs.py` | `fp_inputs.npz` 및 원본 bbox/참조 입력 |
| 4 | 외형 특징·추가 FP 적용 | `ball_fp_evidence.py`, `score_ball_false_positives.py`, `apply_ball_fp_reduction.py:apply` | FP `filtered_ball_coordinates.csv` |
| 5 | 유니폼/몸체 문맥 | `ball_uniform_context.py`, **`export_ball_uniform_context.py:apply`** | 문맥 판단·필터 입력·추가 거절 근거 |
| 6 | 실제 재탐지 요청 큐 | `prepare_ball_redetection_frames.py` | `frames_to_redetect.csv` |
| 7 | 실제 YOLO26X 검출 | `yolo26_football_detection/src/redetect_ball_frames.py` | 11,092프레임·34,784개 X 후보. 완료된 X 작업 재사용 |
| 8 | X 후보 점수/선택 | `infer_ball_yolo26_binary_head.py`, `finalize_ball_yolo26x.py:prepare/finalize` | X-only `all_candidates.csv` |
| 9 | 원본 H 투영·후보 검수 반영 | `export_yolo26x_green_pitch.py:export/audit`, green audit 기록 생성 코드 | `reviewed_ball_pitch.csv` |
| 10 | 원래 SNP 관측 전체 복구 | `restore_full_ball_observations.py:build` | FP 원본+검수된 X 관측에서 전체 restored CSV 생성 |
| 11 | 정밀화 입력·모델·후보 | `harden_ball_precision.py`, `finalize_ball_precision_candidates.py` | 실제 적용 v2/v3 head·후보 점수·판정 입력 |
| 12 | 정밀 좌표 선택 | `apply_ball_precision.py:apply` | precision 전체 좌표 CSV |
| 13 | 고정밀 좌표 선별 | **`export_ball_precision_view.py:main`** | temporal 단계가 실제로 읽는 high-precision CSV |
| 14 | 양방향 시간 보정 | `refine_ball_temporal_ai.py:bank/train/associate/build` | `ball_temporal_ai_20261117/ball_homography_all_frames.csv` |
| 15 | CSV 공 좌표를 새 ATD에 적용 | `export_broadcast_atd_from_input_csv.py` | `broadcast_atd_from_csv_v1` ATD |
| 16 | Y 복원에 필요한 검사 입력 | `analyze_broadcast_atd_coordinate_difference.py`, `visualize_atd_y_difference.py` | Y 복원 코드가 실제 읽고 검증하는 차이 CSV |
| 17 | Y 임계값의 원본 XY 복원 | `restore_atd_original_by_y_difference.py:export()` | `broadcast_atd_y_minus_0p1_original_v1` ATD |
| 18 | 시간 보정 V2 | `correct_broadcast_atd_temporal_v2.py:run()` | `broadcast_atd_temporal_v2` CSV·ATD |
| 19 | 보정본에 Tactical 공 결합 | `merge_tactical_into_corrected_ball_v2.py` | `broadcast_tactical_ball_merge_v2` CSV·ATD |
| 20 | 채택한 V3 국소 보정 | `smooth_merged_ball_teleports_v3.py` | 최종 builder가 직접 사용하는 V3 CSV·ATD |
| 21 | 선수 좌표 통합 | `build_unified_tracking_v1.py` | 최종 tracking CSV/TXT/XML/roster |
| 22 | 최종 영상 하나 | unified용 compiler·`render_unified_tracking_v1.py`·native renderer | `unified_tracking.mp4` |

SNP 전반 입력은 `843c33078559fe095b96e5fd7ecd5c5a15c2a29e`, 후반 입력은
`39f6e1ddc8f7a2081f058a48f3adc384666bdeea`와 바이트 일치한다. 이미 보정된 CSV를 가져오는
것이 이 로컬 경로의 시작이다. 그 상류 내부 생성기를 로컬 필수 실행 단계로 추가하지 않았다.

원래 H 방향은 정규화 경기장→영상이다. 최초 변환은 동일 source_frame의 H 역행렬을
사용하며 전반 기존 정규화 값은 유지한다. 최초 단계의105.3×68m와 최종 XML의105×68m를
구분하고, ATD 적용은 정규화 XY를 전달한다.

## 4. 파일 단위로 잘못 분류하지 않도록 확정한 사항

- `export_ball_uniform_context.py`는 필수 큐/판정 입력을 생산한다. 단순 뷰어로 제외하지 않는다.
- `export_ball_precision_view.py`는 실제 고정밀 CSV를 생산한다. 단순 뷰어가 아니다.
- `fill_ball_bidirectional.py`는 공통 함수 `predict_gap`, `homography_project`,
  `local_histogram` 때문에 필요하다. 그 파일의 옛 standalone `prepare` 출력은 최종 입력이 아니다.
- `compare_broadcast_ball_coordinates.py`는 상수/공통 함수 import 때문에 포함한다.
  별도 비교 영상 생성은 좌표 생산 순서에 넣지 않는다.
- benchmark/classifier 모듈은 실제 공유 함수·모델/임계값 준비에 필요한 부분만 사용한다.
  N/S/M/L 비교 결과를 최종 좌표로 넣지 않았다.
- `decide_ball_yolo26_candidates.py`의 **임계값 JSON**은 사용하지만,
  S+new-X 혼합 `all_candidate_verdicts.csv`는 최종 좌표 입력이 아니다.
- Y 차이 분석/시각화 두 파일은 검사 CSV를 생성하므로 필요한 입력 준비 단계다.
- `deliver_ball_temporal_ai.py`와 후속 `finalize_*` 완료/영상/JSON 패키징은 좌표 생산자가
  아니므로 필수 좌표 실행 목록에 넣지 않았다. 최종 영상용 renderer는 별도로 포함했다.

Y 복원의 기존 CLI는 `prepare/render/verify`다. **`... export` CLI 명령은 존재하지 않는다.**
좌표를 바꾸는 함수는 `export()`이며 CLI `prepare`는 영상 준비도 함께 호출한다.
시간 보정 V2의 CLI는 `python starter/correct_broadcast_atd_temporal_v2.py correct`다.

## 5. 제외를 확정한 분기

| 제외 대상 | 확인 근거 |
|---|---|
| `correct_broadcast_ball.py`, `verify_corrected_broadcast_ball.py`, `ball_tracking_correction_20261116/` | 실제 생산자가 참조하지 않음. 최초 병합→FP 배열이 원본 H 변환본과 직접 일치 |
| 더 강한 V4 평활화 | 최종/V3 manifest의 `rejected_V4_used=false`, 실제 입력은 V3 해시 |
| 별도 `submission_tracking/` ATD baseline | 최종 데이터의 입력 경로에 없음 |
| 옛 `ball_bidirectional_fill_20261116` 결과 | restored 코드가 폐기된 부분 결과로 명시하며 그 좌표를 읽지 않음 |
| 3D·롱패스·골대 카메라·미채택 AIRBORNE 대체 | 최종 입력 관계에 없음 |
| 중간 비교 영상/HTML·모델 비교 좌표 | 최종 좌표 입력으로 쓰지 않음. 공유 함수만 필요한 경우 모듈은 유지 |

## 6. 원문·해시 검증 결과

전방에서는 최초 병합→FP 배열을 직접 비교하고, precision의 `before_precision_*`, temporal의
`previous_pass_*`를 전체164,487행에서 이전 좌표와 대조했다. 후방에서는 최종→V3→Tactical
merge→시간V2→Y복원→CSV적용→temporal 입력 해시를 확인했다. 기존 보고서의 경로명을
다시 적는 것만으로 적용을 판정하지 않았다.

V3 실행 원문 SHA는 `407b2a0fe7edb1cd2cd82c02b4c93b74e4d46fa991421e6232b3158c3faa0b77`,
현재 소스 SHA는 `a657cc109bbc1843eb044109db4f51025d1ddf3633094302533263a3f7b987ec`다.
차이는 검증 보고서 복사 경로 `/private/tmp`→`outputs/ball_tracking_validation` 한 표현뿐이다.
이 표현만 메모리에서 되돌리면 당시 SHA와 정확히 일치하며 좌표 계산 코드는 같다.

## 7. 이 정리의 완료 범위

소스49파일을 원문 바이트/해시·의존성으로 정리했다. 최초 변환기 원문과 현재 체크아웃에
없는 정렬4파일도 별도 코드 모음 안에 포함했다. 기존 root 소스·원본 입력·최종 데이터·영상은
변경하지 않았다. 원본 스냅샷·이미 쓰인 모델/검수 자산이 필요한 경로이며, 그 자산을 새로
만들거나 새 end-to-end 실행에 성공했다고 주장하지 않는다.

점유 H/A/D/X 분류와 대회 제출199,319행 CSV 생성은 이 트래킹 코드 범위에 넣지 않는다.
이번에는 Git stage/commit/push나 외부 업로드를 하지 않았다.

## 8. 현재 체크아웃에서 재실행하기 전의 입력 조건

코드 선택/정리는 완료했지만, 현재 입력 파일 상태까지 실행 준비 완료라고 표시하지 않는다.

| 입력 | 현재 상태 | 당시 원본 식별자 |
|---|---|---|
| Broadcast ATD | 134bytes Git LFS 포인터, 실내용 로컬 캐시 없음 | SHA256 `a71c1c9391e564032446be91ffbd7564205b47966060181532d80a338685e458`,141,966,465bytes |
| Tactical ATD | 134bytes Git LFS 포인터, 실내용 로컬 캐시 없음 | SHA256 `f130e350d2bfdf734b7e32f072931fccc6d015c1e837d082064875fe0a700431`,195,714,285bytes |
| Broadcast H | 당시 CRLF, 현재 LF. 줄끝을 제외한 내용 동일 | 원문 hash assert에 맞는 당시 형식 복사본 필요 |

따라서 원본 ATD2개의 **포인터 OID 일치와 실제 내용 해시 검증을 구분**했다.
이번 선언 연결71건 중55건은 실제 바이트 해시 일치이며, 나머지는 LFS 포인터 식별자 또는
H의 줄끝 차이로 따로 기록했다. 실제 저장된 산출물30개 해시와 최종4데이터 파일은 대조했다.
기존 산출물477파일의 크기·수정 시각도 유지됐다. 원본 데이터 경로를 덮어쓰거나 LFS pull을
실행하지 않았다. 자세한 상태는 코드 모음의 `VERIFICATION.json`에 있다.
