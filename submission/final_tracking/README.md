# 최종 반영 트래킹 코드

이 폴더는 현재 최종 데이터의 실제 입력 관계로 확인한 소스 모음입니다. 새 알고리즘이나 새 통합 runner를 작성하지 않았습니다. 기존 좌표·영상·모델을 재생성하지 않았습니다.

**최종 대상:** `outputs/unified_tracking_v1/` — V3 공 + Tactical 선수46슬롯, Broadcast164,487프레임.

## 확정한 실행 범위

| 순서 | 단계 | 코드 |
|---|---|---|
| 00_alignment | Broadcast/Tactical frame alignment | `starter/final_tracking_alignment.py` |
| 01_fixed_SNP_assets | Pinned incoming corrected SNP first/second-half assets |  |
| 02_initial_H_merge | Initial original-H projection and half merge | `starter/audit_corrected_ball_homography.py` |
| 03_FP | Additional FP selection | `starter/prepare_ball_fp_inputs.py`<br>`starter/ball_fp_evidence.py`<br>`starter/score_ball_false_positives.py`<br>`starter/apply_ball_fp_reduction.py` |
| 04_uniform_context | Uniform/context evidence for queue and later precision | `starter/ball_uniform_context.py`<br>`starter/export_ball_uniform_context.py` |
| 05_queue | Targeted full-frame redetection queue | `starter/prepare_ball_redetection_frames.py` |
| 06_X_detection | Actual YOLO26X full-frame detections | `yolo26_football_detection/src/redetect_ball_frames.py` |
| 07_X_score | Score actual X candidates and select | `starter/infer_ball_yolo26_binary_head.py`<br>`starter/finalize_ball_yolo26x.py` |
| 08_X_pitch_review | Original-H projection and recorded candidate review | `starter/export_yolo26x_green_pitch.py`<br>`starter/audit_yolo26x_green_balls.py`<br>`starter/audit_yolo26x_green_independent.py` |
| 09_restore | Restore original SNP observations and local X replacements | `starter/restore_full_ball_observations.py` |
| 10_precision_inputs | Precision features/model/candidate preparation | `starter/harden_ball_precision.py`<br>`starter/finalize_ball_precision_candidates.py` |
| 11_precision | Precision coordinate selection | `starter/apply_ball_precision.py` |
| 12_precision_export | Select high-precision coordinates | `starter/export_ball_precision_view.py` |
| 13_temporal | Two-ended temporal candidate recovery/prediction | `starter/refine_ball_temporal_ai.py` |
| 14_ATD | Apply the selected CSV to a new Broadcast ATD | `starter/export_broadcast_atd_from_input_csv.py` |
| 15_Y_guard | Required Y-difference guard inputs | `starter/analyze_broadcast_atd_coordinate_difference.py`<br>`starter/visualize_atd_y_difference.py` |
| 16_Y_restore | Restore original ATD XY at normalized Y threshold | `starter/restore_atd_original_by_y_difference.py` |
| 17_temporal_V2 | Two-ended temporal correction V2 | `starter/correct_broadcast_atd_temporal_v2.py` |
| 18_tactical_ball | Tactical ball replacement in the corrected destination | `starter/merge_tactical_into_corrected_ball_v2.py` |
| 19_V3 | Accepted local V3 repair | `starter/smooth_merged_ball_teleports_v3.py` |
| 20_unified | Retained V3 ball + Tactical players | `starter/build_unified_tracking_v1.py` |
| 21_final_video | One full-length final tracking video | `starter/compile_ball_tracking_renderers.py`<br>`starter/render_unified_tracking_v1.py`<br>`starter/render_unified_tracking_native_v1.m` |

## 역할을 구분해서 사용할 것

- `CODE_MANIFEST.json`: 실제 입력/출력 경로와 해시, 파일별 역할·의존성·원본 위치.
- `CODE_FILES.txt`: 이 폴더에 포함된 소스만 나열한 목록. 데이터·가중치·검수 기록은 포함하지 않습니다.
- 최초 H 변환기는 Git에 남은 원문을 이 폴더에만 바이트 그대로 복사했습니다. 기존 작업 폴더의 삭제 파일은 복원하지 않았습니다.
- 현재 체크아웃에 없는 정렬 공통4파일은 동일해시 보존본에서 복사했습니다. 원래 작업 폴더는 변경하지 않았습니다.
- `export_ball_uniform_context.py`는 큐/판정 입력을 만들고, `export_ball_precision_view.py`는 실제 고정밀 CSV를 만듭니다. 이름에 export/view가 있어도 필요한 단계입니다.
- `fill_ball_bidirectional.py`는 공통함수만 사용합니다. 옛 `prepare` 출력은 실행 경로에서 제외합니다.
- `compare_broadcast_ball_coordinates.py`, benchmark/classifier 모듈은 필요한 import/helper 때문에 포함됩니다. 별도 비교 영상이나 다른 모델 결과를 좌표 경로에 섞지 않습니다.
- Y 차이 분석/시각화2파일은 다음 Y 복원 코드가 읽는 검사 CSV를 만들기 때문에 필요합니다.
- `restore_atd_original_by_y_difference.py`의 좌표함수는 `export()`입니다. CLI에 export 옵션은 없고 prepare는 영상 준비까지 실행합니다.
- V3 현재 원문의 차이는 검증 보고서 복사 경로1곳뿐이며, 기록된 실행 원문 SHA까지 대조했습니다.

## 실행 조건

원본 데이터와 고정 SNP 입력, 기존 검출/참조/승인/모델 자산을 manifest에 기록된 상대경로에 준비해야 합니다. 이 코드 모음만으로 공식 영상에서 새로운 객체 판정을 자동 생성했다고 주장하지 않습니다. 기존 결과를 덮어쓰는 일괄 실행은 하지 않습니다.

Y 좌표만 생성하려면 기존 함수 `export()`를 호출해야 합니다. `python starter/restore_atd_original_by_y_difference.py export`는 잘못된 명령입니다. 시간 보정 V2의 실제 CLI는 `python starter/correct_broadcast_atd_temporal_v2.py correct`입니다.

최초 변환은 `python starter/audit_corrected_ball_homography.py --skip-visuals`로 원래 H 변환·병합 부분만 실행할 수 있습니다. 이것은 별도의 선/H 보정기와 다릅니다. 이번 작업에서는 실행하지 않았습니다.

마지막 기존 실행 순서(입력 준비 및 새 출력 경로가 필요한 명령, 이번 미실행):

```bash
python starter/build_unified_tracking_v1.py
python starter/compile_ball_tracking_renderers.py --target unified
python starter/render_unified_tracking_v1.py prepare
python starter/render_unified_tracking_v1.py render
python starter/render_unified_tracking_v1.py verify
```

## 제외한 분기

- Discarded line/H repair: `starter/correct_broadcast_ball.py`, `starter/verify_corrected_broadcast_ball.py`, `data/derived/ball_tracking_correction_20261116`
- Discarded stronger smoothing: `V4 smoothing`
- Separate raw-ATD baseline: `submission_tracking/`, `outputs/submission_tracking_v1/`
- Old partial-fill standalone output: `data/derived/ball_bidirectional_fill_20261116/`
- Model comparison coordinate alternatives: `all_candidate_verdicts.csv (mixed S+new-X)`
- Non-coordinate packaging/finalizers: `starter/finalize_ball_precision_delivery.py`, `starter/finalize_broadcast_atd_temporal_v2.py`, `starter/finalize_merged_ball_smoothed_v3.py`, `starter/deliver_ball_temporal_ai.py`
- Other experiments: `3D`, `longpass`, `goalpost camera`, `unused AIRBORNE replacement`

## 검토 상태

소스 원문·입력 해시·의존성·구문을 확인한 로컬 검토본입니다. 새 end-to-end 실행이나 최종 점유 CSV 재현 완료를 뜻하지 않습니다. 기존 객체 검수/고정 판정이 들어 있는 원문도 포함하므로 공개 저장소에 올릴 파일·라벨 출처는 별도 검토해야 합니다. Commit/push/업로드는 하지 않았습니다.

## 현재 checkout의 입력 준비 상태

Broadcast/Tactical ATD2개는 Git LFS 포인터이며 실제 내용은 로컬 캐시에 없습니다. 포인터 OID는 당시 입력 해시와 일치하지만 실내용 검증/재실행은 이번에 하지 못합니다. 원본 H는 당시 CRLF와 현재 LF의 줄끝 차이만 있습니다. 과거 원문 assert에 맞는 원본 형식 복사본을 준비해야 하며, 이번에는 현재 데이터 파일을 변경하지 않았습니다. 세부 상태는 VERIFICATION.json의 LFS_inputs_not_materialized와 line_ending_only_input_variants를 보세요.
