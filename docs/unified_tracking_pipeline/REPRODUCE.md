# 확인된 트래킹 실행 순서와 범위

현재 순서/코드 역할의 기준은 [확정 경로](../submission_pipeline/FINAL_TRACKING_PIPELINE.md)와
[코드 모음](../../submission/final_tracking/README.md)이다. 파일 이름만 보고 viewer/검사/생성기를
분류하지 않았다. 새 자동 실행기는 만들지 않았다.

## 먼저 필요한 입력

SNP 고정 전·후반 보정 CSV, Broadcast/Tactical 원본 ATD/XML/H/영상, 기존 bbox/검출 bank,
모델·특징·승인/검수 기록이 필요하다. 코드만으로 새 객체 판정을 만든다고 설명하지 않는다.
현재 checkout의 ATD2개는 LFS 포인터이며 실내용이 없으므로 먼저 별도 데이터 작업본에
실데이터를 준비해야 한다. H는 CRLF/LF만 달라 원문 해시 검사를 위해 당시 형식 복사본이 필요하다.
기존 결과를 지우거나 덮어쓰는 절차는 진행하지 않았다.

## 실행할 함수/단계의 경계

1. 원본 matcher로 정렬표 생성. 점유 blank/X 준비표는 최종 tracking 입력으로 사용하지 않는다.
2. 고정 SNP 입력에서 `audit_corrected_ball_homography.py --skip-visuals`: 원본 H 변환·최초 병합.
3. FP 입력/bbox·특징·score·승인 적용.
4. uniform context infer/score → **export_ball_uniform_context.py apply** → 요청 큐.
5. actual X full-frame detector 또는 검증된 완료 X raw 재사용 → X score → X-only finalizer.
6. green export와 기존 검수 입력 audit → reviewed pitch.
7. restore_full_ball_observations.py:build. 옛 fill standalone prepare 결과는 읽지 않는다.
8. precision 모델/후보 입력 → apply_ball_precision.py → **export_ball_precision_view.py**의 high-precision CSV.
9. refine_ball_temporal_ai.py의 bank/train/associate/build. 과거 operational_model_selection learned 설정 유지.
10. export_broadcast_atd_from_input_csv.py로 새 ATD 생성.
11. analyze_broadcast_atd_coordinate_difference.py → visualize_atd_y_difference.py로 필수 Y 검사 CSV 생성.
12. restore_atd_original_by_y_difference.py의 **export() 함수**로 XY 복원.
13. correct_broadcast_atd_temporal_v2.py **correct**.
14. merge_tactical_into_corrected_ball_v2.py로 보정본에 Tactical 공 대체.
15. validate_ball_smoothed_v3_model.py가 만드는 model-selection 입력 → smooth_merged_ball_teleports_v3.py.
16. build_unified_tracking_v1.py로 V3 공과 Tactical 선수 통합.
17. unified용 compiler → unified renderer prepare/render/verify.

Y 복원의 CLI는 prepare/render/verify다. **`... export` CLI는 존재하지 않는다.** `prepare`는 좌표 export와
영상 준비를 함께 호출하므로 좌표만 처리할 때에는 기존 export() 함수가 작업 경계다.
`fill_ball_bidirectional.py`는 predict_gap/homography_project/local_histogram 공유 함수용이며
standalone prepare를 새 좌표 단계로 실행하지 않는다.

`deliver_ball_temporal_ai.py`, `finalize_ball_precision_delivery.py`,
`finalize_broadcast_atd_temporal_v2.py package`, `finalize_merged_ball_smoothed_v3.py package`는
기존 결과를 JSON/영상/검증 자료로 포장하는 단계다. 현재 필수 좌표 생산 목록에서는 제외했다.
필요하면 기존 원문 보관본에서 부가 산출물 범위로 별도 검토한다.

## 최종 마무리 명령

아래는 필요한 입력이 준비되고 해당 출력 경로가 비어 있는 별도 작업본에서의 기존 명령이다.
현재 최종 결과는 보존하며 이번 정리에서 실행하지 않았다.

```bash
python starter/build_unified_tracking_v1.py
python starter/compile_ball_tracking_renderers.py --target unified
python starter/render_unified_tracking_v1.py prepare
python starter/render_unified_tracking_v1.py render
python starter/render_unified_tracking_v1.py verify
```

환경 버전은 코드 모음의 ENVIRONMENT_POSTPROCESS.json / ENVIRONMENT_DETECTOR.json에 보존했다.
전체 경로를 fresh install/end-to-end로 이번에 실행했다고 표시하지 않는다.
