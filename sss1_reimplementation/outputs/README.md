`predictions.csv`: 원본 픽셀 bbox, confidence, pelvis/ground, 선택적인 world 좌표.
결측은 빈 필드, JSON에서는 null. `world_valid=False`인 좌표는 평가의 world 검출에서 제외됩니다.
`results.json`: COCO image_id/category_id/xywh bbox/score와 유효할 때만 position_on_pitch.
`timing.json`: 프레임별 추론 및 IO 포함 전체 처리시간. 초기 모델 로딩 제외, 첫 프레임 warmup 포함.
`visualizations/`: 프레임별 bbox·두 점·유효 world 좌표 표시.

`validation/`은 이 작업에서 실제 실행한 CPU 테스트 증거입니다.
합성 fixture 평가 점수와 공개 샘플 탐지는 SynLoc 벤치마크 실측이 아닙니다.
