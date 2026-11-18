# Dead Final · TranSPORTmer Model 1

선수와 공의 추적 좌표에서 누락된 공 위치를 복원한 뒤 프레임별 데드볼을 판정한다. 학습은 Dead Final 1의 attention 투영 가중치에서 시작한다. 분류 출력에 여섯 가지 중단·재개 보정 규칙을 적용한다.

- `EXPERIMENT_PLAN.md`에 데이터 분할, 초기화, 학습과 선택 기준을 기록했다.
- `weights/ball_restorer.pt`와 `weights/dead_best.pt`는 DFL 7경기 전체로 학습한 대회 적용용 가중치다. DFL 평가용 가중치는 경기 ID를 접두어로 쓴다.
- `predictions/dfl/`에는 평가용 14개 하프의 `frame_id`, `truth_dead`, `dead_probability`, `before_rules`, `after_rules`가 있다.
- `predictions/BAR-ATH/`의 `frame_status.csv`와 `features.csv`는 전술 영상 원본 프레임 기준이다. `state`는 `D`, `ALIVE`, `X` 중 하나이며 `ball_observed`는 제공 ATD의 공 관측 여부다.
- `comparison/summary.csv`, `comparison/by_match.csv`, `comparison/by_half.csv`에서 실측 점수를 확인할 수 있다. `training/run_manifest.json`과 `validation.json`에는 입력·모델 설정과 파일 해시가 있다.

로컬 실행 안내는 저장소의 `experiments/deadball/transportmer/dead_final_model1/README.md`에 있다. 평가용 기존 Dead Final 1은 실행 폴더의 `previous_dead_final`에 보관한다.
