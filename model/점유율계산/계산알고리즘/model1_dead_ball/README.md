# model1_dead_ball: dead_ball로 D를 정하고 model1로 H/A를 정하는 점유 모델 (Phase 2)

`model1`은 D를 "긴 관측 누락"으로 대신 추정한다. 이 모델은 그 자리에 **dead_ball(Phase 1)의 데드볼 판정**을 넣는다.

## 입력

| 입력 | 위치 |
|---|---|
| Phase 1 `dead_ball` | `model/영상추출/추출csv/dead_ball/<실행ID>/features.csv`의 `frame, graded, dead, dead_source` |
| 대회 ATD | `data/tactical/` (model1과 같은 최근접 팀 관측, `starter/baseline.nearest_player_labels`) |

공개·비공개 GT는 예측에 쓰지 않는다.

## 판단 방식

- model1과 같은 관측 비용·전환 비용을 쓴다(`config.json`의 값은 model1 그대로).
- **채점 구간에서 dead = 1인 프레임은 D만, dead = 0인 프레임은 H/A만 허용한다.** 그래서 데드볼 경계는 dead_ball이 정하고, 같은 인플레이 구간 안의 H↔A 전환만 model1 비용이 정한다.
- 채점 구간 밖은 model1처럼 제약 없이 푼다(채점에 들어가지 않는다).

## 실행

저장소 루트에서, 대회 데이터가 `data/`에 있는 환경으로 직접 실행한다.

```bash
python model/점유율계산/계산알고리즘/model1_dead_ball/run.py                    # dead_ball부터 전부
python model/점유율계산/계산알고리즘/model1_dead_ball/run.py --skip-dead-ball   # 이미 만든 dead_ball 사용
python model/점유율계산/계산알고리즘/model1_dead_ball/test_model.py             # 합성 데이터 검사
```

개별 실행: `possession.py`(예측) → `validate.py`(형식 검사) → `evaluation.py`(공개 5분 평가).
`--dead-run <실행ID>`로 쓸 dead_ball 실행을, `--run-id`로 이 예측의 실행ID를 정한다(기본은 dead_ball 실행ID와 같게).

## 출력

| 위치 | 파일 |
|---|---|
| `model/모델별결과csv/model1_dead_ball/<실행ID>/` | `prediction.csv`(frame,label), `frame_diagnostics.csv`, `possession_share.csv`, `run_manifest.json`, `validation.json` |
| `model/점유율계산/성능지표/model1_dead_ball/<실행ID>/` | `scoreboard.csv`, `reports/`, `RESULTS.md`, `evaluation_manifest.json` |
| `model/점유율계산/성능지표/모델_성능_비교.csv` | `model1_dead_ball__<실행ID>` 줄 추가(같은 이름이면 바꿈) |

`frame_diagnostics.csv`의 `input_source`: `dead_ball`(D로 고정), `model1_team`(H/A를 model1이 정함), `model1_ungraded`(채점 밖).

## 한계

- H/A는 공에 가장 가까운 선수 팀만 본다(model1). 공이 안 보이는 인플레이 구간은 전환 비용으로 앞뒤 팀을 이어 간다.
- 데드볼 경계 오류는 dead_ball의 오류를 그대로 물려받는다. dead_ball의 근거와 한계는 `model/영상추출/모델/dead_ball/README.md`와 저장소 `LESSONS.md`.
- 공개 5분 점수는 파이프라인 확인용이다. 설정을 고르거나 고치는 데 쓰지 않는다.
