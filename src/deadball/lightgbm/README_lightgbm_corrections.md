# LightGBM v2 데드볼 보정

**영상에 적용한 방식**은 DFL 7경기로 학습한 LightGBM의 데드볼 확률에 **라인아웃·재개·패스 등 여섯 규칙을 결합**하는 것이다. [실행 노트북](../../../notebooks/deadball/LightGBM_deadball_corrections.ipynb)

## 실행

저장소 루트에서 Python 3.12 환경에 설치한다. macOS에서 LightGBM을 불러올 때 OpenMP 오류가 나면 `brew install libomp`를 실행한다.

```bash
python -m pip install -r src/deadball/lightgbm/requirements.txt "scipy==1.18.1" notebook
jupyter notebook notebooks/deadball/LightGBM_deadball_corrections.ipynb
```

| 마지막 셀의 `RUN_MODE` | 준비할 입력 | 실행 결과 |
|---|---|---|
| `competition` | `data/tactical/`에 `BAR-ATH_tactical_atd.xml`, `BAR-ATH_tactical_atd.txt`. DFL 학습 가중치는 저장소에 포함 | 전술 영상 채점 프레임별 `frame_status.csv` |
| `full` | 공개 DFL 7경기 XML 21개를 `data/external/dfl/raw/`에 배치. 기존 Parquet 캐시도 사용 가능 | 7경기 학습·평가와 조건별 결과 |
| `smoke` | `full`과 동일 | 첫 평가 경기 `J03WPY` 점수를 화면에 표시 |

1. 데이터 다운로드 경로를 확인한다. [DFL 공개 데이터와 대회 ATD 준비](../../../data/README.md)
2. DFL XML을 받은 경우 저장소 루트에서 `python src/dfl_load.py all`로 변환한다.
3. 설치한 Python 커널을 선택하고 마지막 셀의 `RUN_MODE`를 설정한 뒤 **Run All**을 누른다.
4. `artifacts/deadball/notebook_runs/lightgbm_corrections_<모드>/`에서 결과를 확인한다. 기존 결과 폴더가 있으면 실행 전에 `DFL_CORRECTIONS_RUN_ID`를 새 이름으로 지정한다.

BAR–ATH의 `frame`은 전술 영상의 0부터 시작하는 프레임 번호다. `dead_probability`는 선수 특징에서 얻은 기본 확률이고, `state`는 보정 이후 `D` 또는 `ALIVE`다. 공개 후반 킥오프 전 채점 구간은 영상에서 사용한 것과 동일하게 D로 표시한다.

## 여섯 보정

| 오류 | 추적 좌표로 적용하는 보정 |
|---|---|
| ① 데드볼 시작 늦음 | 공의 바깥 방향 교차와 재개가 이어지거나, 양 팀 감속과 공 진행 중단이 겹치면 데드볼 진입 비용을 낮춘다. |
| ② 데드볼 시작 빠름 | 확인된 라인 교차 직전에는 데드볼 진입 비용을 높인다. |
| ③ 재개 판정 늦음 | 킥·스로인 방출 이후 공의 이동에서 인플레이 전환 비용을 낮춘다. 공 관측이 약하면 근처 선수 움직임과 기본 확률을 함께 본다. |
| ④ 재개 판정 빠름 | 킥 전 접근·침투 같은 준비 동작에서는 인플레이 전환 비용을 높인다. |
| ⑤ 느린 인플레이 오탐 | 패스·수신 또는 공과 선수의 동행이 추정되면 인플레이에 힘을 싣는다. 확실한 라인 교차 주변은 교차를 우선한다. |
| ⑥ 짧은 데드볼 누락 | 짧은 간격에 교차와 재개가 함께 확인되면 두 전환 비용을 낮춘다. 별도 학습 조건에서는 중단·재개용 LightGBM도 시험한다. |

## DFL 평가 결과

25fps에서 6경기 학습·1경기 평가를 7번 수행했다. 규칙 설정은 학습 쪽에 속한 별도 검증 경기에서 선택했다. 프레임 정답은 DFL `BallStatus`, 이벤트는 오류 유형 집계에 사용했다. 휘슬·심판 신호는 이번 결과에 포함되지 않는다.

| 조건 | 데드볼 구간 F1 | 인플레이 구간 F1 | 데드볼 프레임 F1 | 공 노이즈 데드볼 구간 F1 |
|---|---:|---:|---:|---:|
| 기준 LightGBM + 라인아웃 | 0.8740 | 0.8770 | 0.9459 | 0.8353 |
| ⑤ 느린 인플레이 보정 | 0.8865 | 0.8944 | 0.9517 | 0.8449 |
| **①~⑥ 규칙 결합** | **0.8909** | **0.8920** | **0.9531** | **0.8482** |
| ①~⑥ 규칙 + 중단·재개용 LightGBM | 0.8755 | 0.8783 | 0.9495 | 0.8359 |

- **구간 F1 (Spell F1)**은 정답·예측 구간을 IoU로 맞춘 F1이다. 데드볼과 인플레이를 따로 계산한다.
- **공 노이즈**는 추적 좌표에 평균 2초 길이의 누락을 넣고(목표 누락 비율 25%) 0.20m 흔들림을 추가한 검증 조건이다.
- `RESULTS.md`에서 전체 비교표를 본다. `comparison.csv`는 조건별 평균, `per_match.csv`는 경기별 점수, `error_by_case.csv`는 고친 오류와 새 오류, `error_clips.csv`는 영상 확인용 프레임 구간이다. `run_manifest.json`과 `validation.json`에서 실행 설정과 검증 결과를 확인할 수 있다.
