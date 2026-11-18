# DFL 경합 전환 재판정

DFL 선수·볼 추적으로 기존 5fps PathCRF의 경합 후보를 25fps에서 다시 판정한다.

Colab 업로드용 노트북은 [DFL_Transition_Colab.ipynb](../../../../notebooks/possession/DFL_Transition_Colab.ipynb)다. 데이터·모델 묶음은 `data/transition_final/colab/transition_colab_data.zip`이다. 실행과 Drive·CSV 저장 순서는 [COLAB.md](COLAB.md)에 정리했다.

## 파일

- `parameters.csv`에 확정 파라미터, 시간 측정 후 검토할 값, 구현 가정을 기록한다.
- `EXPERIMENT_PLAN.md`에 데이터 분할, 모델, 학습과 평가 절차를 기록한다.
- `src/possession/pathcrf_refiner/run.py`는 `benchmark`, `train`, `test` 진입점을 제공한다.
- 같은 소스 디렉터리의 `inputs.py`, `refiner.py`, `prepare_tracking.py`가 입력과 모델을 구현한다.
- 실행 중 캐시·가중치·예측은 Git에서 제외되는 `data/transition_final/`에 저장한다.

## 로컬 실행

저장소 루트에서 실행한다. 현재 Mac의 Python과 이미 준비된 패키지 경로를 `run_local.sh`가 사용한다. 별도 환경에서는 `TRANSITION_PYTHON`과 `PYTHONPATH`를 지정한다.

```sh
sh 'src/possession/pathcrf_refiner/run_local.sh' benchmark --device mps --microbatch 32 --bench-steps 5 --full-validation-probe --output data/transition_final/local_run
sh 'src/possession/pathcrf_refiner/run_local.sh' train --device mps --microbatch 32 --output data/transition_final/local_run
sh 'src/possession/pathcrf_refiner/run_local.sh' test --device mps --microbatch 32 --output data/transition_final/local_run
```

학습 예산 변경은 별도 출력 경로와 함께 실행한다. 아래는 epoch당 4,000창으로 줄이는 예시다.

```sh
sh 'src/possession/pathcrf_refiner/run_local.sh' train --device mps --microbatch 32 --epoch-window-cap 4000 --output data/transition_final/cap4000
sh 'src/possession/pathcrf_refiner/run_local.sh' test --device mps --microbatch 32 --epoch-window-cap 4000 --output data/transition_final/cap4000
```

같은 train 명령을 다시 실행하면 `last.pt`의 모델·Adam 상태·완료 epoch·난수 상태에서 이어진다. epoch 도중 중단하면 마지막 완료 epoch 이후부터 재실행한다. `best.pt`는 검증 Spell F1 기준이다. 예산이나 입력을 바꾸면 새 출력 경로를 사용한다.

## 필요한 로컬 자료

- `data/external/dfl/*_tracking.parquet`, `*_meta.parquet`
- `data/transition_final/source/`의 공식 좌표 추적 캐시 또는 `data/external/dfl/raw/`의 기존 XML.gz
- 공식 PathCRF checkout과 `saved/140/model/state_dict_best_acc.pt`
- `data/pathcrf_dfl/scene_two_models_v1/`의 `manifest.json`, `full/best.pt`, `full/fit.json`

데드볼 기준은 이 구현의 기본값으로 같은 5·1·1 분할을 사용한 기존 full-half scene 모델을 선택했다. 사용자가 보유한 다른 dead-ball 모델을 적용하려면 해당 모델의 프레임 정렬 출력과 학습 분할을 확인해서 `frozen_predictions` 어댑터를 연결한다. 이번 측정에 사용한 모델 경로와 SHA256은 manifest에 남는다.

`run_local.sh`는 기존 공식 running-feature 캐시도 재사용한다. 이 캐시는 5fps PathCRF용이다. 25fps 속도·가속도는 관측 단절과 데드볼 경계를 나눠 새로 계산한다.

## Colab GPU 실행

로컬 측정 후 같은 폴더, 저장소 코드와 위 자료를 Colab에 옮긴다. 공개 원본 데이터의 추가 다운로드는 필요하지 않다. PyTorch가 포함된 GPU 런타임에서 `requirements.txt`의 보조 패키지를 설치하고 실행한다. `--upstream`은 옮겨 둔 공식 checkout 위치다.

```sh
python 'src/possession/pathcrf_refiner/run.py' benchmark --device cuda --source-device cpu --upstream /content/pathcrf --microbatch 32 --bench-steps 10 --full-validation-probe --output data/transition_final/colab_run
python 'src/possession/pathcrf_refiner/run.py' train --device cuda --source-device cpu --upstream /content/pathcrf --microbatch 32 --output data/transition_final/colab_run
python 'src/possession/pathcrf_refiner/run.py' test --device cuda --source-device cpu --upstream /content/pathcrf --microbatch 32 --output data/transition_final/colab_run
```

실행 결과를 보존하려면 `--output`을 마운트한 Google Drive 아래에 둔다. 학습 입력은 런타임 로컬 디스크에 둔다. CPU, M5, Colab 실측은 각각 기록한다.

## 검증

```sh
TRANSITION_PYTHON=/path/to/python sh 'src/possession/pathcrf_refiner/run_local.sh' verify
```

CRF 전체 경로 열거와의 일치, 결측 라벨 구간, 경계 Viterbi와 패딩을 확인한다. DFL 짧은 학습·검증·재개 확인 기록은 실험계획서에 남긴다.

## 경합 후보 모델의 채택 결과

5fps PathCRF에서 표시한 구간만 25fps 모델로 재판정하고 고정 보정을 적용했다. 이 채택 조건의 설정은 `inference_parameters.json`에 있다. 위 로컬 명령의 원래 실험과 전체 25fps 실험은 별도 설정과 평가 결과를 사용한다.

DFL 학습 5경기·검증 J03WQQ·평가 J03WR9 조건에서, 예측 D 마스크를 고정한 기존 J03WR9 평가의 Spell F1은 76.22%였다. 가짜 전환 48개, 누락 112개, 매칭 시점 평균 절대오차 140ms다. 같은 홀드아웃으로 확인한 진단 수치다.

| 항목 | 경로 |
|---|---|
| 기존 29에폭 실험의 결과·예측 | `artifacts/possession/pathcrf/candidate25_dfl/historical_run/` |
| best·마지막 체크포인트 | 같은 실험의 `weights/best.pt`, `weights/last.pt` |
| 추론 진입점·가중치 공유 방식 | [possession README](../../../../src/possession/possession/README.md) |

5fps 1차 보정 결과의 중복 사본은 `artifacts/possession/pathcrf/correct_5fps_v1/`의 CSV·Parquet·가중치·요약을 참조한다. 캐시 `scores/`는 기존 경로에서 사용할 수 있다. 5fps 기준 모델의 오답 분석이 두 실행에서 일치하는 CSV도 같은 파일을 참조한다.

`notebooks/possession/`의 노트북은 기존 Colab ZIP에 의존한다. 그 ZIP 안 경로와 내장된 실행 코드는 당시 버전 재현용이다. 로컬 연구 코드의 재배치와 독립적이다.
