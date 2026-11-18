# Dead Final 1에서 Model 1로 초기화하는 실험

저장소 루트에서 실행한다. `check`는 DFL 캐시 21개, Dead Final 1 가중치 7개, 재사용할 공 복원 가중치, 대회 전술 ATD, 모델 규칙 및 장치를 확인한다. 학습은 `train`을 실행할 때 시작한다.

```sh
sh experiments/deadball/transportmer/dead_final_model1/run_local.sh check
sh experiments/deadball/transportmer/dead_final_model1/run_local.sh train
sh experiments/deadball/transportmer/dead_final_model1/run_local.sh promote
```

기본 장치는 Mac GPU(`mps`)다. 환경이 별도로 있으면 `MODEL1_PYTHON`에 Python 실행 경로를 지정한다. 패키지 구성은 기존 DFL TranSPORTmer 환경을 사용한다. NVIDIA 환경에서는 `check`와 `train`에 `--device cuda`를 붙인다.

초기 가중치의 학습 경기와 체크섬은 Dead Final 1 실험 기록에서 확인할 수 있다. 일곱 가중치는 `artifacts/deadball/transportmer/dead_final_1/weights/`를 읽는다. [초기화 명세](../dead_final_1/reference/initialization.json)

실행 중간 결과는 `artifacts/deadball/transportmer/dead_final_model1_warm_v1/`에 저장한다. 동일한 입력과 설정으로 `train`을 다시 시작하면 검증된 완료 폴드 및 가중치를 재사용한다. `promote`는 전체 실험과 BAR–ATH CSV 생성 및 검증까지 완료된 우수 후보를 설치한다. 이전 폴더는 실행 폴더의 `previous_dead_final`에 보관한다.

설치한 가중치로 바르셀로나 CSV를 다시 만들 때는 다음 명령을 쓴다.

```sh
sh experiments/deadball/transportmer/dead_final_model1/run_local.sh predict --work-dir artifacts/deadball/transportmer/dead_final_model1_inference
```

[실험계획](EXPERIMENT_PLAN.md)
