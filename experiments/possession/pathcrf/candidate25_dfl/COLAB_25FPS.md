# 전체 25fps Colab 실행

DFL_25fps_25epochs_MinLoss.ipynb를 Colab에 업로드하고 GPU 런타임에서 모두 실행한다. 실행 코드가 노트북 안에 내장되어 있으므로 별도 Python 파일 업로드는 필요 없다. 기존 Drive의 MyDrive/transition_final/transition_colab_data.zip을 재사용한다.

기본 설정은 전체 인플레이 25fps, 25 epoch, 얼리스탑 없음, 최소 검증 loss 체크포인트, AMP FP16, 유효 배치 64/실제 배치 64다. 5학습·J03WQQ 검증·J03WR9 평가 분할을 유지한다. 5fps H/A와 경합 게이트를 사용하지 않으며 기존 예측 D만 구간 경계에 사용한다. H/A CE·가중 전환 CE·CRF NLL을 더한 검증 loss로 best.pt를 고른다. 검증 loss가 3회 정체하면 학습률을 절반으로 낮추고 25 epoch까지 계속한다.

새 실행 경로는 MyDrive/transition_final/runs/dfl_25fps_25epochs_minloss/다. 실제 파라미터는 parameters.csv, 설정 설명은 parameter_descriptions.csv, epoch별 결과는 training_history.csv, 평가 결과는 holdout_comparison.csv, 원본 축 예측은 estimated_D_primary_J03WR9_H1.csv/H2.csv에 저장된다. 마지막 완료 epoch의 모델·optimizer·AMP scaler·난수 상태는 last.pt에 저장된다. 같은 노트북과 RUN_NAME으로 재실행하면 이어서 학습한다. 설정을 바꾸면 새 RUN_NAME을 사용한다.

기존 하이브리드 결과를 덮어쓰지 않는다. 실행 시간 벤치마크는 기본 비활성이고 RUN_BENCHMARK=True로 켤 수 있다. 노트북과 Python 문법만 로컬에서 확인했으며 실제 Colab GPU 학습은 사용자가 실행한다.
