# Colab 실행과 저장

1. `notebooks/possession/DFL_Transition_Colab.ipynb`를 Colab의 노트 업로드에서 연다.
2. 런타임 유형을 GPU로 선택한다.
3. 첫 셀을 실행하고 Google Drive 연결을 허용한다.
4. Google Drive 웹에서 MyDrive/transition_final 폴더에 data/transition_final/colab/slim/transition_colab_data.zip을 파일 업로드한다. Git 이력을 제외한 약 284MiB 묶음이다. Colab 첫 셀은 Drive에 올린 파일을 바로 읽는다.
5. 설정 셀에서 RUN_NAME, EPOCH_WINDOW_CAP, MAX_EPOCHS, MICROBATCH를 확인한다.
6. 시간 측정 셀을 실행한다. 할당 GPU 이름, 업데이트 시간, 전체 검증 시간, 최대 epoch 예상 시간이 출력된다.
7. 학습 셀을 실행한다. 학습 완료 후 평가·CSV 저장 셀을 실행한다.

## 기본 설정

| 항목 | 값 |
|---|---|
| 분할 | 5학습·1검증·1평가 |
| 기본 모델 | 5fps PathCRF |
| 경합 재판정 | 25fps, 4초, 100시점 |
| 크기 | 임베딩 128, 헤드 4, 공간·시간 블록 2 |
| 총 배치 | 64 |
| 실제 계산 배치 | 16, 4회 누적 |
| 옵티마이저 | Adam, lr 0.001 |
| epoch | 최대 40, 검증 5회 정체 시 종료 |
| 학습 창 상한 | 기본 0으로 전체 창 사용 |
| 빠른 실행 예시 | EPOCH_WINDOW_CAP = 2000 |
| 볼 결측 학습 목표 | 자연 결측 포함 9.5% |
| 예측 D | 동일 5·1·1로 학습된 기존 scene 모델 |

epoch당 창 수는 GPU 시간 측정 후 설정 셀에서 바꾼다. 본 학습 시작 후 설정을 바꾸면 RUN_NAME도 바꿔 새 실행 폴더를 사용한다. 처음 입력을 올리는 시간은 인터넷 업로드 속도에 따라 달라진다.

## 자동 저장 위치

Google Drive의 MyDrive/transition_final/runs/RUN_NAME/에 저장한다.

| 파일 | 내용 |
|---|---|
| parameters.csv | 해당 실행에 적용한 epoch와 창 상한을 포함한 파라미터 |
| colab_settings.json | 노트북의 실행 설정 |
| runtime_info.json | 실제 GPU와 PyTorch 버전 |
| manifest.json | 분할과 모델 출처, 입력 체크포인트 SHA256 |
| benchmark.json | GPU 실측과 학습시간 추정 |
| training.json | epoch별 검증 Spell F1과 loss |
| training_history.csv | 학습 이력 CSV |
| best.pt | 검증 Spell F1이 가장 높은 모델 가중치 |
| last.pt | 마지막 완료 epoch의 가중치·Adam 상태·난수 상태 |
| holdout.json | 테스트 경기의 기본 모델과 재판정 모델 비교 |
| holdout_comparison.csv | 예측 D와 정답 D 조건별 비교표 |
| estimated_D_primary_J03WR9_H1.csv, H2.csv | 예측 D를 적용한 프레임별 결과 |
| oracle_D_diagnostic_J03WR9_H1.csv, H2.csv | 정답 D 진단의 프레임별 결과 |
| benchmark.log, train.log, test.log | 실행 로그 |

프레임별 CSV 열은 match, half, frame, seconds, truth, dead, predicted_dead, base, refined, candidate다. H=0, A=1, D=2, 결측 정답=-1이다. 경기 원본 프레임 축에 정렬된다.

학습 완료 후 아래 코드를 Colab에서 실행하면 CSV·JSON·모델 파일을 Google Drive에서 찾을 수 있다.

```python
from pathlib import Path
saved = Path('/content/drive/MyDrive/transition_final/runs') / RUN_NAME
print(saved)
print(*sorted(p.name for p in saved.iterdir()), sep='\n')
```

## 중단 후 재개

같은 노트북에서 Drive 연결과 압축 해제부터 순서대로 실행한다. 같은 RUN_NAME과 설정으로 학습 셀을 실행하면 마지막 완료 epoch에서 이어진다. epoch 중간에 끊기면 그 epoch는 다시 실행한다. 브라우저 인터넷이 끊겨도 Colab 런타임이 유지되면 학습이 계속될 수 있다. 런타임이 종료되면 Drive 체크포인트에서 재개한다.

기존 parameters.csv, 로그와 CSV는 같은 RUN_NAME 아래 저장된다. 다른 설정의 결과를 보존하려면 새 RUN_NAME을 사용한다.
