# Dead Final 데드볼 실험

## TranSPORTmer Model 1 · 하프 단위 7폴드

[실행 코드](run_model1.py)는 기존 Dead Final 1 가중치에서 출발하는 Model 1을 학습하고, 같은 원시 확률에 네 가지 판정 조건을 적용해 점수와 CSV를 만든다.

### 준비와 실행

1. Python 3.12 환경에서 장치에 맞는 명령 하나로 의존성을 설치한다. Mac에서 LightGBM이 `libomp`를 찾지 못하면 `brew install libomp`를 실행한다.

   ```sh
   # NVIDIA CUDA 12.8
   python -m pip install -r ../../experiments/transportmer_dfl/requirements-lock.txt
   # Mac
   python -m pip install -r requirements-model1.txt "torch==2.8.0"
   ```

2. 저장소에 포함된 [DFL 7경기 XML 압축본](../../data/README.md)을 첫 실행에서 Parquet로 변환한다. `data/external/dfl/`에 이미 Parquet 캐시가 있으면 그대로 읽는다. 초기 가중치 7개와 [학습 경기·해시 기록](weights/initialization.json)은 이 폴더의 `weights/`에 있다. 별도 복사본을 쓰려면 `--weights-dir <경로>`에서 가중치와 같은 폴더에 `initialization.json`을 둔다.
3. 이 폴더에서 확인하고 실행한다.

   ```sh
   python run_model1.py --check
   python run_model1.py
   ```

   장치는 CUDA → Mac MPS → CPU 순으로 잡는다. 지정하려면 `--device cuda` 또는 `--device mps`를 붙인다. CUDA 기본 마이크로배치는 24, MPS·CPU는 8이고 논리 배치는 모두 24다. 중단 후 같은 명령을 실행하면 동일한 입력·설정에서 저장된 단계와 완료 폴드를 이어 간다. 새 실험은 `--output <새 폴더>`로 지정한다.

### 데이터와 초기 가중치

| 항목 | 설정 |
|---|---|
| 분할 | [기존 고정 분할](../../experiments/lightgbm_dfl/dfl_half_7fold.json), 시드 7. 경기 7개에서 얻은 14개 하프를 사용하며 각 폴드는 학습 풀 12개, 평가 2개다. |
| 모델 | 공 복원기와 데드볼 분류기. 128차원, 16헤드, FFN 512, 9.6초 창, 입력 6.25Hz, 출력 25fps. |
| 특징 | 선수·공 좌표, 연속 위치 차분, 좌표 관측 여부. 공 트랙을 목표 40.7%, 평균 6초 길이로 가리고 복원한다. DFL 이벤트는 분류와 보정 입력에 쓰지 않는다. 데드볼 정답은 `BallStatus`다. |
| 초기 가중치 | 폴드마다 기존 Dead Final 1 체크포인트를 고르고 호환되는 attention의 Q/K/V와 출력 투영 24개 텐서를 복사한다. 구조가 달라지는 입력층·정규화층·FFN·출력층은 Model 1 초기값을 사용한다. 공 복원기는 해당 폴드 안에서 학습한다. |
| 학습 | 창 이동 240프레임, 최대 25에폭, 조기 종료 patience 4. 공 복원 학습률 0.001, 분류 학습률 0.0002. 학습 풀 안 한 하프로 에폭을 고르고 학습 풀 12개 하프로 재학습한다. |

기존 체크포인트는 한 경기 전체를 빼고 학습했다. 이번 폴드의 평가 하프 두 개는 서로 다른 경기여서, 기존 가중치 중 겹침이 가장 적은 파일을 골라도 한 하프는 사전 학습에 들어가 있다. 결과표에 이 하프를 따로 표시하고 **초기 가중치가 보지 않은 하프의 점수**도 남긴다. 이 실행은 가중치 재사용과 학습 시간을 보는 실험이다. 기존 하프 7폴드의 **0.9097**은 원래 Model 1 실험의 참고값이다.

### 판정 네 가지

| CSV 열 | 적용 |
|---|---|
| `raw_dead` | 모델의 원시 확률에 0.5 임계값 적용 |
| `six_rules` | ① 경계를 넘거나 중단 신호가 있을 때 D 진입을 쉽게 함 ② 라인아웃 직전의 이른 D 억제 ③ 공이 재개된 움직임에서 인플레이 전환을 쉽게 함 ④ 재개 준비 중에는 D 유지 ⑤ 패스·운반 구간의 인플레이 보존 ⑥ 짧은 이탈과 재개 쌍 보존 |
| `line_out` | 6보정 위에 관측 공의 라인아웃 고정. 바깥 방향으로 움직이고 경계 밖 0.36m 이상이 2프레임 연속 확인되면 바깥 관측 구간을 D로 고정한다. 직전 플레이가 확인되면 해당 프레임을 인플레이로 고정한다. |
| `line_out_viterbi` | 같은 고정 조건을 디코더에 넣어 하프의 비터비 경로를 다시 계산한다. 고정 구간 주변의 판정도 바뀔 수 있다. |

추가 라인아웃의 고정은 관측 공만 사용한다. 6보정은 Model 1의 기존 방식대로 관측·복원 궤적을 함께 보고, 복원 위치에서만 나온 교차는 주변 관측이 받쳐줄 때 강제 근거로 쓴다. 점수는 하프 안에서 IoU 0.3부터 0.7까지 계산한 구간 F1을 폴드별로 평균한 후 7폴드를 평균한다.

### 실행 후 확인할 결과

기본 출력은 이 폴더의 `model1_7fold/`다. 아래 파일은 실행을 마치면 생성된다.

| 산출물 | 내용 |
|---|---|
| [결과표](model1_7fold/comparison/RESULTS.md), [요약 CSV](model1_7fold/comparison/summary.csv) | 원시 모델과 보정 세 단계 각각의 데드볼·인플레이 Spell F1, 데드볼 프레임 F1 |
| [하프별·폴드별 점수](model1_7fold/comparison/) | `by_half.csv`, `by_fold.csv`, 사전 학습에 포함되지 않은 평가 하프의 점수 |
| [프레임별 예측 CSV](model1_7fold/comparison/predictions/) | 14개 하프의 프레임 ID, 시간, 정답, 확률, 네 단계 판정, 초기 학습 겹침 표시 |
| [파라미터](model1_7fold/parameters.json), [학습 시간](model1_7fold/comparison/fit_times.csv) | 학습 설정, 가중치 출처·해시, 선택 에폭과 폴드별 소요 시간 |
| `ball_*_plain_train.refit.pt`, `dead_*_plain_restored_train.refit.pt` | 각 폴드의 공 복원기·데드볼 분류기 최종 가중치. 정확한 파일명은 `parameters.json`에 기록 |

참고로 기존 무이벤트 Model 1의 하프 7폴드 참고점수는 원시 모델 **0.7536**, 6보정 **0.9097**(데드볼 Spell F1)이었다. 이번 초기화와 추가 규칙의 결과는 새 결과표에서 확인한다. [기존 점수](../../experiments/transportmer_dfl/reference/summary.csv)


[기존 Dead Final 1의 경기 단위 실험 기록](README_DEAD_FINAL_1.md)
