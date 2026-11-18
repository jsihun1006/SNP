# dead_ball: 선수 움직임 기반 데드볼 판정 (Phase 1)

**보정 실험과 BAR–ATH 적용** · [LightGBM v2 보정 안내](README_lightgbm_corrections.md)

전술 ATD에서 **프레임마다 데드볼 여부**를 만든다. H/A 점유는 판단하지 않는다.
Phase 2 점유 모델이 D 상태를 정할 때 입력으로 쓰라고 만든 특징 파일이다
(예: model1의 "긴 공 누락 → D" 대신 `dead`를 사용).

## 방식

```text
선수 특징(팀 단위 속도·대형·라인까지 거리 등, 공 미사용)
→ LightGBM: 공개 DFL/IDSSE 7경기로 학습(대회 경기 라벨 사용 안 함)
→ 2상태 비터비 평활화(전환 벌점 40)
→ 공이 라인을 넘는 것이 보이면 그 순간 데드로 들어가는 벌점 0, 그런 라인 아웃은 공이 안에서 다시 보이고
  모델이 인플레이로 볼 때까지 무조건 데드
→ 데드볼 시작을 라인 아웃 프레임에 맞춤
→ 공개 킥오프(FAQ.md): 후반 110808–113741은 D
```

- 공은 라인 아웃 판정에만 쓴다. **공이 안 보인다는 이유만으로 D로 보지 않는다.**
- 학습·검증 데이터의 공 상태(DFL은 데드볼 동안 공 고정, Metrica 샘플은 공 삭제)가 정답을 드러내므로
  모델 입력에는 공 특징을 넣지 않았다.
- 코드는 저장소 `src/`에 있고, 이 폴더는 진입점과 형식 검사만 둔다. 설정과 근거는 `src/config.py`.

## 근거 (정답이 있는 공개 데이터)

| 검증 | 인플레이 구간 F1* |
|---|---|
| DFL 7경기, 경기 하나씩 빼고 학습 (`src/deadball/eval_crossing.py`) | 0.877 |
| Metrica 샘플 2경기 (학습에 안 씀, 라인 아웃 처리 전, `src/deadball/metrica_eval.py`) | 0.79 / 0.81 |

\* 공식 채점기의 spell 점수를 인플레이 = H, 데드 = D로 두고 계산한 값. 데드볼이 점유 구간을 얼마나 정확히 나누는지 본다.

## 실행

저장소 루트에서, 대회 데이터가 `data/`에 있는 환경으로 실행한다.

```bash
python -m pip install -r src/deadball/lightgbm/requirements.txt
python src/deadball/lightgbm/extract.py                  # src 파이프라인 실행 + 내보내기
python src/deadball/lightgbm/extract.py --skip-pipeline  # 이미 만든 결과만 내보내기
python src/deadball/lightgbm/validate.py artifacts/deadball/lightgbm/<실행ID>
```

## LightGBM 모델 파일

| 항목 | 값 |
|---|---|
| 파일 | `artifacts/deadball/lightgbm/dfl_final/weights/model_all_v2.txt` (LightGBM 텍스트 모델, 약 2.8 MB) |
| 재학습본 SHA-256 | `ada5dec2f4cf5b3685d013e8363588405e1c96b6fa012837680e85f5ec0bad84` |
| 기존 HSJ 배포본 SHA-256 | `18a46eb5dd1e89f1057edc163e04b5f33b178a5c63123d7b4b2078724683737e` |
| 학습 데이터 | Sportec Open DFL(IDSSE) 7경기. Bassek et al. 2025, Scientific Data 12, 195. CC-BY 4.0(DFL 표기) |
| 학습 설정 | `src/config.py` (특징 세트 v2, `lgbm_params`, `sim_seed` 7) |

이 저장소의 모델은 공개 DFL 7경기로 재학습한 버전이다. 원본 데이터 출처와 실험 결과 경로는 [data/README.md](../../../data/README.md)를 참고한다.

직접 학습하려면 DFL 원본 XML 7경기 × 3파일을 `data/external/dfl/raw/`에 둔다. 경기 ID는 `src/dfl_load.py`의 `MATCHES`에서 확인할 수 있다.

```bash
python src/dfl_load.py all
python src/deadball/train_dfl.py --save-final
```

## 출력: `artifacts/deadball/lightgbm/<실행ID>/` (Git 제외)

| 파일 | 내용 |
|---|---|
| `features.csv` | 프레임당 1행, 199,319행 |
| `frame_status.csv` | 프레임당 1행: `frame,time_seconds,processed,ball_seen` |
| `run_manifest.json` | 입력 파일·모델 파일 해시, 설정값, 코드 커밋, 패키지 버전 |
| `validation.json` | `validate.py` 검사 결과 |

### `features.csv` 열

| 열 | 의미 |
|---|---|
| `frame` | 전술 영상 0기반 프레임 (ATD `frame_id` = frame + 1) |
| `time_seconds` | frame / 25 |
| `graded` | 채점 구간(전·후반)이면 1 |
| `dead_score` | 모델의 프레임별 데드볼 확률(평활화 전, 0~1) |
| `dead` | 최종 판정. 1 = 데드볼. 채점 구간 밖은 항상 0 |
| `dead_source` | 데드인 이유: `model`, `model+out_start`(시작을 라인 아웃에 맞춤), `forced_line_out`, `public_kickoff`. 인플레이면 빈칸 |

## 한계

- 선수 움직임만 보므로, 공이 돌고 있는 느린 빌드업을 짧은 데드볼로 볼 때가 있다.
- 2D 트래킹 흔들림으로 라인 근처 공이 잠깐 밖으로 찍히면 가짜 라인 아웃이 될 수 있다(`dead_source = forced_line_out`).
- Metrica 기준으로 데드볼을 실제보다 4~5%p 넓게 잡는다. DFL 숫자는 대회 성능보다 낙관적이다.
- 자세한 시도·폐기 기록은 [연구 기록](../../../docs/research/LESSONS.md).

## DFL 선수별 attention 실험

- **노트북** · [dfl_transportmer.ipynb](../../../notebooks/deadball/dfl_transportmer.ipynb). 저장소 루트에서 PyTorch 2.15와 SciPy가 설치된 커널로 셀을 위에서부터 실행한다.
- **입력** · `data/external/dfl/`의 트래킹·이벤트·경기 정보 Parquet. 선수별 좌표와 관측 마스크로 9.6초 창을 만들고, `BallStatus`로 학습한다.
- **검증** · 6경기 학습·1경기 평가를 7번 반복한다. 6.25Hz 모델 확률을 하프별로 25fps에 맞추고 기존 비터비·라인아웃 보정을 적용한다.
- **실측** · 인플레이 구간 F1은 보정 전 0.5222, 보정 후 0.5898이다. 기존 LightGBM은 0.8450, 0.8770이다. 선수별 attention 실험은 Metal에서 준비부터 평가까지 약 13분 5초 걸렸다.
- **결과** · `artifacts/deadball/transportmer/dfl_transportmer_20261112_01/`의 `model_summary.csv`·`per_match.csv`·`training_history.csv`. 학습 경기와 입력·가중치 해시는 `run_manifest.json`에 기록한다.
- **프레임 출력** · `matches/<경기ID>/features.csv`의 `dead_score`는 데드볼 확률, `dead_viterbi`는 비터비 판정, `dead`는 라인아웃 보정 후 판정이다. `frame_status.csv`의 `model_sampled`는 모델이 직접 계산한 시점을 표시한다. `source_frame_id`는 DFL 원본 프레임 ID다.

### 공 좌표 복원과 데드볼 공동 학습

- **실행** · 같은 Python 환경에서 [dfl_transportmer_joint.ipynb](../../../notebooks/deadball/dfl_transportmer_joint.ipynb)의 셀을 순서대로 실행한다. 입력은 동일한 DFL Parquet 7경기다.
- **설정** · 임베딩 128, 16헤드, 9.6초 창, 최대 60 epoch. 공 좌표는 모두 가리고, 실제로 관측된 인플레이 좌표에만 복원 손실을 계산한다.
- **조기 종료** · 학습 쪽 6경기의 각 하프 중앙 12% 창을 내부 검증에 사용한다. 6 epoch부터 비터비 적용 후 인플레이 구간 F1을 보고, 5회 연속 0.001 이상 개선되지 않으면 멈춘다. 바깥 평가 경기는 별도로 유지한다.
- **실측** · 7경기 평균은 라인아웃 보정 후 데드볼 프레임 F1 0.9159, 재현율 92.06%, 인플레이 구간 F1 0.7745다. 조기 종료는 15~24 epoch, 평가용 모델 선택은 10~19 epoch였다. 준비부터 평가까지 약 2시간 26분 걸렸다.
- **결과** · `artifacts/deadball/transportmer/dfl_transportmer_joint_20261112_01/`. `model_summary.csv`와 `per_match.csv`에 데드볼 프레임 F1·정밀도·재현율, 인플레이 구간 F1, 인플레이 공 좌표 ADE를 저장한다. 프레임 출력 컬럼은 위 실험과 같다.

### tranSPORTmer_D · 데드볼 전용

- **실행** · [tranSPORTmer_D.ipynb](../../../notebooks/deadball/tranSPORTmer_D.ipynb). 같은 폴더에 있는 기본 및 공동 학습 노트북의 모델·전처리를 재사용한다.
- **비교** · 공동 학습과 같은 입력 해시, 128차원·16헤드, 9.6초 창, 최대 60 epoch와 조기 종료를 적용한다. 손실은 `4 × 데드볼 BCE`다.
- **실측** · 7경기 평균은 라인아웃 보정 후 데드볼 프레임 F1 0.9067, 인플레이 구간 F1 0.7492다. 공동 학습보다 각각 0.0093, 0.0253 낮다. 평가 모델은 7~20 epoch에서 선택됐고, 전체 2시간 27분 걸렸다.
- **결과** · `artifacts/deadball/transportmer/transportmer_d_20261112_01/`에 `per_match.csv`, `model_summary.csv`, `comparison.csv` 및 경기별 프레임 출력이 저장된다.

### tranSPORTmer_D · 어려운 구간 오버샘플링

- **실행** · [tranSPORTmer_D_oversampling.ipynb](../../../notebooks/deadball/tranSPORTmer_D_oversampling.ipynb). 데드볼 전용 실험의 모델·손실·6경기 학습/1경기 평가를 재사용한다.
- **조건 B** · 10초 미만 중단의 시작·재개 ±1초에 닿는 학습 창에 3배 가중치를 준다. 에폭당 추출 창 수는 기존과 같다.
- **조건 C** · 첫 5 epoch는 B처럼 학습한다. 이후 실제 인플레이 창에서 연속 2초 이상 데드볼로 잘못 예측한 창에도 같은 3배 가중치를 준다.
- **평가** · 인플레이 Spell F1, 데드볼 프레임 F1, 3초 미만·3~10초 미만 중단의 재현율, 인플레이 오탐 구간 수를 비교한다.
- **실측** · 라인아웃 보정 후 7경기 평균 프레임 F1/Spell F1은 기존 0.9067/0.7492, B 0.8998/0.7312, C 0.8835/0.7192다. 10초 미만 중단 검출은 160개 중 101/98/93개, 인플레이 오탐은 64/68/46개였다. 학습·평가는 B 약 2시간 54분, C 약 2시간 35분 걸렸다.
- **결과 경로** · `artifacts/deadball/transportmer/transportmer_d_oversampling_20261112_01/`의 `comparison.csv`와 조건별 `per_match.csv`, `model_summary.csv`, `run_manifest.json`.
