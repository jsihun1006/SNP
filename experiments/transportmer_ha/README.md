# DFL 인플레이 점유 전환 실험

TranSPORTmer가 인플레이 프레임의 점유 팀을 홈(H) 또는 원정(A)으로 예측한다. 공개 DFL 7경기 중 개발 6경기에서 OOF 예측과 오답노트를 만들고, 홀드아웃 1경기에서 일반 학습 B와 오답 집중 학습 C를 비교한다.

## Colab에서 실행하기

로컬 작업 복사본의 `data/transportmer_ha/`에 준비한 아래 파일 두 개가 필요하다. 두 파일은 별도로 보관한다. 노트북에 실험 코드와 볼 결측률·구간 길이 집계가 포함되어 있다. ZIP은 공개 DFL 7경기의 트래킹·이벤트·메타데이터 Parquet 21개를 담고 있으며 크기는 약 85 MiB다. 노트북 셀 3에서 ZIP의 SHA-256을 확인하므로 함께 준비한 ZIP을 사용한다. [DFL 공개 원본 및 CC BY 4.0](https://doi.org/10.6084/m9.figshare.28196177)

| 파일 | 사용 위치 |
| --- | --- |
| `DFL_TranSPORTmer_Colab_로컬캐시.ipynb` | Colab 노트북 업로드 |
| `DFL_7matches_Parquet.zip` | 노트북 셀 3의 파일 선택 |

1. Colab에서 노트북을 열고 **런타임 → 런타임 유형 변경 → GPU**를 선택한다.
2. **런타임 → 모두 실행**을 누르고 Google Drive 연결을 승인한다.
3. 셀 3에 파일 선택이 나타나면 ZIP을 올린다. Parquet는 `내 드라이브/DFL_TranSPORTmer/parquet_local/`에 저장된다. 다음 실행부터는 이 캐시를 재사용한다.
4. 셀 4에서 배치 24로 짧은 학습·추론을 한 뒤 `projected_hours`를 출력한다. 이어서 본 실험을 실행한다. T4에서 21,235 학습 스텝의 벤치마크 예상치는 1.82~1.92시간이었다. 데이터 준비와 Drive 저장 시간은 별도다. 각 OOF fold를 마치면 `remaining_hours`가 실측 학습 속도로 갱신된다.
5. 셀 5에서 6경기 OOF 평활화 평균과 홀드아웃 B/C 비교표를 확인한다. 결과 경로는 `내 드라이브/DFL_TranSPORTmer/seed2027_local_cache/`다.

학습은 epoch가 끝날 때마다 모델과 optimizer 상태를 Drive에 저장한다. 중지하거나 런타임이 종료되면 같은 노트북을 GPU로 연결하고 **모두 실행**을 눌러 마지막으로 저장한 epoch 다음부터 이어간다. 완료한 OOF 예측과 DFL 캐시도 재사용한다. 브라우저 연결만 끊겼다면 다시 접속해 셀 4가 실행 중인지 먼저 확인한다. [Colab 런타임 안내](https://research.google.com/colaboratory/faq.html#using-colab)

## 실험 순서

1. **경기 단위 분할.** Seed `2027`로 홀드아웃 `J03WOH`를 고정한다. 개발 경기 `J03WOY`, `J03WQQ`, `J03WR9`, `J03WN1`, `J03WMX`, `J03WPY`에서 매번 한 경기를 제외하고 나머지 5경기로 학습한다. 한 경기의 전·후반은 같은 fold에 둔다.
2. **입력 준비.** DFL의 `BallStatus`로 데드볼 경계를 정하고, 인플레이의 `ball_owning_team`을 H=0, A=1로 매핑한다. 팀 라벨 결측은 손실과 평가에서 마스킹한다. 원본 프레임 번호와 시간은 출력까지 유지한다. MTAH에서 데드볼 모델이 인플레이로 분류한 프레임의 볼 결측률과 결측 구간 길이 분포로 DFL 볼 좌표에 연속 결측을 모사한다. 합성 처리분 중 25%를 선형 보간으로 대체하는 것은 시뮬레이션 가정이다. 각 하프의 증강은 고정된 seed로 생성해 비교 조건 간 재사용한다.
3. **OOF 학습과 예측.** 정규화 좌표, 홈·원정·볼 구분, 관측 마스크와 속도, 속도 유효 마스크를 사용한다. 속도를 계산할 때 두 샘플 사이의 원본 프레임까지 같은 트랙이 관측됐는지 확인한다. 샘플링은 6.25 Hz, 창은 최대 60시점이다. 창은 하프와 데드볼을 넘지 않으며 재개 첫 시점에는 재개 표시를 넣는다. 학습 창을 epoch마다 섞고 창 내부 순서를 유지한다. 학습 창 간격은 4.8초, 추론 창 간격은 2.4초다. 겹치는 logit을 평균하고 인플레이 구간 안에서 25 fps로 보간한다.
4. **모델과 학습 예산.** 임베딩 128, attention head 16, FFN 512로 설정한다. 시간 attention 2층과 선수 간 attention 1층을 한 블록으로 두 번 적용한다. 배치는 24이며 BCE with logits, AdamW `weight_decay=0.01`, gradient clipping `5.0`을 사용한다. OOF의 각 모델은 학습률 `2e-4`로 5 epoch 워밍업한 뒤 `1e-4`로 15 epoch 학습한다.
5. **평활화와 오답노트.** 개발 6경기의 원시 예측, 50프레임 다수결, P(A)의 15·25·35프레임 이동평균을 비교한다. 경기별 spellF1의 평균으로 고르고 동률이면 2초 미만 점유 재현율을 비교한다. 원시·선택 예측이 모두 틀린 구간은 C의 오답 풀에 넣는다. 평활화가 해결하거나 새로 만든 오답도 집계한다. 오답노트에 원본 프레임 범위, 시작 시점의 정답·예측, 전환 오류 유형과 상황 태그를 기록한다. 구간 전체 라벨은 프레임별 CSV에서 확인할 수 있다.
6. **B/C 홀드아웃 비교.** 개발 6경기로 공통 5 epoch 워밍업 체크포인트를 만든다. B는 일반 창 24개, C는 일반 창 12개와 OOF 오답 구간에 겹치는 창 12개를 사용한다. C는 오류 유형, 오답 구간, 주변 창 순으로 복원추출한다. 두 조건 모두 `1e-4`에서 일반 학습 15 epoch 분량의 같은 스텝 수로 학습한다. 모델과 평활화를 고정한 뒤 홀드아웃에 같은 볼 증강과 D 마스크를 적용한다. 주 지표는 spellF1이며 가짜 전환, 누락 전환, 매칭된 전환의 평균 절대 시점 오차(ms), 2초 미만 점유 재현율도 기록한다. 전환은 방향이 같고 차이가 25프레임 이내일 때 매칭한다.

## 결과 파일

| 파일 | 내용 |
| --- | --- |
| `manifest.json`, `ball_profile.json` | 분할·seed·소스 해시, 볼 결측 집계 |
| `development_halves.json`, `holdout_augmentation.json` | 하프별 프레임과 실제 볼 마스킹·보간 집계 |
| `benchmark.json`, `remaining.json`, `oof_timing.json`, `timing.json` | 장치, 벤치마크, 남은 시간, 실측 시간 |
| `oof_smoothing_by_match.csv`, `oof_smoothing_mean.csv`, `oof_selection.json` | 경기별 평활화 점수, 평균, 선택 결과 |
| `error_ledger.csv`, `oof_probabilities/` | OOF 오답 구간과 원본 프레임별 정답·예측 |
| `locked_holdout_recipe.json`, `holdout_comparison.csv` | B/C 스텝 수와 오답 샘플링 집계, 홀드아웃 1경기 비교 |
| `holdout_B/`, `holdout_C/` | 동일한 홀드아웃의 B/C 프레임별 예측 |
| `reproduce.sh`, `colab_run.log` | Colab에서 사용한 명령과 실행 로그 |

셀 4가 시작할 때 `reproduce.sh`를 저장한다. 이 스크립트를 저장한 뒤에는 노트북의 셀 1~3을 실행한 Colab 런타임에서 아래 명령으로 벤치마크와 본 실험을 재현할 수 있다.

```sh
bash /content/drive/MyDrive/DFL_TranSPORTmer/seed2027_local_cache/reproduce.sh
```
