# 5fps PathCRF 보정 비교

기존 PathCRF 가중치를 고정하고 6개 보정과 조합을 비교했다. 학습 경기 5개, 검증 J03WQQ, 테스트 J03WR9다. 테스트 경기는 이전 실험에서 이미 결과를 확인한 경기이므로 이번 수치는 재사용한 홀드아웃에서의 추가 진단이다.

보정값과 조합 선택은 검증 경기의 주최측 평균 Spell F1로 정했다. 동률이면 2초 미만 점유 재현율, 이후 보정 강도가 낮은 조건을 우선했다. 모든 조건은 이전 runs/dfl_5train_1val_1test의 5fps 결과와 같은 D 마스크를 사용한다. 원본 146,810프레임의 기준 예측과 프레임 축이 정확히 일치하는 것을 확인했다.

| 방법 | 구현 | 검증에서 고른 값 |
|---|---|---|
| 1 | 원본 동적 CRF의 H↔A 전환 점수에 경합 가점과 비경합 벌점 추가 | 강도 0으로 비활성 |
| 2 | 학습 5경기에서 킥·동일 팀 수신·상대 팀 수신·아웃 빈도를 측정하고 중심화한 −log 빈도로 조정 | 강도 1.0 |
| 3 | 볼 궤적의 꺾임·감속과 해당 팀 선수 근접이 함께 관측된 터치 후보로 경계 이동 | ±10프레임, ±0.4초 |
| 4 | 기본 모델 전환에 대해 작은 LightGBM으로 실제 전환 여부 분류 후 낮은 확률 경계 삭제 | 임계값 0으로 비활성 |
| 5 | 볼 움직임·팀 거리·경합·기본 모델 점수의 앞뒤 0.2초 맥락으로 유지·H→A·A→H 확률 학습 | 강도 1.0 |
| 6 | 학습 경기의 H/A 점유 길이 분포를 반마르코프 팀 점유 디코딩에 반영 | 강도 1.0 |

1번은 원본의 학습된 점수에 상대 보정을 더한다. 원본에 고정 벌점 하나가 있다고 가정하지 않았다. 2번은 빈도가 낮은 가로채기에 더 높은 비용을 줄 수 있어 가로채기 재현율 개선이 자동으로 따라오지는 않는다. 3번의 궤적 후보는 터치 정답이 아니다. 후보의 관측 근거와 선수 근접 조건을 사용한다. 5번은 동적 CRF 위에 LightGBM 확률을 보정값으로 더하는 구조다. 6번은 팀 단위 길이 사전을 쓰므로 선수 경로를 별도로 재학습하지 않는다.

학습한 작은 모델은 각각 150트리, 최대 깊이 4, 15 leaves, lr 0.04다. 전환 확률 모델은 75,899개 5fps 사례, 검증 모델은 1,420개 기본 예측 전환으로 학습했다. 팀 점유 정답과 터치 기반 경로 빈도 모두 학습 경기만 사용했다.

기존 25fps 신경망은 이 비교에서 사용하지 않았다. 볼은 현재 DFL 관측 좌표를 사용하고 합성 결측을 추가하지 않았다. 모든 보정은 인플레이와 하프 경계를 보존한다. 내부 점수 추출은 약 225초, 캐시 이후 작은 모델 학습·검증 탐색·테스트는 약 25초였다.

## 재현

로컬 실측 환경의 임시 패키지 경로를 사용하는 명령이다. 필요한 패키지는 torch, numpy, pandas, scipy, pyarrow, kloppy, lightgbm과 scikit-learn이다.

```sh
env PYTHONPATH=/private/tmp/transition5-correction-deps:/private/tmp/dfl-ha-torch:/private/tmp/mtah-dfl-runtime.cz3a8v:/private/tmp/pathcrf-runtime MPLCONFIGDIR=/private/tmp/pathcrf-mpl /Users/jsh/.cache/codex-runtimes/codex-primary-runtime/dependencies/python/bin/python3.12 'src/possession/pathcrf_refiner/correct_5fps.py' prepare
env PYTHONPATH=/private/tmp/transition5-correction-deps:/private/tmp/dfl-ha-torch:/private/tmp/mtah-dfl-runtime.cz3a8v:/private/tmp/pathcrf-runtime MPLCONFIGDIR=/private/tmp/pathcrf-mpl /Users/jsh/.cache/codex-runtimes/codex-primary-runtime/dependencies/python/bin/python3.12 'src/possession/pathcrf_refiner/correct_5fps.py' run
```

원본 점수 캐시는 Git에서 제외되는 data/transition_final/correct_5fps_v1/scores/에 저장한다. 결과표·원본 축 CSV·작은 모델·파라미터 사본은 artifacts/possession/pathcrf/correct_5fps_v1/에 있다. 이 runs 경로도 Git에서 제외했다. 두 모델 파일은 transition_probability.txt와 transition_verifier.txt다.

## 결과

| 조건 | Spell F1 | 가짜 전환 | 누락 전환 | 시점 MAE(ms) | 짧은 점유 재현율 |
|---|---:|---:|---:|---:|---:|
| base | 71.81% | 71 | 111 | 252.9 | 30.85% |
| correction_1 | 71.81% | 71 | 111 | 252.9 | 30.85% |
| correction_2 | 72.47% | 60 | 115 | 250.3 | 30.85% |
| correction_5 | 72.83% | 53 | 124 | 171.7 | 28.72% |
| correction_3 | 72.72% | 70 | 110 | 167.1 | 32.98% |
| correction_4 | 71.81% | 71 | 111 | 252.9 | 30.85% |
| correction_6 | 71.60% | 71 | 113 | 251.6 | 30.85% |
| combined | 72.49% | 52 | 126 | 142.4 | 29.79% |

검증에서 선택된 조건은 combined다. 단독 correction_5의 테스트 수치가 더 높았지만 최종 조건은 사전 검증 선택을 유지했다.
