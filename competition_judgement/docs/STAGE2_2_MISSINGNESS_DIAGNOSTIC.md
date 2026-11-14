# Stage A — Temporal Missingness Distribution & Feature Distortion Analysis

2026-11-09. **Stage A만 구현·검증·전체 로컬 실행했다. 기존 Full-window 정책은 유지한다.** Positive/Negative Evidence, v2-3 판단식, 신경망 학습, 점유 판단은 구현하거나 변경하지 않았다.

최종 출력: `data/derived/competition_judgement/missingness_stage_a_final_20261109/`.
초기 실행 폴더 `missingness_stage_a_20261109/`도 삭제·덮어쓰기 없이 보존했다. 초기 실행 뒤 공식 Full-only NaN 비율의 분모를 수정하고, 비대칭성 histogram과 같은 side 내부 위치 비교를 추가했다. 실험 seed·삭제 수·표본 선택·허용 기준을 결과에 맞춰 바꾸지 않았다. 아래 수치는 최종 출력 기준이다.

## 1. 핵심 결론과 해석 범위

* 공식 구간의 군집 관측 **745,891행 중 Full-reliable 191,868행, 25.72%**를 그대로 재현했다. Window 실패의 가장 흔한 개별 요인은 **전역 H/A 관측 인원 변화: 354,504행, 47.53%**다. 실패 offset에서는 현재/해당 시점 QC의 합집합이 가장 크다. 이 두 분모를 구분해야 한다.
* 관측 지원 부족은 균등하지 않다. observed 결측률은 중심 ±10에서 **0.279%**, 양끝 50개 offset에서 **11.739%**였다. 현재 존재하는 군집을 기준으로 Anchor를 만든 조건부 표본이므로, 이 차이를 Tracking 오류 위치 분포로 해석할 수 없다.
* **같은 연속 10개 결측을 과거 50프레임 안에서 이동**했을 때, 거리 기반 접근 특징의 고정 배치별 MAE는 **0.0462–0.1599m**였다. 각 Window에서 위치별 예측 범위의 평균은 **0.3335m**, P95는 **0.8200m**다. 같은 결측률이라도 위치는 중요하다.
* 중심 근처가 모든 특징의 최악 위치는 아니다. 접근·미래 평균은 끝부분 제거에도 민감하다. 반면 현재부터 이어지는 **지속시간 하한은 중심 가까운 단일 공백에도 크게 줄어든다.** 중심 보존·10개 결측에서 far-past 하한 손실 MAE는 0.481s, 중심 양옆 삭제는 3.312s였다. 후자의 두-run 배치와 side별 결측 수 차이도 함께 명시한다.
* Validation에서 reliable 기반 Full/P90/P80/P70/P60의 Coverage는 **21.78/26.66/31.13/36.22/42.42%**다. 이는 가용성 증가다. 실제 미관측 값의 정확도나 경합 탐지 개선을 입증하지 않는다.

**확인된 사실**은 mask/QC/관측 수의 분포, 고정 Anchor에서 합성 삭제로 생긴 특징 변화, 가상 정책의 Coverage다. **추론**은 특징별 지원 정책을 따로 검토할 필요가 있다는 것이다. 실제 경합·마킹 분류 정확도, Tracking 위치 오차, 관측되지 않은 선수의 이동, 사라진 정보의 정확한 복원은 이번 자료로 검증하지 못한다.

## 2. 기존 코드·데이터 감사

읽은 자료는 `STAGE2_2_DESIGN_AUDIT.md`, `STAGE2_2_TEMPORAL_RELATIONS_REPORT.md`, `STAGE2_1_LOCAL_CLUSTER_REPORT.md`, 실제 `temporal_source.py`, `temporal_features.py`, `temporal_robustness.py`와 기존 player-only 품질·cache/split 코드다. 기존 FrameStore 바이너리와 v2-1/v2-2 결과만 재사용했다. ATD/XML/영상/원시 텍스트를 다시 파싱하지 않았다.

입력:

| 입력 | 실제 경로·역할 |
|---|---|
| v2-1 Frame CSV | `data/derived/competition_judgement/local_v2_1_20261109/local_frame_features.csv`: frame/period/전역 인원/QC |
| v2-2 Frame CSV | `data/derived/competition_judgement/temporal_v2_2_final_20261109/frame_temporal_features.csv`: 최종 축 대조, 199,319행 |
| v2-2 Long CSV | 같은 폴더 `cluster_temporal_features.csv`: 789,005행의 원래 mask·현재 Anchor geometry·참조 특징 |
| 선수 cache | `data/derived/relational_temporal_v1_20261109/raw_cache/`: player-only, 199,319 × 46 × 2, metres |
| pair cache | v2-2 `pair_cache/`: kernel/3m/5m 배열, 199,319 × 24 × 22, read-only mmap |
| 분할 | `data/derived/relational_temporal_v1_20261109/split_manifest.json`: 기존 분할 그대로 |

경기장 크기는 기존 cache의 105.3 × 68m다. 25fps, 원본 내부 frame 0..199318을 보존한다. Track Slot과 cluster_id를 시간적 인물/군집 ID로 사용하지 않는다.

### 2.1 정확한 mask 정의

현재 군집 중심 c, 지름 D에서 고정 Anchor 반경 `A=max(6m,D+3m)`를 사용한다. offset은 −50..+50이다.

```
record[s]   = 같은 half + 같은 split + 연속 record run 내부의 지원
observed[s] = record[s] AND Anchor 안 H 관측>=1 AND A 관측>=1
same[s]     = 전역 nH[s]==nH[0] AND nA[s]==nA[0]
reliable[s] = observed[s] AND strict[s] AND strict[0] AND same[s]
Full        = all(reliable[-50:+50])
```

`strict[0]=False`이면 존재하는 모든 offset의 reliable이 False다. 이는 101회 독립 관측 실패가 아니라 **현재 품질 하나의 Window 전체 gate**다. 현재 QC와 해당 offset QC를 독립 원인처럼 더하지 않는다.

Strict QC는 팀별 유효 관측 8–11명, 동일 팀 0.2m 이내 중복 의심 없음, pitch 밖 좌표 없음, partial/invalid 좌표 없음이다. 완전히 비어 있는 비활성 Slot 자체를 모두 실패로 세지는 않는다. 계산용 좌표에서는 pitch 경계의 ±2m를 넘어선 관측과 비유한 좌표만 제외한다. 중복 의심은 합치거나 인물로 재식별하지 않는다. 인원이 같아도 같은 사람이 추적됐다는 뜻이 아니다.

`record=False`인 위치는 현재 row sentinel을 gather한 뒤 제외한다. 다른 half/split의 실제 값을 참조하지 않는다. 원래 mask는 little-bit-order 101비트/13바이트 hex다. CSV에서 문자열로 읽어 선행 0을 보존했다. **789,005행 모두**에서 support와 reliable 정의를 재계산해 정확히 일치함을 검사했다.

현재 cache에는 record ID gap이 **0개**다. 따라서 이번 `record=False`는 원시 파일 record 소실을 뜻하지 않는다. 공식 구간에서는 half/split 경계 밖 지원이고, period0의 43,114행은 분석 범위 밖이다.

### 2.2 Test 사용 제한

전체 745,891행의 mask 분포는 배치 출력의 기술 통계로만 집계했다. Test 특징값으로 정책을 비교하거나 선택하지 않았다. **가상 Full/Partial Coverage 비교는 Train/Validation만**, donor는 Train만, 특징 왜곡 실험은 Full-reliable Validation만 사용했다. GT·볼·Smart Tagging·영상 판단은 입력이나 선택 기준에 사용하지 않았다. 기존 테스트의 합성 fixture 검사를 제외한 새 분석은 player-only다.

## 3. Full Coverage 손실의 원인

분모는 군집-현재시점 Window 745,891개, offset 75,334,991개다. 동일 frame의 여러 Anchor/radius가 포함된다. 고유 frame 수와 같지 않다.

| 요인 | 한 번 이상 발생 Window | 전체 Window 비율 | 실패 offset | 전체 offset 비율 | 그 요인만 있는 Window |
|---|---:|---:|---:|---:|---:|
| 경계/record 지원 없음 | 2,476 | 0.33% | 61,536 | 0.082% | 1,375 |
| Anchor 내부 한 팀 이상 없음 — ambiguous | 226,313 | 30.34% | 4,741,064 | 6.29% | 95,528 |
| 현재 Strict QC 실패 | 180,466 | 24.19% | 18,213,643 | 24.18% | 0 |
| offset Strict QC 실패 | 322,543 | 43.24% | 18,218,515 | 24.18% | 25,688 |
| 현재 대비 전역 H/A 인원 변화 | 354,504 | 47.53% | 13,645,742 | 18.11% | 88,763 |

원인별 비율은 합산할 수 없다. 현재 QC가 실패하면 offset 0의 QC도 실패하므로 현재 QC 단독 Window가 0인 것은 구현상 당연하다. 위 다섯 factor 중 둘 이상이 발생한 Window는 **342,669개**, 실패 554,023개 중 **61.85%**다. 현재/offset QC를 하나로 합치면 여러 원인 domain이 겹친 Window는 294,357개, 실패 중 53.13%다. QC domain만 실패한 Window는 74,000개다.

현재/offset QC의 **offset 합집합은 21,286,725개**, 실패 offset 31,010,963개의 68.64%다. Window 빈도로는 인원 변화가 가장 크고, offset 빈도로는 QC가 크다. 둘을 섞어 단일 인과 원인이라고 단정하지 않는다.

관측은 가능하지만 신뢰 조건 때문에 제외된 offset은 **26,208,363개**, 그런 offset을 포함한 Window는 **439,291개**다. observed Full은 517,476개(69.38%), record Full은 743,415개(99.67%)다. 낮은 reliable Coverage가 대부분 record 자체 소실 때문에 발생했다는 해석은 맞지 않는다.

경계 지원 부족 2,476개는 half 경계 365개/8,257 offsets, split 경계 2,111개/53,279 offsets다. raw record gap 원인은 0개다. full 조건을 가상으로 한 factor만 제거했을 때의 통과 수는 경계 193,243, Anchor 부재 287,396, 현재 QC 191,868, offset QC 217,556, 인원 변화 280,631이다. **두 QC factor를 함께 제거**하면 265,868이다. 이는 논리적 ablation이며 정책 완화 권고나 오류 없는 관측 수가 아니다.

첫 실패 위치·가장 중심에 가까운 실패 위치의 histogram, 원인별 run/max-run/run-count histogram을 저장했다. 대부분 원인의 시간순 첫 실패 중앙값은 −50이다. Window 왼쪽에서 이미 실패 중인 **좌측 검열**이 많다는 뜻이며 실제 누락 시작 시점으로 읽으면 안 된다. 경계 원인의 첫 실패 중앙값은 +2였다. 원인별 run 중앙값/P95는 Anchor 부재 14/33, 현재 QC 101/101, offset QC 38/101, 인원 변화 27/48프레임이다.

## 4. 시간적 분포

### 4.1 다섯 구간의 지원

| 구간 | offsets | observed 비율 | reliable 비율 |
|---|---|---:|---:|
| Far Past | −50..−26 | 87.62% | 49.67% |
| Near Past | −25..−1 | 98.82% | 67.22% |
| Present | 0 | 100.00% | 75.81% |
| Near Future | +1..+25 | 98.91% | 67.27% |
| Far Future | +26..+50 | 88.90% | 50.50% |

관측 지원 부족은 중심보다 양끝에 많고, 과거 쪽이 조금 많다. 미래−과거 결측 개수의 Window당 평균은 observed −0.341프레임, reliable −0.221프레임이다. 합계뿐 아니라 −50..+50 비대칭성 histogram도 보존했다.

중심 ±10 대 양끝 50개 offset의 결측률은 observed **0.279/11.739%**, reliable **27.915/49.915%**다. 현재 Strict Window만 비교하면 reliable **4.908/33.929%**다. 현재 low-quality Window는 reliable 전체가 0이므로 시간 위치 차이가 사라진다.

**Anchor는 현재 군집에서 만들어졌고 현재 observed는 정의상 True**다. 시간이 멀어지며 한 팀이 고정 원 밖으로 실제 이동했을 수 있다. 이 생존/선택 조건을 무시하고 양끝의 높은 결측을 Tracking 성능 저하라고 부르지 않는다.

### 4.2 짧은 오류인가, 긴 구간인가

| mask | 결측 run 수 | run 중앙값/P95 | ≤5프레임 run 비중 | 결측 offset 중 ≤5 run 비중 | 최대 run |
|---|---:|---:|---:|---:|---:|
| observed | 310,410 | 14/33 | 18.30% | 3.53% | 50 |
| reliable | 759,093 | 28/101 | 11.19% | 0.78% | 101 |

짧은 단발성 지원 실패가 전체 결측량의 대부분이라는 증거는 없다. current Strict만 제한해도 reliable run 중앙값/P95는 21/47프레임이고 ≤5 run의 offset 비중은 1.89%다. low-quality의 길이101 run은 현재 QC gate의 결과다. 이 run을 4초간 선수 Tracking이 실제로 끊겼다고 해석하면 안 된다.

whole-window run과 다섯 구간 내부에서 잘린 run을 구분했다. 구간별 run 수·최대 연속 길이·run count 분포·원인 offset 분포는 JSON/CSV에 있다. 양 팀의 현재 대비 인원 변화뿐 아니라 인접한 유효 record 간 전역 인원 변화율도 별도로 저장했다.

### 4.3 radius·크기·품질

3m Full은 56,689/255,603=22.18%, 5m는 135,179/490,288=27.57%다. observed 양끝 결측은 3m 13.73%, 5m 10.70%; 중심은 0.380/0.226%다. 현재 Strict 565,425행 중 Full 191,868행, low 180,466행 중 Full 0행이다.

군집 크기별 Far Past/Far Future observed 비율은 N=2에서 85.67/86.93%, N=3에서 92.37/94.24%, N=4–5에서 97.25/98.12%, N≥6에서 99.41/99.47%였다. 큰 군집은 원래 지름에 따라 Anchor가 더 넓어지는 효과도 있다. 동일 공간 크기·위치·관측 구성으로 통제한 비교가 아니므로 크기가 신뢰도를 인과적으로 높였다고 해석하지 않는다.

## 5. 통제 실험 설계

### 5.1 기준 Window와 독립 단위

기존 Validation은 전반 66811..78730, 후반 166381..178288이다. 기존 Train/Test 범위는 변경하지 않았다. Validation Full-reliable는 24,452행이며, 각 split 시작부터 고정 500프레임(20초) 블록을 만들고 ±50 support 전체가 같은 블록 안에 있는 **19,561행**을 표본 pool로 사용했다.

블록×radius마다 최대32개를 seed2027로 추출했다. 최종 **1,778 Window: 3m 818, 5m 960, 30 support 블록**이다. feature 값이나 오차를 보고 표본을 고르지 않았다. 동일 cohort를 모든 위치·개수·연속성 조건에 사용했다. 이전 v2-2 robustness의 12×151 probe와는 다른, 이번의 Full-reliable Validation 진단 cohort다. 따라서 이전 robustness 오차와 직접 대조하지 않는다.

기존 pair cache를 read-only로 열고 동일 Anchor의 시계열을 재구성했다. 기준 10개 특징이 동결 v2-2 CSV와 허용오차 1e−6 내에서 모두 일치했다. 재구성 중 모든 기준 offset의 reliable=True를 다시 확인했다. 원래 좌표를 다른 입력으로 보간하거나 누락 선수의 실제 값을 추정하지 않았다.

### 5.2 결측 개수·위치

정확히 1/5/10/20/30개를 삭제한다. A/B/D/E는 각각25개 Far Past/Near Past/Near Future/Far Future 구간의 중앙에 연속 배치한다. C는 중심을 보존한 −15..−1,+1..+15의30개 후보 안 중앙 배치다. F는 100개 비중심 offset의 균등 간격, G는 같은 domain의 무작위 비복원 추출이다. H는 **동일 삭제 수·중심 보존을 충족하는 Train reliable-mask 패턴을 빈도 가중 추출**한다.

중심 삭제 실험에서는 같은 총 삭제 수 중 1개를 offset0에 사용하고 나머지만 해당 domain에 배치했다. **중심 상실은 Anchor 상실 실패**로 별도 집계했고 특징 오차·방향 비교에서 제외했다. 예를 들어 m=1의 중심 삭제 조건은 위치에 관계없이 같은 실패다. 외부에서 제공한 현재 Anchor를 알고 있다고 가정해 정상 점수를 만들지 않았다.

25개 구간에30개를 배치하는 조합은 비교 불가다. 중심 삭제도 나머지29개가25개를 초과하므로 비교 불가다. 수를 줄이지 않았다. 보조 center-past/future one-run 조건까지 포함해 총495조건 중12개 불가, **483개 계산**했다.

핵심 위치 통제는 추가한 **one-run sweep**이다. 현재를 보존한 과거50/미래50 각각에서 길이 m의 연속 run을 가능한 모든 위치에 이동했다. count·연속 run 길이·side가 같고 위치만 다르다. `same_side_location_sensitivity.csv`는 고정 배치 전체 표본 MAE의 최소/최대와, Window별 위치 범위 통계를 분리한다. Window별 best/worst는 사후 진단적 범위이며 실제 운용에서 선택 가능한 예측기가 아니다.

### 5.3 연속성

10개 삭제에 `[10]`, `[5,5]`, `[2,2,2,2,2]`, `[1]×10`을 past/future에서 비교했다. 총수·side·평균 offset(−25.5 또는 +25.5)을 통제했다. 삭제 위치 집합·분산·중심에서 가장 가까운 공백까지 동시에 동일하게 유지할 수는 없다. **이 비교는 run 수와 시간적 퍼짐이 함께 바뀌는 실험**이며 순수 연속성 효과라고 주장하지 않는다.

### 5.4 진단 estimator와 원래 계산식

S는 기존 `max exp(−d/3m)`, d는 최근접 거리, E3/E5는 원래 hard-edge 수다. V는 삭제 뒤 남은 offset, P/F는 과거/미래다. 관측 가능한 값만 평균하며 빈 support는 NaN이다.

| 특징 | 진단 estimator | 단위 |
|---|---|---|
| observed_approach | S0 − mean(S[P∩V]) | 무차원 |
| observed_future_change | mean(S[F∩V]) − S0 | 무차원 |
| observed_nearest_approach_m | mean(d[P∩V]) − d0 | m |
| observed_nearest_future_change_m | mean(d[F∩V]) − d0 | m |
| approach_focus_separation_score | clip(S0−μP,0,1) × clip(S0−μF,0,1) | 무차원 |
| stable_close_score | occupancy_r(V) × mean(S[V]) × (1−range(S[V])) | 무차원 |
| transient_excess_over_mean | clip(S0−mean(S[V]),0,1) | 무차원 |
| close_duration_lower_bound_s | 현재부터 연속 reliable/삭제 안 된 active sample 수 /25 | s, 하한 |
| observed_occupancy_3m/5m | mean(Er>0 on V) | 비율 |

세 Full-only 특징(approach-focus, stable, transient)은 원래 정책에서 한 번이라도 결측이면 **공식 값 NaN**이다. 표의 plug-in은 **진단 전용 별도 estimator**다. 공식 NaN 비율과 진단 estimator의 계산 가능 비율을 함께 기록했고 기존 v2-2의 공식 출력을 채우지 않았다.

지속시간은 관측 평균으로 복원하지 않는다. 원래 `contiguous_duration`의 **하한**을 그대로 호출한다. 추가로 unknown이 모두 active일 수도 있다고 놓고 첫 관측된 inactive까지의 **Window 내부 상한**을 만든다. 이는 참값 추정치가 아니라 구간이며, 기록 범위 밖 실제 관계 지속시간의 상한은 아니다. 기준 하한을 포함하는지와 구간 폭을 기록했다. 삭제 실험에서는 기준값 포함률100%였지만 실제 누락 값 복원 성공률을 뜻하지 않는다.

### 5.5 오차·통계

각 특징의 AE 평균/중앙값/P95, signed bias, RMSE, NaN, finite 수, 계산 가능 비율을 저장했다. Signed bias는 추정−기준이다. 중심 상실 비율과 중심 보존 조건부 오차를 분리하고 `controlled_feature_comparison.csv`에는 전체 cohort 분모도 명시했다.

방향 반전은 사전 deadband를 양쪽에 적용했다. S 변화는0.02, 거리 변화는0.1m를 넘는 기준 방향만 분모로 쓰고 추정치가 반대 deadband를 넘어야 flip이다. 추정치가 deadband 안으로 사라진 경우는 direction loss로 따로 집계한다. 0 부근 진동을 명확한 반전으로 세지 않는다.

작은 오차 진단 cutoffs는 무차원0.02/m0.1/s0.2, 큰 오차는 이 값의5배다. 이는 경합 판정이나 정확도 기준이 아니라 비교용 고정 engineering tolerance다. 특징별 물리적 중요도가 같다는 보증은 없다.

대응 위치 비교의 CI는 **500프레임 support 블록 단위 bootstrap 400회, seed2027**이다. Window·radius·같은 frame 군집·반복 실험을 독립 재표집하지 않는다. 블록끼리 사용한 원시 support가 겹치지 않음을 검증했다. 이웃 블록의 실제 경기 동작까지 통계적으로 독립이라는 보증은 없으며, 30블록·한 경기 내부의 탐색적 불확실성이다. 다른 경기로의 CI가 아니다.

## 6. 같은 결측률, 다른 위치의 결과

### 6.1 중심 보존·10개 삭제

모든 행은 같은1,778 Window다. 아래는 진단 estimator MAE다.

| 삭제 배치 | 접근 S | 최근접 접근 m | 미래 변화 S | 지속 하한 손실 s |
|---|---:|---:|---:|---:|
| Far Past 중앙 연속10 | 0.01346 | 0.10422 | 0 | 0.48115 |
| Near Past 중앙 연속10 | 0.01275 | 0.09925 | 0 | 1.32576 |
| 중심 양옆10, 현재 보존 | 0.00932 | 0.07073 | 0.00952 | 3.31186 |
| Near Future 중앙 연속10 | 0 | 0 | 0.01442 | 1.40344 |
| Far Future 중앙 연속10 | 0 | 0 | 0.01364 | 0.52553 |
| 균등 분산10 | 0.00029 | 0.00265 | 0.00027 | 2.95822 |
| 무작위 분산10 | 0.00261 | 0.02046 | 0.00262 | 2.71640 |
| Train 실제 구조, 정확히10 | 0.00909 | 0.07515 | 0.00956 | 0.39591 |
| 중심 직전 −10..−1 | 0.01876 | 0.14348 | 0 | 1.63453 |

미래 삭제가 과거 평균 접근에 영향을 주지 않는0은 정의상 결과다. 중심 양옆10은 두 side에 나뉘어 두 run을 형성하므로 Far Past10과 순수 위치-only 비교가 아니다. **그 통제를 충족하는 결과는 다음 one-run 실험이다.**

같은 과거 side의 연속10 run을41개 위치에 옮기면 고정 배치별 MAE는 접근 S **0.00660–0.01941**, 최근접 접근 **0.04621–0.15987m**다. 가장 큰 고정 배치 오차는 −50..−41이었다. 미래 변화 S는 미래 side에서 **0.00665–0.02057**였다. Window별 위치에 따른 예측 범위 평균/P95는 접근 S **0.04266/0.09661**, 최근접 접근 **0.33345/0.81998m**, 미래 S **0.04493/0.11103**이다. 같은 결측 수에서 위치만 달라도 평균 변화가 달라진다.

특정 대응 비교인 중심 직전 −10..−1 대 Far Past 중앙 run은 최근접 접근 MAE가 **+0.03927m**, block-bootstrap95% CI **[0.03205,0.04594]m** 커졌다. 접근 S는 +0.00530 [0.00443,0.00610]. 반면 stable score는 −0.00403 [−0.00494,−0.00299]만큼 **작아졌다**. 중심 결측이 모든 특징에 더 나쁘다는 가설은 지지하지 않는다.

삭제 수30의 실제-pattern 조건에서는 명확한 sign flip이 최근접 접근 **3.09%**, 미래 S **2.54%**, 접근 S **2.06%**였다. m10에서는 대부분0에 가깝지만 실제-pattern 미래 S는0.20%였다. 낮은 flip만 보고 안정적이라고 할 수 없으며, deadband 안으로 방향이 사라지는 비율과 연속 오차를 함께 봐야 한다.

### 6.2 열 가지 특징 모두의 오차 예시

| 특징 | Far Past10 MAE | 균등10 MAE | Train pattern10 MAE |
|---|---:|---:|---:|
| observed_approach | .013455 | .000290 | .009093 |
| observed_future_change | 0 | .000270 | .009557 |
| observed_nearest_approach_m | .104218 | .002652 | .075148 |
| observed_nearest_future_change_m | 0 | .002204 | .073489 |
| approach_focus_separation_score, 진단 | .000642 | .000016 | .000889 |
| stable_close_score, 진단 | .010172 | .000507 | .016857 |
| transient_excess_over_mean, 진단 | .006569 | .000115 | .006959 |
| close_duration_lower_bound_s | .481147 | 2.958223 | .395906 |
| observed_occupancy_3m | .026966 | .002187 | .028096 |
| observed_occupancy_5m | .011201 | .001091 | .013115 |

모든 고정 삭제 조건의 진단값은 중심을 보존하면 계산 가능했다. **공식 Full-only 3개 값은100% NaN**이다. 중심 삭제 조건에서는10개 특징 모두 Anchor 상실로 점수 비교하지 않았다. 실제-pattern 무조건 전이에서는 한 side 전체 소실로 일부 평균 특징도 NaN이 됐다.

### 6.3 지속시간과 연속성

과거 side의10개 삭제, 평균 offset−25.5에서 run 수1/2/5/10의 접근 S MAE는 **.00660/.00328/.00218/.00238**, 지속 하한 손실은 **.901/1.143/1.326/1.363초**였다. 미래 side의 지속 하한 손실도 **.925/1.178/1.365/1.403초**로 증가했다.

여러 짧은 공백이 평균을 잘 근사해도 중심에 가까운 첫 공백이 연속 지속 하한을 끊는다. **긴 run 자체보다 가장 가까운 공백과 연속 지원 길이가 더 직접적인 요인**이다. 동일한 연속10개만의 위치 sweep에서도 과거 하한 손실은 고정 배치에 따라 **.272–1.635초**였다. 단일프레임 결측도 현재 바로 옆이면 긴 연속 관계의 하한을 잃을 수 있다. 이 하한 감소는 잘못된 분리 확정이 아니라 정보 보류다.

### 6.4 3m/5m 차이

Far Past10의 접근 S MAE는3m .01463/5m .01245, 거리 접근은 .10076/.10717m, 지속 하한 손실은 .40357/.54725초였다. 5m가 observed Coverage가 높다고 모든 특징의 왜곡도 작지는 않다. 모든 조건에서 radius별 같은 기준 cohort를 유지했고 `size=2/3+`도 별도 저장했다. 두 radius 표본은 공간·크기 분포가 같지 않으므로 radius의 순수 인과 효과는 아니다.

## 7. 실제 실패 패턴을 반영한 전이

Train 관측은3m171,414행,5m335,901행이다. 각 radius의 record/observed/reliable 전체101비트 실패 패턴을 빈도와 함께 보존했다. 이 표현은 run 길이, 과거/미래 비대칭, 중심 실패, 여러 구간 동시 결측을 **동시에 그대로 유지**한다. 원인별 위치·중복 분석은 원래 Train mask 감사 통계로 분리했다. 임의 run-length만 새로 조합해 원래 시간적 공존 구조를 없애지 않았다.

고정 삭제 수 H는 Train에서 같은 수의 결측·중심 보존이라는 **사전 조건만** 적용했다. 별도의 무조건 전이에서는 유리한 패턴을 고르지 않고 전체 Train 패턴 빈도로3회 추출했다. clean 패턴과 current-QC gate로101개 모두 제외되는 패턴도 포함했다. Validation 오차를 보고 donor를 제거하지 않았다.

reliable donor를 무조건 전이하면 중심 상실은3회 각각 **400/388/436개(22.50/21.82/24.52%)**였다. 이를 제외한 조건부 접근 S MAE는 **.02113/.01973/.01923**, 최근접 접근은 **.16652/.14828/.15045m**, stable score는 **.04275/.03794/.03827**, 지속 하한 손실은 **.76955/.75062/.76212초**였다. 부분적으로 한 side가 전부 사라진 경우까지 포함한 원래 cohort 기준 계산 가능 비율은 별도 CSV에 있다.

observed donor는 현재 군집 조건상 중심을 삭제하지 않았다. 접근 S MAE **.00685–.00768**, 최근접 접근 **.05373–.05884m**, 지속 하한 손실 **.209–.223초**였다. record donor는 Train의 record 지원 부족 자체가 드물어 대부분 clean이었고 평균 오차가 거의0이었다. 이는 경계 부근 삭제가 안전하다는 증거가 아니라 **그 donor 분포에서 삭제 노출이 드물었다**는 뜻이다.

reliable rep0에서 양쪽 최소 support가90–100% 미만인74개 Window도 지속 하한 손실 MAE .305s/P95 1.724s였다.80–90%의84개는 .378s/P95 1.416s였다. 높은 지원율만으로 지속시간을 보장하지 못한다.

실제 결측은 비무작위일 수 있다. 원래 full-reliable인 다른 장면에 패턴을 옮기면 움직임과 결측 사이의 상관을 보존할 수 없다. 이 실험은 **관측된 형태를 가진 삭제의 민감도**이며 실제 미관측 정보의 오류량 추정이 아니다. low-quality 좌표 자체의 잡음·팀 오분류·슬롯 교체는 재현하지 못했다.

## 8. Partial 정책: 채택하지 않은 진단 비교

양쪽 각각 `ceil(50 × fraction)`개 지원과 현재 유효를 요구한다. 90/80/70/60%는 각각 양쪽45/40/35/30개다. 전체101개 평균 지원율 하나로 대체하지 않았다.

### 8.1 실제 Validation Coverage

분모112,286개 Validation 군집 Window, 현재 Strict/low 모두 포함한다.

| 정책 | reliable 기준 통과 수 | Coverage | Full 대비 증가 pp | observed 기준 Coverage |
|---|---:|---:|---:|---:|
| Full | 24,452 | 21.78% | 0 | 71.09% |
| P90 | 29,935 | 26.66% | 4.88 | 75.86% |
| P80 | 34,958 | 31.13% | 9.36 | 80.83% |
| P70 | 40,665 | 36.22% | 14.44 | 86.10% |
| P60 | 47,631 | 42.42% | 20.64 | 91.05% |

observed 기반 열은 QC/인원 조건을 무시한 **수학적 계산 가능성 상한 성격의 비교**다. 동일한 신뢰도가 아니다. reliable 기준은 현재 low-quality를 항상 보류하므로 어떤 Partial에서도 그 그룹의 Coverage는0이다. 반대로 observed는 low-quality도 계산할 수 있으나 정확도 검증 자료가 없다.

radius별 Strict Validation reliable Full→P80은3m **6,506→10,039/30,584**,5m **17,946→24,919/58,008**이다. 실제 정책 표에는 Train/Validation × radius × Strict/low × observed/reliable를 모두 분리했다. Test 정책 비교는 하지 않았다.

P80에 조건을 각각 독립적으로 하나씩 추가하면 중심±10 전부 지원 **34,652개(30.86%)**, 최대 연속 공백≤1 **25,875(23.04%)**,≤5 **29,987(26.71%)**,≤10/25 **34,958(31.13%)**, 전역 인원 변화 없음 **31,394(27.96%)**다. 현재 Strict와 현재 Anchor 양 팀 관측 조건은 reliable 정의에 이미 들어 있어 추가 감소가 없다. gate들은 따로 비교했으며 결합한 새 정책을 채택하지 않았다.

### 8.2 Coverage와 오차는 별개

실제 실패 Window의 clean 진실은 없으므로 실제 Coverage 표에 실제 복원 오차를 붙일 수 없다. 합성 삭제에서는 같은 기준값을 알고 있으므로 정책을 적용해 왜곡을 계산했다.

중심 보존10개 삭제는 어느 쪽에 몰아도 P80을 통과하고 Full은 모두 보류한다. 균등10의 접근 S/거리 오차는 고정 작은 tolerance 안에 **100%** 들어갔다. 그러나 지속 하한은 작은 오차2.0e−1s 안에 **0.28%**만 들어갔고, ≥1s 손실은 **96.85%**였다. 중심 양옆10도 P80 Coverage100%지만 지속 하한 ≥1s 손실이 **98.88%**였다.

Far Past10의 Full 보류 중 작은 오차 비율은 접근 S **76.66%**, 거리 접근 **60.01%**, 지속 하한 **25.76%**다. 같은 P80에서도 중심 직전10의 거리 접근 작은 오차는 **45.56%**, 큰 오차(≥.5m)는 **1.46%**였다. 이 비율은 합성 조건의 비율이며 실제 보류 Window 전체 중 안전하게 구제 가능한 비율이 아니다.

### 8.3 다음 검증에 대한 제안 — 정책 미채택

* 접근/미래 평균·occupancy: 각 side 지원 수와 위치·빈 구간 폭·signed bias를 함께 보는 특징별 지원 실험의 근거가 있다. 균등 분산에서는 비교적 안정적이었지만 한쪽 끝의 장기 결측과 실제-pattern에서는 오차가 커졌다. 공통P80을 안전하다고 선언하지 않는다.
* stable score: min/max/range 누락에 민감하고 무조건 reliable-pattern 전이에서 양의 bias가 .032–.037이었다. 평균만의 지원율과 별도로 extrema 지원을 검증해야 한다.
* approach-focus/transient: 절대 오차가 작은 사례가 있어도 값 자체가0에 몰릴 수 있다. 상대적 해석과 극값 소실을 검증하기 전에는 공식 Full 조건을 유지하는 쪽이 보수적이다.
* 지속/최초 형성/재접근: 단순 Partial 평균을 시간 길이로 읽지 않는다. 현행 Full/연속 구간의 하한·검열 semantics를 유지하고, gap 위치와 bounds를 전달하는 인터페이스를 추가 검증하는 것이 타당하다. 하한은 Partial에서 기록 가능하지만 원래 전체 지속값과 같다고 주장할 수 없다.
* current Strict와 전역 인원 동일 조건: 서로 다른 gate다. 현재 QC를 완화하면 Window 전체 mask가 바뀌므로 단순 지원 비율 조정과 분리해 연구해야 한다. 동일 인원도 동일 인물을 보장하지 않는다. 이번 실험은 QC 오류가 있는 좌표의 신뢰도 완화 근거를 제공하지 않는다.

## 9. 구현·입출력·검증

기존 파일을 수정하지 않고 다음만 새로 만들었다.

| Python 모듈 | 역할·주요 Shape |
|---|---|
| `missingness_diagnostics.py` | 기존 mask decode `[N,101]`, factor code `[N,101]`, 전체 스트림 집계 |
| `missingness_cache.py` | read-only pair cache 재사용, 고정 Anchor 시계열 `[N,101]` |
| `controlled_temporal_masking.py` | 정확한 삭제 mask `[N,101]`, 진단·공식 adapter별10개 `[N]` 특징, 오차 |
| `partial_window_diagnostics.py` | RLE, 양쪽별 가상 지원 predicate `[N]` |
| `missingness_experiments.py` | Train donor, 동일 Validation cohort, 삭제 실험·대응 block CI |
| `run_missingness_stage_a.py` | 신규 폴더 전용 CLI·기록·원본 CSV 대조 |
| `summarize_missingness_stage_a.py` | 원인/구간/run CSV, 같은 side 위치 통제·전체 cohort 분모 |
| `verify_missingness_stage_a.py` | 원본 축·참조 support·유한성·시작 시점 보존 검사 |

새 설정은 `competition_judgement/configs/missingness_stage_a.json`, 합성 검사는 `competition_judgement/tests/test_missingness_stage_a.py`다. 기존 소스·문서·설정의 정책을 수정하지 않았다.

시작 시 기존233개 테스트는 모두 통과했다. 최종은 **377 passed in31.14s: 기존233 + 그 외 root tests119 + 신규25**, 실패0이다. 초기 합성 fixture에서 Pandas3의 read-only 배열을 변경하려던 테스트1개가 실패했으며 fixture에서 명시적으로 copy한 뒤 해결했다. 알고리즘의 품질 기준을 고친 것이 아니다.

새 검사에는 정확한 삭제 수/중심 실패,25개 구간 초과 불가,run 길이·평균 위치 통제, 기존 공식 함수10개 결과 일치, 숨긴 실제 값 변경 시 진단 출력 불변, 지속 bounds·현재 inactive, side별 support, hex 선행0,half/split/gap sentinel,QC gate 원인,NaN·deadband·Train donor·block bootstrap이 포함된다.

실제 실행에서도 원본 Frame CSV199,319행과 새 Frame CSV 축, Long789,005행, official745,891행, Full191,868행,1,778개 참조의 원래10특징 일치, Validation 경계·20초블록 support 비중복, 비교 CSV의 Inf 없음 등을 검사했다. 기존 **소스·문서198개 SHA256**, **원본/기존 산출물8,301개 size/mtime**가 모두 시작 시점과 같다. 체크포인트·기존 baseline·기존 경기 파생 결과를 보존했다. 검증 기록은 `verification.json`, 시작 snapshot은 `preservation_start.json`이다.

환경은 Conda mtah2027/Python3.11.17/NumPy2.4.6/Pandas3.1.0, macOS arm64 CPU다. 최종 mask 감사+기준 시계열+실험 실행은 **60.66초**, 이후 요약/추가 same-side 집계·테스트·보존 검증 시간은 별도다. peak RAM은 측정하지 않았다. 외부 전송·서버 실행·Git Commit/Push는 하지 않았다. 새 데이터 출력은 기존 Git ignore에 포함된다.

## 10. 산출물과 재현

모든 아래 파일은 최종 로컬 폴더 안에 있다.

* `stage_a_findings.json`: 가장 먼저 읽을 핵심 수치 요약. `summary.json`, `experiment_summary.json`, `missingness_summary.json`: 실행·원인·분포·donor 전체 집계.
* `cause_summary.csv`, `cause_overlap.csv`: 원인별 분자/분모·단독/중복·첫 실패·RLE.
* `temporal_region_summary.csv`, `missing_run_summary.csv`: 다섯 구간/품질/radius/크기별 지원과 연속성.
* `frame_missingness_summary.csv`: **0..199318 전체199,319행**. official 군집 없는 frame은 군집 수0, 평균 지원은 NaN이다. 점유 label은 없다.
* `controlled_conditions.csv`: 모든 조건·비교 불가 사유·중심 상실·지속 bounds.
* `controlled_feature_comparison.csv`: 위치/연속성/실제패턴 전이의 특징별 AE·bias·RMSE·방향·NaN·전체 및 조건부 계산 비율. 기초 집계는 `controlled_feature_errors.csv`.
* `same_side_location_sensitivity.csv`: **같은 삭제 수·같은 side·같은 run 길이에서 위치만 다른 핵심 결과**.
* `one_run_location_sensitivity.csv`: 양쪽을 합친 Window별 best/worst 범위. 반대쪽 삭제가 접근에0영향인 자명한 경우도 있으므로 same-side 표를 우선 읽는다.
* `paired_block_comparisons.csv`: 대응 오차 차이와20초 block CI.
* `actual_policy_coverage.csv`, `synthetic_policy_comparison.csv`: 실제 Train/Validation Coverage와 합성 조건부 오차를 구분.
* `empirical_support_errors.csv`: 양쪽 최소 지원율 구간별 실제-pattern 전이 오차.
* `train_region_cooccurrence.csv`: Train의 다섯 시간 구간 결측 동시 발생32개 조합. reliable에서는 모든 구간 실패112,973행, 양끝 두 구간만 실패33,597행이었다. 전체101비트 패턴과 함께 해석한다.
* `reference_keys.csv`, `reference_temporal_arrays.npz`, `train_pattern_library.npz`: 재현용 로컬 전용 자료. 외부 업로드 대상이 아니다.
* `preregistered_diagnostic_config.json`, `verification.json`, `preservation_start.json`, `test_results.json`: 설정·출처·검증 기록.

프로젝트 루트에서 기존 폴더와 **다른 새 출력 이름**으로 실행한다.

```bash
conda activate mtah2027
export PYTHONPATH="$PWD/competition_judgement/src:$PWD/MTAH_Relational_Starter:$PWD"
export PYTHONDONTWRITEBYTECODE=1
python -m pytest -q -p no:cacheprovider tests MTAH_Relational_Starter/tests competition_judgement/tests
python -m competition_judgement.run_missingness_stage_a \
  --output data/derived/competition_judgement/missingness_stage_a_reproduction
python -m competition_judgement.summarize_missingness_stage_a \
  --run-dir data/derived/competition_judgement/missingness_stage_a_reproduction
python -m competition_judgement.verify_missingness_stage_a \
  --run-dir data/derived/competition_judgement/missingness_stage_a_reproduction
```

이번 실행의 보존 검사에는 실행 전 생성한 snapshot을 `--preservation-snapshot /private/tmp/cj_stage_a_start.json`으로 전달했다. 재현 시 보존 비교가 필요하면 그 재현 실행 **이전** snapshot을 전달한다. snapshot 없이 실행한 verifier는 정합성만 검사하며 이전 입력 보존을 새로 입증하지 않는다. source hash와 설정은 산출물에 남는다. CLI는 기존 출력 폴더 덮어쓰기를 거부한다.

## 11. 반드시 답해야 하는 질문

**Q1. 25.72% Coverage 손실의 가장 큰 원인은?** Window 빈도 기준 전역 H/A 관측 수 변화47.53%가 가장 흔하다. 실패 offset의68.64%에는 현재/offset QC 중 하나가 걸린다. 현재 low-quality24.19%는101개 전체를 차단한다. 경계 지원 부족은0.33%로 작다. Anchor 한 팀 부재30.34%는 이동/누락 ambiguous다. 원인 겹침이 커 단일 원인 점유율처럼 더하지 않는다.

**Q2. 같은 결측률에서 위치 영향은 얼마나 큰가?** 같은 연속10개·같은 past side에서도 거리 접근 MAE .0462–.1599m, Window별 위치에 따른 예측 범위 평균 .3335m/P95 .8200m였다. 대응 center-past 대 far-past 차이는 .0393m [95% block CI .0320,.0459]다. 위치 정보 없이 지원율 하나로 오차를 표현하기 어렵다.

**Q3. 중심과 양끝 영향은 다른가?** 다르다. 지속 하한은 중심 바로 옆 공백에 크게 줄고 평균·range 특징은 양끝의 비대표적 제거도 중요하다. 중심이 모든 특징의 최악 위치라는 결론은 아니다. 현재 자체가 사라지면 Anchor 실패로 점수 비교를 보류한다.

**Q4. 개수보다 연속성이 중요한 특징은?** 지속시간·onset/재접근 해석은 현재부터 이어지는 연속 지원과 첫 공백에 직접 의존한다. 여러 단발성 공백도 평균을 거의 바꾸지 않으면서 지속 하한을 크게 낮춘다. run 길이·중심 거리·분산을 완전히 분리하지 못한 연속성 대조의 한계도 남는다.

**Q5. Partial에서 비교적 안정적인 특징은?** 이 Full-reliable Validation cohort의 균등/짧은 분산 삭제에서는 observed 평균 접근/미래 변화·occupancy가 비교적 안정적이었다. 실제-pattern·끝부분 집중·긴 결측에서는 오차와 방향 소실이 늘었다. 무조건적인 안정성이나 low-quality 좌표의 신뢰성을 확인한 것은 아니다.

**Q6. Full 유지가 더 타당한 특징은?** 전체-window의 extrema/range를 포함하는 안정성·피크·복합패턴을 공식 점수로 해석할 때는 추가 검증 전 Full을 유지한다. 지속시간은 Full 또는 알려진 연속 구간의 하한/검열 표기로 유지하며 Partial 평균으로 전체 길이를 추정하지 않는다. 모든 특징에 동일한 Partial 기준을 확정할 근거는 없다.

**Q7. Stage B 전에 무엇을 검증해야 하는가?** 특징별 양쪽 support·위치·긴 공백·중심연속성·extrema 지원, 현재 QC와 offset QC의 역할 분리, 인원 변화의 양과 시점, 관측 부재의 ambiguous 표기를 사전 고정 조건으로 추가 검증해야 한다. 낮은 QC의 좌표 오류는 이번 완전관측 삭제 실험으로 검증되지 않았고, 실제 경합 정답도 없다. 이번 결과만으로 정책을 채택하거나 Stage B 증거 판단식을 만들지 않았다.
