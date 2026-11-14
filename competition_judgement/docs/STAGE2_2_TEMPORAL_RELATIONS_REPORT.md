# v2-2 Bidirectional Temporal Relation Analysis 실행 보고서

2026-11-09. **Spatial Anchor를 기본으로 하는 오프라인 양방향 시간 특징**을 구현하고 합성 검증·전체 로컬 실행·Validation 교란 검사를 완료했다. 현재 관계의 공간과 시간 양상을 기술하며 실제 경합·마킹·점유의 정답은 생성하지 않는다. v1/v2-1 코드·설정·결과, 원본·A/B·제출 Baseline을 보존했다. 최종 후보 규칙·threshold·H/A/D/X는 바꾸지 않았다.

## 1. 주요 결론

같은 현재 배치라도 과거 접근·미래 유지/이탈에 따라 다른 특징이 나온다. 순서를 뒤집어 Window 평균을 같게 해도 접근/미래 변화는 달라져 단순 이동평균이 잃는 정보를 표현한다. 이는 합성에서 확인한 **움직임 양상 구분**이며 실제 경기의 일반 마킹과 볼 경합을 분류한 성능이 아니다.

관측 부족을 실제 분리로 확정하지 않도록 보수적으로 처리했다. 공식 군집 관측745,891행 중191,868행(25.72%)만 전체101개 reliable 관측 조건을 만족했다. frame으로 세면3m33,129개,5m49,835개에서 적어도 하나의 valid pattern anchor가 있었고, 두 radius 중 하나라도 있는 frame은49,835/158,857(31.37%)였다. 군집 행과 고유 frame의 coverage를 혼동하지 않는다.

독립적인 프레임별 dropout10/20/30%와 단기 composition 제거에서는 full-quality 패턴 지원이0이 됐다. 이를 낮은 오차나 강건성 성공으로 해석하지 않는다. 가짜 신호를 보류하는 대신 강한 coverage 손실이 생겼으며, 기준을 결과에 맞춰 완화하지 않았다.

## 2. 기존 자료·구현 감사와 새 파일

[사전 감사](STAGE2_2_DESIGN_AUDIT.md)에 v2-1 설계/실행 보고서, Stage1, FrameStore/QA API와 실행 전 설정을 기록했다. 기존 Frame CSV199,319행/Long CSV789,005행의 SHA·키를 확인했다. 실제 Python은 `competition_judgement/src/competition_judgement/`, FrameStore는 `MTAH_Relational_Starter/mtah_relational/frame_store.py`에 있다. 원시 parser와 인물 ID 연결은 새로 만들지 않았다.

| 신규 파일 | 역할 |
|---|---|
| [temporal_source.py](../src/competition_judgement/temporal_source.py) | read-only FrameStore와 v2 QA를 사용한 frame-local 거리/kernel 캐시, 경계 support |
| [temporal_features.py](../src/competition_judgement/temporal_features.py) | 고정 Anchor 시계열·양방향 변화·지속 하한·검열·품질 마스크 |
| [temporal_matching.py](../src/competition_judgement/temporal_matching.py) | reciprocal-unique인 인접 공간 대응만 기록, 병합/분리 ambiguity 보존 |
| [analyze_temporal_relations.py](../src/competition_judgement/analyze_temporal_relations.py) | 전체 추출·Frame/Long CSV·집계·원본 보존 CLI |
| [temporal_robustness.py](../src/competition_judgement/temporal_robustness.py) | 기존 Validation support/seed의 고정 probe 비교·redetection matching 진단 |
| [test_temporal_relations.py](../tests/test_temporal_relations.py) | 신규 합성20개 검사 |
| [temporal_relations_v2_2.json](../configs/temporal_relations_v2_2.json) | Window·물리 범위·matching·품질 지원 설정, fitting 없음 |

기존 README/manifest/설정/소스는 수정하지 않았다. v2-1 feature 정의를 그대로 사용하고 새 코드와 문서만 추가했다. 기존 `PACKAGE_MANIFEST.json`은 v1 스냅샷이며 v2-2 목록은 본 보고서와 로컬 실행 source hash를 따른다.

## 3. Window와 경계

기본 R=50, offsets −50..+50, 과거50·현재1·미래50의101개 관측이다. 과거/미래 각각2초이며 처음/마지막 sample 시각의 차는4초다. config의 `window_radius_frames`로25/75도 지정할 수 있고 합성에서 길이·정렬을 검사했다. 실제 경기의25/75 비교 실험은 이번에 실행하지 않았다.

참조 가능한 run은 **같은 official half + 같은 Train/Validation/Test block + 연속 record ID**다. period0은 분석 범위 밖이며 시간 support를0으로 유지한다. source/cache를 지우거나 과거 반/다른 split의 관측으로 채우지 않는다. raw record gap도 run을 끊는다. 전체 CLI는 기존 intact199,319 cache를 요구하고 손상 record를 보간해 길이를 맞추지 않는다. sparse FrameStore의 gap 처리는 합성으로 검증했다.

가용 support가 없는 array 위치는 다른 블록이 아닌 현재 row의 sentinel로 gather한 뒤 False mask로 완전히 제외한다. 보간·forward fill·고정0 관측 생성은 하지 않는다. frame별 실제 past/future 지원 수와 bitmask를 기록한다.

**미래를 사용하는 오프라인 특징이다.** 실제 streaming 점유 판단에 그대로 사용할 수 없고, offline v1(현재+과거1초)과 동일한 정보 예산의 비교도 아니다.

## 4. Spatial Anchor: 기본 방법

현재 군집 중심c와 지름D에서 `A=max(6m,D+3m)`인 고정 원을 둔다. D는 anchor 영역 크기 계산용이고3m/5m contact radius와 다르다. 이 원은 현재 component를 포함하지만 큰 사슬에서는 넓은 영역이 되어 다른 관계도 포함할 수 있다. motion compensation이나 영구 ID는 없다.

각 s=t−R..t+R에서 QA 후 원 안 관측의 H/A 집합을 다시 선택한다. 위치가 원 밖으로 나간 경우와 tracking에서 빠진 경우를 실제 동일 선수 추적으로 구분할 수 없으므로 둘 다 관측 ambiguity가 될 수 있다.

H-A 거리 d에 대해 기존 inclusive edge `d≤3m`, `d≤5m`를 유지한다. 보조 연속 kernel은 `K(d)=exp(−d/3m)`다. frame-local pair cache는 hard 비교를 float64 거리에서 수행하고 kernel만 float32로 저장한다. 새 시간 특징에 clean target·볼·태깅을 넣지 않는다.

두 팀이 원 안에 관측될 때:

* S_s=max K(d_HA): 해당 공간의 가장 가까운 상대 쌍에 대한 soft proximity, [0,1]. 원래 v2-1 contact_strength와 다른 정의다.
* M_s=mean K(d_HA): 원 안 모든 H×A 조합의 평균 kernel, [0,1].
* dmin_s=−3log(S_s), m: float32 kernel에서 복원하므로 작은 반올림 오차가 있다.
* E_r(s)=count(d≤r): 원 안 cross-edge 수.
* C_r(s)=edge에 참여하는 서로 다른 현재 관측 수: contact size이며 최대 component 크기와 같지 않다.
* 원 안에 중심이 들어오는 v2-1 component 수/최대 N: `nearby_cluster_*`. 큰 component의 모든 멤버가 원 안에 있다는 뜻이나 동일 집단의 성장 추적은 아니다.

원 경계의 관측 선택은 hard하므로 kernel이 모든 불연속을 없애지는 않는다. 최근접 쌍이 바뀌어도 S가 이어질 수 있다. 이는 **공간의 관계 변화**이지 같은 두 선수의 마킹 지속이나 볼 경합 추적이 아니다.

## 5. 관측·품질·지원 조건

각 offset의 mask를 세 층으로 구분한다.

| mask/flag | 조건·의미 |
|---|---|
| record | 같은 run 안 실제 record가 있음 |
| observed | record이고 원 안에 양 팀 관측이 있음; 수학적으로 관계 계산 가능 |
| reliable | observed + 현재/해당 시점 strict QC + 현재와 같은 전역 H/A 관측 수 |
| pattern_score_valid | 전체101개 reliable=True |
| partial_bidirectional_available | 양쪽 reliable sample≥ceil(0.8R). 지원 표시에만 사용하며 기본 pattern 점수를 완화하지 않음 |
| global_count_change | 존재하는 record 중 현재와 다른 전역 H/A 관측 수가 있었음 |
| local_count_change | observed 시점 사이 원 안 H/A 수가 변했음. observed0이면 NaN |

strict는 기존 팀별8–11명·duplicate/bounds/partial/invalid 기준이며 확률적 confidence가 아니다. global count가 같아도 누락/교체·팀 오분류가 없다는 보증은 아니다. local count 변화는 실제 유입/유출과 관측 구성을 함께 반영하므로 기록만 하고 원인을 확정하지 않는다.

`record/observed/reliable_mask_hex`는 offset−R부터 순서대로 little-bit-order pack한 mask다. 기본101비트/13바이트/26 hex 문자이며 마지막 padding bits는0이다. 이를 관측 값이나 모델 feature로 채우지 않는다. 지원 수만 같고 유효 offset 위치가 다른 경우도 있으므로 noise 비교에서는 mask 자체의 일치를 확인했다.

observed 기술 통계는 실제 정의된 sample만 집계하고 표본 수·mask를 보존한다. 품질이 낮아도 기술 통계가 계산될 수 있다. **기본 reliable 변화/패턴 점수는 full-window 조건 밖에서 NaN**이다. 0은 계산된 변화/강도가0이라는 뜻이지 관측 부족 대체값이 아니다.

## 6. 시간 특징의 정의

P는 offsets−R..−1, F는1..R, W는−R..R이다. μ_Q는 Q의 observed-valid 값만의 평균이며0개면NaN이다. S0은 현재 proximity다. 별도 reliable pattern 조건은 아래에 명시한다.

| 출력 또는 공통 이름 | 계산식·단위·관측 조건 |
|---|---|
| record/observed/reliable_past/future_frames | 각각 P/F의 True 수, frame |
| full_record_window | 모든 W record mask True |
| window_observed_fraction | observed 개수/(2R+1) |
| window_strict_fraction | record AND 해당 시점 strict 개수/(2R+1) |
| present_anchor_soft / soft_pair_mean | 현재 S0/M0, [0,1]; 현재 observed 필요 |
| observed_past/future/window_soft_mean | μ_P(S), μ_F(S), μ_W(S) |
| observed_window_soft_std/max/range | observed W의 population std(ddof0)/최대/(최대−최소). std≤0.5, 나머지는[0,1] |
| observed_approach | S0−μ_P(S); 양수면 현재 가까움이 과거 평균보다 큼 |
| observed_future_change | μ_F(S)−S0; 음수면 미래 평균 가까움이 작음 |
| observed_nearest_approach_m | μ_P(dmin)−dmin0, m. 개인 approaching velocity나 m/s가 아님 |
| observed_nearest_future_change_m | μ_F(dmin)−dmin0, m. 양수면 미래 평균 거리가 더 큼 |
| short_approach | S0−가장 가까운 과거 floor(R/2)개 observed 평균 |
| long_approach | S0−나머지 먼 과거 observed 평균 |
| observed_occupancy_3m/5m | μ_W(1[E_r>0]), [0,1] |
| observed_edge_growth_r / future_change_r | E_r(0)−μ_P(E_r), μ_F(E_r)−E_r(0), 쌍 수 단위 |
| observed_contact_size_growth_r / future_change_r | C_r(0)−μ_P(C_r), μ_F(C_r)−C_r(0), 관측 수 단위 |
| present_nearby_cluster_count/max_size | 원 안에 중심이 있는 현재 v2-1 군집 수/최대 N |
| observed_nearby_cluster_*_past_growth/future_change | 위 군집 요약의 현재−과거 평균/미래 평균−현재 |
| reliable_approach / reliable_future_change | full reliable일 때 위 soft 변화, 아니면NaN |
| approach_focus_separation_score | full reliable일 때 clip(S0−μ_P,0,1)×clip(S0−μ_F,0,1), [0,1] |
| stable_close_score | full reliable일 때 해당 r의 occupancy×μ_W(S)×(1−soft range), [0,1] |
| transient_excess_over_mean | full reliable일 때 clip(S0−μ_W(S),0,1), [0,1] |
| global_v2_moving_mean_r | 같은 run에서 strict+defined인 원래 frame-level 최대 v2-1 강도의 centered 평균. Anchor 밖 다른 지역일 수도 있음 |
| global_v2_moving_support_r | 위 평균에 사용된 정의된 sample 수;0이면 평균NaN |

접근→집중→분리는 경합의 필수 조건이 아니다. 점수의0/양수나 안정 점수로 실제 마킹/경합 클래스를 확정하지 않는다. approach_focus와 stable_close가 달라도 모두 축구 의미가 아닌 정의된 기하 양상이다.

### 지속·형성·해체·재접근

현재부터 과거/미래로 `reliable AND E_r>0`인 연속 prefix 길이를 Lp/Lf로 구한다. invalid를 건너뛰어 연결하지 않는다.

* `close_duration_lower_bound_s=(Lp+1+Lf)/25`, `past/future_close_lower_bound_s=Lp/25,Lf/25`. frame-bin 합계이므로101개가 전부 근접이면4.04초다. 처음/마지막 sample 시각 간격4초와 다르며 실제 사건 길이의 확정값이 아니다.
* 원래 현재 군집에서는 현재 edge가 존재한다. noise의 고정 probe처럼 관측은 valid지만 현재 E=0이면 duration0과 `current_relation_not_active`다. 관측이 불충분하면 durationNaN과 `current_observation_unknown`이다.
* prefix 직전/직후가 reliable한 비접촉이면 관측된 지역 관계 종료/시작으로 기록한다. `onset_offset_frames=−Lp`는 직전 known-off가 있을 때 첫 관측된 close offset이며, 실제 인물 관계의 최초 발생 정답이 아니다.
* Window 끝·record/split 경계·invalid가 먼저 나오면 onset/end_censored=True다. `window_censored`, `boundary_censored`, `observation_or_region_ambiguous`를 구분한다. 한 팀이 원 밖으로 나간 경우도 마지막 범주다.
* `reapproach_observed`는 현재 close이고 미래 전체가 reliable일 때 off→on이 관측되었는지0/1, 그 외NaN이다. hard threshold의 흔들림도 off→on을 만들 수 있다.

## 7. Conservative Cluster Matching: 보조 진단

동일 run의 인접 frame에서만, centroid step≤2m·N 비율 min/max≥0.5·대리 원판 IoU≥0.1인 연결을 만든다. 원판 반경=max(1m,diameter/2)이며 실제 선수 집합의 겹침/인물 동일성을 보장하지 않는다. 양끝 strict QC와 전역 H/A 인원 일치도 요구한다.

서로 후보가 정확히 하나인 reciprocal-unique일 때만 `(상대 frame, 그 frame 내부 cluster_id, center step)`를 기록한다. 여러 후보·병합·분리는 geometric_ambiguous, 관측/인원 문제는 observation_ambiguous다. 후보0은 no_clear_spatial_match로 남기며 실제 소멸이라고 확정하지 않는다. 경계/period0에는 연결하지 않는다. 대응을 이어 영구 track을 만들지 않았다.

공식 구간 next 대응의 분포:

| 현재 radius | status | 군집 관측 행 |
|---|---|---:|
| 3m | reciprocal_unique_geometry | 188,919 |
| 3m | observation_ambiguous | 64,298 |
| 3m | no_clear_spatial_match | 2,294 |
| 3m | geometric_ambiguous | 83 |
| 3m | boundary_or_outside | 9 |
| 5m | reciprocal_unique_geometry | 364,184 |
| 5m | observation_ambiguous | 122,060 |
| 5m | no_clear_spatial_match | 4,028 |
| 5m | boundary_or_outside | 15 |
| 5m | geometric_ambiguous | 1 |

reciprocal-unique553,103행은 형식적인 기하 조건 통과 수다. 실제 동일 집단 추적 정확도가 아니며 ambiguous84행만큼만 불확실하다는 뜻도 아니다. observation-ambiguous186,358행과 no-clear6,322행도 별도 불확실성이다. 합성 원시 좌표에서3m 집단2→1 병합과1→2 분리를 실제로 만들었고, 둘 다 강제 대응하지 않고 geometric_ambiguous로 유지했다.

## 8. 합성 양상과 단순 평균의 차이

실제 경기를 보거나 라벨링하지 않고8v8 합성 관측에서 중심의 두 상대 관측 거리만 바꿨다. 현재 배치는 모두 동일하다. 단위는 distance kernel 기준의 dimensionless 값이다.

| 합성 입력 | 현재 S | Window mean | 과거 접근 | 미래 변화 | 접근-집중-분리 | 안정 근접 |
|---|---:|---:|---:|---:|---:|---:|
| constant_close | 0.716531 | 0.716531 | 0.000000 | 0.000000 | 0.000000 | 0.716531 |
| approach_then_hold | 0.716531 | 0.482318 | 0.473110 | 0.000000 | 0.000000 | 0.100261 |
| hold_then_separate | 0.716531 | 0.482318 | 0.000000 | -0.473110 | 0.000000 | 0.100261 |
| approach_peak_separate | 0.716531 | 0.248105 | 0.473110 | -0.473110 | 0.223833 | 0.020466 |
| single_observation_dropout | 0.716531 | 0.716531 | 0.000000 | 0.000000 | NaN | NaN |

`approach_then_hold`와 `hold_then_separate`는 현재 S와 Window 평균이 같지만 변화 방향이 다르다. 지속 근접은 pattern score0/안정0.7165, 짧은 피크는 pattern0.2238/안정0.0205였다. 같은 평균이나 현재 강도만으로 이 차이를 모두 표현할 수 없다. 이 합성의 이름은 입력 생성 시나리오이며 실제 경기의 정답 라벨이 아니다.

single dropout은 원래 관측된 soft 값 평균이 변하지 않아도 primary pattern을NaN으로 보류하고 종료 원인을 ambiguous로 기록했다. 이 동작을 실제 경합 종료 검출로 해석하지 않는다. 실제 경기에서는 움직임·압박·마킹·시야 변화가 비슷한 공간 신호를 낼 수 있어 semantic 분류는 미검증이다.

## 9. 실제 분포와 v1/v2-1 비교

공식 군집 관측은3m255,603행/5m490,288행이다. 다음은 raw 시간 지원과 품질 지원을 나눈 결과다.

| 현재 radius | 현재 군집 행 | full record101 | partial 양쪽80% | full reliable101 |
|---|---:|---:|---:|---:|
| 3m | 255,603 | 254,718 | 79,013 | 56,689 (22.18%) |
| 5m | 490,288 | 488,697 | 178,723 | 135,179 (27.57%) |

기록 범위가 부족한 행은2,476개지만, quality/count/local visibility 때문에 패턴 지원은 훨씬 더 줄었다. radius별 population과 anchor 원 크기가 다르므로3m/5m coverage 비율을 통제된 radius 우열로 읽지 않는다. 같은 집단의 두 radius 또는 인접 frame의 반복 행을 독립 경합 사건으로 세지 않는다.

아래는 각 radius의 **같은 full-reliable cohort**에서 계산한 분포다. duration을 제외하면 dimensionless이며 occupancy는0..1이다.

| 현재 radius / 특징 | 중앙값 | P05 | P95 |
|---|---:|---:|---:|
| 3m / observed_approach | 0.027734 | -0.160009 | 0.238395 |
| 3m / observed_future_change | -0.029971 | -0.233555 | 0.162142 |
| 3m / observed_window_soft_std | 0.096989 | 0.021882 | 0.202149 |
| 3m / observed_occupancy_3m | 0.881188 | 0.445545 | 1.000000 |
| 3m / observed_occupancy_5m | 1.000000 | 0.851485 | 1.000000 |
| 3m / approach_focus_separation_score | 0.000000 | 0.000000 | 0.034130 |
| 3m / stable_close_score | 0.269344 | 0.082117 | 0.509540 |
| 3m / close_duration_lower_bound_s | 3.440000 | 1.640000 | 4.040000 |
| 5m / observed_approach | 0.005676 | -0.180316 | 0.185697 |
| 5m / observed_future_change | -0.008416 | -0.183269 | 0.181504 |
| 5m / observed_window_soft_std | 0.077042 | 0.018126 | 0.198175 |
| 5m / observed_occupancy_3m | 0.445545 | 0.000000 | 1.000000 |
| 5m / observed_occupancy_5m | 1.000000 | 0.554455 | 1.000000 |
| 5m / approach_focus_separation_score | 0.000000 | 0.000000 | 0.015689 |
| 5m / stable_close_score | 0.238836 | 0.080863 | 0.452290 |
| 5m / close_duration_lower_bound_s | 4.040000 | 2.160000 | 4.040000 |

부호의 기술 집계(아주 작은 양수도 포함, 후보 cutoff 아님):

| radius | 과거 접근>0 | 미래 변화<0 | 양쪽 조건의 곱>0 | 재접근 on / 정의 표본 |
|---|---:|---:|---:|---:|
| 3m | 34,598 | 35,220 | 18,958 | 3,335 / 56,689 |
| 5m | 71,480 | 73,131 | 31,141 | 4,818 / 135,179 |

Window에 끝까지 남아 있는 접촉이 많아 지속 하한은4.04초에서 검열된다. 이를 긴 실제 마킹 관계의 확정 길이로 해석하지 않는다. 원 안에서 가장 가까운 쌍이 바뀌거나 여러 다른 관계가 이어져도 근접 지속이 길어질 수 있다.

### Validation의 동일 frame에서 세 표현 비교

v1 현재/과거1초 지수, v2-1 현재 local 최대 강도, static moving mean, v2-2 양방향 score를 함께 정의할 수 있는 **같은 Validation frame**만 선택했다. static mean도101개가 모두 정의된 조건을 적용했다. 선택은 상관 기술 통계의 support 제한이며 파라미터/특징 선택을 하지 않았다. 아래는 Frame-level 최대 shape score와의 Spearman 상관이다.

| radius | 공통 frame | v1 score | 현재 v2-1 최대 강도 | static moving mean |
|---|---:|---:|---:|---:|
| 3m | 2,507 | 0.360 | 0.500 | 0.177 |
| 5m | 6,668 | 0.393 | 0.330 | 0.252 |

낮은/다른 상관과 합성의 순서 차이는 이동평균과 같은 값이 아님을 보이지만, 경합 성능 향상을 입증하지 않는다. Frame의 최대 shape와 최대 stable 값은 다른 현재 anchor에서 나올 수 있다. v1은 과거만, v2-2는 미래도 사용하므로 동일 정보 예산의 정확도 비교도 아니다. 기존 v1의 후보·비후보별 같은 cohort 분포는 `validation_comparison.json`에 보관했다.

## 10. Validation robustness와 가짜 변화 보류

v1/v2-1과 동일한12×151 support와 seed2027이다. ±50 전체를 가지는 각 support 중앙51개, 총612개 frame에서 clean current cluster3,191개를 고정 probe로 사용했다. 모두 Validation이고 support 밖을 빌려오지 않았다. clean의 full-reliable probe는1,032개였다.

jitter/dropout/permutation/team swap/composition을 기존 방식으로 적용했다. spatial anchor는 clean의 위치/범위에 고정해 비교하므로 군집 redetection 전체의 강건성이 아닌 **동일 공간 probe의 조건부 민감도**다. 별도로 다시 탐지한 군집의 adjacent-match 분포도 기록했다. noise 뒤 동일 cluster_id를 동일 집단으로 강제 매칭하지 않는다.

raw observed 특징은 전체 observed-mask가 같은 표본끼리 비교했다. quality-gated score/지속은 양쪽 모두 full reliable인 같은 표본끼리 비교했다. 표의 soft는 현재 kernel 강도, hard3/5는101개 관측의 occupancy이므로 연산·시간 평균의 차이가 있다. 범위가 같다는 이유로 정확도 순위를 매기지 않는다.

| 교란 | 동일 observed mask 수 | soft abs차 P95 | hard3 occupancy P95 | hard5 occupancy P95 | 공통 full reliable | shape P95 | duration P95(초) |
|---|---:|---:|---:|---:|---:|---:|---:|
| jitter_0.1 | 2,715 | 0.051744 | 0.029703 | 0.019802 | 1,028 | 0.010551 | 0.76 |
| jitter_0.5 | 2,348 | 0.228344 | 0.168317 | 0.089109 | 856 | 0.044589 | 2.60 |
| jitter_1 | 2,029 | 0.381098 | 0.257426 | 0.257426 | 671 | 0.099985 | 4.04 |
| dropout_0.1 | 400 | 0.313881 | 0.158416 | 0.049505 | 0 | NaN | NaN |
| dropout_0.2 | 117 | 0.481232 | 0.237624 | 0.029703 | 0 | NaN | NaN |
| dropout_0.3 | 24 | 0.493772 | 0.216337 | 0.069307 | 0 | NaN | NaN |
| composition_0.2 | 600 | 0.390918 | 0.069307 | 0.019802 | 0 | NaN | NaN |

Permutation/team swap에서는3,191개 probe의 observed 지원이 같았고,1,032개 full-reliable probe의 패턴·지속 변화도0이었다. raw 특징 차이도0이었다.

독립적인 프레임별 dropout은 전역 인원을 자주 바꾼다. dropout10/20/30%와 composition20%에서는 full reliable 패턴이0개라 shape/duration 비교를 내지 않았다. **NaN은 오차0이나 성능 향상이 아니다.** observed 특징에는 dropout10%에서 과거 접근 차 P95약0.279처럼 원래 입력과 큰 차이가 생겼다. 이는 가짜 변화가 raw feature에 들어올 수 있다는 증거이며, 기본 pattern score는 이를 보류했다. 같은 인원 수의 재획득/교체는 이 guard로 모두 찾을 수 없다는 한계가 남는다.

3m보다5m occupancy가 일부 작은 교란에 덜 변했지만1m jitter에서는 차이가 작아졌고, kernel도0.1/0.5/1m에서 점점 크게 변했다. hard edge를 연속 kernel로 대체하거나 특정 radius를 승자로 선택하지 않았다. 미관측과 실제 분리의 원인을 완전히 식별한 것이 아니며 tracking 정확도·실제 경합 정확도는 측정하지 않았다.

## 11. 테스트·전체 출력·보존

기존213개를 먼저 재실행해 모두 통과했고, 최종 **233 passed in24.74s**(신규20), 실패0개다. 접근/유지/분리/재접근, 같은 현재·같은 평균의 다른 순서, dropout/지역 이탈의 ambiguity, hard 경계의 smooth kernel, ±25/50/75, split sentinel·순수 record gap·half boundary·period0, permutation/H-A swap, reciprocal merge/split, source/csv 정렬·기존 출력/덮어쓰기 보호를 검사했다. 별도 원시 합성 좌표의 merge/split assert도 통과했다.

개발 중 pandas3의 bool 열에 NaN을 넣는 type 오류를 테스트가 찾아 nullable 의미의 float flag로 고쳤다. 고정 probe에서 현재 접촉이 알려진0인 경우와 관측 unknown인 경우도 분리해 검증했다. 수정 후 최종 전체 재실행의 두 CSV는 초기 전체 실행과 byte SHA까지 동일했다(실제 현재 군집에서는 그 known-empty probe 분기가 발생하지 않음). 설정이나 threshold를 재탐색한 실험이 아니다.

최종 결과는 Frame **199,319행×37열**, Cluster **789,005행×110열**이다. 원래 군집 키와 현재35개 공간 특징을 보존하고 시간 특징·matching 진단을 추가했다. 최초/최종 frame0..199318, 중복/누락 없음, Long 키의 순서/개수도 v2-1과 같음을 확인했다.

전체 CSV를 chunk로 재읽어 모든 경계 지원 수, mask 포함 관계(reliable⊆observed⊆record), full score의 validity/NaN·범위, 지속 하한, 인접 matching frame±1 및 후보1개, Frame/Long 집계 일치, Inf 없음, v1 참조 열을 확인했다. Long과 Frame의 현재 관측 부족을 비후보로 바꾸지 않았고 새로운 candidate_state도 만들지 않았다.

작업 시작 시 보호한 기존 소스·문서187개 SHA256과 원본/기존 산출물8,276개 size/mtime은 최종까지 동일했다. A/B/Checkpoint/Baseline을 변경하지 않았고, v2-1 입력 CSV의 SHA도 원래 verification과 일치했다. 평가 GT 라벨을 파싱하거나 사용하지 않았다(보호파일 byte hash 감사와는 구분). 기존 회귀 pytest의 합성 neural 검사는 실제 경기 A/B 재학습이 아니다.

환경은 mtah2027 / Python3.11.17 / NumPy2.4.6 / Pandas3.1.0 / macOS arm64 CPU다. 최종 전체 추출·기본 집계 기록 시간은251.06초이며 이후 보존 검사·Validation robustness·보조 통계 시간은 제외한다. peak 메모리는 측정하지 않았다. 신경망·새 fitting·후보 임계값 조정·외부 업로드·웹 서버·Git push는 하지 않았다. 전체 deployment는 Test 시간 블록도 포함하지만 특징/설정 선택·튜닝은 Test에서 하지 않았다.

## 12. 재현 및 로컬 파일

프로젝트 루트에서 새 결과 폴더를 지정한다.

```bash
conda activate mtah2027
export PYTHONPATH="$PWD/competition_judgement/src:$PWD/MTAH_Relational_Starter:$PWD"
export PYTHONDONTWRITEBYTECODE=1
python -m pytest -q -p no:cacheprovider MTAH_Relational_Starter/tests competition_judgement/tests
python -m competition_judgement.analyze_temporal_relations --out-dir data/derived/competition_judgement/temporal_v2_2_reproduction
python -m competition_judgement.temporal_robustness --run-dir data/derived/competition_judgement/temporal_v2_2_reproduction
```

`--cache`, `--v2-dir`, `--split-manifest`, `--config`로 기존 입력 경로를 지정한다. 출력은 data/derived/competition_judgement 아래로 제한하고 덮어쓰기를 거부한다. pair kernel/hard-edge 캐시는 frame-local 물리 관계이며 학습 embedding이 아니다. 실제 경기에서 복원 가능한 모든 파생물은 Git ignore 및 로컬 전용이다.

최종 폴더: `data/derived/competition_judgement/temporal_v2_2_final_20261109/`.

* [frame_temporal_features.csv](../../data/derived/competition_judgement/temporal_v2_2_final_20261109/frame_temporal_features.csv): 약57.1MB.
* [cluster_temporal_features.csv](../../data/derived/competition_judgement/temporal_v2_2_final_20261109/cluster_temporal_features.csv): 약866.7MB.
* [temporal_report.json](../../data/derived/competition_judgement/temporal_v2_2_final_20261109/temporal_report.json): 전체·radius별 지원·특징 분포·matching 집계.
* [temporal_robustness.json](../../data/derived/competition_judgement/temporal_v2_2_final_20261109/temporal_robustness.json): 고정 Validation support·조건별 동일 지원 비교.
* [validation_comparison.json](../../data/derived/competition_judgement/temporal_v2_2_final_20261109/validation_comparison.json): 동일 Validation frame의 v1/v2/static-mean 비교.
* [synthetic_scenarios.json](../../data/derived/competition_judgement/temporal_v2_2_final_20261109/synthetic_scenarios.json): 합성 양상·원시 merge/split 예시 집계.
* [verification.json](../../data/derived/competition_judgement/temporal_v2_2_final_20261109/verification.json), [output_validation.json](../../data/derived/competition_judgement/temporal_v2_2_final_20261109/output_validation.json): hash·원본 보존·최종 검사.
* `pattern_descriptions.json`, `frame_coverage.json`, `preregistered_config.json`, `pair_cache/`, `LOCAL_ONLY.txt`.

CSV는 필요한 열만 usecols로 읽고, mask hex는 `dtype='string'`로 지정해 leading zero를 보존한다. 다른 nullable 열은 low_memory=False 또는 명시적 nullable dtype을 사용한다. 파일 내용이나 pair cache를 공개 GitHub/외부 서비스/LLM으로 옮기지 않는다.

## 13. v2-3 인터페이스와 남은 한계

v2-3에 전달할 것은 원래 `(frame,radius_m,cluster_id)`와 현재 공간 특징, 시간 변화 벡터·지원 counts/masks, full/partial 품질 eligibility, 검열된 지속·onset/reapproach, 인접 matching status다. Frame37열에는 radius별 현재 군집 수·valid anchor 수·현재 strong 관계 수·최대 shape/stable/변화와 static 평균·matching ambiguity 비율이 있다. Frame의 서로 다른 최댓값이 같은 anchor에서 나오리라 가정하지 않고 필요하면 Long으로 돌아가야 한다.

기본 후보 state나 cutoff는 정의하지 않았다. v2-3은 NaN/unknown을 실제 비경합으로 채우지 않고, partial 관측을 어떻게 사용할지 별도 사전 정책으로 결정해야 한다. 강한 full-window guard의 coverage 손실은 미해결 문제이며 이번 결과를 보고 완화하지 않았다.

**Q1. 정적 군집 대비 추가 정보는?** 과거 접근 방향·단기/장기 차이, 현재 피크가 과거/미래에 비해 얼마나 두드러지는지, 유지 시간 하한·재접근·관측 공백/경계 검열을 표현한다. 같은 정적 강도나 평균으로 환원되지 않는다.

**Q2. 지속 마킹과 짧은 접근·집중·분리를 구분하는가?** 합성의 지속 근접과 짧은 피크는 구분했다. 실제 마킹/볼 경합의 semantic 분류는 아직 검증하지 않았다. anchor 안에서 다른 선수 쌍이 교체되거나 여러 관계가 이어지는 경우도 같은 시계열을 만들 수 있다.

**Q3. Tracking 누락의 가짜 변화는 어떻게 처리하는가?** raw 관측 변화와 reliable 패턴을 나누고, 관측/인원 변화·한 팀의 지역 부재·모호한 대응에 unknown/NaN·검열을 부여한다. 실제 분리인지 지역 이탈/미관측인지 불명확하면 확정하지 않는다. 같은 인원 수의 재획득/팀 오분류까지 완전히 식별하지 못하고, 그 대가로 많은 Window를 보류한다.
