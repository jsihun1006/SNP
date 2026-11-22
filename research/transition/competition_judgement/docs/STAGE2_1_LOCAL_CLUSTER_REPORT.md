# v2-1 Local Mixed Cluster Detection 실행 보고서

2026-11-09. 신경망 없는 **현재 프레임의 국소 공간 특징 추출**을 구현하고 전체 로컬 실행을 완료했다. v1 코드·threshold·state는 보존했다. 새 경합 후보 규칙, 실제 경합 확률/정답, H/A/D/X, 군집 시간 추적, 미래 프레임 분석은 구현하지 않았다.

## 1. 결론

v2-1은 작은 접촉 집단 여러 개, 하나의 큰 연결 사슬, 군집의 물리적 크기·밀도·구성을 별도로 표현한다. 공식 구간에서3m 혼합 군집이2개 이상인 frame은78,716개, v1 not_candidate이지만 사전 정의한 강한 국소 관계가 있는 frame은50,662개였다. 이는 **기존 표현에서 구분하기 어려운 관측 관계의 추가 표현**이며 실제 경합을 새로 맞힌 수가 아니다.

국소 특징을 관측 누락에 무조건 강하다고 결론내릴 수 없다. 계산 가능한 coverage는 넓지만 max contact strength는 공통 유효 support에서 v1 점수보다 더 크게 변한 조건이 많았다. 5m는 작은 좌표 jitter의 edge 변화에 상대적으로 덜 민감했으나 더 큰 사슬과 dropout 민감성도 보였다.

## 2. 감사·파일·재사용

[사전 설계 감사](STAGE2_1_DESIGN_AUDIT.md)에 요청 문서와 실제 API 차이, 실행 전 고정 정의를 기록했다. 실제 위치는 루트 `mtah_relational/`이 아닌 `MTAH_Relational_Starter/mtah_relational/`이다.

기존 read-only `FrameStore`와 `assess_observations/POLICY`를 재사용했다. `max_bipartite_cluster_size`는 최대 크기만 반환하므로 모든 집합을 반환하는 새 BFS를 작성하고 결과를 대조했다. `dynamic_graph.build_graph`의 같은 팀 kNN/self-loop/먼 상대 fallback과 float32 tolerance는 이번 graph 정의와 달라 직접 호출하지 않았다. 원시 parser·ID 정렬·좌표 보간은 추가하지 않았다.

신규 파일:

| 파일 | 역할 |
|---|---|
| [local_clusters.py](../src/competition_judgement/local_clusters.py) | 현재 관측의3m/5m bipartite component 전수 추출, canonical ID와 radius 간 포함 관계 |
| [local_features.py](../src/competition_judgement/local_features.py) | 모든 군집 특징·프레임 요약·strict/계산 가능성 구분 |
| [local_diagnostics.py](../src/competition_judgement/local_diagnostics.py) | 기존 Validation support/seed의 jitter/dropout·한 관측 삭제 검사 |
| [analyze_local_clusters.py](../src/competition_judgement/analyze_local_clusters.py) | 전체 추출·Long CSV·v1 비교 Case·집계·출처/보존 감사 CLI |
| [compare_local_robustness.py](../src/competition_judgement/compare_local_robustness.py) | 정확히 같은 유효 frame에서 v1/local 강도 변화 비교, composition active span 분리 |
| [test_local_clusters.py](../tests/test_local_clusters.py) | 신규 합성26개 검사 |
| [local_clusters_v2_1.json](../configs/local_clusters_v2_1.json) | 실행 전 고정한 물리적 descriptor·Case 설정; fitting 없음 |

기존 v1의 README/manifest/소스/설정/보고서는 이번에 수정하지 않았다. `PACKAGE_MANIFEST.json`은 v1 당시 스냅샷이며 이번 v2 파일 목록은 위 표와 로컬 실행 hash에 기록했다. A/B·원본·제출 Baseline도 변경하지 않았다.

## 3. Graph와 radius 간 중복 해석

Node는 현재 QA view의 finite 관측이다. 원래 normalized xy는 기존 캐시에서105.3×68m의 float64 미터 좌표로 이미 변환되어 있다. 같은 팀≤0.2m duplicate 의심은 flag만 하고 유지한다. 피치 밖2m margin 초과만 기존 정책대로 제외하며 그 안의 outside 관측도 flag를 보존한다. nonfinite는 노드에서 제외한다. 모든 남은 노드는 observed=True이고 보이지 않는 선수를 노드로 만들지 않는다.

각 frame에서 전체 Euclidean 거리 행렬을 한 번 계산한다. `team_i≠team_j` 및 `distance≤r`일 때만 무방향 edge를 만들며 r은3m/5m다. 같은 팀 edge·self-loop·zone 경계·먼 상대 fallback은 없다. 실제 edge가 있는 connected component를 모두 기록하고 isolated node는 포함하지 않는다. 두 팀을 포함하는2명 집단도 유효한 component다.

ID는 `(frame,radius_m,cluster_id)`로만 해석한다. 좌표 signature로 정렬해 입력 순열과 팀 교환에 대해 재현되지만, 다음 frame에서 같은 번호라고 같은 선수 집단이 아니다. source slot과 frame 간 군집을 연결하지 않았다.

모든3m component는 한5m component에 포함된다. `family_id_5m`과 `same_node_set_across_radii`로 같은 시점의 포함/일치 관계를 기록한다. 공식 구간에서 동일 node set이 두 radius에 그대로 등장한 대응은163,234개였다. 이것은 두 scale에 저장된 관측의 대응 수이며 실제 경합 사건 수가 아니다. radius별 수치를 따로 보고하며3m와5m 행이나 여러 frame의 행을 더해 경합 횟수로 사용하지 않는다.

## 4. 군집 특징의 모든 계산식

N=nH+nA, E=현재 radius의 H-A 무방향 edge 수, d=군집 내 미터 거리, c=관측 좌표 평균이다. 연결 군집에서 nH,nA≥1, N≥2, E≥1이다.

| 특징 | 정의·단위 |
|---|---|
| n_players / n_home / n_away | QA 후 군집 내 관측 수 N/nH/nA; 실제 고유 인물 수 보증 아님 |
| n_cross_edges | H×A 조합에서 d≤r인 쌍 수 E; 양방향으로 두 번 세지 않음 |
| n_unique_contact_players | edge에 참여하는 서로 다른 현재 관측 수. 이 component 정의에서는 N과 같음 |
| cluster_centroid_x/y | c=Σp/N, m |
| cluster_radius_m | Σ||p−c||/N, m. enclosing radius나 최대 반경이 아님 |
| cluster_bbox_length_m / width_m | max(x)−min(x), max(y)−min(y), m |
| cluster_bbox_area_m2 | 위 length×width, m². 공선/중복이면0도 정의된 값 |
| cluster_max_pairwise_distance_m | 모든 팀 조합 포함 max(d_ij), m; 연결 사슬의 실제 확장을 표현 |
| min_cross_distance_m | 군집 내 모든 H×A 거리의 최소, m |
| mean_cross_edge_distance_m | Σ d_edge/E, m; 현재 r의 edge만 평균 |
| cross_pair_count_3m/5m | 해당 component 멤버의 전체 H×A 쌍 중 d≤3/5인 수; component 생성 radius와 별개 |
| cross_pair_density | E/(nH×nA), [0,1]. 전체 경기장 인원을 분모로 쓰지 않음 |
| home_fraction / away_fraction | nH/N, nA/N, [0,1], 합1 |
| team_balance | 2min(nH,nA)/N, (0,1]; 수적 균형의 보조 지표 |
| component_size | N, n_players와 의도적으로 같은 값 |
| edge_density | E / C(N,2)=2E/[N(N−1)], [0,1]; H-A 가능 쌍만 분모로 하는 cross density와 다름 |
| max_degree / mean_degree | 최대 H-A 이웃 수 / 2E/N; 같은 팀 이웃 제외 |
| contact_strength | balance × cross_pair_density × mean(exp(−d_edge/3m)), [0,1]. 두 radius에 동일한3m 감쇠 길이 |
| small_mixed_cluster | N≤3. 2명과3명 집단을 포함 |
| compact_mixed_cluster | diameter≤2r AND mean radius≤r. 기본3m이면6m/3m, 비교5m이면10m/5m |

강도는 경합 확률이 아니고 최적 가중치를 학습한 지표도 아니다. 곱은 구성·연결·거리의 동시 조건을 기술하기 위한 사전 정의다. strong relation이라는 Case용 보조 조건은 compact이면서 strength≥0.5다. 균형 잡힌 두 관측만 있을 때 d≤3ln2≈2.079m로 해석된다. 실제 영상에 맞춰 수치를 바꾸지 않았다.

## 5. 프레임 요약과 결측 의미

프레임별 필수 unsuffixed 지표는 기본3m 값의 alias이며5m는 `_5m`로 별도 보존한다. 양 radius에 다음 값이 있다.

* n_local_clusters: 모든 non-isolated mixed component 수.
* max_local_cluster_size: N 최대값; n_small/n_compact_mixed_clusters는 위 사전 조건에 맞는 군집 수.
* max_local_cross_density / max_local_contact_strength: 군집별 지표의 최대값.
* total_unique_players_in_mixed_clusters: 같은 radius에서 disjoint component의 N 합. source 중복이 있으면 물리적 인물 중복까지 제거한 수가 아니다.
* largest_cluster_radius_m: 최대 N 군집의 평균 반경. 동률이면 canonical 첫 군집이다. 반경 자체의 최댓값과 다르다.
* max_cluster_diameter_m / n_cross_edges: 모든 군집의 지름 최댓값 / edge 합계.
* local_metric_available, local_cluster_extrema_available_3m/5m, quality_tier, strict_quality_valid와 원래 QA 사유.

양 팀이 관측되지만 edge가 없으면 군집 count/size/접촉 관측 수는0, 군집 extrema는 NaN이다. 한 팀이 없으면 count까지 NaN이고 local_metric_available=False다. 없는 군집의 max strength를 관측된0 강도로 채우지 않는다. 미래 값·시간 스무딩·인물 속도는 입력에 없다.

## 6. 전체 출력과 발생 빈도

`local_frame_features.csv`는 **199,319행×62열**, `local_clusters.csv`는 **789,005행×35열**이다. 모든 원본 frame0..199318을 유지한다. Long CSV에서 같은 frame에 여러 radius·cluster 행이 있을 수 있다. 공식158,857프레임에서의 집계는 다음과 같다. 아래 비율 분모는 공식 전체이며 frame 간 중복 관측을 사건으로 세지 않았다.

| radius | 군집 있는 frame | 여러 군집 frame | 군집 관측 행 |
|---|---:|---:|---:|
| 3m | 125,479 (78.99%) | 78,716 (49.55%) | 255,603 |
| 5m | 150,032 (94.44%) | 130,906 (82.40%) | 490,288 |

계산 가능한158,110프레임으로 분모를 바꾸면 여러 군집 비율은3m49.79%,5m82.79%다. 관측 가능한 군집0인 frame은 각각32,631/8,078개다.

radius가 커지면 기존 집단끼리 합쳐지지만, 이전 isolated 관측으로 새로운2명 집단도 생긴다. 따라서 **non-isolated mixed component 수는 radius에 따라 단조 감소하지 않는다.** 한편 최대 군집 크기와 혼합 군집에 포함된 관측의 합집합은3m⊆5m 관계를 만족한다.

### 크기 분포 (공식 구간의 군집 관측 행)

| 관측 수 N | 3m | 5m |
|---:|---:|---:|
| 2 | 217,347 | 369,312 |
| 3 | 23,664 | 66,549 |
| 4 | 7,381 | 20,823 |
| 5 | 3,587 | 11,633 |
| 6 | 1,827 | 6,510 |
| 7 | 1,020 | 4,735 |
| 8 | 509 | 4,301 |
| 9 | 166 | 2,415 |
| 10 | 84 | 1,644 |
| 11 | 18 | 1,431 |
| 12 | 0 | 595 |
| 13 | 0 | 250 |
| 14 | 0 | 44 |
| 15 | 0 | 46 |

2명 집단은3m의85.03%,5m의75.33%였다. 프레임 수준으로3m에서는 큰 단일 군집(N≥4)만 있는 경우4,040개, 작은 군집만 여러 개(Nmax≤3)가68,761개, 큰 군집과 작은 군집이 함께 있는 경우9,578개였다. 5m는 각각7,430/88,978/40,491개였다. 이 구조를 단일 최대 군집 수치와 구분해 저장했다.

### Compactness 분포

각 숫자 묶음은 중앙값/P95/최대, 단위m다.

| radius | 평균 반경 분포 | 지름 분포 | compact 관측 행 |
|---|---|---|---:|
| 3m | 1.099 / 1.576 / 3.805 | 2.237 / 4.087 / 12.633 | 253,011 (98.99%) |
| 5m | 1.849 / 3.153 / 7.056 | 3.794 / 8.665 / 24.859 | 476,235 (97.13%) |

3m edge 사슬도 지름12.63m,5m 사슬은24.86m까지 확장됐다. component 크기가 작거나 모든 edge가 가깝다는 사실과 전체 공간 집중은 다르다. compact 비율은2/3명 집단이 대부분인 영향이 크며, 높은 비율 자체가 실제 경합을 잘 찾는다는 근거가 아니다. cross density의 중앙값은 두 radius 모두1로 포화됐고, 최대 density 하나만으로 집단의 의미를 구별하기 어렵다.

## 7. v1과 같은 frame의 네 Case

사전 정의한 Case는 서로 배타적인 분류가 아니며 실제 경합 정답이 아니다. 기본3m와 비교5m를 분리한다.

1. v1 not_candidate이지만 compact+strength≥0.5인 국소 군집이 적어도 하나 존재.
2. v1 score≥원래 고정 threshold(0.3147864495529943)이면서 어떤 군집의 diameter>2r. 점수가 높아도 공간적으로 긴 집단을 포함할 수 있는 경우.
3. non-isolated 혼합 component가2개 이상 존재.
4. 같은 반>0·split·연속 frame에서 양끝 local metric이 정의되고 관측 팀 인원이 변하면서, 군집 수가 변하거나 최대 N 차≥2 또는 edge 수 차≥2인 경우. 원인을 tracking 오류로 확정하지 않는 **동반 변화**다.

| Case / radius | frame | 공식 전체 비율 |
|---|---:|---:|
| Case 1, 3m | 50,662 | 31.89% |
| Case 2, 3m | 1,381 | 0.87% |
| Case 3, 3m | 78,716 | 49.55% |
| Case 4, 3m | 754 | 0.47% |
| Case 1, 5m | 38,681 | 24.35% |
| Case 2, 5m | 3,427 | 2.16% |
| Case 3, 5m | 130,906 | 82.40% |
| Case 4, 5m | 938 | 0.59% |

Case1의3m50,662개 중49,076개는 v1 score 자체가 threshold 미만이고1,586개는 score가 높아도 causal 최소 지속 조건 때문에 not_candidate이었다. 5m는37,449/1,232개다. 따라서 Case1 전체를 전역 분모의 희석 때문이라고 설명하지 않는다. 'v1에서 포착하지 못했다'는 것은 이 알고리즘 출력과 국소 descriptor가 다르다는 뜻이며 실제 false negative를 뜻하지 않는다.

Case3의3m78,716개 중 strict59,690/low19,026개였다. Case4의 유효 직전 비교는158,104개, 인원 변화가 있는 비교는2,051개였다. 이 중 구조 변화 조건을 충족한754/938개를 기록했다. source slot을 선수 ID로 연결하거나 실제 누락 원인을 판정한 것이 아니다.

Case별 특징 분포(중앙값/P95):

| Case / radius | 해당 radius 최대 N | 최대 지름(m) | 최대 강도 |
|---|---|---|---|
| 1, 3m | 2.0 / 3.0 | 2.315 / 3.574 | 0.631 / 0.846 |
| 2, 3m | 7.0 / 9.0 | 6.842 / 9.708 | 0.448 / 0.784 |
| 3, 3m | 2.0 / 5.0 | 2.732 / 5.102 | 0.591 / 0.843 |
| 4, 3m | 2.0 / 6.4 | 2.801 / 6.196 | 0.592 / 0.870 |
| 1, 5m | 3.0 / 6.0 | 4.841 / 10.290 | 0.622 / 0.839 |
| 2, 5m | 9.0 / 12.0 | 11.957 / 16.237 | 0.405 / 0.795 |
| 3, 5m | 3.0 / 7.0 | 4.977 / 11.315 | 0.466 / 0.795 |
| 4, 5m | 5.0 / 11.0 | 6.534 / 13.036 | 0.422 / 0.824 |

모든 Case의 평균·P05·P95·최대·정의된 표본 수와 품질별 수는 로컬 JSON에 있다. frame 최대 N/지름/강도는 서로 다른 군집에서 나올 수도 있으므로 한 대표 집단의 결합 특징으로 해석하지 않는다.

## 8. 품질별 계산 가능 범위

공식 구간의 strict-valid111,733개(70.3356%), strict는 아니지만 local available46,377개(29.1942%), 상대 관계 미정의747개(0.4702%)를 구분했다. 전체 local available은 **158,110개(99.5298%)**다. v1의 보류47,124개를 비후보로 바꾸거나 v2 후보를 선언하지 않았다.

| quality tier | frame | 3m 군집 있는 frame | 5m 군집 있는 frame |
|---|---:|---:|---:|
| strict_valid | 111,733 | 93,104 | 109,408 |
| low | 46,377 | 32,375 | 40,624 |
| insufficient | 747 | 0 | 0 |

관측이 적어도 실제 보이는2명 관계의 거리·구성·집중 정도는 계산할 수 있다. 그러나 군집 크기·강도·존재 자체가 누락의 영향을 받고, 관측되지 않은 상대 때문에 실제 집단이 분할돼 보일 수도 있다. 같은 팀 중복은 삭제하지 않았으므로 실제 고유 인물 수나 신뢰도 확률도 아니다. bounds/missing/duplicate/invalid 사유를 유지했다. metadata 팀 매핑의 정합성은 관측의 실제 팀 분류가 정확하다는 보증이 아니다.

공식 구간 밖에도 공간 특징을 계산하지만 period/in_official_block과 원래 v1 상태를 그대로 보존한다. 이 관측을 실제 경기 경합 정답으로 쓰지 않는다.

## 9. Validation 노이즈·누락

v1과 동일한12개×151프레임(1,812개 고유 관측), seed2027을 사용했다. 조건 간 같은 난수를 스케일해 jitter0.1/0.5/1m 및 nested dropout10/20/30%를 만들었다. composition은 각 support offset65..85의20% 관측 삭제다. 물리적 GT나 경합 label은 없고 원본은 메모리 복사만 했다. support 밖 시간 흐름이나 서로 다른 선수의 ID를 연결하지 않았다.

군집 수 변화율은 전체 paired local-available 관측에서의 비율, N 변화는 최대 군집 크기 차의 P95다. edge 변화는 같은 원래 관측 위치에 대한 within-frame 합성 pairing에서 symmetric difference/union의 평균이다(둘 다 edge0이면0). 이 QA pairing은 시계열 선수 ID가 아니다.

| 교란 | 군집 수 변화율 3m/5m | 최대 N 차 P95 3m/5m | edge 변화 평균 3m/5m | local 유효 / strict 유효 |
|---|---|---|---|---:|
| jitter_0.1 | 10.38% / 7.84% | 0.0 / 1.0 | 0.059 / 0.032 | 1,812 / 1,422 |
| jitter_0.5 | 39.13% / 35.71% | 2.0 / 2.0 | 0.291 / 0.162 | 1,812 / 1,415 |
| jitter_1 | 53.97% / 55.57% | 2.0 / 3.0 | 0.490 / 0.322 | 1,812 / 1,385 |
| dropout_0.1 | 24.89% / 39.96% | 2.0 / 2.0 | 0.169 / 0.184 | 1,812 / 1,265 |
| dropout_0.2 | 43.16% / 65.67% | 2.0 / 3.0 | 0.325 / 0.360 | 1,812 / 831 |
| dropout_0.3 | 57.12% / 81.02% | 3.0 / 4.0 | 0.466 / 0.510 | 1,812 / 358 |
| composition_0.2 | 7.06% / 8.89% | 0.0 / 1.0 | 0.047 / 0.049 | 1,812 / 1,327 |

Permutation/H-A swap에서는 군집 수·크기·대칭 local 강도가 모두 불변이었다. v1의 score 차이도 약1e-16 반올림 수준이었다. permutation의 edge matching 통계는 계산하지 않고 invariant 특징으로 확인했다.

5m edge는 작은 jitter에서 상대 변화가 작았다. 하지만1m jitter에서 군집 수 변화율은5m가 더 컸고, dropout에서는5m의 집단 수/크기 변화가 더 컸다. 따라서 radius 하나가 모든 지표에서 더 안정적이라고 선택하지 않는다. 원래 tracking은 정답 위치가 아니고5m가 더 정확하다는 결론도 아니다.

### v1 지수와 local 최대 강도의 동일 support 비교

각 radius마다 원래/교란 local 강도와 원래/교란 v1 score가 **모두 정의된 정확히 같은 frame**만 비교한 표다. 이 supplementary 계산은 비교 support를 맞추는 목적이며 특징·규칙·threshold를 바꾸지 않았다. local extrema가 없는 frame을0으로 채우지 않고 누락 정의 수도 원래 robustness JSON에 기록했다.

| 교란 / radius | 같은 frame 수 | v1 abs차 P95 | local 강도 abs차 P95 |
|---|---:|---:|---:|
| jitter_0.1, 3m | 1,330 | 0.046795 | 0.060349 |
| jitter_0.1, 5m | 1,402 | 0.047539 | 0.057799 |
| jitter_0.5, 3m | 1,251 | 0.100357 | 0.257847 |
| jitter_0.5, 5m | 1,368 | 0.113845 | 0.286109 |
| jitter_1, 3m | 1,158 | 0.133794 | 0.361598 |
| jitter_1, 5m | 1,325 | 0.147136 | 0.414532 |
| dropout_0.1, 3m | 1,033 | 0.089487 | 0.159287 |
| dropout_0.1, 5m | 1,159 | 0.103721 | 0.241918 |
| dropout_0.2, 3m | 622 | 0.100467 | 0.193946 |
| dropout_0.2, 5m | 733 | 0.121574 | 0.295593 |
| dropout_0.3, 3m | 261 | 0.109062 | 0.253206 |
| dropout_0.3, 5m | 311 | 0.136337 | 0.321653 |
| composition_0.2, 3m | 1,241 | 0.068150 | 0.000000 |
| composition_0.2, 5m | 1,313 | 0.068435 | 0.000000 |

양쪽 점수 범위는[0,1]이지만 목적·정의·시간 문맥이 다르므로 차이 크기를 정확도로 비교할 수 없다. 이 표에서 여러 jitter/dropout 조건의 local 강도 변화가 더 컸으므로 **local이 전반적으로 더 강건하다고 주장하지 않는다.** 반면 dropout30%에서도 local metric은1,812개 모두 계산됐지만 strict 유효는358개만 남았다. 이것은 관측 정보 coverage 차이이지 신뢰도 향상이 아니다.

composition의 전체 support에서는 대부분의 frame을 건드리지 않아 local P95가0이었다. 실제 교란한252개 frame 중 공통 유효 support로 제한하면3m93개에서 v1/local P95=0.072269/0.235532,5m103개에서0.081968/0.365559였다. 전체 구간의0을 단기 누락에 대한 불변성으로 읽으면 안 된다. 국소 최대값은 가장 강한 관측 집단이 남으면 바뀌지 않을 수도 있으며, 그것이 전체 군집 구조의 안정성을 뜻하지도 않는다.

### 한 관측 삭제: 소멸과 분열 분리

clean Validation의 각 component에서 각 멤버 관측을 한 번씩 제거했다. mixed component가2개 이상 살아남는 분열, 고립 관측까지 포함한 분리, mixed 관계가 전부 소멸한 경우를 나눴다. 분모는 frame 수가 아니라 **군집-멤버 삭제 trial 수**이며 큰 군집과 반복 frame이 더 많이 반영된다.

| radius | 삭제 trial | 원래 N≥3 trial | mixed≥2로 분열 | 고립 포함 분리 | mixed 소멸 |
|---|---:|---:|---|---:|---:|
| 3m | 7,391 | 2,449 | 72 (0.97%; N≥3에서 2.94%) | 861 | 5,373 |
| 5m | 16,099 | 7,263 | 345 (2.14%; N≥3에서 4.75%) | 2,134 | 9,599 |

2명 집단에서 하나가 사라지는 것은 소멸이며 분열로 세지 않는다. 현재 표본에서5m가 큰 component를 더 많이 포함해 분열 가능 trial과 분열 비율도 더 컸다. 이것을 실제 경기의 tracking 실패 확률로 해석하지 않는다.

## 10. 테스트·실제 정합성·보존

기존187개를 먼저 실행해 모두 통과했다. 최종 suite는 **213 passed in21.31s**(기존187+신규26), 실패0개다. 두 관측·별/사슬/여러 집단, 정확한3/5m 경계, zone 밖 연결, 빈/한 팀 부재/관측은 있으나 edge0, permutation/H-A swap, duplicate·NaN/Inf·bounds, raw read-only 보존, 단일 관측 삭제, radius 포함 관계, 같은 support 비교, synthetic cache→CSV와 전체199,319축을 검사했다.

합성에서 관계 없는 먼 관측을 추가했을 때 local strength는 같고 v1은 전역 분모의 영향으로 작아지는 것도 확인했다. 이는 설계 목적의 수학적 성질이지 실제 dropout 전체에서 우월하다는 증거가 아니다. 초기 한 fixture의 두 집단 간 간격을 정확히3m로 둔 채 분리돼야 한다고 기대한 검사는 inclusive 정의에 맞게 fixture를 수정했다. 남은 실패는 없다.

실제 전체 출력에서도 다음을 확인했다.

* frame0..199318 연속·유일, Long primary key `(frame,radius,cluster_id)` 유일, frame 내부 ID0..K−1.
* 모든 cluster에 양 팀/실제 edge 존재, 크기 합·degree·density 분모 일치, cluster 수와 Frame CSV 집계 일치.
* 3m→5m parent 포함·동일 멤버 flag 정합성, nested 최대 크기/관측 합집합 관계.
* 각 radius의 edge 합이 기존 v1의 전역 pair count와 모든 계산 가능 frame에서 정확히 같음.
* cluster 숫자에 NaN/Inf 없음, frame 미정의 값은 NaN이며 Inf 없음.
* 기존 helper와31개 frame×2radius=62개 최대 component 크기 대조 통과.
* 기존 v1 state/score와 새 CSV의 v1 참조 열이 동일하며 원래 CSV의 byte SHA도 그대로.
* 시작 시 보호한 기존 소스·문서176개 SHA와 원본/산출물8,268개 size/mtime 동일. A/B/체크포인트/제출 Baseline 보존.

원래 출력의 코드 hash는 추출/기본 robustness 당시 코드다. 이후 추가한 동일-support 보조 분석 모듈까지 최종 `output_validation.json`에 hash를 기록했다. 기본 특징 추출 코드는 그 사이 변경하지 않았다. 기존 pytest에는 합성 neural 검사가 포함되지만 실제 경기 신경망 재학습은 하지 않았다.

실행은 Conda mtah2027, Python3.11.17, NumPy2.4.6, Pandas3.1.0, macOS arm64 CPU였다. 본 추출/기본 진단 기록 시간73.60초이며 마지막 보존 검사와 이후 동일-support 보조 분석 시간은 제외한다. peak 메모리는 측정하지 않았다. Test 시간 블록은 전체 deployment 특징에 포함하되 Test·GT·볼·Smart Tagging으로 특징을 고르거나 튜닝하지 않았다. 영상을 보고 라벨이나 규칙을 만들지 않았고, 외부 전송·웹 서버·Push도 수행하지 않았다.

## 11. 실행 명령·파일 읽기

프로젝트 루트에서 기존 결과와 다른 새 out-dir를 사용한다.

```bash
conda activate mtah2027
export PYTHONPATH="$PWD/competition_judgement/src:$PWD/MTAH_Relational_Starter:$PWD"
export PYTHONDONTWRITEBYTECODE=1
python -m pytest -q -p no:cacheprovider MTAH_Relational_Starter/tests competition_judgement/tests
python -m competition_judgement.analyze_local_clusters --out-dir data/derived/competition_judgement/local_v2_1_reproduction
python -m competition_judgement.compare_local_robustness --run-dir data/derived/competition_judgement/local_v2_1_reproduction
```

CLI의 `--cache`, `--v1-dir`, `--split-manifest`, `--config`로 기존 경로를 지정한다. 전체 출력은 data/derived/competition_judgement 아래로 제한하고 덮어쓰기를 거부한다. Git ignore를 실제 확인했다.

최종 로컬 폴더는 `data/derived/competition_judgement/local_v2_1_20261109/`다.

* [local_frame_features.csv](../../data/derived/competition_judgement/local_v2_1_20261109/local_frame_features.csv): 전체199,319행, 약78.5MB.
* [local_clusters.csv](../../data/derived/competition_judgement/local_v2_1_20261109/local_clusters.csv): Long 군집 관측, 약218.7MB.
* [local_cluster_report.json](../../data/derived/competition_judgement/local_v2_1_20261109/local_cluster_report.json): 모든 분포·Case·기본 robustness 집계.
* [equal_support_robustness.json](../../data/derived/competition_judgement/local_v2_1_20261109/equal_support_robustness.json): 공통 유효 support 및 composition active span 비교.
* [verification.json](../../data/derived/competition_judgement/local_v2_1_20261109/verification.json), [output_validation.json](../../data/derived/competition_judgement/local_v2_1_20261109/output_validation.json): 보존·hash·최종 출력/테스트 검사.
* `preregistered_config.json`, `LOCAL_ONLY.txt`: 사전 설정과 로컬 보관 조건.

nullable boolean/NaN 열이 있으므로 pandas로 전체 CSV를 읽을 때 `low_memory=False`를 사용하거나 해당 flag를 `dtype='boolean'`로 지정한다. 필요한 열만 `usecols`로 읽으면 메모리를 줄일 수 있다. 모든 경기 파생물은 로컬 전용이며 내용을 대화나 공개 저장소로 옮기지 않는다.

## 12. v2-2에 전달할 인터페이스와 미구현 범위

다음 단계에 제공 가능한 것은 원본 frame/period/split과 품질·가용성, frame-local summaries `[T,F]`, 각 시점의 ragged cluster features `[K_t,Fc]`, radius/family의 같은 시점 포함 관계다. 미정의 값에는 validity를 보존하고 양 팀 구조가 필요하면 기존 read-only cache의 현재 관측을 함께 쓸 수 있다.

군집 ID를 시계열 인물/집단 ID처럼 사용하지 않는다. 시간 window는 half/split/record gap을 넘지 않고 자연 결측 frame을 삭제하지 않아야 한다. v1 state/score는 변경 없는 참조 신호이며 점유 정답이 아니다. 이 보고서는 미래 window 계산·군집 매칭·후보 판정 변경을 구현하지 않는다.

**Q1. v1이 표현하기 어려운 국소 관계를 추가로 구분하는가?** 그렇다. 전체 인원으로 희석되지 않는 군집 내부 거리·균형·연결 정도와 독립 집단 수, 물리적으로 긴 component를 구분했다. 다만 Case1은 실제 경합 recall 개선이 아니다.

**Q2. 3m와5m의 구성·안정성 차이는?** 3m는 더 좁은 접촉 관계,5m는 더 많은 관측과 큰 사슬을 포함한다. 5m edge는 작은 jitter의 상대 변화가 작았지만 군집 수·크기·누락 민감성까지 일괄 우수하지 않았다. 어느 radius도 승자로 고르지 않았다.

**Q3. 불완전 관측에서 유용한 국소 정보는?** 실제 보이는 혼합 집단의 거리·구성·반경·지름·독립 집단 수는 낮은 coverage에서도 기록할 수 있다. 관측되지 않은 멤버나 실제 볼 경합을 복원한 것은 아니며 quality/가용성과 함께만 사용해야 한다.
