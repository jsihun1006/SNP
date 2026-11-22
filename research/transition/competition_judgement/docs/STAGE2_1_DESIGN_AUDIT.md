# v2-1 사전 설계·코드 감사

2026-11-09. v1 README·Stage0/1 보고서·02 설계, detector/quality/artifacts/diagnostics 및 실제 mtah_relational의 features/dynamic_graph/FrameStore를 확인했다. 기존187개 테스트를 먼저 재실행해187 passed in21.52s였다. 기존 코드·문서176개 SHA256과 원본/기존 산출물8,268개 size/mtime을 로컬 보존 목록에 기록했다. 기존 v1 코드/설정/출력은 수정하지 않는다.

## 재사용 경계

* 원시 parser를 만들지 않고 기존 read-only FrameStore를 사용한다. 관측 정제는 기존 assess_observations/POLICY(2m margin·0.2m 같은 팀 중복 의심)를 그대로 호출한다. 자연 결측을 보간하거나 중복을 합치지 않는다.
* `max_bipartite_cluster_size`는 최대값만 반환하므로 모든 component를 반환하는 새 함수를 추가한다. 합성 및 고정31개 실제 frame×2radii에서 기존 최대값과 대조한다.
* 기존 `dynamic_graph.build_graph`에는 같은 팀 kNN·self-loop·10m 밖 최근접 상대 fallback이 있어 새 graph로 직접 사용하지 않는다. float32 normalization과 radius tolerance도 이번의 미터 단위 inclusive 비교와 다르다.
* 한 프레임의 float64 거리 행렬을 한 번 계산해3m·5m mask와 모든 군집 특징에 재사용한다. source slot은 시계열 인물 ID로 쓰지 않는다.

## 실행 전 고정한 정의

3m/5m에서 H-A 거리≤r인 무방향 edge만 연결하며, edge가 있는 connected component를 모두 반환한다. 2명 집단도 보존한다. node는 QA 후 현재 관측이다. 고립 노드·같은 팀 edge는 component에 포함하지 않는다. 좌표에 따른 canonical 정렬은 프레임 내부 재현용이며 시간적 identity가 아니다.

같은 프레임의3m component는 정확히 하나의5m component에 포함된다. 각 행에5m family ID와 양 radius의 node set 일치 여부를 기록한다. radius별 수치를 따로 집계하며 두 scale의 행을 더해 실제 경합 횟수로 계산하지 않는다. 서로 다른 프레임의 번호도 동일 집단을 의미하지 않는다.

* radius: 군집 중심까지 평균 거리. diameter: 모든 군집 관측 쌍의 최대 거리. chain의 크기와 공간 집중을 분리한다.
* cross_pair_density=E/(nH×nA); edge_density=2E/[N(N−1)]. E는 해당 radius의 H-A 무방향 edge 수다.
* team_balance=2min(nH,nA)/N.
* contact_strength=team_balance × cross_pair_density × mean(exp(−d_edge/3m)), 범위[0,1]. 두 radius 모두 동일3m 감쇠 길이를 사용한다. 관측 관계의 기술 지표이며 경합 확률이 아니다.
* small: N≤3; compact: diameter≤2r 및 평균 radius≤r. 기본3m에서는 diameter≤6m/radius≤3m를 뜻한다. 최적값이 아니라 직접 해석 가능한 사전 조건이다.
* strong descriptive relation: compact이면서 strength≥0.5. 균형 잡힌 완전 연결 두 관측에서는 거리≤3ln2≈2.079m에 해당한다. 최종 후보 규칙이 아니다.
* 필수 unsuffixed 프레임 지표는 기본3m의 alias이며5m는 suffix로 따로 보존한다. 가장 큰 군집은 N 최대, 동률이면 공간 정렬 첫 component다.
* 양 팀 관측은 있으나 edge가 없으면 count0, 군집 extrema는 NaN. 한 팀이 없으면 count까지 NaN이며 local_metric_available=False다. strict와 계산 가능성은 별개다.

## 비교 Case와 perturbation

Case1: v1 not_candidate + strong local relation. Case2: v1 score≥원래 고정 threshold + diameter>2r. Case3: 해당 radius의 혼합 component≥2개. Case4: 직전 프레임과 관측 인원 변화가 동반되고, component 수가 변하거나 최대 크기 차≥2 또는 edge 수 차≥2인 경우다. Case4는 같은 period>0/split/contiguous pair, 양쪽 local-metric available에서만 계산하는 기술 진단이다. 실제 추적 오류나 경합 원인을 확정하지 않는다. 미래 프레임·인물 연결은 사용하지 않는다.

Validation support12×151과 seed2027은 v1과 동일하다. jitter0.1/0.5/1m, dropout10/20/30%, permutation/H-A swap, offset65..85의20% composition 제거를 고정한다. 또 각 clean component의 현재 관측을 하나씩 제거해, 남은 mixed component≥2로 분열되는 경우·고립 관측 포함 분리·혼합 연결 소멸을 각각 센다. 이것은 같은 시점의 합성 관측 삭제이며 프레임 간 추적이 아니다.

v1 threshold/state를 변경하지 않으며 v2의 새 후보 state를 만들지 않는다. 실제 영상/볼/SmartTag/GT를 기준으로 조건을 바꾸거나 Test에서 특징을 선택하지 않는다. 전체 frame 출력은 Test 시간 블록도 포함하지만, 특징 설계·노이즈 진단과 별개인 고정 deployment 산출물이다. 실제 정확도는 평가하지 않는다.
