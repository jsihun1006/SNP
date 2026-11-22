# 02 — 규칙 기반 경합 후보 탐지 설계 v1

## 0. 구현 철학
어떤 선수가 공을 실제로 통제했는지 알지 못하므로 `competition_judgement`는 **경합 후보의 정량적 단서와 관측 품질**을 계산한다. 확률 출력/정답 라벨은 없다. 단순 위치뿐 아니라 짧은 시계열에서 접근→근접 집단 형성→유지→분리를 추적한다. 과적합을 줄이기 위해 초기에는 기존 규칙을 보존하고 소수의 해석 가능한 지표만 사용한다.

## 1. 기존 구현을 기준선으로 고정
`mtah_relational.candidate_baseline.CausalContestedIndex`의 기존 수식을 임의로 변경하지 않는다.

```text
contact = (3m 이내 상대가 있는 홈팀 인원 + 원정팀 인원)/(nH+nA)
cluster = max_cross_cluster_3m/(nH+nA)
proximity = exp(-cross_min_m/3)
spatial_contact_index = 0.5*contact + 0.3*cluster + 0.2*proximity

# 관측이 양 끝과 중간 모두 유효한 과거 25프레임 변화
contact_delta = contact[t] - contact[t-25]
approach_m_per_s = cross_nearest_mean_m[t-25] - cross_nearest_mean_m[t]
approach_index = 0.5*clip(contact_delta,0,1) + 0.5*clip(approach_m_per_s/5,0,1)
contested_index = 0.8*spatial_contact_index + 0.2*approach_index
```

과거1초 이용 불가능 시 기존 baseline처럼 현재 spatial index만 사용하는 경로를 유지하고 `temporal_available=False`로 기록한다. **점수는 0~1이라도 확률이 아니다.**

## 2. 입력 특징 (물리 단위+validity)
- `cross_pairs_3m/5m/10m`: 반대 팀 전체 선수 쌍의 거리 임계값 이하 개수(포함경계).
- `cross_density_rm`: 위 쌍 개수/(관측 홈×원정 인원). 인원 누락을 완전히 보정하지 못함.
- `contact_player_fraction`: 반대 팀3m 안에 적어도 한 명이 있는 서로 다른 선수의 비율.
- `max_cross_cluster_3m`: H-A edge로 연결된 가장 큰 component의 노드 수(고립 노드 제외). 연쇄 연결은 물리적 동일 지점의 작은 군집이 아님.
- `cross_min_m`, `cross_nearest_mean_m`: 최소 상대거리와 양 팀 선수 각각의 평균 최근접 상대거리.
- `home_*`,`away_*`의 중심/폭/길이/compactness: 시간적 구조 변화·차후 점유 팀 추론 용도로 별도 보존.
- 8×6=48 Zone의 `max_mixed_zone`, zone density, concentration: **보조 비교 및 시각화**, 전역 쌍 후보를 zone 경계로 제한하지 않음.

각 지표마다 수학적 valid mask와 실측 품질을 구분한다. 원본 CSV 컬럼의 단위/정의를 검사하고 임의 이름 변경을 피한다.

## 3. 후보 상태 분리
`candidate_state` 세 가지:
- `candidate`: 고정 사전 규칙을 통과한 **후보**.
- `not_candidate`: 관측이 충분하고 해당 규칙에선 후보가 아님. 실제 비경합 정답 아님.
- `insufficient_observation`: 관측 부족·해석 불가능. 점수도 NaN을 허용. 비경합으로 채우지 않음.

추가 `quality_tier` 및 `reason_codes`로 `strict_valid`, low_players, duplicate_suspect, out_of_bounds, temporal_gap, undefined_pair, outside_official_block 등 기록한다. 변수가 부족할 때 fallback의 근거/결측 비중이 보고되어야 한다.

## 4. 1차 threshold: 사전 고정한 비지도 분포 기준
`configs/initial_policy.json`의 임시 시작 조건: **strict-valid Train 고유 프레임**에서 `contested_index` 상위10%(q=0.90)를 `candidate` 기준으로 사용한다. 이 숫자는 사람이 영상에서 라벨링해 얻은 최적값이 아니라 **사전 지정된 상대적 후보 검출량 제어 설정**이다. 임계값은 Train에서 한 번 적합하고 검증·전체 출력에는 고정한다.

시간 안정화: 기본 5개 연속 유효 프레임(25fps에서0.2초), 미상 구간 관통 연결0프레임. 지속시간과 coverage·후보 구간 수를 보고한다. 0.85/0.90/0.95 quantile 및 min_run1/5/10은 **사전 등록 민감도 표만** 출력하고 영상을 보고 이 중 승자를 선택하지 않는다. 하나의 후보 상태를 이미 계산한 후에 보고서나 영상 검토로 특정 장면에 맞춰 상수를 수정하지 않는다.

단일 지수 임계값만으로 경합을 확정할 수 없다. 추후 복합 조건/히스테리시스 여부는 **후보의 물리적 의미·결측 복원 민감도에 근거한 별도 버전 설계**로 기록한다. 무라벨 상태에서 '정답 경합 recall'이라는 지표를 출력하지 않는다.

## 5. 품질 정책 비교
`strict` 기존 지수(엄격 관측)와 `coverage-aware` 보조 후보 분석을 구분한다. 후자는 수학적으로 계산 가능한 최소한의 지표를 유지하되 low reliability로 표시하는 정책이다. 처음부터 다른 분모·임의 보간·확률 보정을 섞지 않는다. **현재 strict에서 평가 범위 70.3%만 유효하므로, 관측 보류 29.7%를 자동 비경합으로 치환하지 않는 것이 핵심**이다.

## 6. 시간 변화 점검
시점별 점수뿐 아니라 동일 반/contiguous run의 `approach -> concentrated -> dispersal` 변화량을 로그로 기록한다. 선수 ID가 없으므로 개인 속도/가속도가 아니라 **집합 통계의 변화**로 명명한다. 계단식 접촉 수는 노이즈에 취약하므로, 집합 구성 변화와 품질 전환을 함께 출력한다. split/half/gap을 넘겨 차분하거나 smoothing하지 않는다.

## 7. 산출물
전체199,319행 `candidate_frames.csv`와 구간 요약 `candidate_segments.csv`(연속적 상태/품질 변경 시 경계), aggregate `quality_and_sensitivity.json`. 출력 칼럼 계약은 `schemas/candidate_frame_columns.json`. 모든 원본 프레임·모델 체크포인트 보존. AI가 라벨링한 정답 파일이나 대회 H/A/D CSV를 만들지 않는다.
