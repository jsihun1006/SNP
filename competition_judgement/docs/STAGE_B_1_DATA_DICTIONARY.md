# Stage B.1 Pattern Profile 데이터 사전

메인 결과는 `evidence_pattern_b1_final_20261110/cluster_pattern_profiles.csv.gz`와 `frame_pattern_profiles.csv.gz`다. 실제 경기 파생물이므로 로컬에서만 연다. JSON 사전은 같은 결과 폴더의 `data_dictionary.json`이다.

## 공통 정보와 mask 참조

군집 key는 `(frame,radius_m,cluster_id)`, frame은 원본0-based다. ID는 해당 frame 안에서만 의미가 있다. `period/split`은 기존 계약이다. `cluster_centroid_x/y`, `anchor_radius_m=max(6,D+3)`가 고정 공간 원을 지정한다. 기존 `n_players/n_home/n_away`는 현재 component 관측 수이며 `current_anchor_home/away`는 고정 원 안 관측 수다. `current_edges/current_soft`는 해당 원의 관측 관계다. 둘의 모집단이 다르다.

Masks는 bit-order little의 hex **문자열**이다. −50..+50의101비트는13bytes/26chars, N2 path는−10..+10의21비트3bytes/6chars다. leading0을 버리지 않는다. 같은 mask를 특징마다 복사하지 않고 의존 구간을 참조한다.

| mask | 의미 |
|---|---|
| record_mask | 같은 half/split/연속 기록 지원 |
| observed_mask | Anchor 안 양 팀 관측 있음 |
| reliable_mask | 원래 v2-2의 현재/offset QC·전역 인원 동일 조건 |
| strict_mask | record 지원 중 Strict QC |
| global_changed_mask | 알려진 record에서 현재와 다른 전역 H/A 관측 수 |
| local_changed_mask | observed 시점에서 현재와 다른 Anchor H/A 수 |
| path_mask | N2의 별도 기하 corridor 지원. 고정 원 reliable mask와 다른 정보 |

숫자·문자열 mask는 결측 관측을 복원하는 feature가 아니다. 현재 Strict는 `current_strict`, path/matching의 원래 원인과 상태는 `path_reason`, `prev_match/next_match`다.

## Individual Evidence

P1/P2/P3/P4/N1/N2/N3는 원래 Stage B의 무차원[0,1] 수식이다. N1–N3는 Alternative Evidence이며 실제 비경합이 아니다. `_state`, `_reason`, `_reliable_n`, `_max_gap`, `_composition_flag`가 각 특징에 붙는다. 상태 UNKNOWN이면 값은 NaN이다. 계산된0은 관측된 수학적0이다.

| 특징 | 필수 offsets | mask basis |
|---|---|---|
| P1 | −50..0 | reliable_mask |
| P2 | −25..0 | reliable_mask, finite component geometry≥20 |
| P3 | −25..25 | reliable_mask, 인접±1 matching 포함 |
| P4 | 0..50 | reliable_mask, 알려진 future end/Window censor |
| N1 | −50..50 | reliable_mask, 기존 연속 하한/구성 조건 |
| N2 | −10..10 | **path_mask**, reciprocal-unique corridor |
| N3 | −50..50 | reliable_mask |

`_reliable_n/_max_gap`은 이 범위에서의 개수/최대 연속 False 길이다. mask의 범위 slice로 결측 위치와 QC를 읽는다. `_composition_flag` bit0=알려진 전역 수 변화, bit1=observed 국소 H/A 수 변화다. flag0도 gap 속 구성이나 인물 동일성의 증명이 아니다.

| Observation | 의미 |
|---|---|
| COMPLETE | 필요한 범위와 feature별 기존 정보 조건이 충족됨. 전체101 Full과 다름 |
| EXPERIMENTAL_PARTIAL | 필요한 범위가 불완전하지만 동결 Stage B Partial 조건 통과. 실험용 |
| UNKNOWN | 부족/품질/구성/기하 모호성으로 계산 불가. NaN |

`available_mask` bits0..6은 P1,P2,P3,P4,N1,N2,N3 순서다. 크기 high/low가 아니라 확보 여부만 저장한다. `legacy_101_full`은 비교용 원래101 지원 flag다.

P3는 `P3_k_change`, `P3_density_change`, `P3_branch_term`, `P3_matching_ambiguity`를 따로 둔다. P4의 원래 optional 항목은 `P4_term_mask` bits0..3=soft/distance/known-end/expansion이다. 이 내부 term 변동은 기존 수식의 특성이므로 보존했다.

`close_duration_lower_bound_s`, `onset_censored/end_censored`, `future_end_reason`는 기존의 **관측된 연속 지속 하한과 검열 정보**다. 새 Score Bounds를 만든 것이 아니다. `soft_range/std/occupancy`, `approach_s/m`, `future_drop_s`, `radius/diameter_contract_m`, `edge_growth`는 해석용 원시 통계다.

## Pattern Profile

| 코드 | 필수 Evidence | 검증할 기하 양상 |
|---|---|---|
| A | P1,P2 | 접근 또는 국소 집중. `A_convergence`와 `A_concentration` 분리 |
| B | P1,P3,P4 | 중심 부근 peak 앞의 실제 rise와 이후 fall, 실제 K/E 변화 |
| C | N1,N3 | 연속 근접·작은 S/d 변동·K/E 구조 유지 |
| D | N2 | 명확한 corridor의 centroid 이동·shape 유지·현재 원 안 scope |

각 코드에 `_obs/_seq/_outcome/_reason/_missing_mask`를 둔다. required bitset은 사전에 고정되어 행마다 반복하지 않는다. `_missing_mask = required & ~available_mask`. 현재 필요한 값이 있어도 N2 경로가 현재 고정 원 밖이면 D의 observation은 UNKNOWN일 수 있다. 이때 missing mask0과 scope reason이 함께 의미를 설명한다.

| Sequence | 의미 |
|---|---|
| VERIFIED | 사전 고정한 기하 시계열 조건 충족. 실제 축구 사건 검증 아님 |
| UNVERIFIED | 자료/순서 미확인 또는 정의한 기하 조건 불충족. outcome/reason과 함께 읽음 |
| NOT_APPLICABLE | Frame의 관측된 empty mixed graph처럼 현재 Anchor 자체가 없는 경우 |

| Outcome | 의미 |
|---|---|
| SUPPORTED | 같은 Anchor에서 패턴의 필수 관측과 기하 조건 확인 |
| NOT_OBSERVED | 필요한 자료는 충분하나 정의된 기하 조건이 나타나지 않음 |
| UNKNOWN | 필요한 관측이나 시간 검증이 부족함. 패턴 부재 아님 |

EXPERIMENTAL_PARTIAL은 VERIFIED로 자동 승격하지 않는다. A–D는 비배타적이다. A/B의 동시 관측, C/D의 동시 관측이 가능하다. 대표 한 유형을 강제하지 않는다.

`B_peak_offset/B_rise_end_offset/B_fall_start_offset`은101개 지원이 있을 때만 기록한다. `B_structure_changed`는 실제 K/E 변동이며 matching ambiguity와 별개다. `B_reapproach`는 기존 future off→on 관측 flag다. `path_motion_m/path_*_range`는 기존 N2 지표, `path_inside_anchor`는 profile 결합 범위다. 원래 시계열은 복제 저장하지 않고 read-only cache에서 재구성한다.

## Reason code

| 코드 | 의미 |
|---|---|
| OK | 지원/기하 조건 충족 |
| NO_RECORD / CURRENT_UNOBSERVED / CURRENT_QC | record/현재 양팀 관측/현재 QC 부족 |
| BAND_SUPPORT / CENTER_GAP / LONG_GAP | 기존 Partial guard 실패 |
| GLOBAL_COMPOSITION / LOCAL_COMPOSITION | 알려진 인원 구성 변화 |
| GEOMETRY / MATCHING / PATH / UNDEFINED | 필요한 기하·대응·수식의 정의 부족 |
| EVIDENCE_UNKNOWN | 원시 시계열 없는 입력에서 값이 qualified되지 않았음 |
| END_AMBIGUOUS | 실제 observed 종료와 공백/이탈을 구분 불가 |
| MISSING_EVIDENCE | 패턴 필수 Evidence UNKNOWN |
| SEQUENCE_PARTIAL / SEQUENCE_UNVERIFIED | 시계열 완전 지원/검증 부족 |
| NO_TIME_ARRAY | Evidence-only 입력에 원시 시간 배열이 없음 |
| CONDITION_NOT_MET | 충분한 자료에서 사전 기하 조건 불충족 |
| ANCHOR_SCOPE_OR_PATH_UNKNOWN | N2 경로가 모호하거나 seed 원 범위를 벗어남 |
| ANCHOR_SUPPORTED | Frame에 실제 supported source 있음 |
| ALL_ANCHORS_NOT_MET | 해당 Frame의 모든 현재 Anchor가 충분하지만 조건 불충족 |
| ANCHORS_UNKNOWN | 평가 불가 Anchor가 남아 있어 Frame 부재를 확정할 수 없음 |
| NO_OBSERVED_MIXED_ANCHOR | Strict 관측에서 현재 mixed graph가 비어 있음 |

이는 주된 사유 코드다. 모든 중복 원인과 실제 누락 원인을 확정하는 분류가 아니다.

## Frame과 Legacy

Frame은0..199318 전체 순서다. 각 패턴의 `_source_radius_m/_source_cluster_id`는 같은 frame의 Long key를 참조한다. 지원 Anchor가 여러 개면 작은 radius/ID를 표시용으로 선택한다. 다른 패턴은 다른 source를 가질 수 있으며 그것들을 한 사건으로 묶지 않는다.

`available_3m/5m_views`, `supported_3m/5m_views`는 radius별 관측 view 수다. 합을 실제 경합 횟수로 해석하지 않는다. `supported_pattern_types_n`도 유형 존재 개수이며 점수나 사건 수가 아니다.

지원 source가 없고 어떤 Anchor라도 평가 불가이면 Frame은 UNKNOWN이다. 모든 Anchor가 충분하나 조건 불충족이면 NOT_OBSERVED다. Strict한 empty graph는 NO_OBSERVED_MIXED_ANCHOR로 분리한다. 기존 v1 상태를 패턴 정답으로 사용하지 않는다.

기존 P/N 평균과 사용 항목은 `legacy_diagnostics.csv.gz`에 key별 `legacy_` 접두사로만 복사한다. 메인 profile에는 종합 P/N·차감·고정 점수·확률·Score Bounds가 없다. Legacy 값은 profile 계산 경로로 들어가지 않는다.
