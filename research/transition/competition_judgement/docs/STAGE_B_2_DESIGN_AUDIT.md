# Stage B.2 사전 설계 감사

2026-11-10. 구현 전 기존 테스트 **440 passed in 31.35s**를 확인했다. B.1 source/values/profiles/frames/diagnostics/runner/verifier/config, Stage A/B missingness·temporal 정의와 보고서, B.1 보고서·사전·RULES.md를 확인했다. 기존 235개 코드/문서 SHA256와 8,408개 원본/파생 파일 size/mtime을 `/private/tmp/cj_b2_start.json`에 기록했다. 기존 사용자 변경도 보존한다.

## 기존 정책과 해석

현재 `PatternSource`는 읽기 전용 player/pair cache와 고정 Anchor를 사용한다. record는 같은 half/split/연속 기록, observed는 원 안 양 팀 존재, reliable은 observed와 offset/current Strict QC 및 현재 대비 전역 H/A 관측 수 일치다. 현재 Anchor는 `max(6m,현재 diameter+3m)`다. 같은 수는 같은 선수라는 뜻이 아니다. 미관측은 Tracking 누락/영역 이탈을 구분할 수 없다.

B.1은 필요한 Evidence의 관측 상태와 실제 시간 순서를 분리한다. A/B/C/D, P1..N3와 기존 Full/Partial의 정의·결과는 변경하지 않는다. B의 전체101개 peak/구조 검증, C의 엄격한 전체101개 안정성은 Local 변화와 별개다. D/N2의 기존21개 reciprocal-unique 경로는 고정 원 밖으로 나갈 수 있으며 B.1은 결합 시 이를 보류한다.

Stage A/B에서 평균 변화도 결측 위치에 민감했고, 현재 인접 공백은 지속 하한을 크게 줄였다. 이를 Partial 정책 완화로 해결하지 않는다. B.2는 **정해진 더 짧은 의존 구간의 완전 관측** 또는 명시적인 연속 하한을 기록한다.

## 사전 고정 수식

설정은 `configs/local_temporal_b2.json`에 실제 구현·Validation 전에 고정했다. 25fps, scopes는 PS=-10..0, PN=-25..0, FS=0..10, FN=0..25다. 현재 포함11/26개이며 sample span은0.4/1초다. 유리한 구간을 검색하지 않는다.

L1/L2: 해당 scope 전체 reliable, finite S/d, 현재 Strict QC, 국소 H/A 관측 수 불변을 요구한다. 이는 지역 구성 변화를 같은 관계의 접근/분리로 오해하지 않기 위한 보수적 보류이며 선수 동일성을 보증하지 않는다. scope 안에서만 trailing median5를 만든다. 첫 median과 마지막 median의 차이 ΔS, Δd 및 **모든** lag5 차이의 평균을 기록한다. Rise는 ΔS≥.02, Δd≤−.1m와 lag 평균 방향 일치, Fall은 반대다. 값은 기존 deadband에서 가져온 기하 조건이며 오차 보증이 아니다. 조건 불충족은 NOT_OBSERVED, 지원 부족은 UNKNOWN이다. PS 비교의 첫 median은−10..−6, 마지막은−4..0, FS는0..4와6..10이다. PN/FN도 동일한 고정 끝점 원칙이다.

L3: 네 scope 각각 전체 reliable일 때 radius별 E>0가 모든 시점에 성립하면 OBSERVED다. 충분히 관측됐으나 중간 known E=0이면 NOT_OBSERVED다. 불완전하면 UNKNOWN이되, 현재부터 연결된 reliable+close prefix의 관측 수/첫끝 offset/기간 하한 n/25/검열 및 S,d,K,E range는 별도로 보존한다. 공백 이후를 연결하지 않는다. 현재가 unknown이면 하한도 NaN, 현재 known non-close이면0이다. 지속 근접에 구조 안정성을 요구하지 않으며 range가 크다는 사실을 따로 남긴다.

L4: 원래21개 링크의 **현재에 연결된 연속 부분**만 쓴다. 각 offset의 원래 frame/half/split/QC/global/component H/A 수 조건과 seed 원 내부 조건을 유지한다. 원래 경로가 끝난 뒤 다른 군집에 연결하거나 공백 너머로 뛰지 않는다. 2개 이상일 때 순변위/누적 경로 길이(m), sample span(s), R/D/density range를 기록한다. 길이가 수치 tolerance보다 크면 이동이 관측됐다는 의미의 OBSERVED, 정지하면 NOT_OBSERVED다. 1m 또는 shape 안정 한계를 짧은 경로에 적용하지 않는다. jitter도 이동을 만들 수 있으므로 실제 같은 선수의 동반 이동·경합으로 해석하지 않는다.

## 출력·순서·검증

Short/Near의 Rise/Fall 3×3 조합을 같은 row/Anchor 안에서만 독립 sequence로 저장한다. 전체 pattern을 승격하지 않는다. L3도 past/future를 구분한다. scope의 reliable 수/첫끝/최대 gap/연속성/구성 변화/reason, 원시 기하 변화와 matching ambiguity를 기록한다. masks/원래 Evidence/시간 배열을 복제하지 않는다.

Frame은199,319행이며 각 signal별 출처를 둔다. Rise-then-Fall은 군집 row에서 먼저 만들고 집계하므로 다른 Anchor의 Rise/Fall을 합칠 수 없다. radius별 view 수를 사건 수로 합산하지 않는다. 관측된 Anchor가 없는 Frame은 Local 평가 UNKNOWN+NO_CURRENT_ANCHOR다.

Validation만 분포/회수/민감도에 사용한다. Stage A1,778 cohort와 seed·동일 삭제, B.1 tracking support를 재사용한다. 동결 B.1 결과를 전체 key·P1..N3·A/B/C/D와 read-only 재계산 대조한다. 설정은 결과에 맞춰 재조정하지 않는다. 합성 기대는 수학적 동작이며 축구 정답이 아니다. 볼/태깅/GT/영상 라벨, 학습/후보/v2-3/점유/Git/외부 전송은 하지 않는다.
