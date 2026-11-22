# 03 — 테스트와 성능 보고(정답 없는 상태)

## 반드시 통과할 합성 테스트
1. 3/5/10m 정확히 경계에 있는 H-A 쌍의 포함 여부.
2. 3m 내 상대가 있는 **고유 선수 수** 계산(쌍 수와 혼동하지 않음).
3. H-A로만 연결된 bipartite component, 연결 사슬과 동일 지역 집단의 차이.
4. 48 Zone 경계를 양쪽으로 넘는 근접1m 쌍도 **전역** 관계에는 포함.
5. 선수 배열 순열 무관 출력; H/A 교환 시 대칭 점수 동일, 팀별 개별 값 교환.
6. 비어있는 팀, 한 명뿐인 팀, 자연 결측·NaN·bounds/duplicate 의심, 전부 미관측: `insufficient_observation`과 NaN/validity 유지.
7. 팀 8→7명 등 strict→low coverage 전환은 0 경합으로 바꾸지 않음.
8. 현재 frame 변경이 과거 **causal** 후보 점수/상태를 바꾸지 않음.
9. half/official block/record gap 경계에서 시계열 파생값·연속 후보 구간 끊김.
10. 같은 seed·분포 산출·설정으로 결정적 재현. Train quantile 계산에 Validation/Test 포함하지 않음.
11. 기존 `CausalContestedIndex` 결과와 해당 strict baseline 수치가 합치하는지.
12. 전 프레임 199,319개, ID 0..199318 중복/누락 없고 공식 H/A/D/X 레이블을 생성하지 않음.
13. 볼 필드/SmartTag decoy/공개 GT를 바꾸거나 아예 제거해도 동일 candidate 출력.
14. 전체 코드가 외부 네트워크·업로드·Git push를 사용하지 않고 기존 소스/결과를 변경하지 않음.

## 관측 오류 강건성: 정답이 없으므로 안정성만 측정
원본 데이터 자체를 수정하지 않고 별도 in-memory source wrapper로 합성 perturbation을 만들어 감도 분석. 동일 seed, 동일 frame support, 동일 조합을 비교한다.
- xy jitter Gaussian SD 0.1/0.5/1.0m
- 관측선수 random dropout 10/20/30%
- 특정 짧은 구간 관측 구성 20% 변경
- Home 전체1m 이동(실제 공간 변화 대조) — 다른 perturbation과 물리적 강도가 동등하지 않음을 명시
- frame별 선수 순열·H/A swap

관찰 지표: 후보 점수 변화 p50/p95, candidate↔noncandidate 변화 수, `insufficient_observation` 전환 수, 총 후보 coverage, 구간 fragmentation, 관측 인원/품질 변화에 대한 조건부 민감도.

**주의:** 원본 추적 자체가 완전한 정답이 아니다. 여기서 '정확도', '실제 경합 F1'은 계산할 수 없다. 좌표에 실제 이동을 가한 대조에서는 점수 변화가 자연스럽고 무조건 불변이 바람직하지 않다.

## 다른 경기 정답 라벨을 나중에 받을 경우
외부 경기 데이터/영상 접근 가능성, 사용권, 주최 측 감사 가능성을 우선 확인한다. MTAH 제공 경기에 수작업 경합 라벨을 붙이지 않는다. 외부 정답을 도입한다면 본 Phase 코드와 명확히 분리하고, 별도 도메인 교란/홀드아웃 설계를 만든다.

## 재현·보고
- 원래 코드를 건드리지 않은 상태에서 pytest 회귀 테스트를 우선 실행.
- candidate 임계값을 Train에서 고정한 뒤 Validation에서는 분포/강건성 **진단**만 진행. 공개 GT/영상 수작업 기준으로 threshold 탐색 금지.
- 출력 경로는 `data/derived/competition_judgement/` 아래에 run별 별도 폴더 사용. 기존 output 덮어쓰기 금지.
- 기여도와 한계 보고: 정상 vs 저품질, zone vs 전역 거리, 접근 vs 근접, threshold sensitivity, unknown 비율.
- 최종 H/A/D `spellF1` 개선은 **연동 구현 후** 별도 공식 채점·규정 준수 절차에서만 판단 가능하다. 현재 후보 지수를 강제로 D나 H↔A로 변환하지 않는다.
