# v2-3 Candidate Decision 사전 감사

2026-11-10. 기존477개 테스트를 먼저 실행하여 **477 passed in31.76s**를 확인했다. 기존245개 코드·문서 SHA256와8,423개 원본·파생 파일 size/mtime을 `/private/tmp/cj_v23_start.json`에 보존했다. 이 감사와 `configs/candidate_v23.json`은 v2-3 구현 및 Validation 후보 분포를 확인하기 전에 작성했다.

## 읽은 구현과 계약

요청한 Stage1/v2-1/v2-2/Stage A/Stage B/B.1/B.2 보고서와 실제 source/feature/profile/집계/진단 코드, 설정·데이터 사전을 확인했다. RULES/FAQ의 제공 경기 수작업 라벨 금지·공개 GT 튜닝 금지·경기/추출물 외부 전송 금지도 재확인했다. 기존 제출 baseline/A/B 모델/원본은 입력이나 변경 대상이 아니다.

v1 `detector.py`는 기존 CausalContestedIndex를 그대로 사용한다. contact/최대 군집/최근접 거리의 공간 지수와 과거25프레임 변화를 결합하며, Train strict q=.90 threshold 및 causal5연속 확인을 사용한다. split/half/gap/QC 경계에서 run을 끊는다. 본 단계는 원래 `candidate_frames.csv`를 비교용으로만 읽고 재적합하지 않는다.

v2-1 `local_clusters.py`는 같은 frame의 H/A 간 inclusive distance≤3m/5m edge만으로 connected component를 만든다. 고립 선수는 군집이 아니다. 3m component는 같은 frame의5m family에 포함되며 family/cluster_id는 영구 ID가 아니다. 원래 coordinate signature에 따른 canonical order는 순열/H/A 교환에 불변이다.

B.1 `pattern_values.py`, `pattern_profiles.py`는 개별 P1~N3 수식과 관측 정책을 유지한다. COMPLETE는 해당 실제 의존 구간 완전 지원 및 원래 구성/기하 조건, EXPERIMENTAL_PARTIAL은 원래 Stage B guard 통과, UNKNOWN은 지원·해석 부족이다. 값0은 unknown이 아니다. A/B의 SUPPORTED는 필요한 Evidence COMPLETE와 raw 순서 확인이며, NOT_OBSERVED는 충분한 관측에서 조건 불충족이다. A는 과거 접근/집중, B는 전체101개 peak/order/구조 재편이다. C/D는 대체적인 기하 설명이며 경합 부재 정답이 아니다.

B.2는 fixed past short−10..0/near−25..0, future short0..10/near0..25를 따로 사용한다. scope 내부 median5 및 모든 lag5 평균 방향을 확인한다. direction 판정은 scope 전체 reliable/finite와 local H/A 수 불변을 요구한다. 같은 row의 rise/fall O/O만 Local Rise→Fall이다. 중간 공백은 연결하지 않는다. Motion은 원래21개 경로의 현재 연결 부분이며 미세 jitter에도 비영 이동을 반환하므로 후보 또는 veto 근거로 쓰지 않는다.

## Spatial Gate — 현재 관측부터 확인

공식 half/record, 현재 frame와 cluster의 Strict QC, 현재 실제 군집 매핑, 양 팀 존재, 실제 해당 radius cross-edge, finite하고 일관된 centroid/R/D/density/count, 현재 Anchor 안 양 팀 관측을 요구한다. 현재 gate 실패는 INSUFFICIENT다. 시간 Evidence가 높아도 승격하지 않는다. perturbation에서 깨끗한 현재 군집을 모호하게 대응시키거나 찾지 못하면 fixed-region 시계열이 있어도 현재 Anchor 매핑이 부족하므로 보류한다.

전체 Frame에 현재 군집 row가 전혀 없을 때는 별도로 처리한다. 공식·Strict·양 팀 관측이 충분하면 현재3/5m mixed graph 없음이 관측됐으므로 NO_DETECTED_SIGNAL+SPATIAL_NO_MIXED_CLUSTER, 시간 평가 NOT_APPLICABLE다. QC 부족/공식 밖이면 INSUFFICIENT다. 이 예외는 공간 부재 관측이며 시간적 비경합 판단이 아니다.

## 고정한 상태 규칙

같은 key `(frame,radius_m,cluster_id)`에서만 결정한다.

1. Spatial gate 실패 → INSUFFICIENT.
2. A 또는 B SUPPORTED → PATTERN_CANDIDATE. A/B 각각과 동시 Local 근거를 별도bit로 남긴다.
3. A/B 미지원이며 같은 scope의 Short/Near Local Rise→Fall → LOCAL_CANDIDATE. 두 scope는 따로 보존한다.
4. 충분한 부재 판정: A/B 모두 COMPLETE+NOT_OBSERVED이고4개 Short/Near Rise/Fall 모두 완전 지원+NOT_OBSERVED일 때만 NO_DETECTED_SIGNAL.
5. 나머지 유효 공간 군집은 REVIEW. rise-only/fall-only/다른 scope의 부분 변화/partial Evidence/전체 패턴 unknown/시간 부족/공간만 관측 등을 reason bit로 남긴다.

개별 positive Evidence 값이 양수라는 이유만으로 후보를 만들지 않는다. C/D, N1/N2/N3, Local Proximity/Motion은 진단용이며 어떤 값도 후보를 취소하지 않는다. 새 점수·가중치·확률·물리적 threshold를 도입하지 않는다. 동결 입력의 SUPPORTED/Local RF와 지원 metadata가 모순되면 조용히 신뢰하거나 수정하지 않고 입력 계약 오류로 중단한다.

## Frame 집계·중복·시간 의미

완성된 Anchor 상태만 집계한다. 우선순위는 PATTERN > LOCAL > REVIEW > INSUFFICIENT > NO_DETECTED_SIGNAL로 고정한다. 다른 Anchor가 평가 불가인데 Frame 전체의 부재를 확정하지 않도록 INSUFFICIENT를 NO보다 앞에 둔다. 동률은 작은radius/cluster_id로 정하고 점수로 선택하지 않는다. 대표 근거·reason은 선택한 한 row에서 가져온다. Radius별 출처/상태/후보 view 수와 전체 상태별 view 수를 보존한다. 서로 다른 군집의 증거를 합성하지 않는다.

Radius 중복은 기존 frame-local3m→5m family와 same-node-set 관계를 진단으로만 사용한다. 후보 view 수·후보 frame 수는 사건 수가 아니다. ID를 연결하는 episode/spell stitching은 구현하지 않는다.

본 알고리즘은 offline이다. 후보의 기록된 근거 A/B/Short/Near에 각각0/50/10/25프레임의 미래 의존성을 표시한다. 여러 근거를 보존하면 그 최대 범위를 기록한다. REVIEW/충분한 동적 부재 평가의 이론적 최대 범위는50, 공간 실패/관측된 empty graph는0이다. A만으로 판정된 row도 전체 파이프라인은 offline이며 실시간 보장을 하지 않는다. v1은 고정 Train calibration 아래 현재·과거만 사용하는 causal decision이라는 차이를 명시한다.

## 검증·분석 계획

합성20가지 상황과 source permutation/team swap, current dropout, 경계, 다른 Anchor 혼합 금지, 높은 Alternative의 비veto를 검증한다. 전체 output199,319/789,005 정렬, old key 정확한 join 및 input hash/mtime 보존을 확인한다.

Validation112,286개에서 상태·근거·QC/radius·원래 COMPLETE/PARTIAL 구성 및 unknown reason을 집계한다. v1과는 같은 Validation Frame축의 교차표·후보 frame coverage·연속 frame run 길이만 비교한다. run은 half/split/gap에서 끊는 기술 통계이며 episode가 아니다. Precision/Recall/F1은 계산하지 않는다.

Stage A1,778 cohort/Train donor/seed와 B.1/B.2의12×151 tracking support를 재사용한다. 동일 Anchor 상태 전환, 후보 발생/소실, gate/지원 집합 변화와 공통 유효 교집합을 분리하고3/5m별로 기록한다. 모호한 현재 매핑은 보류한다. Validation 결과가 예상과 달라도 규칙을 바꾸지 않는다. Test/GT/볼/SmartTag/영상 라벨, 외부 전송, 학습, H/A/D/X, Commit/Push는 사용하지 않는다.
