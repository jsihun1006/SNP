# v2-3 — Contested-Ball Candidate Decision 실행 보고서

2026-11-10. 설계 감사, 독립 구현, 전체 로컬 실행, Validation 진단 및 보존 검증을 완료했다. **동일한 Local Cluster Anchor에서 관측된 기하 패턴을 근거로 검토할 후보를 선정한다.** 실제 볼 경합 사실·확률·점유 라벨을 예측한 결과가 아니다.

## 1. 완료 결과와 전체 구조

`현재 공간 군집 → Spatial Gate → 동결 A/B 및 Local Rise→Fall → Anchor 상태 → 출처를 보존한 Frame 집계` 순서다. v1 점수·Alternative Evidence는 판단의 정답이나 veto로 사용하지 않는다.

- 전체 **199,319 Frame / 789,005 군집 관측**과 기존 key 순서를 유지했다.
- Validation112,286개 군집에서 **PATTERN8,850 / LOCAL3,503 / REVIEW74,313 / NO_DETECTED_SIGNAL1,926 / INSUFFICIENT23,694개**였다.
- Validation23,828개 Frame 중 후보는 **6,139개(25.76%)**였다. 실제 독립 경합 사건 수가 아니다.
- 기존477개를 포함해 **522개 테스트 통과**(신규45개,32.92초). 최종 실패0개다.
- 기존245개 코드·문서 SHA256 및8,423개 원본·기존 산출물 size/mtime이 유지됐다. 주요 동결 입력5개도 전체 SHA256이 동일했다.
- 전체 실행·진단은 **52.89초**, 이후 별도 검증에서 모든 군집 판정과 Frame/Radius 출처를 다시 계산했다.

설정은 [candidate_v23.json](../configs/candidate_v23.json), 사전 감사는 [STAGE2_3_DECISION_DESIGN_AUDIT.md](STAGE2_3_DECISION_DESIGN_AUDIT.md)다. 두 파일은 v2-3 구현과 Validation 후보 분포를 보기 전에 작성했다. 규칙을 결과에 맞춰 재조정하지 않았다.

## 2. 현재 Spatial Gate

다음 조건을 모두 요구한다.

1. 현재 실제 군집 record가 공식 half 안에 존재한다.
2. 현재 frame/군집 및 동결 profile의 Strict QC가 유효하다.
3. 현재 군집이 명확히 존재한다. perturbation의 fixed clean probe에서는 명확한 현재 군집 bridge가 있어야 한다.
4. 현재 군집에 Home/Away가 각각 존재한다.
5. 해당 radius의 **실제 cross-edge 수가 양수**다.
6. centroid/R/D/density/contact가 finite이고 count·geometry가 일관된다. 관측 수는 비음수 정수, 전체=H+A, edge≤H×A, R≥0/D≥R, density/contact는 원래 범위 안이다.
7. 현재 고정 Anchor의 양 팀 관측·soft/edge가 유효하다.

어느 조건이든 실패하면 INSUFFICIENT다. 과거·미래의 높은 지표로 현재 gate를 우회하지 않는다. 실패 mask는 복수 원인을 보존하고 primary reason은 고정 순서로 정한다. 새 성능 최적화용 공간 threshold는 도입하지 않았다.

**현재 mixed graph가 없는 Frame은 별도 처리한다.** 공식·Strict·양 팀 관측이 충분하고 v2-1의3/5m 군집 수가 모두0이면 `NO_DETECTED_SIGNAL / SPATIAL_NO_MIXED_CLUSTER`, 시간 평가는 `NOT_APPLICABLE`이다. QC 부족이나 미관측을 같은 부재로 처리하지 않는다. 군집이 없으면 cluster sidecar row를 인위적으로 만들지 않는다.

## 3. 다섯 상태의 정확한 정의

| 상태 | 조건과 의미 |
|---|---|
| PATTERN_CANDIDATE | Spatial Gate 유효 + 동결 A 또는 B SUPPORTED. 접근/집중 또는 전체 일시적 재편의 기하 패턴을 선정 근거로 사용 |
| LOCAL_CANDIDATE | Gate 유효, A/B 미지원 + 같은 Anchor의 Short 또는 Near Rise/Fall 모두 OBSERVED이고 해당 scope 완전 지원 및 같은 row의 sequence 확인 |
| REVIEW | Gate 유효지만 위 후보 근거 또는 충분한 동적 부재 판정이 없음. 일부 관측·시간 지원 부족·공간만 확인된 상태를 보존 |
| NO_DETECTED_SIGNAL | Gate 유효 + A/B 모두 COMPLETE/NOT_OBSERVED + 네 가지 Short/Near Rise/Fall 모두 완전 지원/NOT_OBSERVED. 관측 부족에는 부여하지 않음 |
| INSUFFICIENT | 현재 record/QC/군집 매핑/geometry/핵심 현재 관측 부족으로 Anchor 후보 판단 불가 |

REVIEW는 높은 경합 가능성을 뜻하지 않는다. NO_DETECTED_SIGNAL도 실제 비경합 정답이 아니다. 공간적으로 확인된 empty graph는 앞 절의 Frame 예외이며, 시간적으로 충분히 관측된 비경합 사례로 취급하지 않는다.

판정 우선순위는 현재 gate → A/B → 같은 scope의 Local RF → 충분한 동적 부재 → 그 밖의 REVIEW다. REVIEW와 부재 사이에서는 **충분한 부재 조건을 먼저 확인**하며, 조건 불충족/unknown을 분리한다.

## 4. 동일 Anchor 근거와 Alternative 처리

기본 key는 `(frame,radius_m,cluster_id)`다. B.1/B.2를 key·순서가 정확하게 일치하는 상태로 결합한다. SUPPORTED인데 COMPLETE/VERIFIED가 아니거나 Local RF와 실제 Rise/Fall 상태·지원 metadata가 모순되면 입력 계약 오류로 중단한다. 원래 결과를 조용히 수정하지 않는다.

`observed_basis_mask`는 A/B/Short RF/Near RF의 독립 bits1/2/4/8이다. `decision_basis_mask`는 현재 gate를 통과한 후보의 근거다. A와Local이 함께 관측되면 대표 상태는PATTERN이지만 둘 다 남는다. Short/Near를 하나의 유리한 scope로 합치지 않으며, 서로 다른 scope의 Rise/Fall도 RF로 만들지 않는다.

N1/N3는 지속·안정 근접의 참고 Evidence, C/D는 대체적인 기하 Pattern, N2/Local Motion은 진단 정보다. **어떤 값도 후보를 자동 제거하거나 상쇄하지 않는다.** 실제 경합에서도 지속 근접할 수 있고, 기하 matching/비영 centroid 이동은 동일 선수의 동반 이동을 증명하지 않기 때문이다.

실제로 후보와 D가 함께 SUPPORTED인 군집은 PATTERN에서2,591개, LOCAL에서1,534개였다. 이들을 제거하지 않았다. N1/N2/N3의 상태별 유효 수·중앙값/P90은 `alternative_evidence_diagnostic.csv`에 있다. 이 분포를 이용해 veto나 새 cutoff를 고르지 않았다. C는 기존대로 Validation SUPPORTED0개다.

## 5. Frame 집계와 Radius 중복

완료된 각 Anchor 상태의 우선순위는 **PATTERN > LOCAL > REVIEW > INSUFFICIENT > NO_DETECTED_SIGNAL**이다. 한 Anchor가 평가 불가인데 다른 Anchor의 known absence만으로 Frame 전체를 부재로 단정하지 않도록 INSUFFICIENT를 NO보다 앞에 둔다. 동률은 작은 radius, 작은 canonical cluster_id 순서다. 점수의 최댓값을 선택하지 않는다.

Frame에는 상태별·radius별 Anchor view 수, 후보 존재 여부, radius별 대표 출처와 reason, 전체 대표 Anchor의 key/근거/reason, 현재 QC·양 팀 관측 및 불충분 여부를 기록한다. 다른 군집의 A/Rise/Fall/P1/P4를 합쳐 새 후보를 만들지 않는다.

Validation3m39,616개/5m72,670개 view를 유지했다. 후보는3m4,694개,5m7,659개다. 기존3m→5m family의 같은 frame 기하 관계를 사용한 진단에서:

- 같은 node set인3m view는24,394개다.
- 3m view와 대응5m parent가 모두 후보인 관측은4,020개다.
- 한5m family에 여러3m component가 포함될 수 있으므로 이를 독립 사건으로 더하거나 강제로 하나의 사건으로 합치지 않는다.
- `radius_overlap_diagnostic.csv`는 상태 조합·same-node 여부·고유 parent 수·centroid 간 거리 집계다. centroid 거리는 진단일 뿐 후보 조건이 아니다.

cluster_id/family_id는 시간적 영구 ID가 아니다. episode/spell stitching을 구현하지 않았다.

## 6. Offline 및 미래 정보 범위

이 파이프라인은 **Offline Decision**이다. 현재를 포함한101개 Window와 미래 관측을 사용하는 B.1/B.2를 입력으로 받는다. 실시간 또는 인과적 예측 모델이라고 표현하지 않는다.

보존된 후보 근거의 최대 미래 의존성은 A0/B50/Short RF10/Near RF25프레임이다. 여러 근거가 있으면 최대 범위를 기록한다. `decision_future_frames`와 `decision_uses_future`는 **그 row의 기록된 후보 근거 의존성**이며 파이프라인이 미래 데이터를 전혀 읽지 않았다는 뜻이 아니다. A만으로 지원되는 후보의0도 streaming 실행 보장은 아니다.

REVIEW/충분한 동적 부재 평가의 이론적 최대 budget은50프레임이다. 해당 미래가 실제로 모두 관측됐다는 뜻은 아니며 unknown 이유는 별도 보존한다. 현재 공간 실패나 관측된 empty graph의 판정 budget은0이다.

v1은 고정 Train calibration 아래 현재와 과거25프레임을 사용하는 causal decision이다. v2-3과 정보 예산·선택 목적·확인 지연이 다르다.

## 7. Validation 후보 분포와 판단 이유

기존 Validation은 전반66811..78730/후반166381..178288이며 변경하지 않았다. 아래 군집 표의 분모는112,286개 **반복 Anchor view**다.

| 상태 | 군집 수 | 비율 | Frame 수 | Frame 비율(분모23,828) |
|---|---:|---:|---:|---:|
| PATTERN_CANDIDATE | 8,850 | 7.88% | 4,819 | 20.22% |
| LOCAL_CANDIDATE | 3,503 | 3.12% | 1,320 | 5.54% |
| REVIEW | 74,313 | 66.18% | 10,941 | 45.92% |
| NO_DETECTED_SIGNAL | 1,926 | 1.72% | 202 | .85% |
| INSUFFICIENT | 23,694 | 21.10% | 6,546 | 27.47% |

군집 후보12,353개(11.00%)와 Frame 후보6,139개(25.76%)는 분모가 다르다. Frame의202개NO 중152개는 현재 관측된 empty graph,50개는 모든 Anchor의 충분한 동적 부재에 의한 집계였다.

| 근거/주요 reason | 관측 수 |
|---|---:|
| A만 지원 / B만 지원 / A+B | 8,381 / 209 / 260 |
| A 근거 전체 / B 근거 전체 | 8,641 / 469 |
| Short RF 관측 전체 / Near RF 관측 전체 | 1,649 / 4,772 |
| LOCAL 대표: Short만 / Near만 / 둘 다 | 813 / 2,321 / 369 |
| REVIEW: Rise, Fall 미확인 | 17,470 |
| REVIEW: Fall, Rise 미확인 | 20,916 |
| REVIEW: 서로 다른 scope의 부분 변화 | 1,111 |
| REVIEW: Partial dynamic Evidence | 1,770 |
| REVIEW: 전체 Pattern 미확인 | 33,000 |
| REVIEW: 모든 시간 정보 unknown인 Spatial-only | 46 |

근거별 수는 중복될 수 있으며 전체 후보 수에 단순 합산하지 않는다. REVIEW primary reason과 별개로 복수 reason bits/원래 unknown 사유를 보존했다.

## 8. 관측 신뢰성·COMPLETE/PARTIAL

현재 Strict88,592개 군집 중PATTERN8,850/LOCAL3,503/REVIEW74,313/NO1,926개였다. 현재Low-quality23,694개는 모두INSUFFICIENT였다. 현재 공간은 유효하지만 시간 지원만 부족하면 REVIEW이며 INSUFFICIENT로 모두 묶지 않는다.

PATTERN 후보의P1은8,850개 모두COMPLETE였다. A/B의 필요한 근거가 다르므로 모든P1~P4가 동시에COMPLETE일 필요는 없다. 예를 들어PATTERN 후보의P4는COMPLETE3,138/Partial28/UNKNOWN5,684개이며, A만으로 성립한 후보의 불필요한 미래를 요구하지 않았다.

LOCAL 후보는 원래P1이COMPLETE816/Partial356/UNKNOWN2,331개, P4가COMPLETE1,044/Partial48/UNKNOWN2,411개였다. 이들을 Full Pattern으로 승격하지 않았으며, 후보 근거는 **실제로 완전 관측된 해당 Short/Near의 RF**다. Partial 개별 Evidence의 크기만으로 후보가 된 row는 없다.

REVIEW의 원래 unknown 사유는 B:MISSING_EVIDENCE64,964회, A:MISSING_EVIDENCE57,694회, P3:GLOBAL_COMPOSITION29,027회, P4:GLOBAL_COMPOSITION27,909회, P4:LOCAL_COMPOSITION22,716회 등이었다. 사유는 중첩되므로 합산하지 않는다. `upstream_unknown_reason_summary.csv` 및 cluster의 `upstream_unknown_reasons`에서 추적할 수 있다. 원래UNDEFINED/BAND_SUPPORT 등의 이름을 보존하며, 이를 위치 정답의 오류라고 새로 분류하지 않는다.

3m의PATTERN/LOCAL/REVIEW/NO/INSUFFICIENT는3,074/1,620/25,577/313/9,032개, 5m는5,776/1,883/48,736/1,613/14,662개다. 모집단·Anchor 크기가 달라 통제된 radius 우열 비교가 아니다.

## 9. v1 일치·불일치와 연속 Frame

같은 Validation23,828개 Frame의 교차표다. **v1은 정답이 아니다.**

| v1 상태 | v2-3 PATTERN | LOCAL | REVIEW | NO | INSUFFICIENT |
|---|---:|---:|---:|---:|---:|
| candidate | 346 | 151 | 1,377 | 0 | 0 |
| not_candidate | 4,473 | 1,169 | 9,564 | 202 | 0 |
| insufficient_observation | 0 | 0 | 0 | 0 | 6,546 |

v1 후보는1,874개(7.86%), v2-3 후보는6,139개(25.76%)이고 양쪽에서 선정된 Frame은497개다. v1 후보지만 v2-3 REVIEW인1,377개, v1 not_candidate지만 새PATTERN/LOCAL인5,642개가 있었다. 후보량 차이는 서로 다른 규칙과 미래 정보 사용의 결과이며 성능 우위를 의미하지 않는다. Precision/Recall/F1·실제False Positive를 계산하지 않았다.

연속 run은 같은 half/split·연속 frame·같은 출력 상태에서만 세며 사건을 연결하지 않는다.

| 모델·상태 | 연속 run 수 | 중앙 길이 | P95 | 최대 |
|---|---:|---:|---:|---:|
| v1 candidate | 100 | .46s | 2.648s | 6.00s |
| v2-3 PATTERN | 254 | .36s | 2.510s | 4.32s |
| v2-3 LOCAL | 164 | .24s | .794s | 1.00s |
| v2-3 REVIEW | 414 | .62s | 3.428s | 11.56s |
| v2-3 NO | 8 | 1.00s | 1.424s | 1.48s |

PATTERN↔LOCAL 전환도 서로 다른 상태 run으로 세었다. 인접 Frame의 대표 Anchor가 바뀔 수 있어 이 길이는 한 집단·한 경합 Episode의 길이가 아니다.

## 10. 고정 결측 실험

Stage A의 고정1,778개/30블록 Validation Window, bitwise 동일한 원래 temporal 배열, seed2027 및 Train donor를 재사용했다. 원래 clean 상태는PATTERN408/LOCAL37/REVIEW1,213/NO120개였다. 총117조건 중105개 계산/12개INCOMPARABLE이며 삭제 수를 줄이지 않았다.

| 동일 cohort의 삭제 조건 | PATTERN | LOCAL | REVIEW | INSUFFICIENT | 후보 소실 |
|---|---:|---:|---:|---:|---:|
| 먼 과거1개, 현재 보존 | 0 | 178 | 1,600 | 0 | 267 |
| 먼 미래1개, 현재 보존 | 380 | 49 | 1,349 | 0 | 16 |
| 현재 직전1개, 현재 보존 | 0 | 0 | 1,778 | 0 | 445 |
| 현재 직후1개, 현재 보존 | 380 | 0 | 1,398 | 0 | 65 |
| 무작위1개, 현재 보존 | 187 | 69 | 1,522 | 0 | 189 |
| 현재1개 삭제 | 0 | 0 | 0 | 1,778 | 445 |

먼 과거 결측은 B.2 Local 자체를 바꾸지 않지만 A/B의 원래 충분한 지원을 끊을 수 있다. 따라서 PATTERN→LOCAL 또는REVIEW 전환이 생긴다. 먼 미래 결측에서도 과거만 필요한 A는 남는다. Local이 남았다고 전체Pattern을 복구한 것은 아니다.

현재 보존 실험의 Spatial Gate 교집합은1,778개지만 위 결측 조건에서는 양쪽 모두 전체 동적 지원을 가진 교집합이0이었다. 이 상태 변화를 동일 full-information 조건의 정확도 저하로 부를 수 없다. 현재 삭제는 Anchor 상실로INSUFFICIENT다. 조건/radius별 전체5×5 전환 및 지원 교집합은 `masking_robustness.csv`에 있다.

## 11. Tracking perturbation 민감도

기존12×151개 Validation support의612개 중심에서 만든3,191개 fixed clean Anchor probe를 재사용했다(3m1,195/5m1,996). 현재 군집은 reciprocal-unique bridge가 명확할 때만 재대응시켰다. 대응이 없거나 모호하면INSUFFICIENT이며 다른 군집 ID를 강제로 붙이지 않았다.

Jitter/Dropout은 support 전체 관측에, composition은 각 support의offset65..85에 기존 함수·seed 그대로 적용했다. clean 추적도 실제 위치 정답은 아니다.

| 조건 | PATTERN | LOCAL | REVIEW | INSUFFICIENT | 신규/소실 후보 | 상태 변경 |
|---|---:|---:|---:|---:|---:|---:|
| Clean | 351 | 70 | 2,177 | 593 | — | — |
| Jitter .1m | 517 | 69 | 1,954 | 651 | 213 / 48 | 325 |
| Jitter .5m | 303 | 98 | 1,876 | 897 | 210 / 230 | 778 |
| Jitter 1m | 74 | 74 | 1,603 | 1,433 | 116 / 389 | 1,301 |
| Dropout10% | 0 | 0 | 2,005 | 1,186 | 0 / 421 | 1,188 |
| Dropout20% | 0 | 0 | 1,198 | 1,993 | 0 / 421 | 1,823 |
| Dropout30% | 0 | 0 | 511 | 2,680 | 0 / 421 | 2,255 |
| Composition20% | 168 | 3 | 1,875 | 1,145 | 0 / 250 | 826 |

.5m/1m Jitter에서는 별도NO17/7개도 있었다. .1m에서PATTERN→REVIEW/INSUFFICIENT24개, REVIEW→LOCAL25개가 발생했다. .5m는169/75개,1m는319/66개였다. **새 후보 발생은 성능 향상이 아니다.** 현재 관측 구성·edge 변화·median/기존Pattern 순서 및 지원 조건이 함께 달라졌다.

공통 지원을 구분하면:

| Jitter | 양쪽 Spatial Gate 유효 | 같은 비영 dynamic support bits | 그 안 상태 변경 | 양쪽 전체 dynamic 지원 | 그 안 상태 변경 |
|---|---:|---:|---:|---:|---:|
| .1m | 2,540 | 2,161 | 239 | 289 | 65 |
| .5m | 2,274 | 808 | 237 | 175 | 101 |
| 1m | 1,718 | 251 | 73 | 46 | 32 |

support bits는 A/B와Local4개가 평가 가능한지의 집합이다. 같은bits가 전체 원시 관측 mask나 같은 선수를 보증하지 않는다. 강도가 커질수록 교집합도 바뀌므로 서로 다른 분모의 비율을 무조건 비교하지 않는다. 양쪽 모두 충분한 전체 동적 지원이 있는 표본에서도 변화가 남아, 지원 소실만이 민감도의 원인은 아니었다.

Radius별 .1m의신규/소실/상태 변경은3m56/19/114개,5m157/29/211개였다. .5m는3m31/86/284,5m179/144/494개이며1m는3m18/154/535,5m98/235/766개다. 어느 radius를 우수한 것으로 선택하지 않았다.

Dropout에서는 모든 후보가 소실됐지만 대부분REVIEW/INSUFFICIENT로 보류됐다. 실제 비경합으로 바뀐 것이 아니다. Slot 순열/Home-Away 교환은 **모든 결정 열이 정확히 동일**했고 상태 전환0개였다.

## 12. 구현·출력·재현·보존

새 Python 모듈5개와 테스트1개를 추가했다. 기존 코드는 변경하지 않았다.

| 파일 | 역할 |
|---|---|
| [candidate_decision_v23.py](../src/competition_judgement/candidate_decision_v23.py) | 같은 Anchor의 공간 gate·동결 상태·지원에서13개 결정 필드 생성 |
| [candidate_frames_v23.py](../src/competition_judgement/candidate_frames_v23.py) | 완성된 Anchor 상태의 Frame/radius별 집계·출처 보존 |
| [candidate_diagnostics_v23.py](../src/competition_judgement/candidate_diagnostics_v23.py) | Validation 분포/지원/Alternative, v1 교차표, radius 중복, 고정 민감도 |
| [run_candidate_v23.py](../src/competition_judgement/run_candidate_v23.py) | 읽기 전용 B.1/B.2 join과 전체 실행·hash 보존 |
| [verify_candidate_v23.py](../src/competition_judgement/verify_candidate_v23.py) | 모든 결정 재계산, Frame 출처·원래 파일 보존 검증 |
| [test_candidate_v23.py](../tests/test_candidate_v23.py) | 지정20상황을 포함한 신규45개 합성 검사 |

최종 경로는 `data/derived/competition_judgement/candidate_v23_final_20261110/`다. 지정11개 파일을 모두 생성했다. 주요 출력은 **Cluster789,005×16열 / Frame199,319×39열**, gzip 약4.94MiB/1.26MiB다. 원래 Evidence 값·Pattern/Local profile·시간 배열·QC mask를 다시 복사하지 않았다. key와결정/출처/지원 metadata를 가진 Sidecar이며 상세 원본은기존key로join한다.

추가로 `candidate_support_summary.csv`, `alternative_evidence_diagnostic.csv`, `consecutive_frame_runs.csv`, `upstream_unknown_reason_summary.csv`, `masking_conditions.csv`, `tracking_probe_correspondence.csv`, 원래 입력 SHA·고정config·보존snapshot을 저장했다. 모든bit/정확한열/단위/의미는 `data_dictionary.json`에 있다.

초기 전체 실행의 Radius 진단에서 기존family_id와cluster_id를 바꾼 조인 키가 중복되는 오류를 발견해 **신규 진단 모듈만 수정**하고 회귀 검사를 추가했다. 미완료 출력은 `candidate_v23_incomplete_radius_join_20261110/`로 보존했다. 이 폴더는 최종 결과가 아니다. 후보 규칙·설정은 수정하지 않았다.

강한 합성Jitter로 특정radius의군집이 사라지는 경우도 검사했다. 기존B.2의 빈batch 호출은 지원하지 않는 경로여서 기존함수를 바꾸지 않고, 현재 군집이0개인 경우 Frame의관측된empty graph 처리로 분기한다. 실제 전체 run과고정probe 진단은 비어 있지 않은batch만 기존함수에 전달했다.

`verification.json`은 모든군집 상태·reason·bit·미래 metadata를 동결입력에서 재계산하고, Frame 대표/radius별 출처·전체축을 재현했다. 이전245개 파일과8,423개 산출물은 변경되지 않았다. 최종522개 테스트 실패0개이며 실제 데이터 전체 실행도 완료했다. 기존회귀 검사의 합성신경망 검사와실제경기 재학습은 구분한다. 신규모델 학습은 하지 않았다.

실행 환경은 `mtah2027`, macOS arm64 CPU/Python3.11.17, NumPy2.4.6/pandas3.1.0/pytest9.2.0이다. 추가 설치는 필요하지 않았다. 프로젝트root에서 새폴더로재현한다.

```bash
conda activate mtah2027
export PYTHONDONTWRITEBYTECODE=1
export PYTHONPATH="$PWD/competition_judgement/src:$PWD/MTAH_Relational_Starter:$PWD"

python -m pytest -q -p no:cacheprovider tests MTAH_Relational_Starter/tests competition_judgement/tests > /private/tmp/cj_v23_reproduction_tests.log 2>&1

python -m competition_judgement.run_candidate_v23 \
  --output data/derived/competition_judgement/candidate_v23_reproduction

python -m competition_judgement.verify_candidate_v23 \
  --run-dir data/derived/competition_judgement/candidate_v23_reproduction \
  --preservation-snapshot data/derived/competition_judgement/candidate_v23_final_20261110/preservation_start.json \
  --tests-log /private/tmp/cj_v23_reproduction_tests.log
```

기존 출력 폴더가 있으면 덮어쓰기를 거부한다. 원본 ATD 재파싱 없이 기존 로컬 cache/산출물을 사용한다. 서버·외부 업로드·Git Commit/Push는 하지 않았다.

## 13. 해석 한계와 이후 과제

이번에 검증한 것은 **정해진 공간·시간 관측 조건과 추적 가능한 선정 이유**다. 볼 위치·실제경합 정답·선수 영구 ID가 없으므로 정확도·False Positive 감소·실제 비경합 판정·H/A/D/X 개선은 확인할 수 없다.

현재 Anchor의 최근접 상대 쌍과 구성원이 시간에 따라 바뀔 수 있다. 같은 관측 수도 같은 선수라는 보증이 없다. 지역 이탈과 Tracking 누락을 완전히 구분할 수 없다. 큰 Anchor는 여러 관계를 포함할 수 있고, 3m/5m 및 인접 Frame은 강하게 중복된다. 모든 통계는 한 경기의 상관된 반복 관측에 대한 기술 통계이며, 독립 경합 사건이나 일반화 성능을 추정하지 않는다.

REVIEW가 많고 NO가 적다는 결과는 보수적인 관측 정책과 선정 규칙의 결과다. 이 빈도를 목표로 기준을 조정하지 않았다. .1m Jitter에서도 상태 변화가 커서 후보가 안정적인 축구 의미를 갖는다고 주장할 수 없다. 실제 결측은 합성 삭제와 다를 수 있다. 높은 Coverage만으로 개선이라고 결론 내리지 않는다.

향후 Episode 설계에는 같은 frame의 radius 중복과 시간상 Anchor 교체/병합·분리/누락을 구분하는 별도 연결 정책, 검열·불확실성 보존이 필요하다. 이후 H/A/D/X 단계에서는 이 후보 상태를 독립적인 상황 routing 단서로만 전달하고, 볼 관측 품질·점유 판단·대회 제출 축과의 연계를 별도로 설계해야 한다. 이번에는 Episode, 새로운 종합 점수·확률, 점유 판단, 학습·GT/영상 기반 튜닝을 구현하지 않았다.
