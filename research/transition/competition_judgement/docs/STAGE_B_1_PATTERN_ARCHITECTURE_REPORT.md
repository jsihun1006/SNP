# Stage B.1 — Spatiotemporal Evidence & Pattern Architecture

2026-11-10. 설계 감사·독립 구현·전체 로컬 실행·Validation 진단을 완료했다. **개별 Evidence → 특징별 관측 지원 → 비배타적 Pattern Profile**로 구조를 바꿨다. 새로운 Positive/Negative 평균·고정 종합 점수·차감·Score Bounds·확률·경합 후보는 만들지 않았다.

최종 결과는 `data/derived/competition_judgement/evidence_pattern_b1_final_20261110/`다. 기존 모든 단계의 코드·설정·결과와 원본/모델/baseline을 보존했다. 이 문서의 패턴은 실제 경합·비경합 정답이 아니다.

## 1. 결과와 핵심 해석

* **199,319 Frame / 789,005 군집 관측**을 출력했다. 공식745,891개와 원래 Full-reliable191,868개(25.72%)를 그대로 유지했다.
* Stage B의14개 Full/Partial 개별 값과 비교 가능한 전체 행에서 **최대 절대 차이0**이었다. 수식을 바꾸지 않았다.
* Validation112,286개에서 COMPLETE는 P1 41,788/P2 39,729/P3 50,798/P4 21,694/N1 8,674/N2 50,357/N3 24,452개다. 전체101개가 아니라 실제 필요 구간을 사용한 결과다.
* 정의한 기하 패턴을 지원한 군집 관측은 **A 8,641, B 469, C 0, D 23,055개**다. 반복 frame·겹친 Anchor·3/5m view가 포함되며 실제 사건 건수가 아니다.
* **C를 완전히 평가할 수 있던8,674개 모두에서 soft range가 사전 기준 .02를 넘었다.** 낮은 발생량을 보고 기준을 재선택하지 않았다. 실제 지속 근접/마킹이 없다는 뜻이 아니다.
* 단일 frame의 pattern별 출처를 따로 유지했다. 서로 다른 군집의 P1/P3/P4를 합쳐 B를 만들거나3/5m 관측을 독립 사건으로 합산하지 않는다.
* 결측에서 패턴을 많이 보류한다. jitter에도 개별 값과 지원/미지원이 달라졌다. **새 구조의 성공은 의미 일관성·불확실성 보존이지 탐지 정확도/강건성 개선을 입증한 것이 아니다.**

## 2. 감사와 변경 이유

[사전 감사](STAGE_B_1_PATTERN_DESIGN_AUDIT.md)에 Stage1/v2-1/v2-2/Stage A/Stage B 보고서와 실제 수식·관측 정책·cache/matching/집계 코드를 대조했다. 기존 Stage B는 개별 항목의 정의가 명확했지만 종합 평균의 유효 항목과 Negative family가 달라질 때 숫자의 의미가 바뀌었다. 같은 frame의 P/N 최대값도 다른 Anchor에서 나올 수 있었다.

Stage A/B에서는 결측 위치에 따른 평균/극값 편향, N1의 연속 지속 하한 손실, 관측 구성 변경의 숨김 문제가 확인됐다. 이를 새로운 평균이나 미관측 정보 복원으로 해결하지 않고 **지원 상태와 검증된 기하 구조를 분리**했다.

`configs/pattern_b1.json`을 실제 B.1 구현/Validation 전에 고정했다. Stage B 설정은 읽기 전용으로 참조하고 SHA256을 저장한다. 실제 데이터에서 수식·noise scale·sequence 기준을 반복 조정하지 않았다. 기존 source slot/cluster_id를 영구 인물/집단 ID로 간주하지 않는다.

## 3. 개별 Evidence와 실제 의존 구간

원래 함수의7개 수식·scale·P4 optional term은 독립 `pattern_values.py`에 그대로 보존했다. 원래 source/kernel/geometry/matching을 재사용한다. **서로 다른 Evidence를 평균하는 경로는 없다.** 아래 평균은 원래 개별 Evidence 내부 평균이다.

`g(x;s)=clip(x/s,0,1)`. μP/μF는 과거/미래50개, PN/FN은 가까운25개, PF는 먼 과거25개 평균이다. 모두 reliable 관측만 사용한다.

| 특징 | 보존한 수식 | 필요한 offsets |
|---|---|---|
| P1 | mean(g(μP(d)−d0;3m),g(S0−μP(S);.25),g(μPN(S)−μPF(S);.25)) | −50..0 |
| P2 | 원래 R/D 축소·contact/edge/density 증가5항 평균 | −25..0 |
| P3 | 원래 K/density 변화와 복수 매칭 후보항3개 평균 | −25..25, ±1 matching 포함 |
| P4 | soft/거리 미래 변화·known end·가능한 D확장 평균 | 0..50 |
| N1 | 연속 지속 하한/4.04 × 구조 안정 × 평균 접근/분리 부재 | −50..50 |
| N2 | 순변위/1m clip × exp(−D range/1m−R range/.5m−density range/.25) | 기하 경로−10..10 |
| N3 | occupancy×exp(−S std/.1)×(1−S range)×(1−peak/.25 clip) | −50..50 |

정확한7개 원래 수식은 [Stage B 보고서](STAGE2_2_5_EVIDENCE_REPORT.md)와 사전 감사에 있다. 모든 score는 무차원[0,1]이다. d/R/D/centroid 변화는 m, edge는 쌍 수, contact는 현재 관측 수, 기존 duration 하한은 s다. 새 Score Lower/Upper Bound는 만들지 않았다.

현재 Anchor는 그대로 `A=max(6m,현재 diameter+3m)`의 고정 원이다. 시간별 S/d/E/C는 원 안 관측을 다시 선택한 집합 통계다. R/D/K는 centroid가 원 안인 component의 최대 반경/지름/개수다. R/D의 최대값이 같은 component라는 보증도 없고, 큰 component의 모든 멤버가 원 안이라는 보증도 없다.

P4는 원래 duration 함수가 past까지 반환해도 **future end reason만** 사용한다. N2는 고정 원 reliable mask와 다른21개 움직이는 기하 corridor를 사용한다. 이 실제 의존성을 관측 판정에 반영했다.

## 4. Feature-specific Observation Policy

### COMPLETE

필수 구간 전체의 원래 신뢰 관측과 해당 Evidence의 기존 구성/기하 조건을 요구한다. 전체101 Full을 일괄 요구하지 않는다. 현재 Strict QC는 유지한다. P1/P2는 불필요한 미래, P4는 불필요한 과거 때문에 보류하지 않는다.

P2의 finite geometry≥20, P2/P4/N1의 local 구성 조건, P3의 matching 관측 조건, P4의 known end/censor를 유지했다. N2는 정확한 frame 간격·같은 half/split·각 시점 QC·전역/대응 component H/A 수와 reciprocal-unique 경로를 검사한다.

COMPLETE는 지정한 관측 조건의 완전성이다. 같은 인원이라도 같은 사람이 아니며, 실제 위치 정확도·경합 정답을 보증하지 않는다. P1/P3처럼 원래 local 구성 불변을 요구하지 않는 특징은 해당 변화 flag를 보존한다.

### EXPERIMENTAL_PARTIAL

COMPLETE가 아니면 **동결 Stage B Partial 정책을 그대로** 적용한다. P1/P2의 중앙±10 추가 요구까지 유지했다. 예를 들어 P1의 과거가 완전하면 미래 공백과 무관하게 COMPLETE지만, 과거가 불완전하면 원래 Partial guard를 사용한다. 새로운 Partial 최적화를 하지 않았다.

N1의 불완전101 Window는 계산 가능해도 반드시 실험 상태다. Stage B에서 한 프레임 공백만으로 MAE .0412/P95 .1286을 보였으므로 운영 신뢰 상태로 승격하지 않는다. N3도 extrema의 실제 보존을 주장하지 않는다.

### UNKNOWN

관측/QC/구성/기하/수식이 부족하면 NaN이다. 낮은 값0과 다르다. feature마다 state/reason/필수 구간의 reliable 수·최대 gap·구성 flag를 저장하고 공통 masks를 한 번만 참조한다. N2는 별도21비트 path mask를 참조한다.

## 5. Layer 1과 Layer 2

### Layer 1 — Evidence Combination

`available_mask`는 P1,P2,P3,P4,N1,N2,N3의 확보 여부 bits다. 값의 high/low가 아니다. 각 패턴의 required list와 `_missing_mask`가 어떤 필수 값이 없는지 보여준다. P1/P3가 있어도 P4가 UNKNOWN이면 B의 전체 근거가 확보되지 않았다.

P3의 K변화·density변화·복수 후보항·matching ambiguity를 각각 보존한다. P3의 ambiguity는 N2의 명확한 대응과 다른 정보다. 복수 후보만으로 실제 merge/split을 확인하지 않는다.

### Layer 2 — Temporal Pattern Verification

원래 cache에서 시간별 S/d/E/K/geometry를 가져온다. **Evidence 값이나 Window 평균만으로 순서를 확정하지 않는다.** 원시 배열이 없는 `without_sequence()` 입력은 항상 Sequence UNVERIFIED/Outcome UNKNOWN이다. 필수 값이 있어도 시간 검증과 별개다.

5프레임 trailing median과 lag5를 사전에 정했다. 각0.2초이며 S/d를 smoothing한다. soft .02/distance .1m는 Stage A deadband, D의 이동/shape scale은 Stage B에서 가져왔다. 실제 tracking 오차의 통계적 상한값은 아니다.

필요한 sequence 구간에 공백이 있으면 median 결과를 관측값으로 보간·대체하지 않고 검증을 보류한다. 일부 Experimental Partial 평균은 남아도 VERIFIED로 자동 승격하지 않는다. 구조 변화가 실제 동일 선수들의 변화라는 가정도 하지 않는다.

## 6. 패턴 정의

### A — Convergence / Local Concentration

필수 Evidence는 P1/P2다. 접근과 집중을 별도 flag로 기록한다.

* 접근: `S0−μP(S)≥.02`, `μP(d)−d0≥.1m`, 그리고 가까운 과거(끝 offset−25..0) 안 실제 lag5 median 쌍에서 S증가≥.02/d감소≥.1m.
* 집중: PN 대비 R/D 축소 각각≥.1m이고 실제−5→0 R/D 축소도 각각≥.1m이거나, −5→0 contact와 edge가 각각1 이상 증가.

P1/P2가 모두 필요한 관측을 갖추고 두 flag 중 하나라도 성립하면 A SUPPORTED다. 두 flag를 합산하지 않는다. 현재 단순 근접만으로 A를 지원하지 않는다. 단기 raw 변화와 원래 평균 특징은 다르므로 Evidence의 확보를 score>0과 동일시하지 않는다.

### B — Transient Reconfiguration

필수 Evidence는 P1/P3/P4다. 전체101개 reliable 원자료에서 다음을 함께 검사한다.

1. 5프레임 trailing median S의 **처음 전역 최고점**이 중심±5에 있다. 동률은 가장 이른 offset이다.
2. far-past median→peak와 peak→far-future median에서 S차≥.02, 반대 방향 d차≥.1m다.
3. 실제 lag5 rise가 peak 이전에 끝나고 실제 lag5 fall이 peak 이후 시작한다. 각 rise/fall은 S와 d 두 조건을 만족한다.
4. 같은 Window에서 실제 K 또는 E의 range≥1이다. matching ambiguity항만으로 이 조건을 충족할 수 없다.

peak/rise-end/fall-start offset과 구조 변화를 기록한다. 이후 observed off→on은 별도 `B_reapproach`다. 여러 관계가 같은 원 안에 생기거나 최근접 상대 쌍이 바뀔 수 있다. VERIFIED는 **정의한 공간 시계열의 순서**이며 한 축구 사건의 인과관계나 같은 선수 관계의 증명이 아니다. Window 내 구조 변화가 그 선수들의 동일 사건에 속한다는 보증도 없다.

### C — Persistent Proximity

필수 Evidence는 N1/N3다. 전체101개가 연속 reliable+현재 radius의 close이고 S range≤.02, d range≤.1m, K/E range=0이어야 한다. N1/N3의 값은 각각 보존하며 합산하지 않는다.

4초 sample span/4.04초 frame-bin 안의 안정적 지역 근접을 설명한다. Window 밖의 장기 마킹이나 실제 비경합 정답이 아니다. 기존 duration 하한/검열 정보도 그대로 남긴다.

### D — Shape-preserving Co-Movement

필수 Evidence는 N2다. 원래 reciprocal-unique21개 경로에서 centroid 순변위≥1m, D range≤1m, R range≤.5m, density range≤.25를 요구한다. scale은 원래 Stage B 값이다.

**같은 Spatial Anchor 결합을 위해 모든 경로 centroid가 현재 seed 원 안에 있어야 한다.** 밖이면 N2 값은 보존하지만 D는 scope mismatch로 UNKNOWN이다. 원래 경로는 영구 ID가 아니고 실제 같은 선수의 동반 이동을 확인한 것이 아니다. 순변위이므로 왕복 이동을 놓칠 수 있다.

### 상태의 분리

각 패턴은 Observation(COMPLETE/EXPERIMENTAL_PARTIAL/UNKNOWN), Sequence(VERIFIED/UNVERIFIED/NOT_APPLICABLE), Outcome(SUPPORTED/NOT_OBSERVED/UNKNOWN)을 가진다.

SUPPORTED는 충분한 동일 Anchor 자료에서 기하 조건 확인, NOT_OBSERVED는 충분한 자료에서 그 조건 불충족, UNKNOWN은 자료/검증 부족이다. NOT_OBSERVED도 실제 비경합을 뜻하지 않는다. N/A는 Frame의 관측된 empty mixed graph 등이다. 패턴은 비배타적이다.

## 7. Validation Coverage와 실제 분포

기존 Validation112,286개 군집 관측만 분석했다. Train/Validation/Test split을 그대로 유지하고 Test는 전체 배치 출력에만 포함했다. 현재 Strict88,592/low23,694개다. low에서는 모든 시간 Evidence가 UNKNOWN이다.

### 7.1 특징별 지원

| 특징 | COMPLETE | EXPERIMENTAL_PARTIAL | UNKNOWN | COMPLETE 비율 |
|---|---:|---:|---:|---:|
| P1 | 41,788 | 4,122 | 66,376 | 37.22% |
| P2 | 39,729 | 511 | 72,046 | 35.38% |
| P3 | 50,798 | 828 | 60,660 | 45.24% |
| P4 | 21,694 | 357 | 90,235 | 19.32% |
| N1 | 8,674 | 699 | 102,913 | 7.72% |
| N2 | 50,357 | 0 | 61,929 | 44.85% |
| N3 | 24,452 | 2,336 | 85,498 | 21.78% |

기존 Full에서 보류됐으나 실제 필요 구간이 완전해 새 COMPLETE가 된 수는 P1 17,336/P2 24,958/P3 26,346/P4 8,249/N2 32,490개다. 기존 Partial조차 보류됐던 표본 중에는 P1 3,187/P2 3,966/P4 1,756개가 새 COMPLETE였다. 실제 수치 의존성과 무관한 반대쪽 중앙 지원 때문에 보류됐던 경우다.

N1/N3는 전체101개를 사용하므로 기존 Full보다 COMPLETE를 늘리지 않았다. N2는 나머지80프레임이 없어도 필요한21개 경로를 완전히 관측하면 COMPLETE다. Coverage 변화는 새로운 신뢰도 calibration이나 실제 정확도 향상이 아니다.

### 7.2 패턴 availability와 발생은 다름

분모112,286개 Anchor view다. availability는 필요한 Evidence가 모두 확보된 C/E의 합이며 Sequence VERIFIED와 다르다.

| 패턴 | Evidence 확보 C+E | 그중 COMPLETE | SUPPORTED | 충분하나 조건 불충족 | 최종 UNKNOWN |
|---|---:|---:|---:|---:|---:|
| A | 27,947 (24.89%) | 25,440 | 8,641 (7.70%) | 16,799 | 86,846 |
| B | 14,990 (13.35%) | 13,445 | 469 (0.42%) | 12,976 | 98,841 |
| C | 9,173 (8.17%) | 8,674 | **0** | 8,674 | 103,612 |
| D | 50,357 (44.85%) | 50,357 | 23,055 (20.53%) | 27,302 | 61,929 |

A/B/C에서 Experimental Partial인2,507/1,545/499개는 Evidence는 있지만 순서를 검증하지 않아 최종 UNKNOWN이다. D는 원래21개 경로가 완전해야 해서 Experimental Partial은0개다.

C의 COMPLETE8,674개 모두 S range>.02였다. S range의 P05/median/P95는 **.0721/.2574/.6167**이었다. 이 중4,456개는 근접 지속 하한4.04초였지만 안정 조건까지 충족한 것은 아니었다. **실제 지속 근접이 없다는 결론이 아니라 이번 엄격한 정의가 지지되지 않았다는 결과**다. 기준을 완화하지 않았다.

모든 패턴 Outcome이 UNKNOWN인 군집 관측54,226개와 모든 패턴 NOT_OBSERVED인2,643개를 구분했다. supported bitmask0인82,972개를 전부 패턴 부재라고 해석할 수 없다.

### 7.3 다중 패턴과 Frame 출처

같은 Anchor에서 A∩B 260개, A∩D 2,499개, B∩D 168개이며 A/B/D 동시76개도 남아 있다. C는0이다. 단일 유형을 강제하지 않았다.

현재 군집이 있는 Validation23,280 Frame에서 서로 다른 패턴 유형이 둘 이상 지원된 Frame은 **3,272개(14.05%)**다. 이 경우 각각 다른 Anchor의 패턴일 수 있으므로 하나의 복합 사건으로 합치지 않는다. 같은 Anchor overlap은 별도의 Long 집계다.

모든 Validation23,828 Frame 중 네 패턴 모두 UNKNOWN은11,941개, 모두 NOT_OBSERVED는309개다. 현재 군집이 없는 strict empty graph는 별도 reason/Sequence N/A를 유지한다. 원래199,319축에 gap을 만들지 않는다.

## 8. Stage B와 비교·중복·불확실성

실제 전체 데이터에서 원래 Full/Partial 개별 값이 정의된 행은 전부 새로운 개별 값과 일치했다. `evidence_value_equality.csv`의14개 비교 최대 차이는0이다. Validation도 동일하다.

새 메인에는 P/N 종합 점수가 없으므로 **가변 cross-Evidence 평균의 의미 변경 문제를 제거**했다. UNKNOWN P4를 다른 항목으로 보완하거나 N1→N3 family 교체로 하나의 숫자를 만들지 않는다. 단, 원래 P4 내부 optional expansion 항목은 바꾸지 않았으므로 `P4_term_mask`를 유지한다. 모든 수식 내부의 가변 항목까지 제거했다고 주장하지 않는다.

기존 Stage B의 Full/Partial P/N·used mask·family source는 별도 `legacy_diagnostics.csv.gz`에 key별로 복사했다. profile 입력으로 쓰지 않는다. 예를 들어 Stage B의10개 Far Past 삭제에서는 Positive1,439개가 계산돼도 모두 항목 구성이 달라 직접 비교할 수 없었다. B.1은 그때 각 특징의 상태·A/B의 검증 보류·D의 동일21개 지원을 각각 전달한다.

N1/N3의 공통9,173개에서 Pearson **.7156**, Spearman **.7426**으로 중복성이 여전히 크다. C는 두 값을 합산하지 않고 연속성과 안정성을 별도 근거로 보존한다. 실제 비경합으로 해석하지 않는다.

P3 raw matching ambiguity는25,893개 관측에서 기록됐고, 복수 후보항이 양수인 관측은14개였다. 이들을 구조 변화의 정답으로 사용하지 않았다. N2에 모호한 대응이 있으면 NaN/UNKNOWN이며, D도 확인하지 않는다. 같은 관측 인원·reciprocal-unique라도 실제 멤버 동일성을 증명하지 못한다.

## 9. 결측 통제와 Tracking 강건성

### 9.1 동일 Stage A/B cohort

기존 **1,778 Window,30개 고정 support block,seed2027**을 재사용했다. 네 공통 시계열 soft/nearest/3m·5m edge는 기존 배열과 **bitwise equal**이었다. 1/5/10/20/30개 결측, 위치10종, 중심 보존/삭제, 연속성8종, Train record/observed/reliable donor3회씩을 사용했다.117조건 중105계산/12구간 용량 초과 비교 불가다. 원시 시간 배열을 새 파일에 복제하지 않았다.

* Far Past1개 삭제: P1은1,778개 모두 Experimental Partial, MAE .001636. P2/P3/P4/N2는 실제 scope가 완전한 표본에서 COMPLETE와 값이 유지됐다.
* N1은647개에서 Experimental Partial/MAE **.041202**, N3는1,778개에서 Experimental Partial/MAE .001273이었다. N1을 새 COMPLETE로 승격하지 않았다.
* 중심−1 삭제: P1/P2/P3/N1/N2/N3가 보류됐지만 **P4는 미래 구간이 완전한1,013개에서 COMPLETE/값 동일**했다. 불필요한 과거 때문에 미래 특징을 차단하지 않는다.
* Near Future1개 또는 Far Future10개 삭제: P1/P2의 과거가 완전하면 A의 지원380개가 유지됐다. B는 미래 순서 검증을 보류했고 D의575개 지원은 필요한21개가 남은 조건에서 유지됐다.
* Far Past1개 삭제에서 A380개/B55개의 원래 지원은 UNKNOWN으로 바뀌었다. D는경로 밖 삭제라 그대로였다. 지지하던 신호가 사라졌다는 뜻이 아니라 **현재 기준으로 검증할 수 없음**이다.

Feature의 `same_mask_n`은 전체101이 아니라 각 특징의 실제 의존 구간에서 동일한 유효 mask를 가진 공통 표본 수다. 처음 진단에서 이 보조 분모를101기준으로 집계한 것을 수정했고 initial CSV를 별도 보존했다. Pattern의 observation unknown과 outcome unknown도 별도 열로 분리했다. 수식·지원·시계열 조건·sample·seed는 바꾸지 않았다.

실제 결측은 움직임과 비무작위로 연관될 수 있다. 완전 관측 구간에 삭제를 옮긴 결과를 실제 미관측 정보의 오류량으로 해석하지 않는다. 알고리즘 구조상 불완전한 필요한 구간을 VERIFIED로 승격한 경우는 없었다. 실제 가짜 경합의 오탐률을 측정한 것은 아니다.

### 9.2 선수 좌표 jitter/dropout/composition

기존12×151 Validation support의 중앙51개×12=612 Frame, **3,191개 clean Anchor probe**를 고정했다. 초기 cache의1812개 raw 관측을 복사해 perturb하고 원본은 변경하지 않았다.

현재 probe 대응은 동일 frame/radius에서 기존 reciprocal-unique 기하 조건이 명확한 경우만 쓴다. ID가 같다는 이유로 연결하지 않는다. 고정 원 안 S/거리/edge 변화는 bridge가 없어도 평가할 수 있는 범위만 남기고, N2/P3 matching은 unknown으로 둔다. jitter.1/.5/1m의 unique-current bridge 수는3,115/2,829/2,188개, dropout.1/.2/.3은2,642/2,099/1,643개였다.

| 조건 | A 지원/UNKNOWN | B 지원/UNKNOWN | C 지원/UNKNOWN | D 지원/UNKNOWN |
|---|---|---|---|---|
| clean(순열 불변 조건) | 339/2,269 | 18/2,726 | 0/2,957 | 506/1,558 |
| jitter .1m | 501/2,321 | 30/2,736 | 0/2,966 | 389/1,752 |
| jitter .5m | 306/2,660 | 8/2,888 | 0/3,020 | 0/2,667 |
| jitter 1m | 71/3,066 | 8/3,045 | 0/3,099 | 0/3,185 |
| dropout10/20/30% 각각 | 0/3,191 | 0/3,191 | 0/3,191 | 0/3,191 |
| 단기 composition20% | 168/2,917 | 0/3,191 | 0/3,191 | 100/2,907 |

분모는 모든 열3,191이다. 표에서 남은 부분은 충분한 관측에서 조건 불충족이다. dropout에서 전부 UNKNOWN인 것은 실제 패턴이 사라졌다는 뜻이 아니고0오차도 아니다. composition에서 일부 P1/P2/P4/N2가 남은 것은 필요 구간이 교란 span을 피하거나 충분했던 조건부 결과다.

jitter .1m의 공통 유효 표본에서 P1 MAE .02491/P95 .08511, N1 .07628/.32352, N2 .16189/.50265였다. .5m에서는 N2 .36128/.78792였다. 지원 cohort가 달라지고 hard edge·shape extrema·matching의 변화가 함께 작용하므로 단일 원인이나 정확도 순위로 읽지 않는다.

jitter .1m에서 A는 clean 지원29개를 잃고191개가 새로 지원됐고, B는12개가 새로 지원됐다. 이는 **합성 좌표 변화에 대한 기하 출력 변화**이지 False Positive 수가 아니다. 작은 jitter에도 패턴/지원이 바뀌므로 강건성이 개선됐다고 주장하지 않는다.

Source Slot 순열/H-A 교환에서는3,191개 probe의 모든 개별 값과 패턴 outcome이 불변이었다. 같은 Anchor·같은 실제 scope의 공통 유효/동일 mask 표본으로 수치 차이를 계산했으며 보류 진입·해제는 별도로 집계했다. 반복 Window와3/5m 관측을 독립 사건으로 사용하지 않았다.

## 10. 합성·전체 테스트·출력 검증

기존404개를 시작 시 재실행해 통과했다. 신규 검사는 다음의20개 요구 상황을 포함한다.

정적 근접, 일정 shape 이동, 급격한 접근, 접근 후 유지/분리/재접근, 국소 집중, 관계 재편, merge/split처럼 보이는 복수 후보, matching ambiguity, 중심·far-past·future 결측, 인원 변화, 같은 현재/다른 이력, 다른 Anchor의 Evidence, 같은 값/다른 support, Slot/H-A 순열, half/split/record 경계, 충분한 관측의 조건 불충족을 검사했다.

실제 기존 geometry/matching 코드를 합성 좌표에 적용하는 integration 검사도 포함했다. P1/P4가 높아도 최고점이 다른 위치여서 순서가 맞지 않으면 B를 확인하지 않았고, 원시 배열 없는 입력도 UNVERIFIED로 남겼다. UNKNOWN을0으로 채우거나 실제 비경합 label을 만들지 않았다.

최종은 **440 passed in29.64s(기존404+신규36), 실패0**이다. 결과 폴더의 `test_results.json`과 `verification.json`에도 기록했다. 실제 전체 Long을 재읽어 원래789,005개 키, 점수/상태의 finite/NaN 관계, actual support 상태, full count191,868, 패턴 proof·source·Frame 재집계를 검사했다. 기존 source/문서223개 SHA256와 원본/기존 산출물8,377개 size/mtime도 그대로다.

환경은 Conda mtah2027/Python3.11.17/NumPy2.4.6/Pandas3.1.0, macOS arm64 CPU다. 본 전체 추출/초기 진단은 **297.93초**, 이후 보조 진단 분모 정리·테스트·최종 보존 검사는 별도다. peak RAM은 측정하지 않았다. 외부 설치·서버·업로드·Commit/Push는 수행하지 않았다. 실제 경기 neural 재학습은 없고 기존 회귀 테스트의 합성 검사는 별개다.

## 11. 산출물·데이터 사전·재현

[데이터 사전](STAGE_B_1_DATA_DICTIONARY.md)에 각 필드의 scope·단위·mask·reason을 정리했다. 출력의 JSON 사전에도 같은 상태/bit 정의가 있다.

| 산출물 | 역할 |
|---|---|
| `cluster_pattern_profiles.csv.gz` | 789,005행×127열. 현재 공간·7개 Evidence·개별 지원·A–D profile |
| `frame_pattern_profiles.csv.gz` | 199,319행×45열. pattern별 존재/보류/출처·radius별 view 수 |
| `legacy_diagnostics.csv.gz` | 기존 종합값·구성의 별도 읽기 전용 복사 |
| `feature_support_summary.csv` | feature별 C/E/U·품질/radius별 분모 |
| `pattern_availability.csv` | availability와 지원/충분한 불충족/UNKNOWN 구분 |
| `temporal_sequence_validation.csv` | observation/sequence/outcome/reason별 집계 |
| `legacy_comparison.csv`, `evidence_value_equality.csv` | 개별 값 동일성·새 scope COMPLETE 비교 |
| `masking_robustness.csv`, `masking_conditions.csv` | Stage A cohort의 위치/연속성/Train-pattern 삭제 |
| `tracking_robustness.csv`, `tracking_probe_correspondence.csv` | fixed-probe jitter/dropout/composition과 대응 가용성 |
| `pattern_overlap.csv`, `frame_multilabel_summary.json` | 비배타적 유형/출처를 가진 관측 빈도 |
| `alternative_evidence_diagnostic.json` | N1/N3 중복과 P3 ambiguity 진단 |
| `summary.json`, `verification.json`, `test_results.json`, `data_dictionary.json` | 설정·실행·검증·사전 |

새 Long은 gzip 약126.85MB, Frame1.28MB, Legacy10.68MB다. 기존 Stage B Long155.70MB/232열보다 메인 정보를 줄였다. 공통 masks는 한 번만 저장하고 N2의 다른21개 basis만 별도로 저장했다. 원시 시간 배열·JSON 문자열 덩어리·모든 군집 정보를 Frame에 중복 저장하지 않는다. 메인 출력에는 Legacy 종합 점수 열이 없다.

신규 Python은 독립 `pattern_values.py`, `pattern_source.py`, `pattern_profiles.py`, `pattern_frames.py`, `pattern_diagnostics.py`, `run_pattern_b1.py`, `verify_pattern_b1.py`다. 설정은 `configs/pattern_b1.json`, 테스트는 `tests/test_pattern_b1.py`다. 이전 파일은 수정하지 않았다.

프로젝트 루트에서 새 폴더로 재현한다.

```bash
conda activate mtah2027
export PYTHONPATH="$PWD/competition_judgement/src:$PWD/MTAH_Relational_Starter:$PWD"
export PYTHONDONTWRITEBYTECODE=1
python -m pytest -q -p no:cacheprovider tests MTAH_Relational_Starter/tests competition_judgement/tests
python -m competition_judgement.run_pattern_b1 \
  --output data/derived/competition_judgement/evidence_pattern_b1_reproduction
python -m competition_judgement.verify_pattern_b1 \
  --run-dir data/derived/competition_judgement/evidence_pattern_b1_reproduction \
  --preservation-snapshot /private/tmp/cj_b1_start.json \
  --tests-log /private/tmp/cj_b1_tests_final_delivery.log
```

snapshot과 tests log는 해당 실행 전에/중에 생성한 파일을 사용한다. 이번 snapshot은 `/private/tmp/cj_b1_start.json`이었다. CLI는 기존 output 폴더 덮어쓰기를 거부한다. gzip CSV는 pandas로 필요한 열만 읽고 mask는 문자열 dtype, `low_memory=False`를 사용한다. 실제 경기 파생물은 로컬 전용이다.

## 12. v2-3 인터페이스와 남은 한계

전달할 것은 현재 key/Anchor/QC, 개별값과 C/E/U·scope·이유·masks, 네 패턴의 observation/sequence/outcome/실제 변화·missing bits다. 패턴은 어떤 Evidence가 확보됐고 어떤 순서가 실제 관측됐는지를 설명한다. Legacy 평균은 입력이 아니다.

Cross-Evidence 가변 평균 문제는 제거했지만 결측 정보·구성/팀 오류·같은 인원의 멤버 변경·영구 ID 부재·Anchor 이탈 ambiguity는 해결되지 않았다. P4의 원래 내부 optional term, N1 Partial의 지속 하한 민감성, N3 extrema 소실도 남는다. 순서 규칙은 사전 고정한 실험용 기하 규칙이며 실제 축구 의미의 검증이 아니다.

특히 C의0 발생과 dropout의 전체 보류는 그대로 남긴 결과다. 이 결과만으로 지원 정책을 운영용으로 채택하거나 후보 cutoff를 맞추지 않는다. v2-3 최종 후보·점유 판단·A/B 재학습·C Fusion·실제 경기 라벨링·GT/볼/태깅 기반 튜닝은 수행하지 않았다.
