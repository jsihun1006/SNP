# Stage B — Positive & Negative Contest Evidence

2026-11-09. **Stage B만 구현하고 전체 로컬 실행·Validation 분석을 완료했다.** P1–P4/N1–N3는 선수 관측의 기하·시간 양상이다. 볼 경합 확률, 경합/마킹 정답, v2-3 후보 상태, H/A/D/X를 생성하지 않는다. 사전 설정은 결과를 보고 조정하지 않았다.

최종 로컬 폴더는 `data/derived/competition_judgement/evidence_stage_b_final_20261109/`다. 기존 v1/v2-1/v2-2/Stage A, 원본, A/B, 제출 baseline은 보존했다.

## 1. 핵심 결과

* 전체 **199,319 Frame / 789,005 군집 관측**을 출력했다. 공식 구간745,891개와 원래 Full-reliable191,868개(25.72%)를 재현했다. 모든 `(frame,radius_m,cluster_id)`와 기존 v2-2 Full mask가 정확히 일치한다.
* Validation112,286개 군집 관측에서 종합 Positive/Negative의 Full Coverage는 각각 **21.78%**다. 실험용 Partial은 **44.81/51.29%**다. 서로 다른 특징별 지원·유효 항목 조합을 사용한 결과이며 공통P80의 대체 정책이 아니다.
* **N1/N3의 공통 Full 표본8,674개에서 Pearson .7103, Spearman .7405**였다. 중복성이 상당하다. Negative는 N1을 우선하고 N1이 invalid일 때만 N3를 같은 persistence family의 대체값으로 사용한다. N1+N3를 함께 더하지 않는다.
* 같은 군집에서 P와N이 사전 reporting band .5 이상인 관측은 Full0개, Partial15개였다. 두 값을 상보적으로 만들거나 충돌 시 하나를 지우지 않았다. 이15개는 실제 경합15건이 아니다.
* Frame의 서로 다른 Anchor 최대값을 한 사건으로 묶으면 오해가 생긴다. Validation Full에서 두 독립 최대값이 모두 .5 이상인77개 Frame은 **전부 서로 다른 출처**였다. 같은 Positive 대표 Anchor의 Negative까지 확인하면0개다.
* Stage A의 **동일1,778개 기준 Window / 30개 support block / 같은 삭제 seed·패턴**을 재사용했다. 중심 인접 공백은 대부분 Evidence를 보류하게 했다. 먼 과거1개 공백만으로도 N1 MAE .0412/P95 .1286이었다. **Partial N1의 현재 지원 조건이 왜곡을 충분히 제한한다고 확인하지 못했다.** 조건을 사후 완화하거나 재조정하지 않았다.

Coverage 증가는 더 많은 관측에서 수학적 descriptor를 계산한다는 뜻이다. 경합 탐지 정확도, False Positive Rate 감소, 실제 마킹 식별 개선을 검증한 결과가 아니다.

## 2. Stage A·기존 구현 감사

요청한 두 문서는 root `docs/`가 아니라 `competition_judgement/docs/`에 있었다. 실제 읽은 자료/구현은 다음과 같다.

* `STAGE2_2_MISSINGNESS_DIAGNOSTIC.md`, `STAGE2_2_TEMPORAL_RELATIONS_REPORT.md`.
* `controlled_temporal_masking.py`, `partial_window_diagnostics.py`, `missingness_cache.py`, `missingness_experiments.py`.
* v2-1 `local_features.py`/`local_clusters.py`.
* v2-2 `temporal_source.py`, `temporal_features.py`, `temporal_matching.py`.
* 기존 FrameStore·QC·split 계약과 기존 테스트.

[사전 감사](STAGE_B_DESIGN_AUDIT.md)와 `configs/evidence_stage_b.json`을 Evidence 구현·실제 Validation 분석 전에 작성했다. 기존 파일의 정책을 수정하지 않고 새 `evidence_*` 모듈로 구현했다.

Stage A에서 가져온 사실은 다음과 같다.

1. 전체 Full25.72%와 **Validation reliable-P80 31.13%는 분모가 다르다**. 본 모듈의 Validation Full 기준은24,452/112,286=21.78%다.
2. 같은 결측 수라도 위치에 따라 접근/미래 평균과 extrema가 달라졌다. 개수만으로 신뢰를 보장하지 않는다.
3. 연속 지속 하한은 현재에서 가장 가까운 공백에 민감했다. 관측된 값의 평균으로 지속시간을 복원하지 않는다.
4. 현재 Strict QC 실패는101개 모두를 차단한다. 전역 인원 변화는 해당 offset의 reliable을 제외하며 별도 시점·횟수로 기록해야 한다.
5. Anchor 안 한 팀이 없어지는 것은 지역 이탈과 Tracking 누락을 구분하지 못한다. `observed=False`를 실제 분리/이벤트 종료로 확정하지 않는다.

원래 masks를 그대로 사용한다.

```
record   = same half + same split + contiguous record run
observed = record AND Anchor 안 H>=1 AND A>=1
reliable = observed AND strict[t] AND strict[offset]
           AND global_nH[offset]==global_nH[t]
           AND global_nA[offset]==global_nA[t]
```

전역 count 정보는 **record가 존재하면 local Anchor가 미관측이어도 별도로 알려져 있다**. 새 구현에서 이 가용성을 분리했다. 합성 frame 삭제 진단에서는 그 삭제 위치의 전역 count도 unknown으로 처리하여 숨긴 값을 이용하지 않는다. count가 같아도 같은 선수들이라는 보증은 없다.

## 3. Anchor·입력·출력의 의미

각 현재 군집의 centroid c와 지름 D0를 사용해 기존 `A=max(6m,D0+3m)` 원을 고정한다. 과거/미래 동일 원 안의 관측을 다시 선택한다. 시간적 ID를 만들지 않는다.

* `S=max exp(-d_HA/3m)`, `d=-3 log S`, 원래 inclusive3m/5m edge와 contact participant 수를 재사용한다.
* `E`는 Anchor 안 cross-edge 수, `C`는 edge에 참여한 서로 다른 **그 시점 관측** 수다. `rho=E/(nH_anchor*nA_anchor)`다.
* `K`는 원 안에 centroid가 들어오는 해당 radius의 v2-1 component 수다.
* `R`/`D`는 위 지역 component들의 **최대 mean-radius / 최대 diameter**다. 없으면 NaN이며0으로 채우지 않는다. 두 최대값의 출처 component는 다를 수 있다.

R/D는 raw contact participant 집합 전체의 반경/지름이 아니다. centroid가 원 안인 큰 component의 일부 멤버는 원 밖에 있을 수 있다. P2/P3/P4는 현재 군집을 영구 추적한 변화가 아니라 **현재 군집이 지정한 지역의 관측 구조 변화**다. 이 구분이 점수의 중요한 한계다.

N2만 별도 기하 corridor를 사용한다. 같은 radius의 기존 reciprocal-unique 인접 대응을 현재부터 과거/미래 각각10단계만 따라간다. 매 Window에서 새로 만드는 임시 참조이며 영구 군집 ID나 선수 ID가 아니다.

모든 점수는 **무차원[0,1] descriptor**다. 원시 변화량은 m/관측 수/쌍 수/초/비율 단위로 함께 기록한다. 현재 공간 특징·팀별 관측 수·centroid·반경·지름·밀도·contact strength는 시간 Evidence가 invalid여도 Long에 보존한다. Frame에도 별도 Spatial 대표 Anchor를 보존했다.

## 4. Positive/Negative의 수학적 정의

`g(x;s)=clip(x/s,0,1)`. 모든 평균·표준편차·range는 **사용 가능한 reliable 관측만** 사용한다. 빈 support는 NaN이다. 과거/미래 각50개를 P/F, 가까운25개를 PN/FN, 먼 과거25개를 PF라 한다. μ는 해당 집합 평균, sd는 population 표준편차다. 현재가 알려져 있어야 한다.

### P1. Rapid Convergence

```
aS = S0 - μP(S)
ad = μP(d) - d0                          [m]
short = S0 - μPN(S)
long  = S0 - μPF(S)
P1 = mean(g(ad;3m), g(aS;.25), g(long-short;.25))
```

거리 감소·soft 증가·near/far 대비를 함께 기술한다. `long-short=μPN(S)-μPF(S)`다. 실제 개별 선수의 속도/가속도나 최초 접근 시간은 아니다. 결측 위치에 따른 편향을 제거한 추정량이라고 주장하지 않고, 각 band 지원·gap·중심 지원을 기록한다.

### P2. Local Concentration

```
P2 = mean(
 g(μPN(R)-R0;3m),
 g(μPN(D)-D0;6m),
 g(C0-μPN(C);4 observations),
 g(log(1+E0)-μPN(log(1+E));log(3)),
 g(rho0-μPN(rho);.25))
```

R/D 평균은 실제 component geometry가 정의된 PN sample만 사용하고 최소20개를 요구한다. 현재/과거의 전역·국소 H/A 관측 구성이 달라지면 보류한다. 이는 가짜 집중을 증명한 분류가 아니라 **이동과 누락을 구분 못하는 관측 구성 ambiguity에 대한 보류**다. 인원이 같아도 멤버 교체/팀 오류는 남는다.

### P3. Relationship Reconfiguration

```
k = (abs(K0-μPN(K)) + abs(μFN(K)-K0))/2
r = (abs(rho0-μPN(rho)) + abs(μFN(rho)-rho0))/2
b = max(prev_candidate_count-1,0) + max(next_candidate_count-1,0)
P3 = mean(g(k;2 clusters), g(r;.25), clip(b/2,0,1))
```

candidate_count는 기존2m centroid step·size ratio≥.5·disk IoU≥.1의 인접 기하 후보 수다. 여러 후보는 병합/분리의 **가능한 기하 증거**이며 실제 멤버의 merge/split 판정이 아니다. geometric ambiguity를 그대로 보존하고 N2에는 사용하지 않는다. observation ambiguity/경계이면 P3도 보류한다. 후보0을 실제 소멸로 확정하지 않는다.

### P4. Subsequent Separation

```
P4 = available-term mean(
 g(S0-μF(S);.25),
 g(μF(d)-d0;3m),
 I(현재부터 이어진 관계가 reliable한 known-off에서 끝남),
 g(μF(D)-D0;6m))
```

앞의 세 항목이 정의돼야 한다. 미래 component geometry가 전혀 정의되지 않으면 expansion만 제외하고3개 항목 평균으로 기록한다. `P4_term_mask`의 bits0/1/2/3은 soft/distance/known-end/expansion이다. 비교 시 이 내부 term 구성도 확인한다.

현재부터 이어지는 prefix가 관측 부족/지역 이탈 ambiguity에서 먼저 끊기면 P4를 보류한다. 알려진 종료 또는 미래 전체가 이어져 Window 경계에서 검열된 경우만 이 조건을 통과한다. 알려진 종료 후 먼 곳에 부분 공백이 있는 경우는 나머지 개별 지원 조건을 검사한다. 지역 관계 종료는 같은 선수의 분리나 볼 경합 종료를 뜻하지 않는다.

### N1. Persistent Marking Proxy

```
L = 현재부터 양쪽으로 연속 reliable AND E>0인 sample 수 /25 [s]
T = clip(L/4.04s,0,1)
Q = exp(-sdW(rho)/.25 - sdW(K)/(1 cluster))
A = 1 - max(g(aS;.25), g(S0-μF(S);.25))
N1 = T * Q * A
```

연속 하한·지역 count/density 안정과 접근/분리 평균 변화의 부재를 결합한다. L은 실제 장면 길이 확정값이 아니며 gap을 건너뛰지 않는다. `onset_censored/end_censored`, `past_end_reason/future_end_reason`을 원래 함수에서 유지한다. Window 밖의 장기 마킹이나 같은 선수의 지속 관계를 확인한 것은 아니다. 모든 반경/shape 변화의 부재를 검증하는 것도 아니다.

### N2. Co-Movement Proxy

±10의21개 기하 대응에서 다음을 모두 요구한다.

1. 매 인접 연결이 reciprocal-unique, 정확한 원래 frame 간격.
2. 같은 half/split, 모든 해당 시점 Strict QC, 전역 H/A 인원 일정.
3. 대응 component의 H/A 관측 수 일정.

centroid 양끝 변위 M, 경로상의 diameter/radius/density range를 ΔD/ΔR/Δrho라 하면:

```
N2 = g(M;1m) * exp(-ΔD/1m - ΔR/.5m - Δrho/.25)
```

경로의 처음/마지막 sample 차이는0.8초다. 개인 속도·벡터를 계산하지 않는다. 모호한 대응/관측/경계/구성 변경이면 **NaN**이며 `N2_path_reason`, 현재 prev/next status와 candidate 수를 기록한다. 왕복해서 순변위가 작으면 움직임이 있어도 낮게 나올 수 있다. 서로 가까운 다른 집단으로 연결되는 오류 가능성도 남는다.

### N3. Stable Proximity

```
O = μW(I(E>0))
peak = max(S0-μW(S),0)
N3 = O * exp(-sdW(S)/.1) * (1-rangeW(S)) * (1-g(peak;.25))
```

낮은 soft 변동·높은 contact occupancy·작은 현재 peak를 기술한다. Partial의 extrema가 원래 extrema라는 보증은 없으므로 `partial_N3_extrema_unverified=True`를 별도 기록한다. N1과 상관이 높아도 동일 수식은 아니며 N1은 연속성과 구성 안정, N3는 soft 분포가 중심이다.

## 5. 특징별 관측 정책

### 공통 원칙

* Full은 원래101개 reliable을 모두 요구한다. 추가 정보 조건 때문에 개별 Evidence Coverage는 원래 Full 이하일 수 있다.
* Partial도 현재 observed+Strict를 요구하고 reliable에 남은 값만 쓴다. 현재 QC 실패를 완화하지 않는다.
* 지원 개수는 네 band FP/NP/NF/FF 각각25개다. 중앙 지원은 표의 ±r 전체가 연속 reliable인 조건이다. 단, max-gap과 전역/국소 구성 조건은 각 특징의 scope 안에서 검사한다.
* `global_constant`는 해당 scope의 record에 알려진 전역 H/A 수가 현재와 같은지 검사한다. local 구성은 양 팀이 observed인 시점의 Anchor H/A 수를 비교한다. Partial의 누락 위치에서 구성 불변을 입증한 것은 아니다.

| 특징 | Partial 최소 band FP/NP/NF/FF | 중앙 완전 지원 | scope 최대 gap | 추가 조건 |
|---|---|---|---:|---|
| P1 | 20/23/0/0 | ±10 | past≤5 | 위치별 수·현재 QC 보존 |
| P2 | 0/23/0/0 | ±10 | near-past≤2 | 해당 scope 전역/국소 수 일정, geometry≥20 |
| P3 | 0/24/24/0 | ±10 | ±25≤1 | 전역 수 일정, 인접 matching 관측 지원 |
| P4 | 0/0/23/22 | ±10 | future≤3 | 전역/국소 수 일정, 종료 해석 가능 |
| N1 | 20/25/25/20 | **±25** | whole≤3 | 전역/국소 수 일정, 연속 하한·검열 유지 |
| N2 | 0/0/0/0 | ±10 | path≤0 | **21개 경로 전부 명확함** |
| N3 | 23/23/23/23 | ±10 | whole≤2 | extrema 보장 안 됨 flag |

이 숫자는 Stage A가 안전성을 보증한 최적 정책이 아니다. 구현 전 고정한 **실험용 guard**다. 특히 본 실험에서 N1의 왜곡 제한이 충분하지 않은 사례가 나왔지만 사후 수정하지 않았다.

모든 점수마다 `{mode}_{P1..N3}_valid/reason/support/observed_n/reliable_n/max_gap/global_count_changed/local_composition_changed`를 기록한다. `mode`는 `full` 또는 `partial`이다. support는 `full_reliable`, `experimental_partial`, `spatial_only`, `insufficient_observation`이다. Partial namespace라도 원래 Full이면 support는 full_reliable다.

공통 metadata에는 네 band/과거/미래/중앙 지원 수,101비트 masks, Window 최대 gap, 현재 QC/관측, 주변 QC 실패 수, 전역 count 차이 offset과 횟수·인접 전이 시점, local 구성 변화, matching uncertainty가 있다. reason은 우선순위가 있는 **주된 보류 사유**이므로 이유별 빈도를 전체 중복 원인의 합집합이라고 읽지 않는다.

## 6. 종합 Evidence와 결측 항목 처리

Positive는 유효 P1–P4가 **최소2개**일 때 그 항목들만의 산술평균이다. `positive_used_mask` bits0..3과 `positive_used_n`이 항목과 동일 가중치를 지정한다. 유효 항목이 부족하면 NaN이다.

Negative는 두 family의 available mean이다.

1. Persistence family = **N1이 valid이면 N1, 그렇지 않을 때만 valid N3**.
2. Movement family = valid N2.

한 family라도 유효하면 그 family를 사용한다. `negative_persistence_source`, `negative_uses_N2`, `negative_family_n`으로 항목과 가중치를 명시한다. N1/N3를 합산하거나 최댓값으로 증폭하지 않는다. 단, fallback에 따라 같은 Negative 숫자라도 근거가 다를 수 있다.

결측 항목을0으로 대체하지 않는다. 코드의 masked sum은 수치 연산일 뿐 분모는 실제 valid 항목/가족 수다. **0은 계산된 증거의0**이고 NaN은 모름이다. `negative=1-positive`, 합1 정규화, P−N 최종 지수는 없다. Full/Partial을 별도 열로 유지하며 항목 구성이 다르면 같은 척도의 신뢰 점수로 직접 대조하지 않는다.

예를 들어 정적인 근접 관계에서 N1=N3=1, N2=0이면 Negative는 **(1+0)/2=.5**다. 이것이 .5의 경합/비경합 확률이라는 뜻은 아니다. N1이 없고 N2만0이어도 계산된 Negative는0일 수 있으므로 `family_n/source`를 버리면 잘못 해석하게 된다.

## 7. Validation 실제 결과

원래 Validation은 전반66811..78730, 후반166381..178288이다. Train/Test 분할을 변경하지 않았다. 분석에는 Validation112,286개 군집 관측만 썼다. 현재 Strict88,592, low23,694다. Test는 전체 배치 출력에만 포함하고 기준·점수·지원 선택에 사용하지 않았다.

### 7.1 Coverage와 분포

평균/median/P95는 **계산 가능한 각 항목의 표본**에 조건부다. 서로 다른 항목의 모집단이 같지 않다.

| 특징 | Full 유효 n / Coverage | Full mean / median / P95 | Partial 유효 n / Coverage | Partial mean / median / P95 |
|---|---|---|---|---|
| P1 | 24,452 /21.78% | .2012/.1060/.7118 | 42,723 /38.05% | .2105/.1197/.7242 |
| P2 | 14,771 /13.15% | .0295/.0057/.1736 | 36,274 /32.30% | .0304/.0049/.1702 |
| P3 | 24,452 /21.78% | .1536/.1244/.4033 | 51,626 /45.98% | .1619/.1333/.4067 |
| P4 | 13,445 /11.97% | .1553/.0367/.5633 | 20,295 /18.07% | .1549/.0392/.5628 |
| N1 | 8,674 /7.72% | .3811/.1983/1.0000 | 9,373 /8.35% | .3724/.1881/1.0000 |
| N2 | 17,867 /15.91% | .3520/.3025/.8271 | 50,357 /44.85% | .3239/.2664/.7976 |
| N3 | 24,452 /21.78% | .2591/.2022/.6909 | 26,788 /23.86% | .2557/.1984/.6904 |
| Positive | 24,452 /21.78% | .1555/.1333/.3795 | 50,319 /44.81% | .1445/.1203/.3864 |
| Negative | 24,452 /21.78% | .2968/.2424/.7429 | 57,590 /51.29% | .3006/.2418/.7633 |

P2는 원래 Full 중 local composition ambiguity7,748개와 geometry 지원 부족1,933개가 추가 보류됐다. P4는 local composition ambiguity11,007개가 보류됐다. N1은 local composition ambiguity15,778개가 보류됐고 N2는 경로 불명확6,585개가 보류됐다. Full이 모든 개별 관계의 신뢰성을 증명하는 조건은 아니다.

current low23,694개에서는 **모든 시간 Evidence Coverage0**다. 현재 Strict만의 종합 Full Coverage는27.60%, Partial Positive56.80%, Negative65.01%다. low의 공간 관측 자체는 품질 표시와 함께 남긴다. 낮은 QC 좌표의 정확도가 향상됐다고 주장하지 않는다.

radius3/5·품질별 전체 수치와 사유는 `validation_distributions.csv`, `validation_reason_counts.csv`에 있다. 원래 v1/v2-1 지수를 새 score의 입력으로 넣지 않았다.

### 7.2 P/N 상관·N1/N3 중복

| 공통 표본 | n | Pearson | Spearman |
|---|---:|---:|---:|
| Full N1/N3 | 8,674 | .7103 | .7405 |
| Partial N1/N3 전체 | 9,173 | .7156 | .7426 |
| Partial-only N1/N3 | 499 | .8155 | .7758 |
| Full Positive/Negative | 24,452 | −.5672 | −.6057 |
| Partial Positive/Negative | 45,058 | −.4133 | −.4345 |
| Partial-only Positive/Negative | 20,606 | −.2522 | −.2794 |

N1/N3는 중복성이 크지만 완전히 같은 값은 아니다. N1의 연속성·구성 guard가 강해 유효 표본도 다르다. 상관을 이유로 추가 가중치를 맞추거나 학습하지 않았다. Negative의 Full persistence source는 N1 8,674개/N3 15,778개, Partial은 N1 9,373개/N3 17,615개다. persistence가 없고 N2만 있는 경우도 명시적으로 남는다.

음의 P/N 상관은 독립 정의된 기하 특성이 상반된 양상에 반응한 관측 결과다. 상보 수식이나 실제 경합/비경합 분류 정확도의 증거가 아니다. 반복 frame·겹친 Anchor를 포함하므로 독립 표본의 유의성 검정으로 읽지 않는다.

### 7.3 동시에 높은 경우와 Frame 출처

사전 고정 `.5`는 분포 보고용 band이며 후보 임계값이 아니다.

* 같은 군집: Full0/24,452, Partial15/45,058=**0.0333%**가 두 종합 점수 모두 .5 이상이었다. Partial-only는15/20,606=.0728%다. 전부 보존했다.
* Validation에 군집이 존재하는23,280 Frame에서 독립 P/N max가 모두 높았던 수는 Full77/Partial308이다.
* Full77개는 전부 서로 다른 source Anchor였다. Partial308개 중300개도 source가 달랐다.
* Positive 대표 Anchor의 **그 군집 Negative**까지 높았던 Frame은 Full0/Partial9개였다. 이는 모든 same-anchor 동시-high를 세는 알고리즘은 아니다. 독립 max와 coherent 대표값의 의미 차이를 보여준다.

Frame CSV의 각 최대값에는 source radius/cluster_id, valid 수, support를 기록했다. `negative_at_positive_representative`는 같은 Positive 대표 군집의 Negative다. 서로 다른 최대값을 결합해 후보를 만들지 않았다.

### 7.4 기존 지수와 상관

아래는 같은 Validation frame 신호를 각 Anchor에 대응한 Pearson이다. v1은 전역/과거 기반, v2-1은 frame 전체 최대값이므로 현재 Anchor 밖 다른 관계일 수도 있다.

| 종합 지표 | v1 | v2-1 max3m | v2-1 max5m |
|---|---:|---:|---:|
| Full Positive | .1136 | .1274 | .1108 |
| Full Negative | −.0617 | .0260 | .0128 |
| Partial Positive | .0174 | .0918 | .0926 |
| Partial Negative | −.0837 | .0087 | .0111 |

공통 유효 분모가 신호마다 다르다. 고유 Frame max 기준 Full Positive/v1 상관은 .3114(n6,768), Partial은 .1863(n10,877)이었다. Anchor 반복 가중과 Frame max 집계 결과를 섞지 않았다. 모든 P1–N3별 Pearson/Spearman/공통n은 `validation_correlations.csv`에 있다. 낮은 상관을 기존 지수보다 정확하거나 새 의미를 검증했다는 증거로 쓰지 않는다.

## 8. Stage A와 동일 Window의 결측 통제

### 8.1 재사용·비교 조건

Stage A 기준1,778개 key와 원래 soft/nearest/3m·5m edge 시계열이 재구성 결과와 정확히 맞음을 확인했다. 30개 support block, radius3m818/5m960, seed2027을 유지했다. 1/5/10/20/30개,10개 위치 조건, 중심 보존/삭제, 연속성8조건, Train donor의 record/observed/reliable 각3회 전이를 재사용했다. 총117조건 중25개 구간 용량을 초과한12조건은 비교 불가, **105조건 계산**이다. Stage A의378개 exhaustive position sweep은 다시 수행하지 않았다.

기준 개별 Full 유효 수는 P1/P3/N3 각각1,778, P2 1,101, P4 1,013, N1 647, N2 1,316이다. 원래101개 reliable이어도 모든 개별 score가 정의되지는 않는다.

비교 target은 clean **Full**이고 estimate는 삭제 후 **Partial**이다. 기본 유효/새로 계산 가능/보류/공통 비교 가능 수를 분리한다. 종합 P는 사용 항목 mask, 종합 N은 persistence source와 N2 사용 여부가 같을 때만 같은 조합의 오차를 계산한다. P4 자체와 P4를 사용하는 Positive는 **P4 내부 term mask까지 동일**해야 한다. 항목 구성 변경을 오차0으로 기록하지 않는다.

최종 통제 진단은 `masking_final/`에 있다. 최초 진단을 보존하고 P4 내부 항목 구성 일치 검사까지 보완해 새 하위 폴더에서 재계산했다. 수식·점수·정책·sample·seed는 바꾸지 않았다. 현재 코드로 새 전체 run을 재현하면 처음부터 이 검사로 root masking CSV를 생성한다.

### 8.2 같은 삭제 수·다른 위치

* 중심을 삭제하면 Anchor 자체를 모르는 별도 실패로 남기며 모든 시간 score를 NaN 처리했다.
* 중심 보존이라도 ±1 인접 공백은 중앙 연속 guard 때문에 대부분/해당 조건 전체를 보류했다. **이는 낮은 오차가 아니라 계산 불가**다.
* 10개 Far Past 중앙 삭제에서 P1/N1/N3는 전부 보류됐다. P2/P3/P4/N2는 원래 scope가 영향을 받지 않는 표본에서 유지됐다.
* 10개 Far Future 삭제에서는 P1/P2/P3/N2가 유지되고 P4/N1/N3는 보류됐다.
* 10개 무작위 삭제의 P1은107개만 계산됐고 MAE .00548/P95 .02019였다. P4는3개만 남았다. 이 조건부 작은 오차를1,778개 전체의 안정성으로 일반화하지 않는다.

N2의 path 밖 삭제 오차0은 **동일한21개 자료를 사용하는 결과**다. 누락된 경로를 복원한 성능이 아니다. center-past/future one-run과 far run의 bootstrap 대응 비교는 중앙 정책 때문에 공통 유효0인 경우가 많아 CI를 추정할 수 없었다. null CI를 차이0으로 해석하지 않는다.

### 8.3 N1 지속 하한과 N3 extrema

Far Past1개 삭제에서 N1의 기준 유효647개를 모두 계산했지만 **MAE .04120/P95 .12862**였다. 현재 연속 prefix를 끊는 공백은 삭제한1개 외의 먼 sample들도 지속 하한에서 잃게 만든다. 중앙51개 완전 지원과 gap≤3만으로 N1의 왜곡을 작게 제한하지 못한다. 현 Partial N1은 실험 상태이며 운영 신뢰 정책으로 채택하지 않는다.

N3는 같은1개 Far Past 삭제에서1,778개 유효, MAE .001273/P95 .003790, bias+.000572였다. 정확히1개 Train-pattern 삭제에서는1,680개만 유효했고 MAE .002753/P95 .009083, bias+.002081이었다. random5는326개만 유효, MAE .002793/P95 .008541였다. 엄격한 band 조건을 통과한 부분집합 결과다. min/max/range 소실의 영향이0이거나 전체 Partial에서 안전하다고 확인한 것은 아니다.

10개 삭제의1-run/2-run/5-run/10-run 비교도 동일 수·side·평균 offset을 유지해 실행했다. 여러 공백이 중앙 band에 들어오면 아예 보류되며, 이 조건부 Evidence 결과로 순수 연속성 효과를 식별하기 어렵다. Stage A에서 측정한 raw duration 하한 손실은 여전히 유효하다. 가려진 정보를 연결·보간하지 않았다.

### 8.4 종합 점수의 항목 변경

10개 Far Past 삭제에서 Positive는1,439개가 계산 가능했지만 **모두 사용 항목이 달라** 동일 구성 비교0개였다. Negative도 N2만 남은 경우가 많아 동일 구성 비교0개였다. 10개 Far Future에서 Positive가 그대로 계산되는 경우도 P4 제외로 조합이 바뀔 수 있다. 따라서 Coverage가 높아졌다는 이유로 기존 Full 평균과 직접 대조하거나 차이가 작다고 선언할 수 없다.

Train reliable 패턴 무조건 전이 rep0의 Positive는828개 계산/동일 구성566개, Negative는944개 계산/동일 구성494개였다. 반복3회·범위별 자세한 값은 최종 `masking_distortion.csv`다. 삭제 구조와 실제 움직임의 상관을 보존하지 않는 전이이므로 실제 Tracking 오차량으로 해석하지 않는다.

관측 구성 guard의 한계도 드러났다. 세 조건에서 각각1개씩, 삭제가 구성 변화의 관측을 숨겨 **clean에서는 보류됐던 N1/P4/P2가 Partial에서 새로 계산 가능**해졌다. `newly_valid_n`에 별도로 남겼고 clean target이 없으므로 성공 사례/작은 오차로 세지 않았다. observed 시점의 인원 불변은 gap 안에서도 불변이라는 증명이 아니다. 이 문제는 v2-3 전에 검토할 미해결 사항이며 결과에 맞춰 이번 guard를 다시 조정하지 않았다.

### 8.5 한계

낮은 QC의 실제 좌표 잡음·팀 오류·누락 멤버의 이동을 재현하지 못했다. 원래 Full-reliable Validation만 기준으로 삼은 선택 편향이 있다. 짧고 매끄러운 구간, 계산 가능한 조건만의 오차가 작게 보일 수 있다. 인접 Window/같은 frame 군집/radius view는 독립 사건이 아니다. block bootstrap은 겹치지 않는500프레임 support block 단위400회이며, 한 경기30블록의 탐색적 비교다. 다른 경기로 일반화한 신뢰구간이나 탐지 정확도는 없다.

## 9. 합성 검증·실제 정합성·보존

합성 기대 동작은 사전 감사에 먼저 정의했다. 새 검사는 다음을 확인한다.

| 시나리오 | 확인한 수학적 동작 |
|---|---|
| 정적 장기 근접 | P1–P4≈0, N1/N3≈1, N2=0 |
| 일정 shape의 이동 | 명확한21개 경로,1.2m centroid 변위에서 N2≈1 |
| 접근 후 분리 | P1/P4 증가 |
| 접근 후 유지 | P1 증가, P4≈0 |
| 여러 관측 집중 | 같은 H/A 구성에서 P2 증가, 구성 변경이면 보류 |
| 기하 병합·분리 | P3 증가, ambiguity 때문에 N2 NaN |
| 중심 인접/중심 삭제 | 연속 하한 감소·Evidence 보류, 현재 삭제는 전부 NaN |
| 양끝 삭제 | P1/P2/P3/N2 등 scope에 따라 보존, N1/N3는 해당 guard로 보류 |
| 전역 인원 변화 | local 관측과 독립적인 시점/횟수, 현재 QC 실패와 구분 |
| matching ambiguity·split | N2가 불명확 경로나 경계를 건너가지 않음 |

실제 기존 geometry/matching 함수를 합성 좌표에 적용하는 integration 검사도 포함했다. Frame 내 slot 순열·H/A 교환 불변성, 숨긴 값 변경 시 Partial score 불변성, 전부 결측일 때 NaN/Inf 안전성, 중복 가중 방지, Full/Partial 완전 지원 일치, Frame 출처와 coherent pair, SciPy 없는 tie-rank Spearman, P4 항목 구성 변경 비교 제외도 검사했다.

기존377개를 시작 시 재실행해377 passed였다. 최종은 **404 passed in31.48s: 기존377+신규27, 실패0**이다. 실행 기록은 `test_results.json`에도 남겼다. 초기 부동소수점0의 exact 비교, 합성 matching fixture의 연결 key 누락은 fixture를 수정했다. 전역 count 가용성을 local observed와 혼동한 구현 문제는 합성 감사에서 수정 후 전체 추출을 새 폴더에서 다시 했다. 사전 설정값을 Validation 결과로 바꾸지 않았다.

실제 출력 verifier는 전체 Long을 다시 읽어 기존 키/Full mask, 점수[0,1], Inf0, current QC 실패 시 모든 시간 점수NaN, 종합 수식과 Negative family dedup을 확인했다. Long에서 Frame max와 모든 source key/valid count를 재집계하고 coherent Negative도 대조했다. Frame 축은0..199318 그대로다.

기존 소스·문서**210개 SHA256**, 원본/기존 산출물**8,344개 size/mtime**가 시작과 같음을 확인했다. checkpoint/baseline/Stage A 결과도 보존했다. 외부 데이터 전송, 서버, Git Commit/Push는 수행하지 않았다.

실행은 macOS arm64 CPU, Conda mtah2027, Python3.11.17, NumPy2.4.6/Pandas3.1.0이다. 새로운 모델·학습 의존성은 설치하지 않았다. 선택 라이브러리 SciPy가 없어 첫 상관 진단이 중단됐고, **average tie ranks의 Pearson으로 Spearman을 계산**하도록 수정·테스트한 뒤 완료된 추출물을 읽어 진단만 재개했다. offset 문자열의 Pandas dtype warning은 명시적 dtype/low_memory=False로 처리했다. 핵심 Evidence를 다시 튜닝하지 않았다.

최종 추출에서 Validation 파일 완성까지 약313초, 진단 재개6.67초였으며, 최종 비교 보완·검증·테스트 시간은 별도다. peak RAM은 측정하지 않았다.

## 10. 파일·실행 명령

신규 Python은 `competition_judgement/src/competition_judgement/` 내부다.

| 파일 | 역할 |
|---|---|
| `evidence_source.py` | read-only 원래 cache·matching·현재 지역 시계열/임시 경로 |
| `evidence_features.py` | P1–N3, 특징별 Full/Partial 지원·이유, 종합 mean/family dedup |
| `evidence_frames.py` | 독립 maximum·개별 출처·coherent 대표 pair |
| `evidence_diagnostics.py` | Validation 분포/상관·동일 cohort 마스킹/구성 일치 비교 |
| `run_evidence_stage_b.py` | 전체 추출·진단 CLI |
| `resume_evidence_diagnostics.py` | 완료된 추출 뒤 아직 없는 진단만 재개 |
| `diagnose_evidence_masks.py` | 새 폴더에서 통제 마스킹 진단 재실행 |
| `verify_evidence_stage_b.py` | 전체 출력 대조·Spatial 대표 추가·보존 검사 |
| `audit_evidence_delivery.py` | 최종 보존·테스트 기록·Stage A 네 시계열 bitwise 대조·코드 manifest |

설정은 `configs/evidence_stage_b.json`, 테스트는 `tests/test_evidence_stage_b.py`다. 기존 파일은 수정하지 않았다.

최종 출력 파일:

* `cluster_evidence.csv.gz`: **789,005행**, 원래 연결 키, 현재 공간 특징, 두 namespace의 세부/종합 Evidence·지원·원인·품질·matching.
* **`frame_evidence_with_spatial.csv.gz`**: **199,319행**, 독립 maxima+출처/coherent pair와 현재 Spatial 대표. 현재 대표는 N이 가장 큰 군집, tie는 작은 radius/ID다. 진단/표시용이며 후보 선택이 아니다.
* `frame_evidence.csv.gz`: Spatial 보강 이전 동일 Frame Evidence. 원래 출력도 보존했다.
* `validation_evidence.csv.gz`, `validation_distributions.csv`, `validation_correlations.csv`, `validation_reason_counts.csv`.
* `validation_joint_summary.json`, `validation_frame_joint_summary.json`: same-anchor와 독립 Frame max를 분리.
* **`masking_final/`**: 최종 통제 비교 CSV·조건·참조 keys와 Evidence·block 비교. root masking CSV도 중간 진단으로 보존했다.
* `preregistered_config.json`, `summary.json`, `verification.json`, `preservation_start.json`, `test_results.json`, `final_manifest.json`.

gzip CSV는 Pandas가 자동 해제한다. mask/offset은 문자열이다. 예:

```python
import pandas as pd
run = 'data/derived/competition_judgement/evidence_stage_b_final_20261109'
frame = pd.read_csv(f'{run}/frame_evidence_with_spatial.csv.gz')
for chunk in pd.read_csv(f'{run}/cluster_evidence.csv.gz', chunksize=10000,
        low_memory=False, dtype={'record_mask_hex': str,
        'observed_mask_hex': str, 'reliable_mask_hex': str,
        'global_count_transition_offsets': str, 'global_count_different_offsets': str}):
    # 여기서도 실제 경기 파생물은 로컬에서만 처리한다.
    pass
```

프로젝트 루트에서 기존 결과와 다른 폴더로 재현한다.

```bash
conda activate mtah2027
export PYTHONPATH="$PWD/competition_judgement/src:$PWD/MTAH_Relational_Starter:$PWD"
export PYTHONDONTWRITEBYTECODE=1
python -m pytest -q -p no:cacheprovider tests MTAH_Relational_Starter/tests competition_judgement/tests
python -m competition_judgement.run_evidence_stage_b \
  --output data/derived/competition_judgement/evidence_stage_b_reproduction
python -m competition_judgement.verify_evidence_stage_b \
  --run-dir data/derived/competition_judgement/evidence_stage_b_reproduction
```

시작 snapshot을 전달하는 `--preservation-snapshot`은 선택 인자다. 이번 실행에서는 `/private/tmp/cj_stage_b_start.json`을 사용했다. snapshot 없이 verifier를 실행하면 출력 정합성만 확인하며 기존 파일 보존을 새로 입증하지 않는다. 설정/원본/완료된 출력 폴더를 덮어쓰지 않는다. 실제 파생물은 Git ignore에 포함되며 업로드 대상이 아니다.

## 11. v2-3에 전달할 인터페이스와 남은 문제

전달 가능한 것은 `(frame,radius_m,cluster_id)`와 현재 Anchor, `full_*`/`partial_*` P1–N3/P/N, 항목 mask/가족 source/지원 scope/valid/reason/QC/시간 masks/인원 전이/검열/matching uncertainty다. Frame max에는 항목별 source key를 함께 전달한다. **Partial 항목 구성이 다른 종합 점수를 동일한 Full 기준으로 자동 임계화하면 안 된다.**

다음 판단 모듈 전에 해결해야 할 문제는 N1 Partial의 지속 하한 왜곡, 중앙 공백에서 거의 전부 보류되는 Coverage, aggregate 구성 변경의 비교 불가능성, extrema의 미관측 편향, 같은 관측 수에서 멤버 변경/팀 오류를 식별하지 못하는 한계다. N2는 기하 proxy일 뿐 실제 동일 선수 co-movement를 확인하지 못했고 P3도 실제 merge/split 정답이 아니다.

이 보고서는 관측 가능한 Evidence와 보류 이유를 전달하는 데서 끝난다. v2-3 후보 판단식·임계값 튜닝·H/A/D/X·A/B/C 학습은 구현하지 않았다.
