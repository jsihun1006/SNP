# Stage B.2 — Local Temporal Evidence Extension

2026-11-10. 설계 감사, 독립 구현, 합성 검증, 전체 로컬 실행 및 Validation 진단을 완료했다. **전체 패턴을 판단할 수 없어도, 사전에 정한 짧은 구간에서 실제 관측된 기하 변화만 보존한다.** 전체 Pattern A/B/C/D의 정의·Outcome과 P1~N3는 그대로다. Local 결과는 경합 정답·확률·후보 판정이 아니다.

## 1. 완료 결과

- 전체 **199,319 Frame / 789,005 군집 관측**, 공식 구간745,891개를 동일 key로 유지했다.
- Validation112,286개에서 Short Rise/Fall 계산 가능 비율은 **59.71% / 60.02%**, Near는 **40.36% / 40.26%**다. 기존 Full101 정책을 완화한 비율이 아니라, 서로 다른 의존 구간의 지원 비율이다.
- B=UNKNOWN98,841개 안에서 같은 Anchor의 Local Rise→Fall을 **Short1,498개 / Near3,252개** 확인했다. 기존 B는 모두 UNKNOWN으로 보존했다.
- D=UNKNOWN61,929개 중 **38,115개**에서21개 미만의 현재 연결 경로를 확보했다. 짧은 경로에1m 기준을 적용하지 않았다.
- 기존 P1~N3와 A/B/C/D를 포함한 B.1의 모든 재계산 profile 열을 원래 CSV와 대조했다. **수치 최대 차이0, Outcome·mask 문자열 동일**이었다.
- 기존440개를 포함한 **477개 테스트 통과**(신규37개,33.14초). 최종 실패/미해결 테스트는 없다.
- Stage A1,778개/30블록 cohort와 원래 배열이 bitwise 동일했다. 삭제105조건을 계산했고, 불가능한12조건은 INCOMPARABLE로 유지했다. Tracking9조건은 원래12×151프레임/612개 중심/3,191개 Anchor probe를 재사용했다.

전체 실행은 `mtah2027`, Python3.11.17/NumPy2.4.6/pandas3.1.0/pytest9.2.0, arm64 CPU에서 **369.64초** 걸렸다. 학습이나 추가 의존성 설치는 하지 않았다.

## 2. 사전 감사와 보존

[설계 감사](STAGE_B_2_DESIGN_AUDIT.md)와 [고정 설정](../configs/local_temporal_b2.json)을 구현·Validation 분석 전에 작성했다. 기존 B.1의440개 테스트를 먼저 통과시켰다(31.35초). Validation 결과에 따라 기준을 수정하지 않았다.

B.1의 source/values/profiles/frames/diagnostics/runner/verifier와 설정, Stage A/B의 missingness·temporal 정의 및 보고서, 공식 RULES.md를 확인했다. 읽기 전용 player/pair cache와 원래 공간 군집·reciprocal-unique matching을 재사용했다. ATD를 다시 파싱하거나 pair cache를 재생성하지 않았다. 합성 tracking 진단에서만 player cache의 로컬 복사본으로 공간 군집을 다시 계산했다.

기존235개 코드/문서 SHA256와8,408개 원본·파생 파일의 size/mtime이 모두 보존됐다. 사용자 작업 중이던 파일, A/B checkpoint, 원본, 제출용 baseline도 변경하지 않았다. Test는 전체 정렬 출력/보존 대조에만 포함했고 정책 선택·분포·민감도 분석은 기존 Validation에 한정했다. 볼·Smart Tagging·GT·영상 라벨, 서버·외부 전송, Git Commit/Push는 사용하지 않았다.

## 3. 수학적 정의와 관측 조건

### 공통 입력

원래 현재 군집의 중심을 고정하고 Anchor 반경 `a=max(6m,현재 군집 diameter+3m)`를 유지한다. 각 시간에 이 원 안의 관측을 다시 선택한다.

`S(u)=max exp(-d_HA(u)/3m)`, `d(u)=-3log S(u)`이며 S는 무차원, d는 m다. **S와 d는 같은 최근접 관계 정보의 변환이지 독립적인 두 증거가 아니다.** 최근접 상대 쌍이 바뀔 수 있고 같은 관측 수가 같은 선수라는 보증도 없다.

record는 같은 half/split/연속 기록, observed는 Anchor 안 양 팀 존재, reliable은 observed와 현재/해당 offset의 Strict QC 및 현재 대비 전역 H/A 관측 수 일치다. 기존 정의를 바꾸지 않았다. 존재하지 않는 offset은 다른 split의 값을 가져오지 않고 현재 sentinel을 gather하되 record=False로 제외한다.

| Scope | 원래 offset | 관측 수 | sample span | 첫/마지막 median 지원 |
|---|---|---:|---:|---|
| Past Short | -10..0 | 11 | .4s | -10..-6 / -4..0 |
| Past Near | -25..0 | 26 | 1s | -25..-21 / -4..0 |
| Future Short | 0..10 | 11 | .4s | 0..4 / 6..10 |
| Future Near | 0..25 | 26 | 1s | 0..4 / 21..25 |

각 scope를 독립 평가한다. 한쪽이 실패했다고 유리한 Short/Near를 대표로 선택하지 않는다. scope 바깥 값을 smoothing에 넣지 않는다.

### L1 / L2 — Local Rise / Fall

scope 내부 trailing median5를 `mS(j), md(j)`라고 한다.

`ΔS=mS(last)-mS(first)`, `Δd=md(last)-md(first)`.

`lagS=mean_j(mS(j+5)-mS(j))`, `lagd=mean_j(md(j+5)-md(j))`.

평균은 scope 안의 **모든** lag5 비교를 쓴다(Short2쌍, Near17쌍). 가장 큰 상승·하락 구간을 검색하지 않는다.

- Rise: 과거 scope의 `ΔS≥.02`, `Δd≤-.1m`, `lagS>tol`, `lagd<-tol`.
- Fall: 미래 scope의 `ΔS≤-.02`, `Δd≥.1m`, `lagS<-tol`, `lagd>tol`.
- `tol=1e-12`는 수치 tolerance다. .02/.1m는 기존 deadband에서 가져온 사전 기하 기준이며 실제 Tracking 오류의 통계적 보증이 아니다.

전체 scope가 reliable이고 S/d가 finite이며 지역 H/A 관측 수가 현재와 같아야 판정한다. 충분한 지원에서 조건 불충족이면 NOT_OBSERVED(실제 변화량0 포함), 결측/QC/지역 구성 변화면 UNKNOWN이다. 지역 구성 변화는 실제 이탈·교대 관측·누락을 구분할 수 없어 보류한다. 미관측 값은 median 계산 전에 NaN으로 가리고 UNKNOWN 변화량을 출력하지 않는다. Fall은 상대 팀 관측 소멸을 분리로 해석하지 않는다.

### L3 — Local Persistent Proximity

현재 radius(3/5m)의 E가 해당 scope 모든 관측에서 양수이고 전체 scope reliable이면 OBSERVED다. 충분히 관측됐으나 known E=0가 있으면 NOT_OBSERVED, 지원이 부족하면 UNKNOWN이다. **S/d/K/E의 안정성은 이 조건에 넣지 않는다.** 따라서 기존 C와 다르다.

별도로 현재부터 과거/미래 방향으로 이어진 `reliable AND E>0`의 연속 prefix만 보존한다. 첫 unknown 또는 known non-close에서 멈춘다. 공백 뒤의 값은 연결하지 않는다.

- 관측된 연속 프레임 수 n, 첫/끝 offset.
- frame-bin 지속 하한 `n/25`초. sample span `(n-1)/25`초와 다르다.
- 현재 자체가 unknown이면 하한 NaN, 현재 known non-close이면0.
- unknown 또는 고정 scope 끝에서 끊기면 censored, known non-close에서 끝나면 그 방향의 종료는 관측됨.
- 실제 연속 prefix 안의 S/d/K/E range를 별도 기록한다. 이 prefix 밖 정보는 미확인이다.

지역 근접의 연속 존재만 확인하며 동일 선수 쌍·안정적 구조·실제 마킹을 확인한 것은 아니다. 구성 변화가 있어도 지역 E의 연속 존재라는 관측은 남기고 변화 flag/range를 함께 제공한다.

### L4 — Local Motion

기존21개 reciprocal-unique 경로 안에서 현재를 포함하는 **최대 연속 유효 부분**을 선택한다. 선택 기준은 관측 가용성뿐이며 큰 변위를 찾지 않는다. 원래 frame 간격/half/split/QC/전역 및 matched component H/A 수 조건을 유지하고 각 centroid가 현재 seed 원 안에 있어야 한다. 모호한 링크·공백·구성 변경·원 밖에서 멈추며 이후 경로에 다시 연결하지 않는다.

2개 이상 연속 sample이면 다음을 보존한다.

- `motion_net_m=||c_last-c_first||`.
- `motion_path_length_m=Σ||c_(j+1)-c_j||`.
- `motion_span_s=(last-first)/25`.
- 해당 경로의 R/D/density range, 첫/끝 offset, retained sample 수와 양쪽 검열.

경로 길이가 수치 tolerance보다 크면 OBSERVED, 완전히 정지하면 NOT_OBSERVED,2개 미만이면 UNKNOWN이다. **짧은 경로에1m 또는 shape 안정 한계를 적용하지 않았다.** 이 상태는 비영(非零) centroid 이동을 측정했다는 의미뿐이다. jitter·구성 교체도 이동을 만들 수 있으며 실제 동일 선수들의 Co-Movement를 확인한 것이 아니다. 현재 인접 ambiguity flag와 경로 지원/보류 사유를 저장한다. 더 먼 링크의 정확한 실패 원인은 기존 cache/matching에서 재구성해야 한다.

## 4. 전체 Pattern과 Local Sequence의 차이

같은 Anchor row 안에서 Past Rise와 Future Fall의 3×3 상태를 Short/Near별로 저장한다. O/O만 `LOCAL_RISE_THEN_FALL`, O/U는 `RISE_OBSERVED_FALL_UNKNOWN`이다. 나머지도 조합별로 보존한다. 현재를 사이에 둔 고정 시간 순서만 확인하며 전체101프레임의 peak가 현재 근처였다는 보증은 없다.

Validation의 Local Rise→Fall은 Short1,649개 / Near4,772개다. Short에서 Rise만 관측되고 Fall은 unknown인4,711개도 버리지 않았다(Near9,130개). 두 scope를 합쳐 단일 사건 수를 만들지 않는다.

기존 Validation A/B/C/D SUPPORTED는 **8,641 / 469 / 0 / 23,055개**로 동일하다. A의 접근·집중, B의 full peak/order/구조, C의101개 안정성, D의21개1m/shape 검증은 바뀌지 않았다. Local R/D/contact/K/E의 raw endpoint 변화도 같이 제공하되 원래 P2나 전체 A의 확정을 대신하지 않는다.

## 5. Validation Coverage

분모112,286개 군집 관측이다. 겹친 시간·Anchor 및3/5m view가 포함되므로 독립 사건 수가 아니다.

| Local 결과 | OBSERVED | NOT_OBSERVED | UNKNOWN | 계산 가능 |
|---|---:|---:|---:|---:|
| Rise Short | 19,676 | 47,370 | 45,240 | 59.71% |
| Rise Near | 22,418 | 22,905 | 66,963 | 40.36% |
| Fall Short | 18,606 | 48,793 | 44,887 | 60.02% |
| Fall Near | 21,440 | 23,763 | 67,083 | 40.26% |
| Proximity Past Short | 73,349 | 5,311 | 33,626 | 70.05% |
| Proximity Past Near | 54,136 | 10,768 | 47,382 | 57.80% |
| Proximity Future Short | 73,572 | 5,088 | 33,626 | 70.05% |
| Proximity Future Near | 54,651 | 10,687 | 46,948 | 58.19% |
| Motion | 88,472 | 0 | 23,814 | 78.79% |

3m/5m view의 Short Rise 계산 가능 비율은56.93%/61.22%, Near는36.97%/42.22%였다. 현재 Strict88,592개에서는 Short Rise75.68%, Near51.16%, Motion99.86%였다. Low-quality23,694개는 모든 Local 판정을 UNKNOWN으로 보존했다.

Motion은21개 연속50,357개,2~20개 연속38,115개다. 관측 가능한88,472개 전부에서 수치상 비영 이동이 있었다. **높은 Motion Coverage가 변별력이나 의미 있는 이동을 뜻하지 않는다.** 이 단계에서는 raw geometry 보존이 목적이며 noise scale에 근거 없는 Co-Movement threshold는 만들지 않았다.

## 6. 기존 UNKNOWN에서 추가 확보한 정보

아래는 전체 패턴 UNKNOWN인 군집 안에서 Local이 OBSERVED였던 수다. 항목·scope·패턴끼리 중복되며 합산하지 않는다.

| 기존 패턴 UNKNOWN | 분모 | Rise Short / Near | Fall Short / Near | Rise→Fall Short / Near | Proximity Past Short / Near | Motion |
|---|---:|---:|---:|---:|---:|---:|
| A | 86,846 | 14,262 / 11,740 | 12,619 / 13,572 | 1,206 / 2,279 | 47,982 / 29,586 | 63,040 |
| B | 98,841 | 16,478 / 17,257 | 15,596 / 15,803 | 1,498 / 3,252 | 60,748 / 42,868 | 75,028 |
| C | 103,612 | 17,529 / 18,099 | 16,691 / 17,630 | 1,548 / 3,461 | 65,133 / 46,685 | 79,799 |

D UNKNOWN61,929개 중 Local Motion38,115개를 확보했다. 이는 충분한21개 경로나1m/shape 조건을 확인했다는 뜻이 아니다. C가0개인 결과도 완화하지 않았다. 짧은 지역 근접의 존재와4초 전체 구조 안정성은 다른 관측이다.

## 7. 결측·Jitter·Dropout 민감도

### 동일 원본·동일 삭제 조건

Stage A의 고정 Full-reliable1,778개/30개 블록 및 원래 seed2027, Train donor library를 재사용했다. 단, Full-reliable이어도 지역 구성 변경 또는 경로 대응 때문에 B.2는 처음부터 UNKNOWN일 수 있다. 깨끗한 cohort의 Short Rise UNKNOWN242개, Near511개였다. Full101 reliable과 Local의 의미 조건을 혼동하지 않는다.

결측1/5/10/20/30개, 위치10조건×현재 삭제 여부, 연속성8조건, 실측 donor9조건을 시도했다.105조건 계산/12조건 비교 불가이며,25개 band에30개를 줄여 넣지 않았다.

- 먼 과거 또는 먼 미래1개 삭제는 모든 Local 판정과 raw값을 바꾸지 않았다. 필요 없는 구간의 공백을 패턴의 공백과 분리했다.
- offset−1 하나 삭제 시 Past Rise/Proximity는 전부 UNKNOWN이지만 Future는 원래 결과를 보존했다. offset+1은 반대다.
- 현재 삭제 시 모든9개 Local 판정이 UNKNOWN이다. Anchor 자체를 잃은 경우를 정상 점수와 비교하지 않았다.
- random1개 삭제에서 Short Rise UNKNOWN은242→416, Near는511→823이었다. 동일 결측률에서도 위치에 따라 확보 정보가 달랐다.
- 삭제 후에도 계산 가능한 고정 구간의 ΔS/Δd/lag5 값은 **모든 조건에서 변화 MAE0**이었다. 이는 완전 구간만 계산하기 때문이며, missing-value 복원이나 Tracking noise 강건성을 입증한 것이 아니다.
- Past Short 지속 하한 변화는 far-past1개에서 MAE0, random1개에서MAE.0211s/P95.20s, offset−1에서MAE.3828s/P95.40s였다. **실제 지속시간 추정 오류가 아니라 삭제로 줄어든 관측 하한의 변화**다. retained run 양끝이 같은 경우의 MAE는0이었다.
- 단발/긴 공백 모두 고정 direction 구간 안에 들어오면 UNKNOWN이 된다. 반면 prefix/Motion은 첫 공백 위치에 따라 retained 범위가 달라진다. 서로 다른 범위의 raw변위/기간 변화와 같은 범위의 변화는 `same_scope_n/mae`로 분리했다.

### Tracking perturbation

기존 B.1과 같은3,191개 fixed clean Anchor probe다. 재검출 군집은 명확한 reciprocal-unique bridge만 허용하고 강제 ID 연결을 하지 않았다. Jitter는 모든 관측에, Dropout은 전체 support의 각 frame에 적용된다. composition은 기존 함수대로 각151개 support의 offset65..85에서만 관측 삭제를 적용한다. 이는 합성 진단이며 실제 오류 분포의 추정이 아니다.

| 조건 | Short Rise O / U | Short Rise 상태 변경 | 공통 Δd 표본 | Δd 변화 MAE(m) | Motion O |
|---|---:|---:|---:|---:|---:|
| Clean | 634 / 1,074 | 0 | 2,117 | 0 | 2,595 |
| Jitter .1m | 677 / 1,164 | 339 | 2,017 | .0855 | 2,498 |
| Jitter .5m | 581 / 1,830 | 1,282 | 1,358 | .3971 | 2,023 |
| Jitter 1m | 316 / 2,444 | 1,684 | 747 | .7347 | 1,236 |
| Dropout10% | 0 / 3,191 | 2,117 | 0 | NaN | 297 |
| Dropout20% | 0 / 3,191 | 2,117 | 0 | NaN | 38 |
| Dropout30% | 0 / 3,191 | 2,117 | 0 | NaN | 28 |
| Composition20% | 227 / 2,373 | 1,299 | 818 | 0 | 1,544 |

Jitter .1m에서 Rise OBSERVED가634→677로 늘었지만 신규150개·소실107개를 포함한 상태 변화이며 개선이 아니다. Soft 변화 MAE는.01356/.05826/.08805였다. 강도별 공통 유효 표본이 다르므로 MAE를 무조건 같은 조건으로 비교할 수 없다.

Motion의 누적 경로 길이는 특히 jitter에 민감했다. 유지된 경로 범위가 같아도.1m에서MAE1.378m(n=1,893),.5m에서10.713m(n=607)의 변화가 있었다. 비영 이동 관측은 실제 이동 방향·동일 집단·의미 있는 속도의 증명이 아니다.

Dropout에서는 고정 구간의 방향/Proximity가 모두 UNKNOWN이었다. 현재/offset QC, 전역 관측 수 일치와 연속 국소 지원을 유지했기 때문이다. Local Motion은 일부 남았지만10%에서 신규OBSERVED38개도 발생했다. 관측 소실로 모호했던 대응이 바뀌는 현상을 개선으로 부를 수 없다. 삭제 후 raw 방향 값이 계산 불가능하면 변화 MAE도NaN이며0으로 대체하지 않는다.

Slot 순열과Home/Away 교환에서는 **모든 Local 열의 수치·상태가 불변**이었다. 실제 Tracking 오차나 같은 인물의 추적을 확인한 검증은 아니다.

## 8. 구현과 출력 인터페이스

신규 Python은 아래6개 모듈과1개 테스트 파일이다. 기존 코드의 변경은 없다.

| 파일 | 입력·출력과 역할 |
|---|---|
| [local_temporal_source.py](../src/competition_judgement/local_temporal_source.py) | 기존 read-only source 확장. 시계열[N,101] 및 일시적인 path 좌표[N,21,2]/지원[N,21] 반환. 영구 ID 없음 |
| [local_temporal_features.py](../src/competition_judgement/local_temporal_features.py) | named series+radius[N]→Local 표[N,138]. scope별 상태/지원/원시 변화/sequence |
| [local_temporal_frames.py](../src/competition_judgement/local_temporal_frames.py) | 출처를 보존한 존재 여부를 전체 Frame으로 집계. 다른 Anchor의 Rise/Fall을 합성하지 않음 |
| [local_temporal_diagnostics.py](../src/competition_judgement/local_temporal_diagnostics.py) | Validation 분포, UNKNOWN 안의 정보 회수, 고정 cohort 삭제/Tracking 대조 |
| [run_local_temporal_b2.py](../src/competition_judgement/run_local_temporal_b2.py) | 전체 추출, 동결 B.1 재계산 대조, 진단·사전·summary 생성 |
| [verify_local_temporal_b2.py](../src/competition_judgement/verify_local_temporal_b2.py) | 출력 재독해, key/Frame source/sequence/연속 지원·보존 검증 |
| [test_local_temporal_b2.py](../tests/test_local_temporal_b2.py) | 지정20개 상황을 포함한37개 합성 검증 |

설정은 `configs/local_temporal_b2.json`, 감사는 `docs/STAGE_B_2_DESIGN_AUDIT.md`다. 신뢰 mask/원래 Evidence/원시 시간 배열은CSV에 복제하지 않았다. 메인 Sidecar는 **789,005×141열(key3+Local138)**, Frame은 **199,319×60열**이다. 압축 크기는 약128MiB/2.6MiB다.

Local outcome은OBSERVED/NOT_OBSERVED/UNKNOWN이고, reason은scope 공통 및direction/motion별로 제공한다. Past/Future Proximity와Short/Near sequence를 구분하기 위해 지시문의 예시인 단일 `local_proximity_short`/`local_sequence_code`로 합치지 않았다. 정확한 열 이름·bit·단위는 실행 폴더의 `data_dictionary.json`에 기록했다.

`local_support_code`는4개scope의 완전 지원과Motion 최소2점 지원의bitset이며 강도/확률이 아니다. scope별 관측 수/최장gap/첫끝/연속성/구성 변화, prefix의 관측 하한/검열, Motion 원시 기하와ambiguity를 보존한다. 원래 QC와mask는 동결 B.1을key로join한다.

Frame은signal별 출처 `(frame,source_radius_m,source_cluster_id)`와radius별view 수를 가진다. 어떤 Anchor의Rise와 다른 Anchor의Fall이 모두OBSERVED여도 같은row에서 만든sequence가 없으면RF로 만들지 않는다. sequence-view-mask는 같은row에서 계산한9가지sequence의 존재 집합이며, Frame 전체에 하나의sequence를 가정하지 않는다. Anchor가 없는Frame의Local은UNKNOWN이다.

## 9. 출력·재현·검증

최종 폴더: `data/derived/competition_judgement/local_temporal_b2_final_20261110/`.

지정한10개 파일(cluster/frame CSV.gz, coverage/recovery/masking/tracking CSV, legacy preservation/verification/summary/dictionary JSON)을 모두 생성했다. 추가로config, masking 조건, tracking bridge, 보존snapshot, LOCAL_ONLY.txt를 저장했다.

`verification.json`에서 전체Cluster key 순서, Frame 번호0..199318, scope 완전 지원, Motion 연속성, Frame source/sequence 재현, 기존 파일 보존 및477개 테스트를 확인했다. 실제 경기에서도 전체 실행을 완료했다. 실제 경합/점유 정답에 대한 정확도는 검증하지 않았다.

프로젝트root에서 아래와 같이 재현한다. runner/verifier가 기존 폴더 덮어쓰기를 거부하므로 새로운 출력 이름을 사용한다.

```bash
conda activate mtah2027
export PYTHONDONTWRITEBYTECODE=1
export PYTHONPATH="$PWD/competition_judgement/src:$PWD/MTAH_Relational_Starter:$PWD"

python -m pytest -q -p no:cacheprovider tests MTAH_Relational_Starter/tests competition_judgement/tests > /private/tmp/cj_b2_reproduction_tests.log 2>&1

python -m competition_judgement.run_local_temporal_b2 \
  --output data/derived/competition_judgement/local_temporal_b2_reproduction

python -m competition_judgement.verify_local_temporal_b2 \
  --run-dir data/derived/competition_judgement/local_temporal_b2_reproduction \
  --preservation-snapshot data/derived/competition_judgement/local_temporal_b2_final_20261110/preservation_start.json \
  --tests-log /private/tmp/cj_b2_reproduction_tests.log
```

이번 실행의pytest 로그는 `/private/tmp/cj_b2_verified_tests.log`, 보존된 집계 결과는 `verification.json`에 있다. 추가 테스트 작성 중 합성FrameStore의read-only period 배열을 직접 변경하려던 검사가 실패했고, 합성fixture의 복사본을 사용하도록 수정했다. 이를 위해 원본이나 제품 코드를 바꾸지 않았다.

## 10. 한계와v2-3에 전달할 범위

확인한 것은 국소 방향·연속 근접·기하 이동이라는 추가 정보와 그 관측 지원·결측/노이즈 민감도다. 정보 복원, 개인ID 또는 실제 마킹/볼 경합을 식별하는 능력을 확인한 것은 아니다.

공간Anchor에는 다른 상대 쌍·군집이 들어오고 나갈 수 있다. 관측 인원이 같아도 구성 교체를 검출하지 못한다. 관측 부재가 지역 이탈인지Tracking 누락인지도 여전히ambiguous하다. 현재 근처의 한 프레임 결측으로 고정scope가UNKNOWN이 되어 짧은 방향 정보도 잃는다. 관측 정책을 완화하지 않고 보수성을 유지한 결과다.

합성 삭제는 실제 결측의 비무작위성이나 좌표 정확도를 재현하지 못한다. Jitter의 신규OBSERVED, Partial Motion의 높은Coverage나 누적 이동은 성능 개선 또는False Positive 감소의 근거가 아니다. 겹치는Window·Anchor·3/5m view는 상관된 반복 측정이다. 이 표는 기술 통계이며 독립 표본 검정이나 신뢰구간을 주장하지 않는다.

향후v2-3에는 `(frame,radius_m,cluster_id)`로 동결B.1과Local Sidecar를join하여 **원래Pattern 상태, Local별O/N/U, 실제scope/지원/구성·matching 불확실성, raw 변화, 같은Anchor의sequence**를 각각 전달할 수 있다. UNKNOWN을0으로 바꾸거나, Local로 전체Pattern을 승격하거나, Motion을Co-Movement 확정으로 취급하면 안 된다. 후보식·가중치·임계값 및H/A/D/X 판단은 이번에 구현하지 않았다.
