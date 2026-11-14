# Stage B.1 — Spatiotemporal Evidence & Pattern Architecture 사전 감사

2026-11-10. Stage1/v2-1/v2-2/Stage A/Stage B 보고서와 실제 코드·설정·출력을 확인했다. 기존223개 파일 SHA256, 원본/기존 산출물8,377개 size/mtime을 구현 전 `/private/tmp/cj_b1_start.json`에 기록했다. 기존404개 테스트를 먼저 실행한다. 이전 버전은 수정하지 않는다.

## 정확한 수식·의존성

기존 수식의 `g(x;s)=clip(x/s,0,1)`, 모든 평균은 reliable 관측만 사용한다. 원래101개 Full gate와 실제 수식의 의존 구간은 다르다.

| Evidence | 보존할 수식 요약 | 실제 수치 의존성 |
|---|---|---|
| P1 | mean(g(과거 평균 d−d0;3m),g(S0−과거 S;.25),g(near S−far S;.25)) | −50..0 |
| P2 | 반경/지름 축소, contact/edge/density 증가5항 평균, 원래 scale 유지 | −25..0, geometry≥20·구성 gate |
| P3 | 지역 K/density 변화와 복수 matching 후보항3개 평균 | −25..25, adjacent±1 포함 |
| P4 | 미래 soft/거리 분리·known end·가능한 지름 확장 available mean | 0..50. 기존 duration 함수가 past도 반환하지만 P4는 future reason만 사용 |
| N1 | 연속 근접 하한/4.04 × count/density 안정 × 접근·분리 부재 | 전체101개 |
| N2 | centroid 순변위/1m × exp(−D range/1m−R range/.5m−density range/.25) | ±10 기하 경로21개. 고정 원 inside 관측과 다른 지원 basis |
| N3 | occupancy×exp(−S std/.1)×(1−S range)×(1−peak/.25 clip) | 전체101개 |

기존 P/N 가변 평균은 사용 항목과 Negative family가 바뀌어 숫자의 의미가 달라진다. B.1은 이를 Legacy 파일로만 참조하며 새 profile의 입력으로 쓰지 않는다. 새 cross-Evidence 평균/차감/점수 bounds는 만들지 않는다.

## 관측 정책

COMPLETE는 위 의존 구간 전체의 reliable 지원과 기존 feature별 구성/기하 조건을 요구한다. P1/P2의 필요 없는 미래 때문에 COMPLETE를 차단하지 않는다. N2는 원래 reciprocal-unique 경로의 QC·global/team-count·frame/half/split·구성 검사를 재사용하며 고정 원의101 mask로 대신하지 않는다.

COMPLETE가 아니면 기존 Stage B Partial 정책을 **그대로** 적용한다. 이 정책의 P1/P2 중앙±10 추가 요구도 유지한다. N1의 불완전 Window는 반드시 EXPERIMENTAL_PARTIAL이고 범위/연속성 불확실성이 남는다. 정의 불가/조건 실패는 UNKNOWN+NaN이다. 기존 Strict/count 정의를 완화하거나 결측을 채우지 않는다.

N1은 한 프레임 결측에도 MAE .0412/P95 .1286이었고 N3 extrema는 부분 관측으로 보장되지 않는다. Stage A/B에서 결측 위치·연속성과 관측 구성 숨김 문제가 확인됐다. 같은 인원은 같은 사람이라는 보증이 아니다. Anchor 이탈과 누락은 ambiguous다.

P3는 K/density의 구조 변화와 복수 후보항·매칭 ambiguity를 별도 필드로 분리한다. matching ambiguity만으로 실제 merge/split을 확정하지 않는다. N2는 영구 ID가 아니며 같은 선수의 동반 이동을 확인하지 않는다.

## 패턴·시간 규칙 — 결과 보기 전에 고정

설정은 `configs/pattern_b1.json`이다. Evidence 크기를 high/low로 나누지 않는다. Layer1은 확보된 Evidence bitset/상태만 저장한다. Layer2는 원래 raw 시계열에서 아래 실험용 기하 조건을 검사한다.

5프레임 trailing median, lag5(각0.2초)를 사용한다. soft .02/거리 .1m는 Stage A deadband, D의 motion/shape 한계는 기존 Stage B physical scale을 재사용한다. noise 오차 보장값이나 축구 의미 calibration이 아니다. 실제 시계열이 없거나 sequence용 필요 구간에 공백이 있으면 UNVERIFIED로 보류한다. 임의 보간/0 대체는 하지 않는다.

* A(P1/P2): 접근과 공간 집중 flag를 별도로 보존. 접근은 과거 평균 대비 S증가/d감소와 가까운 과거의 실제 lag 증가, 집중은 PN 대비 R/D축소 및 실제 −5→0 수축 또는 contact/edge 증가다. 둘 중 하나가 확인되면 A를 지원한다. 필수 Evidence가 부족하면 부분 근거만 남기고 전체 A는 UNKNOWN이다.
* B(P1/P3/P4): 전체 reliable101개에서 smoothed 최고점이 중심±5, far-past→peak→far-future의 S/d 변화와 실제 lag rise/fall의 시간 순서, 실제 K/E 변화가 함께 있어야 한다. P1/P4나 복수 매칭 후보만으로 VERIFIED를 만들지 않는다. 재접근은 별도 observed off→on flag다.
* C(N1/N3): 전체101개 연속 close, S range≤.02/d range≤.1m, K/E 변동 없음인 안정적 지역 관계다. N1/N3는 합산하지 않는다.
* D(N2): 원래21개 명확한 기하 경로에서 순변위≥1m, D/R/density range가 기존1m/.5m/.25 scale 이내다. 같은 Spatial Anchor 원칙을 위해 모든 경로 centroid가 현재 seed 원 안에 있어야 D를 결합한다. 경로가 원 밖이면 N2 값은 보존하되 D는 scope mismatch로 UNKNOWN이다.

패턴마다 Observation(COMPLETE/EXPERIMENTAL_PARTIAL/UNKNOWN), Sequence(VERIFIED/UNVERIFIED/NOT_APPLICABLE), Outcome(SUPPORTED/NOT_OBSERVED/UNKNOWN)을 분리한다. 자료가 충분하나 조건 불충족이면 NOT_OBSERVED, gap/의미 없는 입력이면 UNKNOWN이다. VERIFIED는 정의한 기하 조건만 뜻한다. 패턴은 비배타적이며 대표1개 유형을 강제하지 않는다.

## 출력과 검증 계획

동일 군집 key 안에서만 profile을 만들고 Frame은 pattern별 출처와 radius별 Anchor-view 수를 따로 저장한다. 다른 군집 또는3/5m의 Evidence를 합성하지 않는다. 독립 max로 같은 사건을 만들었던 Stage B의 해석 위험을 제거한다. 다른 Anchor 중 하나라도 판정 불가이면 frame의 전역적 패턴 부재를 확정하지 않는다.

실제 출력은 새 `evidence_pattern_b1_final_20261110/`, 데이터 사전은 별도 문서다. Legacy 종합 점수/used mask/family source는 기존 key를 참조하는 읽기 전용 별도 sidecar로 보존한다. 공통 masks는 한 번만, N2의 다른21개 mask만 별도로 둔다. 시계열 배열을 복제 저장하지 않는다.

Validation은 기존 분할·Stage A1,778 cohort/Train donor/seed와 기존12×151 perturbation support를 재사용한다. jitter/dropout/composition 뒤 군집 ID 동일성을 강제하지 않고 같은 fixed Anchor에서 수치 비교하며 N2/P3 연결은 명확한 현재 기하 대응이 있을 때만 허용한다. Test/GT/볼/태깅/실제 영상 라벨은 선택에 사용하지 않는다. 낮은 Coverage도 그대로 보고한다.
