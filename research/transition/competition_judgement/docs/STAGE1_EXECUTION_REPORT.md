# Stage 1 실행 보고서 — 선수 관계 기반 경합 후보

작성일: 2026-11-09. **Roadmap Stage0/1만 구현·실행했다.** A/B/C의 공간 복원 개선은 중단하고 기존 모델을 동결 보존했다. 이 모듈은 상황별 점유 판단의 독립적인 후보 신호다. 실제 경합 확률·정답·H/A/D/X·점유 전환을 출력하지 않는다.

## 1. 실행 결과와 해석

실제 tactical199,319프레임을 로컬 CPU에서 전부 처리했다. 공식158,857프레임 안에서 **후보9,936프레임(6.2547%), 비후보101,797프레임(64.0809%), 관측 불충분47,124프레임(29.6644%)**이었다. 후보446개 구간의 총 출력시간397.44초, 평균0.8911초, 중앙값0.56초, P95=3.57초, 최대9.12초다.

이 빈도는 규칙이 선택한 구간의 양이다. 실제 경합 빈도·정밀도·recall·점유 개선이 아니다. `not_candidate`도 실제 비경합 정답을 뜻하지 않는다.

## 2. 감사·재사용·문서 차이

[Stage0 감사](STAGE0_AUDIT.md)에 시작 문서, docs/00–05, references/의 모든7개 문서, 공식 RULES/FAQ/CHALLENGE/DATA/EVALUATION, 실제 Python API와 문서 차이를 정리했다. references는 과거 스냅샷이며 현행 코드를 우선했다. A/B 후속 진단까지 존재하지만 C/Fusion은 구현하지 않았다.

재사용한 경로:

| 기존 파일/함수 | 사용 범위 |
|---|---|
| `games.py:half_video_blocks`, `GAMES`, `asset_paths` | 공식25fps·105.3×68m·전후반·전체 frame 계약 |
| `data_quality.py:read_metadata`, `assess_observations`, `iter_player_frames` | Local/Visiting 매핑·슬롯/단위·품질 정책; 파싱 재구현 없음 |
| `features.py:geometry_features`, `max_bipartite_cluster_size` | 전역3/5/10m 쌍·거리·최근접·팀 구조·혼합 component |
| `candidate_baseline.py:CausalContestedIndex` | 원래 수식 그대로 재현, 신경망/fit 없음 |
| `phase1.py:run_phase1`와 기존 features.csv | 기존 전 프레임 기하·품질 재사용; 파일 부재 시 기존 reader로 새 폴더에 생성 |
| `frame_store.py:open_frame_cache` | 원본 player-only cache를 read-only로 열고 robustness 입력으로 사용 |
| `splits.py:SplitManifest`, `make_split_manifest` | 기존 반별70/15/15 시간 블록·fingerprint 대조 |
| `zones.py:zone_metrics`, 기존 zones_48.csv | 48구역 보조 통계; 전역 쌍에 구역 경계 적용 안 함 |
| `spatial_schema.py`, `preprocessing.py` | 수학적 validity/strict eligibility 구분과 팀별 구조 보존 원칙 확인 |

개인 ID가 아닌 source slot을 연결하는 과거 벡터/속도 함수는 사용하지 않았다. 기존 팀 구조를 사용하고 `CausalContestedIndex`의 집합 접근 변화와 같은 방식으로 집합 통계의 과거 변화만 계산했다. 모델의 scalers/embeddings/checkpoints는 입력이 아니다.

## 3. 새 구현과 입력·출력

신규 Python은 모두 `competition_judgement/` 아래에 있다.

| 파일 | 역할 |
|---|---|
| [quality.py](../src/competition_judgement/quality.py) | 기존 player-only feature allowlist·frame/QC adapter·strict/metric 구분 |
| [detector.py](../src/competition_judgement/detector.py) | 원래 지수 재현·split-aware causal 변화·Train quantile·지속 조건·상태/reason |
| [artifacts.py](../src/competition_judgement/artifacts.py) | 전체 CSV·구간 분할·품질/빈도/지속 집계·기존 zone 샘플 연결 |
| [diagnostics.py](../src/competition_judgement/diagnostics.py) | 고정 Validation support와 seed의 원본 보존 perturbation·민감도 |
| [cli.py](../src/competition_judgement/cli.py) | Stage0/1 실행·출처/원본 검증·새 로컬 폴더 원자적 저장 |
| [test_stage1.py](../tests/test_stage1.py) | 경기 파일 없는 신규 합성31개 테스트 |

패키지 `__init__.py`, README와 tests/README를 갱신했고 Stage0/본 보고서를 추가했다. PACKAGE_MANIFEST.json은 현재 코드·문서의 파일 목록과 hash로 갱신했다. 기존 `initial_policy.json`과 필수 schema는 변경하지 않았다. 기존 A/B·전처리·Baseline Python을 수정한 파일은0개다.

입력은 `PlayerFrame.xy_m[Nslot,2]`, 현재 team code와 parse flags 또는 기존 frame-local geometry/quality table이다. slot은 인물 ID가 아니다. CLI는 기존 cache의 원본 ATD metadata/raw SHA와 팀 코드를 대조하며 전체 cache/CSV의 frame·period 축을 검증했다. 별도 사전 고정31개 관측에서 cache로 재계산한 geometry/QC와 기존 CSV도 대조했다. 영상 FPS를 새로 디코딩해 측정한 검사는 아니다.

출력은 **199,319행 ×132열**, `frame=0..199318`이다. 필수 schema20개 열과 다음의 확장 열을 함께 제공한다.

* `baseline_score`, `baseline_temporal_available`: 원본 CausalContestedIndex의 전체 반 기준 결과.
* `candidate_score`, `spatial_contact_index`, `approach_change_index`, signed `nearest_approach_m_per_s`, `contact_fraction_delta_1s`.
* `home_*`, `away_*`: 중심·min/max/robust 폭·길이·radius·bbox area·density·최근접 동료거리. 각 `valid_*`를 함께 보존한다. 두 명 미만의 구조 통계는 NaN, 중심은 한 명 이상에서 정의한다.
* 전역 `cross_pairs_3m/5m/10m`, `cross_density_*`, 최근접 거리·최대 혼합 component와 validity. 상대 집합이 없을 때 기존0 pair를 관측된 비접촉으로 내보내지 않고 NaN+False로 바꾼다.
* 팀별 중심/폭/길이/compactness·쌍/군집/중심 간 거리의 `*_delta_prev_1s`. 모두 과거 방향이며 개인 속도/가속도가 아니다.
* `split`, `record_boundary`, `count_change_available`, `count_change_prev_frame`, `count_stable_1s`, `candidate_pending`, `above_threshold_run_frames`.
* `quality_tier`, `reason_codes`: low_players/overcount/duplicate_suspect/out_of_bounds/undefined_pair, history_boundary/history_warmup/temporal_gap/spatial_fallback, outside_official_block 등.
* `zone_sample_available`, 최대 구역 혼합·밀집·인원 집중도와 혼합 구역 수. 기존1fps의6,354개 frame에만 값이 있으며 나머지는 NaN이다. forward-fill하지 않는다.

`candidate_segments.csv`는 전체 상태·품질·반·split이 연속인 구간을1,797행으로 분할한다. 구간 전체 길이 합은199,319이고 frame의 중복/누락은 없다. 후보 구간은 이 중446행이다.

## 4. 고정 규칙과 시간적 의미

원본 지수는 그대로 사용한다.

```text
contact = 3m 안에 상대가 있는 고유 선수 수 / 관측 선수 수
cluster = 최대 H-A 3m 연결 component / 관측 선수 수
spatial = 0.5*contact + 0.3*cluster + 0.2*exp(-cross_min_m/3)
change = 0.5*clip(contact[t]-contact[t-25],0,1)
       + 0.5*clip((nearest_mean[t-25]-nearest_mean[t])/5,0,1)
score = 0.8*spatial + 0.2*change
```

5/10m 쌍·밀집도·팀 구조·48구역은 보조 근거로 기록하며 원래 지수에 새 가중치를 추가하지 않았다. 이탈은 signed 변화와 현재 근접도의 감소로 관찰하며 별도 정답 단계로 분류하지 않는다.

현재 포함26개 관측이 모두 strict-valid이며 같은 반/연속 기록/같은 split일 때만 change를 사용한다. 이력이 부족하면 현재 spatial로 fallback하고 변화는 NaN이다. signed nearest approach가 음수면 최근접 거리 증가 방향이다. 접근·유지·이탈 단서를 기록할 뿐 실제 경합 단계를 정답으로 이름 붙이지 않는다.

원래 baseline 전체 반 결과는 기존 CSV와 atol1e-14에서 일치했고 temporal availability도 동일했다. candidate 전용으로 split에서 이력을 끊으면서 **75프레임**의 점수가 baseline과 달라졌다. 원본 baseline의 시간 변화 가용101,558개와 달리 candidate는101,483개다. 기존 수식을 수정한 효과가 아니라 경계 정책 차이다. strict 유효111,733개 중10,250개는 spatial fallback이다.

임계값은 **고유 Train strict-valid80,538프레임**에서 q=.90을 한 번 적합했다. 실제 threshold는 **0.3147864495529943**, 비교는 `>=`, quantile은 NumPy linear다. 겹치는 window로 같은 raw frame을 반복 집계하지 않았다. Validation/Test/공개 GT·영상 장면을 fit에 사용하지 않았다.

5프레임 연속 유효·threshold 통과가 현재까지 확인되면 candidate를 출력한다. 첫4개는 pending인 not_candidate이며 소급 변경하지 않는다. half/split/gap/관측 불충분에서 run을 끊고 unknown gap을 연결하지 않는다. 따라서 첫 지원 관측부터 출력까지0.16초 확인 지연이 있고, 후보 출력 segment 자체는 최소0.2초를 보장하지 않는다. **106개 출력 segment가5프레임 미만**이다. 이는 causal 확인과 사후 minimum-duration filtering의 차이다. 이번에는 미래를 보는 후처리로 짧은 구간을 지우지 않았다.

임계값은 Train 전체로 오프라인 calibration한다. 'causal'은 이 고정 정책 아래 현재/과거 관측만으로 점수·상태를 계산한다는 뜻이지, Train fitting 자체가 미래를 모르는 online 절차라는 뜻이 아니다.

## 5. 품질에 따른 coverage와 후보 발생

| 공식 구간의 관측 범주 | 프레임 | 출력 |
|---|---:|---|
| strict-valid | 111,733 (70.3356%) | 후보9,936 / 비후보101,797 |
| metric-valid이지만 low-quality | 46,377 (29.1942%) | 관측 불충분 |
| 상대 관계 계산 불가 | 747 (0.4702%) | 관측 불충분 |
| 합계 | 158,857 | 후보6.2547%, 보류29.6644% |

metric-valid coverage는 **158,110/158,857=99.5298%**다. 이는 계산 가능한 비율이며 관측을 신뢰하거나 후보를 확정할 수 있는 비율이 아니다. Roadmap Stage2의 coverage-aware 후보 정책은 구현하지 않았다. strict와 수학적 계산 가능 범위의 차이를 보이는 데 그쳤고 완화한 규칙의 후보 빈도는 보고하지 않는다.

품질 사유는 중첩된다: 관측8명 미만14,456개, 11명 초과19,572개, 같은 팀0.2m 중복 의심952개, 피치 밖 관측22,876개, 상대 관계 미정의747개다. 합산하면 중복 계산된다. strict 기준은 각각8–11명·duplicate/bounds/partial/invalid 미검출이며 관측 선수의 실제 신원·팀 정확성·22명 완전 추적의 보증이 아니다. 비활성 source slot의 자연 결측과 실제 선수 누락률을 혼동하지 않는다.

| 공식 반 | strict 유효 | 후보 frame | 후보/전체 | 후보 segment |
|---|---:|---:|---:|---:|
| 1 | 57,085 | 4,960 | 6.2416% | 207 |
| 2 | 54,648 | 4,976 | 6.2678% | 239 |

전체199,319행의 상태는 후보9,936·비후보101,797·보류87,586이다. 공식 구간 밖40,462개도 행/관측 특징을 보존하고 `outside_official_block`과 함께 후보 판정을 보류했다. 이 상태는 범위 밖에서 실제 관측이 없다는 뜻도, 비경합 정답도, X 라벨 생성도 아니다. 기존 baseline의 X를 읽어 candidate를 제외하지 않았다.

직전 연속 관측과 인원이 같은 경우 현재 strict-valid110,681개 중 후보9,777개(8.8335%), 인원이 바뀐 경우 strict-valid1,048개 중 후보159개(15.1718%)였다. 원래 인원 변화 여부의 비교 가능 프레임은 각각156,791/2,060개이며 양끝 관계가 없는6개는 제외했다. 선수 추가·누락이 지수에 영향을 줄 수 있지만 이런 차이를 실제 경합률이나 인과효과로 해석하지 않는다. 하위 집합을 골라 시간축이 잘리므로 이 조건부 집합의 segment 수/지속시간은 실제 전체 후보 fragmentation과 직접 비교하지 않는다.

Train/Validation의 후보 수는6,695/1,874, 유효 관측 내 비율은8.3128%/10.8437%다. Train q=.90과 지속 조건 때문에 출력 비율이 정확히10%일 필요는 없다. full-match 산출물은 Test 시간 블록도 deployment routing으로 포함하지만 Test 정답·성능 채점·정책 선택은 하지 않았다. 어떤 split의 빈도에 맞춰 상수를 재선택하지 않았다.

## 6. 사전 지정 threshold/run 민감도

아래는 전체 공식 구간의 **기술 통계**이며 최적값 선택표가 아니다. 모든 threshold는 같은 Train strict 관측에서 계산하고 기본 q=.90/min_run5를 유지했다. Validation 기술 통계도 JSON에 별도 기록했다.

| q | min_run | 후보 frame | 후보/공식 전체 | segment | 중앙 지속(초) |
|---:|---:|---:|---:|---:|---:|
| 0.85 | 1 | 18,959 | 11.93% | 867 | 0.44 |
| 0.85 | 5 | 15,889 | 10.00% | 627 | 0.56 |
| 0.85 | 10 | 13,101 | 8.25% | 465 | 0.64 |
| 0.90 | 1 | 12,134 | 7.64% | 627 | 0.40 |
| 0.90 | 5 | 9,936 | 6.25% | 446 | 0.56 |
| 0.90 | 10 | 7,986 | 5.03% | 320 | 0.64 |
| 0.95 | 1 | 5,236 | 3.30% | 354 | 0.36 |
| 0.95 | 5 | 3,983 | 2.51% | 258 | 0.44 |
| 0.95 | 10 | 2,896 | 1.82% | 174 | 0.46 |

낮은 threshold와 짧은 confirmation이 더 많은 구간을 내보낸다는 성질을 확인했을 뿐 정확도 향상을 측정한 것이 아니다.

## 7. Tracking 노이즈·결측 민감도 — Validation만

미리 정한 Validation 각 블록의 균등6개, 총12개의151프레임 support(1,812개 고유 관측)에서 seed2027을 사용했다. 모든 조건이 같은 support와 원래 Train threshold를 사용한다. 각 support는 독립적으로 history/run을 초기화하므로 전체 Validation 추론의 정확한 부분집합과는 startup 조건이 다르다. 전체 Validation 강건성의 대표 표본이나 독립 사건1,812개라는 주장도 하지 않는다.

Gaussian jitter와 random dropout은 support 전체에 적용했다. composition은 각 support의 offset65..85에서20% random 관측을 제거한 **집합 구성 변화**이며 동일 인물 추적 실험이 아니다. physical shift는 같은 span의 Home 관측 전체를 +x1m 이동한 대조다. dropout과 물리적 강도가 동등하지 않으며 물리적 변화에 대한 불변성을 요구하지 않는다. 모든 변경은 복사한 메모리 관측에만 적용했다.

clean은 제공 추적 관측이지 완전한 위치 정답이 아니다. clean에서는 유효1,404·보류408·후보238프레임, 후보 segment12개였다. 점수 차이는 양쪽 점수가 유효한 **교집합**에서만 계산했다. flip은 양쪽 상태가 후보/비후보인 경우만, unknown 전환은 별도로 센다.

| 변형 | 변형 후 유효 | 공통 점수 | abs차 p50 | abs차 p95 | 후보↔비후보 | 보류 진입/해제 | 후보 frame | segment |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| jitter_0.1 | 1,422 | 1,402 | 0.003757 | 0.047539 | 49 | 2 / 20 | 204 | 14 |
| jitter_0.5 | 1,415 | 1,368 | 0.030341 | 0.113845 | 160 | 36 / 47 | 102 | 17 |
| jitter_1 | 1,385 | 1,326 | 0.045748 | 0.147104 | 193 | 78 / 59 | 39 | 13 |
| dropout_0.1 | 1,265 | 1,166 | 0.036158 | 0.106740 | 125 | 238 / 99 | 71 | 13 |
| dropout_0.2 | 831 | 744 | 0.047723 | 0.125205 | 107 | 660 / 87 | 17 | 3 |
| dropout_0.3 | 358 | 318 | 0.049719 | 0.138205 | 51 | 1086 / 40 | 0 | 0 |
| composition_0.2 | 1,327 | 1,313 | 0.000000 | 0.068435 | 46 | 91 / 14 | 216 | 13 |
| home_shift_1 | 1,404 | 1,404 | 0.000000 | 0.052354 | 3 | 0 / 0 | 235 | 11 |
| permutation_0 | 1,404 | 1,404 | 0.000000 | 0.000000 | 0 | 0 / 0 | 238 | 12 |
| team_swap_0 | 1,404 | 1,404 | 0.000000 | 0.000000 | 0 | 0 / 0 | 238 | 12 |

관측 누락30%에서는1,812개 중358개만 strict 유효해지고 보류1,454개, 후보0개였다. 이는 비경합으로 바뀐다는 뜻이 아니라 강한 관측 부족과 causal run 중단이다. 단기 composition에서도 보류가 늘고 score 차이가 발생했다. 순열·팀 교환의 실제 표본 상태 flip은0이었고 score 차이는 부동소수점 반올림 수준이었다.

작은 jitter/dropout에서 strict 유효 수가 늘어날 수도 있다. 의심 중복·bounds/overcount가 자동 policy에서 사라지기 때문이며 tracking이 좋아졌다고 판단한 것이 아니다. 근접 threshold의 계단 변화·26개 연속 QC·5개 run의 영향이 함께 반영되므로 단일 원인으로 분리하지 않는다. 임계값을 noise 결과에 맞춰 조정하지 않았다.

## 8. 테스트·실제 출력 검증·보존

기존 suite를 먼저 실행해156개 모두 통과했다. 최종 합성 suite는 **187 passed in22.53s**(기존156+신규31), 실패0개다. 합성 테스트에서 거리 포함 경계, 고유 접촉 선수 수, bipartite chain, zone 경계를 넘는 근접, 순열/팀 교환과 ordered 구조 교환, 빈/한 명/저인원, partial/Inf/bounds/duplicate, 자연 누락과 strict 전환, 고정 threshold의 causal prefix 불변, half/split/gap reset, Train-only fit·중복 거부, signed 이탈, full199,319축, ball/tag/GT/future-delta decoy 배제, perturbation 재현·원본 보존, state/CSV/zone 샘플·덮어쓰기 거부를 확인했다.

개발 중 pandas Series 단언 방식과 경계 변수 덮어쓰기 오류를 합성 테스트가 찾아 수정했다. 최종 실제 실행은 수정 후 코드이며 실패가 남아 있지 않다. 출력 validity 정리 과정에서 singleton 구조 통계의0을 NaN으로 보강하고 reason을 상세화한 뒤 새 최종 폴더로 다시 실행했다. 초기 실행 폴더는 보존했으며 두 실행의 frame/candidate_score/state는 정확히 같았다. 규칙·quantile·모델 선택을 반복한 실험이 아니다.

실제 산출물 재읽기에서도 frame199,319개 연속·유일, segment 전체 분할, 후보의 strict eligibility와 과거5프레임 지원, undefined metric/구조의 NaN, unsampled zone의 NaN, 숫자 Inf 없음, 실행 당시 source SHA 일치를 확인했다. `NaN`은 결측 표시로 허용하며 실제 비경합0으로 채우지 않는다.

작업 시작 시 기록한 기존 코드·문서·baseline **148개 SHA256**, 기존 원본/산출물 **8,252개 size/mtime**이 최종까지 동일했다. CLI 실행 자체도 당시 존재한 모든 파일을 snapshot/대조했다. baseline4개 모두 공식 `possession_metrics.validate_submission`의199,319행/frame0..199318 검사를 통과했다. 라벨 재생성·원시 볼 기반 추론·GT 점수 채점은 하지 않았다. evaluation 보호파일의 byte hash 감사는 했지만 GT 라벨을 파싱하거나 입력/정답/튜닝에 사용하지 않았다.

실행 환경은 Conda mtah2027, Python3.11.17, NumPy2.4.6, Pandas3.1.0, macOS-26.6.2-arm64-arm-64bit, CPU. 실제 경기 candidate 실행에서는 신경망 framework를 import하거나 학습하지 않았다. 기존 회귀 pytest에는 합성 Torch forward/backward·소형 학습 검사가 포함되며 실제 경기 모델 재학습과는 구분한다. 기록된 계산·진단 시간은 약 36.23초이며 CSV/마지막 보존 검사 시간은 제외한다. peak 메모리는 측정하지 않았다.

## 9. 실행 명령과 결과 위치

프로젝트 루트에서:

```bash
conda activate mtah2027
export PYTHONPATH="$PWD/competition_judgement/src:$PWD/MTAH_Relational_Starter:$PWD"
export PYTHONDONTWRITEBYTECODE=1
python -m pytest -q -p no:cacheprovider MTAH_Relational_Starter/tests competition_judgement/tests
python -m competition_judgement.cli --out-dir data/derived/competition_judgement/stage1_reproduction
```

동일한 폴더가 있으면 덮어쓰지 않는다. CLI 결과는 `data/derived/competition_judgement/` 하위로 제한하며 git ignore를 실제 확인했다. 원시 parser는 추가하지 않았고 feature CSV 부재 시 기존 Phase1 함수만 호출한다. raw cache는 기존 temporal build로 준비되어 있어야 한다. 결과를 GitHub·외부 LLM·원격 로그에 보내지 않고 로컬 서버도 실행하지 않았다.

이번 최종 결과는 `data/derived/competition_judgement/stage1_v1_final_20261109/`에 있다.

* [candidate_frames.csv](../../data/derived/competition_judgement/stage1_v1_final_20261109/candidate_frames.csv): 전체 프레임 관계·품질·후보 상태.
* [candidate_segments.csv](../../data/derived/competition_judgement/stage1_v1_final_20261109/candidate_segments.csv): 모든 상태/품질 구간.
* [quality_and_sensitivity.json](../../data/derived/competition_judgement/stage1_v1_final_20261109/quality_and_sensitivity.json): 빈도·품질·민감도 집계.
* [policy_resolved.json](../../data/derived/competition_judgement/stage1_v1_final_20261109/policy_resolved.json): fit한 threshold·설정·출처·분할 fingerprint.
* [verification.json](../../data/derived/competition_judgement/stage1_v1_final_20261109/verification.json), [output_validation.json](../../data/derived/competition_judgement/stage1_v1_final_20261109/output_validation.json): 원본 보존·baseline·출력/테스트 검사.

프레임별 CSV는 실제 경기의 비공개 파생물이다. 응답·공개 저장소에 내용을 붙이거나 업로드하지 않는다.

## 10. 한계와 다음 작업의 경계

* 볼 없는 마킹·압박·세트피스 대기와 실제 볼 경합은 같은 선수 배치를 가질 수 있다. 현재 지수로 식별 불가능한 경우가 있다.
* 최대 혼합 component는 연결 사슬을 포함한다. 크기가 큰 component가 같은 좁은 지점의 경합이라는 뜻은 아니다.
* 관측 정규화된 pair/cluster 비율도 MNAR 누락·중복·팀 오분류를 보정하지 못한다. 폭·길이·중심의 변화 역시 관측 선수 구성 변화일 수 있다.
* 엄격 QC는 공식 구간29.66%를 보류한다. tracking이 어려운 실제 경합을 더 많이 놓칠 수 있지만 정답이 없어 누락률/recall을 산출할 수 없다.
* history 불가 시 spatial fallback은 temporal 혼합 점수와 분포가 다르다. 단일 Train quantile·causal confirmation의 효과와 경계 시작 bias를 기록했으며 이번에는 새 점수/보정으로 바꾸지 않았다.
* 단일 경기·고정 seed·초기 engineering policy다. q=.90과run5는 최적값이나 확률 calibration이 아니며 다른 경기로의 일반화도 검증하지 않았다.
* 실제 경합 라벨·공식 점유 결과를 생성/검증하지 않았다. 후보만으로 D, X 또는 H↔A 전환을 결정하지 않는다. 실제 ball validity도 이번 모듈에서 측정하지 않았으므로 후보 구간이 곧 ball-error 구간이라고 주장하지 않는다.

Stage2의 완화 품질 정책·시간 안정성 보완, Stage3 점유 routing, Stage4 선택적 신경망 비교는 실행하지 않았다. 다음 작업이 필요하다면 unknown의 원인을 유지하면서 coverage를 보완하는 별도 사전 정책부터 설계해야 한다. 영상 수작업 경합 라벨/장면별 임계값 조정, A/B 재학습, C/Fusion 및 H/A/D 최종 판단 변경은 이번 범위에 포함되지 않는다.
