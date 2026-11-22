# 01 — 프레임, 데이터 품질 및 규정 계약

## 원본 계약 (기존 보고서 확인)
- 전체 프레임 `0..199318`, **199,319개**, 25fps. ATD `frame_id=frame+1`.
- 공식 분석/평가 전반 `[11185,90651]`, 후반 `[110808,190197]`: 총 **158,857프레임**. 공식 `games.half_video_blocks()`만 사용한다.
- 공식 경계 밖 **40,462프레임**을 삭제하지 않는다. 이 영역은 별도 QA 및 최종 제출 라우팅 대상으로 보존하되 경합의 '비경합 정답'으로 간주하지 않는다.
- 선수 좌표는 normalized에서 **105.3m × 68.0m**로 환산. XML 105m로 혼용하지 않는다.
- XML Score Local/Visiting Team ID로 H/A를 매핑한다. 소스 슬롯 46개는 지속적 선수 ID가 아니다. source-slot 차분·인물별 속도 추정 금지.
- 볼, speed, Smart Tag, 공개 GT는 **경합 후보 모듈의 입력이나 임계값에 사용하지 않는다**.
- Phase1의 `features.csv`, `symmetric_features.csv`는 전체199,319행이며 모듈 재사용 가능. CSV가 없으면 기존 player-only reader로 생성한다.

## X 해석 — 이전 대화의 혼동 방지
X는 공식 제출에서 **점유를 판단하지 않겠다는 abstention**이며, 자동으로 '경합', '관측 결측', '경기 중단'을 뜻하지 않는다. 공식 평가 범위 밖의 여러 프레임이 X로 작성될 수 있지만, **X가 붙었다는 사실 자체로 candidate 입력에서 제외하지 않는다.** 기존 baseline이 X로 예측했는지는 분석용 provenance일 수 있으나 진실 라벨이 아니다. 전체 결과 프레임 인덱스를 유지한다.

## 실제 관측 품질의 알려진 한계
기존 strict quality 정책: 팀마다 관측8–11명, 동일 팀≤0.2m 중복 의심·경계 밖·invalid 관측 등의 조건. 공식 구간의 strict quality-valid는 **111,733/158,857 (70.34%)**; 나머지47,124개가 품질 기준에서 제외된다. 이들은 '경합 없음'이 아니다.

정책은 명확히 구분한다.
1. `strict_valid`: 기존 품질 QA 기준. 기존 baseline 재현에만 사용 가능.
2. `metric_valid`: 특정 거리/쌍/군집 특징을 수학적으로 계산할 충분한 관측이 존재. 각 특징별로 다를 수 있다.
3. `reliability`: 관측 인원, bounds, 중복 의심, 전후반/연속 시점 유효성 등을 근거로 고/낮음/불충분 분리.
4. 출력은 `candidate`/`not_candidate`/`insufficient_observation` 상태를 유지. 산출 불가한 feature는 NaN+validity mask, 0으로 강제하지 않는다.

**관측 품질을 완화한다고 실제 경합/중복의 정답을 얻는 것은 아니다.** 낮은 신뢰도 구간에는 별도 신뢰 표시를 둔다. 각 완화 정책의 coverage와 불확실성을 보고하고 본 실행의 채택 기준은 영상을 보고 조정하지 않는다.

## 금지·허용 범위
공식 소스(작업 시 로컬 파일로 재확인):
- https://github.com/metrica-sports/more-than-a-hack-2027/blob/main/RULES.md
- https://github.com/metrica-sports/more-than-a-hack-2027/blob/main/FAQ.md
- https://github.com/metrica-sports/more-than-a-hack-2027/blob/main/EVALUATION.md

금지: 제공 경기 영상/데이터에서 사람이 경합 여부나 점유를 특정 프레임·구간 단위로 라벨링, 수정, 검증 목표 생성하거나 그렇게 얻은 결과로 threshold/상수를 조정; 공개5분 GT로 모델/임계값 튜닝; 외부 human-generated data about provided match; 재생성 불가능한 제출; 제공 데이터·파생물의 third-party hosted/cloud LLM 업로드.

허용: 일반적 영상 이해·디버깅/특징 설계; 제공 경기 좌표만으로 **자동** 생성한 점수/의사 후보; 다른 경기의 사람이 만든 라벨(주최 측이 접근·검증 가능하고 방법 문서화된 경우). 이번 Phase는 어떤 human label도 사용하지 않는다.

## 분할과 시간의 유효성
- 현행 Train/Validation/Test 블록은 `references/PHASE2_TEMPORAL_DATA.md`를 확인하고 실제 split manifest와 교차 검사한다.
- 정책 분포 임계값(예: Train quantile)은 **고유 Train 원시 프레임**에서만 적합한다. 겹치는 Window별 반복 카운트 금지.
- 미래가 필요한 offline 변화 분석은 별도 열로 표기하며 현재 causal score와 혼합하지 않는다.
- 전후반/record gap/period0 경계에서 추세와 지속 구간을 절대 이어 붙이지 않는다.
- 199,319행 서식을 유지하되 실제 분석 유효성과 경합 후보 여부를 분리한다.

## 로컬 보안
실제 경기 자료·NPY/CSV/checkpoint·프레임별 candidate 파일은 `data/derived/competition_judgement/`에 저장하고 `.gitignore`를 확인한다. 대화에 raw 경기 배열을 넣거나 업로드하지 않는다. 외부 웹/LLM 원격 보고 및 Push 금지.
