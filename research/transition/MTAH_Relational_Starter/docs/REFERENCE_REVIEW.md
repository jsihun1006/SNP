# 기존 코드 및 논문 검토 — MTAH 활용성 평가

**판단 원칙:** 사용자가 공유한 6개 GitHub 항목 + 2023년 논문을 대상으로, 공개 README/문서를 재확인했다. **논문/코드의 원래 데이터와 학습 조건을 우리의 현재 라벨 없는 단일 경기와 혼동하지 않는다.**

## 1. 핵심 채택할 논문·코드

### [Borghesi et al., 2023] Using Temporal Convolutional Networks to estimate ball possession in soccer games

- 논문: https://www.sciencedirect.com/science/article/pii/S0957417423002816
- DOI: https://doi.org/10.1016/j.eswa.2023.119780
- 내용: TCN으로 시계열 tracking trajectory를 인코딩. 선수의 순서에 의미가 없으므로 permutation-invariant aggregate(sum, self-attention 등)를 실험. `live/dead` 분류 후 `home/away` 판별하는 두 네트워크 접근도 비교함.
- 우리에게 유용: **선수 입력 순서 불변 처리**, 전체 선수 중 유의미한 관계에 attention, `D`와 `H/A`를 분리하는 판단 구조, per-frame 대신 temporal context.
- 주의: 원 논문은 라벨 데이터로 성능 검증한 **지도학습 연구**. 보고된 accuracy는 현재 MTAH 데이터의 성능이 아님. 데이터는 공개되지 않음.
- 적용 결정: 네트워크 구조와 논리 채택, TCN 자체는 선택적 A/B comparison. 현재 설계는 Bi-LSTM 우선.

### [Kim et al., 2023 / Ball Radar] Ball Trajectory Inference from Multi-Agent Sports Contexts Using Set Transformer and Hierarchical Bi-LSTM

- 소스: https://github.com/hyunsungkim-ds/ballradar
- 논문: https://arxiv.org/abs/2306.08206
- 내용: 선수 집합에서 Set Transformer와 Hierarchical Bi-LSTM을 사용. 공 소유 선수 classifier → 조건부 볼 궤적 regressor → 후처리의 계층 구조.
- 우리에게 유용: **선수 순서에 영향받지 않는 인코딩(Set Transformer)** 및 **Bi-LSTM**. 경합 이후 볼 위치 모델을 별도 개발할 때 설계 참고.
- 주의: 원래는 공 위치·이벤트 등이 있는 경기로 훈련/평가. **우리 MTAH 단일 경기에서 공 정답을 가린 채 곧바로 학습 가능한 순수 비지도 모델이 아님**. 원본 그대로 실행하기 어렵고 입력 스키마 적합화 필요.
- 적용 결정: 경합 후보를 정리한 뒤 ball-location auxiliary 실험에 참고. 경합 모델 내부에는 볼 좌표 사용하지 않음.

### [Kim et al., 2026 / PathCRF] PathCRF: Ball-Free Soccer Event Detection via Possession Path Inference from Player Trajectories

- 소스: https://github.com/hyunsungkim-ds/pathcrf
- 논문: https://arxiv.org/abs/2602.12080
- 내용: 선수 궤적을 fully connected dynamic graph로 표현하고, Set Attention 기반 인코딩 + CRF로 논리적으로 일관된 possession path를 추론해 on-ball event를 검출.
- 우리에게 유용: **선수 간 관계 그래프**, 구간별 전환 일관성, CRF 기반 출력 시퀀스 제약.
- 주의: 원본은 다른 데이터(Sportec 7경기, synchronized event/possession states)로 학습. **라벨 없는 단일 경기에서 원본 학습 절차 직접 적용 불가.** 프레임별 단일 볼 보유 edge 정의는 MTAH의 팀 단위 H/A/D 목표보다 세밀함. 46 슬롯은 ID가 연속적이지 않으므로 개인 trajectory 기반 전이도 그대로 차용 불가.
- 적용 결정: Dynamic Graph와 temporal consistency 부분을 자기지도 인코더/별도 decoder로 재설계.

## 2. 부분적으로 참조할 구현

### [samuehuang / football-cv-analytics]

- 소스: https://github.com/samuehuang/football-cv-analytics
- 내용: 비디오에서 선수/공 추적→궤적 정제→볼 기반 점유·이벤트→전술 지표. team centroid, width, length, compactness, inter-team spacing 등 추출. 제공된 스모크/일관성 검사 사례는 정답 기반 표준 accuracy가 아님.
- 활용: **team shape 및 compactness 계산, tracking coverage QA, 단계별 모듈 분리** 참조.
- 주의: 전체 파이프라인은 영상에서 시작하며 ball detector/ball state를 사용한다. **공 없이 선수의 움직임만으로 볼 위치를 정확히 학습하는 비지도 모델이라고 볼 근거는 없음**. MTAH는 선수 좌표가 이미 제공되므로 영상 tracking 구현은 우선순위 낮음.
- 적용 결정: 공간 특징·진단 구조 참조, 영상 파이프라인은 제외.

### [Tryolabs / soccer-video-analytics]

- 소스: https://github.com/tryolabs/soccer-video-analytics
- 내용: 중계 영상에서 물체와 볼 검출을 기반으로 점유·패스 카운트를 데모. README는 볼 모델을 제한적 toy/overfit이라고 명시.
- 활용: **나중에 broadcast 영상 활용 시** 영상 tracking/overlay 도구로 검토.
- 주의: tactical ATD가 이미 있는 현재 모델에는 불필요. 방송 화면도 tactical과 타임라인 비동기.
- 적용 결정: **지금은 보류**.

## 3. 이번 모델의 직접 근거에서 제외

### [ML-interns-aau / footballtracking — implementation_plan.md]

- 소스: https://github.com/ML-interns-aau/footballtracking/blob/main/docs/implementation_plan.md
- 내용: 카메라 4점 calibration 설정 및 nearest-ball-player 기반 점유 히스테리시스의 **구현 계획 문서**. 초기 예시 1.5m possession radius, 5frames 유지 및 ball speed 판단.
- 주의: **계획 문서이지 검증된 모델 성능이 아님**. 현재 경합 목표는 볼을 사용하지 않으며 ID와 ball tracking에 종속되는 방식은 부적합.
- 결정: **모델은 미채택**. 추후 temporal hysteresis 설계의 일반적 참고만 가능.

### [Roboflow / sports pull request #36]

- 소스: https://github.com/roboflow/sports/pull/36
- 내용: ball distance + temporal smoothing 기반 possession tracking·pass counts·overlay 추가 제안.
- 확인 상태: 2026-11-09 조회 시 PR **open, not merged**.
- 주의: 별도 모델 검증 근거 부족, ball-dependent이며 `spellF1` 최적화에 필요한 제어권 판별과는 다른 문제.
- 결정: **지금은 제외**.

## 4. 채택하는 아이디어와 실제 구현의 차이

| 출처 | 설계에 가져올 아이디어 | 지금 직접 재사용하지 않는 것 |
|---|---|---|
| Borghesi 2023 | set/order invariant, player relevance, live/dead 분리 | supervised weights & reported accuracy |
| Ball Radar 2023 | Set Transformer + hierarchical Bi-LSTM | ball-label-supervised regressor |
| PathCRF 2026 | dynamic graph, CRF consistency | event-labeled pipeline, player identity assumptions |
| football-cv-analytics | team shape metrics, modular QA | video detection/tracking, ball-dependent possession |
| Tryolabs | future broadcast video tools | present ball-free contest model |
| AAU implementation plan | smoothing conceptual references | unverified camera/ball-dependent possession |
| Roboflow PR 36 | general postprocessing references | unmerged ball-dependent code |

## 5. 우리 모델의 새 기여 목표 (아직 검증 전)

1. **실제 선수 ID가 없는 상황에서도** 프레임별 관계를 선수 순서에 불변하게 인코딩.
2. 경합 후보 추론에 **상대 팀 간 근접 그래프 + 팀 내부 구조 + 시간 전후 변화**를 동시에 사용.
3. **Raw graph + engineered features** 두 stream으로 자기지도 표현 학습.
4. 공·Smart Tagging 없이 경합 가능성 높은 구간 발견(경합 진실값이 아님).
5. 최종적으로 별도 possession-state decoder에 전환 맥락으로 공급.

## 6. 재현 가능성, 라이선스

참고 코드를 그대로 복사하지 말고 각 저장소의 LICENSE를 확인할 것. 아이디어를 참고해 독립 구현하면 의존성 및 license 관리가 단순해진다. 특히 Ball Radar와 PathCRF의 학습/평가 데이터 약관은 MTAH 경기 데이터와 무관하다.
