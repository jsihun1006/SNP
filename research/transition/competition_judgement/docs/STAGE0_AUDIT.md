# Stage 0 감사 (2026-11-09)

시작 문서, docs/00–05, references/의 7개 보고서 및 실제 `mtah_relational`의 data_quality/features/spatial_schema/candidate_baseline/preprocessing/phase1/zones/frame_store/splits 코드를 읽었다. 공식 RULES/FAQ/CHALLENGE/DATA/EVALUATION과 games.py 및 제출 validator를 확인했다. 외부 저장소를 다시 다운로드하거나 경기 원시 내용을 응답으로 전송하지 않았다.

## 실제 경로와 문서 차이

* Python은 루트가 아닌 `MTAH_Relational_Starter/mtah_relational/`에 있다. 새 모듈은 `competition_judgement/src/competition_judgement/`에만 추가한다.
* 기존 `CausalContestedIndex.step(frame, period, features, quality_valid)`는 반환 키가 `contested_index`, `index_valid`다. 새 출력의 `candidate_score`/상태로 연결하되 원래 수식은 변경하지 않는다.
* features의 실제 접촉/군집 비율 이름은 `cross_contact_fraction_3m`, `cross_cluster_fraction_3m`이다. 출력 계약의 `contact_player_fraction`, `cross_cluster_fraction`은 별도 alias다.
* 빈 팀에서 기존 geometry는 pair/cluster count를0으로 내보낸다. 새 candidate 파일에서는 해당 값에 NaN과 수학적 validity를 적용해 관측된 비접촉과 구분한다. 원래 특징 CSV는 보존한다.
* references는 스냅샷이다. 현재 B 진단은 루트 `docs/B_MODEL_DIAGNOSTIC.md`까지 진행되었고 기존 합성 suite는156개다. C 모델/Fusion은 아직 구현하지 않았다. 과거 문서의 신경망 미구현·미설치 설명은 당시 기록이다.
* 기존 움직임 벡터의 source-slot 차분은 지속적 선수 ID를 보장하지 못하므로 재사용하지 않는다. 검증된 팀 구조와 집합 통계의 과거25프레임 변화만 사용한다. 미래 delta는 allowlist에서 제외한다.
* 기존 지수는 half/gap에서 reset하며 split은 알지 못한다. 원본 지수의 전체 반 기준 결과는 `baseline_score`로 정확히 재현한다. candidate 전용 이력은 split에서도 reset한다. 이에 따른 경계 부근 차이는 실행 JSON에 별도로 기록한다.
* Roadmap은 coverage-aware 후보 정책의 구현을 Stage2로 분리한다. 이번 Stage1에서는 strict 상태와 별개로 metric-valid coverage만 보고하고, 낮은 품질을 후보/비후보로 재분류하는 새 완화 정책은 채택하지 않는다.
* 5프레임 조건은 causal confirmation으로 구현한다. 첫4개는 pending인 not_candidate이며 나중에 소급하지 않는다. 따라서 출력 후보 segment는0.2초보다 짧을 수 있다. 최소 지지 관측 길이와 출력 segment 길이를 구분한다.

## 보호·기준 검사

작업 전 기존 소스·문서·baseline148개 SHA256, 원본과 이전 파생물8,252개 size/mtime을 로컬 감사 파일에 기록했다. 작업 전부터 수정되어 있던 starter notebook3개는 그대로 보존한다. `.gitignore`의 `data/*` 규칙은 새 결과에도 적용된다.

기존 합성 pytest는 **156 passed in21.02s**였다. 공식 `possession_metrics.validate_submission`으로 baseline4개의199,319행/frame0..199318을 모두 확인했다. GT를 열거나 라벨을 재생성하지 않았다. 최종 보존 확인과 새 모듈 결과는 `STAGE1_EXECUTION_REPORT.md` 및 로컬 verification JSON을 따른다.

## 사전 고정 실행

기존 initial_policy.json을 변경하지 않는다. 고유 strict-valid Train의 점수 q=.90을 한 번 fit하고 고정한다. q=.85/.90/.95 × min_run1/5/10은 감도표만 작성한다. 실제 데이터에서 후보 정답·영상 라벨·GT로 승자나 상수를 고르지 않는다.

Validation 각 블록에서 균등한6개, 총12개의151프레임 support를 noise 검사에 사용한다. Gaussian0.1/0.5/1m, random dropout10/20/30%, offset65..85의20% 단기 dropout, 같은 span의 Home +x1m 이동, 순열·팀 교환을 seed2027로 고정했다. threshold는 원래 Train 값 그대로다. 이것은 tracking 정확도나 경합 정확도가 아닌 민감도 검사다.
