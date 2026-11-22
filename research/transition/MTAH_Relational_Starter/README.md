# MTAH 2027 — Player Relationship & Contest Detection Starter

> **현재 상태:** 데이터·Window·A 모델에 이어 **Dynamic Graph + Bi-LSTM B 모델**의 로컬 학습·Validation 복원·추적 노이즈 분석까지 완료했습니다. 합성 pytest143개 통과. 이번 B는 A/선형 보간보다 복원 성능이 낮았습니다. Fusion·군집·경합 확률·H/A/D 판별기는 아직 없습니다.

**최신 B 모델:** [5단계 실행 기록](../docs/PHASE5_MODEL_B.md)에서 graph/edge8 정의,43,853개 parameter, B12/B20 예산과 선택 epoch, 특징별 오차·노이즈·분산·실행 명령을 확인하세요. 실제 결과는 로컬 `data/derived/model_b_v1_20261109/`이며 A12/A20과 기존 제출 baseline은 보존했습니다. B의 `team_embedding[B,151,2,32]`은 H/A를 각각 유지합니다. 어느 branch를 Fusion에 채택할지는 결정하지 않았습니다.

현재 구현·환경·품질 통계는 [1단계 실행 기록](../docs/PHASE1_DATA_RELATIONS.md)을 먼저 읽으세요.

**2단계 추가:** 반별 시간 분할, 151프레임 WindowDataset, 연속8–13프레임 동시 마스킹, Train-only 정규화, 입력/타깃 분리와 추론 padding을 구현했습니다. [시계열 데이터 실행 기록](../docs/PHASE2_TEMPORAL_DATA.md)에 98개 합성 테스트와 실제 데이터 검증 결과를 정리했습니다. 신경망 학습은 하지 않았습니다.

```bash
# 프로젝트 루트, mtah2027 환경. 새 로컬 출력 폴더만 사용합니다.
export PYTHONPATH="$PWD/MTAH_Relational_Starter:$PWD"
python -m mtah_relational.temporal_cli build --out-dir data/derived/relational_temporal_v1
```

2단계 결과는 `open_temporal_dataset`으로 다시 열 수 있습니다. 모델 입력은 `[B,151,53]`, 복원 target은 별도 `[B,151,13]`이며 target을 모델 forward에 전달하지 않습니다.

## A 모델 학습·평가 (4단계)

[4단계 실행 기록](../docs/PHASE4_MODEL_A.md)에 파일 목록·shape·split·실제 MPS 학습 결과·특징별 MAE를 정리했습니다. 이번 소형 실험에서는 **선형 보간이 A 모델보다 복원 성능이 좋았습니다.** 이 실험은 경합/점유 정확도를 평가하지 않습니다.

```bash
conda activate mtah2027
export PYTHONPATH="$PWD/MTAH_Relational_Starter:$PWD"
# 의존성이 없을 때만: python -m pip install -r MTAH_Relational_Starter/requirements-model-a.txt
export PYTORCH_ENABLE_MPS_FALLBACK=1
python -m pytest -q -p no:cacheprovider MTAH_Relational_Starter/tests
python -m mtah_relational.train_model_a \
  --data data/derived/relational_temporal_v1_20261109 \
  --feature-cache data/derived/model_a_feature_cache_v1 \
  --out-dir data/derived/model_a_new_run \
  --config MTAH_Relational_Starter/configs/model_a_small.json
python -m mtah_relational.plot_model_a --run-dir data/derived/model_a_new_run
```

시계열 데이터 출력이 없는 환경에서는 temporal build를 먼저 실행하고 `--data`에 그 출력 경로를 지정합니다. 기존 out-dir를 덮어쓰지 않습니다. 이번 실제 결과는 `data/derived/model_a_run_v2_20261109/`에 로컬 보관했습니다. PNG 그래프는 서버 없이 파일로 열면 됩니다. `best.pt`, 캐시와 임베딩을 포함한 경기 파생 결과는 업로드하지 마세요.

## 이 패키지의 목적

Metrica의 **단일 경기 tactical ATD 선수 좌표**로 선수 간 공간적 관계와 시간적 변화 특징을 추출하고, 이후 **Dynamic Graph Encoder + Two-Stream self-supervised Bi-LSTM**의 입력으로 사용한다. 최종 목표는 단순한 경합 후보 발견을 넘어 **경합 전후의 조직 변화와 점유 유지/전환**을 추론하고 대회의 `H/A/D/X` 파일을 생성하는 것이다.

- 분석 대상: FC Barcelona (H) vs Athletic Club (A), 25fps, 199,319 tactical frames
- 경합(contested)은 **별도 잠재 상태**이며, 공식 `D`(Dead Ball)와 **다르다**.
- Smart Tagging: 이번 설계에서 **제외**.
- 선수 좌표: 경합 모델 입력. 볼 좌표는 경합 모델에서 **제외**, 향후 점유 보조 모듈에서만 별도 검토.
- 수동 라벨링과 공개 5분 정답을 이용한 훈련/튜닝: **금지**.
- 경기 원본 ATD/영상/그로부터 재구성 가능한 데이터의 공개 저장소 업로드: **금지**.

## 중요한 사용 전제

**이 폴더는 기존 공식 MTAH 리포지토리의 추가 패키지입니다.** `games.py`의 상수와 로컬 ATD XML/TXT만 필요합니다. `starter/`·평가 코드·Smart Tagging에 의존하지 않습니다. 패키지를 루트로 옮길 필요가 없습니다.

### VS Code / macOS 실행

1. VS Code에서 `more-than-a-hack-2027` 폴더를 열고 터미널에서 기존 `mtah2027` Conda 환경을 활성화.
2. 현재 `MTAH_Relational_Starter/` 안의 패키지와 테스트를 그대로 유지합니다.
3. 프로젝트 루트에서 실행합니다. pytest가 없으면 표시된 설치 명령을 로컬에서 실행하세요:

```bash
conda activate mtah2027
export PYTHONPATH="$PWD/MTAH_Relational_Starter:$PWD"
# 필요할 때만: python -m pip install -r MTAH_Relational_Starter/requirements-relational.txt
python -m pytest -q -p no:cacheprovider MTAH_Relational_Starter/tests
python -m mtah_relational.cli check
python -m mtah_relational.cli phase1 --out-dir data/derived/relational_phase1 --zone-stride 25
```

`python -m mtah_relational.cli check` 명령은 실제 ATD 파일이 로컬 `data/tactical/`에 있는 경우에만 성공합니다. 데이터가 없다면 MTAH 공식 안내대로 압축을 풀어 배치하거나 `MTAH_DATA_DIR`을 지정하세요. **ChatGPT 등 외부 클라우드에 경기 파일을 전송하지 마세요.**

선택적으로 source-slot 배열도 저장하려면 새 출력 폴더를 사용합니다:

```bash
python -m mtah_relational.cli phase1 --out-dir data/derived/relational_phase1_nodes --save-nodes
```

**생성 결과** (비공개 로컬 `data/derived/`):

- `features.csv`: 전 프레임의 H/A 개별 구조·상대 거리·품질·경합 후보 지수와 과거1초 관계 변화.
- `symmetric_features.csv`: 대칭24개 특징과 관측 유효성 mask.
- `zones_48.csv`: 기본25프레임 간격의 보조48구역 지표. 전역 거리는 구역 경계와 무관합니다.
- `quality_report.json`, `schema.json`: 전체 품질 집계·정책·데이터 인터페이스.
- `source_slot_xy_m.npy`, `observed.npy`, `team_codes.npy`: `--save-nodes` 선택 시 생성하는 로컬 전용 배열. 슬롯은 인물 ID가 아닙니다.

출력 폴더가 이미 있으면 덮어쓰지 않습니다. 기존 `extract/plot`은 기술 통계용으로 남겨 두었으며 미래 delta를 포함할 수 있어 모델 입력으로 그대로 사용하지 않습니다.

> **주의:** 코드가 현재 계산하는 것은 비지도 학습의 입력 특징이지, 경합 정답 라벨이나 승자 판단이 아닙니다. 그래프 신경망과 Bi-LSTM 학습은 다음 단계에서 개발합니다.

## 폴더 구성

```text
more-than-a-hack-2027/         # 기존 Metrica 저장소
├── games.py                   # 기존
├── starter/                   # 기존
├── evaluation/                # 기존
├── data/                      # 경기 원본/파생 결과: Git에 올리면 안 됨
│   ├── tactical/
│   └── derived/               # 코드 실행 시 생성
├── MTAH_Relational_Starter/
│   ├── mtah_relational/       # cli, data_quality, features, spatial_schema,
│   │                         # candidate_baseline, preprocessing, phase1, zones
│   ├── tests/                # 합성 데이터 전용 pytest
│   ├── docs/                 # 최초 설계·참조 연구 문서
│   ├── scripts/fetch_reference_sources.sh
│   ├── CODEX_REFERENCE_STUDY_PROMPT.md
│   ├── CODEX_START_HERE.md
│   └── requirements-relational.txt
└── docs/                     # 연구 결과·현재 구현 계획·1단계 실행 기록
```

## 바로 읽을 문서

1. [CODEX_START_HERE.md](CODEX_START_HERE.md) — Codex에게 그대로 전달할 구현 지시
2. [docs/PROJECT_SPEC.md](docs/PROJECT_SPEC.md) — 연구 문제와 전체 모델 설계
3. [docs/REFERENCE_REVIEW.md](docs/REFERENCE_REVIEW.md) — 지금까지 받은 GitHub·논문 정리
4. [docs/IMPLEMENTATION_PLAN.md](docs/IMPLEMENTATION_PLAN.md) — 단계별 작업 및 테스트 기준
5. **[CODEX_REFERENCE_STUDY_PROMPT.md](CODEX_REFERENCE_STUDY_PROMPT.md) — Codex에게 보내는 외부 코드 학습 지시문 (먼저 실행)**
6. **[docs/EXTERNAL_CODE_STUDY_GUIDE.md](docs/EXTERNAL_CODE_STUDY_GUIDE.md) — 실제 GitHub 파일/함수/학습 방식까지 검토한 심화 노트**
7. **[docs/REFERENCE_TO_MODEL_BLUEPRINT.md](docs/REFERENCE_TO_MODEL_BLUEPRINT.md) — 외부 모델의 아이디어를 우리 모듈로 옮기는 인터페이스·테스트 명세**

## 핵심 검증 포인트

- 원본 ATD `frame_id`(1-based) → 제출 `frame`(0-based): `frame = frame_id - 1`.
- pitch normalized coordinate (0–1) → metre: `(105.3*x, 68.0*y)` (공식 `games.py` 기준).
- 46개 track slot ≠ 46명의 고유 선수. 프레임당 관측 인원이 변하고 tracker ID가 끊긴다.
- 처리 분할: 전후반 경계를 넘어 Bi-LSTM window를 만들지 않는다.
- 마스킹 자기지도학습 시 동일 시점 원시 좌표와 파생 특징을 **같이 가려** 정보 누출을 막는다.
- 48구역은 보조 통계/시각화; 실제 3m 거리는 연속 좌표에서 계산한다.
- 순수 비지도 군집의 번호를 경합 정답이라고 단정하지 않는다.

## 출처

- [MTAH 공식 저장소](https://github.com/metrica-sports/more-than-a-hack-2027)
- [MTAH 데이터 명세](https://github.com/metrica-sports/more-than-a-hack-2027/blob/main/DATA.md)
- [MTAH 규정](https://github.com/metrica-sports/more-than-a-hack-2027/blob/main/RULES.md)
- [MTAH 제출 규격](https://github.com/metrica-sports/more-than-a-hack-2027/blob/main/SUBMISSION.md)

## Codex가 검토했던 외부 저장소의 실제 소스까지 읽게 하기 (선택)

`docs/EXTERNAL_CODE_STUDY_GUIDE.md`는 실제 소스 파일별 역할과 pinned commit 링크를 포함하므로, **인터넷 연결 없이도 문서 중심으로 학습할 수 있습니다**. 코드의 모든 구현을 직접 검토하려면, VS Code 터미널에서 다음 스크립트를 수동 실행하세요. 외부 공개 GitHub 코드만 **MTAH 프로젝트 바깥의 형제 폴더**로 복제합니다. 실행이나 학습은 자동으로 하지 않습니다.

```bash
bash MTAH_Relational_Starter/scripts/fetch_reference_sources.sh
```

다운로드 위치 기본값은 `../mtah_external_references/`입니다. Codex에게 이 폴더까지 보여 주려면 VS Code 워크스페이스에 폴더를 추가하세요. **다른 사람의 GitHub 코드를 본인 프로젝트에 무단 복사해 공개하지 마세요.**

연구 노트부터 작성하게 하려면 Codex에 `CODEX_REFERENCE_STUDY_PROMPT.md` 내용을 보낸 뒤 문서상 경로들을 읽게 하세요.
