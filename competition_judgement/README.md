# competition_judgement

선수 좌표·팀 소속으로 근접·혼합·밀집과 과거1초 집합 변화를 계산하는 규칙 기반 경합 **후보** 모듈이다. Stage0/1 구현과 전체199,319프레임 로컬 실행을 완료했다. 경합 확률이나 H/A/D/X 점유 라벨을 출력하지 않는다.

먼저 [Stage1 실행 보고서](docs/STAGE1_EXECUTION_REPORT.md)를 읽는다. 실제 API와 문서 차이는 [Stage0 감사](docs/STAGE0_AUDIT.md)에 기록했다. 설계는 docs/00–05, 과거 연구 스냅샷은 references/에 있다.

## 실행

프로젝트 루트의 기존 `mtah2027` 환경을 사용한다. NumPy·Pandas·pytest를 사용하며 새로운 의존성 설치나 신경망 실행은 필요 없다.

```bash
conda activate mtah2027
export PYTHONPATH="$PWD/competition_judgement/src:$PWD/MTAH_Relational_Starter:$PWD"
export PYTHONDONTWRITEBYTECODE=1
python -m pytest -q -p no:cacheprovider competition_judgement/tests
python -m competition_judgement.cli --out-dir data/derived/competition_judgement/stage1_reproduction
```

기존 결과 폴더는 덮어쓰지 않는다. 실제 경기 결과는 `data/derived/competition_judgement/` 아래로 제한한다. 모든 결과가 Git ignore 대상이며 서버 없이 로컬 파일로 읽는다.

기본 입력:

* `data/derived/relational_phase1_v1_20261109/features.csv`: 기존 player-only 기하·품질. 없으면 기존 Phase1 reader로 새 실행 폴더에 생성한다.
* 위 폴더의 `quality_report.json`, `zones_48.csv`: 정책 계약과1fps 보조 구역 통계. 구역 파일이 비었으면 기존 Phase1을 zone_stride25로 재생성해야 한다.
* `data/derived/relational_temporal_v1_20261109/raw_cache/`: 기존 player-only 캐시. read-only robustness 입력이다.
* 위 temporal 폴더의 `split_manifest.json`: 반별70/15/15 raw-time 분할.
* 공식 ATD metadata/raw: 메타데이터·캐시 출처 hash 확인용. 새 parser는 작성하지 않았다.

경로는 `--features`, `--cache`, `--split-manifest`, `--config`로 바꿀 수 있다. raw cache가 없으면 기존 `mtah_relational.temporal_cli build`로 새 로컬 캐시를 먼저 준비한다.

## 결과와 의미

* `candidate_frames.csv`: frame0..199318 전체, 기하/품질·과거 변화·score/state/reason.
* `candidate_segments.csv`: 연속 state/quality/반/split 기준으로 모든 행을 분할한 구간.
* `policy_resolved.json`: Train 고유 strict 관측에서 적합한 고정 임계값·설정·분할·출처 fingerprint.
* `quality_and_sensitivity.json`: 전체/반/품질 집계와 사전 지정 감도표·Validation 노이즈 검사.
* `verification.json`, `LOCAL_ONLY.txt`: 보존 및 로컬 보관 기록.

상태는 `candidate`, `not_candidate`, `insufficient_observation`이다. strict 관측에서만 후보 여부를 판단한다. metric-valid인 low-quality 관측도 후보/비후보를 확정하지 않는다. 관측 완화 후보 정책은 Stage2에 남겼다.

5프레임 연속 threshold 통과를 현재까지 확인하고 후보를 출력한다. 첫4프레임은 pending인 not_candidate이며 나중에 소급하지 않는다. split/half/gap/관측 불충분에서 이력이 끊긴다. 따라서 출력 후보 segment 자체가 최소5프레임인 것은 아니다.

## 한계와 보존

가까운 마킹·압박·재시작 대기와 실제 볼 경합은 선수 좌표만으로 확정 구분할 수 없다. 후보 빈도는 실제 경합 빈도가 아니며 QC coverage는 recall이 아니다.

기존 mtah_relational·A/B 체크포인트·starter/evaluation/games.py·원본 데이터를 보존했다. 신경망 재학습, C/Fusion, 점유 decoder, Smart Tagging·볼·GT 사용은 제외했다. 영상 수작업 라벨이나 장면별 상수 튜닝을 하지 않았다. 실제 경기 자료·복원 가능한 파생물은 외부 서비스나 공개 GitHub에 업로드하지 않는다. Push하지 않는다.
