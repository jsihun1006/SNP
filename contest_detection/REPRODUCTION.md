# 입출력 및 최종 실행 경로

## 실제 최종본

- Entry point: `competition_judgement/src/competition_judgement/run_candidate_v23.py:main`.
- 핵심: `candidate_decision_v23.decide_clusters`, `candidate_frames_v23.CandidateFrames`.
- 고정 설정: `competition_judgement/configs/candidate_v23.json`; 참조 설정 `pattern_b1.json`, `local_temporal_b2.json`, `evidence_stage_b.json`.
- 테스트: `competition_judgement/tests/test_candidate_v23.py`의 45개 신규 합성 검사와 앞 단계 테스트.
- 최종 산출물의 원래 위치: `data/derived/competition_judgement/candidate_v23_final_20261110/`.
- 미완료 `candidate_v23_incomplete_radius_join_20261110/`는 최종본이 아니다.

## 입력 계약

원본 ATD는 `games.asset_paths(source='tactical')`의 `data/tactical/BAR-ATH_tactical_atd.xml`과 `_atd.txt`다. 메타데이터에서 팀 소속을 읽고 정규화 좌표를 경기장 metre 단위로 변환한다. raw slot 46개는 선수 영구 ID가 아니다. 볼/speed/tag/GT를 특징 계산에 사용하지 않는다. 공식 축은 tactical video의 0-based `frame=0..199318`, 25fps, period=0/1/2다. 공식 평가 block은 half1 11185..90651, half2 110808..190197이다.

FrameStore는 `frame_ids.npy[N]`, `periods.npy[N]`, `xy_m.npy[N,46,2]`, `team_codes.npy[46]`와 `parse_missing/partial/invalid.npy[N,46]`, `cache.json`을 사용한다. 팀 코드는 0/1이며 source fingerprint·pitch·fps·slot/행 수를 검증한다. NaN/Inf와 관측/QC mask를 구분한다. `np.load(...,mmap_mode='r',allow_pickle=False)`로 기존 cache를 연다.

v2-3 runner는 다음 5개를 직접 읽고 SHA256으로 보존 확인한다. CSV 실제 요구 열은 runner의 `profile_columns()` / `local_columns()`가 단일 근거다.

| 입력 (base=`data/derived/competition_judgement/`) | 역할 |
|---|---|
| `evidence_pattern_b1_final_20261110/cluster_pattern_profiles.csv.gz` | key + 현재 공간 + 개별 Evidence 지원/상태 + A/B/C/D obs/seq/outcome |
| `local_temporal_b2_final_20261110/cluster_local_temporal.csv.gz` | 동일 key/순서 + Short/Near Local 상태/지원/순서 |
| `local_v2_1_20261109/local_frame_features.csv` | 전체 frame, period, QC, 팀 수, radius별 군집 수 |
| `local_v2_1_20261109/local_clusters.csv` | radius family 진단 및 원래 군집 기하 |
| `stage1_v1_final_20261109/candidate_frames.csv` | Validation v1 교차표 비교, 정답 아님 |

runner는 결정 추출 뒤 고정 missingness/jitter 진단도 수행한다. 따라서 위 5개만으로 **전체 runner**를 실행할 수는 없다. 추가로 아래 자료가 필요하다.

- `data/derived/relational_temporal_v1_20261109/raw_cache/`와 같은 폴더의 `split_manifest.json`.
- `temporal_v2_2_final_20261109/cluster_temporal_features.csv`, `preregistered_config.json`, `pair_cache/pair_kernel.npy`, `pair_3m.npy`, `pair_5m.npy`.
- `missingness_stage_a_final_20261109/reference_keys.csv` 및 `reference_temporal_arrays.npz` 및 `train_pattern_library.npz` (고정 Validation reference·동결 시계열·Train donor).
- 관련 Stage A/B/B.1/B.2 설정 파일. 원래 `run_pattern_b1.load_source`, `missingness_cache.ReadOnlyAnchorCache`, `pattern_diagnostics`의 입력 경로를 따른다.

위 이름은 현재 원본 코드에 고정된 상대 경로다. 공개 clone은 코드만 포함하므로 라이선스 자료와 cache를 로컬로 따로 준비해야 한다. 값 자체의 복사/공개 없이 results/artifact_manifest.json에 원래 상대 경로와 SHA256을 남겼다. feature 함수만 사용하는 경우 전체 경기 산출물은 필요 없고 해당 함수의 배열/mask 입력만 준비하면 된다.

## 출력 계약

`cluster_candidate_decisions.csv.gz`: 789,005행 × 16열. key=`frame,radius_m,cluster_id`; state/reason, spatial gate/failure, observed/decision basis, dynamic support, complete/partial support, review reason, 미래 의존성, upstream unknown 사유를 담은 sidecar다. 원래 Evidence/time 배열을 복사하지 않는다.

`frame_candidate_decisions.csv.gz`: 199,319행 × 39열. 원래 frame 순서, 대표 상태/reason/Anchor, radius별 출처·개수, 현재 strict/양 팀 관측, 미래 의존성 등을 유지한다. **이 sidecar에는 split/period 열이 없다.** 원래 frame 테이블과 split manifest로 join해야 한다. 정확한 열 목록은 results/artifact_validation.json과 원래 `data_dictionary.json` 참조.

나머지는 `summary.json`, `data_dictionary.json`, `preregistered_config.json`, `legacy_input_preservation.json` 및 Validation 집계/robustness CSV다. 공개 repo에 이 실제 경기 sidecar나 reference 시계열은 업로드하지 않는다.

## Windows 실행

README대로 venv를 만든 뒤 repo root에서 실행한다. 기존 출력 폴더는 runner가 거부한다.

```powershell
$python = Join-Path $PWD '.venv\Scripts\python.exe'
& $python contest_detection/run.py v2_3 --help
$runName = 'candidate_v23_reproduction_' + (Get-Date -Format 'yyyyMMdd_HHmmss')
& $python contest_detection/run.py v2_3 --config competition_judgement/configs/candidate_v23.json --output "data/derived/competition_judgement/$runName"
```

직접 module을 호출하려면 FEATURE_REUSE.md의 PowerShell PYTHONPATH/UTF8 설정 후 다음을 사용한다.

```powershell
& $python -m competition_judgement.run_candidate_v23 --output "data/derived/competition_judgement/$runName"
```

## 기존 자료가 없는 새 로컬 환경의 생성 순서

**v1만 준비할 때**는 [입력 생성 안내](V1_INPUT_GENERATION.md)의 `python contest_detection/run.py prepare_v1`을 사용한다. 기존 `build_frame_cache`, `make_split_manifest`, `run_phase1`을 호출해 v1의 필수 캐시·분할·특징만 준비한다. 원본 ATD 두 파일은 별도 입력이다.

아래는 **전체 재현용 참고 명령**이며 이번에 실행하지 않았다. 라이선스 ATD가 있는 별도의 새 clone/데이터 작업 공간에서만 실행한다. 날짜 폴더가 이미 있으면 기존 자료를 보존하고 새 작업 공간을 사용한다. 현재 runner의 고정 상대 경로 때문에 중간 폴더 이름을 임의 변경하면 다음 단계가 찾지 못한다.

먼저 FEATURE_REUSE.md의 PowerShell PYTHONPATH를 설정한다.

```powershell
& $python -m mtah_relational.temporal_cli build --out-dir data/derived/relational_temporal_v1_20261109 --seed 2027
& $python contest_detection/run.py v1 --out-dir data/derived/competition_judgement/stage1_v1_final_20261109
& $python contest_detection/run.py v2_1 --out-dir data/derived/competition_judgement/local_v2_1_20261109
& $python contest_detection/run.py v2_2 --out-dir data/derived/competition_judgement/temporal_v2_2_final_20261109
& $python contest_detection/run.py stage_a --output data/derived/competition_judgement/missingness_stage_a_final_20261109
& $python contest_detection/run.py stage_b --output data/derived/competition_judgement/evidence_stage_b_final_20261109
& $python contest_detection/run.py stage_b_revised --output data/derived/competition_judgement/evidence_pattern_b1_final_20261110
& $python contest_detection/run.py stage_b_2 --output data/derived/competition_judgement/local_temporal_b2_final_20261110
& $python contest_detection/run.py v2_3 --output data/derived/competition_judgement/candidate_v23_reproduction
```

v1은 기본 Phase1 `features.csv`/quality/zones가 없으면 자체 신규 output/player_features에 player-only 특징을 생성한다. 기존 features를 쓸 경우 기본 경로 `data/derived/relational_phase1_v1_20261109/`의 features.csv, quality_report.json, zones_48.csv 계약을 만족해야 한다. `--features`, `--cache`, `--split-manifest` 인자로 v1 입력을 바꿀 수 있다. 최종 runner 전체를 일반적인 다른 경기 CLI로 바꾸지는 않았다.

## 검증 및 계산 부담

원래 v2-3 전체 추출+고정 진단은 macOS CPU에서 52.89초로 기록돼 있다. B.1 약297.93초/B.2 약369.64초 등의 전처리는 별도다. 하드웨어/디스크/캐시 상태에 따라 달라지고 peak RAM은 측정하지 않았다. 현재 전체 source/cache/요약 검증 대상 208개는 약5.11GB이며, 이 크기는 전체 원본 경기 데이터 용량이 아니다. GPU는 규칙 계산에 필요 없다.

새 실행의 값 비교는 key·순서·상태·mask·NaN 위치를 보존하고 float 허용오차 `atol=1e-12,rtol=0`을 사용한다. seed 2027과 고정 config/cohort를 유지한다. 원래 `verify_candidate_v23`는 새 결과에 verification이 없을 때만 쓸 수 있고, 477 초과 테스트 로그 및 **현재 컴퓨터의** 코드/산출물 snapshot을 요구한다. 과거 절대경로·mtime snapshot을 Windows에 그대로 적용하지 않는다. 합성 419개 로그는 이 과거 >477 guard를 만족하지 않으므로 직접 대체할 수 없다. 이 제한을 통과시키려고 verifier를 수정하지 않았다.

기존 결과만 읽는 감사 도구는 다음처럼 새 report 폴더에 실행한다. 큰 파일을 hash하고 테이블을 읽지만 full inference는 하지 않는다.

```powershell
& $python contest_detection/audit_artifacts.py . (Join-Path $env:TEMP ('contest_audit_' + (Get-Date -Format 'yyyyMMdd_HHmmss')))
```
