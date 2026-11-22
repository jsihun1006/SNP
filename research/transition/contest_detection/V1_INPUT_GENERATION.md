# v1 실행 입력을 만드는 코드

실행 진입점은 `prepare_v1_inputs.py`이며 기존 생성 함수를 그대로 호출한다. 캐시·시간 분할·기하 특징을 생성하고 모델 학습이나 경합 후보 추론은 실행하지 않는다. 원본 ATD 두 파일은 대회에서 제공받아야 하는 입력이며 코드로 만들어내는 자료가 아니다. 압축파일이나 실제 경기 데이터는 이번 커밋에 포함하지 않는다.

## 어떤 코드가 어떤 파일을 만드는가

| 생성 파일 | 기존 생성 함수 | 실제 소스 |
|---|---|---|
| raw_cache/cache.json | `build_frame_cache` | `MTAH_Relational_Starter/mtah_relational/frame_store.py` |
| raw_cache/frame_ids.npy | `build_frame_cache` | 위와 동일 |
| raw_cache/periods.npy | `build_frame_cache` | 위와 동일 |
| raw_cache/xy_m.npy | `build_frame_cache` | 위와 동일 |
| raw_cache/team_codes.npy | `build_frame_cache` | 위와 동일 |
| raw_cache/parse_missing.npy | `build_frame_cache` | 위와 동일 |
| raw_cache/parse_partial.npy | `build_frame_cache` | 위와 동일 |
| raw_cache/parse_invalid.npy | `build_frame_cache` | 위와 동일 |
| split_manifest.json | `make_split_manifest` + `SplitManifest.save` | `MTAH_Relational_Starter/mtah_relational/splits.py` |
| features.csv | `run_phase1` | `MTAH_Relational_Starter/mtah_relational/phase1.py` |
| quality_report.json | `run_phase1` | 위와 동일 |
| zones_48.csv | `run_phase1(zone_stride=25)` | 위와 동일 |

원래 함수는 이미 `transition` 브랜치의 보존 커밋 `d19600b462fdae2530deb65bf3c9ab54c733a723`에 포함돼 있다. 이번에는 이를 한 번에 실행하는 진입점과 생성 파일별 설명·합성 검사를 추가했다. 알고리즘을 새로 작성하거나 기존 생성 함수를 변경하지 않았다. `run_phase1`은 원래 동작에 따라 `symmetric_features.csv`, `schema.json`, `LOCAL_ONLY.txt`도 생성한다.

필요한 원본 입력:

```text
data/tactical/BAR-ATH_tactical_atd.xml
data/tactical/BAR-ATH_tactical_atd.txt
```

기본 생성 경로:

```text
data/derived/
    relational_temporal_v1_20261109/
        split_manifest.json
        raw_cache/                 # 위 8개 파일
    relational_phase1_v1_20261109/
        features.csv
        quality_report.json
        zones_48.csv
        ...                        # 원래 함수의 부가 메타데이터/특징
```

## Windows PowerShell

Python 3.11 환경과 원본 ATD를 준비한 새 clone에서 실행한다.

```powershell
git clone --branch transition --single-branch https://github.com/jsihun1006/SNP.git
Set-Location SNP
py -3.11 -m venv .venv
$python = Join-Path $PWD '.venv\Scripts\python.exe'
& $python -m pip install -r contest_detection/requirements-core.txt

# 입력 존재 여부와 출력 경로만 확인. 실제 데이터 읽기/생성 없음.
& $python contest_detection/run.py prepare_v1 --plan

# 기존 함수를 호출해 v1 입력 12개 및 원래 부가 산출물 생성.
& $python contest_detection/run.py prepare_v1

# 입력 생성 후, 필요할 때 v1 후보 추론을 별도로 실행.
& $python contest_detection/run.py v1 --out-dir data/derived/competition_judgement/v1_run01
```

ATD를 외부 드라이브에 둔다면 실행 전에 환경 변수를 설정한다. 이 설정은 입력 생성과 이후 v1 추론에 공통으로 적용한다.

```powershell
$env:MTAH_DATA_DIR = 'D:\MTAH'
# D:\MTAH\tactical\BAR-ATH_tactical_atd.xml / .txt가 있어야 한다.
& $python contest_detection/run.py prepare_v1 --plan
& $python contest_detection/run.py prepare_v1
```

`--data-dir`로 생성기 입력만 지정할 수도 있지만, 이후 v1 추론에서도 같은 원본을 찾도록 `MTAH_DATA_DIR`를 설정해야 한다. 생성 경로는 repo의 data/derived 아래를 기본으로 한다.

## 기존 파일이 있을 때

생성 대상 중 하나라도 이미 있으면 전체 실행은 쓰기 전에 중단한다. 덮어쓰기 옵션은 제공하지 않는다. 필요한 그룹만 골라 생성할 수 있다.

```powershell
& $python contest_detection/run.py prepare_v1 --only cache
& $python contest_detection/run.py prepare_v1 --only split
& $python contest_detection/run.py prepare_v1 --only features
```

새 경로를 지정하려면 `--cache-dir`, `--split-manifest`, `--features-dir`를 사용한다. 이후 v1의 `--cache`, `--split-manifest`, `--features`에도 대응하는 경로를 전달한다. 그룹별 생성은 원래 함수의 보존 규칙을 따르며, 중간 실패 시 이미 완료된 그룹을 삭제하지 않는다. 재실행할 때 미완료 그룹만 선택한다.

`--only split`은 원본 트래킹 없이 games.py의 공식 전후반 구간에서 시간 분할만 만든다. cache/features 생성은 원본 두 파일을 모두 요구한다.

## 검증 범위

새 진입점은 60프레임·4슬롯 합성 ATD로 실제 기존 생성 함수를 호출해 검사했다. 전체 12개 필수 파일 생성, frame/period 정렬, 25프레임 간격의 48구역 출력, 원본 hash/mtime 보존, 기존 출력 거부, 그룹별 실행, 누락 입력, 중첩 출력 거부, plan 무기록 동작을 확인한다.

이번 작업에서 실제 경기 전체의 캐시/특징을 다시 생성하거나 후보를 추론하지 않았다. Windows 실제 실행은 후속 검증 대상이다. CPU NumPy/pandas 계산이며 PyTorch/CUDA는 필요하지 않는다. v1 판정값을 TranSPORTmer 정답 라벨로 사용하지 않는 기존 원칙도 유지한다.

이번 실행: 기존 기본 검사 251개 + 신규 생성기 검사 7개 = **258 passed in 2.47s**. 명령은 `python contest_detection/run.py test MTAH_Relational_Starter/tests/test_prepare_v1_inputs.py`이며 stdout은 results/prepare_v1_tests.txt에 보존했다. 실제 실행기 `prepare_v1 --plan`도 다른 작업 디렉터리에서 파일 생성 없이 확인했다.
