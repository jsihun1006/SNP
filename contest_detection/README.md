# 경합 탐지 이력 복원 / transition

2026-11-15에 보존한 MTAH 선수 좌표 기반 **경합 후보** 코드다. 원래 `competition_judgement/src/competition_judgement/`와 `MTAH_Relational_Starter/` 구조를 유지했다. 알고리즘 소스와 설정은 원본 byte를 보존했고, `features.py`는 기존 함수 객체를 그대로 내보낸다. TranSPORTmer 구현·학습·전체 경기 재추론은 수행하지 않았다.

- [개발 이력](DEVELOPMENT_HISTORY.md): v1 → v2-1 → v2-2 → Missingness → Evidence → B.1 → B.2 → v2-3.
- [버전별 경로·복원 한계](VERSION_MANIFEST.md), [Windows 호환성](WINDOWS_COMPATIBILITY.md), [입출력·재현](REPRODUCTION.md), [특징 재사용](FEATURE_REUSE.md).
- [v1 입력 파일 생성 코드·실행법](V1_INPUT_GENERATION.md): 원본 ATD에서 캐시·분할·기하 특징을 생성한다. `python contest_detection/run.py prepare_v1`로 실행한다.
- [최종 작업 보고](FINAL_REPORT.md), [검증 결과](VALIDATION.md), [원본 산출물 경로·SHA256](results/artifact_manifest.json), [원본 소스 SHA256](results/source_snapshot.json).

기준은 `origin/KKJ`의 `17f527eb31c4bc2a8f9dc61d0c08e93cd1173808`이다. fetch 후 접근 가능한 Git log/reflog에서 이 경합 패키지의 과거 커밋은 발견되지 않았다. 현재 로컬의 단계별 모듈·설정·테스트와 11월 9–10일 산출물의 hash 기록을 연결했다. `versions/`는 증거와 경로를 안내하며, 존재하지 않는 과거 소스나 커밋을 만들어내지 않는다. 신규 커밋은 11월 15일의 보존·호환성 작업이다.

## Windows 11 / PowerShell 빠른 시작

Python 3.11 계열을 권장한다. 원래 검증 환경은 Python 3.11.17 / NumPy 2.4.6 / pandas 3.1.0 / pytest 9.2.0 / macOS arm64 CPU다. Windows·RTX 5080에서의 실제 실행은 아직 검증하지 않았다.

```powershell
git clone --branch transition --single-branch https://github.com/jsihun1006/SNP.git
Set-Location SNP
py -3.11 -m venv .venv
.\.venv\Scripts\python.exe -m pip install -r contest_detection/requirements-core.txt
.\.venv\Scripts\python.exe contest_detection/run.py verify-snapshot
.\.venv\Scripts\python.exe contest_detection/run.py test
```

이미 clone했다면, 기존 수정 사항을 보존하면서 별도 worktree를 만든다.

```powershell
git fetch origin
git worktree add --detach ..\SNP-transition origin/transition
Set-Location ..\SNP-transition
```

venv 활성화 없이 직접 Python을 호출하므로 실행 정책 변경이 필요 없다. `run.py`는 `pathlib.Path`, `os.pathsep`, Python subprocess 인자 목록으로 repo root·import 경로·UTF-8 설정을 구성한다. 실행 위치에 관계없이 동작하며 경로의 공백도 보존한다. 아래처럼 Python 실행 파일을 변수로 지정해도 된다.

```powershell
$python = Join-Path $PWD '.venv\Scripts\python.exe'
& $python contest_detection/run.py v2_3 --help
# 라이선스가 있는 로컬 입력/기존 캐시를 REPRODUCTION.md대로 준비한 경우에만 실행
$runName = 'candidate_v23_windows_' + (Get-Date -Format 'yyyyMMdd_HHmmss')
& $python contest_detection/run.py v2_3 --output "data/derived/competition_judgement/$runName"
```

기본 테스트는 경기 데이터·동영상·Torch·GPU가 필요 없다. `--legacy`는 기존 관계 모델 합성 검사와 영상 디버거까지 포함한다. `requirements-legacy.txt`와 별도 PyTorch가 필요하다. 저장소 전체의 방송 처리 검사까지 포함하는 `--repository`는 별도 외부 자산·의존성을 요구하며 이번 환경에서 14개 실패가 확인됐다. 자세한 범위와 결과는 VALIDATION.md에 기록했다.

```powershell
& $python -m pip install -r contest_detection/requirements-legacy.txt
# Torch 설치는 WINDOWS_COMPATIBILITY.md 참조
& $python contest_detection/run.py test --legacy
# 선택 사항: 경합 외 코드까지 포함하는 광역 검사
& $python contest_detection/run.py test --repository
```

macOS/Linux도 같은 launcher를 사용한다.

```sh
python contest_detection/run.py verify-snapshot
python contest_detection/run.py test
```

## 알고리즘·결과의 의미

v1은 Train-only threshold와 과거 25프레임 변화를 이용하는 causal 후보 탐지다. v2-1은 3m/5m 혼합 군집, v2-2는 현재 고정 Anchor의 ±50프레임 관계, Stage A는 결측과 관측 지원, Stage B/B.1은 개별 Evidence와 순서 검증, B.2는 같은 Anchor·같은 Short/Near scope의 Rise→Fall을 다룬다.

v2-3은 현재 공간 gate를 통과한 Anchor에서 A/B 지원 → PATTERN, 같은 scope의 완전 관측 Rise→Fall → LOCAL, 충분한 동적 부재 → NO_DETECTED_SIGNAL, 나머지 유효 공간 → REVIEW, 현재 관측 불가 → INSUFFICIENT를 출력한다. Frame 집계는 PATTERN > LOCAL > REVIEW > INSUFFICIENT > NO 우선순위이며 서로 다른 Anchor의 근거를 합쳐 후보를 만들지 않는다. v2-3은 최대 미래 50프레임을 사용하는 offline 계산이다.

확인된 기존 결과는 전체 199,319 Frame / 789,005 Cluster view다. Validation 23,828 Frame 중 후보 6,139개(25.76%), Validation 112,286 Cluster view 중 PATTERN 8,850 / LOCAL 3,503개다. 0.1m jitter 실험의 고정 3,191 probe에서 신규 후보 213 / 소실 48개였다. 후보 수는 정확도나 독립 경합 사건 수가 아니다.

## 향후 TranSPORTmer

직접 재사용 가능한 함수와 입력 mask 계약은 FEATURE_REUSE.md에 있다. Local Density·Cross-team Proximity·시간 변화·관측된 근접 지속시간·Rise→Fall·Evidence·Missingness를 구분한다. m/s 단위 Approach Velocity나 영구 선수 ID 기반 Cluster Persistence는 아직 독립 기능으로 구현되지 않았다.

**v2-3 규칙 기반 판정은 TranSPORTmer의 정답 라벨이 아니다.** 추후 DFL 이벤트 기반 학습과 Ablation Study를 별도로 설계해야 한다. 이번 브랜치는 코드 보존과 Windows 준비 범위이며, 모델·학습기·DFL 라벨러를 추가하지 않았다. 원본 경기 자료·프레임/군집 파생물·동영상·캐시·체크포인트는 Git에 포함하지 않는다.
