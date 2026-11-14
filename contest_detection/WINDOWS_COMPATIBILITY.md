# Windows 11 / RTX 5080 준비

## 의존성과 검증 범위

| 용도 | 패키지·권장 환경 | 상태 |
|---|---|---|
| 순수 규칙/특징 계산 및 기본 합성 검사 | Python 3.11 계열, NumPy 2.4.6, pandas 3.1.0, pytest 9.2.0 | 현재 macOS CPU에서 검증; requirements-core.txt |
| 기존 관계 모델의 합성 회귀 검사 | PyTorch 별도 설치 | 현재 환경 torch 2.15.0; GPU 불필요 |
| 선택 영상 디버거/plot | opencv-python 5.0.0.93, matplotlib 3.11.2 | requirements-legacy.txt; Windows 영상 codec 별도 확인 |
| 향후 TranSPORTmer | Windows 11 x64, Python, CUDA 대응 PyTorch, NVIDIA driver | 이번에 구현/설치/학습하지 않음 |
| 저장소의 별도 Tryolabs 방송 처리 | norfair, imageio-ffmpeg, 외부 inference/YOLO source 등 | 경합 핵심 의존성이 아님; 이번 광역 검사 실패 원인 일부 |

정확한 원래 버전과 Windows 설치 가능한 wheel은 구분한다. pinned core를 설치할 수 없으면 오류와 실제 설치 버전을 남겨 별도 호환 환경에서 합성 회귀를 실행한다. 조용히 버전을 바꿔 원래 결과와 동일하다고 주장하지 않는다. `pip freeze`와 Python/OS/GPU 정보를 로컬 환경 기록으로 보관한다. SciPy는 핵심 계산 필수가 아니다. 과거 진단의 Spearman은 average tie ranks의 Pearson으로 구현돼 있다.

## CUDA는 향후 모델용 별도 설치

RTX 5080의 compute capability는 12.0 (`sm_120`)이다. [NVIDIA GPU 목록](https://developer.nvidia.com/cuda/gpus)에서 확인했다. PyTorch 2.12 문서는 Blackwell에 CUDA 13.0 이상 wheel을 안내하고 해당 CUDA 13.0 환경의 Windows driver 최소 580.88을 명시한다. 설치 때 선택한 wheel의 현재 driver 요구사항도 확인한다. [PyTorch 2.12 릴리스](https://pytorch.org/blog/pytorch-2-12-release-blog/)

다음은 공식 버전 표에 있는 torch 2.14.1/cu130을 사용한 **향후 Windows 설치 예시**다. 이 저장소에서 검증한 Windows lockfile은 아니다. 설치 당시 [PyTorch 설치 선택기](https://pytorch.org/get-started/locally/) 및 [공식 버전별 설치 명령](https://pytorch.org/get-started/previous-versions/)의 Windows/Pip/Python/CUDA 조합을 확인한다. 아래 명령은 이번 작업에서 실행하지 않았다.

```powershell
$python = Join-Path $PWD '.venv\Scripts\python.exe'
nvidia-smi
& $python -m pip install torch==2.14.1 --index-url https://download.pytorch.org/whl/cu130
# 장치 확인과 작은 CUDA 연산만 수행하는 수락 검사 예시, 학습 없음
& $python -c "import torch; print(torch.__version__, torch.version.cuda); assert torch.cuda.is_available(); print(torch.cuda.get_device_name(0), torch.cuda.get_device_capability(0), torch.cuda.get_arch_list()); x=torch.ones((2,2),device='cuda'); print((x@x).cpu())"
```

일반 wheel 실행과 custom CUDA extension 컴파일 환경은 별도로 취급한다. CUDA Toolkit/nvcc 및 MSVC Build Tools는 나중에 선택할 extension이 실제로 요구할 때 검토한다. torchvision/torchaudio/DFL loader 등도 현재 규칙 계산을 위해 일괄 설치하지 않는다. 원래 macOS/MPS 환경을 Windows로 복사하지 말고 Windows venv를 새로 만든다. 향후 학습기의 Windows multiprocessing spawn/entrypoint와 DataLoader도 별도 검증 과제다.

## 확인한 플랫폼 의존성

| 위치/문제 | 영향과 이번 처리 |
|---|---|
| 기존 README·보고서의 `export`, `$PWD`, `:` 구분 PYTHONPATH, Bash `\` 줄 연결 | 역사 문서를 보존. 새 README는 PowerShell과 공통 Python launcher 제공 |
| 기존 보고서의 `/private/tmp/cj_*.log`, `/Users/...` 경로 | 과거 환경 증거이며 Windows에서 존재하지 않음. 현재 실행 인자에는 사용하지 않음. 로그는 `Join-Path $env:TEMP ...` 또는 새 repo-local 경로 사용 |
| runner의 `Path.cwd()`와 상대 config 경로 | repo root 실행 필요. `run.py`가 subprocess cwd를 root로 고정 |
| 최신 runner의 날짜가 붙은 상대 cache/input 디렉터리 | OS와 무관하지만 다른 경기/폴더 구조를 지원하지 않음. 새 경로 옵션을 추가하지 않고 REPRODUCTION.md에 현재 계약 명시 |
| `Path.is_relative_to` | Python 3.9+ 필요. 재현은 기존 3.11 권장 |
| 기존 일부 `read_text/write_text`, `gzip.open(...,'wt')`의 encoding 생략 | Windows 코드페이지에 영향 가능. launcher의 자식 Python에 `PYTHONUTF8=1`; direct module 실행 시에도 설정 |
| 원래 verifier들의 `str(relative_path)` metadata key | Windows는 `\`, Unix는 `/`가 나올 수 있음. 계산값과 달리 manifest byte/경로 key는 OS 차이가 있을 수 있음. 신규 manifest는 as_posix 사용 |
| Git autocrlf가 source bytes 변경 | .gitattributes에서 보존 소스 `-text`, 새 도구 `eol=lf`. source SHA 확인 가능 |
| gzip header 시간·float/CSV serialization | 계산 동일성과 압축 파일의 byte hash 동일성은 다름. 기존 sha는 원래 파일 무결성용; 새 출력은 key/schema/값 기준 비교 |
| `.npy` memmap + `TemporaryDirectory` + `os.rename` | `frame_store.py`, `phase1.py`, `temporal_cli.py`의 Windows 열린 파일/정리/rename 동작은 미검증. read-only 기존 cache 재사용 우선. 필요 시 원본 보존한 별도 수정 및 회귀 검증 |
| OpenCV GUI, H.264/avc1 encoder | Windows wheel의 GUI/encoder 가용성에 따라 영상 테스트/preview가 실패할 수 있음. 기본 테스트에서 영상 제외; 특징 계산에는 영향 없음 |
| 과거 preservation snapshot의 절대경로·mtime | 컴퓨터 간 복사로 mtime이 달라져 원래 verifier를 그대로 적용하기 어려움. 새 실행 전에 해당 컴퓨터의 snapshot 작성 필요 |

원래 경합 Python src/tests에는 `/Users`, `/private/tmp` 실행 절대경로나 Bash subprocess 호출이 발견되지 않았다. 이미 대부분 Path를 사용한다. 이를 이유로 Windows 실제 실행 성공을 단정하지 않는다. 원본 계산 코드에 호환성 수정은 없고 신규 launcher/adapter/문서/gitattributes만 추가했다.

## Windows 인수 검사

1. 새 clone에서 `verify-snapshot`과 `test` 실행, 실제 wheel 버전 기록.
2. 필요하면 `test --legacy` 실행. 합성 관계 모델은 CPU로 충분하며 영상 codec 실패는 별도 기록.
3. 라이선스가 있는 기존 cache를 읽기 전용으로 열고 frame id/period/team/NaN 및 key 순서 검사.
4. 소량 동일 입력에 원래 함수와 새 adapter를 적용해 허용오차 1e-12, NaN 위치, 상태/mask 동일 여부 확인.
5. 향후 CUDA는 위 작은 연산으로 별도 확인. Windows 전체 경기 실행과 TranSPORTmer 학습은 이번 검증 범위 밖이다.
