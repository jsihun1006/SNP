# Windows 처음 실행하기

Windows 10/11 **Intel/AMD 64bit**, 인터넷 연결이 필요합니다. 기존 Python·Conda 환경에 패키지를 설치하지 않습니다.
프로젝트 폴더 안에 Python 3.10 가상환경 `sss1`을 만듭니다.

1. Git이 있으면 PowerShell에서 다음 명령으로 새 폴더에 받습니다.

   ```powershell
   git clone --branch SSS --single-branch https://github.com/jsihun1006/SNP.git SNP-SSS
   Set-Location .\SNP-SSS\sss1_reimplementation
   ```

   Git이 없어도 [SSS 브랜치 ZIP](https://github.com/jsihun1006/SNP/archive/refs/heads/SSS.zip)을
   내려받아 압축을 풀고 `sss1_reimplementation` 폴더를 열면 됩니다.

2. **`WINDOWS_CPU.cmd`를 더블클릭**합니다. PowerShell에서는 `& .\WINDOWS_CPU.cmd`를 실행합니다.
   Python 3.10 x64가 없으면 winget으로 설치합니다. winget이 없으면 안내하는
   [Python 3.10.11 x64 설치 파일](https://www.python.org/downloads/release/python-31011/)을 설치한 후 다시 실행합니다.
   가상환경 생성 → 패키지 설치 → 환경 점검 → CPU 단위 테스트 → 가중치·샘플 다운로드 → 실제 BAAT 추론을 실행합니다.
   처음에는 패키지와 약 53MB의 YOLO26-Large 가중치를 다운로드하므로 시간이 걸립니다.

3. 성공하면 `outputs\first_run\CoreCPU\sample\visualizations`에 사람 검출 이미지가 생깁니다.
   같은 폴더의 `predictions.csv`, `results.json`, `timing.json`에서 좌표와 시간을 확인할 수 있습니다.
   샘플은 축구 영상이 아닌 공개 테스트 사진입니다. `status.json`의 `completed: true`로 전체 완료를 확인합니다.

4. 내 이미지나 영상을 실행합니다. 경로를 따옴표로 감싸십시오.

   ```powershell
   & .\RUN_WINDOWS.cmd "C:\videos\match.mp4"
   # 또는 옵션을 지정합니다.
   powershell.exe -NoProfile -ExecutionPolicy Bypass -File .\scripts\run_windows.ps1 -InputPath "C:\images\frame.jpg" -Mode baat -Device cpu -OutputPath "C:\sss_results"
   ```

   기본 결과 위치는 `outputs\run_날짜_시간`입니다. 영상은 각 프레임을 독립 처리하며 시간적 선수 ID는 추적하지 않습니다.
   CPU에서는 큰 영상 처리에 오래 걸리므로 첫 검증은 이미지 한 장으로 진행하십시오.

NVIDIA GPU가 있을 때는 `WINDOWS_CUDA.cmd`로 전체 Pose 의존성까지 설치할 수 있습니다.
설치 조합은 torch 2.1.0+cu118 / torchvision 0.16.0 / mmcv 2.1.0입니다.
CUDA Toolkit·Docker·WSL을 별도로 설치할 필요는 없으며 호환되는 NVIDIA 드라이버는 필요합니다.
새 GPU가 이 구형 PyTorch 조합에서 지원되지 않으면 GPU 검증 단계에서 실패 원인을 남깁니다.
CPU 설치·실행은 GPU 없이 가능합니다. CUDA 프로필은 Body-7 초기화 가중치도 다운로드합니다.

```powershell
& .\WINDOWS_CUDA.cmd
powershell.exe -NoProfile -ExecutionPolicy Bypass -File .\scripts\run_windows.ps1 -InputPath "C:\images\frame.jpg" -Device cuda:0
```

**Pelvis/Ground 및 경기장 좌표가 필요하면** SynLoc 데이터로 실제 학습한 2점 Pose 체크포인트와
해당 이미지·프레임의 카메라 보정 정보가 필요합니다. Body-7 파일은 학습 완료 2점 모델이 아닙니다.
가중치·보정 정보가 없을 때 좌표를 임의 생성하지 않습니다. 기본 첫 실행은 선수 bbox까지 검증합니다.

```powershell
powershell.exe -NoProfile -ExecutionPolicy Bypass -File .\scripts\run_windows.ps1 -InputPath "C:\images\frame.jpg" -PoseCheckpoint "C:\models\trained_two_point.pth" -Camera "C:\datasets\camera.json" -Device cuda:0
```

오류가 나면 `outputs\install`의 최근 `.log`와 `outputs\first_run\프로필\status.json`, `environment.json`을 확인합니다.
다운로드된 가중치는 SHA256을 검증하며 손상되었으면 자동 사용하지 않습니다.
`WINDOWS_*.cmd`의 PowerShell 실행 정책 옵션은 해당 실행 프로세스에만 적용하며 PC 전체 정책을 변경하지 않습니다.
설치기를 다시 실행해 이어갈 수 있으며, 기존 환경에 다른 Python 버전이 설치되어 있어도 그대로 둡니다.

자동 Windows 검증은 GitHub의 **Actions → SSS Windows first run**에서 확인할 수 있습니다.
[첫 Windows 실행 검증](https://github.com/jsihun1006/SNP/actions/runs/43567391206)은
깨끗한 Windows Server 2025 x64 runner에서 CPU 테스트 42개와 실제 BAAT 샘플 추론을 통과했습니다.
설치 로그와 결과 이미지는 해당 실행의 `sss-windows-cpu-first-run` artifact에서 내려받을 수 있습니다.
CUDA 프로필·일반 PC의 Python 자동 winget 설치 분기는 별도 실기 검증 대상입니다.
`WinError 126` 등 DLL 로딩 오류가 나면 [Microsoft Visual C++ x64 런타임](https://aka.ms/vs/17/release/vc_redist.x64.exe)을
설치하고 다시 실행하십시오. 경로 길이 오류는 `C:\work\SNP-SSS`처럼 짧은 위치에 받아 확인하십시오.
데이터 준비·학습·정량 평가는 [README.md](README.md), 재현 범위는 [reproduction_report.md](reproduction_report.md)를 참조하십시오.
