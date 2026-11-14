# Windows 첫 실행 배포 — 2026-11-14

기준 브랜치: `origin/KKJ` (`17f527e`). 배포 브랜치: `SSS`.
다른 MTAH 작업과 미커밋 변경을 보존하기 위해 별도 worktree에서 커밋합니다.

추가한 진입점:

- `WINDOWS_CPU.cmd`: Python 3.10 x64 탐색/설치, 별도 sss1 환경, CPU 테스트, 실제 YOLO26 BAAT 샘플 실행.
- `WINDOWS_CUDA.cmd`: 같은 과정과 CUDA/MMCV 검증, Body-7 초기화/미학습 Pose 구조 검증.
- `RUN_WINDOWS.cmd`, `scripts/run_windows.ps1`: 이미지·영상 입력과 선택적인 Pose·보정 파일 지정.
- `WINDOWS_START.md`: Git clone/ZIP부터 시작하는 Windows 전용 안내.
- `.github/workflows/sss-windows.yml`: 클린 Windows runner에서 CoreCPU 설치·테스트·실제 샘플 추론 및 로그 보관.

설치기는 기존 Python/Conda에 패키지를 설치하지 않으며 프로젝트의 `sss1` 환경만 사용합니다.
Python을 못 찾으면 winget 또는 공식 설치 안내를 사용하고, Microsoft Store 실행 별칭을 Python으로 오인하지 않습니다.
sskit은 commit 고정 소스 ZIP으로 설치하므로 ZIP으로 프로젝트를 받은 PC에서도 Git 없이 설치할 수 있습니다.
모델 가중치는 Git에 넣지 않고 공식 URL에서 받으며 SHA256을 검증합니다.
한국어/공백이 있는 Windows 이미지 경로는 Python 파일 I/O와 OpenCV codec을 조합해 읽고 저장합니다.
PowerShell 스크립트와 CMD에는 Windows checkout 시 CRLF를 적용합니다.

실제로 수행한 검증:

- 별도 clean worktree에서 전체 **45개 테스트 통과**, skip/실패 없음.
- `scripts.first_run --profile CoreCPU`의 환경 점검 → 핵심 테스트 → 가중치 검증 → 실제 BAAT 추론 완료.
- Microsoft PowerShell 7.4.6 실제 파서로 `.ps1` 두 파일 문법 검사 통과.
- 한글·공백 이미지 경로, 손상 체크포인트 거부, 불완전 다운로드 정리, GPU 없는 CUDA 프로필 거부 검증.

검증 호스트는 macOS ARM64입니다. Windows PowerShell 5.1/Windows x64에서의 실제 설치·실행,
CUDA 학습 성공을 로컬 테스트 결과로 주장하지 않습니다.
교차 플랫폼 pip 해석은 호스트 Python 3.11 marker를 평가하는 한계가 있어 Windows 설치 성공의 근거로 사용하지 않았습니다.
Windows 실제 검증은 위 Actions 작업의 결과 및 내려받을 수 있는 로그를 기준으로 확인합니다.

이전 구현의 38개 테스트와 연구 결과는 `reproduction_report.md`에 별도 이력으로 남아 있습니다.
SynLoc 학습 데이터와 학습된 2점 체크포인트는 여전히 별도 준비 대상입니다.
