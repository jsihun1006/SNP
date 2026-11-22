# 환경 조사 — 2026-11-13 (Asia/Seoul)

작업 루트: `/Users/kkj_1120/performance_analyst/projects/metrica_hack/more-than-a-hack-2027`
독립 프로젝트: 위 루트의 `sss1_reimplementation`.
Windows 환경에 접속된 상태가 아니라 **macOS 26.6.2 ARM64**에서 작업했습니다.
Windows GPU 모델·VRAM·드라이버는 이 호스트에서 조사할 수 없으므로 값을 만들어 넣지 않았습니다.

| 항목 | 실제 조사 결과 |
|---|---|
| 기본 python3 | Apple Python 3.9.6, torch/numpy/cv2/pytest 없음 |
| Conda | `/opt/anaconda3/bin/conda`, 26.5.3 |
| 별도 sss1 | 프로젝트 내부 `sss1/`, Python 3.11.17 venv, 기존 Conda env 변경 없음 |
| NVIDIA / VRAM | nvidia-smi 없음, NVIDIA GPU 확인 안 됨 |
| CUDA Toolkit | nvcc 없음 |
| PyTorch CUDA | 사용 불가, GPU 학습 차단 |
| 실제 CPU PyTorch 쌍 | torch 2.2.2 / torchvision 0.17.2 |
| YOLO26-Large | ultralytics 8.4.211, 공식 yolo26l.pt CPU 추론 실행 |
| SAHI | 0.11.22, 공식 get_sliced_prediction 실행 |
| OpenCV / NumPy / SciPy | 4.10.0.84 / 1.26.4 / 1.13.1 |
| MMEngine / MMPose | 0.10.7 / 1.3.2 |
| 공식 SynLoc devkit | sskit 0.6.0, commit e64b7d89741b7bdb7e4c6d41ed6796b38516360c |

현재 설치 상태·CUDA 실행 검사는 `outputs/validation/environment.json`을 기준으로 합니다.
macOS 검증 의존성은 `configs/constraints-macos-validation.txt`, 설치 전체 버전은
`outputs/validation/pip-freeze-macos.txt`에 기록합니다. Windows의 lock file로 사용하면 안 됩니다.

Windows 제안 조합은 Python 3.10 x64, torch 2.1.0+cu118, torchvision 0.16.0+cu118,
mmcv 2.1.0, mmengine 0.10.7, mmpose 1.3.2, mmdet 3.2.0입니다.
PyTorch/torchvision/xtcocotools/pycocotools의 cp310 win_amd64 배포 파일과
OpenMMLab의 `cu118/torch2.1` index에 있는
`mmcv-2.1.0-cp310-cp310-win_amd64.whl`을 확인했습니다.
증거: `outputs/validation/windows_wheel_metadata.json`, `provenance.json`.
이는 배포 파일·버전 조건 확인이며 **Windows에서 import/학습을 실행했다는 뜻이 아닙니다**.
실제 GPU 종류·드라이버에 따라 이 조합이 지원되지 않을 수 있으며, CUDA convolution/backward 검사 실패 시 학습은 중단합니다.
공식 설치 문서: https://mmpose.readthedocs.io/en/latest/installation.html

발생 오류와 처리:

1. chumpy 0.70의 격리 빌드가 `No module named pip`로 실패했습니다.
   별도 환경에서 setuptools/wheel을 고정하고 이 패키지만 `--no-build-isolation`으로 설치했습니다.
2. SAHI 0.11.36 설치가 torch>=2.4.1 요구로 torch/torchvision을 업그레이드했습니다.
   해당 환경은 새 sss1 안에만 있었고, SAHI 0.11.22와 원래 CPU 검증 쌍으로 복원했습니다.
   Windows 설치에는 constraints를 적용해 같은 문제를 방지합니다.
3. mmcv-lite는 기본 import는 되지만 MMPose 모델 import에 필요한 `mmcv._ext`가 없습니다.
   이를 전체 MMPose 검증 성공으로 취급하지 않았습니다.
4. macOS mmcv 소스 빌드에서 Apple Clang 21과 torch 2.2.2의 `is_arithmetic` 특수화 진단이 충돌했습니다.
   로컬 빌드에만 `-Wno-invalid-specialization`을 적용해 컴파일 단계는 통과했습니다.
5. 다음 링크 단계에서 MacOSX27 SDK의 `arm64e.x1`을 링커가 인식하지 못했습니다.
   기존에 설치된 MacOSX15.4 SDK를 해당 빌드에만 지정했습니다. 이 조치는 Windows 설치 경로에 포함하지 않습니다.
   이 설정으로 mmcv 2.1.0 확장 빌드·설치에 성공했습니다 (`mmcv-build-sdk-compat.log`).
6. MMEngine의 설정 실행 컨텍스트에는 `__file__`이 없어 처음 모델 smoke가 실패했습니다.
   설치된 프로젝트 패키지 위치를 pathlib로 찾아 경로를 계산하도록 수정한 뒤 모델 forward·증강을 실행했습니다.

최종 CPU Pose 검증: 실제 RTMCCHead/GAU 2×2 Attention, SimCC 출력과 gradient,
Body-7 backbone 566개 텐서 shape 일치, 전체 CSPNeXt-X + 2점 Head forward,
flip/scale/rotation/CoarseDropout·SimCC 학습 파이프라인 실행.
Head는 미학습이므로 선수 위치 정확도나 학습된 모델 추론 성공을 의미하지 않습니다.
증거는 `outputs/validation/pose_model.json`, `cpu-full.xml`입니다.
최종 전체 CPU 테스트는 38개 통과, `pip check`는 의존성 오류 없음입니다.
최종 probe에서 torchvision CPU NMS와 MMPose 전체 모델 import가 모두 통과했습니다.

원본 ATD·모델·다른 Python 환경은 변경하지 않았습니다. 프로젝트의 가상환경·가중치·샘플 데이터는 Git에서 제외합니다.

2026-11-14 후속 검증: [Windows Server 2025 x64의 깨끗한 runner](https://github.com/jsihun1006/SNP/actions/runs/43567391206)에서
Python 3.10.11, torch 2.1.0+cpu, torchvision 0.16.0+cpu 설치와 `pip check`,
CPU 테스트 42개, YOLO26-Large 실제 BAAT 추론이 통과했습니다.
위 macOS 조사 및 초기 Windows 미실행 표기는 2026-11-13의 기록입니다.
현재 Windows CPU 검증 범위와 다운로드 가능한 증거는 `WINDOWS_RELEASE_REPORT.md`를 기준으로 합니다.
