# BAR-ATH 공 후보 검출: PC1 전반

`SNP-ball-tracking.zip`의 `ball_only/README_KO.md`와 기본 PC1 실행 스크립트를 사용했습니다. 소스는 ZIP 사본이며 Git 원격 저장소 복제가 완료된 상태는 아닙니다.

## 확인한 환경

- Python 3.11.9, PyTorch 2.9.1+cu128, torchvision 0.24.1+cu128
- Ultralytics 8.4.212, NumPy 1.26.4, OpenCV 4.10.0.84
- NVIDIA GeForce RTX 5080, CUDA 연산 확인, `pip check` 통과
- 공식 YOLO26x 모델, SHA256: `9fdd44a31c504547ffb81d2c6d9e6dac3493c8eaa8b0398d3f43bae6c7003e92`
- 영상: `H:\data\broadcast\BAR-ATH_broadcast.mp4`
- 영상 SHA256: `3dc2a7d952302d33df69a26e3d8935648015f8d7e1b86bf614436b7009abf9f8` (기본 구간 설정과 일치)
- 영상 형식: 1920×1080, 25fps, 164,487프레임

## 실행 범위와 저장 위치

- PC1 전반: 원본 재생 시간 `00:04:46.84` 이상 `00:57:45.52` 미만
- 원본 프레임 `[7171, 86638)`, 총 79,467프레임
- 결과 폴더: `H:\data\ball-tracking-results\first_half`
- 기본 설정 유지: FP32, 전체 입력 1536, 확대 입력 960, 원본 타일 640, 20% 겹침, 배치 1/4/8, 청크 32, 확대 영역 최대 2
- 기본 GPU 평균 사용률 목표 98~99.5%, 기존 온도 제어 유지

2프레임 시험에서 원본 프레임 7171, 7172의 공 후보 출력 및 완료 파일 생성을 확인했습니다. 이 짧은 시험 시간으로 전반 완료 시간을 추정하지 않습니다. 현재 진행률과 예상 시간은 실제 실행의 `progress.json`을 확인하세요.

## 상태 확인, 중단, 재개

이 폴더에서 PowerShell로 실행합니다.

```powershell
.\check_pc1.ps1
.\stop_pc1.ps1
# 중단이 완료된 것을 확인한 후 재개:
.\run_pc1.ps1 -Resume
```

`stop_pc1.ps1`은 현재 청크가 저장된 뒤 중단하도록 `STOP` 파일을 만듭니다. 재개 스크립트는 `STOP`을 제거하고 같은 입력·설정·출력 경로를 사용합니다. 실행 중인 작업을 중복 실행하지 마세요.

백그라운드 실행 로그는 `pc1.stdout.log`, `pc1.stderr.log`입니다. 결과 폴더의 `progress.json`은 청크마다, `progress_5min.jsonl`은 5분마다 갱신됩니다. 완료 시 `job_complete.json`과 통합된 `candidates.jsonl`이 생깁니다. 중단·재개에 사용하는 `chunks`는 보존하세요.

이 출력은 공 후보의 이미지 좌표입니다. 최종 경기 공 트래킹이나 경기장 좌표의 정확도를 검증한 결과는 아닙니다. 두 PC 결과를 병합할 때는 원본 영상과 모델 파일의 SHA256이 서로 같아야 합니다.
