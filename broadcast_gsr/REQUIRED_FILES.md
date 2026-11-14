# GSR 브랜치: Windows 실행에 필요한 파일

이 문서는 `KKJ`에서 분기한 `GSR` 브랜치를 새 Windows 컴퓨터에서 실행할 때 준비할 파일을 정리한다. 코드와 합성 테스트를 포함하며, 경기 영상·ATD·Homography 원본·모델 가중치는 Git에 포함하지 않는다.

## Git에서 받는 파일

저장소 루트에서 브랜치를 내려받는다. 기존 작업 폴더에 수정 사항이 있으면 별도 clone을 사용한다.

```bat
git clone --branch GSR https://github.com/jsihun1006/SNP.git SNP-GSR
cd SNP-GSR
```

| 경로 | 역할 |
|---|---|
| `broadcast_gsr/setup_windows.cmd` | 독립 `.venv`에 Python 패키지 설치, 고정 버전 외부 소스 다운로드 |
| `broadcast_gsr/run_windows.cmd` | `doctor`, `smoke`, `run`, `calibrate`, `compare-atd` 실행 |
| `broadcast_gsr/requirements-core.txt`, `requirements-models.txt` | 기본·모델 의존성 목록 |
| `broadcast_gsr/configs/mtah_broadcast.json` | 영상·가중치·Homography·출력 경로와 추론 옵션 |
| `broadcast_gsr/*.py` | 모델·추적·좌표 변환·후처리·비교 어댑터 |
| `broadcast_gsr/tests/test_core.py` | 합성 기하·원본 추적기·출력 보존 테스트 |
| `broadcast_gsr/README.md` | 실행 예시, 출력 스키마, 구현·검증 범위 |
| `docs/broadcast2pitch_audit_20261113/` | 공식 코드 함수 분석, 논문과의 차이, 감사 증거·평가 계획 |

`broadcast_gsr/` 코드는 기존 `starter/`, 점유 모델, 평가 코드를 import하지 않는다. 외부 소스·패키지는 아래 설치 명령으로 준비한다.

## 컴퓨터에 먼저 설치할 것

- Windows 10/11 x64
- Python 3.11 x64와 Python launcher (`py -3.11` 명령 사용)
- Git for Windows
- NVIDIA GPU를 쓸 경우 선택한 PyTorch CUDA 빌드와 호환되는 드라이버

```bat
REM CPU
broadcast_gsr\setup_windows.cmd cpu

REM NVIDIA CUDA 12.8 빌드를 선택할 경우 위 명령 대신 실행
REM broadcast_gsr\setup_windows.cmd cu128
```

스크립트는 기존 Conda 환경 대신 `broadcast_gsr/.venv/`를 사용한다. `broadcast_gsr/vendor/`에는 아래 공개 소스를 받는다. 모델 가중치는 자동 다운로드하지 않는다.

| 다운로드 대상 | 고정 커밋 |
|---|---|
| `yinmayoo185/SoccernetGSR` | `dbfb65c05c847e8e50baa47aa91ace960cc5e600` |
| `KaiyangZhou/deep-person-reid` | `f8cd150fdf77e8d9e1ed143b7f308c2c609ded50` |

## 별도로 준비할 경기 파일

기본 설정의 경로는 다음과 같다. 대소문자·파일명을 확인하고, 실제 위치가 다르면 `configs/mtah_broadcast.json`을 변경한다. Windows JSON 경로에는 `D:/MTAH/...`처럼 `/`를 사용할 수 있다.

| 파일 | 기본 위치 | 필요한 기능 |
|---|---|---|
| 중계 MP4 | `data/broadcast/BAR-ATH_broadcast-001.mp4` | 실제 영상 추론·새 H 추정 |
| 제공 Homography | `data/broadcast/BAR-ATH_broadcast_homography.csv` | 제공 H로 선수 픽셀을 경기장 좌표로 변환 |
| broadcast ATD TXT | `data/broadcast/BAR-ATH_broadcast_atd.txt` | `compare-atd`로 기존 ATD와 위치 집합 차이 진단할 때만 |

ATD TXT/XML과 tactical 영상은 `run`의 선수 검출·추적 입력이 아니다. 제공 H 대신 `calibrate`로 새 H를 만들면 `run --calibration-csv <생성한 calibration.csv> --calibration-origin learned_sfr`로 지정한다. 이때 SFR 가중치가 추가로 필요하다.

영상과 Homography·ATD의 프레임 대응은 따로 확인해야 한다. 감사 당시 로컬 broadcast MP4 메타데이터는 164,487프레임, 제공 H/ATD는 164,488행이었다. 이를 근거로 임의의 1프레임 이동이나 행 삭제를 적용하지 않는다.

## 별도로 준비할 공식 모델 가중치

[저자가 공개한 체크포인트 폴더](https://drive.google.com/drive/folders/1kgZGxUYGkYhM9AwHzjuSVo7r7FFZ16mh)에서 필요한 파일을 받아 `broadcast_gsr/weights/`에 넣는다.

| 파일명 | 필수 여부 | 용도 |
|---|---|---|
| `yolox_soccernet.pth.tar` | 실제 `run`에 필수 | YOLOX 검출 |
| `sports_model.pth.tar-60` | 실제 `run`에 필수 | OSNet ReID |
| `CLIP_Jersey.pth` | `--jersey clip`에 필요 | 역할·등번호·색상 인식 |
| `SoccernetGSR_EfficientNet_Best.pth` | `calibrate`에 필요 | 영상 기반 새 Homography 추정 |

기본 `jersey=none`으로 YOLOX·OSNet 가중치 두 개만 사용하면 팀·등번호는 미확정으로 남는다. 인식과 좌표 추정이 함께 필요하면 CLIP을 켜고 필요한 H를 준비한다. LLaMA 가중치는 현재 Windows 실행 경로에서 사용하지 않는다. 다른 pretrained OSNet 파일을 SportsMOT 가중치 대신 이름만 바꿔 사용하지 않는다.

## 파일 배치 예시

```text
SNP-GSR/
  broadcast_gsr/
    configs/mtah_broadcast.json
    weights/
      yolox_soccernet.pth.tar
      sports_model.pth.tar-60
      CLIP_Jersey.pth
      SoccernetGSR_EfficientNet_Best.pth
    vendor/                         설치 스크립트가 생성
    .venv/                          설치 스크립트가 생성
  data/
    broadcast/
      BAR-ATH_broadcast-001.mp4
      BAR-ATH_broadcast_homography.csv
      BAR-ATH_broadcast_atd.txt       비교할 때만 필요
    derived/broadcast_gsr/            실행 결과의 기본 위치
```

가중치·외부 clone·가상환경·경기 데이터·실행 결과는 Git에서 제외한다. 생성 결과에는 CSV, MP4, manifest와 선택적인 특징 파일이 포함된다. 출력 폴더가 이미 있으면 덮어쓰지 않으므로 실행마다 새 이름을 지정한다.

## 준비 확인과 첫 실행

가중치 없이 설치·입출력을 점검하려면 합성 검증을 사용한다.

```bat
broadcast_gsr\run_windows.cmd smoke --out data\derived\broadcast_gsr\smoke_001
broadcast_gsr\.venv\Scripts\python.exe -m unittest discover -s broadcast_gsr\tests -v
```

실제 영상·가중치를 배치한 다음 실행한다.

```bat
broadcast_gsr\run_windows.cmd doctor
broadcast_gsr\run_windows.cmd run --start-frame 10000 --max-frames 100 --device cpu --out data\derived\broadcast_gsr\pilot_001
```

`10000`은 명령 예시이며 검증된 라이브 장면 번호가 아니다. 실제 연속 라이브 구간을 확인하고 지정한다. GPU·CLIP을 사용할 경우 `--device cuda --jersey clip`을 지정하고 해당 가중치를 준비한다. `doctor`가 실패하면 누락 파일·패키지·경로를 먼저 해결한다.

검증 기록은 macOS에서 합성 테스트 15개 통과다. Windows 설치·CUDA·공식 가중치 실추론·MTAH 정확도는 미검증이다. `smoke` 성공이나 실행 완료를 모델 정확도 검증으로 해석하지 않는다. 이 코드는 점유 H/A/D 예측이나 독립 ball tracking을 제공하지 않는다.
