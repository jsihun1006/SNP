# 공 검출 코드만 두 컴퓨터로 나눠 실행하기

전달·커밋 단위는 이 `ball_only/` 폴더다. `broadcast_gsr/ball_pilot.py`부터 복사하면 기존 선수 결과·H·캐시 경로에 다시 의존하므로 이 패키지를 사용한다.

이 패키지는 영상에서 **공 후보의 이미지 좌표**를 생성한다. 선수 검출, 팀 분류, ATD, homography, 기존 분석 캐시는 필요하지 않다. 실제 경기 공을 고른 최종 트래킹이나 보정된 ATD를 출력한다고 보장하지 않는다. 현재 모델은 마이크·유니폼 로고를 공으로 오검출할 수 있으며, 고해상도·확대·분할만으로 100% 검출이 보장되지는 않는다.

## 전달할 것

두 PC에 같은 코드, **동일한 전체 영상 파일**, **동일한 `yolo26x.pt` 가중치**를 준비한다. BAR-ATH 제공 영상은 포함된 정렬 구간 설정을 사용하며, 다른 영상은 실제 시작·끝 구간을 지정한다.

- Git에 포함: `ball_only/`의 Python 소스, 설치·실행 스크립트, 요구사항, 문서, 테스트.
- 별도 복사: 원본 전체 MP4, `yolo26x.pt`(현재 파일 약 119MB).
- Git에서 제외: 영상, 가중치, 결과, 가상환경, 캐시, 임시파일. 패키지의 `.gitignore`에 반영했다.

기본 경로 외에 결과나 모델을 저장할 경우 해당 경로도 Git 커밋 대상에서 제외해야 한다.

가중치는 기존 PC의 `H:\SNP-GSR\.runtime\cache\yolo26_weights\yolo26x.pt`를 별도 복사하면 된다. 새 폴더의 `ball_only\weights\yolo26x.pt`에 놓아도 되며 이 폴더는 Git에서 제외된다. 이 경로는 기존 PC에서 찾기 위한 안내일 뿐, 실행 코드에 H 드라이브가 고정돼 있지 않다.

## 각 PC 설치

검증 환경은 Windows, Python 3.11, NVIDIA CUDA GPU, PyTorch 2.9.1+cu128, torchvision 0.24.1+cu128이다. PC2의 GPU 종류·VRAM·드라이버에 따라 속도와 가능한 배치 크기는 달라진다. CPU로 자동 전환하지 않는다.

Python 3.11과 NVIDIA 드라이버가 먼저 설치돼 있어야 한다. Python이 없는 PC에서는 설치 프로그램의 설치 위치를 원하는 드라이브로 지정한다. 이 스크립트는 Python을 C 드라이브에 자동 설치하지 않는다.

프로젝트를 공간이 있는 드라이브에 둔 뒤 **프로젝트 루트(`ball_only`의 상위 폴더)**에서 실행한다.

```powershell
powershell -NoProfile -ExecutionPolicy Bypass -File .\ball_only\setup_windows.ps1
```

Python을 자동으로 못 찾으면 실제 설치 경로를 지정한다.

```powershell
powershell -NoProfile -ExecutionPolicy Bypass -File .\ball_only\setup_windows.ps1 -PythonExe "H:\Tools\Python311\python.exe"
```

가상환경은 `ball_only\.venv`, pip·모델 설정 캐시는 `ball_only\.cache`, 임시파일은 `ball_only\.tmp`에 생성된다. 설치 스크립트는 CUDA PyTorch를 공식 PyTorch CUDA 12.8 인덱스에서 설치한 다음 `requirements.txt`를 설치하고 CUDA 사용 가능 여부를 검사한다.

## 전반·후반 구간 정하기

시간은 **경기장 시계가 아닌 원본 MP4의 재생 시간**이다. 포함된 `jobs.bar_ath.json`은 `main`의 tactical↔broadcast 정렬 코드를 원본 데이터에 실행해 얻은 broadcast 구간이다.

| 작업 | 시작 프레임 포함 | 종료 프레임 제외 | 원본 영상 재생 시간 |
| --- | ---: | ---: | --- |
| PC1 전반 | 7,171 | 86,638 | 00:04:46.84 ~ 00:57:45.52 |
| PC2 후반 | 86,823 | 163,279 | 00:57:52.92 ~ 01:48:51.16 |

기준 코드는 `main`의 `42023fe`에 있는 `model/영상추출/모델/tracking_alignment/final_tracking_alignment.py`이다. 공 궤적·선수 배치 기준점과 이후 구간 확인을 거쳐 반별 일정 시차를 적용한 결과이며, **반 전체의 동기화나 MP4의 모든 컷·리플레이 대응을 검증한 것은 아니다.** `jobs.bar_ath.json`은 이 제한과 원본 영상 SHA256을 기록한다. 실행기는 영상 해시가 다르면 이 설정으로 추론을 시작하지 않는다.

정렬에 사용한 ATD와 프레임별 라벨 CSV는 Git에 포함하지 않는다. 공 검출 실행 시에는 그 원본 ATD나 tactical 파일도 필요 없다. 설정의 두 구간만 재사용한다.

영상 길이의 절반을 하프타임으로 추정하지 않는다. 다른 영상이나 수정된 편집본에는 포함된 BAR-ATH 설정을 적용하지 않고 실제 broadcast 경계를 다시 확인한다.

### 경기 시작 시각을 아직 모르는 경우

공 검출 자체는 방송 영상의 첫 프레임부터 가능하다. `--start 0 --end EOF`로 전체 영상을 처리할 수 있으며, 이때는 도입부·광고·하프타임도 포함된다.
현재 검출기가 영상에서 킥오프나 전·후반 시작을 실시간 자동 판별하는 것은 아니다. BAR-ATH는 `main` 정렬 코드로 미리 산출한 구간 설정을 읽는다. 다른 영상의 전·후반은 정렬 코드를 실행하거나 원본 재생 시간에서 경계를 확인해야 한다.
기존 tactical 영상의 시작 프레임이나 경기 시계를 broadcast의 시작·끝 시각으로 그대로 사용하지 않는다.

```powershell
.\ball_only\.venv\Scripts\python.exe -m ball_only.run --video "PATH_TO_BROADCAST.mp4" --weights "PATH_TO_yolo26x.pt" --out ".\ball_only\outputs\whole_video" --start 0 --end EOF --part whole_video
```

- 입력 형식: `HH:MM:SS` 또는 `HH:MM:SS.mmm`.
- 시작 프레임 포함, 종료 프레임 제외: `[start, end)`.
- 경계는 원본 FPS의 프레임 간격에 맞춰야 한다. 25fps에서는 0.04초 간격이다.
- 두 구간은 겹치지 않게 입력한다. 양쪽 결과는 원본 전체 영상의 0부터 시작하는 프레임 번호를 유지한다.
- `jobs.example.json`은 다른 영상의 시간을 기록하는 빈 템플릿이다. `null` 경계를 채운 뒤 `-RangesFile`로 지정한다. 값이 비어 있으면 실행하지 않는다.
- `-Start`와 `-End`를 둘 다 생략하면 `jobs.bar_ath.json`에서 해당 PC의 구간을 읽는다. 사용자 설정은 `-RangesFile`로 바꿀 수 있다.
- 직접 구간을 지정하려면 `-Start`와 `-End`를 모두 준다. 이 경우에는 `-RangesFile`을 함께 주지 않는다.

## PC1: 전반

BAR-ATH 제공 영상은 다음 명령으로 포함된 전반 구간을 사용한다. 영상·가중치 경로는 각 PC의 실제 위치로 바꾼다.

```powershell
powershell -NoProfile -ExecutionPolicy Bypass -File .\ball_only\run_pc1.ps1 -Video "H:\data\broadcast\BAR-ATH_broadcast.mp4" -Weights "H:\models\yolo26x.pt"
```

기본 출력은 `ball_only\outputs\first_half`이다. `-Output`으로 바꿀 수 있다.

## PC2: 후반

```powershell
powershell -NoProfile -ExecutionPolicy Bypass -File .\ball_only\run_pc2.ps1 -Video "H:\data\broadcast\BAR-ATH_broadcast.mp4" -Weights "H:\models\yolo26x.pt"
```

기본 출력은 `ball_only\outputs\second_half`이다. 코드의 영상·가중치 경로는 고정값이 아니므로 PC마다 다른 경로를 사용해도 된다. 파일 내용과 검출 설정은 같아야 한다.

다른 구간이나 다른 영상은 직접 지정할 수 있다.

```powershell
$firstStart = Read-Host '원본 영상에서 전반 시작 HH:MM:SS'
$firstEnd = Read-Host '원본 영상에서 전반 끝 HH:MM:SS'
powershell -NoProfile -ExecutionPolicy Bypass -File .\ball_only\run_pc1.ps1 -Video "PATH_TO_BROADCAST.mp4" -Weights "PATH_TO_yolo26x.pt" -Start $firstStart -End $firstEnd
```

정렬 라벨에서 새 설정을 재생성할 때는 `ranges_from_alignment.py`를 사용한다. 이 명령은 로컬 라벨을 읽고 두 구간 메타데이터만 내보내며, 라벨이나 원본 추적 좌표를 업로드하지 않는다.

```powershell
.\ball_only\.venv\Scripts\python.exe -m ball_only.ranges_from_alignment --labels "PATH_TO_final_tracking_broadcast_labels.csv" --fps 25 --video-frames 164487 --out ".\ball_only\jobs.local.json"
powershell -NoProfile -ExecutionPolicy Bypass -File .\ball_only\run_pc1.ps1 -Video "PATH_TO_BROADCAST.mp4" -Weights "PATH_TO_yolo26x.pt" -RangesFile ".\ball_only\jobs.local.json"
```

## 수행하는 탐지

- 전체 입력 1536의 YOLO26x FP32 공 후보 탐지.
- 전체 화면 후보 주변의 원본 640픽셀 영역을 입력 960으로 확대 탐지.
- 원본 640픽셀 타일, 20% 겹침으로 전체 화면 분할 탐지.
- 같은 입력 크기를 묶어 처리: 기본 전체 배치 1, 확대 배치 4, 타일 배치 8.

기본 확대 영역은 프레임당 최대 2개이다. `-MaxRois 0`으로 모든 전체 화면 후보 주변을 확대할 수 있으나 후보 수에 따라 시간이 증가한다. 전체 화면 후보가 없으면 중앙 확대 영역을 사용하며, 전체 화면 분할 탐지는 별도로 전체 픽셀 범위를 다룬다. 확대·분할 결과는 같은 YOLO 모델 계열의 관측이며 독립 모델의 투표로 간주하지 않는다.

메모리가 부족하면 `-RoiBatch 1 -TileBatch 2`처럼 **배치 크기만** 명시적으로 줄일 수 있다. 해상도·FP32 정밀도·탐지 모드를 자동으로 낮추지 않는다. 설정을 바꾸고 재실행할 때는 기존 결과와 섞이지 않도록 새로운 출력 폴더를 쓴다.

실행 스크립트는 GPU 평균 사용률 목표를 98~99.5%로 요청한다. 순간 사용률 고정값은 보장하지 않으며 온도 제어는 유지한다. 각 PC 처리 시간은 실제 GPU와 확대 영역 수, 영상 읽기 속도로 달라진다. 기존 짧은 배치 벤치마크의 전체 영상 시간 환산은 이 새 실행기의 전체 완료 실측이 아니다.

## 중단·재개·합치기

`progress.json`에 구간 진행률과 남은 시간을 저장하고, 콘솔과 `progress_5min.jsonl`에 5분 간격으로 진행 상황을 기록한다.
주요 출력 `candidates.jsonl`의 `source_frame`은 전체 영상 프레임 번호, `part_frame`은 해당 PC 구간 안의 프레임 번호이다.
첫 실행에서는 두 PC가 같은 영상을 사용하는지 확인하기 위해 영상과 가중치 SHA256을 계산하며, 이 준비 시간은 영상 저장장치 속도에 따라 달라진다.

중단하려면 해당 출력 폴더에 `STOP`이라는 이름의 파일을 만든다. 저장된 체크포인트 이후의 일부 프레임은 재개할 때 다시 처리할 수 있다. 재개할 때는 `STOP` 파일을 제거하고 **같은 입력·구간·설정·출력 경로**로 같은 명령을 실행한다. 저장된 결과를 다른 설정이나 영상과 섞지 않는다.

두 작업이 완료되면 결과 폴더 둘을 한 PC로 모아 다음 명령으로 합친다. 가중치·영상 내용과 설정 식별자, 구간 겹침을 확인하며 원본 프레임 번호를 유지한다. 전반과 후반 사이 하프타임 공백은 실제 처리 누락과 구분해 확인해야 한다.

병합할 때는 각 PC의 `run_config.json`, `job_complete.json`, `candidates.jsonl` 세 파일만 모아도 된다. `chunks` 폴더는 해당 PC에서 중단·재개할 때 사용하는 기록이다.

```powershell
.\ball_only\.venv\Scripts\python.exe -m ball_only.merge --first ".\ball_only\outputs\first_half" --second ".\ball_only\outputs\second_half" --out ".\ball_only\outputs\merged"
```

이 결과는 후속 공 후보 검증·트래킹의 입력이다. 경기장 좌표가 필요하면 나중에 원본 프레임 번호로 해당 homography와 연결한다.
