# Windows용 독립 중계 트래킹

기존 MTAH 코드·ATD·점유 모델을 수정하지 않는 별도 실행기다. MP4에서 YOLOX 검출과 OSNet 특징을 계산하고, 공식 Deep-EIoU 추적기를 Windows에서 사용할 수 있는 연관 계산 어댑터로 실행한다. 선택적으로 CLIP 속성 인식, SFR Homography 추정, IDATR에서 착안한 후처리를 실행한다.

새 Windows 컴퓨터에서 준비할 영상·가중치·설치 파일은 [REQUIRED_FILES.md](REQUIRED_FILES.md)에 정리했다.

**현재 검증 범위:** macOS의 Python 3.11 + 고정된 기본 패키지에서 합성 영상과 실제 원본 Deep-EIoU를 포함한 테스트 15개를 검증했다. Windows `.cmd` 실행, 공식 가중치 추론, 실제 경기의 정확도·처리 속도는 아직 검증하지 않았다. `smoke`는 합성 검출 입력을 사용하며 신경망 추론 성공을 뜻하지 않는다.

## 1. 설치

Windows 10/11 x64, Python 3.11 x64, Git for Windows를 준비하고 저장소 루트에서 실행한다.

```bat
REM CPU 환경
broadcast_gsr\setup_windows.cmd cpu

REM NVIDIA CUDA 12.8용 PyTorch 환경을 선택하는 경우
REM broadcast_gsr\setup_windows.cmd cu128
```

환경은 `broadcast_gsr\.venv`에만 설치한다. 기존 Conda 환경은 사용하지 않는다. Python launcher의 `py -3.11`로 인터프리터를 선택하므로 Python 3.11 설치가 필요하다. CUDA 선택 시 호환 드라이버가 필요하며, 드라이버는 설치 스크립트가 변경하지 않는다. PyTorch 버전과 인덱스 조합은 [공식 설치 표](https://pytorch.org/get-started/previous-versions/)의 `torch 2.9.1 / torchvision 0.24.1`을 사용한다.

설치기는 공식 소스 두 개를 `vendor/`에 고정 커밋으로 받는다. 이미 존재하는 다른 버전이나 수정된 저장소를 강제로 덮어쓰지 않는다. 원본 `setup.py develop`, `cython_bbox`, `lap`, `unsloth` 설치는 이 실행 경로에 필요하지 않다. `requirements-models.txt`는 CLIP/SFR 선택 기능의 Python 의존성도 설치한다.

공식 모델 가중치는 자동으로 받지 않는다. [저자 공개 체크포인트 폴더](https://drive.google.com/drive/folders/1kgZGxUYGkYhM9AwHzjuSVo7r7FFZ16mh)에서 필요한 파일을 내려받아 다음 위치에 넣는다. 다른 모델의 이름을 바꿔 대신 사용하지 않는다.

```text
broadcast_gsr/weights/
  yolox_soccernet.pth.tar                 필수: 검출
  sports_model.pth.tar-60                 필수: ReID
  CLIP_Jersey.pth                        선택: --jersey clip
  SoccernetGSR_EfficientNet_Best.pth       선택: calibrate
```

```bat
broadcast_gsr\run_windows.cmd doctor
```

`doctor`는 패키지 import, 소스 커밋, 입력 경로, 체크포인트 존재와 영상 메타데이터를 검사한다. 신경망 정확도를 검사하는 명령이 아니다. CLIP을 쓰려면 설정의 `jersey`를 `clip`으로 바꾸거나 실행 시 `--jersey clip`을 지정한다.

## 2. 가중치 없이 설치와 데이터 흐름 확인

```bat
broadcast_gsr\run_windows.cmd smoke --out data\derived\broadcast_gsr\smoke_001
```

24프레임 합성 영상, 합성 검출·특징, 알려진 H로 원본 추적기·좌표 변환·후처리·CSV·오버레이 저장을 점검한다. 같은 출력 폴더가 존재하면 중단한다. 다시 실행할 때는 `smoke_002`처럼 새 폴더를 지정한다.

테스트 전체 실행:

```bat
broadcast_gsr\.venv\Scripts\python.exe -m unittest discover -s broadcast_gsr\tests -v
```

원본 소스가 설치되지 않았다면 원본 연결 테스트는 `skipped`로 표시된다. 성공 개수와 건너뛴 개수를 함께 확인한다.

## 3. 짧은 실제 영상 실행

설정은 [configs/mtah_broadcast.json](configs/mtah_broadcast.json)이다. 기본 영상은 로컬 배포 파일명인 `BAR-ATH_broadcast-001.mp4`를 사용한다. 데이터가 다른 디스크에 있으면 JSON의 경로를 절대 경로로 바꾸거나 아래처럼 지정한다. JSON에서 Windows 경로는 `D:/MTAH/...`처럼 `/`를 쓰면 간단하다.

```bat
broadcast_gsr\run_windows.cmd run --video "D:\MTAH\data\broadcast\BAR-ATH_broadcast-001.mp4" --calibration-csv "D:\MTAH\data\broadcast\BAR-ATH_broadcast_homography.csv" --start-frame 10000 --max-frames 750 --jersey clip --device cuda --out "D:\MTAH\data\derived\broadcast_gsr\pilot_001"
```

`10000`은 실행 예시 프레임 번호다. 라이브 장면이라는 검증값이 아니므로 실제 영상에서 연속 라이브 구간을 확인한 뒤 지정한다. GPU 없이 시도하려면 `--device cpu`를 사용한다. 영상 처리량은 측정 전 단정하지 않는다. 기본 `jersey=none`은 검출·추적만 실행하며, `team_ha`, `jersey`, 속성 신뢰도는 빈값으로 남는다.

원본과 달리 MP4를 직접 읽으며, JPEG 전체 경기 추출은 하지 않는다. 프레임은 원영상의 0-based 번호를 유지한다. `--max-frames`로 구간을 제한한다. 후처리를 끈 실행은 프레임별로 저장하므로 긴 구간도 시도할 수 있지만 전체 경기 실행은 검증하지 않았다. 첫 실행은 100~750프레임에서 환경·메모리·속도를 확인한다.

제공된 H를 사용하는 이 실행은 **선수 검출·추적을 ATD 좌표와 독립적으로 수행**한다. 카메라 보정까지 독립적인 실험은 아래 `calibrate`로 나눠 실행한다. 제공된 H와 ATD의 생성상 독립성은 확인되지 않았다.

## 4. 영상에서 Homography도 새로 추정

SFR 가중치가 필요하다. 추정 실패 프레임을 정상 좌표로 처리하지 않는다.

```bat
broadcast_gsr\run_windows.cmd calibrate --start-frame 10000 --max-frames 100 --device cuda --out data\derived\broadcast_gsr\calibration_001

broadcast_gsr\run_windows.cmd run --start-frame 10000 --max-frames 100 --jersey clip --device cuda --calibration-csv data\derived\broadcast_gsr\calibration_001\calibration.csv --calibration-origin learned_sfr --out data\derived\broadcast_gsr\independent_001
```

`calibrate`는 선택한 구간만 자기 출력 폴더의 `img1/`에 추출한다. `frame_map.csv`가 로컬 JPEG 번호와 원영상 번호를 연결한다. 원본 SFR 모델과 최적화 코드를 사용하되 빈 keypoint 배열과 LM의 제약식 개수 문제를 실행 중 어댑터에서 보정한다. **원본 파일은 수정하지 않으며 원 논문과 동일한 재현 결과라고 주장하지 않는다.** 원본의 identity fallback은 `valid=0`으로 내보낸다. 새 H의 정확도는 필드 라인 오버레이와 독립 표본으로 확인해야 한다.

## 5. 후처리와 팀 매핑

```bat
broadcast_gsr\run_windows.cmd run --start-frame 10000 --max-frames 750 --jersey clip --refine --out data\derived\broadcast_gsr\refined_001
```

`--refine`는 IDATR에서 착안한 별도 변형이다. 시간 간격과 속성 불일치로 분할하고, 같은 카메라 구간 내에서 외형·시간 비중첩·실제 미터 좌표의 이동 가능성을 확인해 병합한다. 원본의 전역 화면 범위를 공간 임계값으로 사용하는 방식과 다르다. 미확정 속성은 판단을 보류한다. 원본 관측·score를 보존하고 `tracks_refined.csv`를 별도로 저장한다. 이 파일이 기본 `tracks.csv`를 덮어쓰지 않는다. 계산 범위는 한 실행당 3,000프레임 이하이며, 병합의 정확도는 미검증이다.

`team_raw`는 `color:blue` 같은 색상 관측 라벨이다. 홈/어웨이가 아니다. 실제 유니폼을 확인한 경우에만 설정의 `team_colors`에 색상→`H`/`A`를 명시한다. `team_confidence`는 검증된 확률이 없어 빈값이다. 기본 `team_colors={}`는 H/A를 확정하지 않는다. 좌/우 위치로 H/A를 자동 부여하지 않는다.

`track_id`는 카메라 구간 내 추적 ID다. 등번호나 로스터의 선수 ID와 같지 않다. 카메라 컷마다 ID namespace를 바꾸며, 서로 다른 컷·카메라의 ID 연결은 구현하지 않았다. CLIP의 `jersey`도 오인식 가능성이 있는 별도 관측값이다.

## 6. 결과

| 파일 | 내용 |
|---|---|
| `manifest.json` | 실행 상태, 소스 커밋, 체크포인트 해시, 입력 서명, 처리 프레임·시간, 검증 범위 |
| `config.json` | 실제 사용한 절대 경로와 옵션 |
| `tracks.csv` | 관측 bbox, 추적 ID, 픽셀·정규화·미터 좌표, 역할·색상·등번호, 품질 flag |
| `frames.csv` | 검출·관측 수, H 유효성, 컷 후보를 모든 처리 프레임에 기록 |
| `overlay.mp4` | bbox/ID와 미니맵. 원본 `tracks.csv` 기준 |
| `tracks_refined.csv`, `refinement.json` | `--refine` 사용 시 별도 후처리 결과와 ID 계보 |
| `features.float32`, `features.json` | `--refine`에 필요한 관측 특징과 shape/dtype |

H 무효·투영 실패는 pitch 좌표를 빈칸으로 둔다. 화면 밖 선수나 누락 프레임의 좌표를 만들지 않는다. `calibration_quality`는 입력 H의 reprojection error이며 값이 작을수록 좋고, 확률이 아니다. `color_similarity`도 확률이 아니다. `detection_score`만 YOLOX objectness×class score를 보존한다. 후처리가 score를 1로 덮어쓰지 않는다.

미터 좌표는 코너 원점의 `x_norm×105.3, y_norm×68.0`이다. x/y 반전이나 하프 방향 교체를 자동 적용하지 않는다. 제공 H와 MP4의 정렬이 별도로 확인되지 않았으면 `video_calibration_alignment_unverified`를 남긴다. 직접 추출한 JPEG→H의 frame map을 사용하는 `--calibration-origin learned_sfr`는 이 제공 H 경고와 구분한다. 장면 유형 기본값은 `unverified`이고 히스토그램 컷 후보는 리플레이 판별기가 아니다. 사람이 확인한 구간은 JSON의 `scene_type="live"`로 기록할 수 있지만 검증 근거는 별도로 보존해야 한다.

처리 도중 실패하면 `manifest.status="failed"`로 남기고 오류를 반환한다. 기존 출력 폴더의 재사용은 거부한다. `manifest.status="complete"`는 요청 구간의 처리 종료를 뜻하며 정확도·Windows 이식성·성능 검증을 뜻하지 않는다.

## 7. 기존 ATD와 비교

```bat
broadcast_gsr\run_windows.cmd compare-atd --tracks data\derived\broadcast_gsr\pilot_001\tracks.csv --atd data\broadcast\BAR-ATH_broadcast_atd.txt --out data\derived\broadcast_gsr\comparison_001
```

같은 broadcast 소스의 원영상 번호와 ATD `frame_id-1`을 가정한 **익명 위치 집합 차이 진단**이다. ATD를 정답으로 보거나, 같은 슬롯을 같은 선수로 보지 않는다. 시간 대응이 따로 검증되었다면 `--alignment-verified`를 명시할 수 있다. 실제 오프셋이 확인된 경우만 `--atd-offset`을 사용한다. 이 옵션은 `ATD frame0 = video frame0 + offset`이다.

거리 gate 이내 일대일 대응의 중앙값/P95와 미매칭 수를 함께 저장한다. gate 밖 대형 오류가 평균에서 사라지므로 미매칭 수를 반드시 같이 본다. 리플레이·클로즈업으로 표시된 행은 제외하며, `unverified` 장면은 진단으로만 취급한다. 수작업 검증 표본과의 위치 오차·IDF1/HOTA는 이 명령이 계산하지 않는다.

현재 로컬 데이터에서 broadcast MP4 메타데이터는 164,487프레임, ATD/H는 164,488행이었다. 원인은 확인하지 않았다. 끝 프레임 삭제나 전체 +1 이동으로 자동 보정하지 않는다. tactical↔broadcast 동기화와 H/A/D·spellF1 평가는 별도 단계다.

## 구현 범위

| 항목 | 상태 |
|---|---|
| 원본 Deep-EIoU 연결·누락 프레임 갱신·검출 인덱스 보존 | 합성 테스트 통과 |
| 좌표계·H 유효성·입출력·오버레이·후처리의 관측 보존 | 합성 테스트 통과 |
| YOLOX / OSNet / CLIP 공식 가중치 로딩·추론 경로 | 코드 작성, 실제 가중치 실행 미검증 |
| SFR 모델과 보정된 최적화 연결 | 코드 작성, 실제 영상 추론 미검증; 합성 최적화 검증 |
| Windows 설치 스크립트 / CUDA | 코드 작성, Windows 실행 미검증 |
| LLaMA/Unsloth 네이티브 Windows, 컷 사이 글로벌 선수 ID | 미구현 |
| 공의 독립 추적, 점유 H/A/D, 기존 모델 대체 | 미구현 |

원본 분석과 개발·평가 계획은 [감사 보고서](../docs/broadcast2pitch_audit_20261113/REPORT.md)에 기록했다. 외부 코드와 가중치는 새 코드의 라이선스로 묶어 재배포하지 않는다. 원본 저장소 루트의 포괄 라이선스는 확인되지 않았고 `sn-trackeval/`의 MIT는 그 하위 코드에 대해서만 확인했다.
