# SSS-1 독립 재구현 — Windows Native

**Windows에서 처음 실행한다면 [WINDOWS_START.md](WINDOWS_START.md)를 먼저 보십시오.**
`WINDOWS_CPU.cmd`가 Python 준비, 별도 환경 설치, CPU 테스트, 가중치 다운로드와 샘플 추론을 실행합니다.

기존 MTAH·ATD·모델을 변경하지 않는 별도 프로젝트입니다. 논문 알고리즘을 구현하고,
공식 YOLO26-Large 사전학습 추론, BAAT·기하·공식 평가 어댑터를 검증합니다.
현재 호스트는 **macOS ARM64**입니다. Windows/GPU 학습 성공이나 논문 성능 재현을 주장하지 않습니다.
실행 증거와 제약은 [environment_report.md](environment_report.md),
[reproduction_report.md](reproduction_report.md)에 구분했습니다.

아래 명령은 모두 **저장소 루트에서 PowerShell**로 실행합니다. 경로에 공백이 있어도 동작합니다.

```powershell
$Repo = (Get-Location).Path
$Project = Join-Path $Repo "sss1_reimplementation"
# Git 없이 브랜치 ZIP으로도 설치할 수 있습니다. 기존 Python 환경은 변경하지 않습니다.
powershell.exe -NoProfile -ExecutionPolicy Bypass -File "$Project\scripts\setup_windows.ps1" -Profile CoreCPU -InstallPython -Verify
$Python = Join-Path $Project "sss1\Scripts\python.exe"
& $Python -m sss1_reimplementation.scripts.probe_environment
& $Python -m pytest sss1_reimplementation/tests -m "not model" -q
```

NVIDIA GPU에서 전체 MMPose를 사용할 때는 같은 별도 환경에 다음 프로필을 설치합니다.
Windows Python 3.10 / torch 2.1.0+cu118 / torchvision 0.16.0 / mmcv 2.1.0
공식 휠 배포 조합을 지정했습니다. 실제 GPU·드라이버 호환 여부는 probe의 CUDA 연산으로 확인합니다.
GPU가 이 구형 CUDA 조합에서 지원되지 않으면 학습을 차단하고 다른 조합을 별도 검증해야 합니다.
`nvidia-smi`의 CUDA 표시는 드라이버의 지원 상한이며 설치된 PyTorch 런타임 버전과 다릅니다.

```powershell
& "$Project\scripts\setup_windows.ps1" -Profile CUDA118
& $Python -m sss1_reimplementation.scripts.probe_environment
& $Python -m pytest sss1_reimplementation/tests -q
& $Python -m sss1_reimplementation.scripts.download_weights
& $Python -m sss1_reimplementation.scripts.validate_pose_model --checkpoint "$Project\checkpoints\cspnext-x_udp-body7_210e-384x288-d28b58e6_20230529.pth" --output "$Project\outputs\pose_model_smoke.json"
```

CoreCPU 프로필은 MMCV 확장 연산을 설치하지 않으므로 `model` 표시가 붙은 MMPose 테스트는 제외합니다.
CUDA 프로필에서도 `--device cpu` 추론·모델 단위 테스트는 가능합니다.
MMCV 소스 빌드로 자동 전환하지 않으며 설치 실패 원인을 그대로 표시합니다.
`mmcv-lite`는 전체 MMPose import에서 필요한 `mmcv._ext`를 제공하지 않습니다.
`requirements.txt`와 `configs/constraints-windows.txt`를 함께 사용하여 의도치 않은 torch 업그레이드를 방지합니다.
Conda를 선호하면 `configs/environment-windows.yml`의 Python 3.10 사양으로 별도 sss1 환경을 만들 수 있습니다.
Windows에서 직접 실행한 환경 보고서는 아직 없습니다.

모듈은 다음처럼 분리했습니다.

| 단계 | 파일 | 역할 |
|---|---|---|
| A | detection/yolo.py | 공식 YOLO26-Large, 원본 픽셀 bbox·confidence |
| B | tiling/baat.py, detection/engine.py | SAHI 격자, Coarse 0.001, BAAT 반복 확장, 좌표 복원·NMS |
| C | pose/, configs/rtmpose_x_two_points.py | CSPNeXt-X, 2점 RTMCCHead·SimCC·GAU, 학습·추론 |
| D | geometry/camera.py | SynLoc 다항식 렌즈 / 명시적 OpenCV 카메라, 전방 광선–Z=0 교점 |
| E | evaluation/, visualization/ | 공식 COCO/sskit 평가, 별도 오차 진단, 비교 이미지 |

BAAT는 1280 타일, overlap 0.2, margin 5, 최대 100회입니다. 현재 타일 경계와 합집합을 취해
확장 중 축소되지 않게 했습니다. 큰 객체는 이미지 전체까지 확장할 수 있으며 조용히 잘라내지 않습니다.
포함 보장은 **coarse에서 검출된 박스의 이미지 내부 부분**에만 해당하고, 반복 제한 시 미수렴을 알립니다.
`sahi` 모드는 같은 병합 설정을 쓰는 직접 작성한 규칙 격자 기준선입니다.
`compare_detection --official-sahi`로 실제 SAHI 패키지 호출과 함께 비교할 수 있습니다.
IoU NMS는 심하게 잘린 신체 조각까지 모두 중복 제거한다고 보장하지 않습니다.

사전학습 YOLO 추론은 2점 체크포인트 없이 실행할 수 있습니다.

```powershell
& $Python -m sss1_reimplementation.scripts.infer --input "C:\datasets\frame.jpg" --detector "$Project\checkpoints\yolo26l.pt" --mode baat --device cpu --output "$Project\outputs\detection"
& $Python -m sss1_reimplementation.scripts.compare_detection --image "C:\datasets\frame.jpg" --checkpoint "$Project\checkpoints\yolo26l.pt" --device cpu --official-sahi --output "$Project\outputs\comparison"
& $Python -m sss1_reimplementation.scripts.demo_geometry --output "$Project\outputs\synthetic"
```

SynLoc 데이터는 [공식 devkit](https://github.com/Spiideo/sskit) 안내에 따라 별도로 확보합니다.
제공된 arena split을 유지하고 train/val을 나눈 뒤 타일을 만드십시오. 기존 ATD를 학습 GT로 사용하지 않습니다.
공식 COCO 이미지 파일명은 각 split 디렉터리 기준입니다. 2점 순서는 pelvis, ground입니다.
원본 주석은 보존하고 새로운 파일로 변환합니다.

```powershell
$SynLoc = "C:\datasets\SpiideoSynLoc"
New-Item -ItemType Directory -Force "$Project\data\annotations" | Out-Null
& $Python -m sss1_reimplementation.scripts.prepare_data --task pose --annotations "$SynLoc\annotations\train.json" --output "$Project\data\annotations\train.json"
& $Python -m sss1_reimplementation.scripts.prepare_data --task pose --annotations "$SynLoc\annotations\val.json" --output "$Project\data\annotations\val.json"
# 모델 학습에는 --data-root 하위 train/와 val/ 이미지도 필요합니다.
# 이미지를 복사하거나 data-root를 별도 준비한 데이터 디렉터리로 지정하십시오.
Copy-Item -Recurse "$SynLoc\train" "$Project\data\train"
Copy-Item -Recurse "$SynLoc\val" "$Project\data\val"
foreach ($Split in @("train", "val")) {
  & $Python -m sss1_reimplementation.scripts.prepare_data --task detection --mode sahi --annotations "$SynLoc\annotations\$Split.json" --images "$SynLoc\$Split" --split $Split --output "$Project\data\detection_sahi"
  & $Python -m sss1_reimplementation.scripts.prepare_data --task detection --mode full --annotations "$SynLoc\annotations\$Split.json" --images "$SynLoc\$Split" --split $Split --output "$Project\data\detection_full"
}
```

원본 `make_coco.py`의 중첩된 2×3 keypoint를 평탄화하고 `num_keypoints`·메타정보를 보완합니다.
다른 17/25개 관절 데이터에서 발목이나 bbox 바닥점을 ground GT로 대신 만들지 않습니다.
타일 학습 시 bbox clipping, visible area 25% 이상 유지, 빈 타일 유지 정책은 **재구현 선택**입니다.
검증용 detection YAML의 val 이미지가 train/와 분리되었는지 확인하십시오.

Detection 학습과 Pose 학습은 CUDA 연산 검사를 통과해야 시작합니다.

```powershell
& $Python -m sss1_reimplementation.scripts.train_detection --data "$Project\data\detection_sahi\dataset.yaml" --checkpoint "$Project\checkpoints\yolo26l.pt" --output "$Project\outputs\train_detector_sahi" --device cuda:0
& $Python -m sss1_reimplementation.scripts.train_detection --data "$Project\data\detection_full\dataset.yaml" --checkpoint "$Project\checkpoints\yolo26l.pt" --output "$Project\outputs\train_detector_full" --device cuda:0
& $Python -m sss1_reimplementation.scripts.train_pose --checkpoint "$Project\checkpoints\cspnext-x_udp-body7_210e-384x288-d28b58e6_20230529.pth" --data-root "$Project\data" --batch-size 128 --accumulate 1 --output "$Project\outputs\train_pose" --device cuda:0
```

Pose: AdamW 1e-4, cosine 최소 5e-6, 700 epoch, flip/scale/rotation/CoarseDropout, HalfBody 제외.
입력 288×384, BN, crop padding·증강 범위·dropout 확률과 PCK 진단 설정은 공개 RTMPose 설정을 바탕으로 한 선택입니다.
Body-7은 backbone 초기화에 사용하며 2점 head는 새로 학습합니다. `--checkpoint` 파일이 없으면 중단합니다.
메모리가 부족하면 `--batch-size 16 --accumulate 8`로 유효 batch 128을 유지할 수 있지만
BN 통계는 batch 128과 같지 않습니다. 실제 변경은 `training_choices.json`에 기록합니다.
다른 초기화는 `--initialization-kind alternative --initialization-note "실험 설명"`으로 명시해야 합니다.
62 epoch 보고를 early stopping으로 사용하지 않았습니다.

2점 체크포인트가 실제로 학습된 후 독립 Pose·ray casting·전체 파이프라인을 실행합니다.
Body-7/COCO 17점 체크포인트를 2점 학습 완료 파일로 넣으면 거부합니다.

```powershell
$Pose = "$Project\outputs\train_pose\epoch_700.pth"
& $Python -m sss1_reimplementation.scripts.infer_pose --image "C:\datasets\frame.jpg" --boxes "C:\datasets\boxes_xyxy.json" --checkpoint $Pose --device cpu --output "$Project\outputs\pose.json"
& $Python -m sss1_reimplementation.scripts.raycast --camera "C:\datasets\camera.json" --u 1200 --v 700
& $Python -m sss1_reimplementation.scripts.infer --input "$SynLoc\val" --coco "$SynLoc\annotations\val.json" --detector "$Project\outputs\train_detector_sahi\yolo26l\weights\best.pt" --pose-checkpoint $Pose --mode baat --device cuda:0 --output "$Project\outputs\synloc_val"
& $Python -m sss1_reimplementation.scripts.evaluate --gt "$SynLoc\annotations\val.json" --predictions "$Project\outputs\synloc_val\results.json" --score-threshold 0.25 --output "$Project\outputs\synloc_val\metrics.json"
```

`boxes_xyxy.json`은 `[{"bbox":[100,100,140,220],"confidence":0.9}]` 형식입니다.
원본 GT bbox를 사용한 upper-bound 실험은 이 API로 수행할 수 있지만 아직 실측하지 않았습니다.
`--coco`는 평가용 image_id를 정확히 유지하고, image별 SynLoc calibration을 자동 사용합니다.
단일 이미지 평가 시 `--image-id`를 GT와 맞추십시오.
보정 JSON은 `model="synloc"`과 width/height/camera_matrix/dist_poly/undist_poly를 넣습니다.
raw SynLoc 카메라 디렉터리의 lens.json·camera_matrix.npy는 공식 `load_camera`로 읽습니다.
외부 보정 JSON은 `model="opencv"`, width/height/K/R/t/distortion, 선택적 pitch_bounds를 사용합니다.
R/t는 world→camera, 단위 metre, Z_world=0이어야 합니다. 좌표계 원점/축은 보정 데이터가 정의합니다.
SynLoc의 다항식을 OpenCV 왜곡 계수로 사용하면 안 됩니다.
경기장 범위가 명시된 경우에만 outside_pitch를 판정합니다. 임의의 105×68 범위를 가정하지 않습니다.

영상은 프레임별 독립 추론입니다. 제공한 카메라는 모든 처리 프레임에 적용됩니다.
카메라 이동·컷이 있는 중계영상의 프레임별 보정은 미구현이므로 고정 보정을 적용한 영상은
카메라가 실제 고정된 구간이어야 합니다. 보정 없이 실행하면 영상 좌표까지만 저장합니다.

```powershell
& $Python -m sss1_reimplementation.scripts.infer --input "C:\datasets\fixed_camera.mp4" --detector "$Project\checkpoints\yolo26l.pt" --pose-checkpoint $Pose --camera "C:\datasets\camera.json" --frame-step 5 --mode baat --device cuda:0 --output "$Project\outputs\video"
```

CSV 필수 열: image_id, frame, detection_id, bbox_x1/y1/x2/y2, confidence, pelvis_u/v,
ground_u/v, world_x/y, world_valid. 실패 원인은 world_reason에 저장합니다.
**detection_id는 매 프레임 다시 시작하며 player_id가 아닙니다.**
결측은 CSV 빈 필드/JSON null입니다. 체크포인트가 없으면 keypoint를 임의 생성하지 않습니다.
world_valid=False 검출도 bbox AP에는 포함하고 world 평가에서는 누락 검출로 남습니다.

평가는 bbox AP50/AP75/AP50:95·small/medium/large AP·COCO AR, 공식 sskit mAP-LocSim
(τ=1m·5m)·precision/recall/F1/frame accuracy를 저장합니다. AP는 [0,1], -1은 없는 크기 구간입니다.
공식 LocSim과 별도로 bbox IoU≥0.5 confidence 순 1:1 매칭으로 Euclidean MAE·median·RMSE,
축별 MAE, matched mean LocSim, GT 전체 분모의 1m 성공률을 계산합니다.
오차 통계는 매칭되고 유효한 world만의 조건부 통계이며 누락률과 함께 보십시오.
`--detection-only`로 world GT 없이 bbox AP만 실행할 수 있습니다. GT가 없으면 점수 생성은 거부합니다.
임계값은 test set 최적화 없이 validation에서 고정해야 하며 기본값 0.25는 재구현 선택입니다.

```powershell
& $Python -m sss1_reimplementation.scripts.ablation --images "$SynLoc\test" --coco "$SynLoc\annotations\test.json" --full-checkpoint "$Project\outputs\train_detector_full\yolo26l\weights\best.pt" --sahi-checkpoint "$Project\outputs\train_detector_sahi\yolo26l\weights\best.pt" --pose-checkpoint $Pose --device cuda:0 --output "$Project\outputs\ablation"
```

`configs/experiments.json`은 C/B1/B2/A1/A2의 학습 도메인과 추론 방식을 분리합니다.
같은 pose·이미지·임계값을 사용하고 입력 파일 SHA256을 manifest로 남깁니다.
학습 provenance는 checkpoint 이름만으로 자동 증명할 수 없으므로 해당 학습 설정도 보존해야 합니다.
결과는 각 실험 폴더 metrics.json/results.json/timing.json/비교 이미지로 저장합니다.
추적·팀 분류·GSR·동적 보정·컷 탐지·ATD 정렬·H/A/D 판단은 `interfaces.py`의 인터페이스만 존재합니다.
