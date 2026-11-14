# SSS-1 재구현 보고서

이 프로젝트는 논문 알고리즘과 공개 컴포넌트를 연결한 독립 재구현입니다.
저자의 SSS-1 학습 완료 가중치나 비공개 소스는 확보하지 않았습니다.
기존 MTAH·ATD·점유 모델은 변경하지 않았습니다.

출처를 다음 버전으로 구분했습니다.

- 논문 v1: https://arxiv.org/html/2609.02705v1
- MMPose 1.3.2: commit 5408bc76f5b848cf925a0d1857899011d8c5b497
- sskit: commit e64b7d89741b7bdb7e4c6d41ed6796b38516360c
- Spiideo 공식 baseline spiideo_scenes: commit bf1b4401a0f12b1f7d2a2e007d287e0f26ca789e
- 구현/의존성 구분: THIRD_PARTY_NOTICES.md, 파일·가중치 해시: outputs/validation/provenance.json

공식 baseline은 YOLOX-Pose 기반이며 SSS-1의 YOLO26 + top-down RTMPose-X와 다릅니다.
baseline의 Docker 설치 절차는 본 프로젝트의 필수 조건으로 가져오지 않았습니다.

논문과 구현의 차이:

| 항목 | 원문/공개 코드 확인 | 이 프로젝트 선택 |
|---|---|---|
| BAAT 경계 | 식 (1)은 현재 경계와 min/max를 명시하지 않아 문자 그대로 구현하면 축소 가능 | 기존 경계와 합집합, floor/ceil로 픽셀 포함 보존 |
| BAAT 보장 | coarse box와 겹친 타일을 반복 확장 | 검출된 box의 이미지 내부 부분에 한정, 최대 반복 후 미수렴 표시 |
| 큰 타일 | 별도 상한 미기재 | 이미지 크기까지 허용, 면적 4배 초과 경고, detector는 1280으로 resize |
| Coarse 입력 | 1280×1280, confidence .001 | 명시적으로 square resize 후 X/Y별 역스케일 |
| 격자 병합 | SAHI 기반, 세부 병합 설정 미기재 | class-aware IoU NMS .5, SAHI 패키지도 같은 설정으로 별도 비교 |
| Head | C_out=2로 GAU token 수가 2, A는 2×2 | 실제 RTMCCHead를 상속, 임의의 별도 geometric attention 추가 안 함 |
| Attention 계수 | 원문에는 1/n, upstream RTMCCBlock은 relu²(QKᵀ/√s), 1/n 없음 | upstream 기본값, 별도 `paper_attention_normalization=True` 실험 옵션 제공 |
| Attention 형태 | 원문의 K(Z)와 공개 코드의 shared base에서 Q/K 생성 표현 차이 | upstream GAU, rel_bias/pos_enc=False. 원본 저자 코드와 동일하다고 주장 안 함 |
| Pose 초기화 | Body-7 | 공개 CSPNeXt-X UDP Body-7 backbone 체크포인트 확보·SHA256 확인; 2점 head는 별도 학습 필요 |
| Backbone 구현 | 공개 설정은 mmdet CSPNeXt-X | 같은 P5/depth1.33/width1.25 구조의 MMPose 내장 CSPNeXt, BN 사용 |
| Pose 세부값 | crop size/증강 확률/weight decay/flip TTA 세부 미기재 | 공개 설정 기반 288×384, wd .05, scale .5~1.5, rotation90, dropout p=.5, flip TTA=False |
| 학습 주기 | 700epoch, epoch62 부근 수렴 보고 | 700epoch cosine 전체 구간, 자동 early stopping 없음 |
| 메모리 조정 | batch128 | batch/accumulation 선택을 별도 JSON 기록, 작은 microbatch의 BN 차이 명시 |
| 타일 학습 라벨 | clipping/retention 세부 미기재 | visible bbox area 25% 이상 유지, 빈 crop 유지, split 후 생성 |
| 좌표 | SynLoc calibration의 Z=0 | sskit 다항식 렌즈 규약, 별도 명시적 OpenCV 보정 어댑터; 가상 homography 없음 |
| 위치 오차 매칭 | 논문의 상세 오차 진단 정책 미기재 | bbox IoU≥.5 confidence 순 1:1 매칭, 공식 world AP와 분리 |
| GT annotation ID | 공식 make_coco는 0부터 시작 | 표준 pycocotools의 unmatched sentinel 충돌 방지를 위해 평가 메모리 복사본에서만 1부터 재번호 |

LocSim 부호를 공식 코드와 대조했습니다. 원문의 식 (2)는 `exp(-ln(.05)*d²/τ²)`로,
ln(.05)<0이므로 d>0일 때 1보다 커집니다. 공식 `sskit.coco.locsim`은
`exp(ln(.05)*dist2/τ²)`를 사용해 d=0이면 1, d=τ이면 .05입니다.
본 구현은 후자를 따르고 공식 evaluator를 호출합니다. 평균 pairwise LocSim과 mAP-LocSim을 구분합니다.

논문 참고 수치: A1 Detection AP50:95=.9128, Public LocSim@1m=97.44,
GT bbox 사용 LocSim=98.86. 모두 **논문 보고값**이며 우리 실측값이 아닙니다.
Challenge 수치는 Introduction의 97.67과 Section4.3/Table1의 96.67이 불일치합니다.
원문을 수정하거나 한 값을 우리 재현 결과로 선택하지 않았습니다.

Phase별 변경·검증·제약:

| Phase | 변경 파일/결과 | 완료 범위와 제약 |
|---|---|---|
| 1 환경/원본 분석 | requirements, configs/*constraints*, setup_windows.ps1, probe_environment.py, 본 보고서 | 호스트/배포 파일 조사 완료. 현재 macOS이므로 Windows 실기 검증은 미완료 |
| 2 Detection/BAAT | detection/, tiling/, tests/test_detection.py·test_tiling.py, compare_detection.py | 단위 테스트 및 공식 YOLO26 실제 CPU 추론·공식 SAHI 비교 실행. SynLoc fine-tuning 미실행 |
| 3 Pose | pose/, configs/rtmpose_x_two_points.py, prepare_data/train_pose/infer_pose/validate_pose_model | 실제 Head/GAU/SimCC·gradient 검증, Body-7 backbone 566/566 shape 일치, 전체 forward·증강 실행. Head 미학습, 실제 SynLoc 학습 미실행 |
| 4 Geometry | geometry/, test_geometry.py, raycast.py | 합성 보정의 왕복·왜곡·edge/far·invalid 검증 및 실제 sskit 함수 수치 대조. 실제 SynLoc calibration 데이터 검증은 미실행 |
| 5 Integration | pipeline.py, infer.py, test_pipeline.py, CSV/JSON/시각화 | 계약 fixture 통합, 실제 YOLO 이미지·2프레임 영상 저장 실행. 학습된 2점 pose를 연결한 실제 통합 추론은 미실행 |
| 6 Reproduction | evaluation/, evaluate.py, ablation.py, configs/experiments.json | 공식 evaluator를 합성 GT로 검증. 실제 SynLoc GT·훈련된 2점 가중치가 없어 C/B1/B2/A1/A2 정량 실험 미실행 |

Phase 3의 실제 모델/학습 검증은 다른 독립 모듈의 검증과 구분합니다.
데이터·GPU가 필요한 완료 조건은 충족되지 않았으며, 가능한 독립 구현과 테스트를 계속한 상태입니다.

실제 pretrained detection 관찰 (GT 없음, 학습 없음):

| 입력 | Full | 직접 SAHI 격자 | BAAT | 공식 SAHI |
|---|---:|---:|---:|---:|
| 공개 bus.jpg 810×1080 | 4 | 4 | 4 | 4 |
| bus 이미지 2배 확대 + 3840×2160 캔버스 | 4 | 10 | 4 | 10 |

이는 검출 개수이며 정확도·Recall·LocSim이 아닙니다. 두 번째 입력은 실제 4K 중계 프레임이 아닙니다.
SAHI의 잘린 신체 조각이 IoU NMS 이후에도 남을 수 있음을 관찰했고, 검출 개수가 많다고 정확도가 높다고 해석하지 않았습니다.
동일 머신의 배경 작업과 warmup 영향이 있으므로 시간은 저장한 timing JSON의 smoke 실행 관찰치로만 보십시오.
모델 초기화 시간은 제외하고 첫 프레임 warmup을 포함합니다.

검증 산출물:

- outputs/validation/cpu-core.xml: CPU 핵심 단위 테스트 JUnit
- outputs/validation/cpu-full.xml 및 tests.log: 실제 MMPose 연산을 포함한 최종 전체 테스트
- outputs/validation/pose_model.json: 전체 모델/Body-7/증강 검증, Head 미학습 명시
- outputs/validation/pretrained_bus/comparison.jpg 및 comparison.json
- outputs/validation/pretrained_4k/comparison.jpg 및 comparison.json
- outputs/validation/cli_image/: 실제 이미지 CSV/results.json/timing.json/visualizations
- outputs/validation/cli_video/: 실제 2프레임 독립 추론 CSV/results.json/timing.json/visualizations
- outputs/validation/synthetic_tiling_comparison.png: 합성 경계 확장 도식
- outputs/validation/mmcv-build*.log: 실제 빌드 실패/수정 기록

최종 실행 결과: **38 tests passed**, 실패·skip 없음 (`cpu-full.xml`, `tests.log`).
실제 MMPose 2점 Head의 2×2 Attention·SimCC KL loss backward, 17점 체크포인트 거부 검증을 포함합니다.
`validate_pose_model`에서 Body-7 초기화와 전체 forward, 증강·타깃 생성, 미학습 Head의
실제 top-down decode API(shape 1×2×2)를 실행했습니다. **미학습 좌표는 성능 평가에 사용하지 않았습니다**.
`pip check`도 의존성 오류 없이 완료했습니다.
경고는 Matplotlib/pyparsing·pkg_resources의 deprecation과 공식 sskit의 빈 예측 F1 분모 0 경고입니다.
빈 예측의 AP=0은 검증했으며, 정의되지 않은 값은 CLI JSON에서 null로 저장합니다.

초기 작업(2026-11-13)의 미실행 항목은 Windows 실기 실행, CUDA/GPU 학습, SynLoc train/val/test 실데이터 검증,
학습 완료 2점 체크포인트를 쓴 통합 위치 추정, 논문 ablation 정량 재현입니다.
미구현 항목과 실행 불가능한 검증 항목을 서로 구분한 상태입니다.

2026-11-14 후속 배포에서는 로컬 전체 테스트 45개 및 실제 Windows CPU 테스트 42개,
깨끗한 sss1 설치와 실제 YOLO26 BAAT 샘플 추론을 통과했습니다.
근거: [Windows CI](https://github.com/jsihun1006/SNP/actions/runs/43567391206), `WINDOWS_RELEASE_REPORT.md`.
CUDA/GPU 학습과 SynLoc 실데이터 성능 재현은 여전히 미실행입니다.

미구현 확장 기능: tracking, team classification, GSR 연결, 동적 프레임 보정,
장면 전환 탐지, ATD 정렬, H/A/D 점유 판정. `interfaces.py`의 계약만 제공합니다.
SynLoc의 고정 보정 전제를 카메라가 움직이는 일반 중계영상으로 확장하려면 별도 구현이 필요합니다.
