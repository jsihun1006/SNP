직접 작성: tiling/baat.py, detection/engine.py, geometry/camera.py, pipeline.py,
평가·시각화·데이터 준비·CLI 어댑터와 tests.

외부 구현을 설치하여 호출:

- Ultralytics YOLO26 / 공식 체크포인트: https://github.com/ultralytics/ultralytics (AGPL-3.0)
- MMPose RTMCCHead, RTMCCBlock, CSPNeXt, SimCC: https://github.com/open-mmlab/mmpose/tree/v1.3.2 (Apache-2.0)
- MMEngine/MMCV: https://github.com/open-mmlab (Apache-2.0)
- SAHI: https://github.com/obss/sahi (MIT)
- Spiideo sskit e64b7d89741b7bdb7e4c6d41ed6796b38516360c: https://github.com/Spiideo/sskit (MIT)
- COCO tools: https://github.com/cocodataset/cocoapi (BSD), xtcocotools: https://github.com/jin-s13/xtcocoapi

RTMPose-X 설정은 아래 공개 설정의 구조·차원·Body-7 초기화 경로를 참고하여 2점용으로 구성했습니다.
Copyright (c) OpenMMLab. All rights reserved. Apache License 2.0:
https://github.com/open-mmlab/mmpose/blob/v1.3.2/projects/rtmpose/rtmpose/body_2d_keypoint/rtmpose-x_8xb256-700e_coco-384x288.py

논문은 알고리즘과 실험 설정의 출처이며, 저자들의 미공개 구현을 입수한 것으로 간주하지 않습니다:
https://arxiv.org/html/2609.02705v1
