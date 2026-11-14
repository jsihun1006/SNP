학습 모델 정의는 `configs/rtmpose_x_two_points.py`, 두 점 Head는 `pose/head.py`에 있습니다.
YOLO26은 공식 Ultralytics 구현을 호출하며 원본 소스를 복사하지 않습니다.
체크포인트는 `checkpoints/`에 별도로 저장합니다. 임의 가중치를 학습 완료 모델로 취급하지 않습니다.
