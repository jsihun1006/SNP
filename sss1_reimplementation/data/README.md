기존 MTAH/ATD는 사용하거나 변경하지 않습니다. SynLoc은 공식 안내에 따라 별도로 확보하십시오:
https://github.com/Spiideo/sskit

권장 배치: `annotations/{train,val,test}.json`, `{train,val,test}/<file_name>`.
공식 SynLoc COCO의 두 keypoint 순서는 pelvis, ground입니다. 중첩된 `[2,3]` 배열을
`prepare_data --task pose`로 COCO의 평탄한 6개 값으로 변환합니다.
이미지 엔트리의 camera_matrix/dist_poly/undist_poly, annotation의 position_on_pitch를 보존합니다.
훈련/검증 arena split을 그대로 유지하고, 타일 생성은 split을 나눈 후에 실행하십시오.
임의 영상을 SynLoc GT로 변환하거나 bbox 바닥점을 ground GT로 대체하지 않습니다.

검증용 `bus.jpg`는 공식 Ultralytics 공개 샘플이며 축구/SynLoc 데이터가 아닙니다.
`bus_4k_upscaled.jpg`는 해당 이미지를 2배 확대해 3840×2160 캔버스에 배치한 입력 크기 테스트입니다.
실제 4K 축구 촬영 데이터라고 주장하지 않습니다.
