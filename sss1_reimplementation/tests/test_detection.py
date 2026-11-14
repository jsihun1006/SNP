import numpy as np
import pytest
from ..types import Detection
from ..detection import DetectionEngine, nms


def test_duplicate_objects_class_aware_nms():
    a=Detection((10,10,30,50),.9)
    assert nms([a,Detection((11,11,31,51),.8)]) == [a]
    assert len(nms([a,Detection((11,11,31,51),.8,1)])) == 2


def test_coarse_restoration_for_4k():
    class Detector:
        def __init__(self): self.calls=[]
        def predict(self, image, confidence):
            self.calls.append((image.shape,confidence))
            return [Detection((100,100,200,200),.1)] if confidence==.001 else []
    detector=Detector()
    engine=DetectionEngine(detector, mode="baat")
    assert engine.predict(np.zeros((2160,3840,3),np.uint8)) == []
    assert engine.last_coarse[0].bbox == (300.,168.75,600.,337.5)
    assert detector.calls[0] == ((1280,1280,3),.001)


def test_tile_offset_restoration_and_duplicate_removal():
    class Detector:
        def predict(self, image, confidence):
            # Original global X is encoded in the test image for deterministic tile offsets.
            offset=int(image[0,0,0])
            return [Detection((85-offset,10,95-offset,40),.9)] if offset<=85 else []
    image=np.zeros((60,180,3),np.uint8)
    image[:,:,0]=np.arange(180,dtype=np.uint8)[None,:]
    result=DetectionEngine(Detector(),mode="sahi",size=100,overlap=.5).predict(image)
    assert len(result)==1 and result[0].bbox==(85,10,95,40)


@pytest.mark.parametrize("box,score", [((0,0,0,10),.5),((0,0,10,10),2),((0,0,float('nan'),10),.1)])
def test_invalid_detection(box,score):
    with pytest.raises(ValueError): Detection(box,score)
