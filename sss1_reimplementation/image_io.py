"""OpenCV codecs with Python file I/O for Unicode Windows user/folder names."""
from pathlib import Path
import cv2
import numpy as np


def read_image(path):
    try:
        encoded=np.fromfile(Path(path),dtype=np.uint8)
    except OSError:
        return None
    return cv2.imdecode(encoded,cv2.IMREAD_COLOR) if encoded.size else None


def write_image(path,image):
    path=Path(path)
    ok,encoded=cv2.imencode(path.suffix,image)
    if not ok: return False
    try:
        encoded.tofile(path)
        return True
    except OSError:
        return False
