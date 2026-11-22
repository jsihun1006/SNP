import numpy as np
from ..image_io import read_image,write_image


def test_image_roundtrip_in_korean_folder_with_spaces(tmp_path):
    directory=tmp_path/"윈도우 사용자 폴더"
    directory.mkdir()
    path=directory/"선수 이미지.png"
    image=np.arange(90,dtype=np.uint8).reshape(5,6,3)
    assert write_image(path,image)
    assert np.array_equal(read_image(path),image)


def test_missing_and_empty_images_are_reported(tmp_path):
    assert read_image(tmp_path/"missing.jpg") is None
    path=tmp_path/"empty.jpg";path.write_bytes(b"")
    assert read_image(path) is None
