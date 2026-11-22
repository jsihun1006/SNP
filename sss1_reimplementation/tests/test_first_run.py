import hashlib
import pytest
from ..scripts.download_weights import download_verified
from ..scripts.first_run import validate_environment


def report():
    return dict(packages={name:dict(import_ok=True) for name in
                         ["numpy","torch","torchvision","ultralytics","cv2","sahi","sskit","xtcocotools"]},
                torchvision_nms_cpu="passed",cuda=dict(gpu_training_allowed=False),mmpose_model_import="missing")


def test_cpu_first_run_does_not_require_gpu_or_pose():
    validate_environment(report(),"CoreCPU")


def test_cuda_first_run_refuses_unavailable_gpu():
    with pytest.raises(RuntimeError,match="CUDA convolution"):
        validate_environment(report(),"CUDA118")


def test_first_run_refuses_broken_binary_pair():
    state=report();state["torchvision_nms_cpu"]="operator missing"
    with pytest.raises(RuntimeError,match="NMS"):
        validate_environment(state,"CoreCPU")


def test_asset_download_checks_digest_and_retains_existing_file(tmp_path):
    source=tmp_path/"source.pt";source.write_bytes(b"test checkpoint bytes, not a model")
    expected=hashlib.sha256(source.read_bytes()).hexdigest()
    target=tmp_path/"weights/target.pt"
    download_verified(source.as_uri(),target,expected)
    assert target.read_bytes()==source.read_bytes()
    target.write_bytes(b"corrupt")
    with pytest.raises(ValueError,match="Checksum mismatch"):
        download_verified(source.as_uri(),target,expected)
    assert target.read_bytes()==b"corrupt"


def test_failed_download_does_not_publish_partial_weights(tmp_path):
    source=tmp_path/"source.pt";source.write_bytes(b"corrupt transfer")
    target=tmp_path/"weights/target.pt"
    with pytest.raises(ValueError,match="checksum mismatch"):
        download_verified(source.as_uri(),target,"0"*64)
    assert not target.exists() and not target.with_suffix(".pt.partial").exists()
