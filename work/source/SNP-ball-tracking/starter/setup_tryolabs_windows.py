"""Prepare public local assets or check the existing Windows CUDA environment.

This script never installs/replaces torch and never uploads competition assets.
"""
import argparse
import hashlib
import importlib.metadata
import json
from pathlib import Path
import platform
import struct
import subprocess
import sys
import urllib.request
import zipfile

ROOT=Path(__file__).resolve().parents[1]
ASSETS=ROOT/"data/derived/tryolabs_broadcast"
REPOS={"upstream":("https://github.com/tryolabs/soccer-video-analytics.git","b00c7c75a1d52bd5ca183521a35fc76fbd89cc0e"),
       "yolov5":("https://github.com/ultralytics/yolov5.git","d3ea0df8b9f923685ce5f2555c303b8eddbf83fd")}
WEIGHTS={"ball.pt":"a2425cd079d0ade7a4d9d4689554abc65c7d675d4abe23503939e757dabaf8e9",
         "yolov5x.pt":"9f27a794fa0308e2606f90565164571d6f0a0ba18a3ce2e5e5d323b71c157859"}


def digest(path):
    h=hashlib.sha256()
    with Path(path).open("rb") as f:
        for block in iter(lambda:f.read(1048576),b""):h.update(block)
    return h.hexdigest()


def validate_weights():
    for name,expected in WEIGHTS.items():
        path=ASSETS/name
        if not path.is_file() or digest(path)!=expected:
            raise RuntimeError(f"Missing or different public checkpoint: {path}. No existing file was overwritten.")


def prepare_assets():
    ASSETS.mkdir(parents=True,exist_ok=True)
    for name,(url,commit) in REPOS.items():
        target=ASSETS/name
        if not target.exists():
            subprocess.run(["git","clone",url,str(target)],check=True)
            subprocess.run(["git","-C",str(target),"checkout","--detach",commit],check=True)
        actual=subprocess.check_output(["git","-C",str(target),"rev-parse","HEAD"],text=True).strip()
        if actual!=commit:raise RuntimeError(f"Existing {name} checkout differs ({actual}); refusing to modify it")
    for name in WEIGHTS:
        target=ASSETS/name
        if target.exists():continue
        if name=="ball.pt":
            archive=ASSETS/"ball.pt.zip.download"
            urllib.request.urlretrieve("https://github.com/tryolabs/soccer-video-analytics/releases/download/v0/ball.pt.zip",archive)
            with zipfile.ZipFile(archive) as z:
                payload=z.read("ball.pt")
            if hashlib.sha256(payload).hexdigest()!=WEIGHTS[name]:raise RuntimeError("Downloaded ball checkpoint checksum mismatch")
            with target.open("xb") as f:f.write(payload)
            archive.unlink()
        else:
            temporary=ASSETS/(name+".download")
            urllib.request.urlretrieve("https://github.com/ultralytics/yolov5/releases/download/v6.2/yolov5x.pt",temporary)
            if digest(temporary)!=WEIGHTS[name]:raise RuntimeError("Downloaded player checkpoint checksum mismatch")
            temporary.rename(target)
    validate_weights()
    print("Public code/weights prepared under git-ignored data/. Copy the private broadcast video locally; it is not downloaded or uploaded.")


def check_environment():
    import torch
    if sys.version_info[:2]!=(3,11) or struct.calcsize("P")!=8:
        raise RuntimeError("Expected Python 3.11 x64 in the existing CUDA virtual environment")
    if torch.__version__.split("+")[0]!="2.9.1" or torch.version.cuda!="12.6":
        raise RuntimeError(f"Expected existing torch 2.9.1 + CUDA12.6, found {torch.__version__} / {torch.version.cuda}. This script never replaces torch.")
    try:
        import torchvision
    except Exception as exc:
        raise RuntimeError("torchvision import failed. Install torchvision 0.24.1 from the cu126 index using --no-deps; keep torch unchanged.") from exc
    if torchvision.__version__.split("+")[0]!="0.24.1":
        raise RuntimeError(f"Expected torchvision 0.24.1, found {torchvision.__version__}")
    if not torch.cuda.is_available():raise RuntimeError("CUDA unavailable: check NVIDIA driver and the active venv")
    # A small functional check, not a video throughput benchmark.
    value=(torch.ones(1,device="cuda")+1).cpu().item()
    nms=torchvision.ops.nms(torch.tensor([[0.,0.,2.,2.]]),torch.tensor([.9]),.45).tolist()
    assert value==2 and nms==[0]
    validate_weights()
    for name,(_,commit) in REPOS.items():
        actual=subprocess.check_output(["git","-C",str(ASSETS/name),"rev-parse","HEAD"],text=True).strip()
        if actual!=commit:raise RuntimeError(f"Pinned source mismatch: {name}")
    import cv2
    video=ROOT/"data/broadcast/BAR-ATH_broadcast-001.mp4"
    cap=cv2.VideoCapture(str(video));metadata=dict(frames=int(cap.get(7)),fps=cap.get(5),width=int(cap.get(3)),height=int(cap.get(4)))
    ok,_=cap.read();cap.release()
    if not ok or metadata!=dict(frames=164487,fps=25.,width=1920,height=1080):
        raise RuntimeError(f"Expected local 164487-frame 1920x1080/25fps broadcast: {video}; found {metadata}")
    result=dict(platform=platform.platform(),python=sys.executable,torch=torch.__version__,torchvision=torchvision.__version__,
                cuda=torch.version.cuda,gpu=torch.cuda.get_device_name(0),vram_bytes=torch.cuda.get_device_properties(0).total_memory,
                numpy=importlib.metadata.version("numpy"),video=metadata)
    print(json.dumps(result,indent=2))


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument("--prepare-assets",action="store_true");p.add_argument("--check",action="store_true")
    a=p.parse_args()
    if not (a.prepare_assets or a.check):p.error("Choose --prepare-assets or --check")
    if a.prepare_assets:prepare_assets()
    if a.check:check_environment()


if __name__=="__main__":main()
