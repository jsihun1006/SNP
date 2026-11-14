import argparse
import hashlib
import json
from pathlib import Path
import urllib.request

ASSETS={
    "yolo26l.pt":"https://github.com/ultralytics/assets/releases/download/v8.4.0/yolo26l.pt",
    "cspnext-x_udp-body7_210e-384x288-d28b58e6_20230529.pth":
        "https://download.openmmlab.com/mmpose/v1/projects/rtmposev1/cspnext-x_udp-body7_210e-384x288-d28b58e6_20230529.pth",
}
SHA256={
    "yolo26l.pt":"9fe3c544f2b19bebad7ea41e76d7ad3d88b7c2f10d11d24430c5311f6b32db26",
    "cspnext-x_udp-body7_210e-384x288-d28b58e6_20230529.pth":
        "d28b58e620a6b1da09427c1c0e21ec419d9a71602379eebcee48ca32bd6f0667",
}
SAMPLE_URL="https://raw.githubusercontent.com/ultralytics/assets/main/im/bus.jpg"


def download_verified(url, path, expected_sha256):
    path=Path(path)
    path.parent.mkdir(parents=True,exist_ok=True)
    if path.exists():
        if hashlib.sha256(path.read_bytes()).hexdigest()!=expected_sha256:
            raise ValueError(f"Checksum mismatch: {path}. Move the existing file aside and retry.")
        return path
    temporary=path.with_suffix(path.suffix+".partial")
    try:
        urllib.request.urlretrieve(url,temporary)
        if hashlib.sha256(temporary.read_bytes()).hexdigest()!=expected_sha256:
            raise ValueError(f"Downloaded file checksum mismatch: {path.name}")
        temporary.replace(path)
    finally:
        temporary.unlink(missing_ok=True)
    return path


def main():
    p=argparse.ArgumentParser()
    p.add_argument("--output",type=Path,default=Path(__file__).resolve().parents[1]/"checkpoints")
    p.add_argument("--only",choices=["all","detection","body7"],default="all")
    p.add_argument("--sample",action="store_true",help="Download the public, non-soccer smoke-test image")
    a=p.parse_args();a.output.mkdir(parents=True,exist_ok=True)
    records=[]
    for name,url in ASSETS.items():
        if a.only=="detection" and not name.endswith(".pt"): continue
        if a.only=="body7" and name.endswith(".pt"): continue
        path=a.output/name
        download_verified(url,path,SHA256[name])
        records.append(dict(file=name,url=url,bytes=path.stat().st_size,
                            sha256=hashlib.sha256(path.read_bytes()).hexdigest(),
                            role="detection-pretrained" if name.endswith(".pt") else "Body-7 backbone initialization ONLY"))
    (a.output/"download_manifest.json").write_text(json.dumps(records,indent=2),encoding="utf-8")
    if a.sample:
        sample_path=Path(__file__).resolve().parents[1]/"data/bus.jpg"
        sample_path.parent.mkdir(parents=True,exist_ok=True)
        if not sample_path.exists():
            urllib.request.urlretrieve(SAMPLE_URL,sample_path)
    print(json.dumps(records,indent=2))


if __name__=="__main__": main()
