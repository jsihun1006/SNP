"""Build an allowlisted code-only Windows V4 ZIP, with SHA256 manifest."""
import argparse
import hashlib
import json
from pathlib import Path
import zipfile

ROOT=Path(__file__).resolve().parents[1]
FILES=[
    "RULES.md", "requirements-tryolabs-v4-windows.txt",
    "windows/run_tryolabs_v4.cmd", "windows/tryolabs_v4_windows.json", "windows/install_tryolabs_v4_bundle.py",
    "starter/windows_tryolabs_v4.py", "starter/package_tryolabs_v4_windows.py",
    "starter/tryolabs_uniform_v2.py", "starter/tryolabs_uniform_v3.py",
    "starter/tryolabs_v3_data.py", "starter/run_tryolabs_classification_v3.py", "starter/evaluate_tryolabs_v3_external.py",
    "starter/tryolabs_uniform_v4.py", "starter/tryolabs_v4_context.py", "starter/tryolabs_v4_data.py",
    "starter/run_tryolabs_classification_v4.py", "starter/analyze_tryolabs_classification_v4.py",
    "starter/calibrate_tryolabs_v4.py", "starter/diagnose_tryolabs_v4_colors.py",
    "starter/configs/tryolabs_classification_v2_hsv.json", "starter/configs/tryolabs_classification_v2_policy.json",
    "starter/configs/tryolabs_classification_v3_hsv.json", "starter/configs/tryolabs_classification_v3_policy.json",
    "starter/configs/tryolabs_classification_v4_hsv.json", "starter/configs/tryolabs_classification_v4_policy.json",
    "tests/test_tryolabs_classification_v3.py", "tests/test_tryolabs_classification_v4.py",
    "tests/test_tryolabs_v4_windows_handoff.py", "docs/TRYOLABS_CLASSIFICATION_V4.md",
    "docs/TRYOLABS_V4_WINDOWS_HANDOFF.md",
]


def build(output, root=ROOT):
    output=Path(output)
    manifest=dict(format_version=1,description="Code/config/tests/docs only; no video, weights, tracks or cache",
                  files={})
    blobs={}
    for name in FILES:
        p=root/name
        if p.is_symlink() or not p.is_file():raise ValueError(f"Missing or symlinked bundle source: {name}")
        data=p.read_bytes()
        if len(data)>1_000_000:raise ValueError(f"Unexpected large source: {name}")
        if name.startswith("data/") or p.suffix.lower() not in (".py",".json",".cmd",".md",".txt"):
            raise ValueError(f"Non-code file rejected: {name}")
        blobs[name]=data
        manifest["files"][name]=dict(size=len(data),sha256=hashlib.sha256(data).hexdigest())
    output.parent.mkdir(parents=True,exist_ok=True)
    with zipfile.ZipFile(output,"x",compression=zipfile.ZIP_DEFLATED) as archive:
        for name,data in blobs.items():archive.writestr(name,data)
        archive.writestr("bundle_manifest.json",json.dumps(manifest,indent=2))
    # Check archive members against the exact allowlist, not a directory glob.
    with zipfile.ZipFile(output) as archive:
        assert set(archive.namelist())==set(FILES)|{"bundle_manifest.json"}
        assert archive.testzip() is None
        for name,data in blobs.items():assert archive.read(name)==data
    return dict(path=str(output),files=len(blobs),bytes=output.stat().st_size,
                sha256=hashlib.sha256(output.read_bytes()).hexdigest(),data_files=0)


if __name__=="__main__":
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument("--output",type=Path,required=True)
    a=p.parse_args()
    print(json.dumps(build(a.output),indent=2))
