"""V4-only audit helpers; existing V1/V2/V3 artifacts are read-only inputs."""
from pathlib import Path

from tryolabs_v3_data import ROOT, ASSETS, digest, write_json, full_frames

OUTPUT = ASSETS / "classification_v4"


def protected_snapshot():
    paths = []
    for directory in [ROOT / "starter", ROOT / "docs", ROOT / "tests", ASSETS]:
        for p in directory.rglob("*"):
            if not p.is_file():
                continue
            parts = p.relative_to(directory).parts
            if any(x in parts for x in [".git", ".venv", "__pycache__", "classification_v4"]):
                continue
            if p.name == ".DS_Store" or p.name.endswith(".lock") or "v4" in p.name.lower():
                continue
            paths.append(p)
    return {str(p.relative_to(ROOT)): {"size": p.stat().st_size, "sha256": digest(p)}
            for p in sorted(set(paths))}


def verify_preserved(before):
    changed = [name for name, old in before.items()
               if not (ROOT / name).is_file() or digest(ROOT / name) != old["sha256"]]
    if changed:
        raise RuntimeError(f"Protected V1/V2/V3 artifacts changed: {changed}")
    return {"passed": True, "protected_files": len(before), "changed_files": []}
