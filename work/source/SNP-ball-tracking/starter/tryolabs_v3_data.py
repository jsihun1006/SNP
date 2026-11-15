"""Read V2 artifacts without modifying/re-running detectors or trackers."""
import gzip
import hashlib
import json
from pathlib import Path
import shutil
import sqlite3
import zlib

ROOT=Path(__file__).resolve().parents[1]
ASSETS=ROOT/"data/derived/tryolabs_broadcast"
OUTPUT=ASSETS/"classification_v3"


def digest(path):
    h=hashlib.sha256()
    with Path(path).open("rb") as f:
        for b in iter(lambda:f.read(1048576),b""):h.update(b)
    return h.hexdigest()


def write_json(path,value):
    with Path(path).open("x",encoding="utf-8") as f:json.dump(value,f,indent=2,ensure_ascii=False,allow_nan=False)


def legacy_frames(cache=ASSETS/"classification_v2"):
    cache=Path(cache)
    raw={r["source_frame0"]:r for r in (json.loads(x) for x in (cache.parent/"preview/detections.jsonl").read_text(encoding="utf-8").splitlines())}
    frozen={}
    for line in (cache/"objects.jsonl").read_text(encoding="utf-8").splitlines():
        r=json.loads(line);frozen[r["source_frame0"],r["track_id"]]=r["label"]
    for line in (cache/"track_observations.jsonl").read_text(encoding="utf-8").splitlines():
        r=json.loads(line);i=r["frame0"]
        yield dict(frame0=i,sequence=f'legacy:{r["segment"]}',objects=[dict(o,frozen_v2_label=frozen.get((i,o["track_id"]))) for o in r["objects"]],
                   balls=raw[i]["balls"],source="legacy750")


def full_frames(cache,ranges,snapshot_dir):
    """Snapshot SQLite+WAL into V3 output, then query only that private copy.

    Stat stability detects concurrent source writes; source SQLite is never opened.
    tracking JSONL (not checkpoint pickle) supplies the unchanged Norfair tracks.
    """
    cache=Path(cache);snapshot_dir=Path(snapshot_dir);snapshot_dir.mkdir(parents=True,exist_ok=False)
    paths=[cache/("run.sqlite"+suffix) for suffix in ("","-wal","-shm") if (cache/("run.sqlite"+suffix)).exists()]
    before={str(p):(p.stat().st_size,p.stat().st_mtime_ns) for p in paths}
    for path in paths:shutil.copyfile(path,snapshot_dir/path.name)
    if before!={str(p):(p.stat().st_size,p.stat().st_mtime_ns) for p in paths}:
        raise RuntimeError("V2 cache changed while copying; pause its writer and retry in a new V3 output")
    db=sqlite3.connect((snapshot_dir/"run.sqlite").as_uri()+"?mode=ro",uri=True)
    try:
        for folder in sorted((cache/"chunks").glob("[0-9]*")):
            meta=json.loads((folder/"chunk.json").read_text(encoding="utf-8"))
            if not any(a<meta["end"] and b>meta["start"] for a,b in ranges):continue
            with gzip.open(folder/"tracking.jsonl.gz","rt",encoding="utf-8") as f,gzip.open(folder/"objects.jsonl.gz","rt",encoding="utf-8") as g:
                for line,oldline in zip(f,g):
                    r=json.loads(line);old=json.loads(oldline);i=r["frame0"]
                    if not any(a<=i<b for a,b in ranges):continue
                    if old["frame0"]!=i:raise ValueError("V2 tracking/classification axes differ")
                    found=db.execute("SELECT payload FROM detections WHERE frame0=?",(i,)).fetchone()
                    if found is None:raise ValueError(f"Missing detection cache at {i}")
                    det=json.loads(zlib.decompress(found[0]))
                    if det["status"]!="ok":raise ValueError(f"Failed source detections at {i}")
                    labels={o["track_id"]:o["label"] for o in old["objects"]}
                    yield dict(frame0=i,sequence=f'full:{r["scene_id"]}',objects=[dict(o,frozen_v2_label=labels.get(o["track_id"])) for o in r["players"]],balls=det["balls"],source="full_cache")
    finally:db.close()


def protected_snapshot():
    paths=[]
    for directory in [ROOT/"starter",ROOT/"docs",ROOT/"tests",ASSETS/"upstream",ASSETS/"yolov5",ASSETS/"preview",ASSETS/"classification_v2",ASSETS/"full_match_v2"]:
        paths += [p for p in directory.rglob("*") if p.is_file() and ".git" not in p.parts and "__pycache__" not in p.parts and p.name!=".DS_Store" and not p.name.endswith(".lock") and "v3" not in p.name.lower()]
    paths += [ASSETS/name for name in ["ball.pt","yolov5x.pt"]]
    return {str(p.relative_to(ROOT)):dict(size=p.stat().st_size,sha256=digest(p)) for p in paths}


def validate_preserved(before):
    changed=[p for p,r in before.items() if not (ROOT/p).is_file() or digest(ROOT/p)!=r["sha256"]]
    if changed:raise RuntimeError(f"Protected source artifacts changed: {changed}")
    return dict(protected_files=len(before),changed_files=[],passed=True)
