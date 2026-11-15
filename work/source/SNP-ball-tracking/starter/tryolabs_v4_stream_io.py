"""Read-only cache snapshots and atomic, deterministic output primitives."""
from contextlib import contextmanager
import gzip
import hashlib
import io
from itertools import zip_longest
import json
import os
from pathlib import Path
import shutil
import sqlite3
import uuid
import zlib


def digest(path):
    h=hashlib.sha256()
    with Path(path).open("rb") as f:
        for block in iter(lambda:f.read(1024*1024),b""):h.update(block)
    return h.hexdigest()


def atomic_json(path, value):
    path=Path(path);temp=path.with_name(path.name+".tmp-"+uuid.uuid4().hex)
    try:
        with temp.open("x",encoding="utf-8",newline="\n") as f:
            json.dump(value,f,ensure_ascii=False,allow_nan=False,separators=(",",":"));f.write("\n");f.flush();os.fsync(f.fileno())
        os.replace(temp,path)
    finally:
        if temp.exists():temp.unlink()


@contextmanager
def lock(output):
    path=Path(output)/".runner.lock"
    with path.open("a+b") as f:
        if f.tell()==0:f.write(b"0");f.flush()
        f.seek(0);acquired=False
        try:
            if os.name=="nt":
                import msvcrt
                msvcrt.locking(f.fileno(),msvcrt.LK_NBLCK,1)
            else:
                import fcntl
                fcntl.flock(f,fcntl.LOCK_EX|fcntl.LOCK_NB)
            acquired=True;yield
        except OSError as e:
            if not acquired:raise RuntimeError(f"Another runner holds {path}") from e
            raise
        finally:
            if acquired:
                f.seek(0)
                if os.name=="nt":msvcrt.locking(f.fileno(),msvcrt.LK_UNLCK,1)
                else:fcntl.flock(f,fcntl.LOCK_UN)


@contextmanager
def jsonl_writer(path):
    # Fixed gzip timestamp and no staging filename in its header: reproducible bytes.
    with Path(path).open("xb") as raw:
        gz=gzip.GzipFile(filename="",fileobj=raw,mode="wb",mtime=0)
        text=io.TextIOWrapper(gz,encoding="utf-8",newline="\n")
        try:yield lambda value:text.write(json.dumps(value,separators=(",",":"),allow_nan=False)+"\n")
        finally:
            text.close();raw.flush();os.fsync(raw.fileno())


def read_jsonl(path):
    opener=gzip.open if str(path).endswith(".gz") else open
    with opener(path,"rt",encoding="utf-8") as f:
        for line in f:
            if line.strip():yield json.loads(line)


def signature(path):
    p=Path(path);return dict(size=p.stat().st_size,sha256=digest(p))


def sqlite_files(cache):
    return [cache/("run.sqlite"+s) for s in ("","-wal","-shm") if (cache/("run.sqlite"+s)).exists()]


def snapshot_database(cache, output):
    cache=Path(cache);work=output/"source_copy";work.mkdir()
    files=sqlite_files(cache)
    if not (cache/"run.sqlite").is_file():raise ValueError(f"No detection DB at {cache}")
    before={p.name:signature(p) for p in files}
    for p in files:shutil.copyfile(p,work/p.name)
    after={p.name:signature(p) for p in sqlite_files(cache)}
    if before!=after or any(signature(work/n)!=info for n,info in before.items()):
        raise ValueError("Source DB/WAL/SHM changed during copy. Stop all V2 cache writers/readers and use a new output")
    # Open ONLY the copies. Backup merges committed WAL into a standalone snapshot.
    source=sqlite3.connect((work/"run.sqlite").resolve().as_uri()+"?mode=ro",uri=True)
    target=sqlite3.connect(output/"source.sqlite")
    try:source.backup(target)
    finally:target.close();source.close()
    with (output/"source.sqlite").open("r+b") as f:os.fsync(f.fileno())
    shutil.rmtree(work)
    return before


def catalog(cache, snapshot, start, end):
    db=sqlite3.connect(snapshot.resolve().as_uri()+"?mode=ro&immutable=1",uri=True)
    try:
        n=db.execute("SELECT COUNT(*) FROM detections WHERE frame0>=? AND frame0<?",(start,end)).fetchone()[0]
        if n!=end-start:raise ValueError(f"Detection cache missing {end-start-n} frames in [{start},{end})")
        rows=db.execute("SELECT start,end,path FROM chunks WHERE end>? AND start<? ORDER BY start",(start,end)).fetchall()
    finally:db.close()
    result=[];cursor=start
    for a,b,name in rows:
        relative=Path(name.replace("\\","/"));folder=(cache/relative).resolve()
        if not folder.is_relative_to(cache.resolve()):raise ValueError("Source chunk path escapes cache")
        if max(a,start)!=cursor or b<=a:raise ValueError(f"Source chunk gap/overlap at {cursor}: [{a},{b})")
        meta=json.loads((folder/"chunk.json").read_text(encoding="utf-8"))
        if (meta["start"],meta["end"])!=(a,b):raise ValueError(f"Source chunk metadata differs: {folder}")
        names=["chunk.json","tracking.jsonl.gz","objects.jsonl.gz"]
        result.append(dict(start=a,end=b,path=str(relative),files={n:signature(folder/n) for n in names}))
        cursor=min(b,end)
    if cursor!=end:raise ValueError(f"Completed tracking cache missing [{cursor},{end})")
    return result


def validate_sources(source):
    if source["kind"]=="prepared":
        if signature(source["path"])!=source["signature"]:raise ValueError("Prepared cache changed")
        return
    cache=Path(source["path"])
    if {p.name:signature(p) for p in sqlite_files(cache)}!=source["database_files"]:
        raise ValueError("Original detection DB/WAL/SHM changed; resume refused")
    for chunk in source["chunks"]:
        for name,info in chunk["files"].items():
            if signature(cache/chunk["path"]/name)!=info:raise ValueError(f"Source chunk changed: {chunk['path']}/{name}")


def records(source, output, ordinal=0):
    if source["kind"]=="prepared":
        for n,row in enumerate(read_jsonl(source["path"])):
            if n>=ordinal:yield n,row
        return
    db=sqlite3.connect((output/"source.sqlite").resolve().as_uri()+"?mode=ro&immutable=1",uri=True)
    start=source["start"];end=source["end"];next_frame=start+ordinal;cache=Path(source["path"])
    try:
        for chunk in source["chunks"]:
            if chunk["end"]<=next_frame:continue
            folder=cache/chunk["path"]
            for name,info in chunk["files"].items():
                if signature(folder/name)!=info:raise ValueError(f"Source changed: {folder/name}")
            count=0
            for tracking,old in zip_longest(read_jsonl(folder/"tracking.jsonl.gz"),read_jsonl(folder/"objects.jsonl.gz")):
                if tracking is None or old is None:raise ValueError(f"Source row count mismatch: {folder}")
                i=chunk["start"]+count;count+=1
                if tracking["frame0"]!=i or old["frame0"]!=i:raise ValueError(f"Source missing/duplicate frame {i}: {folder}")
                if i<next_frame or i<start or i>=end:continue
                if i!=next_frame:raise ValueError(f"Source discontinuity: expected {next_frame}, got {i}")
                found=db.execute("SELECT payload FROM detections WHERE frame0=?",(i,)).fetchone()
                if found is None:raise ValueError(f"No detection at frame {i}")
                det=json.loads(zlib.decompress(found[0]))
                if det["status"]!="ok":raise ValueError(f"Failed cached detection at frame {i}")
                labels={o["track_id"]:o["label"] for o in old["objects"]}
                yield i-start,dict(frame0=i,sequence=f'full:{tracking["scene_id"]}',
                    objects=[dict(o,frozen_v2_label=labels.get(o["track_id"])) for o in tracking["players"]],balls=det["balls"],source="full_cache")
                next_frame=i+1
            if count!=chunk["end"]-chunk["start"]:raise ValueError(f"Incomplete source gzip: {folder}")
        if next_frame!=end:raise ValueError(f"Source ended at {next_frame}, expected {end}")
    finally:db.close()
