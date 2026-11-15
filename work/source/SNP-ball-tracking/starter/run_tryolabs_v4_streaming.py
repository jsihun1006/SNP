"""V4 full-match cached-video streaming runner. No detector or tracker updates.

Committed chunk directories are the transaction journal. No pickle, no source
SQLite connections, no whole-match object arrays, and no changes to frozen V4.
"""
import argparse
from contextlib import ExitStack
import csv
import gzip
import json
import os
from pathlib import Path, PureWindowsPath
import platform
import subprocess
import sys
import uuid

import cv2
import numpy as np

from run_tryolabs_classification_v3 import FrameReader, VIDEO
from run_tryolabs_classification_v4 import comparison, V3_CONFIG, V3_POLICY, V4_CONFIG, V4_POLICY
from tryolabs_v3_data import ROOT, ASSETS
from tryolabs_v4_stream_state import Engine
from tryolabs_v4_stream_io import (atomic_json, digest, signature, lock, jsonl_writer, read_jsonl,
                                   snapshot_database, catalog, validate_sources, records)

TOTAL=164487
VERSION="1.0.0"
CODE=["tryolabs_uniform_v2.py","tryolabs_uniform_v3.py","tryolabs_uniform_v4.py",
      "tryolabs_v4_context.py","tryolabs_v3_data.py","tryolabs_v4_data.py",
      "run_tryolabs_classification_v3.py","run_tryolabs_classification_v4.py",
      "tryolabs_v4_stream_state.py","tryolabs_v4_stream_io.py","run_tryolabs_v4_streaming.py"]
CONFIGS=[V3_CONFIG,V3_POLICY,V4_CONFIG,V4_POLICY]


def fingerprint():
    import imageio_ffmpeg
    return dict(version=VERSION,python=sys.version,platform=platform.platform(),numpy=np.__version__,opencv=cv2.__version__,
                ffmpeg_sha256=digest(imageio_ffmpeg.get_ffmpeg_exe()),
                code={n:digest(ROOT/"starter"/n) for n in CODE},configs={p.name:digest(p) for p in CONFIGS})


def new_engine():
    return Engine(*(json.loads(p.read_text(encoding="utf-8")) for p in CONFIGS))


def validate_windows_output(path):
    p=PureWindowsPath(str(path))
    if not p.is_absolute() or p.drive.upper()!="D:":
        raise ValueError("V4 streaming output must be a new absolute folder on D: drive")


def video_info(path):
    p=Path(path).resolve();cap=cv2.VideoCapture(str(p))
    try:
        frames=int(cap.get(cv2.CAP_PROP_FRAME_COUNT));fps=cap.get(cv2.CAP_PROP_FPS)
        if not cap.isOpened() or frames<=0 or abs(fps-25)>.01:raise ValueError(f"Unreadable/unsupported source video: {p}")
        size=[int(cap.get(3)),int(cap.get(4))]
    finally:cap.release()
    print("HASH_SOURCE_VIDEO",str(p),flush=True)
    return dict(path=str(p),signature=signature(p),frames=frames,fps=fps,size=size)


def initialize_source(out, cache, prepared, start, end):
    if prepared is not None:
        p=Path(prepared).resolve();count=0;keys=set()
        for r in read_jsonl(p):
            key=r["sequence"],r["frame0"]
            if key in keys:raise ValueError(f"Duplicate prepared frame: {key}")
            keys.add(key);count+=1
        if not count:raise ValueError("Empty prepared source")
        return dict(kind="prepared",path=str(p),signature=signature(p),count=count)
    p=Path(cache).resolve();database=snapshot_database(p,out)
    chunks=catalog(p,out/"source.sqlite",start,end)
    return dict(kind="cache",path=str(p),start=start,end=end,count=end-start,database_files=database,
                snapshot=signature(out/"source.sqlite"),chunks=chunks)


def committed(out, manifest, deep=True):
    cursor=0;result=[];manifest_sha=digest(out/"manifest.json")
    for folder in sorted((out/"chunks").glob("[0-9]*")):
        m=json.loads((folder/"chunk.json").read_text(encoding="utf-8"))
        if m["manifest_sha256"]!=manifest_sha or m["start_ordinal"]!=cursor or m["end_ordinal"]<=cursor:
            raise ValueError(f"Committed chunk gap/duplicate/manifest mismatch at {cursor}: {folder}")
        if m["end_ordinal"]>manifest["source"]["count"]:raise ValueError("Chunk extends beyond requested input")
        if deep:
            for name,info in m["files"].items():
                if signature(folder/name)!=info:raise ValueError(f"Committed output checksum mismatch: {folder/name}")
        result.append((folder,m));cursor=m["end_ordinal"]
    return result,cursor


class Encoder:
    def __init__(self,path):
        import imageio_ffmpeg
        self.path=path;self.log=path.with_suffix(".log").open("xb")
        cmd=[imageio_ffmpeg.get_ffmpeg_exe(),"-nostdin","-v","error","-n","-f","rawvideo","-pix_fmt","bgr24",
             "-s","1920x800","-r","25","-i","pipe:0","-an","-c:v","libx264","-threads","1",
             "-preset","veryfast","-crf","21","-pix_fmt","yuv420p","-movflags","+faststart",str(path)]
        self.proc=subprocess.Popen(cmd,stdin=subprocess.PIPE,stderr=self.log)
    def write(self,frame):self.proc.stdin.write(frame.tobytes())
    def close(self):
        try:self.proc.stdin.close()
        except BrokenPipeError:pass
        code=self.proc.wait();self.log.close()
        if code:raise RuntimeError(f"Encoder failed ({code}): {self.path}")


def decode_video(path):
    cap=cv2.VideoCapture(str(path));n=0
    try:
        if not cap.isOpened():raise ValueError(f"Video not readable: {path}")
        fps=cap.get(5);size=[int(cap.get(3)),int(cap.get(4))]
        while cap.read()[0]:n+=1
    finally:cap.release()
    return dict(decoded_frames=n,fps=fps,size=size)


def publish_summary(out,engine,manifest,cursor,state):
    summary=engine.metrics.summary()
    summary.update(state=state,requested_frames=manifest["source"]["count"],committed_frames=cursor,
                   processing_complete=cursor==manifest["source"]["count"],full_match_complete=False,
                   comparison_video_complete=False,scope=manifest["scope"],next_ordinal=cursor)
    if manifest["source"]["kind"]=="cache":summary["next_frame0"]=manifest["source"]["start"]+cursor
    atomic_json(out/"summary.json",summary);atomic_json(out/"status.json",summary)
    return summary


def run(out, *, cache=None, prepared=None, video_path=VIDEO, start=0, end=TOTAL,
        chunk_size=250, resume=False, video=True, limit_chunks=None, hook=None):
    out=Path(out).resolve()
    if not 1<=chunk_size<=2000:raise ValueError("Chunk size must be 1..2000")
    if start<0 or end<=start:raise ValueError("Invalid input frame interval")
    if out.exists() and not resume:raise ValueError(f"Output already exists; use --resume: {out}")
    if resume and not (out/"manifest.json").is_file():raise ValueError("Resume needs a committed manifest")
    if prepared is None and cache is None:raise ValueError("Supply a completed V2 cache or prepared regression input")
    # Prevent accidental writes inside any previous version's output or source tree.
    inputs=[Path(prepared).resolve().parent if prepared else Path(cache).resolve()]
    if any(out==p or out.is_relative_to(p) or p.is_relative_to(out) for p in inputs):
        raise ValueError("Output must be separate from input cache")
    if not resume:out.mkdir(parents=True,exist_ok=False)
    with lock(out):
        engine=new_engine();cursor=0;active_ordinal=None;active_frame=None
        try:
            if resume:
                manifest=json.loads((out/"manifest.json").read_text(encoding="utf-8"))
                if manifest["implementation"]!=fingerprint():raise ValueError("Code/config/software hash changed; resume refused")
                requested=dict(chunk_size=chunk_size,video=video,start=start,end=end)
                if manifest["settings"]!=requested:raise ValueError("Run settings changed; resume refused")
                source=manifest["source"]
                if source["kind"]!=("prepared" if prepared else "cache") or source["path"]!=str(Path(prepared or cache).resolve()):
                    raise ValueError("Input source changed; resume refused")
                validate_sources(source)
                if source["kind"]=="cache" and signature(out/"source.sqlite")!=source["snapshot"]:
                    raise ValueError("Detection snapshot corrupted")
                if video_info(video_path)!=manifest["video"]:raise ValueError("Video changed; resume refused")
                chunks,cursor=committed(out,manifest)
                if chunks:
                    state=json.loads((chunks[-1][0]/"state.json").read_text(encoding="utf-8"))
                    if state["next_ordinal"]!=cursor:raise ValueError("Checkpoint position differs")
                    engine.restore(state["engine"])
                for partial in out.glob(".work-*"):
                    archive=out/"interrupted";archive.mkdir(exist_ok=True);partial.rename(archive/(partial.name+"-"+uuid.uuid4().hex))
            else:
                vi=video_info(video_path)
                if prepared is None and end>vi["frames"]:raise ValueError("Requested interval exceeds video")
                source=initialize_source(out,cache,prepared,start,end)
                manifest=dict(schema=1,implementation=fingerprint(),video=vi,source=source,
                    settings=dict(chunk_size=chunk_size,video=video,start=start,end=end),
                    scope="full_match" if source["kind"]=="cache" and start==0 and end==TOTAL and vi["frames"]==TOTAL else "sample")
                (out/"chunks").mkdir();atomic_json(out/"manifest.json",manifest)
            reader=FrameReader(video_path);iterator=iter(records(source,out,cursor));completed_this_session=0
            try:
                while cursor<source["count"]:
                    a=cursor;b=min(a+chunk_size,source["count"])
                    work=out/f'.work-{a:09d}-{uuid.uuid4().hex}';work.mkdir()
                    encoder=None;first_frame=last_frame=None
                    try:
                        if video:encoder=Encoder(work/"comparison.mp4")
                        with ExitStack() as stack:
                            objects_write=stack.enter_context(jsonl_writer(work/"objects.jsonl.gz"))
                            frames_write=stack.enter_context(jsonl_writer(work/"frames.jsonl.gz"))
                            inputs_write=stack.enter_context(jsonl_writer(work/"input_frames.jsonl.gz"))
                            for ordinal in range(a,b):
                                active_ordinal=ordinal
                                active_frame=source["start"]+ordinal if source["kind"]=="cache" else None
                                n,raw=next(iterator)
                                if n!=ordinal:raise ValueError(f"Source ordinal gap at {ordinal}")
                                active_frame=raw["frame0"];first_frame=active_frame if first_frame is None else first_frame;last_frame=active_frame
                                image=reader.read(active_frame);objects,row=engine.process(raw,image)
                                inputs_write(raw)
                                for obj in objects:objects_write(obj)
                                frames_write(dict(ordinal=ordinal,**row))
                                if encoder:encoder.write(comparison(image,objects,raw,row))
                                if hook:hook("after_frame",ordinal)
                            if b==source["count"]:
                                try:next(iterator)
                                except StopIteration:pass
                                else:raise ValueError("Source contains unexpected extra frames")
                    finally:
                        if encoder:encoder.close()
                    verified=decode_video(work/"comparison.mp4") if video else None
                    if video and (verified["decoded_frames"]!=b-a or verified["fps"]!=25 or verified["size"]!=[1920,800]):
                        raise ValueError("Chunk video frame/format verification failed")
                    atomic_json(work/"state.json",dict(schema=1,next_ordinal=b,manifest_sha256=digest(out/"manifest.json"),engine=engine.export()))
                    for file in work.iterdir():
                        if file.is_file():
                            with file.open("r+b") as f:os.fsync(f.fileno())
                    files={p.name:signature(p) for p in work.iterdir() if p.is_file()}
                    meta=dict(start_ordinal=a,end_ordinal=b,first_frame0=first_frame,last_frame0=last_frame,
                              next_frame0=source["start"]+b if source["kind"]=="cache" else None,
                              manifest_sha256=digest(out/"manifest.json"),config_sha256=manifest["implementation"]["configs"],
                              files=files,video_verification=verified)
                    atomic_json(work/"chunk.json",meta)
                    target=out/"chunks"/f'{a:09d}-{b:09d}'
                    if target.exists():raise ValueError("Refusing to replace committed output")
                    work.rename(target)  # All logs/video/state become visible in ONE same-volume rename.
                    cursor=b
                    if hook:hook("after_commit",b)
                    atomic_json(out/"checkpoint.json",dict(next_ordinal=b,chunk=target.name,state_sha256=files["state.json"]["sha256"]))
                    publish_summary(out,engine,manifest,cursor,"PROCESSING")
                    completed_this_session+=1
                    print("COMMIT",cursor,"/",source["count"],"source",last_frame,"tracks",engine.metrics.data["peak_temporal_tracks"],flush=True)
                    if limit_chunks is not None and completed_this_session>=limit_chunks:break
            finally:
                reader.close()
                if hasattr(iterator,"close"):iterator.close()
            validate_sources(source)
            if signature(video_path)!=manifest["video"]["signature"]:raise ValueError("Video changed during processing")
            return publish_summary(out,engine,manifest,cursor,"PROCESSED" if cursor==source["count"] else "STOPPED")
        except (Exception,KeyboardInterrupt) as e:
            # A directory may have committed just before a crash, while global pointer lags.
            if (out/"manifest.json").exists():
                m=json.loads((out/"manifest.json").read_text());_,saved=committed(out,m,deep=False)
                atomic_json(out/"failure.json",dict(state="INTERRUPTED" if isinstance(e,KeyboardInterrupt) else "FAILED",
                    error_type=type(e).__name__,error=str(e),committed_frames=saved,failed_ordinal=active_ordinal,
                    failed_frame0=active_frame,requested_frames=m["source"]["count"],full_match_complete=False))
            else:
                atomic_json(out/"failure.json",dict(state="INITIALIZATION_FAILED",error_type=type(e).__name__,error=str(e),
                    requested_range=[start,end],full_match_complete=False))
            raise


def verify(out, *, require_full=False, decode=False, _locked=False):
    if not _locked:
        with lock(Path(out)):
            return verify(out,require_full=require_full,decode=decode,_locked=True)
    out=Path(out).resolve();manifest=json.loads((out/"manifest.json").read_text(encoding="utf-8"))
    if fingerprint()!=manifest["implementation"]:raise ValueError("Code/config/software hash changed; verification refused")
    chunks,cursor=committed(out,manifest);position=0;objects_count=0;total_counts={"v3_counts":{},"v4_counts":{}}
    for folder,m in chunks:
        count=0;counts={};frames_count={}
        for row in read_jsonl(folder/"frames.jsonl.gz"):
            if row["ordinal"]!=position:raise ValueError(f"Missing/duplicate output ordinal at {position}")
            if manifest["source"]["kind"]=="cache" and row["frame0"]!=manifest["source"]["start"]+position:
                raise ValueError(f"Missing/duplicate source frame at {position}")
            position+=1;count+=1;frames_count[row["sequence"],row["frame0"]]=0
        for obj in read_jsonl(folder/"objects.jsonl.gz"):
            key=obj["sequence"],obj["source_frame0"]
            if key not in frames_count:raise ValueError("Object has no output frame")
            objects_count+=1;counts[obj["final_label"]]=counts.get(obj["final_label"],0)+1
            for key,field in [("v3_counts","v3_label"),("v4_counts","final_label")]:
                label=obj[field];total_counts[key][label]=total_counts[key].get(label,0)+1
        if count!=m["end_ordinal"]-m["start_ordinal"]:raise ValueError("Chunk row count mismatch")
        if manifest["settings"]["video"]:
            v=decode_video(folder/"comparison.mp4") if decode else m["video_verification"]
            if v["decoded_frames"]!=count:raise ValueError("Chunk video count mismatch")
    if position!=cursor:raise ValueError("Coverage count mismatch")
    complete=cursor==manifest["source"]["count"]
    assembled=out/"assembled/comparison_full.mp4";assembly=out/"assembled/assembly.json";comparison_complete=False
    if assembly.is_file() and assembled.is_file():
        a=json.loads(assembly.read_text())
        comparison_complete=(a["manifest_sha256"]==digest(out/"manifest.json") and a["video"]==signature(assembled)
                             and a["decoded_frames"]==manifest["source"]["count"])
        if decode and comparison_complete:comparison_complete=decode_video(assembled)["decoded_frames"]==cursor
    validate_sources(manifest["source"])
    if signature(manifest["video"]["path"])!=manifest["video"]["signature"]:raise ValueError("Source video checksum changed")
    if complete and chunks:
        e=new_engine();e.restore(json.loads((chunks[-1][0]/"state.json").read_text())["engine"])
        summary=e.metrics.summary()
        if summary["frames"]!=cursor or summary["objects"]!=objects_count:raise ValueError("Aggregate counter mismatch")
        if any(summary[k]!=v for k,v in total_counts.items()):raise ValueError("Class distribution differs from logs")
    report=dict(passed=complete,expected_frames=manifest["source"]["count"],committed_frames=cursor,
                missing_frames=manifest["source"]["count"]-cursor,duplicates=0,objects=objects_count,
                comparison_video_complete=comparison_complete,scope=manifest["scope"],
                full_match_complete=complete and comparison_complete and manifest["scope"]=="full_match",
                source_files_unchanged=True)
    atomic_json(out/"verification.json",report)
    if (out/"summary.json").is_file():
        s=json.loads((out/"summary.json").read_text());s.update(report);atomic_json(out/"summary.json",s)
    if not complete or (require_full and not report["full_match_complete"]):
        raise ValueError("Requested processing/video coverage is not complete: "+json.dumps(report))
    return report


def assemble(out):
    import imageio_ffmpeg
    out=Path(out).resolve()
    with lock(out):
        report=verify(out,_locked=True)
        manifest=json.loads((out/"manifest.json").read_text())
        if not manifest["settings"]["video"]:raise ValueError("This run has no comparison chunks")
        if (out/"assembled").exists():
            if report["comparison_video_complete"]:return report
            raise ValueError("Existing assembled video is unverified; preserved, not overwritten")
        chunks,_=committed(out,manifest)
        playlist=out/"concat.txt"
        with playlist.open("w",encoding="utf-8",newline="\n") as f:
            for folder,_ in chunks:f.write("file '"+(folder.relative_to(out)/"comparison.mp4").as_posix()+"'\n")
        work=out/(".assembly-work-"+uuid.uuid4().hex);work.mkdir();pending=work/"comparison_full.mp4"
        cmd=[imageio_ffmpeg.get_ffmpeg_exe(),"-nostdin","-v","error","-n","-f","concat","-safe","0","-i",str(playlist),
             "-an","-c","copy","-movflags","+faststart",str(pending)]
        subprocess.run(cmd,check=True)
        result=decode_video(pending)
        if result["decoded_frames"]!=manifest["source"]["count"] or result["fps"]!=25:raise ValueError("Assembled video coverage failed")
        with pending.open("r+b") as f:os.fsync(f.fileno())
        atomic_json(work/"assembly.json",dict(manifest_sha256=digest(out/"manifest.json"),video=signature(pending),**result))
        work.rename(out/"assembled")
        return verify(out,_locked=True)


def iter_output(out,name):
    manifest=json.loads((out/"manifest.json").read_text());chunks,_=committed(out,manifest)
    for folder,_ in chunks:yield from read_jsonl(folder/name)


def compare_reference(out,reference):
    from itertools import zip_longest
    objects=0;frames=0
    for got,want in zip_longest(iter_output(out,"objects.jsonl.gz"),read_jsonl(reference/"objects.jsonl")):
        if got!=want:raise ValueError(f"V4 reference object mismatch at observation {objects}")
        objects+=1
    with (reference/"frames.csv").open(encoding="utf-8",newline="") as f:
        for got,want in zip_longest(iter_output(out,"frames.jsonl.gz"),csv.DictReader(f)):
            if got is None or want is None or any(("" if got.get(k) is None else str(got[k]))!=v for k,v in want.items()):
                raise ValueError(f"V4 reference frame mismatch at ordinal {frames}")
            frames+=1
    return dict(frames=frames,objects=objects,all_object_fields_identical=True,all_frame_fields_identical=True)


def regression(output,reference,video_path=VIDEO,chunk_size=137):
    output=Path(output);reference=Path(reference);output.mkdir(parents=True,exist_ok=False)
    first=output/"continuous";second=output/"resumed";prepared=reference/"input_frames.jsonl"
    run(first,prepared=prepared,video_path=video_path,chunk_size=chunk_size,video=False)
    run(second,prepared=prepared,video_path=video_path,chunk_size=chunk_size,video=False,limit_chunks=4)
    run(second,prepared=prepared,video_path=video_path,chunk_size=chunk_size,video=False,resume=True)
    a=compare_reference(first,reference);b=compare_reference(second,reference)
    for name in ["objects.jsonl.gz","frames.jsonl.gz","input_frames.jsonl.gz"]:
        if any(x!=y for x,y in __import__("itertools").zip_longest(iter_output(first,name),iter_output(second,name))):
            raise ValueError(f"Continuous/resume mismatch in {name}")
    m=json.loads((first/"manifest.json").read_text());c1,_=committed(first,m);c2,_=committed(second,json.loads((second/"manifest.json").read_text()))
    state_equal=all(digest(p1/"state.json")==digest(p2/"state.json") for (p1,_),(p2,_) in zip(c1,c2))
    if not state_equal:raise ValueError("Continuous/resume state checkpoints differ")
    report=dict(passed=True,continuous=a,resumed=b,checkpoint_states_identical=state_equal,
                reference_is_manual_ground_truth=False,full_match_complete=False)
    atomic_json(output/"regression.json",report);return report


def main():
    p=argparse.ArgumentParser(description=__doc__);sub=p.add_subparsers(dest="command",required=True)
    for name in ["run","verify","assemble","status","regression"]:
        s=sub.add_parser(name);s.add_argument("--output",type=Path,required=True)
        if name=="run":
            s.add_argument("--cache",type=Path,default=ASSETS/"full_match_v2/windows_gtx1660_full")
            s.add_argument("--prepared",type=Path);s.add_argument("--video-path",type=Path,default=VIDEO)
            s.add_argument("--start",type=int,default=0);s.add_argument("--end",type=int,default=TOTAL)
            s.add_argument("--chunk-size",type=int,default=250);s.add_argument("--resume",action="store_true")
            s.add_argument("--no-video",action="store_true");s.add_argument("--limit-chunks",type=int)
        if name=="verify":s.add_argument("--require-full",action="store_true");s.add_argument("--decode-video",action="store_true")
        if name=="regression":
            s.add_argument("--reference",type=Path,required=True);s.add_argument("--video-path",type=Path,default=VIDEO)
            s.add_argument("--chunk-size",type=int,default=137)
    a=p.parse_args()
    if os.name=="nt":
        try:validate_windows_output(a.output.resolve())
        except ValueError as e:p.error(str(e))
    if os.name!="nt" and PureWindowsPath(str(a.output)).drive:p.error("Windows drive paths require Windows; use a local test directory here")
    try:
        if a.command=="run":r=run(a.output,cache=a.cache,prepared=a.prepared,video_path=a.video_path,start=a.start,end=a.end,
            chunk_size=a.chunk_size,resume=a.resume,video=not a.no_video,limit_chunks=a.limit_chunks)
        elif a.command=="verify":r=verify(a.output,require_full=a.require_full,decode=a.decode_video)
        elif a.command=="assemble":r=assemble(a.output)
        elif a.command=="regression":r=regression(a.output,a.reference,a.video_path,a.chunk_size)
        else:
            m=json.loads((a.output/"manifest.json").read_text());_,cursor=committed(a.output,m,deep=False)
            v=json.loads((a.output/"verification.json").read_text()) if (a.output/"verification.json").exists() else {}
            r=dict(committed_frames=cursor,expected_frames=m["source"]["count"],processing_complete=cursor==m["source"]["count"],
                   last_verified_full_match_complete=v.get("full_match_complete",False),historical_failure=json.loads((a.output/"failure.json").read_text()) if (a.output/"failure.json").exists() else None)
        print(json.dumps(r,indent=2))
    except (Exception,KeyboardInterrupt) as e:
        print(f"ERROR: {type(e).__name__}: {e}",file=sys.stderr);return 130 if isinstance(e,KeyboardInterrupt) else 1
    return 0


if __name__=="__main__":raise SystemExit(main())
