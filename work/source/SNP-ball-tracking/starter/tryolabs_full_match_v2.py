"""Resumable full-broadcast V1/V2 diagnostic. Existing implementations are read-only.

PYTHONHASHSEED=0 <experiment-python> starter/tryolabs_full_match_v2.py --stage detect --device mps
PYTHONHASHSEED=0 <experiment-python> starter/tryolabs_full_match_v2.py --stage process --follow
"""

import argparse
from contextlib import contextmanager
from collections import Counter
import gzip
import hashlib
import json
import logging
import os
from pathlib import Path
import pickle
import shutil
import sqlite3
import subprocess
import time
from types import SimpleNamespace
import warnings
import zlib
import platform

import cv2
import numpy as np
import pandas as pd
import torch

import run_tryolabs_broadcast as legacy
import run_tryolabs_classification_v2 as v2
from tryolabs_uniform_v2 import UniformClassifier, TemporalClassifier, select_candidate
from tryolabs_runtime import file_lock, resolve_device, DetectorPair, log_event, memory_info, install_cpu_nms as cpu_nms

ROOT = legacy.ROOT
OUT = legacy.ASSETS/"full_match_v2"
VIDEO = v2.VIDEO
CONFIG = v2.DEFAULT_CONFIG
TOTAL = 164487
FPS = 25
WIDTH, HEIGHT = 1920, 930


def pack(value):
    return zlib.compress(json.dumps(value,separators=(",",":"),allow_nan=False).encode(),1)


def unpack(value):
    return json.loads(zlib.decompress(value))


def atomic_json(path, value):
    temp = path.with_name(path.name+".tmp")
    with temp.open("w", encoding="utf-8") as f:
        json.dump(value,f,indent=2); f.flush(); os.fsync(f.fileno())
    temp.replace(path)


def connection(out):
    db = sqlite3.connect(out/"run.sqlite",timeout=60)
    db.execute("PRAGMA journal_mode=WAL")
    db.execute("PRAGMA synchronous=FULL")
    db.executescript("""
    CREATE TABLE IF NOT EXISTS detections(frame0 INTEGER PRIMARY KEY,payload BLOB NOT NULL);
    CREATE TABLE IF NOT EXISTS results(frame0 INTEGER PRIMARY KEY,payload BLOB NOT NULL);
    CREATE TABLE IF NOT EXISTS chunks(start INTEGER PRIMARY KEY,end INTEGER NOT NULL,path TEXT NOT NULL,sha256 TEXT NOT NULL);
    """)
    return db


@contextmanager
def stage_lock(out, stage):
    with file_lock(out/f"{stage}.lock"):
        yield


def initialize(out, start, end, batch=16, device="mps", residency="auto", reuse_cache=False):
    out.mkdir(parents=True,exist_ok=True)
    st = VIDEO.stat()
    settings = dict(schema=1,video=str(VIDEO),size=st.st_size,mtime_ns=st.st_mtime_ns,
                    total_source_frames=TOTAL,fps=FPS,start=start,end=end,
                    config_sha256=legacy.digest(CONFIG),policy_sha256=legacy.digest(v2.POLICY),
                    frozen_v1_sha256=legacy.digest(ROOT/"starter/run_tryolabs_broadcast.py"),
                    frozen_v2_sha256=legacy.digest(ROOT/"starter/tryolabs_uniform_v2.py"),
                    weights={name:legacy.digest(legacy.ASSETS/name) for name in ["ball.pt","yolov5x.pt"]},
                    inference_size=640,channel_order="upstream_BGR",dtype="float32",device=device,batch=batch,
                    nms="unchanged upstream thresholds; per-image CPU NMS to avoid GPU batch timeout",
                    scene_policy=dict(hist_cut=.55,gray_cut=.22,structural_cut=.38,hash_cut_bits=26),
                    live_play_status="UNVERIFIED",replay_status="repeated visual signature heuristic only",
                    model_residency=residency,reuse_750_cache=reuse_cache,platform=platform.system())
    manifest = out/"manifest.json"
    if manifest.exists():
        old = json.loads(manifest.read_text(encoding="utf-8"))
        if old != settings:
            raise ValueError("Resume manifest mismatch: source, frozen settings, weights, bounds or inference settings changed")
    else:
        atomic_json(manifest,settings)
        capture_preservation(out)
    connection(out).close()
    return settings


def ranges(indices):
    result=[]
    for i in sorted(set(indices)):
        if result and i == result[-1][1]+1:result[-1][1]=i
        else:result.append([i,i])
    return result


def cached_records(out, start, end):
    db=connection(out)
    try:
        return [(int(i),unpack(b)) for i,b in db.execute("SELECT frame0,payload FROM detections WHERE frame0>=? AND frame0<? ORDER BY frame0",(start,end))]
    finally:db.close()


def seed_existing_cache(db, start, end):
    inserted=0
    cache = legacy.ASSETS/"preview/detections.jsonl"
    if not cache.is_file():
        return 0
    for line in cache.read_text(encoding="utf-8").splitlines():
        raw=json.loads(line);i=raw["source_frame0"]
        if start <= i < end:
            value=dict(frame0=i,persons=raw["persons"],balls=raw["balls"],status="ok",origin="frozen_750_cache")
            inserted+=db.execute("INSERT OR IGNORE INTO detections VALUES (?,?)",(i,pack(value))).rowcount
    db.commit()
    return inserted


def install_cpu_nms():
    cpu_nms()


def capture_preservation(out):
    paths=[ROOT/"starter"/name for name in ["run_tryolabs_broadcast.py","run_tryolabs_classification_v2.py",
           "tryolabs_uniform_v2.py","tryolabs_runtime.py","tryolabs_full_match_v2.py"]]
    paths += [CONFIG,v2.POLICY]+[legacy.ASSETS/name for name in ["ball.pt","yolov5x.pt"]]
    source=VIDEO.stat()
    atomic_json(out/"preservation_before.json",dict(files={str(p.relative_to(ROOT)):{"sha256":legacy.digest(p),"size":p.stat().st_size} for p in paths},
                video=dict(size=source.st_size,mtime_ns=source.st_mtime_ns)))


def exact_mask_range(hsv,ranges):
    """Same inclusive uint8 HSV box union as frozen numpy implementation."""
    result=np.zeros(hsv.shape[:2],bool)
    for bounds in ranges:
        result |= cv2.inRange(hsv,tuple(bounds["lower"]),tuple(bounds["upper"])).astype(bool)
    return result


def install_exact_cpu_adapters():
    """Performance adapters only; frozen source files/settings remain unchanged."""
    import tryolabs_uniform_v2 as uniform
    import inference.base_classifier as base
    from inference.box import Box
    uniform.mask_range=exact_mask_range
    class CroppedCopyBox(Box):
        def __init__(self,top_left,bottom_right,img):
            self.top_left=tuple(int(v) for v in top_left)
            self.bottom_right=tuple(int(v) for v in bottom_right)
            self.img=img[self.top_left[1]:self.bottom_right[1],self.top_left[0]:self.bottom_right[0]].copy()
    base.Box=CroppedCopyBox


def detect(out, manifest, stop_batches=None):
    warnings.filterwarnings("ignore",category=FutureWarning)
    legacy.import_upstream();torch.set_num_threads(2)
    db=connection(out)
    if manifest.get("reuse_750_cache",False):seed_existing_cache(db,manifest["start"],manifest["end"])
    present={i for i, in db.execute("SELECT frame0 FROM detections")}
    pending=[i for i in range(manifest["start"],manifest["end"]) if i not in present]
    if not pending:
        atomic_json(out/"detector_status.json",dict(complete=True,cached=len(present),pending=0));db.close();return
    detectors=None
    cap=cv2.VideoCapture(str(VIDEO));last=-2;completed=0;t0=time.monotonic()
    try:
        detectors=DetectorPair(legacy.load_model,legacy.ASSETS,manifest["device"],manifest.get("model_residency","auto"),out/"runtime_events.jsonl")
        for offset in range(0,len(pending),manifest["batch"]):
            ids=pending[offset:offset+manifest["batch"]];images=[];good=[];records=[]
            for i in ids:
                if i != last+1:cap.set(cv2.CAP_PROP_POS_FRAMES,i)
                ok,image=cap.read();last=i
                if not ok or abs(cap.get(cv2.CAP_PROP_POS_FRAMES)-(i+1))>.1:
                    records.append(dict(frame0=i,persons=[],balls=[],status="decode_failed",origin="source_decode"))
                else:images.append(image);good.append(i)
            if images:
                pp,bb=detectors.predict(images,good)
                if len(pp)!=len(images) or len(bb)!=len(images):raise ValueError("Incomplete detector batch")
                for i,p,b in zip(good,pp,bb):
                    p=p[(p["name"]=="person") & (p["confidence"]>.35)]
                    b=b[b["confidence"]>.3]
                    records.append(dict(frame0=i,persons=p.to_dict("records"),balls=b.to_dict("records"),status="ok",origin="shared_yolo"))
            with db:
                db.executemany("INSERT INTO detections VALUES (?,?)",[(r["frame0"],pack(r)) for r in records])
            completed+=len(ids)
            if offset%max(manifest["batch"],manifest["batch"]*16)==0 or offset+len(ids)>=len(pending):
                count=db.execute("SELECT COUNT(*) FROM detections").fetchone()[0]
                status=dict(complete=count==manifest["end"]-manifest["start"],cached=count,
                            pending=manifest["end"]-manifest["start"]-count,last_committed_frame=max(ids),
                            session_frames=completed,session_seconds=time.monotonic()-t0,
                            session_fps=completed/max(.001,time.monotonic()-t0),
                            model_residency=detectors.residency,memory=memory_info(manifest["device"]))
                atomic_json(out/"detector_status.json",status);print("DETECT",json.dumps(status),flush=True)
            if stop_batches is not None and offset//manifest["batch"]+1>=stop_batches:break
    finally:
        cap.release();db.close()
        if detectors is not None:detectors.close()


class SceneDiagnostics:
    """Unlabelled discontinuity and repeated-appearance hints, never live/replay truth."""
    def __init__(self):
        self.gray=None;self.hist=None;self.hash=None;self.seen={};self.match=None;self.repeated=0

    def update(self, image, frame0, people):
        small=cv2.resize(image,(160,90));gray=cv2.cvtColor(small,cv2.COLOR_BGR2GRAY)
        hsv=cv2.cvtColor(small,cv2.COLOR_BGR2HSV)
        hist=cv2.calcHist([hsv],[0,1],None,[18,8],[0,180,0,256]);cv2.normalize(hist,hist,alpha=1,norm_type=cv2.NORM_L1)
        dct=cv2.dct(cv2.resize(gray,(32,32)).astype(np.float32))[:8,:8].ravel()[1:]
        bits=dct>np.median(dct);phash=int("".join("1" if x else "0" for x in bits),2)
        hist_delta=0. if self.hist is None else float(cv2.compareHist(self.hist,hist,cv2.HISTCMP_BHATTACHARYYA))
        gray_delta=0. if self.gray is None else float(np.abs(gray.astype(float)-self.gray).mean()/255)
        hash_delta=0 if self.hash is None else (phash^self.hash).bit_count()
        cut=(hist_delta>.55 and gray_delta>.22) or (gray_delta>.38 and hash_delta>=26)
        repeated=False
        if frame0%25==0:
            earlier=self.seen.get(phash)
            if earlier is not None and frame0-earlier>250:
                self.repeated=self.repeated+1 if self.match is not None and earlier-self.match==25 else 1
                self.match=earlier;repeated=self.repeated>=3
            else:self.repeated=0;self.match=None
            self.seen[phash]=frame0
        grass=float(((hsv[:,:,0]>=25)&(hsv[:,:,0]<=85)&(hsv[:,:,1]>=40)).mean())
        heights=[(p["ymax"]-p["ymin"])/1080 for p in people]
        median=float(np.median(heights)) if heights else None
        view="CLOSEUP_HYPOTHESIS" if median is not None and median>.25 else ("WIDE_HYPOTHESIS" if len(people)>=6 and median<.15 and grass>.35 else "OTHER_UNVERIFIED")
        self.gray,self.hist,self.hash=gray,hist,phash
        return dict(cut_suspect=bool(cut),hist_delta=hist_delta,gray_delta=gray_delta,hash_delta_bits=hash_delta,
                    visual_hash=str(phash),repeated_sequence_suspect=repeated,matched_earlier_frame0=self.match if repeated else None,
                    view=view,grass_fraction=grass,median_person_height_fraction=median,
                    off_field_suspect=grass<.1,play_state="UNVERIFIED",replay_state="UNVERIFIED")


class Engine:
    def __init__(self, u, config):
        self.u=u;self.config=config;self.diagnostics=SceneDiagnostics();self.scene=-1;self.last=None
        self.reset()

    def reset(self):
        self.scene+=1;self.state=legacy.new_state(self.u)
        self.temporal=TemporalClassifier(self.config["temporal"])
        self.classifier=UniformClassifier(self.config)
        self.teams=[self.u["Team"](name="TEAM_A",abbreviation="TMA"),self.u["Team"](name="TEAM_B",abbreviation="TMB")]
        self.match=self.u["Match"](*self.teams,fps=25);self.match.team_possession=None
        self.evidence=legacy.EvidenceRun()

    def __getstate__(self):
        state=self.__dict__.copy();state.pop("u");return state

    def __setstate__(self,state):
        self.__dict__.update(state);self.u=legacy.import_upstream()

    def step(self, image, raw):
        from norfair import Detection
        i=raw["frame0"];diag=self.diagnostics.update(image,i,raw["persons"])
        reset_reason=None
        if self.last is not None and (i!=self.last+1 or diag["cut_suspect"]):
            reset_reason="FRAME_GAP" if i!=self.last+1 else "SHOT_CUT_HEURISTIC";self.reset()
        self.last=i
        cv2.setRNGSeed(i)
        people=self.u["Converter"].DataFrame_to_Detections(pd.DataFrame(raw["persons"]))
        balls=self.u["Converter"].DataFrame_to_Detections(pd.DataFrame(raw["balls"]))
        for d in people+balls:d.data["source_frame0"]=i
        error=None
        try:transform=self.u["update_motion_estimator"](self.state["motion"],people+balls,image)
        except (cv2.error,ValueError,np.linalg.LinAlgError) as exc:
            error=f"{type(exc).__name__}: {str(exc)[:300]}";reset_reason="MOTION_FAILURE";self.reset();transform=None
        pts=self.state["player_tracker"].update(detections=people,coord_transformations=transform)
        bts=self.state["ball_tracker"].update(detections=balls,coord_transformations=transform)
        pdets=self.u["Converter"].TrackedObjects_to_Detections(pts)
        bdets=self.u["Converter"].TrackedObjects_to_Detections(bts)
        # Keep V1 failure visible instead of allowing an empty image crop to abort a full match.
        v1_error=None
        try:pdets=self.state["classifier"].predict_from_detections(pdets,image)
        except (cv2.error,ValueError) as exc:
            v1_error=f"{type(exc).__name__}: {str(exc)[:200]}"
            for d in pdets:d.data["classification"]="OTHER"
        players=self.u["Player"].from_detections(pdets,self.state["teams"])
        ball=self.u["get_main_ball"](bdets)
        match_error=None
        try:v2.safe_match_update(self.u,self.state["match"],players,ball)
        except (AttributeError,TypeError,ValueError) as exc:
            # Upstream pass counting can dereference a None team for OTHER.
            # Preserve already-computed possession state, isolate pass state,
            # and expose every affected frame rather than hiding the failure.
            from soccer.pass_event import PassEvent
            match_error=f"{type(exc).__name__}: {str(exc)[:200]}"
            self.state["match"].pass_event=PassEvent()
        objects=[];tracks=[]
        estimates={int(t.id):t.estimate.tolist() for t in pts}
        for d in pdets:
            age=i-d.data["source_frame0"]
            track=dict(track_id=d.data["id"],points=d.points.tolist(),age=age,estimate=estimates.get(d.data["id"]),
                       legacy_class=d.data.get("classification","OTHER"),source_frame0=i,scene_id=self.scene)
            tracks.append(track)
            others=[x.points for x in pdets if x.data["id"]!=d.data["id"] and x.data["source_frame0"]==i]
            appearance=self.classifier.classify(image,d.points,others)
            decision=self.temporal.update(d.data["id"],i,appearance,fresh=age==0)
            objects.append({**track,**decision,"bbox_height":float(d.points[1,1]-d.points[0,1]),
                            "bbox_width":float(d.points[1,0]-d.points[0,0])})
        unique=None if len(balls)!=1 else balls[0].points
        candidate,status,reason=select_candidate(objects,unique,ball_count=len(balls))
        known=self.evidence.update(None if candidate is None else candidate["team"])
        if candidate is not None:status="PROXIMITY_20_FRAME_EVIDENCE" if known else "PROXIMITY_PENDING"
        nearest=None
        if unique is not None:
            center=unique.mean(axis=0)
            def distance(o):
                (x1,_),(x2,y2)=np.array(o["points"])
                return min(np.linalg.norm(center-[x1,y2]),np.linalg.norm(center-[x2,y2]))
            fresh=[o for o in objects if o["age"]==0]
            nearest=min(fresh,key=distance) if fresh else None
        old_candidate=None
        if nearest is not None and distance(nearest)<=45 and nearest["legacy_class"] in ("BLUE","LIGHT"):
            old_candidate=nearest["track_id"]
        valid=v2.objects_to_players(self.u,objects,self.teams)
        tracked_ball=self.u["Ball"](None)
        ball_age=None if ball.detection is None else i-ball.detection.data["source_frame0"]
        if ball_age==0 and len(balls)==1:tracked_ball=ball
        v2.safe_match_update(self.u,self.match,valid,tracked_ball)
        row=dict(frame0=i,scene_id=self.scene,reset_reason=reset_reason,diagnostics=diag,status="ok",
                 raw_people=len(people),objects=len(objects),v1_classes=dict(Counter(o["legacy_class"] for o in objects)),
                 v2_classes=dict(Counter(o["label"] for o in objects)),unknown_reasons=dict(Counter(o["reason"] for o in objects if o["label"]=="UNKNOWN")),
                 ball_count=len(balls),ball_confidences=[float(d.data["p"]) for d in balls],ball_tracks=len(bdets),main_ball_age=ball_age,
                 candidate_id=None if candidate is None else candidate["track_id"],candidate_team=None if candidate is None else candidate["team"],
                 v1_candidate_id=old_candidate,evidence_status=status,evidence_reason=reason,fresh_run=self.evidence.count,
                 referees=sum(o["role"]=="REFEREE" for o in objects),eligible_players=len(valid),
                 referee_nearest_excluded=bool(nearest is not None and nearest["role"]=="REFEREE" and distance(nearest)<=45),
                 v1_owner=self.state["match"].team_possession.name,v1_blue_frames=self.state["teams"][0].possession,
                 v1_light_frames=self.state["teams"][1].possession,v2_history_owner=None if self.match.team_possession is None else self.match.team_possession.name,
                 motion_error=error,v1_classification_error=v1_error,v1_match_error=match_error)
        tracking=dict(frame0=i,scene_id=self.scene,players=tracks,balls=[dict(points=d.points.tolist(),track_id=d.data["id"],age=i-d.data["source_frame0"]) for d in bdets])
        return row,tracking,dict(frame0=i,scene_id=self.scene,objects=objects)


def render(image, row, obj, raw):
    left=cv2.resize(image,(960,540));right=left.copy()
    colors={"TEAM_A":(255,140,80),"TEAM_B":(235,160,245),"REFEREE":(0,255,255),"UNKNOWN":(155,155,155)}
    for o in obj["objects"]:
        det=SimpleNamespace(points=np.array(o["points"]))
        old_color={"BLUE":(255,140,80),"LIGHT":(220,220,255),"OTHER":(150,150,150)}[o["legacy_class"]]
        legacy.box(left,det,old_color,f'{o["legacy_class"]} #{o["track_id"]}',bool(o["age"]))
        legacy.box(right,det,colors[o["label"]],f'{o["label"]} | ID:{o["track_id"]}'+(" EXCLUDED" if o["role"]=="REFEREE" else ""),bool(o["age"]))
    for b in raw["balls"]:
        det=SimpleNamespace(points=np.array([[b["xmin"],b["ymin"]],[b["xmax"],b["ymax"]]]))
        for panel in (left,right):
            legacy.box(panel,det,(80,255,80),f'ball? {b["confidence"]:.2f}')
            cv2.circle(panel,tuple(np.rint(det.points.mean(axis=0)*.5).astype(int)),12,(80,255,80),1)
    canvas=np.full((HEIGHT,WIDTH,3),22,np.uint8);canvas[45:585,:960]=left;canvas[45:585,960:]=right
    texts=[("V1: BLUE / LIGHT / OTHER (shared detector)",16,30),("V2: FROZEN HSV / ROLE / UNKNOWN",976,30),
           (f'Frame {row["frame0"]} | file {row["frame0"]/25:.2f}s | scene {row["scene_id"]} | {row["diagnostics"]["view"]}',16,620),
           (f'V1 historical owner {row["v1_owner"]}: BLUE {row["v1_blue_frames"]/25:.2f}s / LIGHT {row["v1_light_frames"]/25:.2f}s | resets at discontinuities',16,657),
           (f'Current V2: {row["evidence_status"]} | candidate {row["candidate_id"]} / {row["candidate_team"]} | run {row["fresh_run"]}/20',16,694),
           (f'Reason {row["evidence_reason"]} | ball candidates {row["ball_count"]}, main age {row["main_ball_age"]} | REFEREE excluded {row["referees"]}',16,731),
           (f'Classes {row["v2_classes"]} | reset {row["reset_reason"]} | errors: motion {bool(row["motion_error"])} / V1 {bool(row["v1_classification_error"])}',16,768),
           (f'Live play: UNVERIFIED | Replay: UNVERIFIED | repeated appearance suspect {row["diagnostics"]["repeated_sequence_suspect"]}',16,805),
           ("HSV support is not correctness probability. REFEREE is an appearance hypothesis; staff confusion remains possible.",16,847),
           ("Local diagnostic only: no labels, accuracy scores, Smart Tagging, tactical data or confirmed controlled possession.",16,894)]
    for text,x,y in texts:legacy.caption(canvas,text,(x,y),scale=.59 if y>760 else .63)
    return canvas


def decode_count(path):
    cap=cv2.VideoCapture(str(path));count=0
    while cap.read()[0]:count+=1
    fps=cap.get(5);size=(int(cap.get(3)),int(cap.get(4)));cap.release()
    return count,fps,size


def recover_chunks(out, db):
    registered={s for s, in db.execute("SELECT start FROM chunks")}
    for path in sorted((out/"chunks").glob("[0-9]*")):
        meta=json.loads((path/"chunk.json").read_text())
        if meta["start"] in registered:continue
        if legacy.digest(path/"state.pkl")!=meta["checkpoint_sha256"]:raise ValueError("Orphan chunk checkpoint corruption")
        with gzip.open(path/"frames.jsonl.gz","rt",encoding="utf-8") as f:rows=[json.loads(line) for line in f]
        if [r["frame0"] for r in rows]!=list(range(meta["start"],meta["end"])):raise ValueError("Orphan chunk has frame gap")
        with db:
            db.executemany("INSERT INTO results VALUES (?,?)",[(r["frame0"],pack(r)) for r in rows])
            db.execute("INSERT INTO chunks VALUES (?,?,?,?)",(meta["start"],meta["end"],str(path.relative_to(out)),meta["checkpoint_sha256"]))


def process(out, manifest, chunk_size=1000, follow=False, stop_chunks=None):
    u=legacy.import_upstream();config=json.loads(CONFIG.read_text());torch.set_num_threads(2);cv2.setNumThreads(1)
    install_exact_cpu_adapters()
    logging.getLogger().setLevel(logging.ERROR)
    chunks=out/"chunks";chunks.mkdir(exist_ok=True);db=connection(out);recover_chunks(out,db)
    latest=db.execute("SELECT end,path,sha256 FROM chunks ORDER BY end DESC LIMIT 1").fetchone()
    if latest:
        cursor=latest[0]
        if legacy.digest(out/latest[1]/"state.pkl")!=latest[2]:raise ValueError("Checkpoint hash mismatch")
        with (out/latest[1]/"state.pkl").open("rb") as f:engine=pickle.load(f)
    else:cursor=manifest["start"];engine=Engine(u,config)
    cap=cv2.VideoCapture(str(VIDEO));cap.set(cv2.CAP_PROP_POS_FRAMES,cursor);committed=0;t0=time.monotonic()
    try:
        while cursor<manifest["end"]:
            end=min(cursor+chunk_size,manifest["end"])
            records=cached_records(out,cursor,end)
            if [i for i,_ in records]!=list(range(cursor,end)):
                if follow:
                    if (out/"detect_failure.json").exists():raise RuntimeError("Detector worker failed; see detect_failure.json")
                    time.sleep(2);continue
                raise RuntimeError(f"Detector cache incomplete for [{cursor},{end})")
            path=chunks/f"{cursor:09d}-{end:09d}"
            staging=chunks/f".work-{cursor:09d}"
            if staging.exists():
                orphan=out/"interrupted";orphan.mkdir(exist_ok=True)
                staging.rename(orphan/f'{staging.name}-{time.time_ns()}')
            staging.mkdir()
            logs={name:gzip.open(staging/f"{name}.jsonl.gz","wt",compresslevel=1,encoding="utf-8") for name in ["frames","tracking","objects"]}
            import imageio_ffmpeg
            cmd=[imageio_ffmpeg.get_ffmpeg_exe(),"-nostdin","-v","error","-n","-f","rawvideo","-pix_fmt","bgr24","-s",f"{WIDTH}x{HEIGHT}","-r","25","-i","pipe:0","-an","-c:v","libx264","-preset","veryfast","-crf","21","-pix_fmt","yuv420p","-movflags","+faststart",str(staging/"comparison.mp4")]
            err=(staging/"encoder.log").open("w");encoder=subprocess.Popen(cmd,stdin=subprocess.PIPE,stderr=err)
            rows=[]
            try:
                for i,raw in records:
                    ok,image=cap.read()
                    if not ok or abs(cap.get(cv2.CAP_PROP_POS_FRAMES)-(i+1))>.1:
                        raise RuntimeError(f"Source exact decode failed at frame {i}; chunk left uncommitted")
                    if raw["status"]!="ok":raise RuntimeError(f"Detector failure at frame {i}: {raw['status']}")
                    row,tracking,objects=engine.step(image,raw)
                    for name,value in [("frames",row),("tracking",tracking),("objects",objects)]:logs[name].write(json.dumps(value,separators=(",",":"))+"\n")
                    encoder.stdin.write(render(image,row,objects,raw).tobytes());rows.append(row)
            finally:
                try:encoder.stdin.close()
                except (BrokenPipeError,OSError):pass
                encoder.wait();err.close()
                for f in logs.values():f.close()
            if encoder.returncode:raise RuntimeError("Chunk encoder failed")
            decoded,fps,size=decode_count(staging/"comparison.mp4")
            if (decoded,fps,size)!=(end-cursor,25.,(WIDTH,HEIGHT)):raise RuntimeError("Chunk video verification failed")
            with (staging/"state.pkl").open("xb") as f:pickle.dump(engine,f,protocol=5);f.flush();os.fsync(f.fileno())
            meta=dict(start=cursor,end=end,decoded_frames=decoded,fps=fps,checkpoint_sha256=legacy.digest(staging/"state.pkl"),runner_sha256=legacy.digest(Path(__file__)),
                      video_sha256=legacy.digest(staging/"comparison.mp4"))
            atomic_json(staging/"chunk.json",meta);staging.rename(path)
            with db:
                db.executemany("INSERT INTO results VALUES (?,?)",[(r["frame0"],pack(r)) for r in rows])
                db.execute("INSERT INTO chunks VALUES (?,?,?,?)",(cursor,end,str(path.relative_to(out)),meta["checkpoint_sha256"]))
            cursor=end;committed+=1
            atomic_json(out/"process_status.json",dict(complete=cursor==manifest["end"],next_frame0=cursor,
                        committed_frames=cursor-manifest["start"],session_seconds=time.monotonic()-t0,last_scene_id=engine.scene))
            print("PROCESS",cursor,"/",manifest["end"],"scene",engine.scene,flush=True)
            if stop_chunks is not None and committed>=stop_chunks:break
    finally:cap.release();db.close()


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument("--stage",choices=["detect","process"],required=True)
    p.add_argument("--output",type=Path,default=OUT/"main")
    p.add_argument("--start",type=int,default=0);p.add_argument("--end",type=int,default=TOTAL)
    p.add_argument("--device",default=None,help="auto, cpu, cuda, cuda:0 or mps; Mac default remains mps")
    p.add_argument("--batch",type=int,default=None)
    p.add_argument("--model-residency",choices=["auto","single","both"],default="auto")
    p.add_argument("--reuse-750-cache",action="store_true",help="Opt in to a local cache; omit for a real inference benchmark")
    p.add_argument("--chunk-size",type=int,default=1000);p.add_argument("--follow",action="store_true")
    p.add_argument("--stop-batches",type=int);p.add_argument("--stop-chunks",type=int)
    args=p.parse_args()
    if os.environ.get("PYTHONHASHSEED")!="0":p.error("Use PYTHONHASHSEED=0 for repeatable temporal ties and resume")
    out=args.output.resolve()
    if not out.is_relative_to(OUT.resolve()):p.error("Output must remain in full_match_v2")
    out.mkdir(parents=True,exist_ok=True)
    try:
        device=resolve_device(args.device or ("mps" if platform.system()=="Darwin" else "auto"))
    except Exception as exc:
        log_event(out/"runtime_events.jsonl","device_selection_failed",error=str(exc))
        raise
    batch=args.batch if args.batch is not None else (16 if device=="mps" else 1)
    if not 0<=args.start<args.end<=TOTAL or batch<1 or args.chunk_size<1:p.error("Invalid frame range or chunk/batch size")
    manifest=initialize(out,args.start,args.end,batch,device,args.model_residency,args.reuse_750_cache)
    with stage_lock(out,args.stage):
        failure=out/f"{args.stage}_failure.json"
        if failure.exists():failure.rename(out/f'{args.stage}_failure_{time.time_ns()}.json')
        try:
            if args.stage=="detect":detect(out,manifest,args.stop_batches)
            else:process(out,manifest,args.chunk_size,args.follow,args.stop_chunks)
        except KeyboardInterrupt:
            atomic_json(out/f"{args.stage}_interrupted.json",dict(stage=args.stage,status="interrupted",resume="Repeat the same command and output directory"))
            raise SystemExit(130)
        except Exception as exc:
            log_event(out/"runtime_events.jsonl","stage_failed",stage=args.stage,error=f"{type(exc).__name__}: {exc}")
            atomic_json(failure,dict(stage=args.stage,type=type(exc).__name__,error=str(exc),target_range=[args.start,args.end-1],
                                    resume="Current detector batch / processing chunk is uncommitted; repeat the same command after fixing the cause"))
            raise


if __name__=="__main__":main()
