"""Small cached-video V3/V4 diagnostic. No inference, training or tracker update.

At most 5,000 frames: whole-match use requires a future streaming/checkpoint runner.
Outputs are exclusive new directories; V3 and all source caches remain unchanged.
"""
import argparse
from collections import Counter, defaultdict
import csv
import json
import os
from pathlib import Path
import subprocess
import sys
import tempfile

import cv2
import numpy as np

from run_tryolabs_classification_v3 import FrameReader, VIDEO, EvidenceRun, caption, draw
from tryolabs_uniform_v2 import crop_region, select_candidate
from tryolabs_uniform_v3 import UniformClassifierV3, TemporalClassifierV3
from tryolabs_uniform_v4 import UniformClassifierV4, TemporalClassifierV4, select_candidate_v4, LABELS, VERSION, assign
from tryolabs_v4_context import frame_context, CameraTransitionMonitor
from tryolabs_v4_data import ROOT, ASSETS, OUTPUT, digest, write_json, full_frames, protected_snapshot, verify_preserved

V3_CONFIG=ROOT/"starter/configs/tryolabs_classification_v3_hsv.json"
V3_POLICY=ROOT/"starter/configs/tryolabs_classification_v3_policy.json"
V4_CONFIG=ROOT/"starter/configs/tryolabs_classification_v4_hsv.json"
V4_POLICY=ROOT/"starter/configs/tryolabs_classification_v4_policy.json"
PREPARED=ASSETS/"classification_v3/validated/input_frames.jsonl"
COLORS={"TEAM_A":(255,140,80),"TEAM_B":(235,160,245),"GK_A":(30,140,255),"GK_B":(235,220,50),
        "REFEREE":(0,255,255),"STAFF":(160,220,180),"OTHER":(240,240,240),"UNKNOWN":(155,155,155)}
os.environ.setdefault("MPLCONFIGDIR",str(Path(tempfile.gettempdir())/"mtah-tryolabs-v4-mpl"))


def read_prepared(path, maximum=5000):
    frames=[]
    with Path(path).open(encoding="utf-8") as f:
        for line in f:
            if not line.strip():continue
            if len(frames)>=maximum:
                raise ValueError("Prepared input exceeds small-run limit; whole-match streaming/resume is not implemented")
            frames.append(json.loads(line))
    return frames


def changes(objects, field):
    previous={};pairs=Counter()
    for o in objects:
        key=o["sequence"],o["track_id"],o.get("v4_epoch") if field in ("final_label","role") else None;old=previous.get(key)
        if old and old[0]+1==o["source_frame0"] and old[1]!=o[field]:pairs[f'{old[1]}->{o[field]}']+=1
        previous[key]=o["source_frame0"],o[field]
    return dict(total=sum(pairs.values()),pairs=dict(pairs))


def comparison(image, objects, raw, row):
    left=cv2.resize(image,(960,540));right=left.copy()
    for o in objects:
        draw(left,o["points"],o["v3_label"],o["track_id"],COLORS[o["v3_label"]])
        draw(right,o["points"],o["final_label"],o["track_id"],COLORS[o["final_label"]])
    for ball in raw["balls"]:
        points=[[ball["xmin"],ball["ymin"]],[ball["xmax"],ball["ymax"]]]
        for panel in [left,right]:draw(panel,points,f'ball? {ball["confidence"]:.2f}',"-",(80,255,80))
    canvas=np.full((800,1920,3),22,np.uint8);canvas[42:582,:960]=left;canvas[42:582,960:]=right
    caption(canvas,"V3 FROZEN BASELINE",(15,28),.7)
    caption(canvas,"V4: SAME DETECTIONS / TRACKS; APPEARANCE + ROLE EVIDENCE",(975,28),.55)
    lines=[f'{raw["sequence"]} | file frame {raw["frame0"]} / {raw["frame0"]/25:.2f}s | live/replay: UNVERIFIED',
           f'V3 candidate {row["v3_candidate_id"]}: {row["v3_status"]}',
           f'V4 candidate {row["v4_candidate_id"]}: {row["v4_status"]} | GK body-near / foot-far {row["gk_body_near_foot_far"]}',
           f'V4 {dict(Counter(o["final_label"] for o in objects))}',
           'Goal geometry is weak; GK_A kit disabled; no reliable staff-zone or glove evidence.',
           'REF/STAFF/OTHER/UNKNOWN excluded. GK allowed. Unchanged 45px foot corners / 20 team frames.']
    for i,line in enumerate(lines):caption(canvas,line,(15,614+32*i),.58)
    return canvas


def visuals(out, objects, samples):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    fig,axes=plt.subplots(8,3,figsize=(13,20))
    for row,label in enumerate(LABELS):
        pp=np.concatenate(samples[label]) if samples[label] else np.zeros((0,3))
        for col,(name,high) in enumerate([("H",180),("S",256),("V",256)]):
            ax=axes[row,col]
            if len(pp):ax.hist(pp[:,col],40,range=(0,high),density=True)
            ax.set_title(f'{label} / {name}: model group, not truth');ax.grid(alpha=.2)
    fig.tight_layout();fig.savefig(out/"hsv_distributions.png",dpi=130);plt.close(fig)
    grouped=defaultdict(list)
    for o in objects:
        if o["age"]==0:grouped[o["sequence"],o["track_id"]].append(o)
    selected=[]
    for key,rr in grouped.items():
        if any(o["goalkeeper_candidates"] or o["final_label"] in ("OTHER","STAFF")
               or o["appearance_label"]=="UNRESOLVED_ORANGE_GREEN_SEARCH" for o in rr):selected.append((key,rr))
    selected.sort(key=lambda pair:(not any(o["goalkeeper_candidates"] for o in pair[1]),pair[0]))
    folder=out/"contact_sheets";folder.mkdir();reader=FrameReader(VIDEO);strips=[];index=[]
    try:
        for key,rr in selected[:80]:
            strip=np.full((230,1320,3),22,np.uint8)
            for column,o in enumerate([rr[0],rr[len(rr)//2],rr[-1]]):
                im=reader.read(o["source_frame0"]);thumb=cv2.resize(im,(320,180))
                pp=np.rint(np.asarray(o["points"])*[320/im.shape[1],180/im.shape[0]]).astype(int)
                cv2.rectangle(thumb,tuple(pp[0]),tuple(pp[1]),(0,255,255),1)
                strip[20:200,column*440:column*440+320]=thumb
                roi,_=crop_region(im,o["points"])
                if roi.size:strip[20:180,column*440+330:column*440+430]=cv2.resize(roi,(100,160),interpolation=cv2.INTER_NEAREST)
                caption(strip,f'{key} f{o["source_frame0"]}',(column*440+3,14),.36)
                caption(strip,f'{o["v3_label"]} -> {o["final_label"]}',(column*440+3,213),.37)
                caption(strip,o["reason"],(column*440+3,227),.28)
            name=f'{key[0].replace(":","_")}_id{key[1]}.jpg';cv2.imwrite(str(folder/name),strip);strips.append(strip)
            index.append(dict(sequence=key[0],track_id=key[1],file=name,observations=len(rr),truth_available=False))
        for page,start in enumerate(range(0,len(strips),6)):
            cv2.imwrite(str(out/f"contact_sheet_{page+1:02d}.jpg"),np.concatenate(strips[start:start+6]))
    finally:reader.close()
    write_json(out/"contact_sheet_index.json",dict(selected_tracks=len(selected),exported_tracks=len(index),tracks=index))


def run(frames,out,video=True,make_visuals=True):
    old=json.loads(V3_CONFIG.read_text(encoding="utf-8"));oldpolicy=json.loads(V3_POLICY.read_text(encoding="utf-8"))
    config=json.loads(V4_CONFIG.read_text(encoding="utf-8"));policy=json.loads(V4_POLICY.read_text(encoding="utf-8"))
    v3=UniformClassifierV3(old,oldpolicy);v4=UniformClassifierV4(config,policy,old,oldpolicy)
    reader=FrameReader(VIDEO);previous=None;objects=[];rows=[];samples=defaultdict(list);rng=np.random.default_rng(4);epoch=0
    encoder=log=None
    if video:
        import imageio_ffmpeg
        log=(out/"encoder.log").open("x")
        cmd=[imageio_ffmpeg.get_ffmpeg_exe(),"-nostdin","-v","error","-n","-f","rawvideo","-pix_fmt","bgr24","-s","1920x800","-r","25","-i","pipe:0","-an","-c:v","libx264","-preset","veryfast","-crf","21","-pix_fmt","yuv420p","-movflags","+faststart",str(out/"comparison_v3_v4.mp4")]
        encoder=subprocess.Popen(cmd,stdin=subprocess.PIPE,stderr=log)
    try:
        with (out/"input_frames.jsonl").open("x",encoding="utf-8") as f:
            for raw in frames:
                i=raw["frame0"];boundary=previous is None or raw["sequence"]!=previous[0] or i!=previous[1]+1
                if boundary:
                    t3=TemporalClassifierV3(oldpolicy);t4=TemporalClassifierV4(policy,oldpolicy);e3=EvidenceRun();e4=EvidenceRun()
                    monitor=CameraTransitionMonitor(policy["scene"]);epoch+=1
                previous=raw["sequence"],i;f.write(json.dumps(raw,separators=(",",":"))+"\n")
                im=reader.read(i);context=frame_context(im,raw["objects"]);current=[]
                transition=monitor.update(im)
                if transition:
                    t4=TemporalClassifierV4(policy,oldpolicy);e4=EvidenceRun();epoch+=1
                for obj in raw["objects"]:
                    others=[o["points"] for o in raw["objects"] if o["track_id"]!=obj["track_id"] and o["age"]==0]
                    b3=v3.classify(im,obj["points"],others);r3=t3.update(obj["track_id"],i,b3,fresh=obj["age"]==0)
                    obs=v4.classify(im,obj["points"],others,context,b3);r4=t4.update(obj["track_id"],i,obs,fresh=obj["age"]==0)
                    if transition:r4=assign(r4,"UNKNOWN",transition)
                    r4.pop("v3_observation",None)
                    record=dict(obj,**r4,sequence=raw["sequence"],source_frame0=i,v3_label=r3["label"],v3_role=r3["role"],
                                v3_team=r3["team"],v3_eligible=r3["eligible"],v3_reason=r3["reason"],
                                snapshot_origin=raw["source"],v3_components=b3.get("components",{}),v4_epoch=epoch)
                    current.append(record);objects.append(record)
                    if make_visuals and obj["age"]==0:
                        roi,_=crop_region(im,obj["points"])
                        if roi.size:
                            pp=cv2.cvtColor(roi,cv2.COLOR_BGR2HSV).reshape(-1,3)
                            samples[record["final_label"]].append(pp[rng.choice(len(pp),min(32,len(pp)),replace=False)])
                balls=raw["balls"];bp=[[balls[0]["xmin"],balls[0]["ymin"]],[balls[0]["xmax"],balls[0]["ymax"]]] if len(balls)==1 else None
                oo=[dict(o,role=o["v3_role"],team=o["v3_team"],eligible=o["v3_eligible"]) for o in current]
                c3,s3,w3=select_candidate(oo,bp,ball_count=len(balls));c4,s4,w4=select_candidate_v4(current,bp,ball_count=len(balls))
                team3=e3.update(c3["team"] if c3 else None);team4=e4.update(c4["team"] if c4 else None)
                if c3:s3="PROXIMITY_20_FRAME_EVIDENCE" if team3 else "PROXIMITY_PENDING"
                if c4:s4="PROXIMITY_20_FRAME_EVIDENCE" if team4 else "PROXIMITY_PENDING"
                handgap=0
                if bp is not None:
                    center=np.mean(bp,axis=0)
                    for o in current:
                        if o["role"]!="GOALKEEPER":continue
                        (x1,y1),(x2,y2)=o["points"]
                        foot=min(np.linalg.norm(center-[x1,y2]),np.linalg.norm(center-[x2,y2]))
                        handgap+=bool(x1<=center[0]<=x2 and y1<=center[1]<=y2 and foot>45)
                row=dict(sequence=raw["sequence"],frame0=i,v3_candidate_id=c3["track_id"] if c3 else None,v4_candidate_id=c4["track_id"] if c4 else None,
                    v3_status=s3,v4_status=s4,v3_reason=w3,v4_reason=w4,ball_candidates=len(balls),gk_body_near_foot_far=handgap,
                    field_view_supported=context["field_view_supported"],broadcast_state="UNVERIFIED",pitch_geometry_verified=False,
                    v4_epoch=epoch,camera_transition_hint=transition)
                rows.append(row)
                if encoder:encoder.stdin.write(comparison(im,current,raw,row).tobytes())
                if len(rows)%125==0:print("V4_POSTPROCESS",len(rows),i,dict(Counter(o["final_label"] for o in current)),flush=True)
    finally:
        reader.close()
        if encoder:
            try:encoder.stdin.close()
            except BrokenPipeError:pass
            encoder.wait();log.close()
    if encoder and encoder.returncode:raise RuntimeError("Video encoder failed")
    with (out/"objects.jsonl").open("x",encoding="utf-8") as f:
        for o in objects:f.write(json.dumps(o,separators=(",",":"))+"\n")
    fields=["sequence","source_frame0","track_id","v3_label","v3_role","appearance_label","final_label","team","role","eligible","reason","age","classification_confidence","goal_geometry_hint","field_view_supported","other_observable","broadcast_state"]
    with (out/"objects.csv").open("x",encoding="utf-8",newline="") as f:
        w=csv.DictWriter(f,fieldnames=fields,extrasaction="ignore")
        w.writeheader();w.writerows(objects)
    with (out/"frames.csv").open("x",encoding="utf-8",newline="") as f:
        w=csv.DictWriter(f,fieldnames=list(rows[0]));w.writeheader();w.writerows(rows)
    decoded=None
    if video:
        cap=cv2.VideoCapture(str(out/"comparison_v3_v4.mp4"));decoded=0
        while cap.read()[0]:decoded+=1
        fps=cap.get(5);cap.release()
        if decoded!=len(rows) or fps!=25:raise RuntimeError("Video verification failed")
    summary=dict(frames=len(rows),objects=len(objects),v3_counts=dict(Counter(o["v3_label"] for o in objects)),
        v4_counts=dict(Counter(o["final_label"] for o in objects)),v3_class_changes=changes(objects,"v3_label"),
        v4_class_changes=changes(objects,"final_label"),v3_role_changes=changes(objects,"v3_role"),v4_role_changes=changes(objects,"role"),
        v3_unknown_fraction=sum(o["v3_label"]=="UNKNOWN" for o in objects)/max(1,len(objects)),
        v4_unknown_fraction=sum(o["final_label"]=="UNKNOWN" for o in objects)/max(1,len(objects)),
        v4_unknown_reasons=dict(Counter(o["reason"] for o in objects if o["final_label"]=="UNKNOWN")),
        transition_table={f'{a}->{b}':sum(o["v3_label"]==a and o["final_label"]==b for o in objects) for a in ["TEAM_A","TEAM_B","REFEREE","UNKNOWN"] for b in LABELS},
        retention={a:dict(v3_observations=n,same_v4=sum(o["v3_label"]==a and o["final_label"]==a for o in objects))
            for a in ["TEAM_A","TEAM_B","REFEREE"] if (n:=sum(o["v3_label"]==a for o in objects))},
        goalkeeper_candidate_observations=sum(bool(o["goalkeeper_candidates"]) for o in objects),
        goalkeeper_candidate_other=sum(bool(o["goalkeeper_candidates"]) and o["final_label"]=="OTHER" for o in objects),
        v3_referee_to_goalkeeper=sum(o["v3_label"]=="REFEREE" and o["role"]=="GOALKEEPER" for o in objects),
        verified_staff_context_player_assignments=sum(o["staff_context_supported"] and o["role"]=="PLAYER" for o in objects),
        actual_staff_player_errors=None,manual_ground_truth=False,accuracy_improvement_proven=False,
        candidate_different_frames=sum(r["v3_candidate_id"]!=r["v4_candidate_id"] for r in rows),
        v3_possession_statuses=dict(Counter(r["v3_status"] for r in rows)),v4_possession_statuses=dict(Counter(r["v4_status"] for r in rows)),
        gk_body_near_foot_far_frames=sum(r["gk_body_near_foot_far"]>0 for r in rows),
        decoded_video_frames=decoded,detector_calls=0,tracker_updates=0,staff_context_available=False,
        live_replay_state_verified=False,goalkeeper_person_identity_verified=False,
        camera_transition_recheck_frames=sum(bool(r["camera_transition_hint"]) for r in rows))
    summary["source_breakdown"]={source:dict(objects=len(oo),v3_counts=dict(Counter(o["v3_label"] for o in oo)),v4_counts=dict(Counter(o["final_label"] for o in oo)))
        for source in sorted({o["snapshot_origin"] for o in objects}) if (oo:=[o for o in objects if o["snapshot_origin"]==source])}
    write_json(out/"summary.json",summary)
    if make_visuals:visuals(out,objects,samples)
    return summary


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument("--source",choices=["prepared","full"],default="prepared")
    p.add_argument("--cache",type=Path);p.add_argument("--starts",type=float,nargs="+",default=[600,1800,3600])
    p.add_argument("--seconds",type=float,default=10);p.add_argument("--extra-cache",type=Path)
    p.add_argument("--extra-start",type=int,default=5000);p.add_argument("--extra-frames",type=int,default=250)
    p.add_argument("--output",type=Path,default=OUTPUT/"validation")
    p.add_argument("--no-video",action="store_true");p.add_argument("--no-visuals",action="store_true")
    a=p.parse_args();out=a.output.resolve()
    if not out.is_relative_to(OUTPUT.resolve()):p.error("Use a new output under classification_v4")
    if not np.isfinite(a.seconds) or a.seconds<=0 or any(not np.isfinite(s) or s<0 for s in a.starts):p.error("Invalid interval")
    if a.extra_start<0 or not 0<=a.extra_frames<=5000:p.error("Extra range must contain at most 5000 frames")
    out.mkdir(parents=True,exist_ok=False)
    before=protected_snapshot();write_json(out/"preservation_before.json",before);stat=VIDEO.stat()
    try:
        if a.source=="prepared":
            source=a.cache or PREPARED;frames=read_prepared(source)
        else:
            if a.cache is None:p.error("--cache must contain V2 run.sqlite and tracking chunks")
            intervals=[(round(s*25),round((s+a.seconds)*25)) for s in a.starts]
            if sum(b-a for a,b in intervals)>5000:raise ValueError("Small-run limit 5000: whole-match streaming/resume is not implemented")
            frames=list(full_frames(a.cache,intervals,out/"source_snapshot"))
            if {r["frame0"] for r in frames}!={i for a,b in intervals for i in range(a,b)}:raise ValueError("Requested completed cache frames are missing; no inference was attempted")
        if a.extra_cache:
            if len(frames)+a.extra_frames>5000:raise ValueError("Combined inputs exceed small-run limit 5000")
            extra=list(full_frames(a.extra_cache,[(a.extra_start,a.extra_start+a.extra_frames)],out/"extra_snapshot"))
            if len(extra)!=a.extra_frames:raise ValueError("Extra requested cache is incomplete")
            frames+=extra
        if not 0<len(frames)<=5000:raise ValueError("Small-run limit is 1..5000 frames; use a future streaming/resume runner for whole match")
        if len({(r["sequence"],r["frame0"]) for r in frames})!=len(frames):raise ValueError("Duplicate source sequence/frame")
        codes=["tryolabs_uniform_v4.py","run_tryolabs_classification_v4.py","tryolabs_v4_context.py","tryolabs_v4_data.py"]
        manifest=dict(version=VERSION,python=sys.version,opencv=cv2.__version__,numpy=np.__version__,device="cpu_postprocess",
            code_sha256={name:digest(ROOT/"starter"/name) for name in codes},
            config_sha256=digest(V4_CONFIG),policy_sha256=digest(V4_POLICY),v3_config_sha256=digest(V3_CONFIG),v3_policy_sha256=digest(V3_POLICY),
            frames=len(frames),video_size=stat.st_size,video_mtime_ns=stat.st_mtime_ns,detector_calls=0,tracker_updates=0)
        write_json(out/"manifest.json",manifest)
        summary=run(frames,out,not a.no_video,not a.no_visuals)
        verification=verify_preserved(before);now=VIDEO.stat()
        if (stat.st_size,stat.st_mtime_ns)!=(now.st_size,now.st_mtime_ns):raise RuntimeError("Original video changed")
        verification.update(video_size_mtime_unchanged=True,input_sha256=digest(out/"input_frames.jsonl"))
        write_json(out/"verification.json",verification)
        print(json.dumps({k:summary[k] for k in ["frames","objects","v3_counts","v4_counts","candidate_different_frames"]},indent=2))
    except Exception as e:
        write_json(out/"failure.json",dict(error_type=type(e).__name__,message=str(e),detector_calls=0,tracker_updates=0))
        raise


if __name__=="__main__":main()
