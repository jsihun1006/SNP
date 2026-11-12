"""CPU-only V2/V3 comparison on identical cached detections AND Norfair tracks.

Default: frozen 750-frame local experiment. --source full reads Windows full-run
chunks/SQLite by snapshotting the DB inside V3 output; source files are untouched.
No torch, model loader, GPU inference, tracker update or manual labels are used.
"""
import argparse
from collections import Counter,defaultdict,OrderedDict
import csv
import gzip
import hashlib
import json
import os
from pathlib import Path
import subprocess
import tempfile
import sys

import cv2
import numpy as np

from tryolabs_uniform_v2 import UniformClassifier,TemporalClassifier,select_candidate,crop_region
from tryolabs_uniform_v3 import UniformClassifierV3,TemporalClassifierV3,VERSION
from tryolabs_v3_data import ROOT,ASSETS,OUTPUT,digest,write_json,legacy_frames,full_frames,protected_snapshot,validate_preserved

V2_CONFIG=ROOT/"starter/configs/tryolabs_classification_v2_hsv.json"
V3_CONFIG=ROOT/"starter/configs/tryolabs_classification_v3_hsv.json"
V3_POLICY=ROOT/"starter/configs/tryolabs_classification_v3_policy.json"
VIDEO=ROOT/"data/broadcast/BAR-ATH_broadcast-001.mp4"
os.environ.setdefault("MPLCONFIGDIR",str(Path(tempfile.gettempdir())/"mtah-tryolabs-v3-mpl"))


class EvidenceRun:
    def __init__(self):self.team=None;self.count=0
    def update(self,team):
        self.count=self.count+1 if team is not None and team==self.team else int(team is not None)
        self.team=team
        return team if self.count>=20 else None


def caption(image,text,xy,scale=.55,color=(230,230,230)):
    cv2.putText(image,str(text),xy,cv2.FONT_HERSHEY_SIMPLEX,scale,color,1,cv2.LINE_AA)


def draw(image,points,label,track_id,color,scale=.5):
    p=np.asarray(points,dtype=float)
    if p.shape!=(2,2) or not np.isfinite(p).all():return
    p=np.rint(p*scale).astype(int)
    cv2.rectangle(image,tuple(p[0]),tuple(p[1]),color,1)
    caption(image,f'{label} | ID:{track_id}',(max(0,p[0,0]),max(12,p[0,1]-3)),.34,color)


def canvas(image,objects,raw,row):
    left=cv2.resize(image,(960,540));right=left.copy()
    colors={"TEAM_A":(255,140,80),"TEAM_B":(235,160,245),"REFEREE":(0,255,255),"UNKNOWN":(145,145,145)}
    for o in objects:
        draw(left,o["points"],o["v2_label"],o["track_id"],colors[o["v2_label"]])
        draw(right,o["points"],o["label"]+(" EXCLUDED" if o["role"]=="REFEREE" else ""),o["track_id"],colors[o["label"]])
    for b in raw["balls"]:
        p=np.array([[b["xmin"],b["ymin"]],[b["xmax"],b["ymax"]]])
        for panel in [left,right]:draw(panel,p,f'ball? {b["confidence"]:.2f}',"-",(80,255,80))
    out=np.full((800,1920,3),22,np.uint8);out[42:582,:960]=left;out[42:582,960:]=right
    caption(out,"V2: FROZEN HSV / TEMPORAL POLICY",(15,28),.65)
    caption(out,"V3: SAME DETECTIONS / SAME TRACKS / NO GPU REINFERENCE",(975,28),.57)
    messages=[f'{raw["sequence"]} | source frame {raw["frame0"]} | file {raw["frame0"]/25:.2f}s',
              f'V2 {dict(Counter(o["v2_label"] for o in objects))} | candidate {row["v2_candidate_id"]} / {row["v2_status"]}',
              f'V3 {dict(Counter(o["label"] for o in objects))} | candidate {row["v3_candidate_id"]} / {row["v3_status"]}',
              f'REFEREE appearance excluded V2 {row["v2_referees"]}, V3 {row["v3_referees"]} | unchanged ball candidates {len(raw["balls"])}',
              'V3 logs separate observed / stabilized / final label; support scores are not correctness probabilities.',
              'No manual truth, accuracy or confirmed main-referee identity. 45px and 20-frame evidence condition preserved.']
    for i,text in enumerate(messages):caption(out,text,(15,614+i*32),.6 if i<4 else .57)
    return out


class FrameReader:
    def __init__(self,path):self.cap=cv2.VideoCapture(str(path));self.last=-2;self.cache=OrderedDict()
    def read(self,i):
        if i in self.cache:return self.cache[i].copy()
        if i!=self.last+1:self.cap.set(cv2.CAP_PROP_POS_FRAMES,i)
        ok,im=self.cap.read();self.last=i
        if not ok or abs(self.cap.get(cv2.CAP_PROP_POS_FRAMES)-(i+1))>.1:raise ValueError(f"Exact source decode failed at {i}")
        self.cache[i]=im
        while len(self.cache)>8:self.cache.popitem(last=False)
        return im.copy()
    def close(self):self.cap.release()


def count_switches(objects,field):
    previous={};total=direct=0
    for o in objects:
        key=o["sequence"],o["track_id"];old=previous.get(key)
        if old and o["source_frame0"]==old[0]+1 and o[field]!=old[1]:
            total+=1;direct+=set([o[field],old[1]])=={"TEAM_A","TEAM_B"}
        previous[key]=(o["source_frame0"],o[field])
    return dict(all_label_changes=total,direct_team_a_team_b_changes=direct)


def feature_summary(objects,version):
    result={}
    for label in ["TEAM_A","TEAM_B","REFEREE","UNKNOWN"]:
        group=[o for o in objects if o["v2_label"]==label] if version==2 else [o for o in objects if o["label"]==label]
        names=["navy","burgundy","lilac","yellow"] if version==2 else ["navy","burgundy","lavender","pink","yellow"]
        result[label]={name:dict(mean=float(np.mean(values)),q10_q50_q90=np.quantile(values,[.1,.5,.9]).tolist()) for name in names
                       if (values:=[(o["v2_components"] if version==2 else o.get("components",{})).get(name,0.) for o in group])}
    return result


def track_statistics(out,objects):
    grouped=defaultdict(list)
    for o in objects:
        if o["age"]==0:grouped[o["sequence"],o["track_id"]].append(o)
    statistics=[]
    for (sequence,track_id),rr in grouped.items():
        counts=Counter(o["v2_label"] for o in rr if o["v2_label"]!="UNKNOWN")
        cohort="UNCERTAIN"
        if counts:
            label,count=counts.most_common(1)[0]
            if count>=3 and count/sum(counts.values())>=.65:cohort=label
        navy=[o["v2_components"].get("navy",0)>=.025 for o in rr];burgundy=[o["v2_components"].get("burgundy",0)>=.025 for o in rr]
        statistics.append(dict(sequence=sequence,track_id=track_id,automatic_v2_track_cohort=cohort,
            fresh_observations=len(rr),first_frame0=rr[0]["source_frame0"],last_frame0=rr[-1]["source_frame0"],
            v2_navy_present_fraction=float(np.mean(navy)),v2_burgundy_present_fraction=float(np.mean(burgundy)),
            v2_dual_color_present_fraction=float(np.mean(np.array(navy)&burgundy)),
            v3_lavender_median=float(np.median([o.get("components",{}).get("lavender",0) for o in rr])),
            v3_pink_median=float(np.median([o.get("components",{}).get("pink",0) for o in rr])),
            v3_shared_color_mean=float(np.mean([o.get("shared_color_fraction",0) for o in rr]))))
    if statistics:
        with (out/"track_color_stability.csv").open("x",encoding="utf-8",newline="") as f:
            w=csv.DictWriter(f,fieldnames=list(statistics[0]));w.writeheader();w.writerows(statistics)
    return statistics


def make_visuals(out,objects,samples):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    fig,axes=plt.subplots(4,3,figsize=(13,11))
    for row,label in enumerate(["TEAM_A","TEAM_B","REFEREE","UNKNOWN"]):
        pp=np.concatenate(samples[label]) if samples[label] else np.zeros((0,3))
        for column,(name,high) in enumerate([("Hue",180),("Saturation",256),("Value",256)]):
            ax=axes[row,column]
            if len(pp):ax.hist(pp[:,column],bins=40,range=(0,high),density=True,color=["#3d68b7","#b58bdd","#b2bb20","#777777"][row])
            ax.set_title(f'{label}: {name} (model group, not truth)');ax.grid(alpha=.15)
    fig.tight_layout();fig.savefig(out/"hsv_distributions.png",dpi=140);plt.close(fig)
    known=[o for o in objects if o.get("features_valid")]
    fig,ax=plt.subplots(figsize=(8,6))
    if known:
        points=np.array([[o["components"]["burgundy"],o["components"]["lavender"]] for o in known])
        ax.scatter(points[::4,0],points[::4,1],s=4,alpha=.3)
    ax.set(xlabel="Burgundy pixel fraction",ylabel="Lavender pixel fraction",title="V3 component coexistence (unlabelled)")
    ax.grid(alpha=.2);fig.tight_layout();fig.savefig(out/"burgundy_lavender_overlap.png",dpi=150);plt.close(fig)


def diagnostic_images(out,objects):
    folder=out/"contact_sheets";folder.mkdir();reader=FrameReader(VIDEO)
    grouped=defaultdict(list)
    for o in objects:
        if not o["age"]:grouped[o["sequence"],o["track_id"]].append(o)
    index=[];rows=[]
    try:
        for key,rr in sorted(grouped.items()):
            picks=[rr[0],rr[len(rr)//2],rr[-1]]
            strip=np.full((230,1320,3),22,np.uint8)
            for column,o in enumerate(picks):
                im=reader.read(o["source_frame0"]);left=column*440
                thumb=cv2.resize(im,(320,180));p=np.asarray(o["points"])*np.array([320/im.shape[1],180/im.shape[0]])
                cv2.rectangle(thumb,tuple(p[0].astype(int)),tuple(p[1].astype(int)),(0,200,255),1)
                strip[20:200,left:left+320]=thumb
                torso,_=crop_region(im,o["points"])
                if torso.size:strip[20:180,left+330:left+430]=cv2.resize(torso,(100,160),interpolation=cv2.INTER_NEAREST)
                caption(strip,f'{key} frame{o["source_frame0"]}',(left+4,14),.35)
                caption(strip,f'V2 {o["v2_label"]} -> V3 {o["label"]}',(left+4,213),.36)
                caption(strip,o["reason"],(left+4,227),.29)
                index.append(dict(sequence=key[0],track_id=key[1],frame0=o["source_frame0"],v2_label=o["v2_label"],v3_label=o["label"],reason=o["reason"],truth_available=False))
            name=f'{key[0].replace(":","_")}_id{key[1]}.jpg';cv2.imwrite(str(folder/name),strip);rows.append(strip)
        for page,start in enumerate(range(0,len(rows),6)):
            cv2.imwrite(str(out/f"contact_sheet_{page+1:02d}.jpg"),np.concatenate(rows[start:start+6],axis=0))
        # Algorithm-selected examples explain changes; no human labels/corrections.
        choices={}
        for o in objects:
            tag="HISTORY_SUPPORTED_A" if o["reason"]=="RECENT_A_HISTORY_WITH_CURRENT_SINGLE_COLOR" else (
                "REFEREE_APPEARANCE" if o["role"]=="REFEREE" else ("YELLOW_ROLE_ABSTENTION" if "YELLOW" in o["reason"] or "REFEREE_TEMPORAL" in o["reason"] else (
                "V2_UNKNOWN_V3_KNOWN" if o["v2_label"]=="UNKNOWN" and o["label"]!="UNKNOWN" else o["reason"] if o["label"]=="UNKNOWN" else None)))
            if tag and tag not in choices:choices[tag]=o
            if o["label"]=="UNKNOWN" and o["reason"] not in choices:choices[o["reason"]]=o
        examples=out/"diagnostics";examples.mkdir()
        for tag,o in choices.items():
            im=reader.read(o["source_frame0"]);display=cv2.resize(im,(960,540))
            draw(display,o["points"],f'{o["v2_label"]} -> {o["label"]}',o["track_id"],(0,200,255))
            bottom=np.full((100,960,3),22,np.uint8)
            caption(bottom,f'{o["sequence"]} / frame{o["source_frame0"]} / ID{o["track_id"]}',(8,20),.5)
            caption(bottom,o["reason"],(8,43),.47)
            caption(bottom,f'Components {o.get("components",{})}',(8,66),.36)
            caption(bottom,'Model diagnostic only; no manually confirmed class or identity',(8,89),.46)
            cv2.imwrite(str(examples/f"{tag}.jpg"),np.concatenate([display,bottom]))
    finally:reader.close()
    with (out/"diagnostic_review.csv").open("x",encoding="utf-8",newline="") as f:
        w=csv.DictWriter(f,fieldnames=["sequence","track_id","frame0","v2_label","v3_label","reason","truth_available"]);w.writeheader();w.writerows(index)
    write_json(out/"diagnostic_example_index.json",{tag:dict(frame0=o["source_frame0"],track_id=o["track_id"],reason=o["reason"]) for tag,o in choices.items()})


def run(frames,out,config_path=V3_CONFIG,policy_path=V3_POLICY,video=True,visuals=True):
    config=json.loads(Path(config_path).read_text(encoding="utf-8"));policy=json.loads(Path(policy_path).read_text(encoding="utf-8"))
    old=json.loads(V2_CONFIG.read_text(encoding="utf-8"));v2=UniformClassifier(old);v3=UniformClassifierV3(config,policy)
    reader=FrameReader(VIDEO);previous=None;objects=[];rows=[];samples=defaultdict(list);rng=np.random.default_rng(20261112)
    encoder=None;log=None
    if video:
        import imageio_ffmpeg
        log=(out/"encoder.log").open("x")
        command=[imageio_ffmpeg.get_ffmpeg_exe(),"-nostdin","-v","error","-n","-f","rawvideo","-pix_fmt","bgr24","-s","1920x800","-r","25","-i","pipe:0","-an","-c:v","libx264","-preset","veryfast","-crf","21","-pix_fmt","yuv420p","-movflags","+faststart",str(out/"comparison_v2_v3.mp4")]
        encoder=subprocess.Popen(command,stdin=subprocess.PIPE,stderr=log)
    try:
        with (out/"input_frames.jsonl").open("x",encoding="utf-8") as frozen:
            for raw in frames:
                i=raw["frame0"];boundary=previous is None or previous[0]!=raw["sequence"] or i!=previous[1]+1
                if boundary:
                    temporal2=TemporalClassifier(old["temporal"]);temporal3=TemporalClassifierV3(policy);e2=EvidenceRun();e3=EvidenceRun()
                previous=raw["sequence"],i
                frozen.write(json.dumps(raw,separators=(",",":"))+"\n")
                im=reader.read(i);current=[]
                for obj in raw["objects"]:
                    others=[o["points"] for o in raw["objects"] if o["track_id"]!=obj["track_id"] and o["age"]==0]
                    points=np.asarray(obj["points"],dtype=float)
                    if points.shape!=(2,2) or not np.isfinite(points).all():
                        from tryolabs_uniform_v2 import metadata
                        r2=metadata("UNKNOWN",reason="INVALID_CACHED_BOX",eligible=False,components={})
                    else:r2=v2.classify(im,obj["points"],others)
                    d2=temporal2.update(obj["track_id"],i,r2,fresh=obj["age"]==0)
                    r3=v3.classify(im,obj["points"],others);d3=temporal3.update(obj["track_id"],i,r3,fresh=obj["age"]==0)
                    record={**obj,**d3,"sequence":raw["sequence"],"source_frame0":i,"v2_label":d2["label"],"v2_role":d2["role"],"v2_eligible":d2["eligible"],
                            "v2_team":d2["team"],"v2_reason":d2["reason"],"v2_components":d2.get("components",{}),
                            "v3_observation_reason":r3["reason"],"snapshot_origin":raw["source"]}
                    objects.append(record);current.append(record)
                    if not obj["age"]:
                        roi,_=crop_region(im,obj["points"])
                        if roi.size:
                            pix=cv2.cvtColor(roi,cv2.COLOR_BGR2HSV).reshape(-1,3)
                            samples[d3["label"]].append(pix[rng.choice(len(pix),min(64,len(pix)),replace=False)])
                oldobjects=[dict(o,label=o["v2_label"],team=o["v2_team"],role=o["v2_role"],eligible=o["v2_eligible"]) for o in current]
                b=raw["balls"]
                points=np.array([[b[0]["xmin"],b[0]["ymin"]],[b[0]["xmax"],b[0]["ymax"]]]) if len(b)==1 else None
                c2,s2,w2=select_candidate(oldobjects,points,ball_count=len(b));c3,s3,w3=select_candidate(current,points,ball_count=len(b))
                t2=e2.update(None if c2 is None else c2["team"]);t3=e3.update(None if c3 is None else c3["team"])
                if c2 is not None:s2="PROXIMITY_20_FRAME_EVIDENCE" if t2 else "PROXIMITY_PENDING"
                if c3 is not None:s3="PROXIMITY_20_FRAME_EVIDENCE" if t3 else "PROXIMITY_PENDING"
                row=dict(sequence=raw["sequence"],frame0=i,v2_candidate_id=None if c2 is None else c2["track_id"],v3_candidate_id=None if c3 is None else c3["track_id"],
                         v2_status=s2,v3_status=s3,v2_reason=w2,v3_reason=w3,v2_referees=sum(o["role"]=="REFEREE" for o in oldobjects),v3_referees=sum(o["role"]=="REFEREE" for o in current),ball_candidates=len(b))
                rows.append(row)
                if encoder:encoder.stdin.write(canvas(im,current,raw,row).tobytes())
                if len(rows)%125==0:print("V3 POSTPROCESS",len(rows),i,dict(Counter(o["label"] for o in current)),flush=True)
    finally:
        reader.close()
        if encoder:
            try:encoder.stdin.close()
            except BrokenPipeError:pass
            encoder.wait();log.close()
    if encoder and encoder.returncode:raise RuntimeError("Comparison encoder failed")
    if not rows:raise ValueError("No input frames")
    with (out/"objects.jsonl").open("x",encoding="utf-8") as f:
        for o in objects:f.write(json.dumps(o,separators=(",",":"))+"\n")
    fields=["sequence","source_frame0","track_id","v2_label","v2_role","v2_reason","observed_label","stabilized_label","final_label","team","role","eligible","reason","classification_confidence","classification_source","age","shared_color_fraction","torso_pixels","torso_overlap","shorts_dark_fraction","socks_dark_fraction","small_border_person","large_bottom_clipped","dark_lower_support","lower_body_dark_fallback_used"]
    with (out/"objects.csv").open("x",encoding="utf-8",newline="") as f:
        w=csv.DictWriter(f,fieldnames=fields,extrasaction="ignore");w.writeheader();w.writerows(objects)
    with (out/"frames.csv").open("x",encoding="utf-8",newline="") as f:
        w=csv.DictWriter(f,fieldnames=list(rows[0]));w.writeheader();w.writerows(rows)
    decoded=None
    if video:
        cap=cv2.VideoCapture(str(out/"comparison_v2_v3.mp4"));decoded=0
        while cap.read()[0]:decoded+=1
        fps=cap.get(5);cap.release()
        if decoded!=len(rows) or fps!=25:raise ValueError("Output frame/fps mismatch")
    summary=dict(frames=len(rows),objects=len(objects),v2_counts=dict(Counter(o["v2_label"] for o in objects)),v3_counts=dict(Counter(o["label"] for o in objects)),
                 v2_unknown_fraction=sum(o["v2_label"]=="UNKNOWN" for o in objects)/max(1,len(objects)),v3_unknown_fraction=sum(o["label"]=="UNKNOWN" for o in objects)/max(1,len(objects)),
                 v2_unknown_reasons=dict(Counter(o["v2_reason"] for o in objects if o["v2_label"]=="UNKNOWN")),v3_unknown_reasons=dict(Counter(o["reason"] for o in objects if o["label"]=="UNKNOWN")),
                 v2_switches=count_switches(objects,"v2_label"),v3_switches=count_switches(objects,"label"),
                 v2_referee_exclusion_frames=sum(r["v2_referees"]>0 for r in rows),v3_referee_exclusion_frames=sum(r["v3_referees"]>0 for r in rows),
                 v3_referee_border_suspects=sum(o["role"]=="REFEREE" and o.get("small_border_person",False) for o in objects),
                 v3_history_assisted_a=sum(o["classification_source"]=="HSV_CURRENT_PLUS_TRACK" for o in objects),
                 candidate_different_frames=sum(r["v2_candidate_id"]!=r["v3_candidate_id"] for r in rows),v2_statuses=dict(Counter(r["v2_status"] for r in rows)),v3_statuses=dict(Counter(r["v3_status"] for r in rows)),
                 frozen_v2_agreement=sum(o["v2_label"]==o.get("frozen_v2_label") for o in objects),frozen_v2_available=sum(o.get("frozen_v2_label") is not None for o in objects),
                 feature_distributions_v2=feature_summary(objects,2),feature_distributions_v3=feature_summary(objects,3),
                 shared_color_observations=sum(o.get("shared_color_fraction",0)>0 for o in objects),
                 decoded_output_frames=decoded,manual_ground_truth_used=False,accuracy_improvement_proven=False,
                 detection_and_track_updates=0,classifier_confidence_is_probability=False)
    per_source={}
    for source in sorted({o["snapshot_origin"] for o in objects}):
        oo=[o for o in objects if o["snapshot_origin"]==source]
        per_source[source]=dict(objects=len(oo),v2_counts=dict(Counter(o["v2_label"] for o in oo)),v3_counts=dict(Counter(o["label"] for o in oo)))
    summary["source_breakdown"]=per_source
    summary["model_disagreement_table"]={f'{a}->{b}':sum(o["v2_label"]==a and o["label"]==b for o in objects) for a in ["TEAM_A","TEAM_B","REFEREE","UNKNOWN"] for b in ["TEAM_A","TEAM_B","REFEREE","UNKNOWN"]}
    summary["team_known_role_unknown"]=sum(o["team"] in ("TEAM_A","TEAM_B") and o["role"]=="UNKNOWN" for o in objects)
    summary["v3_role_counts"]=dict(Counter(o["role"] for o in objects))
    summary["referee_validation"]=dict(appearance_observations=sum(o["role"]=="REFEREE" for o in objects),
       person_identity_verified=False,invalid_referee_eligible=sum(o["role"]=="REFEREE" and o["eligible"] for o in objects),
       rejected_small_border_yellow=sum(o["reason"]=="SIDELINE_OR_CLIPPED_YELLOW" for o in objects),
       large_clipped_lower_dark_assisted=sum(o["role"]=="REFEREE" and o.get("lower_body_dark_fallback_used",False) for o in objects),
       role_pending=sum(o["reason"]=="REFEREE_TEMPORAL_PENDING" for o in objects))
    summary["eligible_fraction_v2"]=sum(o["v2_eligible"] for o in objects)/max(1,len(objects))
    summary["eligible_fraction_v3"]=sum(o["eligible"] for o in objects)/max(1,len(objects))
    statistics=track_statistics(out,objects)
    summary["automatic_track_cohort_statistics"]={label:dict(tracks=len(group),
        v2_navy_present_median=float(np.median([r["v2_navy_present_fraction"] for r in group])),
        v2_burgundy_present_median=float(np.median([r["v2_burgundy_present_fraction"] for r in group])),
        v2_dual_present_median=float(np.median([r["v2_dual_color_present_fraction"] for r in group])),
        v3_pink_support_median=float(np.median([r["v3_pink_median"] for r in group]))) for label in ["TEAM_A","TEAM_B","REFEREE","UNCERTAIN"]
        if (group:=[r for r in statistics if r["automatic_v2_track_cohort"]==label])}
    write_json(out/"summary.json",summary)
    with (out/"external_evaluation_template.csv").open("x",encoding="utf-8",newline="") as f:
        csv.writer(f).writerow(["game_id","evaluation_scope","label_source","image_path","truth","prediction"])
    if visuals:make_visuals(out,objects,samples);diagnostic_images(out,objects)
    return summary


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument("--source",choices=["legacy","full","prepared"],default="legacy")
    p.add_argument("--cache",type=Path);p.add_argument("--starts",type=float,nargs="+",default=[600,1800,3600]);p.add_argument("--seconds",type=float,default=10)
    p.add_argument("--extra-cache",type=Path);p.add_argument("--extra-start",type=int,default=8000);p.add_argument("--extra-frames",type=int,default=250)
    p.add_argument("--output",type=Path,default=OUTPUT/"preview");p.add_argument("--config",type=Path,default=V3_CONFIG);p.add_argument("--policy",type=Path,default=V3_POLICY)
    p.add_argument("--no-video",action="store_true");p.add_argument("--no-visuals",action="store_true")
    a=p.parse_args();out=a.output.resolve()
    if not np.isfinite(a.seconds) or a.seconds<=0 or any(not np.isfinite(s) or s<0 for s in a.starts):p.error("Invalid frame interval")
    if not out.is_relative_to(OUTPUT.resolve()):p.error("Use a new directory under classification_v3")
    out.mkdir(parents=True,exist_ok=False)
    before=protected_snapshot();write_json(out/"preservation_before.json",before);stat=VIDEO.stat()
    if a.source=="legacy":frames=list(legacy_frames(a.cache or ASSETS/"classification_v2"))
    elif a.source=="full":
        if a.cache is None:p.error("--source full requires --cache pointing at V2 run.sqlite/chunks")
        intervals=[(round(s*25),round((s+a.seconds)*25)) for s in a.starts]
        frames=list(full_frames(a.cache,intervals,out/"source_snapshot"))
        wanted={i for start,end in intervals for i in range(start,end)}
        if {r["frame0"] for r in frames}!=wanted:raise ValueError("Not all requested V2 cache frames are available; V3 did not run inference to fill them")
    else:
        if a.cache is None:p.error("--source prepared requires --cache INPUT_FRAMES_JSONL")
        frames=[json.loads(x) for x in a.cache.read_text(encoding="utf-8").splitlines()]
    if a.extra_cache:
        extra=list(full_frames(a.extra_cache,[(a.extra_start,a.extra_start+a.extra_frames)],out/"extra_snapshot"))
        if len(extra)!=a.extra_frames:raise ValueError("Not all requested extra diagnostic frames are cached")
        frames+=extra
    if len({(r["sequence"],r["frame0"]) for r in frames})!=len(frames):raise ValueError("Duplicate cached input frames")
    manifest=dict(implementation_version=VERSION,python=sys.version,opencv=cv2.__version__,numpy=np.__version__,
                  code_sha256={name:digest(ROOT/"starter"/name) for name in ["tryolabs_uniform_v3.py","run_tryolabs_classification_v3.py","tryolabs_v3_data.py"]},
                  config_sha256=digest(a.config),policy_sha256=digest(a.policy),v2_config_sha256=digest(V2_CONFIG),
                  input_frame_count=len(frames),source=a.source,device="cpu_postprocess",detector_and_tracker_invocations=0,video_size=stat.st_size,video_mtime_ns=stat.st_mtime_ns)
    write_json(out/"manifest.json",manifest)
    summary=run(frames,out,a.config,a.policy,not a.no_video,not a.no_visuals)
    verification=validate_preserved(before)
    now=VIDEO.stat()
    if (stat.st_size,stat.st_mtime_ns)!=(now.st_size,now.st_mtime_ns):raise ValueError("Source video changed")
    verification.update(video_size_mtime_unchanged=True,input_frames_sha256=digest(out/"input_frames.jsonl"))
    write_json(out/"verification.json",verification)
    print(json.dumps({k:summary[k] for k in ["frames","objects","v2_counts","v3_counts","v2_unknown_fraction","v3_unknown_fraction","detection_and_track_updates"]},indent=2))


if __name__=="__main__":main()
