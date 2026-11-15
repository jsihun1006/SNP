"""Per-frame frozen V3/V4 execution and portable, lossless JSON checkpoints."""
from collections import deque
import json

import numpy as np

from run_tryolabs_classification_v3 import EvidenceRun
from tryolabs_uniform_v2 import select_candidate
from tryolabs_uniform_v3 import UniformClassifierV3, TemporalClassifierV3
from tryolabs_uniform_v4 import UniformClassifierV4, TemporalClassifierV4, select_candidate_v4, assign, LABELS
from tryolabs_v4_context import frame_context, CameraTransitionMonitor


def encode(value):
    if isinstance(value, np.ndarray):
        return {"type":"ndarray", "dtype":value.dtype.str, "shape":list(value.shape), "items":value.tolist()}
    if isinstance(value, deque):
        return {"type":"deque", "maxlen":value.maxlen, "items":[encode(x) for x in value]}
    if isinstance(value, tuple): return {"type":"tuple", "items":[encode(x) for x in value]}
    if isinstance(value, dict): return {"type":"dict", "items":[[encode(k),encode(v)] for k,v in value.items()]}
    if isinstance(value, list): return [encode(x) for x in value]
    if isinstance(value, np.generic): return value.item()
    return value


def decode(value):
    if isinstance(value, list): return [decode(x) for x in value]
    if not isinstance(value, dict): return value
    kind=value["type"]
    if kind=="ndarray": return np.asarray(value["items"],dtype=np.dtype(value["dtype"])).reshape(value["shape"])
    if kind=="deque": return deque((decode(x) for x in value["items"]),maxlen=value["maxlen"])
    if kind=="tuple": return tuple(decode(x) for x in value["items"])
    if kind=="dict": return {decode(k):decode(v) for k,v in value["items"]}
    raise ValueError(f"Unsupported state type: {kind}")


def bump(mapping, key, count=1): mapping[key]=mapping.get(key,0)+int(count)


class Metrics:
    def __init__(self):
        self.data=dict(frames=0,objects=0,v3_counts={},v4_counts={},v4_unknown_reasons={},transition_table={},
                      v3_statuses={},v4_statuses={},ball_candidate_counts={},candidate_different_frames=0,
                      gk_body_near_foot_far_frames=0,goalkeeper_candidate_observations=0,goalkeeper_candidate_other=0,
                      v3_referee_to_goalkeeper=0,v4_referee_exclusion_frames=0,camera_transition_recheck_frames=0,
                      class_changes_v3_common_epoch=0,class_changes_v4_common_epoch=0,
                      role_changes_v3_common_epoch=0,role_changes_v4_common_epoch=0,
                      candidate_changes_v3=0,candidate_changes_v4=0,time_bins={},peak_frame_objects=0,
                      peak_temporal_tracks=0,peak_camera_history=0)
        self.previous_objects={};self.previous_frame=None

    def add(self, raw, objects, row, engine):
        d=self.data;d["frames"]+=1;d["objects"]+=len(objects)
        d["peak_frame_objects"]=max(d["peak_frame_objects"],len(objects))
        d["peak_temporal_tracks"]=max(d["peak_temporal_tracks"],len(engine.t3.tracks),len(engine.t4.tracks),len(engine.t4.base.tracks))
        d["peak_camera_history"]=max(d["peak_camera_history"],len(engine.monitor.history))
        current={};keybin=str(raw["frame0"]//7500)
        bin_=d["time_bins"].setdefault(keybin,dict(frames=0,objects=0,v3_counts={},v4_counts={},candidate_different_frames=0))
        bin_["frames"]+=1;bin_["objects"]+=len(objects)
        for o in objects:
            a,b=o["v3_label"],o["final_label"]
            bump(d["v3_counts"],a);bump(d["v4_counts"],b);bump(d["transition_table"],a+"->"+b)
            bump(bin_["v3_counts"],a);bump(bin_["v4_counts"],b)
            if b=="UNKNOWN":bump(d["v4_unknown_reasons"],o["reason"])
            candidate=bool(o["goalkeeper_candidates"])
            d["goalkeeper_candidate_observations"]+=candidate
            d["goalkeeper_candidate_other"]+=candidate and b=="OTHER"
            d["v3_referee_to_goalkeeper"]+=a=="REFEREE" and o["role"]=="GOALKEEPER"
            key=(o["sequence"],o["track_id"],o["v4_epoch"])
            old=self.previous_objects.get(key)
            if old and old["source_frame0"]+1==o["source_frame0"]:
                for field,metric in [("v3_label","class_changes_v3_common_epoch"),("final_label","class_changes_v4_common_epoch"),
                                     ("v3_role","role_changes_v3_common_epoch"),("role","role_changes_v4_common_epoch")]:
                    d[metric]+=old[field]!=o[field]
            current[key]={k:o[k] for k in ("source_frame0","v3_label","final_label","v3_role","role")}
        self.previous_objects=current  # At most the previous frame, not all match objects.
        for v in ("v3","v4"):bump(d[v+"_statuses"],row[v+"_status"])
        bump(d["ball_candidate_counts"],str(min(2,row["ball_candidates"])))
        different=row["v3_candidate_id"]!=row["v4_candidate_id"]
        d["candidate_different_frames"]+=different;bin_["candidate_different_frames"]+=different
        d["gk_body_near_foot_far_frames"]+=row["gk_body_near_foot_far"]>0
        d["v4_referee_exclusion_frames"]+=any(o["role"]=="REFEREE" for o in objects)
        d["camera_transition_recheck_frames"]+=bool(row["camera_transition_hint"])
        old=self.previous_frame
        if old and old["sequence"]==row["sequence"] and old["frame0"]+1==row["frame0"]:
            for v in ("v3","v4"):
                if v=="v3" or old["v4_epoch"]==row["v4_epoch"]:
                    d["candidate_changes_"+v]+=old[v+"_candidate_id"]!=row[v+"_candidate_id"]
        self.previous_frame=dict(row)

    def summary(self):
        d=dict(self.data)
        for v in ("v3","v4"):
            d[v+"_unknown_fraction"]=d[v+"_counts"].get("UNKNOWN",0)/max(1,d["objects"])
        d.update(manual_ground_truth=False,accuracy_improvement_proven=False,actual_staff_player_errors=None,
                 live_replay_state_verified=False,detector_calls=0,tracker_updates=0,
                 staff_context_available=False,goalkeeper_person_identity_verified=False)
        return d


class Engine:
    def __init__(self, v3_config, v3_policy, v4_config, v4_policy):
        self.p3=v3_policy;self.p4=v4_policy
        self.v3=UniformClassifierV3(v3_config,v3_policy)
        self.v4=UniformClassifierV4(v4_config,v4_policy,v3_config,v3_policy)
        self.previous=None;self.epoch=0;self.metrics=Metrics();self.reset()

    def reset(self):
        self.t3=TemporalClassifierV3(self.p3);self.t4=TemporalClassifierV4(self.p4,self.p3)
        self.e3=EvidenceRun();self.e4=EvidenceRun();self.monitor=CameraTransitionMonitor(self.p4["scene"])

    def process(self, raw, image):
        i=raw["frame0"]
        boundary=self.previous is None or raw["sequence"]!=self.previous[0] or i!=self.previous[1]+1
        if boundary:self.reset();self.epoch+=1
        self.previous=raw["sequence"],i
        context=frame_context(image,raw["objects"]);current=[]
        transition=self.monitor.update(image)
        if transition:
            self.t4=TemporalClassifierV4(self.p4,self.p3);self.e4=EvidenceRun();self.epoch+=1
        # Expired entries would reset on their next observation in the frozen classes.
        # Pruning them bounds memory without changing that lazy-reset behaviour.
        for classifier,gap in [(self.t3,self.p3["temporal"]["reset_gap_frames"]),
                               (self.t4.base,self.p3["temporal"]["reset_gap_frames"]),
                               (self.t4,self.p4["goalkeeper"]["reset_gap_frames"])]:
            for track_id in list(classifier.tracks):
                if i-classifier.tracks[track_id]["last"]>gap:del classifier.tracks[track_id]
        for obj in raw["objects"]:
            others=[o["points"] for o in raw["objects"] if o["track_id"]!=obj["track_id"] and o["age"]==0]
            b3=self.v3.classify(image,obj["points"],others)
            r3=self.t3.update(obj["track_id"],i,b3,fresh=obj["age"]==0)
            obs=self.v4.classify(image,obj["points"],others,context,b3)
            r4=self.t4.update(obj["track_id"],i,obs,fresh=obj["age"]==0)
            if transition:r4=assign(r4,"UNKNOWN",transition)
            r4.pop("v3_observation",None)
            current.append(dict(obj,**r4,sequence=raw["sequence"],source_frame0=i,v3_label=r3["label"],v3_role=r3["role"],
                                v3_team=r3["team"],v3_eligible=r3["eligible"],v3_reason=r3["reason"],
                                snapshot_origin=raw["source"],v3_components=b3.get("components",{}),v4_epoch=self.epoch))
        balls=raw["balls"]
        bp=[[balls[0]["xmin"],balls[0]["ymin"]],[balls[0]["xmax"],balls[0]["ymax"]]] if len(balls)==1 else None
        oldobjects=[dict(o,role=o["v3_role"],team=o["v3_team"],eligible=o["v3_eligible"]) for o in current]
        c3,s3,w3=select_candidate(oldobjects,bp,ball_count=len(balls))
        c4,s4,w4=select_candidate_v4(current,bp,ball_count=len(balls))
        team3=self.e3.update(c3["team"] if c3 else None);team4=self.e4.update(c4["team"] if c4 else None)
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
                 v4_epoch=self.epoch,camera_transition_hint=transition)
        self.metrics.add(raw,current,row,self)
        return current,row

    def export(self):
        return dict(schema=1,state=encode(dict(previous=self.previous,epoch=self.epoch,t3=self.t3.tracks,
            t4=self.t4.tracks,t4_base=self.t4.base.tracks,e3=vars(self.e3),e4=vars(self.e4),monitor=self.monitor.history,
            metrics=vars(self.metrics))))

    def restore(self, payload):
        if payload["schema"]!=1:raise ValueError("Unknown checkpoint schema")
        s=decode(payload["state"]);self.previous=s["previous"];self.epoch=s["epoch"]
        self.t3.tracks=s["t3"];self.t4.tracks=s["t4"];self.t4.base.tracks=s["t4_base"]
        self.e3.__dict__.update(s["e3"]);self.e4.__dict__.update(s["e4"])
        self.monitor.history=s["monitor"];self.metrics.__dict__.update(s["metrics"])
