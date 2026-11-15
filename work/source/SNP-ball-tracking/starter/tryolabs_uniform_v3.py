"""V3 appearance classification, separate from frozen V2 and possession rules.

Only image pixels and already cached track identity are consumed. No GPU, YOLO,
tracking updates or human labels. Scores are support fractions, not probabilities.
"""
from collections import Counter,deque

import cv2
import numpy as np

from tryolabs_uniform_v2 import crop_region

LABELS=("TEAM_A","TEAM_B","REFEREE","UNKNOWN")
VERSION="3.0.1"
TEAM_NAMES={"TEAM_A":"FC Barcelona","TEAM_B":"Athletic Club"}
ROLES=("PLAYER","REFEREE","GOALKEEPER","STAFF","SUBSTITUTE","UNKNOWN")


def assignment(label,reason,**features):
    return dict(label=label,team=label if label in TEAM_NAMES else (None if label=="REFEREE" else "UNKNOWN"),
                team_name=TEAM_NAMES.get(label),role="PLAYER" if label in TEAM_NAMES else ("REFEREE" if label=="REFEREE" else "UNKNOWN"),
                eligible=label in TEAM_NAMES,reason=reason,classification_confidence=None,
                classification_source="HSV_V3",role_identity_verified=False,**features)


def box_mask(hsv,bounds):
    mask=np.zeros(hsv.shape[:2],bool)
    for b in bounds:mask|=cv2.inRange(hsv,tuple(b["lower"]),tuple(b["upper"])).astype(bool)
    return mask


class UniformClassifierV3:
    def __init__(self,config,policy):self.config=config;self.policy=policy

    def classify(self,frame,points,other_points=()):
        if not isinstance(frame,np.ndarray) or frame.ndim!=3 or frame.shape[2]!=3 or frame.size==0:
            return assignment("UNKNOWN","MISSING_FRAME",partial_team_a=False,features_valid=False,scores={k:0. for k in LABELS[:-1]})
        try:points=np.asarray(points,dtype=float)
        except (TypeError,ValueError):
            return assignment("UNKNOWN","MISSING_OR_INVALID_BOX",partial_team_a=False,features_valid=False,scores={k:0. for k in LABELS[:-1]})
        if points.shape!=(2,2) or not np.isfinite(points).all():
            return assignment("UNKNOWN","MISSING_OR_INVALID_BOX",partial_team_a=False,features_valid=False,scores={k:0. for k in LABELS[:-1]})
        torso,rect=crop_region(frame,points)
        shorts,_=crop_region(frame,points,(.48,.72));lower,_=crop_region(frame,points,(.60,.95));socks,_=crop_region(frame,points,(.80,.98))
        collar,_=crop_region(frame,points,(.08,.20))
        q=self.policy["quality"];height,width=torso.shape[:2]
        if min(height,width)==0:
            return assignment("UNKNOWN","EMPTY_ROI",partial_team_a=False,features_valid=False,scores={k:0. for k in LABELS[:-1]})
        hsv=cv2.cvtColor(torso,cv2.COLOR_BGR2HSV)
        masks={k:box_mask(hsv,v) for k,v in self.config["hsv_components"].items()}
        support={k:float(v.mean()) for k,v in masks.items()}
        overlap_color=masks["burgundy"]&(masks["lavender"]|masks["pink"])
        exclusive_a=masks["burgundy"]&~(masks["lavender"]|masks["pink"])
        exclusive_b=masks["lavender"]&~masks["burgundy"]
        navy=float(masks["navy"].mean());burgundy=float(exclusive_a.mean());lavender=float(exclusive_b.mean())
        dual=min(navy,burgundy)>=q["minimum_secondary_color_fraction"]
        a_support=float((masks["navy"]|exclusive_a).mean())
        # Pink is an auxiliary appearance feature, never an independent team label.
        b_support=min(1.,lavender+self.policy["roles"]["pink_auxiliary_weight"]*support["pink"])
        scores=dict(TEAM_A=a_support if dual else 0.,TEAM_B=b_support if lavender>=self.policy["decision"]["TEAM_B"]["minimum_support"] else 0.,REFEREE=support["yellow"])
        def dark(crop):
            return 0. if crop.size==0 else float((cv2.cvtColor(crop,cv2.COLOR_BGR2HSV)[:,:,2]<=self.config["lower_dark_value_max"]).mean())
        shorts_dark,lower_dark,socks_dark,collar_dark=map(dark,[shorts,lower,socks,collar])
        xa,ya,xb,yb=rect;overlap=0.
        for other in other_points:
            arr=np.asarray(other,dtype=float)
            if arr.shape!=(2,2) or not np.isfinite(arr).all():continue
            (x1,y1),(x2,y2)=arr
            overlap=max(overlap,max(0,min(xb,x2)-max(xa,x1))*max(0,min(yb,y2)-max(ya,y1))/max(1,(xb-xa)*(yb-ya)))
        (x1,y1),(x2,y2)=points;fh,fw=frame.shape[:2]
        pad=max(4,int((x2-x1)*.3));rx1,rx2=max(0,int(x1)-pad),min(fw,int(x2)+pad)
        ry1,ry2=max(0,int(y1)),min(fh,int(y2))
        surround=frame[ry1:ry2,rx1:rx2];green=0.
        if surround.size:
            h=cv2.cvtColor(surround,cv2.COLOR_BGR2HSV);ring=np.ones(h.shape[:2],bool)
            ring[:,max(0,int(x1)-rx1):min(h.shape[1],int(x2)-rx1)]=False
            green=float((((h[:,:,0]>=25)&(h[:,:,0]<=85)&(h[:,:,1]>=40))&ring).sum()/max(1,ring.sum()))
        edge=self.policy["roles"]["image_edge_margin_fraction"]
        small=(y2-y1)/max(1,fh)<self.policy["roles"]["small_border_person_height_fraction"]
        border=small and ((x1+x2)/2<edge*fw or (x1+x2)/2>(1-edge)*fw or y2>(1-edge)*fh)
        # Truncated close-up boxes no longer describe full-body proportions.
        # Only here may the frozen V2 lower ROI supplement the shorts ROI.
        clipped_fallback=bool(self.policy["roles"].get("allow_large_bottom_clipped_lower_dark_fallback",False)
                              and not small and y2>=(1-edge)*fh)
        dark_support=max(shorts_dark,lower_dark) if clipped_fallback else shorts_dark
        quantiles=np.quantile(hsv.reshape(-1,3),[.1,.5,.9],axis=0).tolist()
        features=dict(scores=scores,components=support,exclusive_navy=navy,exclusive_burgundy=burgundy,exclusive_lavender=lavender,
                      shared_color_fraction=float(overlap_color.mean()),torso_pixels=height*width,torso_width=width,torso_height=height,
                      torso_overlap=float(overlap),shorts_dark_fraction=shorts_dark,lower_dark_fraction=lower_dark,
                      socks_dark_fraction=socks_dark,collar_dark_fraction=collar_dark,surrounding_green_fraction=green,
                      small_border_person=bool(border),hsv_q10_q50_q90=quantiles,
                      large_bottom_clipped=clipped_fallback,dark_lower_support=dark_support,
                      lower_body_dark_fallback_used=bool(clipped_fallback and lower_dark>shorts_dark),
                      saturation_q10_q50_q90=[r[1] for r in quantiles],value_q10_q50_q90=[r[2] for r in quantiles],
                      focus_variance=float(cv2.Laplacian(cv2.cvtColor(torso,cv2.COLOR_BGR2GRAY),cv2.CV_32F).var()),
                      features_valid=True,partial_team_a=False,player_role_context_supported=bool(not border and green>=self.policy["roles"]["minimum_surrounding_green_fraction"]))
        if width<q["minimum_torso_width"] or height<q["minimum_torso_height"] or height*width<q["minimum_torso_pixels"]:
            return assignment("UNKNOWN","TOO_SMALL_ROI",**features)
        if overlap>q["maximum_torso_overlap"]:return assignment("UNKNOWN","TORSO_OCCLUDED",**features)
        ranked=sorted(scores,key=lambda k:(-scores[k],k));winner,second=ranked[:2]
        margin=scores[winner]-scores[second];features["score_margin"]=margin
        # A partial current color may ONLY use a recent strong A track later.
        if not dual and a_support>=self.policy["decision"]["TEAM_A"]["minimum_support"] and scores["TEAM_B"]==0 and scores["REFEREE"]<self.policy["decision"]["REFEREE"]["minimum_support"]:
            features["partial_team_a"]=True
            return assignment("UNKNOWN","MISSING_SECOND_TEAM_A_COLOR",**features)
        if scores[winner]<=0 or scores[winner]<self.policy["decision"][winner]["minimum_support"]:
            reason="SHARED_COLOR_UNRESOLVED" if features["shared_color_fraction"]>max(navy,burgundy,lavender) else "LOW_COLOR_SUPPORT"
            return assignment("UNKNOWN",reason,**features)
        if margin<self.policy["decision"][winner]["minimum_margin"]:
            return assignment("UNKNOWN","AMBIGUOUS_TEAM_COLORS",**features)
        if winner=="REFEREE":
            if border:return assignment("UNKNOWN","SIDELINE_OR_CLIPPED_YELLOW",role_hypotheses=["REFEREE","STAFF","SUBSTITUTE"],**features)
            if dark_support<self.policy["roles"]["minimum_dark_shorts_fraction"]:
                return assignment("UNKNOWN","YELLOW_WITHOUT_DARK_SHORTS",role_hypotheses=["REFEREE","GOALKEEPER","STAFF"],**features)
            if green<self.policy["roles"]["minimum_surrounding_green_fraction"]:
                return assignment("UNKNOWN","YELLOW_ROLE_CONTEXT_UNVERIFIED",**features)
            return assignment("REFEREE","YELLOW_SHORTS_CONTEXT_APPEARANCE",**features)
        verdict=assignment(winner,"CURRENT_MULTICOLOR_HSV" if winner=="TEAM_A" else "CURRENT_LAVENDER_WITH_OPTIONAL_PINK",**features)
        if not features["player_role_context_supported"]:
            verdict.update(role="UNKNOWN",eligible=False,reason="TEAM_APPEARANCE_ROLE_UNVERIFIED",role_hypotheses=["PLAYER","STAFF","SUBSTITUTE"])
        return verdict


class TemporalClassifierV3:
    def __init__(self,policy):self.policy=policy["temporal"];self.tracks={}

    def update(self,track_id,frame0,observation,*,fresh=True):
        observation=observation or assignment("UNKNOWN","MISSING_OBSERVATION")
        p=self.policy;raw=observation["label"]
        def output(label,reason,stable="UNKNOWN",source="HSV_V3_TEMPORAL"):
            data=dict(observation);data.update(assignment(label,reason),observed_label=raw,stabilized_label=stable,final_label=label,
                                               classification_source=source,track_id=track_id)
            if label in TEAM_NAMES and not observation.get("player_role_context_supported",False):
                data.update(role="UNKNOWN",eligible=False,reason="TEAM_APPEARANCE_ROLE_UNVERIFIED")
            return data
        if not fresh:return output("UNKNOWN","STALE_TRACK")
        if track_id is None:return output("UNKNOWN","MISSING_TRACK_ID")
        if frame0 is None:return output("UNKNOWN","MISSING_FRAME_INDEX")
        s=self.tracks.get(track_id)
        if s is None or frame0-s["last"]>p["reset_gap_frames"]:
            s=dict(last=frame0,votes=deque(maxlen=p["window"]),stable="UNKNOWN",strong_a=None,conflict=False,ref_run=0,assisted=0)
            self.tracks[track_id]=s
        if s["votes"] and frame0<=s["last"]:raise ValueError("Track observations must be strictly increasing")
        contiguous=not s["votes"] or frame0==s["last"]+1
        s["last"]=frame0;s["votes"].append(raw)
        votes=Counter(s["votes"]);winner,count=sorted(votes.items(),key=lambda v:(-v[1],v[0]))[0]
        if winner!="UNKNOWN" and count>=p["minimum_votes"] and count/len(s["votes"])>=p["majority_fraction"]:s["stable"]=winner
        s["ref_run"]=(s["ref_run"]+1 if contiguous else 1) if raw=="REFEREE" else 0
        if raw in ("TEAM_B","REFEREE"):s["conflict"]=True
        if raw=="TEAM_A":s["strong_a"]=frame0;s["assisted"]=0;s["conflict"]=False
        if raw=="REFEREE":
            # Both pending and confirmed referee appearance are ineligible now.
            if s["ref_run"]<p["minimum_referee_observations"]:return output("UNKNOWN","REFEREE_TEMPORAL_PENDING",s["stable"])
            return output("REFEREE","REFEREE_APPEARANCE_TEMPORALLY_SUPPORTED","REFEREE")
        if raw=="UNKNOWN" and observation.get("partial_team_a"):
            recent=s["strong_a"] is not None and frame0-s["strong_a"]<=p["maximum_assistance_frames"]
            if s["stable"]=="TEAM_A" and recent and not s["conflict"] and s["assisted"]<p["maximum_assistance_frames"]:
                s["assisted"]+=1
                return output("TEAM_A","RECENT_A_HISTORY_WITH_CURRENT_SINGLE_COLOR","TEAM_A","HSV_CURRENT_PLUS_TRACK")
        if raw=="UNKNOWN":return output("UNKNOWN",observation["reason"],s["stable"])
        if raw!=s["stable"]:
            return output("UNKNOWN","TEMPORAL_WARMUP" if s["stable"]=="UNKNOWN" else "TEMPORAL_CONFLICT",s["stable"])
        return output(raw,observation["reason"],s["stable"])
