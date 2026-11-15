"""Conservative V4 role extensions on frozen V3 pixel observations and tracks."""
from collections import deque
import cv2
import numpy as np

from tryolabs_uniform_v2 import crop_region
from tryolabs_uniform_v3 import UniformClassifierV3, TemporalClassifierV3, box_mask
from tryolabs_v4_context import goal_geometry_hint

VERSION = "4.0.1"
LABELS = ("TEAM_A", "TEAM_B", "GK_A", "GK_B", "REFEREE", "STAFF", "OTHER", "UNKNOWN")
TEAM = {"TEAM_A":"TEAM_A", "TEAM_B":"TEAM_B", "GK_A":"TEAM_A", "GK_B":"TEAM_B"}
ROLE = {"TEAM_A":"PLAYER", "TEAM_B":"PLAYER", "GK_A":"GOALKEEPER", "GK_B":"GOALKEEPER",
        "REFEREE":"REFEREE", "STAFF":"STAFF", "OTHER":"OTHER", "UNKNOWN":"UNKNOWN"}


def assign(observation, label, reason, *, appearance=None):
    result = dict(observation)
    result.update(final_label=label, label=label, team=TEAM.get(label, None if label in ("REFEREE","STAFF") else "UNKNOWN"),
                  role=ROLE[label], eligible=label in TEAM, reason=reason,
                  appearance_label=appearance or observation.get("appearance_label", "UNKNOWN"),
                  classification_confidence=None, classification_source="HSV_V4_WITH_CONTEXT",
                  person_identity_verified=False)
    return result


class UniformClassifierV4:
    def __init__(self, config, policy, v3_config, v3_policy):
        self.config, self.policy = config, policy
        self.base = UniformClassifierV3(v3_config, v3_policy)

    def classify(self, frame, points, other_points=(), context=None, base_observation=None):
        base = base_observation if base_observation is not None else self.base.classify(frame,points,other_points)
        r = dict(v3_observation=base, appearance_label=base["label"], quality_valid=False,
                 goalkeeper_candidates=[], goalkeeper_shorts={}, other_observable=False,
                 goal_geometry_hint=False, staff_context_supported=False,
                 broadcast_state=(context or {}).get("broadcast_state", "UNVERIFIED"))
        if not base.get("features_valid") or base["reason"] in ("TOO_SMALL_ROI", "TORSO_OCCLUDED", "EMPTY_ROI"):
            return r
        torso, _ = crop_region(frame, points); shorts, _ = crop_region(frame, points, (.48,.72))
        hsv = cv2.cvtColor(torso,cv2.COLOR_BGR2HSV)
        sh = cv2.cvtColor(shorts,cv2.COLOR_BGR2HSV) if shorts.size else np.zeros((1,1,3),np.uint8)
        kits = {}; shorts_support = {}
        for name, kit in self.config["goalkeeper_kits"].items():
            kits[name] = float(box_mask(hsv,kit["bounds"]).mean()) if kit["enabled"] else 0.
            shorts_support[name] = float(box_mask(sh,kit["bounds"]).mean()) if kit["enabled"] else 0.
        threshold = self.policy["goalkeeper"]["minimum_torso_support"]
        candidates = [name for name, score in kits.items() if score >= threshold]
        search = {name:float(box_mask(hsv,[bounds]).mean()) for name,bounds in self.config["unresolved_search"].items()}
        # Green search includes turf and yellow fringes. It is never a GK_A kit.
        grass = float(cv2.inRange(hsv,(25,40,35),(85,255,255)).mean()/255)
        rule = self.policy["other"]; hh, ww = hsv.shape[:2]
        clear = (ww >= rule["minimum_torso_width"] and hh >= rule["minimum_torso_height"]
                 and hh*ww >= rule["minimum_torso_pixels"]
                 and base["focus_variance"] >= rule["minimum_focus_variance"])
        defined = max(base.get("components",{}).values(), default=0.)
        other = (clear and grass <= rule["maximum_grass_fraction"]
                 and defined < rule["maximum_defined_color_support"]
                 and max(kits.values(),default=0.) < .10 and max(search.values(),default=0.) < .30
                 and base["reason"] == "LOW_COLOR_SUPPORT")
        context = context or {}
        staff = bool(other and context.get("technical_area_verified")
                     and context.get("pitch_relation") == "OUTSIDE"
                     and context.get("camera_motion_compensated") and context.get("nonplayer_motion")
                     and not context.get("substitute_evidence", True))
        r.update(quality_valid=True, goalkeeper_candidates=candidates, goalkeeper_support=kits,
                 goalkeeper_shorts=shorts_support, unresolved_search_support=search,
                 other_observable=bool(other), grass_fraction=grass, roi_clear=bool(clear),
                 staff_context_supported=staff, goal_geometry_hint=goal_geometry_hint(points,context) if candidates else False,
                 field_view_supported=context.get("field_view_supported",False),
                 glove_evidence="NOT_RELIABLY_RESOLVED", torso_pixels=hh*ww,
                 hsv_q10_q50_q90=np.quantile(hsv.reshape(-1,3),[.1,.5,.9],axis=0).tolist(),
                 focus_variance=base["focus_variance"])
        coverage=context.get("green_coverage")
        if coverage is None:
            hhsv=cv2.cvtColor(frame,cv2.COLOR_BGR2HSV)
            coverage=float(cv2.inRange(hhsv,(25,40,35),(85,255,255)).mean()/255)
        r["player_scene_context_supported"]=bool(context.get("field_view_supported") or coverage>=.35)
        if candidates:r["appearance_label"]="TEAL_KIT_CANDIDATE" if candidates==["GK_B"] else "GK_KIT_AMBIGUOUS"
        elif other:r["appearance_label"]="NONMATCHING_OBSERVABLE"
        elif base["label"]=="UNKNOWN" and max(search.values(),default=0.)>=.30:r["appearance_label"]="UNRESOLVED_ORANGE_GREEN_SEARCH"
        return r


class TemporalClassifierV4:
    def __init__(self, policy, v3_policy):
        self.policy=policy; self.base=TemporalClassifierV3(v3_policy); self.tracks={}

    def update(self, track_id, frame0, observation, *, fresh=True):
        observation=observation or {}
        base=self.base.update(track_id,frame0,observation.get("v3_observation"),fresh=fresh)
        r=dict(observation,track_id=track_id,v4_internal_v3_label=base["label"],v4_internal_v3_role=base["role"],
               observed_label=observation.get("appearance_label","UNKNOWN"))
        if not fresh or track_id is None or frame0 is None:
            return assign(r,"UNKNOWN","STALE_OR_MISSING_TRACK_INPUT")
        p=self.policy["goalkeeper"];state=self.tracks.get(track_id)
        if state is None or frame0-state["last"]>p["reset_gap_frames"]:
            state=dict(last=frame0,history=deque(maxlen=p["window"]),gk=None,other_run=0,staff_run=0,other_latched=False,known_uniform_seen=False)
            self.tracks[track_id]=state
        elif frame0<=state["last"]:
            raise ValueError("V4 track observations must increase")
        consecutive=frame0==state["last"]+1
        state["last"]=frame0
        if observation.get("v3_observation",{}).get("label") in ("TEAM_A","TEAM_B","REFEREE"):
            state["known_uniform_seen"]=True
        if not observation.get("quality_valid"):
            state["history"].append((None,False));state["other_run"]=state["staff_run"]=0
            return assign(r,"UNKNOWN",base["reason"])
        candidates=observation["goalkeeper_candidates"]
        # A known uniform or referee observation conflicts with GK appearance.
        if base["label"] in ("TEAM_A","TEAM_B","REFEREE"):
            state["history"].append((None,False));state["other_run"]=state["staff_run"]=0
            if candidates or state["gk"] or state["other_latched"]:
                return assign(r,"UNKNOWN","CURRENT_ROLE_APPEARANCE_CONFLICT")
            if base["label"] in ("TEAM_A","TEAM_B") and (not base["eligible"] or not observation.get("player_scene_context_supported",False)):
                return assign(r,"UNKNOWN","TEAM_ROLE_CONTEXT_UNVERIFIED")
            return assign(r,base["label"],base["reason"])
        match=candidates[0] if len(candidates)==1 else None
        clothing=match and observation["goalkeeper_shorts"].get(match,0.)>=p["minimum_shorts_support"]
        current=match if clothing and observation.get("field_view_supported") else None
        state["history"].append((current,bool(current and observation["goal_geometry_hint"])))
        if match:
            state["other_run"]=state["staff_run"]=0
            if state["other_latched"]:return assign(r,"UNKNOWN","OTHER_TRACK_REQUIRES_ROLE_REESTABLISHMENT")
            if state["known_uniform_seen"]:return assign(r,"UNKNOWN","KNOWN_UNIFORM_TRACK_GK_CONFLICT")
            if state["gk"]==match:
                return assign(r,match,"GK_TRACK_CURRENT_COLOR_RETAINED")
            votes=sum(x[0]==match for x in state["history"])
            goal_votes=sum(x[1] and x[0]==match for x in state["history"])
            if (current and votes>=p["minimum_observations"] and votes/len(state["history"])>=p["minimum_fraction"]
                    and goal_votes>=p["minimum_goal_hints"]):
                state["gk"]=match
                return assign(r,match,"GK_KIT_SHORTS_TRACK_WEAK_GOAL_CONTEXT")
            return assign(r,"UNKNOWN","GK_ROLE_EVIDENCE_PENDING")
        if observation.get("staff_context_supported"):
            state["staff_run"]=(state["staff_run"]+1 if consecutive else 1)
            if state["staff_run"]>=self.policy["staff"]["minimum_observations"]:
                return assign(r,"STAFF","VERIFIED_ZONE_NONPLAYER_TRACK_APPEARANCE")
            return assign(r,"UNKNOWN","STAFF_CONTEXT_TRACK_PENDING")
        state["staff_run"]=0
        if observation.get("other_observable") and not state["gk"] and not state["known_uniform_seen"]:
            state["other_run"]=(state["other_run"]+1 if consecutive else 1)
            if state["other_run"]>=self.policy["other"]["minimum_observations"]:
                state["other_latched"]=self.policy["other"]["latch_until_scene_reset"]
                return assign(r,"OTHER","CLEAR_NONMATCHING_APPEARANCE")
            return assign(r,"UNKNOWN","OTHER_OBSERVATION_PENDING")
        state["other_run"]=0
        return assign(r,"UNKNOWN",base["reason"] if base["label"]=="UNKNOWN" else "UNRESOLVED_ROLE")


def select_candidate_v4(objects, ball_points, *, ball_count=1, ball_age=0):
    """Frozen 45px bottom-corner distance, extended only to eligible goalkeepers."""
    eligible=[o for o in objects if o.get("role") in ("PLAYER","GOALKEEPER")
              and o.get("team") in ("TEAM_A","TEAM_B") and o.get("eligible") and o.get("age",0)==0]
    if ball_count!=1 or ball_points is None or ball_age!=0:return None,"INSUFFICIENT_EVIDENCE","NO_UNIQUE_FRESH_BALL"
    if not eligible:return None,"INSUFFICIENT_EVIDENCE","NO_ELIGIBLE_PLAYER"
    center=np.asarray(ball_points).mean(axis=0)
    def distance(o):
        (x1,_),(x2,y2)=np.asarray(o["points"])
        return min(np.linalg.norm(center-[x1,y2]),np.linalg.norm(center-[x2,y2]))
    closest=min(eligible,key=lambda o:(distance(o),o["track_id"]))
    if distance(closest)>45:return None,"INSUFFICIENT_EVIDENCE","DISTANCE_GT_45PX"
    return dict(closest,distance_px=float(distance(closest))),"PROXIMITY_EVIDENCE","NEAREST_ELIGIBLE_PLAYER"
