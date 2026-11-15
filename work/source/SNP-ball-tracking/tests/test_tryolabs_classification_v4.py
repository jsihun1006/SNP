"""Synthetic behavioural tests and real CPU decoding; not match accuracy."""
import copy
import json
from pathlib import Path
import sys
import tempfile
import unittest
from unittest.mock import patch

import cv2
import numpy as np

ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT/"starter"))
from tryolabs_uniform_v4 import UniformClassifierV4,TemporalClassifierV4,select_candidate_v4,assign
from tryolabs_v4_context import frame_context,goal_geometry_hint,CameraTransitionMonitor
from test_tryolabs_classification_v3 import appearance,BOX,CONFIG as V3C,POLICY as V3P
import run_tryolabs_classification_v4 as runner

CONFIG=json.loads(runner.V4_CONFIG.read_text());POLICY=json.loads(runner.V4_POLICY.read_text())


def candidate(label="GK_B",goal=True,field=True,shorts=.5):
    return dict(v3_observation=dict(label="UNKNOWN",role="UNKNOWN",team="UNKNOWN",eligible=False,
                reason="LOW_COLOR_SUPPORT",features_valid=True,partial_team_a=False),
                quality_valid=True,appearance_label="TEAL_KIT_CANDIDATE",goalkeeper_candidates=[label],
                goalkeeper_shorts={label:shorts},goal_geometry_hint=goal,field_view_supported=field,
                other_observable=False,staff_context_supported=False)


class V4RoleTests(unittest.TestCase):
    def setUp(self):self.c=UniformClassifierV4(CONFIG,POLICY,V3C,V3P)
    def test_teal_is_candidate_not_identity_or_automatic_role(self):
        r=self.c.classify(appearance([(91,150,196)]),BOX)
        self.assertEqual(r["goalkeeper_candidates"],["GK_B"])
        out=TemporalClassifierV4(POLICY,V3P).update(1,0,r)
        self.assertEqual(out["final_label"],"UNKNOWN");self.assertFalse(out["person_identity_verified"])
    def test_gk_a_disabled_no_season_color_assumption(self):
        self.assertFalse(CONFIG["goalkeeper_kits"]["GK_A"]["enabled"])
        for color in [(12,200,230),(58,160,200)]:
            r=self.c.classify(appearance([color]),BOX)
            self.assertNotIn("GK_A",r["goalkeeper_candidates"])
    def test_role_needs_geometry_shorts_and_persistence(self):
        for kwargs in [dict(goal=False),dict(field=False),dict(shorts=0)]:
            t=TemporalClassifierV4(POLICY,V3P)
            for i in range(40):out=t.update(1,i,candidate(**kwargs))
            self.assertEqual(out["final_label"],"UNKNOWN")
        t=TemporalClassifierV4(POLICY,V3P)
        for i in range(10):out=t.update(1,i,candidate())
        self.assertEqual(out["final_label"],"GK_B");self.assertEqual(out["team"],"TEAM_B")
        self.assertEqual(out["role"],"GOALKEEPER");self.assertTrue(out["eligible"])
    def test_gk_retained_outside_goal_with_current_color(self):
        t=TemporalClassifierV4(POLICY,V3P)
        for i in range(10):t.update(1,i,candidate())
        self.assertEqual(t.update(1,10,candidate(goal=False,field=False,shorts=0))["final_label"],"GK_B")
    def test_new_id_gap_and_scene_require_new_evidence(self):
        t=TemporalClassifierV4(POLICY,V3P)
        for i in range(10):t.update(1,i,candidate())
        self.assertEqual(t.update(2,10,candidate())["final_label"],"UNKNOWN")
        self.assertEqual(t.update(1,30,candidate())["final_label"],"UNKNOWN")
        self.assertEqual(TemporalClassifierV4(POLICY,V3P).update(1,31,candidate())["final_label"],"UNKNOWN")
    def test_stale_missing_and_bad_roi_are_unknown(self):
        t=TemporalClassifierV4(POLICY,V3P)
        self.assertFalse(t.update(1,0,candidate(),fresh=False)["eligible"])
        self.assertEqual(t.update(1,1,None)["final_label"],"UNKNOWN")
        for box in [None,[[80,10],[82,14]]]:
            r=self.c.classify(appearance([(91,150,196)]),box)
            self.assertFalse(r["quality_valid"])
    def test_occluded_known_color_never_other(self):
        r=self.c.classify(appearance([(140,70,238)]),BOX,[BOX])
        self.assertFalse(r["other_observable"]);self.assertFalse(r["quality_valid"])
    def test_zero_support_is_not_enough_for_other(self):
        im=np.full((240,240,3),180,np.uint8)
        r=self.c.classify(im,BOX)
        self.assertFalse(r["other_observable"])
    def test_clear_nonmatching_white_texture_other(self):
        im=appearance([(0,0,235)])
        # Achromatic texture is resolvable but not a defined field/GK uniform.
        im[45:108:4,90:130]=(110,110,110)
        r=self.c.classify(im,BOX)
        self.assertTrue(r["other_observable"])
        t=TemporalClassifierV4(POLICY,V3P)
        for i in range(10):out=t.update(1,i,r)
        self.assertEqual(out["final_label"],"OTHER");self.assertFalse(out["eligible"])
        self.assertEqual(out["team"],"UNKNOWN");self.assertEqual(out["role"],"OTHER")
    def test_low_support_known_purple_remains_unknown(self):
        im=appearance([(140,70,238),(0,0,230),(0,0,230),(0,0,230),(0,0,230),(0,0,230)])
        self.assertFalse(self.c.classify(im,BOX)["other_observable"])
    def test_other_cannot_automatically_promote_to_gk_or_player(self):
        t=TemporalClassifierV4(POLICY,V3P);r=candidate();r.update(goalkeeper_candidates=[],other_observable=True)
        for i in range(10):self.assertIn(t.update(1,i,r)["final_label"],["UNKNOWN","OTHER"])
        for i in range(10,40):out=t.update(1,i,candidate())
        self.assertEqual(out["final_label"],"UNKNOWN")
        player=self.c.classify(appearance([(122,140,127),(166,205,174)]),BOX)
        for i in range(40,70):out=t.update(1,i,player)
        self.assertEqual(out["final_label"],"UNKNOWN")
    def test_staff_needs_verified_zone_motion_and_long_track(self):
        im=appearance([(0,0,235)]);im[45:108:4,90:130]=(110,110,110)
        r=self.c.classify(im,BOX)
        self.assertFalse(r["staff_context_supported"])
        ctx=dict(technical_area_verified=True,pitch_relation="OUTSIDE",camera_motion_compensated=True,
                 nonplayer_motion=True,substitute_evidence=False)
        r=self.c.classify(im,BOX,context=ctx)
        t=TemporalClassifierV4(POLICY,V3P)
        for i in range(49):out=t.update(1,i,r)
        self.assertEqual(out["final_label"],"UNKNOWN")
        out=t.update(1,49,r);self.assertEqual(out["final_label"],"STAFF");self.assertFalse(out["eligible"])
        ctx["substitute_evidence"]=True
        self.assertFalse(self.c.classify(im,BOX,context=ctx)["staff_context_supported"])
    def test_same_team_appearance_does_not_imply_staff(self):
        im=appearance([(122,140,127),(166,205,174)],green=False)
        obs=self.c.classify(im,BOX);t=TemporalClassifierV4(POLICY,V3P)
        for i in range(10):r=t.update(1,i,obs)
        self.assertEqual(r["final_label"],"UNKNOWN")
        self.assertEqual(r["appearance_label"],"TEAM_A")
    def test_referee_never_becomes_gk_when_current_evidence_conflicts(self):
        obs=self.c.classify(appearance([(31,242,255)]),BOX);t=TemporalClassifierV4(POLICY,V3P)
        for i in range(10):r=t.update(1,i,obs)
        self.assertEqual(r["final_label"],"REFEREE");self.assertFalse(r["eligible"])
        for i in range(10,20):r=t.update(1,i,candidate())
        self.assertEqual(r["final_label"],"UNKNOWN")
    def test_known_uniform_history_blocks_temporary_nonmatching_other(self):
        obs=self.c.classify(appearance([(122,140,127),(166,205,174)]),BOX)
        t=TemporalClassifierV4(POLICY,V3P)
        for i in range(10):out=t.update(1,i,obs)
        self.assertEqual(out["final_label"],"TEAM_A")
        unknown=candidate();unknown.update(goalkeeper_candidates=[],other_observable=True)
        for i in range(10,40):out=t.update(1,i,unknown)
        self.assertEqual(out["final_label"],"UNKNOWN")
    def test_possession_accepts_goalkeeper_but_excludes_other_roles(self):
        for label in ["GK_A","GK_B"]:
            g=assign({},label,"test");g.update(track_id=1,points=BOX,age=0)
            self.assertEqual(select_candidate_v4([g],[[80,208],[82,212]])[0]["track_id"],1)
        for label in ["REFEREE","STAFF","OTHER","UNKNOWN"]:
            x=assign({},label,"test");x.update(track_id=1,points=BOX,age=0,eligible=True)
            self.assertIsNone(select_candidate_v4([x],[[80,208],[82,212]])[0])
    def test_body_near_ball_does_not_relax_45px_foot_rule(self):
        g=assign({},"GK_B","test");g.update(track_id=1,points=BOX,age=0)
        self.assertIsNone(select_candidate_v4([g],[[108,90],[112,94]])[0])
        self.assertIsNone(select_candidate_v4([],None,ball_count=0)[0])
    def test_goal_geometry_is_weak_not_verified_pitch_or_play(self):
        im=np.full((360,640,3),(30,120,30),np.uint8)
        cv2.line(im,(100,100),(200,110),(255,255,255),3)
        cv2.line(im,(100,100),(100,160),(255,255,255),3)
        cv2.line(im,(200,110),(200,170),(255,255,255),3)
        objs=[dict(points=[[250+i*15,180],[260+i*15,220]],age=0) for i in range(8)]
        ctx=frame_context(im,objs)
        self.assertTrue(goal_geometry_hint([[150,125],[165,170]],ctx))
        self.assertFalse(ctx["pitch_geometry_verified"]);self.assertEqual(ctx["broadcast_state"],"UNVERIFIED")
    def test_camera_transition_rechecks_cut_and_dissolve(self):
        mon=CameraTransitionMonitor(POLICY["scene"])
        green=np.full((64,64,3),(20,150,20),np.uint8);blue=np.full_like(green,(180,20,20))
        for _ in range(15):self.assertIsNone(mon.update(green))
        self.assertEqual(mon.update(blue),"CAMERA_VIEW_TRANSITION_RECHECK")
        mon=CameraTransitionMonitor(POLICY["scene"])
        for _ in range(15):mon.update(green)
        hints=[]
        for weight in np.linspace(0,1,15):hints.append(mon.update(cv2.addWeighted(green,1-weight,blue,weight,0)))
        self.assertTrue(any(hints))


class ReplayTests(unittest.TestCase):
    def test_same_cached_inputs_reproduce_v3_and_v4_without_mutation(self):
        with tempfile.TemporaryDirectory() as temp:
            root=Path(temp);video=root/"source.mp4"
            writer=cv2.VideoWriter(str(video),cv2.VideoWriter_fourcc(*"mp4v"),25,(240,240))
            for i in range(8):writer.write(appearance([(122,140,127),(166,205,174)]))
            writer.release()
            frames=[dict(sequence="synthetic",frame0=i,source="synthetic",balls=[],objects=[dict(track_id=1,points=BOX,age=0)]) for i in range(8)]
            before=copy.deepcopy(frames)
            with patch.object(runner,"VIDEO",video):
                for n in ["one","two"]:
                    out=root/n;out.mkdir();runner.run(frames,out,video=False,make_visuals=False)
            self.assertEqual(frames,before)
            for filename in ["input_frames.jsonl","objects.jsonl","objects.csv","frames.csv","summary.json"]:
                self.assertEqual((root/"one"/filename).read_bytes(),(root/"two"/filename).read_bytes())


if __name__=="__main__":unittest.main()
