"""Synthetic behavior checks; never create labels from competition video."""

import json
from pathlib import Path
import sys
import unittest

import cv2
import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT/"starter"))
from tryolabs_uniform_v2 import UniformClassifier, TemporalClassifier, metadata, select_candidate


CONFIG = json.loads((ROOT/"starter/configs/tryolabs_classification_v2_hsv.json").read_text())
POINTS = [[0,0],[60,100]]


def appearance(hsv_color, *, dark_lower=True):
    hsv = np.full((100,60,3), (60,200,160), np.uint8)
    hsv[15:50,9:51] = hsv_color
    hsv[60:95,9:51] = (0,0,40) if dark_lower else (60,200,160)
    return cv2.cvtColor(hsv,cv2.COLOR_HSV2BGR)


def obj(track_id, team="TEAM_A", role="PLAYER", points=None, eligible=True):
    return dict(track_id=track_id,team=team,role=role,eligible=eligible,age=0,
                points=points or [[10,0],[20,20]])


class UniformTests(unittest.TestCase):
    def setUp(self):
        self.classifier = UniformClassifier(CONFIG)

    def test_zero_scores_are_unknown(self):
        result = self.classifier.classify(appearance((0,0,160)),POINTS)
        self.assertEqual(result["label"],"UNKNOWN")
        self.assertTrue(all(s == 0 for s in result["scores"].values()))
        self.assertIsNone(result["classification_confidence"])

    def test_two_stripes_are_team_a(self):
        image = appearance((122,143,123))
        patch = cv2.cvtColor(np.uint8([[[165,203,181]]]),cv2.COLOR_HSV2BGR)[0,0]
        image[15:50,30:51] = patch
        self.assertEqual(self.classifier.classify(image,POINTS)["label"],"TEAM_A")

    def test_single_navy_not_forced_to_team_a(self):
        self.assertEqual(self.classifier.classify(appearance((122,143,123)),POINTS)["label"],"UNKNOWN")

    def test_lilac_is_team_b(self):
        self.assertEqual(self.classifier.classify(appearance((137,77,239)),POINTS)["label"],"TEAM_B")

    def test_referee_team_and_role_separate(self):
        r = self.classifier.classify(appearance((31,242,255)),POINTS)
        self.assertEqual(r["label"],"REFEREE")
        self.assertEqual(r["role"],"REFEREE")
        self.assertIsNone(r["team"])
        self.assertFalse(r["eligible"])

    def test_yellow_without_dark_lower_is_unknown(self):
        r = self.classifier.classify(appearance((31,242,255),dark_lower=False),POINTS)
        self.assertEqual(r["label"],"UNKNOWN")
        self.assertEqual(r["reason"],"YELLOW_WITHOUT_DARK_LOWER_BODY")

    def test_tiny_box_unknown(self):
        self.assertEqual(self.classifier.classify(appearance((31,242,255)),[[0,0],[3,3]])["label"],"UNKNOWN")

    def test_cyan_goalkeeper_color_is_not_forced_into_a_team(self):
        r = self.classifier.classify(appearance((91,165,192)),POINTS)
        self.assertEqual(r["label"],"UNKNOWN")
        self.assertEqual(r["role"],"UNKNOWN")

    def test_ambiguous_color_tie_unknown(self):
        image = appearance((137,77,239))
        for xa,xb,color in [(9,20,(122,143,123)),(20,30,(165,203,181))]:
            image[15:50,xa:xb] = cv2.cvtColor(np.uint8([[color]]),cv2.COLOR_HSV2BGR)[0,0]
        r = self.classifier.classify(image,POINTS)
        self.assertEqual(r["label"],"UNKNOWN")
        self.assertEqual(r["reason"],"AMBIGUOUS_COLORS")

    def test_occluded_torso_abstains(self):
        r = self.classifier.classify(appearance((137,77,239)),POINTS,[POINTS])
        self.assertEqual(r["label"],"UNKNOWN")
        self.assertEqual(r["reason"],"TORSO_OCCLUDED")

    def test_temporary_flip_does_not_change_stable_memory(self):
        temporal = TemporalClassifier(CONFIG["temporal"])
        a = metadata("TEAM_A",eligible=True,reason="HSV_APPEARANCE")
        b = metadata("TEAM_B",eligible=True,reason="HSV_APPEARANCE")
        for frame in range(10):
            r = temporal.update(1,frame,a)
        self.assertEqual(r["label"],"TEAM_A")
        r = temporal.update(1,10,b)
        self.assertEqual(r["label"],"UNKNOWN")
        self.assertEqual(r["stable_memory"],"TEAM_A")
        self.assertFalse(r["eligible"])
        self.assertEqual(temporal.update(1,11,a)["label"],"TEAM_A")

    def test_unknown_never_inherits_eligible_memory(self):
        temporal = TemporalClassifier(CONFIG["temporal"])
        for frame in range(5):
            temporal.update(1,frame,metadata("TEAM_A",eligible=True))
        r = temporal.update(1,5,metadata("UNKNOWN",eligible=False))
        self.assertEqual(r["label"],"UNKNOWN")
        self.assertFalse(r["eligible"])

    def test_long_track_gap_resets_history(self):
        temporal = TemporalClassifier(CONFIG["temporal"])
        for frame in range(5):
            temporal.update(1,frame,metadata("TEAM_A",eligible=True))
        r = temporal.update(1,30,metadata("TEAM_B",eligible=True))
        self.assertEqual(r["label"],"UNKNOWN")
        self.assertEqual(r["stable_memory"],"UNKNOWN")

    def test_referee_veto_immediate_despite_team_history(self):
        temporal = TemporalClassifier(CONFIG["temporal"])
        for frame in range(5):
            temporal.update(1,frame,metadata("TEAM_A",eligible=True))
        r = temporal.update(1,5,metadata("REFEREE",eligible=False))
        self.assertEqual(r["role"],"REFEREE")
        self.assertFalse(r["eligible"])

    def test_empty_and_referee_only_no_candidates(self):
        for objects in [[],[obj(1,team=None,role="REFEREE",eligible=True)]]:
            candidate,status,_ = select_candidate(objects,[[20,19],[22,21]])
            self.assertIsNone(candidate)
            self.assertEqual(status,"INSUFFICIENT_EVIDENCE")

    def test_nearest_referee_excluded_even_if_eligible_flag_wrong(self):
        objects = [obj(1,team=None,role="REFEREE",eligible=True),obj(2,points=[[30,0],[40,20]])]
        candidate,_,_ = select_candidate(objects,[[19,19],[21,21]])
        self.assertEqual(candidate["track_id"],2)

    def test_far_player_after_referee_exclusion_is_insufficient(self):
        objects = [obj(1,team=None,role="REFEREE"),obj(2,points=[[100,0],[110,20]])]
        candidate,status,_ = select_candidate(objects,[[19,19],[21,21]])
        self.assertIsNone(candidate)
        self.assertEqual(status,"INSUFFICIENT_EVIDENCE")

    def test_unknown_stale_missing_multiple_balls_excluded(self):
        for objects,ball,count in [([obj(1,team="UNKNOWN")],[[20,19],[22,21]],1),
                                   ([dict(obj(1),age=1)],[[20,19],[22,21]],1),
                                   ([obj(1)],None,0),([obj(1)],[[20,19],[22,21]],2)]:
            self.assertIsNone(select_candidate(objects,ball,ball_count=count)[0])

    def test_upstream_match_empty_guard_and_isolated_accumulation(self):
        import run_tryolabs_broadcast as legacy
        from run_tryolabs_classification_v2 import objects_to_players,safe_match_update
        from norfair import Detection
        u = legacy.import_upstream()
        old = legacy.new_state(u)
        old["match"].update([],u["Ball"](None))
        frozen = (old["match"].duration,old["teams"][0].possession,old["teams"][1].possession)
        teams = [u["Team"](name="TEAM_A",abbreviation="TMA"),u["Team"](name="TEAM_B",abbreviation="TMB")]
        match = u["Match"](teams[0],teams[1],fps=25);match.team_possession=None
        ball = u["Ball"](Detection(points=np.array([[19,19],[21,21]]),data={}))
        objects = [obj(1,team=None,role="REFEREE",eligible=True)]
        players = objects_to_players(u,objects,teams)
        self.assertEqual(players,[])
        safe_match_update(u,match,players,ball)
        self.assertIsNone(match.closest_player)
        self.assertIsNone(match.team_possession)
        self.assertEqual(match.ball_distance_threshold,45)
        self.assertEqual(match.possesion_counter_threshold,20)
        self.assertEqual((old["match"].duration,old["teams"][0].possession,old["teams"][1].possession),frozen)


if __name__ == "__main__":
    unittest.main()
