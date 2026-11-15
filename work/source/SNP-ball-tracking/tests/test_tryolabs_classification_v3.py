"""Synthetic correctness and unchanged cached-input replay, not match accuracy."""
import copy
import gzip
import hashlib
import json
from pathlib import Path
import sqlite3
import sys
import tempfile
import unittest
from unittest.mock import patch
import zlib

import cv2
import numpy as np

ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT/"starter"))
from tryolabs_uniform_v3 import UniformClassifierV3,TemporalClassifierV3,assignment
from tryolabs_uniform_v2 import select_candidate
from tryolabs_v3_data import full_frames,digest
import run_tryolabs_classification_v3 as runner
from evaluate_tryolabs_v3_external import evaluate

CONFIG=json.loads(runner.V3_CONFIG.read_text());POLICY=json.loads(runner.V3_POLICY.read_text())
BOX=[[80,10],[140,210]]


def appearance(colors,box=BOX,dark=True,green=True):
    hsv=np.full((240,240,3),(38,130,145) if green else (0,0,180),np.uint8)
    (x1,y1),(x2,y2)=box
    hsv[y1:y2,x1:x2]=(15,70,160)
    xa,xb=int(x1+.15*(x2-x1)),int(x1+.85*(x2-x1))
    ya,yb=int(y1+.15*(y2-y1)),int(y1+.50*(y2-y1))
    for i,color in enumerate(colors):
        a=xa+round(i*(xb-xa)/len(colors));b=xa+round((i+1)*(xb-xa)/len(colors))
        hsv[ya:yb,a:b]=color
    if dark:
        hsv[int(y1+.50*(y2-y1)):int(y1+.72*(y2-y1)),xa:xb]=(0,0,40)
        hsv[int(y1+.80*(y2-y1)):int(y1+.98*(y2-y1)),xa:xb]=(0,0,40)
    return cv2.cvtColor(hsv,cv2.COLOR_HSV2BGR)


class UniformV3Tests(unittest.TestCase):
    def setUp(self):self.c=UniformClassifierV3(CONFIG,POLICY)
    def a(self):return self.c.classify(appearance([(122,140,127),(166,205,174)]),BOX)
    def test_navy_and_burgundy_team_a(self):
        r=self.a();self.assertEqual(r["label"],"TEAM_A");self.assertEqual(r["team_name"],"FC Barcelona")
    def test_lavender_with_pink_team_b(self):
        r=self.c.classify(appearance([(140,70,238),(162,65,187)]),BOX)
        self.assertEqual(r["label"],"TEAM_B");self.assertGreater(r["components"]["pink"],0)
    def test_pink_alone_does_not_assign_team_b(self):
        self.assertEqual(self.c.classify(appearance([(166,65,187)]),BOX)["label"],"UNKNOWN")
    def test_bright_achromatic_shirt_not_team_b(self):
        self.assertEqual(self.c.classify(appearance([(0,0,255)]),BOX)["label"],"UNKNOWN")
    def test_dark_alone_not_team_a(self):
        self.assertEqual(self.c.classify(appearance([(0,0,40)]),BOX)["label"],"UNKNOWN")
    def test_shared_lavender_burgundy_abstains(self):
        r=self.c.classify(appearance([(145,110,145)]),BOX)
        self.assertEqual(r["label"],"UNKNOWN");self.assertEqual(r["reason"],"SHARED_COLOR_UNRESOLVED")
    def test_small_roi_and_occlusion(self):
        self.assertEqual(self.c.classify(appearance([(140,70,238)]),[[80,10],[83,14]])["label"],"UNKNOWN")
        self.assertEqual(self.c.classify(appearance([(140,70,238)]),BOX,[BOX])["reason"],"TORSO_OCCLUDED")
    def test_referee_needs_shorts_context_and_temporal_support(self):
        r=self.c.classify(appearance([(31,242,255)]),BOX)
        self.assertEqual(r["role"],"REFEREE");self.assertFalse(r["eligible"])
        t=TemporalClassifierV3(POLICY)
        self.assertEqual(t.update(1,0,r)["label"],"UNKNOWN")
        self.assertEqual(t.update(1,1,r)["label"],"UNKNOWN")
        self.assertEqual(t.update(1,2,r)["label"],"REFEREE")
    def test_yellow_without_dark_or_green_unknown(self):
        for kwargs in [dict(dark=False),dict(green=False)]:
            self.assertEqual(self.c.classify(appearance([(31,242,255)],**kwargs),BOX)["label"],"UNKNOWN")
    def test_yellow_small_border_staff_not_referee(self):
        box=[[0,100],[12,150]]
        r=self.c.classify(appearance([(31,242,255)],box),box)
        self.assertEqual(r["label"],"UNKNOWN");self.assertEqual(r["reason"],"SIDELINE_OR_CLIPPED_YELLOW")
    def test_large_bottom_clipped_box_can_use_existing_v2_lower_roi(self):
        box=[[80,10],[140,240]]
        im=appearance([(31,242,255)],box,dark=False)
        # Fixed shorts ROI is yellow; only the lower visible strip is dark.
        im[190:240,89:131]=(20,20,20)
        r=self.c.classify(im,box)
        self.assertEqual(r["label"],"REFEREE");self.assertTrue(r["lower_body_dark_fallback_used"])
        self.assertLess(r["shorts_dark_fraction"],.1);self.assertGreater(r["dark_lower_support"],.1)
        uncut=[[80,10],[140,220]]
        r=self.c.classify(im,uncut)
        self.assertEqual(r["label"],"UNKNOWN");self.assertFalse(r["large_bottom_clipped"])
    def test_cyan_goalkeeper_not_forced_to_field_team(self):
        r=self.c.classify(appearance([(91,165,192)]),BOX)
        self.assertEqual(r["team"],"UNKNOWN");self.assertEqual(r["role"],"UNKNOWN")
    def test_team_and_role_are_separate_at_unverified_location(self):
        r=self.c.classify(appearance([(122,140,127),(166,205,174)],green=False),BOX)
        self.assertEqual(r["team"],"TEAM_A");self.assertEqual(r["role"],"UNKNOWN");self.assertFalse(r["eligible"])
    def test_single_color_requires_recent_strong_a(self):
        t=TemporalClassifierV3(POLICY)
        single=self.c.classify(appearance([(122,140,127)]),BOX)
        self.assertEqual(t.update(1,0,single)["label"],"UNKNOWN")
        for i in range(1,5):r=t.update(1,i,self.a())
        self.assertEqual(r["label"],"TEAM_A")
        r=t.update(1,5,single)
        self.assertEqual(r["label"],"TEAM_A");self.assertEqual(r["observed_label"],"UNKNOWN")
        self.assertEqual(r["classification_source"],"HSV_CURRENT_PLUS_TRACK")
        r=t.update(1,30,single);self.assertEqual(r["label"],"UNKNOWN")
    def test_history_does_not_fill_zero_evidence_or_conflict(self):
        t=TemporalClassifierV3(POLICY)
        for i in range(5):t.update(1,i,self.a())
        no=self.c.classify(appearance([(0,0,120)]),BOX)
        self.assertEqual(t.update(1,5,no)["label"],"UNKNOWN")
        b=self.c.classify(appearance([(140,70,238)]),BOX)
        self.assertEqual(t.update(1,6,b)["label"],"UNKNOWN")
        single=self.c.classify(appearance([(122,140,127)]),BOX)
        self.assertEqual(t.update(1,7,single)["label"],"UNKNOWN")
    def test_stale_and_missing_input_safe(self):
        for image,box in [(None,BOX),(appearance([(0,0,40)]),None),(appearance([(0,0,40)]),"bad")]:
            self.assertEqual(self.c.classify(image,box)["label"],"UNKNOWN")
        t=TemporalClassifierV3(POLICY)
        self.assertFalse(t.update(1,0,self.a(),fresh=False)["eligible"])
        self.assertEqual(t.update(1,1,None)["label"],"UNKNOWN")
    def test_referee_never_selected_even_if_flag_wrong(self):
        r=assignment("REFEREE","test");r.update(track_id=1,points=[[0,0],[20,20]],age=0,eligible=True)
        p=assignment("TEAM_A","test");p.update(track_id=2,points=[[30,0],[40,20]],age=0)
        chosen,_,_=select_candidate([r,p],[[19,19],[21,21]])
        self.assertEqual(chosen["track_id"],2)
        self.assertIsNone(select_candidate([r],[[19,19],[21,21]])[0])
    def test_v2_source_and_settings_unchanged(self):
        paths=[ROOT/"starter/tryolabs_uniform_v2.py",runner.V2_CONFIG,ROOT/"starter/configs/tryolabs_classification_v2_policy.json"]
        before=[digest(p) for p in paths];self.a()
        self.assertEqual(before,[digest(p) for p in paths])
    def test_confidence_is_never_probability(self):self.assertIsNone(self.a()["classification_confidence"])


class CacheReplayTests(unittest.TestCase):
    def test_full_cache_read_is_snapshot_only(self):
        with tempfile.TemporaryDirectory() as temp:
            root=Path(temp);cache=root/"v2";cache.mkdir();folder=cache/"chunks/000000005-000000006";folder.mkdir(parents=True)
            (folder/"chunk.json").write_text(json.dumps(dict(start=5,end=6)))
            obj=dict(track_id=1,points=BOX,age=0)
            for name,r in [("tracking",dict(frame0=5,scene_id=2,players=[obj])),("objects",dict(frame0=5,objects=[dict(track_id=1,label="UNKNOWN")]))]:
                with gzip.open(folder/f"{name}.jsonl.gz","wt") as f:f.write(json.dumps(r)+"\n")
            db=sqlite3.connect(cache/"run.sqlite");db.execute("PRAGMA journal_mode=WAL");db.execute("CREATE TABLE detections(frame0 INTEGER PRIMARY KEY,payload BLOB)")
            raw=dict(status="ok",balls=[]);db.execute("INSERT INTO detections VALUES (?,?)",(5,zlib.compress(json.dumps(raw).encode())));db.commit()
            before={str(p):digest(p) for p in cache.rglob('*') if p.is_file()}
            result=list(full_frames(cache,[(5,6)],root/"snapshot"))
            self.assertEqual(result[0]["objects"][0]["frozen_v2_label"],"UNKNOWN")
            self.assertEqual(before,{str(p):digest(p) for p in cache.rglob('*') if p.is_file()})
            db.close()
    def test_same_cached_video_and_tracks_reproduce_outputs(self):
        with tempfile.TemporaryDirectory() as temp:
            root=Path(temp);video=root/"source.mp4"
            writer=cv2.VideoWriter(str(video),cv2.VideoWriter_fourcc(*"mp4v"),25,(240,240));self.assertTrue(writer.isOpened())
            for i in range(8):writer.write(appearance([(122,140,127),(166,205,174)] if i<4 else [(122,140,127)]))
            writer.release()
            frames=[dict(sequence="synthetic",frame0=i,source="synthetic",balls=[],objects=[dict(track_id=1,points=BOX,age=0,frozen_v2_label=None)]) for i in range(8)]
            original=copy.deepcopy(frames)
            with patch.object(runner,"VIDEO",video):
                for name in ['one','two']:
                    out=root/name;out.mkdir();runner.run(frames,out,video=False,visuals=False)
            self.assertEqual(frames,original)
            for filename in ["objects.jsonl","frames.csv","input_frames.jsonl","summary.json"]:
                self.assertEqual((root/'one'/filename).read_bytes(),(root/'two'/filename).read_bytes())


class ExternalEvaluationTests(unittest.TestCase):
    def test_empty_or_pseudo_truth_is_rejected(self):
        with self.assertRaises(ValueError):evaluate([])
        with self.assertRaises(ValueError):evaluate([dict(game_id='external',evaluation_scope='external_match',label_source='algorithm',truth='TEAM_A',prediction='TEAM_A')])
    def test_provided_match_manual_labels_are_rejected(self):
        with self.assertRaises(ValueError):evaluate([dict(game_id='BAR-ATH-20240824',evaluation_scope='external_match',label_source='human_external',truth='TEAM_A',prediction='TEAM_A')])
    def test_external_metrics_including_unknown(self):
        rows=[dict(game_id='synthetic_external',evaluation_scope='external_match',label_source='human_external',truth=k,prediction=k) for k in ['TEAM_A','TEAM_B','REFEREE','UNKNOWN']]
        r=evaluate(rows);self.assertEqual(r['macro_f1'],1.);self.assertEqual(r['known_prediction_coverage'],.75)


if __name__=="__main__":unittest.main()
