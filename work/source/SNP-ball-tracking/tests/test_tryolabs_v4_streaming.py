"""Streaming/resume correctness on synthetic cached inputs, not match accuracy."""
import copy
import csv
import gzip
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

ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT/"starter"))
import run_tryolabs_v4_streaming as runner
from tryolabs_v4_stream_io import signature,lock,read_jsonl,digest
from tryolabs_v4_stream_state import encode,decode
from test_tryolabs_classification_v3 import appearance,BOX
from test_tryolabs_classification_v4 import candidate


def fixture(root,n=32,scene_at=None):
    video=root/"original.mp4";writer=cv2.VideoWriter(str(video),cv2.VideoWriter_fourcc(*"mp4v"),25,(240,240))
    assert writer.isOpened()
    image=appearance([(122,140,127),(166,205,174)])
    for _ in range(n):writer.write(image)
    writer.release();cache=root/"v2";(cache/"chunks").mkdir(parents=True)
    db=sqlite3.connect(cache/"run.sqlite");db.execute("PRAGMA journal_mode=WAL")
    db.executescript("CREATE TABLE detections(frame0 INTEGER PRIMARY KEY,payload BLOB); CREATE TABLE chunks(start INTEGER PRIMARY KEY,end INTEGER,path TEXT,sha256 TEXT);")
    raw=dict(status="ok",balls=[dict(xmin=79,ymin=209,xmax=81,ymax=211,confidence=.9)])
    prepared=[]
    for a in range(0,n,11):
        b=min(n,a+11);folder=cache/"chunks"/f'{a:09d}-{b:09d}';folder.mkdir()
        (folder/"chunk.json").write_text(json.dumps(dict(start=a,end=b)))
        with gzip.open(folder/"tracking.jsonl.gz","wt") as t,gzip.open(folder/"objects.jsonl.gz","wt") as o:
            for i in range(a,b):
                scene=int(scene_at is not None and i>=scene_at)
                obj=dict(track_id=1,points=BOX,age=0,legacy_class="BLUE")
                t.write(json.dumps(dict(frame0=i,scene_id=scene,players=[obj]))+"\n")
                o.write(json.dumps(dict(frame0=i,objects=[dict(track_id=1,label="UNKNOWN")]))+"\n")
                db.execute("INSERT INTO detections VALUES (?,?)",(i,zlib.compress(json.dumps(raw).encode())))
                prepared.append(dict(frame0=i,sequence=f'full:{scene}',objects=[dict(obj,frozen_v2_label="UNKNOWN")],balls=raw["balls"],source="full_cache"))
        db.execute("INSERT INTO chunks VALUES (?,?,?,?)",(a,b,str(folder.relative_to(cache)),"unused-tracker-state"))
    db.commit()
    return cache,video,db,prepared


class StreamingTests(unittest.TestCase):
    def test_windows_output_requires_d_drive(self):
        runner.validate_windows_output(r"D:\MTAH2027\v4_full\new_run")
        for path in [r"C:\SNP\data\new_run",r"new_run",r"\\server\share\new_run"]:
            with self.assertRaises(ValueError):runner.validate_windows_output(path)
    def test_portable_state_includes_numpy_deques_integer_keys_and_roles(self):
        engine=runner.new_engine()
        for i in range(10):engine.t4.update(15,i,candidate())
        for i in range(12):engine.monitor.update(appearance([(91,150,196)]))
        engine.e3.team="TEAM_A";engine.e3.count=19
        exported=engine.export();other=runner.new_engine();other.restore(json.loads(json.dumps(exported)))
        self.assertEqual(exported,other.export())
        self.assertEqual(other.e3.update("TEAM_A"),"TEAM_A")
        self.assertEqual(other.t4.update(15,10,candidate(goal=False,field=False))["final_label"],"GK_B")
        self.assertEqual(len(other.monitor.history),10)
    def test_expired_histories_are_bounded_without_chunk_reset(self):
        engine=runner.new_engine();image=appearance([(122,140,127),(166,205,174)])
        for i in range(35):
            raw=dict(frame0=i,sequence="same",source="synthetic",balls=[],objects=[dict(track_id=i,points=BOX,age=0)])
            engine.process(raw,image)
        self.assertLessEqual(len(engine.t3.tracks),11);self.assertLessEqual(len(engine.t4.tracks),11)
        self.assertEqual(engine.epoch,1)
    def test_cache_resume_preserves_20_frames_and_original_db_wal_shm(self):
        with tempfile.TemporaryDirectory() as temp:
            root=Path(temp);cache,video,db,_=fixture(root,32)
            try:
                before={str(p):signature(p) for p in cache.rglob("*") if p.is_file()}
                for name,limit in [("continuous",None),("resumed",2)]:
                    runner.run(root/name,cache=cache,video_path=video,end=32,chunk_size=7,video=False,limit_chunks=limit)
                runner.run(root/"resumed",cache=cache,video_path=video,end=32,chunk_size=7,video=False,resume=True)
                for name in ["objects.jsonl.gz","frames.jsonl.gz","input_frames.jsonl.gz"]:
                    self.assertEqual(list(runner.iter_output(root/"continuous",name)),list(runner.iter_output(root/"resumed",name)))
                frames=list(runner.iter_output(root/"resumed","frames.jsonl.gz"))
                self.assertEqual(frames[21]["v4_status"],"PROXIMITY_20_FRAME_EVIDENCE")
                self.assertEqual(frames[20]["v4_status"],"PROXIMITY_PENDING")
                self.assertTrue(runner.verify(root/"resumed")["passed"])
                self.assertEqual(before,{str(p):signature(p) for p in cache.rglob("*") if p.is_file()})
                a,_=runner.committed(root/"continuous",json.loads((root/"continuous/manifest.json").read_text()))
                b,_=runner.committed(root/"resumed",json.loads((root/"resumed/manifest.json").read_text()))
                for (p1,_),(p2,_) in zip(a,b):
                    for n in ["state.json","objects.jsonl.gz","frames.jsonl.gz"]:self.assertEqual(digest(p1/n),digest(p2/n))
            finally:db.close()
    def test_mid_chunk_interruption_discards_only_uncommitted_tail(self):
        with tempfile.TemporaryDirectory() as temp:
            root=Path(temp);cache,video,db,_=fixture(root,24)
            try:
                def fail(stage,n):
                    if stage=="after_frame" and n==10:raise KeyboardInterrupt()
                with self.assertRaises(KeyboardInterrupt):runner.run(root/"out",cache=cache,video_path=video,end=24,chunk_size=7,video=False,hook=fail)
                self.assertEqual(json.loads((root/"out/failure.json").read_text())["committed_frames"],7)
                runner.run(root/"out",cache=cache,video_path=video,end=24,chunk_size=7,video=False,resume=True)
                runner.run(root/"reference",cache=cache,video_path=video,end=24,chunk_size=7,video=False)
                self.assertEqual(list(runner.iter_output(root/"out","objects.jsonl.gz")),list(runner.iter_output(root/"reference","objects.jsonl.gz")))
                self.assertTrue(list((root/"out/interrupted").iterdir()))
            finally:db.close()
    def test_commit_survives_crash_before_global_checkpoint(self):
        with tempfile.TemporaryDirectory() as temp:
            root=Path(temp);cache,video,db,_=fixture(root,16)
            try:
                def fail(stage,n):
                    if stage=="after_commit" and n==7:raise RuntimeError("Power-loss boundary")
                with self.assertRaises(RuntimeError):runner.run(root/"out",cache=cache,video_path=video,end=16,chunk_size=7,video=False,hook=fail)
                self.assertFalse((root/"out/checkpoint.json").exists())
                runner.run(root/"out",cache=cache,video_path=video,end=16,chunk_size=7,video=False,resume=True)
                self.assertEqual(runner.verify(root/"out")["committed_frames"],16)
            finally:db.close()
    def test_scene_resets_but_chunk_boundaries_do_not(self):
        with tempfile.TemporaryDirectory() as temp:
            root=Path(temp);cache,video,db,_=fixture(root,30,scene_at=24)
            try:
                runner.run(root/"out",cache=cache,video_path=video,end=30,chunk_size=7,video=False)
                rows=list(runner.iter_output(root/"out","frames.jsonl.gz"))
                self.assertEqual(rows[7]["v4_epoch"],rows[6]["v4_epoch"])
                self.assertEqual(rows[24]["v4_epoch"],rows[23]["v4_epoch"]+1)
                self.assertEqual(rows[24]["v4_status"],"INSUFFICIENT_EVIDENCE")
            finally:db.close()
    def test_source_gap_and_resume_config_or_input_changes_are_rejected(self):
        with tempfile.TemporaryDirectory() as temp:
            root=Path(temp);cache,video,db,_=fixture(root,16)
            try:
                runner.run(root/"out",cache=cache,video_path=video,end=16,chunk_size=7,video=False,limit_chunks=1)
                with self.assertRaisesRegex(ValueError,"settings changed"):
                    runner.run(root/"out",cache=cache,video_path=video,end=16,chunk_size=8,video=False,resume=True)
                with patch.object(runner,"fingerprint",return_value={}):
                    with self.assertRaisesRegex(ValueError,"hash changed"):
                        runner.run(root/"out",cache=cache,video_path=video,end=16,chunk_size=7,video=False,resume=True)
                db.execute("DELETE FROM chunks WHERE start=11");db.commit()
                with self.assertRaisesRegex(ValueError,"changed"):
                    runner.run(root/"out",cache=cache,video_path=video,end=16,chunk_size=7,video=False,resume=True)
                with self.assertRaisesRegex(ValueError,"tracking cache missing"):
                    runner.run(root/"gap",cache=cache,video_path=video,end=16,chunk_size=7,video=False)
            finally:db.close()
    def test_corrupt_output_is_not_overwritten_or_accepted(self):
        with tempfile.TemporaryDirectory() as temp:
            root=Path(temp);cache,video,db,_=fixture(root,8)
            try:
                runner.run(root/"out",cache=cache,video_path=video,end=8,chunk_size=7,video=False)
                next((root/"out/chunks").glob("*/state.json")).write_text("corrupt")
                with self.assertRaisesRegex(ValueError,"checksum"):runner.verify(root/"out")
                with self.assertRaises(ValueError):runner.run(root/"out",cache=cache,video_path=video,end=8,video=False)
            finally:db.close()
    def test_duplicate_source_frame_reports_exact_failed_frame(self):
        with tempfile.TemporaryDirectory() as temp:
            root=Path(temp);cache,video,db,_=fixture(root,8)
            try:
                path=next((cache/"chunks").glob("*/tracking.jsonl.gz"))
                rows=list(read_jsonl(path));rows[2]["frame0"]=1
                with gzip.open(path,"wt") as f:
                    for r in rows:f.write(json.dumps(r)+"\n")
                with self.assertRaisesRegex(ValueError,"frame 2"):
                    runner.run(root/"out",cache=cache,video_path=video,end=8,chunk_size=7,video=False)
                failure=json.loads((root/"out/failure.json").read_text())
                self.assertEqual(failure["failed_frame0"],2);self.assertEqual(failure["committed_frames"],0)
            finally:db.close()
    def test_video_assembly_and_completion_do_not_claim_full_match_for_sample(self):
        with tempfile.TemporaryDirectory(prefix="v4 stream space ") as temp:
            root=Path(temp);cache,video,db,_=fixture(root,4)
            try:
                runner.run(root/"out",cache=cache,video_path=video,end=4,chunk_size=2,video=True)
                report=runner.assemble(root/"out")
                self.assertTrue(report["comparison_video_complete"]);self.assertFalse(report["full_match_complete"])
                with self.assertRaisesRegex(ValueError,"not complete"):runner.verify(root/"out",require_full=True)
                self.assertEqual(runner.decode_video(root/"out/assembled/comparison_full.mp4")["decoded_frames"],4)
            finally:db.close()
    def test_lock_prevents_two_writers(self):
        with tempfile.TemporaryDirectory() as temp:
            with lock(Path(temp)):
                with self.assertRaises(RuntimeError):
                    with lock(Path(temp)):pass
    def test_full_completion_gate_requires_assembled_video(self):
        # A tiny synthetic full extent exercises the completion gate, not the match.
        with tempfile.TemporaryDirectory() as temp, patch.object(runner,"TOTAL",4):
            root=Path(temp);cache,video,db,_=fixture(root,4)
            try:
                runner.run(root/"out",cache=cache,video_path=video,end=4,chunk_size=2,video=True)
                runner.run(root/"paused",cache=cache,video_path=video,end=4,chunk_size=2,video=True,limit_chunks=1)
                runner.run(root/"paused",cache=cache,video_path=video,end=4,chunk_size=2,video=True,resume=True)
                for folder in (root/"out/chunks").iterdir():
                    self.assertEqual(digest(folder/"comparison.mp4"),digest(root/"paused/chunks"/folder.name/"comparison.mp4"))
                with self.assertRaisesRegex(ValueError,"not complete"):runner.verify(root/"out",require_full=True)
                runner.assemble(root/"out")
                self.assertTrue(runner.verify(root/"out",require_full=True,decode=True)["full_match_complete"])
            finally:db.close()


if __name__=="__main__":unittest.main()
