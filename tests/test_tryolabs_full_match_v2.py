"""Full-match storage/discontinuity checks without competition-video labels."""
import copy
import hashlib
import json
from pathlib import Path
import pickle
import sqlite3
import sys
import tempfile
import unittest
from unittest.mock import patch

import numpy as np

ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT/"starter"))
import tryolabs_full_match_v2 as full


class FullMatchTests(unittest.TestCase):
    def test_frame_ranges_exact_inclusive(self):
        self.assertEqual(full.ranges([0,1,2,5,6,9]),[[0,2],[5,6],[9,9]])
        self.assertEqual(full.ranges([]),[])

    def test_cache_is_single_shared_primary_key(self):
        with tempfile.TemporaryDirectory() as temp:
            db=full.connection(Path(temp));raw=dict(frame0=3,persons=[],balls=[],status="ok")
            db.execute("INSERT INTO detections VALUES (?,?)",(3,full.pack(raw)));db.commit()
            with self.assertRaises(sqlite3.IntegrityError):
                db.execute("INSERT INTO detections VALUES (?,?)",(3,full.pack(raw)))
            self.assertEqual(full.cached_records(Path(temp),3,4),[(3,raw)])
            db.close()

    def test_lock_excludes_duplicate_worker_but_allows_pipeline(self):
        with tempfile.TemporaryDirectory() as temp:
            p=Path(temp)
            with full.stage_lock(p,"detect"):
                with self.assertRaises(RuntimeError):
                    with full.stage_lock(p,"detect"):pass
                with full.stage_lock(p,"process"):pass

    def test_scene_diagnostics_remains_unverified(self):
        scene=full.SceneDiagnostics();a=np.full((108,192,3),[0,100,0],np.uint8)
        first=scene.update(a,0,[]);same=scene.update(a,1,[])
        changed=scene.update(np.full_like(a,230),2,[])
        self.assertFalse(first["cut_suspect"]);self.assertFalse(same["cut_suspect"])
        self.assertTrue(changed["cut_suspect"])
        self.assertEqual(changed["play_state"],"UNVERIFIED")
        self.assertEqual(changed["replay_state"],"UNVERIFIED")

    def test_checkpoint_retains_diagnostics_exactly(self):
        scene=full.SceneDiagnostics();rng=np.random.default_rng(4)
        for i in range(3):scene.update(rng.integers(0,255,(108,192,3),dtype=np.uint8),i,[])
        restored=pickle.loads(pickle.dumps(scene))
        im=rng.integers(0,255,(108,192,3),dtype=np.uint8)
        self.assertEqual(scene.update(im,3,[]),restored.update(im,3,[]))

    def test_engine_resets_on_frame_gap_and_cut(self):
        u=full.legacy.import_upstream();config=json.loads(full.CONFIG.read_text());engine=full.Engine(u,config)
        image=np.full((108,192,3),[0,100,0],np.uint8)
        raw=lambda i:dict(frame0=i,persons=[],balls=[],status="ok")
        with patch.dict(engine.u,{"update_motion_estimator":lambda *args,**kwargs:None}):
            row,_,_=engine.step(image,raw(0));scene=row["scene_id"]
            row,_,_=engine.step(image,raw(2))
            self.assertGreater(row["scene_id"],scene)
            self.assertEqual(row["reset_reason"],"FRAME_GAP")
            row,_,_=engine.step(np.full_like(image,230),raw(3))
            self.assertEqual(row["reset_reason"],"SHOT_CUT_HEURISTIC")
            self.assertEqual(row["fresh_run"],0)

    def test_atomic_json_replaces_complete_document(self):
        with tempfile.TemporaryDirectory() as temp:
            p=Path(temp)/"status.json"
            full.atomic_json(p,dict(processed=1));full.atomic_json(p,dict(processed=2))
            self.assertEqual(json.loads(p.read_text()),dict(processed=2))
            self.assertFalse(p.with_suffix('.json.tmp').exists())

    def test_nonfinite_values_cannot_silently_enter_diagnostic_cache(self):
        with self.assertRaises(ValueError):full.pack(dict(value=float('nan')))

    def test_optimized_hsv_mask_exactly_matches_frozen_numpy_boxes(self):
        from tryolabs_uniform_v2 import mask_range
        config=json.loads(full.CONFIG.read_text())
        rng=np.random.default_rng(123);hsv=rng.integers(0,256,(600,800,3),dtype=np.uint8)
        for bounds in config["hsv_components"].values():
            np.testing.assert_array_equal(mask_range(hsv,bounds),full.exact_mask_range(hsv,bounds))
        bounds=[dict(lower=[0,0,0],upper=[255,255,255]),dict(lower=[70,90,110],upper=[70,90,110])]
        np.testing.assert_array_equal(mask_range(hsv,bounds),full.exact_mask_range(hsv,bounds))

    def test_crop_copy_adapter_preserves_pixels_even_at_image_boundaries(self):
        full.legacy.import_upstream()
        from inference.box import Box
        import inference.base_classifier as base
        from unittest.mock import patch
        image=np.random.default_rng(6).integers(0,255,(108,192,3),dtype=np.uint8)
        originals=[Box(a,b,image).img for a,b in [((3,5),(50,100)),((-3,-5),(200,120)),((4,4),(4,4))]]
        original_mask=__import__('tryolabs_uniform_v2').mask_range
        original_box=base.Box
        try:
            full.install_exact_cpu_adapters()
            actual=[base.Box(a,b,image).img for a,b in [((3,5),(50,100)),((-3,-5),(200,120)),((4,4),(4,4))]]
            for a,b in zip(originals,actual):np.testing.assert_array_equal(a,b)
        finally:
            base.Box=original_box;__import__('tryolabs_uniform_v2').mask_range=original_mask


if __name__=="__main__":unittest.main()
