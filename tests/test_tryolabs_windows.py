"""Windows/CUDA branches are mocked; video resume is exercised on CPU only."""
from contextlib import ExitStack
import gzip
import json
from pathlib import Path
import sqlite3
import sys
import tempfile
from types import SimpleNamespace
import unittest
from unittest.mock import Mock,patch

import cv2
import numpy as np

ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT/"starter"))
import tryolabs_runtime as rt
import tryolabs_full_match_v2 as full
import analyze_tryolabs_full_match_v2 as analysis
import benchmark_tryolabs_windows as bench


class FakeModel:
    def __init__(self,name,oom_above=None,error=None):
        self.name=name;self.device="cpu";self.calls=[];self.oom_above=oom_above;self.error=error
    def eval(self):return self
    def to(self,device):self.device=str(device);return self
    def __call__(self,images,size=640):
        self.calls.append((self.device,len(images),size))
        if self.error:raise RuntimeError(self.error)
        if self.oom_above is not None and len(images)>self.oom_above:raise RuntimeError("CUDA out of memory")
        values=[f"{self.name}:{i}" for i in images]
        return SimpleNamespace(pandas=lambda:SimpleNamespace(xyxy=values))


class RuntimeTests(unittest.TestCase):
    def cuda_mocks(self):
        stack=ExitStack()
        stack.enter_context(patch.object(rt,"resolve_device",return_value="cuda:0"))
        stack.enter_context(patch.object(rt,"install_cpu_nms"))
        stack.enter_context(patch.object(rt,"memory_info",return_value={}))
        stack.enter_context(patch.object(rt.torch.cuda,"empty_cache"))
        stack.enter_context(patch.object(rt.torch.cuda,"is_available",return_value=True))
        return stack

    def test_explicit_cuda_never_silently_falls_back(self):
        with patch.object(rt.torch.cuda,"is_available",return_value=False):
            with self.assertRaisesRegex(RuntimeError,"No CPU fallback"):
                rt.resolve_device("cuda")

    def test_cuda_index_is_validated(self):
        with patch.object(rt.torch.cuda,"is_available",return_value=True),patch.object(rt.torch.cuda,"device_count",return_value=1):
            self.assertEqual(rt.resolve_device("cuda"),"cuda:0")
            with self.assertRaises(RuntimeError):rt.resolve_device("cuda:1")

    def test_single_residency_parks_inactive_models_on_cpu(self):
        with self.cuda_mocks():
            pair=rt.DetectorPair(lambda p,device:FakeModel(p.name),Path("assets"),"cuda","single")
            a,b=pair.predict([1,2])
            self.assertEqual(a,["yolov5x.pt:1","yolov5x.pt:2"])
            self.assertEqual(b,["ball.pt:1","ball.pt:2"])
            self.assertTrue(all(m.device=="cpu" for m in pair.models))
            self.assertTrue(all(m.calls==[("cuda:0",2,640)] for m in pair.models))

    def test_oom_splits_batch_preserving_output_order(self):
        with tempfile.TemporaryDirectory() as temp,self.cuda_mocks():
            log=Path(temp)/"events.jsonl"
            pair=rt.DetectorPair(lambda p,device:FakeModel(p.name,oom_above=2),Path("assets"),"cuda","single",log)
            a,b=pair.predict([0,1,2,3],[10,11,12,13])
            self.assertEqual(a,[f"yolov5x.pt:{i}" for i in range(4)])
            self.assertEqual(len(b),4)
            events=[json.loads(x) for x in log.read_text().splitlines()]
            self.assertTrue(any(e["kind"]=="oom_retry_smaller_batch" for e in events))

    def test_oom_at_one_frame_stops_and_logs(self):
        with tempfile.TemporaryDirectory() as temp,self.cuda_mocks():
            log=Path(temp)/"events.jsonl"
            pair=rt.DetectorPair(lambda p,device:FakeModel(p.name,oom_above=0),Path("assets"),"cuda","single",log)
            with self.assertRaisesRegex(RuntimeError,"batch=1"):
                pair.predict([0],[15000])
            self.assertIn("cuda_oom",log.read_text())

    def test_non_oom_cuda_error_is_not_retried_as_a_success(self):
        with self.cuda_mocks():
            pair=rt.DetectorPair(lambda p,device:FakeModel(p.name,error="no kernel image is available"),Path("assets"),"cuda","single")
            with self.assertRaisesRegex(RuntimeError,"uncommitted"):
                pair.predict([0])

    def test_both_residency_can_downgrade_on_oom(self):
        with self.cuda_mocks():
            pair=rt.DetectorPair(lambda p,device:FakeModel(p.name,oom_above=1),Path("assets"),"cuda","both")
            a,b=pair.predict([0,1])
            self.assertEqual(pair.residency,"single")
            self.assertEqual(len(a),2);self.assertEqual(len(b),2)

    def test_load_model_selects_device_and_restores_torch_loader(self):
        model=FakeModel("weights");attempt=Mock(return_value=model)
        fake_common=SimpleNamespace(AutoShape=lambda x:x)
        fake_experimental=SimpleNamespace(attempt_load=attempt)
        old=full.legacy.torch.load
        with tempfile.TemporaryDirectory() as temp:
            path=Path(temp)/"weights.pt";path.write_bytes(b"mock")
            with patch.dict(sys.modules,{"models.common":fake_common,"models.experimental":fake_experimental}),patch.object(full.legacy,"resolve_device",return_value="cuda:0"):
                result=full.legacy.load_model(path,device="cuda")
            self.assertIs(result,model);self.assertEqual(model.device,"cuda:0")
            self.assertEqual(attempt.call_args.kwargs["device"].type,"cpu")
            self.assertIs(full.legacy.torch.load,old)

    def test_model_load_failure_is_recorded(self):
        def fail(*args,**kwargs):raise RuntimeError("checkpoint load error")
        with tempfile.TemporaryDirectory() as temp,self.cuda_mocks():
            log=Path(temp)/"events.jsonl"
            with self.assertRaisesRegex(RuntimeError,"Model load failed"):
                rt.DetectorPair(fail,Path("assets"),"cuda","single",log)
            self.assertIn("model_load_failed",log.read_text())

    def test_windows_lock_branch_uses_one_byte_and_unlocks(self):
        fake=SimpleNamespace(LK_NBLCK=1,LK_UNLCK=2,locking=Mock())
        with tempfile.TemporaryDirectory() as temp,patch.object(rt,"IS_WINDOWS",True),patch.dict(sys.modules,{"msvcrt":fake}):
            path=Path(temp)/"lock"
            with rt.file_lock(path):self.assertEqual(path.read_bytes(),b"0")
            self.assertEqual([c.args[1:] for c in fake.locking.call_args_list],[(1,1),(2,1)])

    def test_windows_lock_conflict_is_clear(self):
        fake=SimpleNamespace(LK_NBLCK=1,LK_UNLCK=2,locking=Mock(side_effect=OSError("locked")))
        with tempfile.TemporaryDirectory() as temp,patch.object(rt,"IS_WINDOWS",True),patch.dict(sys.modules,{"msvcrt":fake}):
            with self.assertRaisesRegex(RuntimeError,"Another worker"):
                with rt.file_lock(Path(temp)/"lock"):pass

    def test_missing_old_cache_is_optional(self):
        with tempfile.TemporaryDirectory() as temp,patch.object(full.legacy,"ASSETS",Path(temp)):
            db=full.connection(Path(temp))
            self.assertEqual(full.seed_existing_cache(db,0,750),0)
            db.close()

    def test_failed_inference_does_not_commit_empty_success(self):
        with tempfile.TemporaryDirectory() as temp:
            pair=Mock();pair.predict.side_effect=RuntimeError("CUDA OOM at batch=1")
            cap=Mock();cap.read.return_value=(True,np.zeros((8,8,3),np.uint8));cap.get.return_value=1
            with patch.object(full.legacy,"import_upstream"),patch.object(full,"DetectorPair",return_value=pair),patch.object(full.cv2,"VideoCapture",return_value=cap):
                with self.assertRaisesRegex(RuntimeError,"CUDA OOM"):
                    full.detect(Path(temp),dict(start=0,end=1,batch=1,device="cuda:0",reuse_750_cache=False))
            db=full.connection(Path(temp));self.assertEqual(db.execute("SELECT COUNT(*) FROM detections").fetchone()[0],0);db.close()

    def test_windows_requirements_do_not_install_torch(self):
        lines=(ROOT/"requirements-tryolabs-windows.txt").read_text().splitlines()
        names={line.strip().split("==")[0].lower() for line in lines if line.strip() and not line.startswith("#")}
        self.assertFalse(names & {"torch","torchvision","torchaudio","thop"})

    def test_ffmpeg_concat_uses_forward_slashes_and_quotes(self):
        path=Mock();path.resolve.return_value.as_posix.return_value="C:/Users/test name/a'b.mp4"
        with patch.object(analysis,"Path",return_value=path):
            self.assertEqual(analysis.concat_entry("ignored"),"file 'C:/Users/test name/a'\\''b.mp4'\n")

    def test_750_benchmark_requests_actual_inference_and_marks_resume(self):
        def fake_child(command,**kwargs):
            out=Path(command[command.index("--output")+1]);out.mkdir(exist_ok=True)
            start=int(command[command.index("--start")+1]);end=int(command[command.index("--end")+1])
            stage=command[command.index("--stage")+1]
            self.assertNotIn("--reuse-750-cache",command)
            full.atomic_json(out/"manifest.json",dict(start=start,end=end))
            db=full.connection(out)
            table="detections" if stage=="detect" else "results"
            with db:db.executemany(f"INSERT OR IGNORE INTO {table} VALUES (?,?)",[(i,full.pack(dict(status="ok"))) for i in range(start,end)])
            db.close()
        with tempfile.TemporaryDirectory() as temp,patch.object(full,"OUT",Path(temp)),patch.object(full,"resolve_device",return_value="cuda:0"),patch.object(bench.platform,"platform",return_value="mock Windows"),patch.object(bench.subprocess,"run",side_effect=fake_child):
            out=Path(temp)/"benchmark"
            first=bench.benchmark(out);second=bench.benchmark(out)
            self.assertTrue(first["fresh_benchmark"]);self.assertFalse(second["fresh_benchmark"])
            self.assertEqual(first["detected_frames"],750);self.assertEqual(first["processed_frames"],750)


class CpuResumeTests(unittest.TestCase):
    def test_real_cpu_video_checkpoint_resume_matches_continuous_run(self):
        # Synthetic CPU replay only: no GPU/model inference or throughput claim.
        with tempfile.TemporaryDirectory() as temp:
            root=Path(temp);video=root/"synthetic.mp4"
            writer=cv2.VideoWriter(str(video),cv2.VideoWriter_fourcc(*"mp4v"),25,(192,108))
            self.assertTrue(writer.isOpened())
            image=np.random.default_rng(7).integers(0,255,(108,192,3),dtype=np.uint8)
            for _ in range(8):writer.write(image)
            writer.release()
            manifest=dict(start=0,end=8,device="cpu",batch=1)
            resumed=root/"resumed";reference=root/"reference"
            for out in [resumed,reference]:
                out.mkdir();full.atomic_json(out/"manifest.json",manifest)
                db=full.connection(out)
                with db:db.executemany("INSERT INTO detections VALUES (?,?)",[(i,full.pack(dict(frame0=i,status="ok",persons=[],balls=[]))) for i in range(8)])
                db.close()
            with patch.object(full,"VIDEO",video):
                full.process(resumed,manifest,chunk_size=2,stop_chunks=2)
                self.assertEqual(json.loads((resumed/"process_status.json").read_text())["next_frame0"],4)
                full.process(resumed,manifest,chunk_size=2)
                full.process(reference,manifest,chunk_size=2)
            a=full.connection(resumed);b=full.connection(reference)
            self.assertEqual(list(a.execute("SELECT * FROM results ORDER BY frame0")),list(b.execute("SELECT * FROM results ORDER BY frame0")))
            a.close();b.close()
            def objects(out):
                result=[]
                for path in sorted((out/"chunks").glob("*/objects.jsonl.gz")):
                    with gzip.open(path,"rt",encoding="utf-8") as f:result.extend(json.loads(x) for x in f)
                return result
            self.assertEqual(objects(resumed),objects(reference))
            self.assertEqual(analysis.coverage(resumed)["processed_frames"],8)


if __name__=="__main__":unittest.main()
