"""Handoff safety/selection and real subprocess tests; no native Windows claim."""
import importlib.util
import json
from pathlib import Path
import sys
import tempfile
import unittest
from unittest.mock import patch
import zipfile

ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT/"starter"))
import windows_tryolabs_v4 as launch
import package_tryolabs_v4_windows as package
spec=importlib.util.spec_from_file_location("v4_bundle_installer",ROOT/"windows/install_tryolabs_v4_bundle.py")
installer=importlib.util.module_from_spec(spec);spec.loader.exec_module(installer)


def config():
    return dict(starts=[600,1800,3600],seconds=10,prepared_candidates=[],full_cache_candidates=[])


def frame():
    return dict(frame0=5,sequence="synthetic",objects=[],balls=[],source="test")


class HandoffTests(unittest.TestCase):
    def test_explicit_prepared_path_with_spaces(self):
        with tempfile.TemporaryDirectory(prefix="v4 space ") as temp:
            p=Path(temp)/"input frames.jsonl";p.write_text(json.dumps(frame())+"\n")
            r=launch.select_source(config(),cache=p,root=Path(temp))
            self.assertEqual(r["source"],"prepared");self.assertEqual(r["frames"],1)
    def test_large_prepared_stops_before_full_match_read(self):
        with tempfile.TemporaryDirectory() as temp:
            p=Path(temp)/"input.jsonl"
            p.write_text("\n".join(json.dumps(dict(frame(),frame0=i)) for i in range(5001)))
            with self.assertRaisesRegex(ValueError,"exceeds 5000"):launch.prepared_count(p)
    def test_empty_duplicate_and_malformed_input_rejected(self):
        with tempfile.TemporaryDirectory() as temp:
            p=Path(temp)/"input.jsonl"
            for data in ["",json.dumps(frame())+"\n"+json.dumps(frame()),'{}\n']:
                p.write_text(data)
                with self.assertRaises(ValueError):launch.prepared_count(p)
    def test_missing_cache_reports_without_running_detector(self):
        with tempfile.TemporaryDirectory() as temp:
            with patch.object(launch.subprocess,"Popen") as popen:
                with self.assertRaisesRegex(ValueError,"No usable cache"):launch.select_source(config(),root=Path(temp))
                popen.assert_not_called()
    def test_complete_chunks_required_not_just_sqlite(self):
        with tempfile.TemporaryDirectory() as temp:
            root=Path(temp);(root/"run.sqlite").write_bytes(b"cache");(root/"chunks").mkdir()
            with self.assertRaisesRegex(ValueError,"do not cover"):launch.validate_full_cache(root,[(0,2)])
            folder=root/"chunks/000000000-000000002";folder.mkdir()
            (folder/"chunk.json").write_text(json.dumps(dict(start=0,end=2)))
            for n in ["tracking.jsonl.gz","objects.jsonl.gz"]:(folder/n).write_bytes(b"placeholder; core validates contents")
            self.assertEqual(launch.validate_full_cache(root,[(0,2)]),2)
    def test_auto_skips_incomplete_full_cache(self):
        with tempfile.TemporaryDirectory() as temp:
            root=Path(temp);bad=root/"bad";good=root/"good"
            for p in [bad,good]:(p/"chunks").mkdir(parents=True);(p/"run.sqlite").touch()
            folder=good/"chunks/000000000-000000002";folder.mkdir()
            (folder/"chunk.json").write_text('{"start":0,"end":2}')
            for name in ["tracking.jsonl.gz","objects.jsonl.gz"]:(folder/name).touch()
            c=dict(config(),starts=[0],seconds=.08,full_cache_candidates=["bad","good"])
            r=launch.select_source(c,root=root)
            self.assertEqual(Path(r["cache"]),good.resolve());self.assertEqual(len(r["skipped"]),1)
    def test_invalid_or_overlapping_full_intervals_rejected(self):
        for starts,seconds in [([0],9999),([0],float("nan")),([1,1],10),([0],.001),([-1],10)]:
            with self.assertRaises(ValueError):launch.intervals(starts,seconds)
    def test_commands_are_argv_no_cuda_or_detector(self):
        selected=dict(source="full",cache="C:/cache with spaces",frames=750)
        commands=launch.build_commands(selected,config(),Path("result"),analysis=True,video=True)
        self.assertEqual(len(commands),2)
        self.assertIn("C:/cache with spaces",commands[0])
        self.assertIn("--starts",commands[0]);self.assertNotIn("--device",commands[0])
        self.assertTrue(all("detect" not in c and "torch" not in c for command in commands for c in command))
    def test_pip_install_preserves_torch_metadata(self):
        with patch.object(launch,"torch_metadata",return_value={"torch":"2.9.1+cu126"}),patch.object(launch.subprocess,"run") as run:
            launch.install_dependencies();args=run.call_args.args[0]
            self.assertIn("requirements-tryolabs-v4-windows.txt",args[-1]);self.assertNotIn("torch",args)
        with patch.object(launch,"torch_metadata",side_effect=[{"torch":"before"},{"torch":"after"}]),patch.object(launch.subprocess,"run"):
            with self.assertRaises(RuntimeError):launch.install_dependencies()
    def test_real_subprocess_logs_and_failure_exit(self):
        with tempfile.TemporaryDirectory(prefix="v4 space ") as temp:
            p=Path(temp)
            launch.tee([sys.executable,"-c","print('cached-only')"],p,p/"ok.log")
            self.assertIn("cached-only",(p/"ok.log").read_text())
            with self.assertRaises(RuntimeError):launch.tee([sys.executable,"-c","raise SystemExit(3)"],p,p/"bad.log")
    def test_bundle_contains_only_allowlisted_code_and_installs_idempotently(self):
        with tempfile.TemporaryDirectory() as temp:
            p=Path(temp);output=p/"handoff.zip";package.build(output)
            with zipfile.ZipFile(output) as z:
                self.assertFalse(any(n.startswith("data/") for n in z.namelist()));z.extractall(p/"bundle")
            target=p/"project";r=installer.install(p/"bundle",target)
            self.assertEqual(r["new_files"],len(package.FILES))
            before={str(f):f.read_bytes() for f in target.rglob("*") if f.is_file()}
            self.assertEqual(installer.install(p/"bundle",target)["new_files"],0)
            self.assertEqual(before,{str(f):f.read_bytes() for f in target.rglob("*") if f.is_file()})
    def test_bundle_conflict_or_tamper_leaves_target_untouched(self):
        with tempfile.TemporaryDirectory() as temp:
            p=Path(temp);package.build(p/"bundle.zip")
            with zipfile.ZipFile(p/"bundle.zip") as z:z.extractall(p/"bundle")
            target=p/"target";(target/"starter").mkdir(parents=True)
            changed=target/"starter/tryolabs_uniform_v3.py";changed.write_text("preserve user's version")
            with self.assertRaisesRegex(ValueError,"NOTHING was installed"):installer.install(p/"bundle",target)
            self.assertEqual(list((target/"starter").iterdir()),[changed])
            (p/"bundle/starter/tryolabs_uniform_v4.py").write_text("tampered")
            with self.assertRaisesRegex(ValueError,"integrity"):installer.install(p/"bundle",p/"fresh")
            self.assertFalse((p/"fresh").exists())
    def test_bundle_rejects_traversal_and_noncode(self):
        for name in ["../video.mp4","/tmp/x.py","data/ball.pt","starter/weights.pt","starter\\x.py","C:/x.py"]:
            with self.assertRaises(ValueError):installer.safe_member(name)
    def test_windows_crlf_existing_source_is_preserved(self):
        with tempfile.TemporaryDirectory() as temp:
            p=Path(temp);package.build(p/"code.zip")
            with zipfile.ZipFile(p/"code.zip") as z:z.extractall(p/"bundle")
            dest=p/"target/starter/tryolabs_uniform_v3.py";dest.parent.mkdir(parents=True)
            original=(p/"bundle/starter/tryolabs_uniform_v3.py").read_bytes().replace(b"\r\n",b"\n").replace(b"\n",b"\r\n")
            dest.write_bytes(original)
            r=installer.install(p/"bundle",p/"target")
            self.assertEqual(r["newline_equivalent_files"],1);self.assertEqual(dest.read_bytes(),original)


if __name__=="__main__":unittest.main()
