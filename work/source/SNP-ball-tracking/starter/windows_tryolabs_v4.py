"""Windows handoff launcher for frozen V4: stdlib bootstrap, no torch import.

No detection/tracking, policy edits or automatic whole-match processing. Python
entry point also works on macOS for local tests; CMD is the Windows entry point.
"""
import argparse
from datetime import datetime, timezone
import importlib.metadata
import importlib.util
import json
import math
import os
from pathlib import Path
import struct
import subprocess
import sys

ROOT = Path(__file__).resolve().parents[1]
ASSETS = ROOT / "data/derived/tryolabs_broadcast"
MAXIMUM = 5000


def local_path(value, root=ROOT):
    path = Path(str(value).replace("\\", "/"))
    return path.resolve() if path.is_absolute() else (root / path).resolve()


def validate_python():
    if sys.version_info[:2] != (3, 11) or struct.calcsize("P") != 8:
        raise ValueError("Use the existing Python 3.11 x64 venv; no new venv or PyTorch installation is performed")


def torch_metadata():
    versions = {}
    for name in ("torch", "torchvision", "torchaudio"):
        try: versions[name] = importlib.metadata.version(name)
        except importlib.metadata.PackageNotFoundError: versions[name] = None
    return versions


def install_dependencies(root=ROOT):
    before = torch_metadata()
    subprocess.run([sys.executable, "-m", "pip", "install", "--upgrade-strategy", "only-if-needed",
                    "-r", str(root / "requirements-tryolabs-v4-windows.txt")], cwd=root, check=True)
    if torch_metadata() != before:
        raise RuntimeError("Installed PyTorch package metadata changed during dependency install; stop and inspect the venv")


def dependencies():
    names = {"numpy": "numpy", "cv2": "opencv-python", "matplotlib": "matplotlib", "imageio_ffmpeg": "imageio-ffmpeg"}
    missing = [distribution for module, distribution in names.items() if importlib.util.find_spec(module) is None]
    if missing:
        raise ValueError("Missing packages: " + ", ".join(missing) + ". Run windows\\run_tryolabs_v4.cmd --install --check")
    # Actual imports catch DLL/ABI failures that package-presence checks miss.
    import cv2
    import numpy
    import matplotlib
    import imageio_ffmpeg
    exe = Path(imageio_ffmpeg.get_ffmpeg_exe())
    if not exe.is_file(): raise ValueError(f"FFmpeg executable not found: {exe}")
    subprocess.run([str(exe), "-version"], stdout=subprocess.DEVNULL, stderr=subprocess.PIPE, check=True)
    return dict(numpy=numpy.__version__, opencv=cv2.__version__, matplotlib=matplotlib.__version__, ffmpeg=str(exe))


def prepared_count(path):
    count = 0; keys = set()
    with path.open(encoding="utf-8") as f:
        for line in f:
            if not line.strip(): continue
            count += 1
            if count > MAXIMUM: raise ValueError(f"Prepared input exceeds {MAXIMUM} frames: {path}")
            row = json.loads(line)
            if not all(k in row for k in ("frame0", "sequence", "objects", "balls", "source")):
                raise ValueError(f"Incomplete cached input at line {count}: {path}")
            if type(row["frame0"]) is not int or not 0 <= row["frame0"] < 164487:
                raise ValueError(f"Invalid frame index at line {count}: {path}")
            if not isinstance(row["objects"], list) or not isinstance(row["balls"], list):
                raise ValueError(f"Invalid object/ball list at line {count}: {path}")
            key = row["sequence"], row["frame0"]
            if key in keys: raise ValueError(f"Duplicate sequence/frame at line {count}: {path}")
            keys.add(key)
    if count == 0: raise ValueError(f"Empty prepared input: {path}")
    return count


def intervals(starts, seconds):
    if not math.isfinite(seconds) or seconds <= 0 or any(not math.isfinite(x) or x < 0 for x in starts):
        raise ValueError("Start times must be nonnegative and duration positive")
    ranges = [(round(x*25), round((x+seconds)*25)) for x in starts]
    if any(b <= a for a, b in ranges) or sum(b-a for a, b in ranges) > MAXIMUM:
        raise ValueError(f"Use a nonempty sample of at most {MAXIMUM} frames; full-match resume is not implemented")
    ranges.sort()
    if any(a < previous_b for (_, previous_b), (a, _) in zip(ranges, ranges[1:])):
        raise ValueError("Sample intervals overlap")
    return ranges


def validate_full_cache(path, ranges):
    if not (path / "run.sqlite").is_file() or not (path / "chunks").is_dir():
        raise ValueError(f"Need run.sqlite AND completed chunks under: {path}")
    covered = set()
    for folder in sorted((path / "chunks").glob("[0-9]*")):
        meta_path = folder / "chunk.json"
        if not meta_path.is_file(): continue
        meta = json.loads(meta_path.read_text(encoding="utf-8"))
        if not any(a < meta["end"] and b > meta["start"] for a, b in ranges): continue
        required = [folder / name for name in ("tracking.jsonl.gz", "objects.jsonl.gz")]
        if not all(p.is_file() for p in required):
            raise ValueError(f"Incomplete tracking/classification chunk: {folder}")
        for a, b in ranges:
            covered.update(range(max(a, meta["start"]), min(b, meta["end"])))
    wanted = {i for a, b in ranges for i in range(a, b)}
    missing = sorted(wanted - covered)
    if missing:
        raise ValueError(f"Completed tracking chunks do not cover {len(missing)} requested frames, first={missing[0]}, last={missing[-1]}: {path}. No inference will be started")
    # Core V4 subsequently checks actual gzip rows and the copied detection DB.
    return len(wanted)


def select_source(config, source="auto", cache=None, root=ROOT):
    ranges = intervals(config["starts"], config["seconds"])
    if cache is not None:
        path = local_path(cache, root)
        mode = ("prepared" if path.is_file() else "full") if source == "auto" else source
        n = prepared_count(path) if mode == "prepared" else validate_full_cache(path, ranges)
        return dict(source=mode, cache=str(path), frames=n, skipped=[])
    candidates = []
    if source in ("auto", "prepared"):
        paths = [local_path(x, root) for x in config["prepared_candidates"]]
        paths += sorted((root / "data/derived/tryolabs_broadcast/classification_v3").glob("*/input_frames.jsonl"))
        candidates.extend(("prepared", p) for p in dict.fromkeys(paths) if p.is_file())
    if source in ("auto", "full"):
        paths = [local_path(x, root) for x in config["full_cache_candidates"]]
        paths += sorted((root / "data/derived/tryolabs_broadcast/full_match_v2").glob("*"))
        candidates.extend(("full", p) for p in dict.fromkeys(paths) if p.is_dir() and (p/"run.sqlite").is_file())
    skipped = []
    for mode, path in candidates:
        try: n = prepared_count(path) if mode == "prepared" else validate_full_cache(path, ranges)
        except (ValueError, OSError, KeyError, TypeError) as e:
            skipped.append(dict(path=str(path), reason=str(e))); continue
        return dict(source=mode, cache=str(path), frames=n, skipped=skipped)
    details = "\n".join(f'  {r["path"]}: {r["reason"]}' for r in skipped)
    raise ValueError("No usable cache found. Provide --cache <V3 input_frames.jsonl or V2 run folder>.\n"
                     f"Project: {root}\nV3: data/derived/tryolabs_broadcast/classification_v3/<run>/input_frames.jsonl\n"
                     "V2: data/derived/tryolabs_broadcast/full_match_v2/<run>/run.sqlite + chunks\n" + details)


def validate_video(root=ROOT):
    import cv2
    path = root / "data/broadcast/BAR-ATH_broadcast-001.mp4"
    if not path.is_file(): raise ValueError(f"Original video missing: {path}")
    cap = cv2.VideoCapture(str(path))
    try:
        shape = int(cap.get(3)), int(cap.get(4))
        fps = cap.get(5); frames = int(cap.get(7)); ok, _ = cap.read()
        if not ok or shape != (1920,1080) or abs(fps-25) > .01 or frames != 164487:
            raise ValueError(f"Expected original 1920x1080/25fps/164487 frames; got {shape}/{fps}/{frames}, readable={ok}: {path}")
    finally: cap.release()
    return dict(path=str(path), width=shape[0], height=shape[1], fps=fps, frames=frames)


def build_commands(selection, config, output, *, tests=False, analysis=True, video=True, visuals=True, root=ROOT):
    commands = []
    if tests:
        commands.append([sys.executable, "-m", "pytest", "-q", "tests/test_tryolabs_classification_v3.py",
                         "tests/test_tryolabs_classification_v4.py", "tests/test_tryolabs_v4_windows_handoff.py"])
    command = [sys.executable, str(root / "starter/run_tryolabs_classification_v4.py"), "--source", selection["source"],
               "--cache", selection["cache"], "--output", str(output)]
    if selection["source"] == "full":
        command += ["--starts", *map(str, config["starts"]), "--seconds", str(config["seconds"])]
    if not video: command += ["--no-video"]
    if not visuals: command += ["--no-visuals"]
    commands.append(command)
    if analysis:
        command = [sys.executable, str(root / "starter/analyze_tryolabs_classification_v4.py"), "--run", str(output)]
        if selection["source"] == "prepared" and (Path(selection["cache"]).parent/"objects.jsonl").is_file():
            command += ["--frozen-v3", str(Path(selection["cache"]).parent/"objects.jsonl")]
        commands.append(command)
    return commands


def tee(command, root, logfile):
    environment = dict(os.environ, PYTHONUTF8="1", PYTHONIOENCODING="utf-8", OPENBLAS_NUM_THREADS="1", OMP_NUM_THREADS="2")
    with logfile.open("x", encoding="utf-8") as f:
        proc = subprocess.Popen(command, cwd=root, env=environment, stdout=subprocess.PIPE,
                                stderr=subprocess.STDOUT, text=True, encoding="utf-8", errors="replace")
        try:
            for line in proc.stdout:
                print(line, end="", flush=True); f.write(line); f.flush()
            code = proc.wait()
        except KeyboardInterrupt:
            proc.terminate()
            try: proc.wait(timeout=10)
            except subprocess.TimeoutExpired: proc.kill(); proc.wait()
            raise
    if code: raise RuntimeError(f"Stage exited with code {code}; see {logfile}")


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--config", type=Path, default=ROOT/"windows/tryolabs_v4_windows.json")
    p.add_argument("--source", choices=["auto","prepared","full"])
    p.add_argument("--cache"); p.add_argument("--output")
    p.add_argument("--install", action="store_true"); p.add_argument("--tests", action="store_true")
    p.add_argument("--check", action="store_true"); p.add_argument("--dry-run", action="store_true")
    p.add_argument("--no-analysis", action="store_true"); p.add_argument("--no-video", action="store_true")
    p.add_argument("--no-visuals", action="store_true")
    a = p.parse_args(); status = None; job = None
    try:
        validate_python()
        if a.install: install_dependencies()
        env = dependencies(); video = validate_video()
        if a.tests and importlib.util.find_spec("pytest") is None:
            raise ValueError("pytest is missing. Run windows\\run_tryolabs_v4.cmd --install --check")
        config = json.loads(a.config.read_text(encoding="utf-8-sig"))
        selection = select_source(config, a.source or config["source"], a.cache)
        stamp = datetime.now(timezone.utc).strftime("%Y%m%d_%H%M%S_%f")
        job = ASSETS / "classification_v4/windows_jobs" / stamp
        output = local_path(a.output) if a.output else job / "result"
        if not output.is_relative_to((ASSETS/"classification_v4").resolve()):
            raise ValueError("Output must be a NEW directory under data/derived/tryolabs_broadcast/classification_v4")
        if output.exists(): raise ValueError(f"Output already exists; preserved: {output}")
        commands = build_commands(selection, config, output, tests=a.tests, analysis=not a.no_analysis,
                                  video=not a.no_video, visuals=not a.no_visuals)
        print(json.dumps(dict(python=sys.executable, packages=env, torch_metadata_only=torch_metadata(),
                              video=video, selection=selection, output=str(output), detector_calls=0), indent=2))
        for cmd in commands: print("CMD:", subprocess.list2cmdline(cmd), flush=True)
        if a.check or a.dry_run:
            print("CHECK PASSED. No postprocessing or inference was started."); return 0
        job.mkdir(parents=True, exist_ok=False)
        status = dict(status="RUNNING", source=selection, output=str(output), commands=commands, completed_stages=0,
                      packages=env, torch_metadata_only=torch_metadata(), detector_calls=0)
        print("Source cache writers must be stopped. Preservation hashing may take time.", flush=True)
        for i, command in enumerate(commands):
            (job/"launcher_status.json").write_text(json.dumps(status,indent=2),encoding="utf-8")
            tee(command, ROOT, job/f"stage_{i+1:02d}.log"); status["completed_stages"] += 1
        summary = json.loads((output/"summary.json").read_text(encoding="utf-8"))
        verification = json.loads((output/"verification.json").read_text(encoding="utf-8"))
        if summary["frames"] != selection["frames"] or not verification["passed"]:
            raise RuntimeError("Output count or preservation verification failed")
        if not a.no_video and summary["decoded_video_frames"] != selection["frames"]:
            raise RuntimeError("Comparison video verification failed")
        status.update(status="SUCCESS", summary=summary, verification=verification)
        (job/"launcher_status.json").write_text(json.dumps(status,indent=2),encoding="utf-8")
        print("SUCCESS:", output); print("LAUNCH LOG:", job/"launcher_status.json")
        return 0
    except (Exception, KeyboardInterrupt) as e:
        if status is not None:
            status.update(status="INTERRUPTED" if isinstance(e,KeyboardInterrupt) else "FAILED", error_type=type(e).__name__,error=str(e))
            (job/"launcher_status.json").write_text(json.dumps(status,indent=2),encoding="utf-8")
        print(f"ERROR: {type(e).__name__}: {e}", file=sys.stderr)
        return 130 if isinstance(e,KeyboardInterrupt) else 1


if __name__ == "__main__":
    raise SystemExit(main())
