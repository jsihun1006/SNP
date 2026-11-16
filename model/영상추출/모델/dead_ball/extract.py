"""데드볼 판정 → Phase 1 특징 파일(팀 규칙: model/README.md 3절).

    python model/영상추출/모델/dead_ball/extract.py                     # 파이프라인 실행 + 내보내기
    python model/영상추출/모델/dead_ball/extract.py --skip-pipeline     # 이미 만든 결과만 내보내기
    python model/영상추출/모델/dead_ball/extract.py --run-id full_25fps_20261108

직접 연 터미널에서 실행할 것(대회 트래킹을 읽는다). 파이프라인은 저장소의 src/에 있다:
    src/apply_competition.py  → data/dead_balls/model_v2/   (데드볼 모델 확률)
    src/final_dead_balls.py   → data/dead_balls/final_model/ (최종 데드볼)
이 스크립트는 두 단계를 차례로 실행한 뒤 결과를 아래 폴더에 팀 규칙 형식으로 쓴다.

출력: model/영상추출/추출csv/dead_ball/<실행ID>/
    features.csv        프레임당 1행: frame,time_seconds,graded,dead_score,dead,dead_source
    frame_status.csv    프레임당 1행: frame,time_seconds,processed,ball_seen
    run_manifest.json   입력·모델·설정·버전·해시
    validation.json     validate.py의 형식 검사 결과
"""

import argparse
import datetime
import hashlib
import json
import os
import platform
import subprocess
import sys

import numpy as np
import pandas as pd

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.abspath(os.path.join(HERE, "..", "..", "..", ".."))
sys.path[:0] = [HERE, ROOT, os.path.join(ROOT, "src"), os.path.join(ROOT, "starter")]

MODEL_NAME = "dead_ball"
OUT_ROOT = os.path.join(ROOT, "model", "영상추출", "추출csv", MODEL_NAME)
FEATURE_COLUMNS = ["frame", "time_seconds", "graded", "dead_score", "dead", "dead_source"]
STATUS_COLUMNS = ["frame", "time_seconds", "processed", "ball_seen"]


def sha256(path):
    digest = hashlib.sha256()
    with open(path, "rb") as handle:
        for chunk in iter(lambda: handle.read(1 << 20), b""):
            digest.update(chunk)
    return digest.hexdigest()


def rel(path):
    """저장소 기준 상대 경로, 운영체제와 상관없이 '/'로."""
    return os.path.relpath(path, ROOT).replace(os.sep, "/")


def run_pipeline():
    for script in ("apply_competition.py", "final_dead_balls.py"):
        print("running src/{} ...".format(script), flush=True)
        done = subprocess.run([sys.executable, os.path.join(ROOT, "src", script)], cwd=ROOT)
        if done.returncode != 0:
            raise SystemExit("src/{} failed (exit {})".format(script, done.returncode))


def build_tables(model_frames, final_frames, segments, signals, n_frames, blocks, fps):
    """(features, frame_status) DataFrame. 입력은 src 파이프라인의 출력 표들."""
    frame = np.arange(n_frames)
    graded = np.zeros(n_frames, dtype=np.int8)
    for a, b in blocks:
        graded[a:b] = 1
    score = np.full(n_frames, np.nan)
    score[model_frames["frame"].to_numpy()] = model_frames["score"].to_numpy(np.float64)
    dead = np.zeros(n_frames, dtype=np.int8)
    dead[final_frames["frame"].to_numpy()] = final_frames["dead"].to_numpy()
    dead[graded == 0] = 0
    source = np.full(n_frames, "", dtype=object)
    for row in segments.itertuples():
        source[int(row.start):int(row.end) + 1] = row.kind
    source[dead == 0] = ""
    seen = np.zeros(n_frames, dtype=np.int8)
    seen[signals["frame"].to_numpy()] = signals["seen"].to_numpy()
    time_seconds = np.round(frame / float(fps), 6)
    features = pd.DataFrame({"frame": frame, "time_seconds": time_seconds, "graded": graded,
                             "dead_score": np.round(score, 4), "dead": dead, "dead_source": source})
    status = pd.DataFrame({"frame": frame, "time_seconds": time_seconds, "processed": 1, "ball_seen": seen})
    return features[FEATURE_COLUMNS], status[STATUS_COLUMNS]


def keeper_setting():
    import keeper_hold
    return dict(keeper_hold.SETTING)


def manifest(out_dir, run_id, inputs, fps, n_frames):
    from config import CONFIG
    import train_dfl
    import dfl_load
    try:
        commit = subprocess.run(["git", "rev-parse", "HEAD"], cwd=ROOT, capture_output=True, text=True).stdout.strip()
        dirty = bool(subprocess.run(["git", "status", "--porcelain", "src"], cwd=ROOT, capture_output=True,
                                    text=True).stdout.strip())
    except OSError:
        commit, dirty = "", None
    versions = {"python": platform.python_version()}
    for name in ("numpy", "pandas", "lightgbm", "pyarrow"):
        try:
            versions[name] = __import__(name).__version__
        except ImportError:
            versions[name] = None
    model_path = train_dfl.model_path(CONFIG["default_feature_set"])
    files = {name: {"path": rel(path), "sha256": sha256(path)} for name, path in inputs.items()}
    outputs = {name: {"path": rel(os.path.join(out_dir, name)),
                      "sha256": sha256(os.path.join(out_dir, name))} for name in ("features.csv", "frame_status.csv")}
    return {
        "model": MODEL_NAME, "run_id": run_id, "phase": 1,
        "created_at": datetime.datetime.now().isoformat(timespec="seconds"),
        "code": {"git_commit": commit, "src_has_uncommitted_changes": dirty,
                 "entry_points": ["src/apply_competition.py", "src/final_dead_balls.py",
                                  "model/영상추출/모델/dead_ball/extract.py"]},
        "versions": versions,
        "source": "tactical", "fps": fps, "frame_range": [0, n_frames - 1], "frame_index": "0-based video frame; "
                  "ATD frame_id = frame + 1",
        "inputs": files,
        "model_file": {"path": rel(model_path), "sha256": sha256(model_path),
                       "feature_set": CONFIG["default_feature_set"],
                       "trained_on": "DFL/IDSSE public matches " + ", ".join(dfl_load.MATCHES)
                                     + " (Bassek et al. 2025, CC-BY 4.0); no labels from the competition match"},
        "settings": {"smooth_choice": CONFIG["smooth_choice"], "final": {k: (list(v) if isinstance(v, tuple) else v)
                                                                         for k, v in CONFIG["final"].items()},
                     "context_offsets_s": CONFIG["context_offsets_s"],
                     "slow_buildup": CONFIG["slow_buildup"], "keeper_hold": keeper_setting()},
        "columns": {
            "graded": "1 inside the two graded halves (games.half_video_blocks), else 0",
            "dead_score": "frame-level dead-ball probability of the DFL-trained model, before smoothing (0-1)",
            "dead": "final dead-ball call (1 dead, 0 not dead). Always 0 where graded = 0",
            "dead_source": "why the frame is dead: model | model+out_start | forced_line_out | public_kickoff; "
                           "empty when dead = 0",
            "ball_seen": "1 when the tactical ATD has a ball position in this frame",
        },
        "outputs": outputs,
        "notes": ["Player-movement model; the ball is used for line-outs, the slow build-up corrections "
                  "(src/slow_buildup.py), penalty kicks and goalkeeper holds (src/keeper_hold.py).",
                  "Ball missing is not taken as dead by itself.",
                  "The public 5-minute ground truth was not used for training or tuning."],
    }


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--run-id", default="full_25fps_" + datetime.date.today().strftime("%Y%m%d"))
    parser.add_argument("--skip-pipeline", action="store_true", help="src 파이프라인을 다시 돌리지 않고 내보내기만")
    args = parser.parse_args()
    from config import CONFIG
    from games import GAME, GAMES, half_video_blocks, require
    import validate

    if not args.skip_pipeline:
        run_pipeline()
    fps = CONFIG["fps"]
    n_frames = GAMES[GAME]["n_video_frames"]
    blocks = [(first, last + 1) for _, first, last in half_video_blocks(GAME)]
    from apply_competition import OUT_DIR as model_dir
    from final_dead_balls import OUT_DIR as final_dir
    paths = {"model_frames": os.path.join(model_dir, "frames.csv"),
             "final_frames": os.path.join(final_dir, "frames.csv"),
             "final_segments": os.path.join(final_dir, "segments.csv"),
             "frame_signals": os.path.join(final_dir, "frame_signals.csv")}
    missing = [p for p in paths.values() if not os.path.exists(p)]
    if missing:
        raise SystemExit("missing pipeline outputs (run without --skip-pipeline): " + ", ".join(missing))
    features, status = build_tables(pd.read_csv(paths["model_frames"], usecols=["frame", "score"]),
                                    pd.read_csv(paths["final_frames"]), pd.read_csv(paths["final_segments"]),
                                    pd.read_csv(paths["frame_signals"], usecols=["frame", "seen"]),
                                    n_frames, blocks, fps)
    out_dir = os.path.join(OUT_ROOT, args.run_id)
    os.makedirs(out_dir, exist_ok=True)
    features.to_csv(os.path.join(out_dir, "features.csv"), index=False)
    status.to_csv(os.path.join(out_dir, "frame_status.csv"), index=False)

    meta_path, raw_path = require(["atd_meta", "atd_raw"])
    inputs = {"atd_raw": raw_path, "atd_meta": meta_path, "smart_tagging": require(["smart_tagging"])[0]}
    inputs.update(paths)
    with open(os.path.join(out_dir, "run_manifest.json"), "w", encoding="utf-8") as handle:
        json.dump(manifest(out_dir, args.run_id, inputs, fps, n_frames), handle, ensure_ascii=False, indent=2)
    report = validate.validate(out_dir, n_frames, blocks, fps)
    graded = features["graded"] == 1
    print("wrote {}".format(rel(out_dir)))
    print("frames {}, graded {}, dead share in graded {:.1%}, validation {}".format(
        len(features), int(graded.sum()), features.loc[graded, "dead"].mean(), "OK" if report["ok"] else "FAILED"))
    if not report["ok"]:
        print(json.dumps(report["failed"], ensure_ascii=False, indent=2))
        raise SystemExit(1)


if __name__ == '__main__':
    main()
