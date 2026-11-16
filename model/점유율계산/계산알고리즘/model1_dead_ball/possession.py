"""Phase 2 점유 예측: D는 dead_ball(Phase 1) 결과로 정하고, H/A는 model1의 최근접 팀 관측 + 전환 비용 디코더로 정한다.

    python model/점유율계산/계산알고리즘/model1_dead_ball/possession.py                     # 가장 최근 dead_ball 실행 사용
    python model/점유율계산/계산알고리즘/model1_dead_ball/possession.py --dead-run full_25fps_20261116

직접 연 터미널에서 실행할 것(대회 ATD를 읽는다). 공개·비공개 GT는 읽지 않는다.

입력:
  Phase 1  model/영상추출/추출csv/dead_ball/<실행ID>/features.csv   (frame, graded, dead)
  ATD      data/tactical/ (model1과 같은 최근접 팀 관측, starter/baseline.nearest_player_labels)
판단:
  model1과 같은 관측 비용·전환 비용(config.json의 model1 값 그대로)을 쓰되, 채점 구간에서
    dead = 1 인 프레임은 D만 허용, dead = 0 인 프레임은 H/A만 허용한다(D 비용 무한).
  채점 구간 밖은 model1과 같이 제약 없이 푼다(채점에 안 들어간다).
  그래서 데드볼 경계는 dead_ball이 정하고, 같은 인플레이 구간 안의 H↔A 전환만 model1 비용이 정한다.
출력: model/모델별결과csv/model1_dead_ball/<실행ID>/   (팀 규칙 model/README.md 4절)
  prediction.csv          frame,label (199,319행, H/A/D)
  frame_diagnostics.csv   frame,predicted_label,input_source,observation,dead,dead_source
  possession_share.csv    scope,team,frames,seconds,share
  run_manifest.json       입력 Phase 1 모델·실행ID·경로·해시, 설정, 코드 해시
"""
import argparse
from datetime import datetime, timezone
import importlib.util
import json
from pathlib import Path
import sys

import numpy as np
import pandas as pd

HERE = Path(__file__).resolve().parent
MODEL1 = HERE.parent / "model1"
sys.path.insert(0, str(MODEL1))
from common import FPS, N_FRAMES, ROOT, sha256, write_json, official_baseline as baseline  # noqa: E402

NAME = "model1_dead_ball"
H, A, D = 0, 1, 2
STATES = np.array(["H", "A", "D"])
BIG = 1e9
DEAD_ROOT = ROOT / "model" / "영상추출" / "추출csv" / "dead_ball"
OUT_ROOT = ROOT / "model" / "모델별결과csv" / NAME


def load_model1():
    spec = importlib.util.spec_from_file_location("_model1_model", MODEL1 / "model.py")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def constrained_costs(base_costs, graded, dead):
    """model1의 관측 비용에 dead_ball 제약을 더한다: 채점 구간에서 dead면 D만, 아니면 H/A만."""
    costs = np.asarray(base_costs, dtype=np.float64).copy()
    g = np.asarray(graded, dtype=bool)
    d = np.asarray(dead, dtype=bool)
    costs[g & d, H] = BIG
    costs[g & d, A] = BIG
    costs[g & ~d, D] = BIG
    return costs


def viterbi(costs, team_switch, dead_switch):
    """model1.decode와 같은 3상태 동적 계획법(비용을 밖에서 받는다)."""
    transitions = np.array([[0, team_switch, dead_switch], [team_switch, 0, dead_switch],
                            [dead_switch, dead_switch, 0]], dtype=float)
    back = np.zeros((len(costs), 3), dtype=np.int8)
    score = costs[0].copy()
    for frame in range(1, len(costs)):
        candidates = score[:, None] + transitions
        parent = candidates.argmin(axis=0)
        back[frame] = parent
        score = costs[frame] + candidates[parent, np.arange(3)]
    labels = np.empty(len(costs), dtype=np.int8)
    labels[-1] = score.argmin()
    for frame in range(len(costs) - 1, 0, -1):
        labels[frame - 1] = back[frame, labels[frame]]
    return labels


def latest_dead_run():
    runs = [p for p in DEAD_ROOT.iterdir() if p.is_dir() and (p / "features.csv").is_file()] if DEAD_ROOT.is_dir() else []
    if not runs:
        raise SystemExit("no dead_ball run found; run model/영상추출/모델/dead_ball/extract.py first")
    return max(runs, key=lambda p: (p / "features.csv").stat().st_mtime)


def share_table(labels, graded):
    rows = []
    for scope, mask in (("graded", graded), ("full_video", np.ones(len(labels), dtype=bool))):
        alive = int(np.sum(mask & (labels != D)))
        for team, code in (("H", H), ("A", A)):
            frames = int(np.sum(mask & (labels == code)))
            rows.append({"scope": scope, "team": team, "frames": frames, "seconds": round(frames / FPS, 2),
                         "share": round(frames / alive, 6) if alive else ""})
    return pd.DataFrame(rows)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dead-run", help="dead_ball 실행ID(기본: 가장 최근)")
    parser.add_argument("--run-id", help="이 점유 예측의 실행ID(기본: dead_ball 실행ID와 같게)")
    parser.add_argument("--config", type=Path, default=HERE / "config.json")
    args = parser.parse_args()
    config = json.loads(args.config.read_text(encoding="utf-8"))
    model1 = load_model1()
    model1.validate_config(config)
    dead_dir = DEAD_ROOT / args.dead_run if args.dead_run else latest_dead_run()
    features_path = dead_dir / "features.csv"
    features = pd.read_csv(features_path, usecols=["frame", "graded", "dead", "dead_source"])
    if len(features) != N_FRAMES or not (features["frame"].to_numpy() == np.arange(N_FRAMES)).all():
        raise SystemExit(f"{features_path}: expected frames 0..{N_FRAMES - 1}")
    graded = features["graded"].to_numpy() == 1
    dead = features["dead"].to_numpy() == 1
    run_id = args.run_id or dead_dir.name
    obs_args = argparse.Namespace(input_mode="atd", data_dir=None, atd_raw=None, atd_meta=None,
                                  raw_predictions=ROOT / "evaluation/results/baseline_raw.csv")
    observations, mode, inputs = model1.read_observations(obs_args)
    base_costs, _, _ = model1.emission_costs(observations, config)
    labels = viterbi(constrained_costs(base_costs, graded, dead), config["team_switch_cost"], config["dead_switch_cost"])
    out = OUT_ROOT / run_id
    out.mkdir(parents=True, exist_ok=True)
    prediction = out / "prediction.csv"
    baseline.write_submission(str(prediction), STATES[labels])
    obs_name = np.where(observations == H, "H", np.where(observations == A, "A", "MISSING"))
    pd.DataFrame({"frame": np.arange(N_FRAMES), "predicted_label": STATES[labels],
                  "input_source": np.where(graded & dead, "dead_ball", np.where(graded, "model1_team", "model1_ungraded")),
                  "observation": obs_name, "dead": dead.astype(int),
                  "dead_source": features["dead_source"].fillna("").to_numpy()}).to_csv(out / "frame_diagnostics.csv",
                                                                                     index=False)
    share_table(labels, graded).to_csv(out / "possession_share.csv", index=False)
    dead_manifest = dead_dir / "run_manifest.json"
    write_json(out / "run_manifest.json", {
        "model": NAME, "run_id": run_id, "phase": 2, "created_at_utc": datetime.now(timezone.utc).isoformat(),
        "inference": "model1 costs with dead_ball constraint; no GT access, fitting, or parameter search",
        "phase1_input": {"model": "dead_ball", "run_id": dead_dir.name,
                         "features": {"path": str(features_path.relative_to(ROOT)).replace("\\", "/"),
                                      "sha256": sha256(features_path)},
                         "run_manifest_sha256": sha256(dead_manifest) if dead_manifest.is_file() else None},
        "atd_input_mode": mode, "atd_inputs": [{"path": str(p), "sha256": sha256(p)} for p in inputs],
        "config": config, "config_sha256": sha256(args.config),
        "code_sha256": {"possession.py": sha256(Path(__file__)), "model1/model.py": sha256(MODEL1 / "model.py")},
        "prediction_sha256": sha256(prediction),
        "label_counts": {label: int(np.sum(labels == i)) for i, label in enumerate(STATES)},
        "graded_dead_share": float(dead[graded].mean()) if graded.any() else None,
        "limitations": ["H/A comes from the nearest player to the ball only (model1); ball missing inside an in-play "
                        "stretch keeps the surrounding team through the switch cost.",
                        "Dead-ball boundaries are taken from dead_ball as they are."]})
    print(f"{NAME}: {N_FRAMES} frames; dead_ball run {dead_dir.name}; wrote {out.relative_to(ROOT)}")
    print("label counts:", {label: int(np.sum(labels == i)) for i, label in enumerate(STATES)})
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
