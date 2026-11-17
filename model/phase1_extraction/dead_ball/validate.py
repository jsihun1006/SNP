"""dead_ball 추출 결과의 형식 검사 → validation.json.

    python model/phase1_extraction/dead_ball/validate.py outputs/phase1/dead_ball/<run_id>

검사: 열 이름·순서, 프레임 0..N-1 연속·중복 없음, time_seconds = frame / fps, graded가 채점 구간과 일치,
dead는 0/1이고 채점 밖에서는 0, dead_score는 0~1 또는 빈칸, dead_source는 dead = 1일 때만 정해진 값,
frame_status.csv의 프레임이 features.csv와 같음. 실패가 있으면 종료 코드 1.
"""

import json
import os
import sys

import numpy as np
import pandas as pd

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.abspath(os.path.join(HERE, "..", "..", ".."))
sys.path[:0] = [ROOT]

FEATURE_COLUMNS = ["frame", "time_seconds", "graded", "dead_score", "dead", "dead_source"]
STATUS_COLUMNS = ["frame", "time_seconds", "processed", "ball_seen"]
SOURCES = {"model", "model+out_start", "forced_line_out", "public_kickoff"}


def validate(out_dir, n_frames, blocks, fps):
    checks = {}
    features = pd.read_csv(os.path.join(out_dir, "features.csv"), keep_default_na=False, na_values=[""])
    status = pd.read_csv(os.path.join(out_dir, "frame_status.csv"))
    frame = features["frame"].to_numpy() if "frame" in features else np.array([])
    checks["features_columns"] = list(features.columns) == FEATURE_COLUMNS
    checks["status_columns"] = list(status.columns) == STATUS_COLUMNS
    checks["row_count"] = len(features) == n_frames
    checks["frames_contiguous_0_based"] = bool(len(frame) == n_frames and (frame == np.arange(n_frames)).all())
    checks["time_seconds"] = bool(np.allclose(features["time_seconds"], frame / float(fps), atol=1e-5))
    expected = np.zeros(len(features), dtype=int)
    for a, b in blocks:
        expected[a:b] = 1
    checks["graded_matches_halves"] = bool((features["graded"].to_numpy() == expected).all())
    dead = features["dead"].to_numpy()
    checks["dead_is_0_1"] = bool(np.isin(dead, (0, 1)).all())
    checks["dead_zero_outside_graded"] = bool((dead[expected == 0] == 0).all())
    score = features["dead_score"].to_numpy(np.float64)
    checks["dead_score_in_0_1_or_blank"] = bool(np.all(np.isnan(score) | ((score >= 0) & (score <= 1))))
    source = features["dead_source"].fillna("").astype(str).to_numpy()
    checks["dead_source_only_when_dead"] = bool(((source != "") == (dead == 1)).all())
    checks["dead_source_values"] = bool(set(source[source != ""]) <= SOURCES)
    checks["status_frames_match"] = bool(len(status) == len(features)
                                         and (status["frame"].to_numpy() == frame).all())
    checks["ball_seen_is_0_1"] = bool(status["ball_seen"].isin((0, 1)).all())
    graded = expected == 1
    report = {"ok": all(checks.values()), "checks": checks, "failed": [k for k, v in checks.items() if not v],
              "n_frames": int(len(features)), "n_graded": int(graded.sum()),
              "dead_share_graded": float(dead[graded].mean()) if graded.any() else None,
              "dead_score_blank": int(np.isnan(score).sum()),
              "dead_source_counts_frames": {k: int((source == k).sum()) for k in sorted(SOURCES)}}
    with open(os.path.join(out_dir, "validation.json"), "w", encoding="utf-8") as handle:
        json.dump(report, handle, ensure_ascii=False, indent=2)
    return report


if __name__ == '__main__':
    from games import GAME, GAMES, half_video_blocks
    target = sys.argv[1]
    result = validate(target, GAMES[GAME]["n_video_frames"],
                      [(first, last + 1) for _, first, last in half_video_blocks(GAME)], GAMES[GAME]["fps"])
    print(json.dumps({k: result[k] for k in ("ok", "failed", "n_frames", "dead_share_graded")}, ensure_ascii=False))
    sys.exit(0 if result["ok"] else 1)
