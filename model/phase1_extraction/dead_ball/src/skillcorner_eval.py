"""DFL로만 학습한 지금 데드볼 모델을 SkillCorner 공개 데이터(A리그 2024/25, 20경기)로 외부 검증한다.

    python src/skillcorner_eval.py

설정은 지금 파이프라인 그대로다(모델 v2, 비터비 벌점 40, 라인 아웃 처리). 이 결과로 기준값을 고르지 않는다.
SkillCorner는 중계 영상 기반 10 fps라 이렇게 맞춘다.
  - 25 fps로 선형 보간(양쪽 프레임이 다 있을 때만). 정답과 하프는 가장 가까운 원래 프레임 값.
  - 좌표 원점이 경기장 중앙이라 (L/2, W/2)를 더한다. 팀은 match.json의 team_id로 홈/원정.
  - 선수 두 가지: detected(화면에 잡힌 선수만), all(SkillCorner가 화면 밖 선수까지 추정한 위치 포함).
    대회 tactical(프레임당 평균 20.5명)은 그 사이에 있다.
  - 공은 is_detected인 프레임만(추정 위치는 쓰지 않는다).
정답: phases_of_play는 공이 인플레이일 때만 정의된다. 경기 구간(match_periods) 안에서 phase 밖 = 데드.
  (skillcorner_check.py: 데드볼이 공이 라인을 넘기 몇 초 전부터 잡혀 있는 경우가 많다. 추정)
데이터: data/external/skillcorner/ (커밋 안 함). 출처: SkillCorner Open Data (MIT).
"""

import glob
import json
import os
import sys

import numpy as np
import pandas as pd

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = __import__("repo_root").ROOT
sys.path[:0] = [HERE, os.path.join(ROOT, "starter")]
from config import CONFIG  # noqa: E402
import compare_smoothing as cs  # noqa: E402
import features as fx  # noqa: E402
import final_dead_balls as fd  # noqa: E402
import metrics  # noqa: E402
import smoothing  # noqa: E402
import train_dfl  # noqa: E402

FPS = CONFIG["fps"]
SC_FPS = 10
SC_DIR = os.path.join(ROOT, "data", "external", "skillcorner")
OUT = os.path.join(SC_DIR, "eval_results.csv")


def load(path):
    mid = os.path.basename(path)
    meta = json.load(open(os.path.join(path, mid + "_match.json"), encoding="utf-8"))
    length, width = float(meta["pitch_length"]), float(meta["pitch_width"])
    players = meta["players"]
    col = {p["id"]: i for i, p in enumerate(players)}
    team = np.array([0 if p["team_id"] == meta["home_team"]["id"] else 1 for p in players], dtype=np.int8)
    frames, rows = [], []
    with open(os.path.join(path, mid + "_tracking_extrapolated.jsonl"), encoding="utf-8") as handle:
        for line in handle:
            r = json.loads(line)
            frames.append(r)
    n = max(r["frame"] for r in frames) + 1
    pos = np.full((n, len(players), 2), np.nan, dtype=np.float32)
    det = np.zeros((n, len(players)), dtype=bool)
    ball = np.full((n, 2), np.nan)
    for r in frames:
        f = r["frame"]
        for p in r["player_data"]:
            i = col.get(p["player_id"])
            if i is None or p["x"] is None:
                continue
            pos[f, i] = p["x"] + length / 2, p["y"] + width / 2
            det[f, i] = bool(p.get("is_detected"))
        b = r["ball_data"]
        if b["is_detected"] and b["x"] is not None:
            ball[f] = b["x"] + length / 2, b["y"] + width / 2
    period = np.zeros(n, dtype=np.int64)
    for p in meta["match_periods"]:
        period[p["start_frame"]:p["end_frame"] + 1] = p["period"]
    alive = np.zeros(n, dtype=bool)
    phases = pd.read_csv(os.path.join(path, mid + "_phases_of_play.csv"))
    for a, b in zip(phases["frame_start"], phases["frame_end"]):
        alive[int(a):int(b) + 1] = True
    dead = (period > 0) & ~alive
    return mid, (length, width), team, pos, det, ball, period, dead


def upsample(values, n_out):
    """10 fps 배열을 25 fps로 선형 보간. 앞뒤 원래 프레임이 모두 있을 때만 값이 있다."""
    src = np.arange(n_out) * SC_FPS / FPS
    lo = np.floor(src).astype(int)
    hi = np.minimum(lo + 1, len(values) - 1)
    w = (src - lo).reshape((-1,) + (1,) * (values.ndim - 1))
    return values[lo] * (1 - w) + values[hi] * w


def nearest(values, n_out):
    return values[np.minimum(np.round(np.arange(n_out) * SC_FPS / FPS).astype(int), len(values) - 1)]


def spells_of(dead, period):
    rows = []
    for p in np.unique(period[period > 0]):
        idx = np.flatnonzero(period == p)
        for a, b in fx.runs_of(dead[idx]):
            rows.append({"start": idx[a], "stop": idx[b - 1] + 1, "period": p, "seconds": (b - a) / FPS})
    return pd.DataFrame(rows, columns=["start", "stop", "period", "seconds"])


def evaluate(score, period, dead, sig):
    choice = CONFIG["smooth_choice"]
    truth = pd.DataFrame({"period": period, "dead": dead.astype(np.int8)})
    spells = spells_of(dead, period)
    crossings = fd.line_crossings(sig)
    base = smoothing.viterbi(score, period, choice["penalty"], choice["bias"]).astype(bool)
    free = np.zeros(len(score), dtype=bool)
    free[crossings] = True
    entry = smoothing.viterbi(score, period, choice["penalty"], choice["bias"], free_entry=free).astype(bool)
    forced, _ = fd.force_line_outs(entry, crossings, sig, score, period)
    out = {}
    for name, state in (("model_only", base), ("plus_line_outs", forced)):
        state = state & (period > 0)
        r = cs.score_one(state.astype(np.int8), truth)
        seg = metrics.segment_metrics(spells, state.astype(np.int8), period)
        seg.pop("found")
        out[name] = dict(alive_spell_f1=r["alive_spell_f1"], dead_f1=r["dead_f1"], pred_spells=r["pred_spells"],
                         true_spells=r["true_spells"], spell_recall=seg["spell_recall"],
                         segment_precision=seg["segment_precision"], start_abs_med=seg["start_abs_med"],
                         end_abs_med=seg["end_abs_med"])
    out["crossings"] = len(crossings)
    return out


def main():
    import lightgbm as lgb
    pd.set_option("display.width", 220)
    model = lgb.Booster(model_file=train_dfl.model_path(CONFIG["default_feature_set"]))
    columns = model.feature_name()
    rows = []
    for path in sorted(p for p in glob.glob(os.path.join(SC_DIR, "*")) if os.path.isdir(p)):
        mid, pitch, team, pos, det, ball10, period10, dead10 = load(path)
        n = int((len(period10) - 1) * FPS / SC_FPS) + 1
        period = nearest(period10, n)
        dead = nearest(dead10, n)
        ball = upsample(ball10, n)
        sig = fd.ball_signals(fx.Match(players=np.zeros((n, 0, 2), np.float32), team=team[:0], ball=ball,
                                       period=period, pitch=pitch))
        for variant in ("detected", "all"):
            p10 = pos.copy()
            if variant == "detected":
                p10[~det] = np.nan
            players = upsample(p10, n).astype(np.float32)
            match = fx.Match(players=players, team=team, ball=ball, period=period, pitch=pitch)
            frame = fx.compute_features(match)
            score = model.predict(frame[columns].astype(np.float32))
            res = evaluate(score.astype(np.float64), period, dead, sig)
            graded = period > 0
            per_frame = np.isfinite(players).all(axis=2)
            for stage in ("model_only", "plus_line_outs"):
                rows.append(dict(match=mid, players=variant, stage=stage, crossings=res["crossings"],
                                 players_per_frame=float(per_frame[graded].sum(axis=1).mean()), **res[stage]))
        print("done", mid, flush=True)
    t = pd.DataFrame(rows)
    t.to_csv(OUT, index=False)
    keys = ["alive_spell_f1", "dead_f1", "spell_recall", "segment_precision", "start_abs_med", "end_abs_med",
            "pred_spells", "true_spells", "players_per_frame", "crossings"]
    print("\n== mean over {} matches ==".format(t["match"].nunique()))
    print(t.groupby(["players", "stage"])[keys].mean().round(3).to_string())
    print("\n== alive-spell F1 per match ==")
    print(t.pivot_table(index="match", columns=["players", "stage"], values="alive_spell_f1").round(3).to_string())
    for v in ("detected", "all"):
        p = t[t["players"] == v].pivot(index="match", columns="stage", values="alive_spell_f1")
        d = p["plus_line_outs"] - p["model_only"]
        print("{}: line-outs - model only: mean {:+.4f}, better {} / worse {}".format(
            v, d.mean(), int((d > 1e-9).sum()), int((d < -1e-9).sum())))
    print("\nreference: DFL (7 matches, leave-one-out) model only 0.845, plus line-outs 0.878; "
          "Metrica 2 games plus line-outs 0.796 / 0.844")
    print("wrote", OUT)


if __name__ == '__main__':
    main()
