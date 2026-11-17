"""Metrica 샘플 3경기로 두 모델을 비교한다: 지금 모델 대 DFL+SkillCorner(비중 맞춤) 모델.

    python src/eval_metrica3.py

결과를 보기 전에 정한 기준(2026-11-13):
  Metrica 3경기 평균(라인 아웃 처리 포함 인플레이 구간 F1)에서 비중 맞춘 모델이 더 높고,
  3경기 중 2경기 이상에서 이기면 바꾼다. 아니면 지금 모델을 유지한다. DFL은 참고만 한다.
3경기는 형식(EPTS 트래킹 + JSON 이벤트)이 1·2경기와 달라 여기서 읽는다(kloppy).
정답은 metrica_load.dead_spells와 같은 규칙: 재개(SET PIECE, 또는 하위 종류가 GOAL KICK인 이벤트) 직전 이벤트
(CARD·SET PIECE 제외, 같은 하프)가 끝난 프레임부터 재개 프레임까지. 0.5초보다 짧은 것과 하프 첫 킥오프는 뺀다.
Metrica는 학습에 쓰지 않는다. 대회 데이터는 쓰지 않는다.
"""

import json
import os
import sys

import numpy as np
import pandas as pd

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = __import__("repo_root").ROOT
sys.path[:0] = [HERE, os.path.join(ROOT, "starter")]
from config import CONFIG  # noqa: E402
import features as fx  # noqa: E402
import final_dead_balls as fd  # noqa: E402
import metrica_load as ml  # noqa: E402
import train_dfl  # noqa: E402
sys.path.insert(0, os.path.join(HERE, "experiments_dropped"))
from train_with_skillcorner import pipeline_scores  # noqa: E402

FPS = CONFIG["fps"]
RAW3 = os.path.join(ml.RAW, "Sample_Game_3_")
MODELS = {"current": train_dfl.model_path(CONFIG["default_feature_set"]),
          "dfl_sc_balanced": os.path.join(ROOT, "data", "external", "skillcorner", "model_dfl_sc_balanced_v2.txt")}


def subtype_names(event):
    sub = event.get("subtypes")
    if isinstance(sub, dict):
        return [sub.get("name") or ""]
    return [s.get("name") or "" for s in (sub or [])]


def load_game3():
    from kloppy import metrica
    ds = metrica.load_tracking_epts(meta_data=RAW3 + "metadata.xml", raw_data=RAW3 + "tracking.txt")
    df = ds.to_df(engine="pandas")
    length, width = ml.PITCH
    home, away = ds.metadata.teams
    team_of = {p.player_id: 0 for p in home.players}
    team_of.update({p.player_id: 1 for p in away.players})
    ids = [c[:-2] for c in df.columns if c.endswith("_x") and c[:-2] in team_of]
    players = np.stack([np.stack([df[i + "_x"].to_numpy(np.float64) * length,
                                  df[i + "_y"].to_numpy(np.float64) * width], axis=1) for i in ids], axis=1)
    ball = np.stack([df["ball_x"].to_numpy(np.float64) * length, df["ball_y"].to_numpy(np.float64) * width], axis=1)
    period = df["period_id"].to_numpy().astype(np.int64)
    match = fx.Match(players=players.astype(np.float32), team=np.array([team_of[i] for i in ids], dtype=np.int8),
                     ball=ball, period=period, pitch=ml.PITCH, frame=df["frame_id"].to_numpy())
    row_of = {int(f): k for k, f in enumerate(df["frame_id"].to_numpy())}

    events = json.load(open(RAW3 + "events.json", encoding="utf-8"))["data"]
    label = np.zeros(len(df), dtype=np.int8)
    spells = 0
    for i, ev in enumerate(events):
        is_restart = ev["type"]["name"] == "SET PIECE" or any(s.startswith("GOAL KICK") for s in subtype_names(ev))
        if not is_restart:
            continue
        before = [e for e in events[:i] if e["type"]["name"] not in ml.SKIP_BEFORE_RESTART
                  and e["type"]["name"] != "SET PIECE" and e["period"] == ev["period"]
                  and e["start"]["frame"] < ev["start"]["frame"]]
        if not before:
            continue
        a, b = int(before[-1]["end"]["frame"]), int(ev["start"]["frame"])
        if b - a < FPS // 2 or a not in row_of or b not in row_of:
            continue
        label[row_of[a]:row_of[b]] = 1
        spells += 1
    return match, label, spells


def main():
    import lightgbm as lgb
    pd.set_option("display.width", 200)
    games = []
    for g in ml.GAMES:
        tracking, events = ml.load_game(g)
        spells = ml.dead_spells(tracking, events)
        games.append(("game{}".format(g), ml.match_from_metrica(tracking), ml.dead_label(tracking, spells), len(spells)))
    m3, l3, s3 = load_game3()
    games.append(("game3", m3, l3, s3))
    models = {name: lgb.Booster(model_file=path) for name, path in MODELS.items()}
    rows = []
    for name, match, label, n_spells in games:
        frame = fx.compute_features(match)
        sig = fd.ball_signals(match)
        for mname, model in models.items():
            score = model.predict(frame[model.feature_name()].astype(np.float32)).astype(np.float64)
            res = pipeline_scores(score, np.asarray(match.period).astype(np.int64), label, sig)
            rows.append(dict(game=name, spells=n_spells, model=mname, **res))
        print("done", name, flush=True)
    t = pd.DataFrame(rows)
    p = t.pivot_table(index="game", columns="model", values="plus_line_outs")
    print("\n== alive-spell F1 with line-outs (decision metric) ==")
    print(p.round(4).to_string())
    print("\nmodel only:")
    print(t.pivot_table(index="game", columns="model", values="model_only").round(4).to_string())
    diff = p["dfl_sc_balanced"] - p["current"]
    wins = int((diff > 1e-9).sum())
    mean_better = p["dfl_sc_balanced"].mean() > p["current"].mean()
    print("\nmean: current {:.4f}, balanced {:.4f} (diff {:+.4f}); balanced wins {} of {} games".format(
        p["current"].mean(), p["dfl_sc_balanced"].mean(), diff.mean(), wins, len(diff)))
    print("decision (pre-registered): {}".format(
        "switch to the balanced model" if mean_better and wins >= 2 else "keep the current model"))
    print("dead spells per game:", dict(zip(t.drop_duplicates("game")["game"], t.drop_duplicates("game")["spells"])))


if __name__ == '__main__':
    main()
