"""실험: 골(골대 사이 골라인 아웃) 뒤 재개까지 데드로 묶는다(final_dead_balls.hold_goal_dead). 세리머니 중 데드볼이
끊겼다가 이어지는 것을 막으려는 것이다. 지금 파이프라인(느린 빌드업 보정 + 라인 아웃 강제)과 비교한다.

    python src/eval_goal_kickoff.py

데이터: DFL 7경기(모델 OOF 확률, 원본 좌표), Metrica 3경기, SkillCorner 20경기(화면 밖 추정 위치 포함).
SkillCorner는 데드볼 동안에도 공을 기록해서 재개 감지를 가장 정직하게 본다. DFL은 데드 동안 공을 고정하고
Metrica는 지운다.
채택 기준(결과 보기 전에 정함, 2026-11-15): 세 데이터 모두 평균이 떨어지지 않고, 합쳐서 나빠진 경기가 좋아진 경기보다 적을 것.
"""

import glob
import os
import sys

import numpy as np
import pandas as pd

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(HERE)
sys.path[:0] = [HERE, os.path.join(ROOT, "starter")]
from config import CONFIG  # noqa: E402
import features as fx  # noqa: E402
import final_dead_balls as fd  # noqa: E402
import slow_buildup as sb  # noqa: E402
import compare_smoothing as cs  # noqa: E402

FPS = CONFIG["fps"]


def run(name, match, score, label):
    sig = fd.ball_signals(match)
    cr = fd.line_crossings(sig)
    cr = cr[(cr > 0) & (match.period[cr] == match.period[cr - 1])]
    base = sb.decode(score, match.players, match.ball, match.pitch, match.period, cr)
    base, _ = fd.force_line_outs(base, cr, sig, score, match.period)
    new, made, missing = fd.hold_goal_dead(base, cr, sig, match.ball, match.pitch, match.period)
    graded = np.asarray(match.period) > 0
    truth = pd.DataFrame({"period": match.period, "dead": label.astype(np.int8)})
    changed = new & ~base
    row = {"data": name.split(":")[0], "match": name.split(":")[1], "goals": len(made) + missing,
           "no_restart": missing, "changed_s": float(changed.sum() / FPS),
           "changed_truly_dead": float(label[changed].mean()) if changed.any() else np.nan,
           "held_s_median": float(np.median([(b - a) / FPS for a, b in made])) if made else np.nan}
    for v, st in (("base", base), ("goal_hold", new)):
        r = cs.score_one((st & graded).astype(np.int8), truth)
        row[v] = r["alive_spell_f1"]
        row[v + "_spells"] = r["pred_spells"]
    row["true_spells"] = r["true_spells"]
    return row


def sources():
    import lightgbm as lgb
    import dfl_load
    import train_dfl
    import metrica_load as ml
    import skillcorner_eval as se
    from eval_metrica3 import load_game3
    for m in dfl_load.MATCHES:
        tracking, events, meta = dfl_load.load_match(m)
        match = fx.match_from_dfl(tracking, meta)
        oof = pd.read_parquet(os.path.join(train_dfl.oof_dir(), "{}_sim.parquet".format(m)))
        match = fx.Match(players=match.players, team=match.team, ball=match.ball,
                         period=oof["period"].to_numpy().astype(np.int64), pitch=match.pitch)
        yield "dfl:" + m, match, oof["score"].to_numpy(np.float64), oof["dead"].to_numpy()
    model = lgb.Booster(model_file=train_dfl.model_path(CONFIG["default_feature_set"]))
    games = []
    for g in ml.GAMES:
        tracking, events = ml.load_game(g)
        games.append(("metrica:game{}".format(g), ml.match_from_metrica(tracking),
                      ml.dead_label(tracking, ml.dead_spells(tracking, events))))
    m3, l3, _ = load_game3()
    games.append(("metrica:game3", m3, l3))
    for name, match, label in games:
        period = np.asarray(match.period).astype(np.int64)
        keep = period > 0
        match = fx.Match(players=match.players[keep], team=match.team, ball=match.ball[keep], period=period[keep],
                         pitch=match.pitch)
        frame = fx.compute_features(match)
        score = model.predict(frame[model.feature_name()].astype(np.float32)).astype(np.float64)
        yield name, match, score, np.asarray(label)[keep]
    feat_dir = os.path.join(se.SC_DIR, "features_all")
    for path in sorted(p for p in glob.glob(os.path.join(se.SC_DIR, "*")) if os.path.isdir(p)
                       and os.path.exists(os.path.join(p, os.path.basename(p) + "_match.json"))):
        mid, pitch, team, pos, det, ball10, period10, dead10 = se.load(path)
        n = int((len(period10) - 1) * FPS / se.SC_FPS) + 1
        period, dead, ball = se.nearest(period10, n), se.nearest(dead10, n), se.upsample(ball10, n)
        players = se.upsample(pos, n).astype(np.float32)
        cached = os.path.join(feat_dir, mid + ".parquet")
        full = fx.Match(players=players, team=team, ball=ball, period=period, pitch=pitch)
        frame = pd.read_parquet(cached) if os.path.exists(cached) else fx.compute_features(full)
        score = model.predict(frame[model.feature_name()].astype(np.float32)).astype(np.float64)
        keep = period > 0
        yield ("skillcorner:" + mid, fx.Match(players=players[keep], team=team, ball=ball[keep],
                                             period=period[keep].astype(np.int64), pitch=pitch),
               score[keep], dead[keep].astype(np.int8))


def main():
    pd.set_option("display.width", 220)
    rows = []
    for name, match, score, label in sources():
        rows.append(run(name, match, score, np.asarray(label).astype(bool)))
        print("done", name, flush=True)
    t = pd.DataFrame(rows)
    t["diff"] = t["goal_hold"] - t["base"]
    print("\n== per match (alive-spell F1) ==")
    print(t.round(4).to_string(index=False))
    print("\n== mean by data set ==")
    print(t.groupby("data")[["base", "goal_hold", "diff", "goals", "no_restart", "changed_s",
                             "changed_truly_dead"]].mean().round(4).to_string())
    better, worse = int((t["diff"] > 1e-9).sum()), int((t["diff"] < -1e-9).sum())
    ok = (t.groupby("data")["diff"].mean() >= -1e-9).all() and worse < better
    print("\nbetter {} / worse {} -> {}".format(better, worse, "PASS" if ok else "not passed"))


if __name__ == '__main__':
    main()
