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
ROOT = __import__("repo_root").ROOT
sys.path[:0] = [HERE, os.path.join(ROOT, "starter")]
from config import CONFIG  # noqa: E402
import features as fx  # noqa: E402
import final_dead_balls as fd  # noqa: E402
import slow_buildup as sb  # noqa: E402

FPS_ = CONFIG["fps"]


# --- 골 뒤 재개까지 데드(실험, 2026-11-15) ---
# 규칙 8조: 골 뒤에는 센터 마크 킥오프로 재개한다. 2D라 크로스바 위로 넘은 공도 골대 사이로 보이므로 골킥(16조)과
# 코너킥(17조) 재개도 끝으로 인정한다. 재개 = 그 자리에 STILL_S초 이상 멈춰 놓였던 공이 빠르게 출발.
# 놓인 공을 요구하는 이유: 세리머니 중 골키퍼가 그물의 공을 차 내거나 선수가 공을 차서 보내는 것은 재개가 아니다.
# 상수는 결과를 보기 전에 정했다. RESTART_TOL_M은 페널티 마크 감지와 같은 좌표 흔들림 허용치.
POST_HALF_M, GOAL_AREA_DEPTH_M, GOAL_AREA_HALF_M, CORNER_ARC_M = 3.66, 5.5, 9.16, 1.0   # 경기 규칙 1조
RESTART_TOL_M, STILL_S = 2.0, 0.5


def goal_crossings(crossings, ball, pitch):
    """라인 아웃 중 골대 사이로 골라인을 넘은 것(넘기 전후 프레임 y 평균). 반환 (프레임, 골 쪽 0: x=0 / 1: x=length)."""
    length, width = pitch
    c = np.asarray(crossings, dtype=np.int64)
    c = c[c > 0]
    x = ball[c, 0]
    y_mid = (ball[c - 1, 1] + ball[c, 1]) / 2
    with np.errstate(invalid="ignore"):
        keep = ((x < 0) | (x > length)) & (np.abs(y_mid - width / 2) <= POST_HALF_M)
    return c[keep], (x[keep] > length).astype(np.int64)


def restart_spots(ball, pitch, side):
    """프레임별로 공이 이 골 뒤의 재개 자리(센터 마크, 그 골의 골 에어리어, 그 골 쪽 코너)에 있는지."""
    length, width = pitch
    bx, by = ball[:, 0].astype(np.float64), ball[:, 1].astype(np.float64)
    depth = bx if side == 0 else length - bx
    t = RESTART_TOL_M
    with np.errstate(invalid="ignore"):
        center = np.hypot(bx - length / 2, by - width / 2) <= t
        goal_area = (depth <= GOAL_AREA_DEPTH_M + t) & (depth >= -t) & (np.abs(by - width / 2) <= GOAL_AREA_HALF_M + t)
        corner = np.hypot(depth, np.abs(by - width / 2) - width / 2) <= CORNER_ARC_M + t
    return [center, goal_area, corner]


def hold_goal_dead(state, crossings, sig, ball, pitch, period):
    """골대 사이 골라인 아웃 c부터 첫 재개(놓인 공이 재개 자리에서 출발) 직전까지 데드. 같은 하프 안에서 재개를
    못 찾으면 바꾸지 않는다. 반환 (새 state, [(c, t)] 바꾼 구간, 재개 못 찾은 수)."""
    out = state.copy()
    n = len(out)
    still = sig["seen"] & (np.nan_to_num(sig["speed"], nan=np.inf) <= CONFIG["final"]["line_out_moving_mps"])
    go = sig["inside"] & sig["moving"]
    k = int(STILL_S * FPS_)
    period = np.asarray(period)
    goals, sides = goal_crossings(crossings, ball, pitch)
    made, missing = [], 0
    for c, side in zip(goals, sides):
        end = c + 1
        while end < n and period[end] == period[c]:
            end += 1
        restart = None
        for spot in restart_spots(ball[c:end], pitch, side):
            placed = np.concatenate(([0], np.cumsum(still[c:end] & spot)))
            idx = np.flatnonzero(go[c:end] & spot)
            idx = idx[idx >= k]
            ok = idx[(placed[idx] - placed[idx - k]) == k]
            if len(ok):
                restart = c + int(ok[0]) if restart is None else min(restart, c + int(ok[0]))
        if restart is None:
            missing += 1
            continue
        out[c:restart] = True
        made.append((int(c), int(restart)))
    return out, made, missing


import compare_smoothing as cs  # noqa: E402

FPS = CONFIG["fps"]


def run(name, match, score, label):
    sig = fd.ball_signals(match)
    cr = fd.line_crossings(sig)
    cr = cr[(cr > 0) & (match.period[cr] == match.period[cr - 1])]
    base = sb.decode(score, match.players, match.ball, match.pitch, match.period, cr, team=match.team)
    base, _ = fd.force_line_outs(base, cr, sig, score, match.period)
    new, made, missing = hold_goal_dead(base, cr, sig, match.ball, match.pitch, match.period)
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
