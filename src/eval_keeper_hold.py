"""실험: 수비팀이 자기 박스 안에서 경기 중에 공을 받은 뒤(골키퍼가 잡은 경우 등) 다시 차거나 던질 때까지는
데드볼이 아니라고 둔다. DFL 7경기, Metrica 3경기, SkillCorner 20경기에서 효과와 손해를 잰다.

    python src/eval_keeper_hold.py

축구 규칙: 골키퍼가 공을 손에 쥐고 있는 동안은 인플레이다(12조). 같은 박스 안 모습인 골킥(16조)은 공이 먼저
골라인을 넘는다. 그래서 '경기 중에 날아온 공'과 '그 전에 라인 넘음이 없음'으로 가른다.
대회 데이터에도 있는 정보만 쓴다(2D 공, 선수 위치와 팀, 라인 넘음). 골키퍼가 누구인지, 공 높이는 쓰지 않는다.

받음(catch) 프레임 c:
  - 공이 보이고 한쪽 페널티 박스(골라인에서 16.5 m, 가운데서 ±20.16 m, 경기 규칙 1조) 안에 있다.
  - 직전 1초 안에 공이 빠르게(ball_moving_mps) 움직였고, c에서는 그보다 느리다(받아서 멈춤).
  - c에서 공에 가장 가까운 선수가 그 박스를 지키는 팀이다(하프마다 팀 평균 x로 어느 골을 지키는지 정한다).
  - c 앞 out_start_window_s 안에 라인 넘음이 없고, 공이 밖에서 보인 적도 없다(골킥·코너킥 준비가 아님).
놓음: c 뒤 처음으로 공이 경기장 안에서 빠르게 움직이는 프레임, 또는 라인 넘음, 또는 하프 끝.
[c, 놓음) 동안은 데드볼이 아닌 것으로 바꾼다(라인 아웃 처리까지 한 결과 위에).
기준값은 config의 기존 값과 경기 규칙의 박스 크기뿐이다. 결과를 보고 기준값을 고르지 않는다.
"""

import glob
import os
import sys

import numpy as np
import pandas as pd

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(HERE)
sys.path[:0] = [HERE, os.path.join(ROOT, "starter"), os.path.join(HERE, "experiments_dropped")]
from config import CONFIG  # noqa: E402
import compare_smoothing as cs  # noqa: E402
import dfl_load  # noqa: E402
import features as fx  # noqa: E402
import final_dead_balls as fd  # noqa: E402
import simulate  # noqa: E402
import smoothing  # noqa: E402
import train_dfl  # noqa: E402

FPS = CONFIG["fps"]
F = CONFIG["final"]
BOX_DEPTH_M, BOX_HALF_WIDTH_M = 16.5, 20.16


def defended_goal(match):
    """하프마다 {팀: 0(x=0 골) 또는 1(x=length 골)}. 팀 평균 x가 작은 팀이 x=0 골을 지킨다."""
    out = {}
    for p in np.unique(match.period[match.period > 0]):
        rows = match.period == p
        mean_x = [np.nanmean(match.players[rows][:, match.team == k, 0]) for k in (0, 1)]
        low = int(np.argmin(mean_x))
        out[p] = {low: 0, 1 - low: 1}
    return out


def keeper_holds(match, sig, crossings):
    """[(c, release)] 수비팀이 자기 박스 안에서 경기 중 공을 받은 뒤 다시 찰 때까지."""
    length, width = match.pitch
    ball = match.ball
    period = np.asarray(match.period)
    n = len(ball)
    with np.errstate(invalid="ignore"):
        in_box0 = sig["seen"] & (ball[:, 0] <= BOX_DEPTH_M) & (np.abs(ball[:, 1] - width / 2) <= BOX_HALF_WIDTH_M)
        in_box1 = sig["seen"] & (ball[:, 0] >= length - BOX_DEPTH_M) & (np.abs(ball[:, 1] - width / 2) <= BOX_HALF_WIDTH_M)
    in_box = in_box0 | in_box1
    moving = sig["moving"]
    recent = np.concatenate(([0], np.cumsum(moving)))
    moved_before = (recent[np.arange(n)] - recent[np.maximum(np.arange(n) - FPS, 0)]) > 0
    candidates = np.flatnonzero(in_box & ~moving & sig["inside"] & moved_before & (period > 0))
    # 같은 받음이 여러 프레임 이어지면 첫 프레임만
    if len(candidates):
        candidates = candidates[np.concatenate(([True], np.diff(candidates) > 1))]
    goal_of = defended_goal(match)
    cross = np.zeros(n, dtype=bool)
    cross[crossings[crossings < n]] = True
    cross_c = np.concatenate(([0], np.cumsum(cross)))
    out_c = np.concatenate(([0], np.cumsum(sig["outside"])))
    window = int(F["out_start_window_s"] * FPS)
    spans, last_end = [], -1
    for c in candidates:
        if c < last_end:
            continue
        lo = max(c - window, 0)
        if cross_c[c + 1] - cross_c[lo] > 0 or out_c[c + 1] - out_c[lo] > 0:
            continue
        pos = match.players[c].astype(np.float64)
        d = np.hypot(pos[:, 0] - ball[c, 0], pos[:, 1] - ball[c, 1])
        if not np.isfinite(d).any():
            continue
        k = int(np.nanargmin(d))
        box_goal = 0 if in_box0[c] else 1
        if goal_of[period[c]][int(match.team[k])] != box_goal:
            continue
        t = c + 1
        while t < n and period[t] == period[c] and not (sig["inside"][t] and moving[t]) and not cross[t]:
            t += 1
        spans.append((c, t))
        last_end = t
    return spans


def run(score, period, sig, crossings):
    choice = CONFIG["smooth_choice"]
    free = np.zeros(len(score), dtype=bool)
    free[crossings[crossings < len(score)]] = True
    state = smoothing.viterbi(score, period, choice["penalty"], choice["bias"], free_entry=free).astype(bool)
    state, _ = fd.force_line_outs(state, crossings[crossings < len(score)], sig, score, period)
    return state & (np.asarray(period) > 0)


def evaluate(name, score, match, label, sig):
    period = np.asarray(match.period).astype(np.int64)
    label = np.asarray(label).astype(np.int8)
    crossings = fd.line_crossings(sig)
    state = run(score, period, sig, crossings)
    spans = keeper_holds(match, sig, crossings)
    after = state.copy()
    changed, changed_true_alive = 0, 0
    span_frames, span_alive = 0, 0
    for a, b in spans:
        span_frames += b - a
        span_alive += int((label[a:b] == 0).sum())
        flip = after[a:b].copy()
        changed += int(flip.sum())
        changed_true_alive += int((flip & (label[a:b] == 0)).sum())
        after[a:b] = False
    truth = pd.DataFrame({"period": period, "dead": label})
    return dict(data=name, holds=len(spans), hold_s_med=np.median([(b - a) / FPS for a, b in spans]) if spans else 0,
                hold_frames_truly_alive=span_alive / max(span_frames, 1),
                frames_changed=changed, changed_truly_alive=changed_true_alive / max(changed, 1),
                f1_before=cs.score_one(state.astype(np.int8), truth)["alive_spell_f1"],
                f1_after=cs.score_one(after.astype(np.int8), truth)["alive_spell_f1"])


def dfl_rows():
    rows = []
    for i, m in enumerate(dfl_load.MATCHES):
        tracking, _, meta = dfl_load.load_match(m)
        raw = fx.match_from_dfl(tracking, meta)
        match = simulate.simulate(raw, seed=CONFIG["sim_seed"])   # 평가 경기와 같은 선수 누락
        match = fx.Match(players=match.players, team=raw.team, ball=raw.ball, period=raw.period, pitch=raw.pitch)
        oof = pd.read_parquet(os.path.join(train_dfl.oof_dir(), "{}_sim.parquet".format(m)))
        rows.append(evaluate("dfl:" + m, oof["score"].to_numpy(np.float64), match, oof["dead"].to_numpy(),
                             fd.ball_signals(match)))
        print("done", m, flush=True)
    return rows


def metrica_rows(model):
    import metrica_load as ml
    from eval_metrica3 import load_game3
    rows = []
    games = []
    for g in ml.GAMES:
        tracking, events = ml.load_game(g)
        games.append(("metrica:game{}".format(g), ml.match_from_metrica(tracking),
                      ml.dead_label(tracking, ml.dead_spells(tracking, events))))
    m3, l3, _ = load_game3()
    games.append(("metrica:game3", m3, l3))
    for name, match, label in games:
        frame = fx.compute_features(match)
        score = model.predict(frame[model.feature_name()].astype(np.float32)).astype(np.float64)
        rows.append(evaluate(name, score, match, label, fd.ball_signals(match)))
        print("done", name, flush=True)
    return rows


def skillcorner_rows(model):
    import skillcorner_eval as se
    rows = []
    feat_dir = os.path.join(se.SC_DIR, "features_all")
    for path in sorted(p for p in glob.glob(os.path.join(se.SC_DIR, "*")) if os.path.isdir(p)
                       and os.path.exists(os.path.join(p, os.path.basename(p) + "_match.json"))):
        mid, pitch, team, pos, det, ball10, period10, dead10 = se.load(path)
        n = int((len(period10) - 1) * FPS / se.SC_FPS) + 1
        period, dead, ball = se.nearest(period10, n), se.nearest(dead10, n), se.upsample(ball10, n)
        players = se.upsample(pos, n).astype(np.float32)
        match = fx.Match(players=players, team=team, ball=ball, period=period, pitch=pitch)
        cached = os.path.join(feat_dir, mid + ".parquet")
        frame = pd.read_parquet(cached) if os.path.exists(cached) else fx.compute_features(match)
        score = model.predict(frame[model.feature_name()].astype(np.float32)).astype(np.float64)
        rows.append(evaluate("skillcorner:" + mid, score, match, dead, fd.ball_signals(match)))
        print("done", mid, flush=True)
    return rows


def main():
    import lightgbm as lgb
    pd.set_option("display.width", 220)
    model = lgb.Booster(model_file=train_dfl.model_path(CONFIG["default_feature_set"]))
    t = pd.DataFrame(dfl_rows() + metrica_rows(model) + skillcorner_rows(model))
    t["set"] = t["data"].str.split(":").str[0]
    t["delta"] = t["f1_after"] - t["f1_before"]
    print("\n== per data set ==")
    g = t.groupby("set")
    summary = pd.DataFrame({
        "games": g.size(), "holds_per_game": g["holds"].mean(), "hold_s_med": g["hold_s_med"].median(),
        "hold_frames_truly_alive": g["hold_frames_truly_alive"].mean(),
        "frames_changed_per_game": g["frames_changed"].mean(),
        "changed_truly_alive": g["changed_truly_alive"].mean(),
        "f1_before": g["f1_before"].mean(), "f1_after": g["f1_after"].mean(), "delta": g["delta"].mean(),
        "better": g["delta"].apply(lambda d: int((d > 1e-9).sum())),
        "worse": g["delta"].apply(lambda d: int((d < -1e-9).sum()))})
    print(summary.round(4).to_string())
    print("\n== per game ==")
    print(t[["data", "holds", "frames_changed", "changed_truly_alive", "f1_before", "f1_after", "delta"]]
          .round(4).to_string(index=False))


if __name__ == '__main__':
    main()
