"""실험: 데드볼 중에 공이 페널티 마크 근처에서 빠르게 출발하면 페널티킥을 찬 순간으로 본다(경기 규칙 1조·14조).

    python src/eval_penalty_spot.py

페널티 마크: 골라인에서 11 m, 경기장 폭의 가운데(1조). 페널티킥은 마크 위의 공을 차서 재개한다(14조).
감지: 지금 파이프라인(⑤ 포함 + 라인 아웃 처리)이 데드로 본 프레임에서, 공이 마크에서 SPOT_TOL_M 안에 보이고
빠르게(ball_moving_mps) 움직이는 첫 프레임. 같은 데드 구간에서는 한 번만.
'긴 데드볼'을 따로 몇 초로 정하지 않는다. 데드 구간 안이라는 조건이 인플레이 중 마크를 지나는 공을 거른다.
DFL은 데드볼 동안 공을 반칙 지점에 고정했다가 차는 순간 마크 근처에서 움직이며 나타난다(페널티 3번 모두 0.1~1.6 m).
SPOT_TOL_M은 좌표 흔들림 허용치다(대회 장면에 맞춘 값이 아니다).
보는 것: 실제 페널티킥을 찾는지(DFL 3, SkillCorner 1, Metrica 1), 다른 곳에서 잘못 걸리는지.
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

FPS = CONFIG["fps"]
SPOT_M = 11.0
SPOT_TOL_M = 2.0
# '긴 데드볼': DFL 7경기 데드볼 713번 길이의 상위 10% 경계(p90). 결과를 보기 전에 정했다(2026-11-14).
LONG_DEAD_S = 49.4


def pipeline_state(score, match):
    sig = fd.ball_signals(match)
    cr = fd.line_crossings(sig)
    cr = cr[(cr > 0) & (match.period[cr] == match.period[cr - 1])]
    state = sb.decode(score, match.players, match.ball, match.pitch, match.period, cr)
    state, _ = fd.force_line_outs(state, cr, sig, score, match.period)
    return state & (np.asarray(match.period) > 0), sig


def penalty_kicks(match, state, sig):
    length, width = match.pitch
    ball = match.ball
    with np.errstate(invalid="ignore"):
        d = np.fmin(np.hypot(ball[:, 0] - SPOT_M, ball[:, 1] - width / 2),
                    np.hypot(ball[:, 0] - (length - SPOT_M), ball[:, 1] - width / 2))
    start = sig["seen"] & (d <= SPOT_TOL_M) & sig["moving"]
    kicks = []
    for a, b in fx.runs_of(state):
        hit = np.flatnonzero(start[a:b + 1])   # 데드 구간 끝 바로 다음 프레임까지(재개 순간)
        if len(hit):
            kicks.append((a + int(hit[0]), (a + int(hit[0]) - a) / FPS))   # (차는 순간, 그때까지 데드 길이 초)
    return kicks


def check(name, match, score, true_kicks):
    state, sig = pipeline_state(score, match)
    found = penalty_kicks(match, state, sig)
    rows = []
    out = []
    for long_only in (False, True):
        kept = [(k, L) for k, L in found if not long_only or L >= LONG_DEAD_S]
        rows = []
        for k, L in kept:
            err = [(k - t) / FPS for t in true_kicks if abs(k - t) <= 5 * FPS]
            rows.append({"data": name, "long_only": long_only, "kick": k, "dead_before_s": L, "true": bool(err),
                         "error_s": err[0] if err else np.nan})
        hit = sum(1 for t in true_kicks if any(abs(k - t) <= 5 * FPS for k, _ in kept))
        out.append((rows, len(true_kicks), hit))
    return out


def sources():
    import lightgbm as lgb
    import dfl_labels
    import dfl_load
    import simulate
    import train_dfl
    import eval_penalty_detect as epd
    import skillcorner_eval as se
    model = lgb.Booster(model_file=train_dfl.model_path(CONFIG["default_feature_set"]))
    for m in dfl_load.MATCHES:
        tracking, events, meta = dfl_load.load_match(m)
        spells = dfl_labels.dead_spells(tracking, events)
        raw = fx.match_from_dfl(tracking, meta)
        sim = simulate.simulate(raw, seed=CONFIG["sim_seed"])
        match = fx.Match(players=sim.players, team=raw.team, ball=raw.ball, period=raw.period, pitch=raw.pitch)
        oof = pd.read_parquet(os.path.join(train_dfl.oof_dir(), "{}_sim.parquet".format(m)))
        yield "dfl:" + m, match, oof["score"].to_numpy(np.float64), [int(s.stop) for s in spells.itertuples()
                                                                     if s.cause == "penalty"]
    feat_dir = os.path.join(se.SC_DIR, "features_all")
    for name, match, pens in epd.skillcorner_sets():
        if name.startswith("skillcorner_detected"):
            continue
        mid = name.split(":")[1]
        frame = pd.read_parquet(os.path.join(feat_dir, mid + ".parquet"))
        score = model.predict(frame[model.feature_name()].astype(np.float32)).astype(np.float64)
        keep = np.asarray(match.period) > 0
        m2 = fx.Match(players=match.players[keep], team=match.team, ball=match.ball[keep],
                      period=np.asarray(match.period)[keep].astype(np.int64), pitch=match.pitch)
        new_index = np.cumsum(keep) - 1
        yield name, m2, score[keep], [int(new_index[min(b, len(keep) - 1)]) for _, b in pens]
    for name, match, pens in epd.metrica_sets():
        keep = np.asarray(match.period) > 0
        m2 = fx.Match(players=match.players[keep], team=match.team, ball=match.ball[keep],
                      period=np.asarray(match.period)[keep].astype(np.int64), pitch=match.pitch)
        frame = fx.compute_features(m2)
        score = model.predict(frame[model.feature_name()].astype(np.float32)).astype(np.float64)
        new_index = np.cumsum(keep) - 1
        yield name, m2, score, [int(new_index[min(max(b, 0), len(keep) - 1)]) for _, b in pens]


def main():
    pd.set_option("display.width", 200)
    rows, summary = [], []
    for name, match, score, kicks in sources():
        for (r, n_true, hit), long_only in zip(check(name, match, score, kicks), (False, True)):
            rows += r
            summary.append({"set": name.split(":")[0], "long_only": long_only, "true": n_true, "found": hit,
                            "detections": len(r), "false": sum(1 for x in r if not x["true"])})
        print("done", name, flush=True)
    s = pd.DataFrame(summary)
    print("\n== by data set (long_only: only inside dead segments of at least {} s) ==".format(LONG_DEAD_S))
    print(s.groupby(["long_only", "set"])[["true", "found", "detections", "false"]].sum().to_string())
    if rows:
        print("\n== detections ==")
        d = pd.DataFrame(rows)
        print(d[d["long_only"]].round(2).to_string(index=False))


if __name__ == '__main__':
    main()
