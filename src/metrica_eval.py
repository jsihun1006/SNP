"""Metrica 샘플 2경기로 (1) 공백과 데드볼의 관계를 정답 기준으로 재고 (2) DFL로만 학습한 모델을 시험한다.

    python src/metrica_eval.py

공개 데이터라 결과를 그대로 출력해도 된다. 대회 데이터는 쓰지 않는다.
"""

import os
import sys

import numpy as np
import pandas as pd

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
from config import CONFIG  # noqa: E402
import features as fx  # noqa: E402
import metrica_load as ml  # noqa: E402
import metrics  # noqa: E402

FPS = CONFIG["fps"]
BUCKETS_S = [0, 1, 3, 5, 10, 20, 30, np.inf]
THRESHOLDS_S = [3, 5, 8, 10, 15, 20, 30]


def gaps_of(seen, period):
    out = []
    for p in np.unique(period):
        rows = np.flatnonzero(period == p)
        out += [(rows[a], rows[b - 1] + 1) for a, b in fx.runs_of(~seen[rows])]
    return out


def gap_report(games):
    lines = ["== ball gaps vs event-based dead balls (both games) =="]
    rows, offsets = [], []
    for g, (tracking, spells, label) in games.items():
        seen = tracking["ball_x"].notna().to_numpy()
        period = tracking["period"].to_numpy()
        gaps = gaps_of(seen, period)
        for a, b in gaps:
            rows.append({"game": g, "seconds": (b - a) / FPS, "frames": b - a, "dead_frames": int(label[a:b].sum())})
        starts = np.array([a for a, _ in gaps])
        stops = np.array([b for _, b in gaps])
        for s in spells.itertuples():
            overlap = np.minimum(stops, s.stop) - np.maximum(starts, s.start)
            k = int(np.argmax(overlap)) if len(overlap) else -1
            if k < 0 or overlap[k] <= 0:
                offsets.append({"game": g, "covered": 0.0, "cause": s.cause})
                continue
            offsets.append({"game": g, "covered": overlap[k] / (s.stop - s.start), "cause": s.cause,
                            "gap_s": (stops[k] - starts[k]) / FPS,
                            "start_off": (starts[k] - s.start) / FPS, "end_off": (stops[k] - s.stop) / FPS})
        lines.append("game {}: ball missing {:.1%}, gaps {}, of missing frames dead {:.1%}, of seen frames dead {:.1%}"
                     .format(g, (~seen).mean(), len(gaps), label[~seen].mean(), label[seen].mean()))
    table = pd.DataFrame(rows)
    lines.append("gap length s p10/p25/p50/p75/p90: " + " / ".join(
        "{:.2f}".format(v) for v in np.percentile(table["seconds"], [10, 25, 50, 75, 90])))
    lines.append("\nby gap length: gaps, share of missing frames, share of those frames truly dead")
    total = table["frames"].sum()
    for low, high in zip(BUCKETS_S[:-1], BUCKETS_S[1:]):
        pick = table[(table.seconds >= low) & (table.seconds < high)]
        if len(pick):
            lines.append("  {:>4}-{:<4} s  gaps {:4d}  frames {:5.1%}  dead {:5.1%}".format(
                low, "inf" if np.isinf(high) else high, len(pick), pick.frames.sum() / total,
                pick.dead_frames.sum() / max(pick.frames.sum(), 1)))
    lines.append("\nrule 'gap >= T s is dead': share of those gap frames truly dead / share of all dead frames covered")
    all_dead = sum(int(label.sum()) for _, _, label in games.values())
    for t in THRESHOLDS_S:
        pick = table[table.seconds >= t]
        lines.append("  T {:>2} s: precision {:5.1%}  covers {:5.1%} of dead frames".format(
            t, pick.dead_frames.sum() / max(pick.frames.sum(), 1), pick.dead_frames.sum() / all_dead))
    off = pd.DataFrame(offsets)
    has = off[off.covered > 0]
    lines.append("\ndead spells overlapping a gap: {}/{}; share of the spell inside its main gap p10/p50/p90 {}".format(
        len(has), len(off), " / ".join("{:.2f}".format(v) for v in np.percentile(has.covered, [10, 50, 90]))))
    for name in ("start_off", "end_off"):
        v = has[name].to_numpy()
        lines.append("  gap {} minus dead {} (s): p10 {:.1f} p25 {:.1f} p50 {:.1f} p75 {:.1f} p90 {:.1f}; "
                     "within 1 s {:.0%}, within 2 s {:.0%}".format(
                         "start" if name == "start_off" else "end", "start" if name == "start_off" else "restart",
                         *np.percentile(v, [10, 25, 50, 75, 90]), (np.abs(v) <= 1).mean(), (np.abs(v) <= 2).mean()))
    lines.append("  by restart type (median start_off / end_off, n):")
    for cause, part in has.groupby("cause"):
        lines.append("    {:12s} {:5.1f} / {:5.1f}  n {}".format(cause, part.start_off.median(), part.end_off.median(),
                                                               len(part)))
    return lines


def model_report(games, feature_set=CONFIG["default_feature_set"]):
    import lightgbm as lgb
    import train_dfl
    import compare_smoothing as cs
    import smoothing
    model = lgb.Booster(model_file=train_dfl.model_path(feature_set))
    columns = model.feature_name()
    choice = CONFIG["smooth_choice"]
    lines = ["\n== DFL-only model on Metrica (player features only, Viterbi penalty {}) ==".format(choice["penalty"])]
    rows, causes = [], []
    for g, (tracking, spells, label) in games.items():
        match = ml.match_from_metrica(tracking)
        frame = fx.compute_features(match)
        score = model.predict(frame[columns].astype(np.float32))
        period = match.period
        state = smoothing.min_duration(smoothing.viterbi(score, period, choice["penalty"], choice["bias"]), period,
                                       int(choice["min_s"] * FPS))
        out = metrics.frame_metrics(label, state, score)
        seg = metrics.segment_metrics(spells, state, period)
        found = seg.pop("found")
        out.update(seg)
        out.update(cs.score_one(state, pd.DataFrame({"period": period, "dead": label})))
        rows.append(dict(game=g, **out))
        causes.append(spells.assign(found=found).groupby("cause")["found"].agg(["sum", "count"]))
        suffix = "" if feature_set == "v1" else "_" + feature_set
        np.save(os.path.join(ml.CACHE, "game{}_score{}.npy".format(g, suffix)), score.astype(np.float32))
    table = pd.DataFrame(rows)
    shown = ["game", "auc", "dead_f1", "dead_precision", "dead_recall", "pred_dead_share", "true_dead_share",
             "spell_recall", "segment_precision", "start_err_med", "start_abs_med", "end_err_med", "end_abs_med",
             "alive_spell_f1", "pred_spells", "true_spells"]
    lines.append(table[shown].round(3).to_string(index=False))
    lines.append("reference, DFL leave-one-match-out mean (v2, model + Viterbi only): dead_f1 0.934, "
                 "spell_recall 0.913, alive_spell_f1 0.845")
    merged = pd.concat(causes).groupby(level=0).sum()
    merged["recall"] = (merged["sum"] / merged["count"]).round(2)
    lines.append("spell recall by restart:\n" + merged.sort_values("count", ascending=False).to_string())
    return lines


def main():
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--features", default=CONFIG["default_feature_set"], choices=sorted(CONFIG["feature_sets"]))
    parser.add_argument("--model-only", action="store_true", help="공백 분석은 건너뛴다")
    args = parser.parse_args()
    pd.set_option("display.width", 220)
    games = {}
    for g in ml.GAMES:
        tracking, events = ml.load_game(g)
        spells = ml.dead_spells(tracking, events)
        games[g] = (tracking, spells, ml.dead_label(tracking, spells))
    lines = ([] if args.model_only else gap_report(games)) + model_report(games, args.features)
    print("\n".join(lines))


if __name__ == '__main__':
    main()
