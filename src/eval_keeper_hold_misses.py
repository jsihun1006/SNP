"""진단: DFL 골키퍼 쥠 정답 구간마다 지금 강제 규칙(keeper_hold.SETTING)이 잡는지, 못 잡으면 왜인지 센다.

    python src/eval_keeper_hold_misses.py

쥠 정답: eval_keeper_hold_size.keeper_hold_spans(DFL 7경기, 인플레이이고 공이 골키퍼 1.5 m 안, 자기 박스, 1초 이상).
조건: ball(원본, 공이 보임), hidden(쥠 구간만 공 삭제, 대회처럼 잡으면 공이 안 보이는 상황).
구간 분류:
  ok_already      규칙 전 파이프라인이 이미 대부분 인플레이(데드 프레임 < 50%)
  fixed           규칙 전엔 대부분 데드, 규칙 뒤 대부분 인플레이
  missed          규칙 뒤에도 대부분 데드 -> reason으로 원인
missed의 reason:
  hidden: 구간 안(시작 1초 전 ~ 끝)에서 시작한 '공 사라짐' 후보가 없으면 no_vanish, 있으면 그 후보의 첫 탈락 조건
  ball:   구간에 겹치는 몰고 나가기 후보의 첫 탈락 조건(없으면 no_carry_candidate)
함께 세는 신호(다음 개선의 잠재 효과):
  fast_arrival    쥠 시작 2초 전 안에 공이 빠르게(ball_moving_mps) 움직였음(슛·크로스·백패스를 받음)
  model_dead_1s   쥠 시작 1초 전에 이미 데드로 판정돼 있었음
"""

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
import keeper_hold as kh  # noqa: E402
from eval_kickoff_fill import pipeline  # noqa: E402
from eval_keeper_hold_size import keeper_hold_spans  # noqa: E402

FPS = CONFIG["fps"]


def first_fail(c):
    for key in ("in_play", "keeper_near", "no_line_out"):
        if not c[key]:
            return key
    return "zone:" + c["zone"]


VERSIONS = {"current": dict(kh.SETTING),
            "v8": dict(kh.SETTING, arrival_model=True, fill_before=True, merge_short_dead=True)}


def classify(cond, m, match, score, spans, ball_sig_for_arrival, setting=None, version="current"):
    sig = fd.ball_signals(match)
    cr = fd.line_crossings(sig)
    cr = cr[(cr > 0) & (match.period[cr] == match.period[cr - 1])]
    plain = pipeline(match, score)
    ruled, _ = kh.keep_holds_alive(plain, match, sig, cr, setting)
    cands = kh.candidates(match, sig, cr, plain)
    carries = kh.carry_candidates(match, sig, cr, plain)
    rows = []
    for a, b in spans:
        before, after = plain[a:b].mean(), ruled[a:b].mean()
        kind = "ok_already" if before < 0.5 else ("fixed" if after < 0.5 else "missed")
        reason = "-"
        if kind == "missed":
            if cond == "hidden":
                cs_ = [c for c in cands if a - FPS <= c["v"] <= b]
                reason = first_fail(cs_[0]) if cs_ else "no_vanish"
                if cs_ and cs_[0]["accepted"]:
                    reason = "accepted_but_still_dead"
            else:
                cs_ = [c for c in carries if c["a"] < b and a < c["b"]]
                reason = cs_[0]["first_fail"] if cs_ else "no_carry_candidate"
        lo = max(a - 2 * FPS, 0)
        rows.append({"version": version, "cond": cond, "match": m, "seconds": (b - a) / FPS, "kind": kind, "reason": reason,
                     "fast_arrival": bool(ball_sig_for_arrival["moving"][lo:a].any()),
                     "model_dead_1s": bool(plain[max(a - FPS, 0)])})
    return rows


def main():
    import dfl_load
    import train_dfl
    pd.set_option("display.width", 200)
    rows = []
    for m in dfl_load.MATCHES:
        tracking, events, meta = dfl_load.load_match(m)
        raw = fx.match_from_dfl(tracking, meta)
        oof = pd.read_parquet(os.path.join(train_dfl.oof_dir(), "{}_sim.parquet".format(m)))
        match = fx.Match(players=raw.players, team=raw.team, ball=raw.ball,
                         period=oof["period"].to_numpy().astype(np.int64), pitch=raw.pitch)
        label = oof["dead"].to_numpy().astype(bool)
        score = oof["score"].to_numpy(np.float64)
        spans = keeper_hold_spans(match, label)
        full_sig = fd.ball_signals(match)
        ball = np.asarray(match.ball, dtype=np.float64).copy()
        for a, b in spans:
            ball[a:b] = np.nan
        hidden = fx.Match(players=match.players, team=match.team, ball=ball, period=match.period, pitch=match.pitch)
        for version, st in VERSIONS.items():
            rows += classify("ball", m, match, score, spans, full_sig, st, version)
            rows += classify("hidden", m, hidden, score, spans, full_sig, st, version)
        print("done", m, flush=True)
    t = pd.DataFrame(rows)
    print("\n== holds by outcome (count, seconds) ==")
    print(t.groupby(["version", "cond", "kind"]).agg(n=("seconds", "size"), seconds=("seconds", "sum")).round(1).to_string())
    t = t[t["version"] == "v8"]
    print("\n(below: v8 only)")
    print("\n== missed holds: reason ==")
    miss = t[t["kind"] == "missed"]
    print(miss.groupby(["cond", "reason"]).agg(n=("seconds", "size"), seconds=("seconds", "sum")).round(1).to_string())
    print("\n== missed holds: signals for the next fix ==")
    print(miss.groupby("cond")[["fast_arrival", "model_dead_1s"]].mean().round(2).to_string())
    print("\n== all holds: fast_arrival share by outcome ==")
    print(t.groupby(["cond", "kind"])["fast_arrival"].mean().round(2).to_string())


if __name__ == '__main__':
    main()
