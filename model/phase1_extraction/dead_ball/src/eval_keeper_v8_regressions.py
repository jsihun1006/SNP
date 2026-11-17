"""진단: v8 골키퍼 규칙이 지금 버전보다 점수를 떨어뜨린 DFL 경기에서 무엇이 바뀌었는지 본다(공개 데이터).

    python src/eval_keeper_v8_regressions.py

DFL 7경기 × (ball, hidden)에서 adopted(지금 파이프라인)와 v8_ab, v8_c, v8_all의 인플레이 구간 F1을 비교하고,
떨어진 경기마다 판정이 달라진 구간을 출력한다: 시작·끝 프레임, 길이, 바뀐 쪽(new_alive = v8이 인플레이로 바꿈),
그 구간의 정답 데드 비율, 그리고 주변 ±10초 정답·adopted·v8의 상태 띠(25프레임 = 1글자, D=데드, .=인플레이).
"""

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
import compare_smoothing as cs  # noqa: E402
import keeper_hold as kh  # noqa: E402
from eval_kickoff_fill import pipeline  # noqa: E402
from eval_keeper_hold_size import keeper_hold_spans  # noqa: E402
from eval_keeper_hold_rule import VARIANTS  # noqa: E402

FPS = CONFIG["fps"]


def strip(arr, a, b):
    return "".join("D" if arr[i:i + FPS].mean() >= 0.5 else "." for i in range(a, b, FPS))


def main():
    import dfl_load
    import train_dfl
    for m in dfl_load.MATCHES:
        tracking, events, meta = dfl_load.load_match(m)
        raw = fx.match_from_dfl(tracking, meta)
        oof = pd.read_parquet(os.path.join(train_dfl.oof_dir(), "{}_sim.parquet".format(m)))
        base_match = fx.Match(players=raw.players, team=raw.team, ball=raw.ball,
                              period=oof["period"].to_numpy().astype(np.int64), pitch=raw.pitch)
        label = oof["dead"].to_numpy().astype(bool)
        score = oof["score"].to_numpy(np.float64)
        graded = np.asarray(base_match.period) > 0
        truth = pd.DataFrame({"period": base_match.period, "dead": label.astype(np.int8)})
        ball = np.asarray(base_match.ball, dtype=np.float64).copy()
        for a, b in keeper_hold_spans(base_match, label):
            ball[a:b] = np.nan
        hidden = fx.Match(players=raw.players, team=raw.team, ball=ball, period=base_match.period, pitch=raw.pitch)
        for cond, match in (("ball", base_match), ("hidden", hidden)):
            sig = fd.ball_signals(match)
            cr = fd.line_crossings(sig)
            cr = cr[(cr > 0) & (match.period[cr] == match.period[cr - 1])]
            plain = pipeline(match, score)
            states = {v: kh.keep_holds_alive(plain, match, sig, cr, VARIANTS[v])[0]
                      for v in ("adopted", "v8_ab", "v8_c", "v8_all")}
            f1 = {v: cs.score_one((s & graded).astype(np.int8), truth)["alive_spell_f1"] for v, s in states.items()}
            for v in ("v8_ab", "v8_c", "v8_all"):
                d = f1[v] - f1["adopted"]
                if d >= -1e-9:
                    continue
                print("\n" + "=" * 90 + "\n{} {} {}: {:.4f} -> {:.4f} ({:+.4f})".format(m, cond, v, f1["adopted"], f1[v], d))
                diff = states[v] != states["adopted"]
                for a, b in fx.runs_of(diff):
                    new_alive = not states[v][a]
                    lo, hi = max(a - 10 * FPS, 0), min(b + 10 * FPS, len(label))
                    print("  frames {}-{} ({:.1f} s) {}, truth dead {:.0%}".format(
                        a, b - 1, (b - a) / FPS, "new_alive" if new_alive else "new_dead", label[a:b].mean()))
                    print("    truth   " + strip(label, lo, hi))
                    print("    adopted " + strip(states["adopted"], lo, hi))
                    print("    {:7s} ".format(v) + strip(states[v], lo, hi))
        print("done", m, flush=True)


if __name__ == '__main__':
    main()
