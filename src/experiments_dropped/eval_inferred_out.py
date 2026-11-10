"""[채택하지 않은 실험] 결과: 추정 라인 아웃 정밀도 19%, DFL 인플레이 구간 F1 0.872 → 0.719. 채택 안 함.

공이 안 보이는 동안의 '추정 라인 아웃'(final_dead_balls.inferred_line_outs)을 DFL로 검증한다.

    python src/eval_inferred_out.py

DFL 공은 데드볼 직전까지 진짜로 움직인다. 공 누락을 흉내 내(simulate.ball_dropout, 점유 평가와 같은 기본 설정)
대회 데이터처럼 공이 안 보이는 구간을 만든 뒤 비교한다:
  visible_only      보이는 라인 아웃만 무조건 데드(지금 최종 단계와 같음)
  plus_inferred     + 추정 라인 아웃도 무조건 데드
추정 라인 아웃의 정밀도(1초 뒤 실제 데드볼 비율)도 본다. 확률은 경기 하나씩 빼고 학습한 데드볼 모델(oof).
"""

import os
import sys

import numpy as np
import pandas as pd

HERE = os.path.dirname(os.path.abspath(__file__))
HERE = os.path.dirname(HERE)  # src/ — 이 파일은 src/experiments_dropped/ 안에 있다(채택하지 않은 실험, 기록용)
sys.path[:0] = [HERE, os.path.join(os.path.dirname(HERE), "starter")]
from config import CONFIG  # noqa: E402
import compare_smoothing as cs  # noqa: E402
import dfl_labels  # noqa: E402
import dfl_load  # noqa: E402
import features as fx  # noqa: E402
import final_dead_balls as fd  # noqa: E402
import metrics  # noqa: E402
import simulate  # noqa: E402
import smoothing  # noqa: E402
import train_dfl  # noqa: E402

FPS = CONFIG["fps"]
F = CONFIG["final"]
from features import runs_of  # noqa: E402


def inferred_line_outs(ball, period, pitch):
    """공이 안 보이게 된 공백마다, 사라지기 직전 위치·속도로 직선을 이어 공이 다시 보이기 전에 경기장 밖으로
    나가면 그 시점을 '추정 라인 아웃'으로 낸다(사라진 뒤 정보는 쓰지 않는다). 새 기준값 없음.

    속도는 마지막으로 보인 프레임과 speed_half_window×2 프레임 전의 위치 차이로 구한다(둘 다 보여야 한다).
    경계는 경기장 크기 + boundary_margin_m(라인 아웃 판정과 같은 여유)."""
    length, width = pitch
    m = F["boundary_margin_m"]
    k = 2 * CONFIG["speed_half_window"]
    seen = np.isfinite(ball).all(axis=1)
    out = []
    for a, b in runs_of(~seen):
        last = a - 1
        if last - k < 0 or b >= len(ball) or period[last] != period[min(b, len(ball) - 1)]:
            continue
        if not (seen[last] and seen[last - k]) or period[last - k] != period[last]:
            continue
        x, y = ball[last]
        if not (-m <= x <= length + m and -m <= y <= width + m):
            continue  # 이미 밖에 있던 공
        vx, vy = (ball[last] - ball[last - k]) * FPS / k
        times = []
        if vx > 0:
            times.append((length + m - x) / vx)
        if vx < 0:
            times.append((-m - x) / vx)
        if vy > 0:
            times.append((width + m - y) / vy)
        if vy < 0:
            times.append((-m - y) / vy)
        if not times:
            continue
        t_exit = min(times)
        if t_exit * FPS < (b - a):  # 다시 보이기 전에 밖으로 나간다
            out.append(a + int(round(t_exit * FPS)))
    return np.array(out, dtype=int)


def main():
    pd.set_option("display.width", 200)
    choice = CONFIG["smooth_choice"]
    share = float(np.mean(CONFIG["sim_ball_missing_share"]))
    gap = float(np.mean(CONFIG["sim_ball_gap_mean_s"]))
    rows, prec = [], []
    for i, m in enumerate(dfl_load.MATCHES):
        tracking, events, meta = dfl_load.load_match(m)
        match = fx.match_from_dfl(tracking, meta)
        match.ball = simulate.ball_dropout(match.ball, np.random.default_rng(CONFIG["sim_seed"] + 100 + i), share, gap)
        sig = fd.ball_signals(match)
        visible = fd.line_crossings(sig)
        inferred = inferred_line_outs(match.ball, match.period, match.pitch)
        oof = pd.read_parquet(os.path.join(train_dfl.oof_dir(), "{}_sim.parquet".format(m)))
        score, period, label = oof["score"].to_numpy(np.float64), oof["period"].to_numpy(), oof["dead"].to_numpy()
        spells = dfl_labels.dead_spells(tracking, events)
        for c in inferred:
            prec.append({"match": m, "dead_1s_later": label[min(c + FPS, len(label) - 1)] == 1,
                         "already_dead": label[c] == 1})
        base = smoothing.viterbi(score, period, choice["penalty"], choice["bias"]).astype(bool)
        for name, cross in (("visible_only", visible), ("plus_inferred", np.union1d(visible, inferred))):
            free = np.zeros(len(score), dtype=bool)
            free[cross] = True
            state = smoothing.viterbi(score, period, choice["penalty"], choice["bias"], free_entry=free).astype(bool)
            state, made = fd.force_line_outs(state, cross, sig, score, period)
            made = len(made)
            r = cs.score_one(state.astype(np.int8), pd.DataFrame({"period": period, "dead": label}))
            seg = metrics.segment_metrics(spells, state, period)
            seg.pop("found")
            rows.append(dict(match=m, variant=name, alive_spell_f1=r["alive_spell_f1"], dead_f1=r["dead_f1"],
                             spell_recall=seg["spell_recall"], segment_precision=seg["segment_precision"],
                             forced=made, crossings=len(cross)))
        rows.append(dict(match=m, variant="model_only", **{k: v for k, v in cs.score_one(
            base.astype(np.int8), pd.DataFrame({"period": period, "dead": label})).items()
            if k in ("alive_spell_f1", "dead_f1")}))
        print("done", m, "visible", len(visible), "inferred", len(inferred), flush=True)
    p = pd.DataFrame(prec)
    print("\n== inferred line-outs: {} ==".format(len(p)))
    print("already dead at the inferred frame {:.1%}; dead 1 s later {:.1%}".format(
        p["already_dead"].mean(), p["dead_1s_later"].mean()))
    t = pd.DataFrame(rows)
    print("\n== per match alive_spell_f1 ==")
    print(t.pivot(index="match", columns="variant", values="alive_spell_f1").round(3).to_string())
    print("\n== mean ==")
    print(t.groupby("variant")[["alive_spell_f1", "dead_f1", "spell_recall", "segment_precision", "forced",
                                "crossings"]].mean().round(3).T.to_string())


if __name__ == '__main__':
    main()
