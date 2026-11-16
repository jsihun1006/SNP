"""진단: 재개 순간 페널티 박스 안 상대 선수 수로 골키퍼 쥠(인플레이)과 데드볼 재개를 가를 수 있는지 DFL에서 잰다.

    python src/eval_keeper_opponents_in_box.py

근거(경기 규칙): 골킥(16조)과 수비 팀이 자기 박스에서 차는 프리킥(13조)은 공을 차기 전까지 상대가 모두 박스 밖이어야
한다. 골키퍼가 경기 중 공을 쥔 동안에는 이런 제한이 없다.
  쥠: eval_keeper_hold_size.keeper_hold_spans의 정답 구간(DFL 7경기, 207개), 재개 = 구간 끝(공이 골키퍼를 떠남)
  데드볼: 정답 데드볼 중 재개가 goal_kick이거나, free_kick이면서 재개 순간 공이 수비 팀 자기 박스 안인 것. 재개 = 데드볼 끝
상대 = 그 박스를 지키지 않는 팀. 박스 경계에서 EDGE_M(1 m) 안쪽에 있는 선수만 센다(흔들림 대비, 페널티킥 감지와 같은 값).
재개 순간과 그 직전 1초 동안의 최댓값 두 가지를 본다.
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
import keeper_hold as kh  # noqa: E402
from eval_keeper_hold_size import keeper_hold_spans  # noqa: E402

FPS = CONFIG["fps"]
EDGE_M = 1.0


def opponents_in_box(match, frame, side, defending_team):
    length, width = match.pitch
    p = match.players[frame].astype(np.float64)
    opp = np.asarray(match.team) != defending_team
    x, y = p[opp, 0], p[opp, 1]
    depth = x if side == 0 else length - x
    with np.errstate(invalid="ignore"):
        inside = (depth <= kh.BOX_DEPTH_M - EDGE_M) & (depth >= -1) & (np.abs(y - width / 2) <= kh.BOX_HALF_M - EDGE_M)
    return int(np.nansum(inside))


def main():
    import dfl_load
    import dfl_labels
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
        length, width = match.pitch
        side_of = kh.defended_side(match)
        period = np.asarray(match.period)
        ball = np.asarray(match.ball, dtype=np.float64)

        def record(kind, frame, side, team):
            frames = range(max(frame - FPS, 0), frame + 1)
            rows.append({"kind": kind, "match": m, "at_restart": opponents_in_box(match, frame, side, team),
                         "max_last_1s": max(opponents_in_box(match, f, side, team) for f in frames)})

        for a, b in keeper_hold_spans(match, label):
            f = min(b, len(period) - 1)
            bx = ball[a, 0]
            side = 0 if bx < length / 2 else 1
            team = [t for t, s in side_of[int(period[a])].items() if s == side][0]
            record("keeper_hold", f, side, team)
        spells = dfl_labels.dead_spells(tracking, events)
        for s in spells.itertuples():
            f = int(s.stop)
            if f >= len(period) or period[f] <= 0 or s.cause not in ("goal_kick", "free_kick"):
                continue
            bx, by = ball[f]
            side = 0 if bx < length / 2 else 1
            depth = bx if side == 0 else length - bx
            if not (depth <= kh.BOX_DEPTH_M and abs(by - width / 2) <= kh.BOX_HALF_M):
                continue
            team = [t for t, sd in side_of[int(period[f])].items() if sd == side][0]
            record(s.cause + "_own_box", f, side, team)
        print("done", m, flush=True)
    t = pd.DataFrame(rows)
    for col in ("at_restart", "max_last_1s"):
        print("\n== opponents inside the box ({}) ==".format(col))
        g = t.groupby("kind")[col]
        print(pd.DataFrame({"n": g.size(), "share_with_any": g.apply(lambda v: (v > 0).mean()).round(3),
                            "mean": g.mean().round(2), "p50": g.median(), "p90": g.quantile(0.9)}).to_string())


if __name__ == '__main__':
    main()
