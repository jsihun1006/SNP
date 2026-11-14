"""주어진 프레임 근처의 라인 넘음 판정과 공 상태를 보여 준다(대회 데이터, 직접 확인용).

    python src/diagnose_air_balls.py 44826 123445 68267 80143
    python src/diagnose_air_balls.py --window-s 15 44826

프레임마다 앞뒤 window-s초 안의 라인 넘음 후보(공이 안에서 밖으로 넘은 프레임)를 모두 보여 준다:
  cross        넘은 프레임(밖에서 처음 보인 프레임)
  accepted     지금 규칙(final_dead_balls.line_crossings)이 라인 아웃으로 인정했는지
  why_not      인정 안 했으면 이유: not_rolling(넘기 전 1초 안에 굴러가지 않음) / fast_back(2초 안에 안에서 빠른 공) /
               back_soon(1.6초 안에 안에서 공이 보임)
  line         넘은 라인: touch_low(y=0 쪽) / touch_high(y=W 쪽) / goal_low(x=0) / goal_high(x=L)
  out_m        밖으로 나간 거리(m, 넘은 프레임)
  speed_in     넘기 직전 1초 공 속도 중앙값
  outside_s    넘은 뒤 공이 밖에서(또는 안 보이게) 머문 시간
  unseen_s     그동안 공이 안 보인 시간
  back_speed   안에서 다시 보인 첫 1초의 공 속도 중앙값
  seg          그 프레임을 덮는 최종 데드볼 구간(start-end, kind), 없으면 -
같은 축(라인)에서 카메라 반대쪽이면 뜬 공이 밖으로 찍히기 쉽다(카메라 위치는 이 스크립트가 모른다).
이 표는 원인 확인용이다. 본 장면에 맞춰 기준값을 고르지 않는다.
"""

import argparse
import os
import sys

import numpy as np
import pandas as pd

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(HERE)
sys.path[:0] = [HERE, ROOT, os.path.join(ROOT, "starter")]
from config import CONFIG  # noqa: E402
import final_dead_balls as fd  # noqa: E402

FPS = CONFIG["fps"]
F = CONFIG["final"]


def main():
    import competition_data
    ap = argparse.ArgumentParser()
    ap.add_argument("frames", type=int, nargs="+")
    ap.add_argument("--window-s", type=float, default=10.0)
    args = ap.parse_args()
    pd.set_option("display.width", 250)
    match, _ = competition_data.load_competition_match()
    length, width = match.pitch
    sig = fd.ball_signals(match)
    ball = match.ball
    n = len(ball)
    accepted = set(int(c) for c in fd.line_crossings(sig))
    segments = pd.read_csv(os.path.join(fd.DEAD_DIR, "final_model", "segments.csv"))
    t = np.flatnonzero(sig["inside"][:-1] & sig["outside"][1:]) + 1      # 모든 넘음 후보
    rolled = np.concatenate(([0], np.cumsum(sig["rolling"])))
    fast_in = np.concatenate(([0], np.cumsum(sig["inside"] & sig["moving"])))
    any_in = np.concatenate(([0], np.cumsum(sig["inside"])))
    confirm, restart = int(F["out_confirm_s"] * FPS), int(F["out_min_restart_s"] * FPS)
    w = int(args.window_s * FPS)
    rows = []
    for f in args.frames:
        near = t[(t >= f - w) & (t <= f + w)]
        seg = segments[(segments["start"] <= f) & (segments["end"] >= f)]
        seg_txt = "-" if not len(seg) else "{}-{} {}".format(int(seg["start"].iloc[0]), int(seg["end"].iloc[0]),
                                                             seg["kind"].iloc[0])
        if not len(near):
            rows.append({"frame": f, "cross": "-", "accepted": "-", "why_not": "no crossing nearby", "seg": seg_txt})
            continue
        for c in near:
            x, y = ball[c]
            line = ("goal_low" if x < 0 else "goal_high" if x > length else
                    "touch_low" if y < 0 else "touch_high")
            out_m = max(-x, x - length, -y, y - width)
            reasons = []
            if rolled[c] - rolled[max(c - FPS, 0)] == 0:
                reasons.append("not_rolling")
            if fast_in[min(c + 1 + confirm, n)] - fast_in[min(c + 1, n)] > 0:
                reasons.append("fast_back")
            if any_in[min(c + 1 + restart, n)] - any_in[min(c + 1, n)] > 0:
                reasons.append("back_soon")
            k = c
            while k < n and not sig["inside"][k]:
                k += 1
            sp_in = sig["speed"][max(c - FPS, 0):c]
            sp_back = sig["speed"][k:k + FPS]
            rows.append({"frame": f, "cross": int(c), "accepted": int(c) in accepted,
                         "why_not": "+".join(reasons) if int(c) not in accepted else "",
                         "line": line, "out_m": round(float(out_m), 1),
                         "speed_in": round(float(np.nanmedian(sp_in)), 1) if np.isfinite(sp_in).any() else np.nan,
                         "outside_s": round((k - c) / FPS, 2),
                         "unseen_s": round(float((~sig["seen"][c:k]).sum()) / FPS, 2),
                         "back_speed": round(float(np.nanmedian(sp_back)), 1) if np.isfinite(sp_back).any() else np.nan,
                         "seg": seg_txt})
    print(pd.DataFrame(rows).to_string(index=False))


if __name__ == '__main__':
    main()
