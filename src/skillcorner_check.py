"""SkillCorner 공개 데이터(A리그 2024/25, 10경기, 중계 기반 10 fps)로 공 기반 규칙의 손해 쪽을 잰다.

    python src/skillcorner_check.py

DFL은 데드볼 동안 공을 고정하고 Metrica는 지워서, '진짜 데드볼인데 공이 움직이는' 경우를 잴 수 없었다.
SkillCorner는 데드볼 동안에도 화면에 잡힌 공을 기록한다(1경기 확인: 데드볼 동안 공 검출 20%).
정답: phases_of_play.csv는 공이 인플레이일 때만 정의된다(README). 같은 하프에서 phase 사이 빈 구간 = 데드볼.
원인: dynamic_events.csv의 game_interruption_after(그 이벤트 뒤의 중단 종류).
공: is_detected인 프레임만 쓴다(추정 좌표 제외). 좌표 원점은 경기장 중앙, 단위 m.

재는 것(규칙 하나를 미리 정해 둔다, 기준값은 config의 기존 값):
  '공이 매 프레임 보이고, 경기장 안이고, line_out_moving_mps(2 m/s)보다 빠르게 움직인다'가 W초 이어지는 창.
  - 인플레이와 데드볼에서 이런 창이 나타나는 비율
  - 데드볼 구간별: 이런 창이 있는지, 있으면 구간 안 어디인지(시작 몇 초 뒤, 재개 몇 초 전)
데이터: data/external/skillcorner/<match_id>/ (커밋 안 함). 출처: SkillCorner Open Data (MIT).
"""

import glob
import json
import os
import sys

import numpy as np
import pandas as pd

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(HERE)
sys.path.insert(0, HERE)
from config import CONFIG  # noqa: E402
from features import runs_of  # noqa: E402

SC_DIR = os.path.join(ROOT, "data", "external", "skillcorner")
SC_FPS = 10
MARGIN = CONFIG["final"]["boundary_margin_m"]
MOVING = CONFIG["final"]["line_out_moving_mps"]
WINDOWS_S = (2, 3, 4)


def load_match(path):
    mid = os.path.basename(path)
    meta = json.load(open(os.path.join(path, mid + "_match.json"), encoding="utf-8"))
    rows = []
    with open(os.path.join(path, mid + "_tracking_extrapolated.jsonl"), encoding="utf-8") as handle:
        for line in handle:
            r = json.loads(line)
            b = r["ball_data"]
            rows.append((r["frame"], r["period"], b["x"], b["y"], b["is_detected"]))
    t = pd.DataFrame(rows, columns=["frame", "period", "x", "y", "det"])
    t = t.set_index("frame").reindex(np.arange(t["frame"].max() + 1))
    phases = pd.read_csv(os.path.join(path, mid + "_phases_of_play.csv"))
    events = pd.read_csv(os.path.join(path, mid + "_dynamic_events.csv"), low_memory=False)
    return mid, meta, t, phases, events


def ball_signals(t, length, width):
    x, y = t["x"].to_numpy(float), t["y"].to_numpy(float)
    seen = (t["det"] == True).to_numpy() & np.isfinite(x) & np.isfinite(y)  # noqa: E712
    inside = seen & (np.abs(x) <= length / 2 + MARGIN) & (np.abs(y) <= width / 2 + MARGIN)
    # 속도: 3프레임 이동평균 위치의 2프레임 차분(약 0.3초 / 0.2초, 25 fps 파이프라인과 비슷한 시간 폭)
    xs = pd.Series(np.where(seen, x, np.nan)).rolling(3, center=True).mean().to_numpy()
    ys = pd.Series(np.where(seen, y, np.nan)).rolling(3, center=True).mean().to_numpy()
    speed = np.full(len(x), np.nan)
    speed[1:-1] = np.hypot(xs[2:] - xs[:-2], ys[2:] - ys[:-2]) * SC_FPS / 2
    with np.errstate(invalid="ignore"):
        moving = inside & (speed > MOVING)
    return seen, moving


def dead_spells(t, phases, events):
    """[(start, stop, cause)] 같은 하프 phase 사이 빈 구간. cause는 직전 중단 이벤트(±2초 안), 없으면 '?'."""
    ph = phases.sort_values("frame_start")
    stops = events[events["game_interruption_after"].notna()][["frame_end", "game_interruption_after"]]
    out = []
    for e, s, p1, p2 in zip(ph["frame_end"][:-1], ph["frame_start"][1:], ph["period"][:-1], ph["period"][1:]):
        if s - e <= 1 or p1 != p2:
            continue
        cause = "?"
        if len(stops):
            k = int(np.argmin(np.abs(stops["frame_end"].to_numpy() - e)))
            if abs(stops["frame_end"].iloc[k] - e) <= 2 * SC_FPS:
                cause = str(stops["game_interruption_after"].iloc[k]).replace("_for", "").replace("_against", "")
        out.append((int(e) + 1, int(s), cause))
    return out


def main():
    pd.set_option("display.width", 200)
    paths = sorted(p for p in glob.glob(os.path.join(SC_DIR, "*")) if os.path.isdir(p))
    window_rows, spell_rows, match_rows = [], [], []
    for path in paths:
        mid, meta, t, phases, events = load_match(path)
        length, width = float(meta["pitch_length"]), float(meta["pitch_width"])
        n = len(t)
        play = t["period"].notna().to_numpy()
        alive = np.zeros(n, dtype=bool)
        for a, b in zip(phases["frame_start"], phases["frame_end"]):
            alive[int(a):int(b) + 1] = True
        alive &= play
        spells = dead_spells(t, phases, events)
        dead = np.zeros(n, dtype=bool)
        for a, b, _ in spells:
            dead[a:b] = True
        seen, moving = ball_signals(t, length, width)
        match_rows.append({"match": mid, "dead_spells": len(spells), "alive_min": alive.sum() / SC_FPS / 60,
                           "dead_min": dead.sum() / SC_FPS / 60, "ball_seen_dead": seen[dead].mean(),
                           "ball_seen_alive": seen[alive].mean()})
        c = np.concatenate(([0], np.cumsum(moving)))
        for w_s in WINDOWS_S:
            w = int(w_s * SC_FPS)
            full = (c[w:] - c[:-w]) == w           # 창 [i, i+w) 전체가 움직이는 공
            for name, mask in (("dead", dead), ("alive", alive)):
                cm = np.concatenate(([0], np.cumsum(mask)))
                inside_mask = (cm[w:] - cm[:-w]) == w
                window_rows.append({"match": mid, "window_s": w_s, "state": name,
                                    "windows": int(inside_mask.sum()), "fires": int((full & inside_mask).sum())})
        w = 3 * SC_FPS
        for a, b, cause in spells:
            hits = [i for i in range(a, b - w + 1) if c[i + w] - c[i] == w]
            spell_rows.append({"match": mid, "cause": cause, "seconds": (b - a) / SC_FPS,
                               "seen_share": seen[a:b].mean(), "fires": bool(hits),
                               "first_s": (hits[0] - a) / SC_FPS if hits else np.nan,
                               "before_restart_s": (b - (hits[-1] + w)) / SC_FPS if hits else np.nan})
        print("done", mid, flush=True)

    m = pd.DataFrame(match_rows)
    print("\n== matches ==")
    print(m.round(3).to_string(index=False))
    wr = pd.DataFrame(window_rows).groupby(["window_s", "state"])[["windows", "fires"]].sum()
    wr["share"] = wr["fires"] / wr["windows"]
    print("\n== windows where the ball is seen inside and moving > {} m/s on every frame ==".format(MOVING))
    print(wr.round(4).to_string())
    s = pd.DataFrame(spell_rows)
    print("\n== dead spells with such a 3 s window: {} of {} ==".format(int(s["fires"].sum()), len(s)))
    print(s.groupby("cause")["fires"].agg(spells="size", with_window="sum").to_string())
    f = s[s["fires"]]
    if len(f):
        early = f["first_s"] <= 5
        print("window starts within 5 s of the spell start: {} of {}".format(int(early.sum()), len(f)))
        print("windows not near the start (> 5 s in): before restart s p10/p50/p90 {}".format(
            " / ".join("{:.1f}".format(v) for v in np.percentile(f.loc[~early, "before_restart_s"], [10, 50, 90]))
            if (~early).any() else "-"))
        short = s["seconds"] < 10
        print("spells shorter than 10 s: {}, with a window {}".format(int(short.sum()), int(s.loc[short, "fires"].sum())))


if __name__ == '__main__':
    main()
