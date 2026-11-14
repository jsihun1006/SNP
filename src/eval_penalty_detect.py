"""실험: 선수 배치만으로 페널티킥 준비를 찾는다(경기 규칙 14조). DFL 7경기, SkillCorner 20경기, Metrica 3경기에서
실제 페널티킥을 찾는지(재현율)와 다른 곳에서 잘못 걸리는지(가짜 감지)를 본다.

    python src/eval_penalty_detect.py

규칙 14조: 차기 전 키커와 수비 골키퍼를 뺀 모든 선수는 페널티 박스 밖, 페널티 마크보다 뒤, 마크에서 9.15 m 밖에
있어야 한다. 골키퍼는 차는 순간 골라인 위에 있어야 한다.
감지(프레임): 한쪽 박스 안 선수가 정확히 2명, 그중 한 명은 골라인 위(GOAL_LINE_M 안, 골대 사이),
마크에서 9.15 m 안의 선수는 그 2명뿐, 보이는 선수가 MIN_VISIBLE명 이상. 이 상태가 MIN_HOLD_S초 이어지면 준비 구간.
차는 순간(추정) = 준비 구간의 끝(선수들이 박스로 들어오는 순간. 규칙상 차기 전에는 들어올 수 없다).
대문자 상수는 좌표 흔들림·누락을 감안한 감지 허용치다(정답이나 대회 장면에 맞춘 값이 아니다).
정답: DFL은 재개가 penalty인 데드볼, SkillCorner는 중단 종류가 penalty인 데드볼(skillcorner_check.dead_spells)과
마크에서 시작한 free_kick 재개(spot_free_kicks), Metrica는 이벤트 하위 종류 PENALTY.
"""

import glob
import json
import os
import sys

import numpy as np
import pandas as pd

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(HERE)
sys.path[:0] = [HERE, os.path.join(ROOT, "starter")]
from config import CONFIG  # noqa: E402
import features as fx  # noqa: E402

FPS = CONFIG["fps"]
BOX_DEPTH, BOX_HALF, SPOT, ARC, POST_HALF = 16.5, 20.16, 11.0, 9.15, 3.66   # 경기 규칙 1조·14조
GOAL_LINE_M = 1.5
MIN_VISIBLE = 16
MIN_HOLD_S = 1.0
SPOT_LABEL_TOL_M = 2.0   # SkillCorner 이벤트 위치가 마크에서 이만큼 안이면 마크 시작으로 본다


def setups(match):
    """[(start, stop, goal_side)] 페널티킥 준비 구간. goal_side 0: x=0 골, 1: x=length 골."""
    length, width = match.pitch
    p = match.players.astype(np.float64)
    x, y = p[..., 0], p[..., 1]
    seen = np.isfinite(x) & np.isfinite(y)
    visible = seen.sum(axis=1)
    out = []
    for side in (0, 1):
        depth = x if side == 0 else length - x
        spot_x = SPOT if side == 0 else length - SPOT
        with np.errstate(invalid="ignore"):
            in_box = seen & (depth <= BOX_DEPTH) & (depth >= -1) & (np.abs(y - width / 2) <= BOX_HALF)
            on_line = seen & (depth <= GOAL_LINE_M) & (np.abs(y - width / 2) <= POST_HALF)
            near_spot = seen & (np.hypot(x - spot_x, y - width / 2) < ARC)
        ok = (in_box.sum(axis=1) == 2) & (on_line & in_box).any(axis=1) & ((near_spot & ~in_box).sum(axis=1) == 0)
        ok &= (near_spot.sum(axis=1) <= 2) & (visible >= MIN_VISIBLE) & (np.asarray(match.period) > 0)
        for a, b in fx.runs_of(ok):
            if b - a >= MIN_HOLD_S * FPS:
                out.append((a, b, side))
    return sorted(out)


def score(name, match, penalty_spells):
    """penalty_spells: [(start, stop)] 정답 페널티 데드볼(차는 순간 = stop)."""
    found = setups(match)
    rows = []
    hit_true = set()
    for a, b, side in found:
        match_k = [k for k, (s, e) in enumerate(penalty_spells) if s - 5 * FPS <= b <= e + 5 * FPS]
        if match_k:
            hit_true.update(match_k)
        rows.append({"data": name, "setup_start": a, "setup_end": b, "hold_s": (b - a) / FPS,
                     "true_penalty": bool(match_k),
                     "kick_error_s": (b - penalty_spells[match_k[0]][1]) / FPS if match_k else np.nan})
    return rows, len(penalty_spells), len(hit_true)


def dfl_sets():
    import dfl_labels
    import dfl_load
    import simulate
    for m in dfl_load.MATCHES:
        tracking, events, meta = dfl_load.load_match(m)
        spells = dfl_labels.dead_spells(tracking, events)
        pens = [(int(s.start), int(s.stop)) for s in spells.itertuples() if s.cause == "penalty"]
        raw = fx.match_from_dfl(tracking, meta)
        yield "dfl:" + m, raw, pens
        yield "dfl_sim:" + m, simulate.simulate(raw, seed=CONFIG["sim_seed"]), pens


def spot_free_kicks(events, pitch, known):
    """SkillCorner가 free_kick으로 분류한 재개 중 페널티 마크에서 시작한 것(2026-11-15 추가). 규칙 1조: 마크는
    골라인에서 11 m, 폭의 가운데. 마크에서 차는 프리킥은 없으므로 페널티킥이다(1874553에서 마크 시작 → 슛 → 골로 확인).
    좌표는 경기장 가운데가 원점. SPOT_LABEL_TOL_M은 이벤트 위치 허용치. 반환 [(kick - 30초, kick)] (25 fps)."""
    import skillcorner_eval as se
    length, _ = pitch
    # 받은 선수의 소유(start_type *_reception)는 뺀다: 다른 곳에서 찬 프리킥을 마크 근처에서 받은 것일 뿐이다
    # (20경기 중 4건: 받은 뒤 슛·패스, 골키퍼가 백패스를 받음).
    e = events[(events["game_interruption_before"] == "free_kick_for")
               & ~events["start_type"].fillna("").astype(str).str.endswith("_reception")]
    with np.errstate(invalid="ignore"):
        on_spot = np.hypot(np.abs(e["x_start"].astype(float)) - (length / 2 - SPOT), e["y_start"].astype(float)) <= SPOT_LABEL_TOL_M
    out = []
    for f10 in sorted(set(int(f) for f in e.loc[on_spot, "frame_start"])):
        k = int(f10 * FPS / se.SC_FPS)
        if not any(abs(k - b) <= 5 * FPS for _, b in known + out):
            out.append((k - 30 * FPS, k))
    return out


def penalty_goal_kickoffs(events, spells, known):
    """페널티 골(2026-11-15 추가). SkillCorner는 페널티킥이 골이 되면 그 뒤 킥오프 소유에만
    game_interruption_before = penalty_for/against를 붙이고, 페널티킥 앞 데드볼에는 중단 종류가 없다('?').
    2006229에서 확인: 166초 데드 → 1.2초 인플레이(차고 골) → 72초 데드 → penalty_against 킥오프.
    차는 순간 = 킥오프 앞 데드볼 바로 전 데드볼의 끝. spells는 skillcorner_check.dead_spells(10 fps). 반환 25 fps."""
    import skillcorner_eval as se
    e = events[events["game_interruption_before"].fillna("").astype(str).str.startswith("penalty")]
    ends = [b for _, b, _ in spells]
    out = []
    for f10 in sorted(set(int(f) for f in e["frame_start"])):
        j = int(np.argmin([abs(b - f10) for b in ends])) if ends else -1
        if j < 1 or abs(ends[j] - f10) > 2 * se.SC_FPS:
            continue
        k = int(ends[j - 1] * FPS / se.SC_FPS)
        if not any(abs(k - b) <= 5 * FPS for _, b in known + out):
            out.append((k - 30 * FPS, k))
    return out


def skillcorner_sets():
    import skillcorner_eval as se
    for path in sorted(p for p in glob.glob(os.path.join(se.SC_DIR, "*")) if os.path.isdir(p)
                       and os.path.exists(os.path.join(p, os.path.basename(p) + "_match.json"))):
        mid, pitch, team, pos, det, ball10, period10, dead10 = se.load(path)
        n = int((len(period10) - 1) * FPS / se.SC_FPS) + 1
        period = se.nearest(period10, n)
        import skillcorner_check as sk
        _, _, t10, phases, events = sk.load_match(path)
        spells = sk.dead_spells(t10, phases, events)
        pens = [(int(a * FPS / se.SC_FPS), int(b * FPS / se.SC_FPS)) for a, b, cause in spells if cause == "penalty"]
        pens += spot_free_kicks(events, pitch, pens)
        pens += penalty_goal_kickoffs(events, spells, pens)
        for variant in ("all", "detected"):
            p10 = pos.copy()
            if variant == "detected":
                p10[~det] = np.nan
            players = se.upsample(p10, n).astype(np.float32)
            yield "skillcorner_{}:{}".format(variant, mid), fx.Match(players=players, team=team, ball=se.upsample(ball10, n),
                                                                      period=period, pitch=pitch), pens


def metrica_sets():
    import metrica_load as ml
    from eval_metrica3 import load_game3, RAW3
    for g in ml.GAMES:
        tracking, events = ml.load_game(g)
        row_of = {int(f): i for i, f in enumerate(tracking["frame"].to_numpy())}
        pen = events[events["Subtype"].fillna("").str.contains("PENALTY")]
        pens = [(row_of.get(int(f), 0) - 30 * FPS, row_of.get(int(f), 0)) for f in pen["Start Frame"]]
        yield "metrica:game{}".format(g), ml.match_from_metrica(tracking), pens
    m3, _, _ = load_game3()
    ev3 = json.load(open(RAW3 + "events.json", encoding="utf-8"))["data"]
    pens3 = []
    for e in ev3:
        sub = e.get("subtypes")
        names = [sub.get("name", "")] if isinstance(sub, dict) else [s.get("name", "") for s in (sub or [])]
        if any("PENALTY" in (s or "") for s in names):
            f = int(e["start"]["frame"]) - 1
            pens3.append((f - 30 * FPS, f))
    yield "metrica:game3", m3, pens3


def main():
    pd.set_option("display.width", 200)
    rows, summary = [], []
    for gen in (dfl_sets, skillcorner_sets, metrica_sets):
        for name, match, pens in gen():
            r, n_true, n_hit = score(name, match, pens)
            rows += r
            summary.append({"data": name, "set": name.split(":")[0], "true_penalties": n_true, "found": n_hit,
                            "detections": len(r), "false_detections": sum(1 for x in r if not x["true_penalty"])})
            print("done", name, flush=True)
    s = pd.DataFrame(summary)
    print("\n== by data set ==")
    print(s.groupby("set")[["true_penalties", "found", "detections", "false_detections"]].sum().to_string())
    d = pd.DataFrame(rows)
    if len(d):
        print("\n== detections ==")
        print(d.round(2).to_string(index=False))


if __name__ == '__main__':
    main()
