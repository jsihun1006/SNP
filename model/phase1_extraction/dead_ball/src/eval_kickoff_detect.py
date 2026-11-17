"""실험: 선수 배치 킥오프 감지(src/kickoff.py)가 실제 킥오프를 찾는지, 다른 곳에서 잘못 걸리는지 본다.

    python src/eval_kickoff_detect.py

정답(킥오프 프레임):
  DFL       재개 종류가 kick_off인 데드볼의 끝 + 각 하프의 첫 프레임(하프 시작 킥오프)
  Metrica   이벤트 하위 종류 KICK OFF(game3은 이벤트 json)
  SkillCorner  game_interruption_before가 있는 소유 중 센터 마크 3 m 안에서 시작한 것 + 각 하프 첫 프레임
5초 안이면 맞은 것으로 센다. DFL은 원본과 선수 누락·흔들림 시뮬레이션, SkillCorner는 화면 밖 추정 위치 포함(all)과
검출 선수만(detected) 두 조건. 가짜 감지가 적어야 다음 단계(골 뒤 세리머니 메우기)에 쓸 수 있다.
"""

import glob
import json
import os
import sys

import numpy as np
import pandas as pd

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = __import__("repo_root").ROOT
sys.path[:0] = [HERE, os.path.join(ROOT, "starter")]
from config import CONFIG  # noqa: E402
import features as fx  # noqa: E402
import kickoff as ko  # noqa: E402

FPS = CONFIG["fps"]


def period_starts(period):
    period = np.asarray(period)
    return [int(i) for i in np.flatnonzero((period > 0) & (np.r_[0, period[:-1]] != period))]


def dfl_sets():
    import dfl_labels
    import dfl_load
    import simulate
    for m in dfl_load.MATCHES:
        tracking, events, meta = dfl_load.load_match(m)
        spells = dfl_labels.dead_spells(tracking, events)
        raw = fx.match_from_dfl(tracking, meta)
        truth = sorted(set([int(s.stop) for s in spells.itertuples() if s.cause == "kick_off"] + period_starts(raw.period)))
        yield "dfl:" + m, raw, truth
        yield "dfl_sim:" + m, simulate.simulate(raw, seed=CONFIG["sim_seed"]), truth


def metrica_sets():
    import metrica_load as ml
    from eval_metrica3 import load_game3, RAW3
    for g in ml.GAMES:
        tracking, events = ml.load_game(g)
        match = ml.match_from_metrica(tracking)
        row_of = {int(f): i for i, f in enumerate(tracking["frame"].to_numpy())}
        ko_ev = events[events["Subtype"].fillna("").str.contains("KICK OFF")]
        truth = sorted(set(row_of[int(f)] for f in ko_ev["Start Frame"] if int(f) in row_of))
        yield "metrica:game{}".format(g), match, truth
    m3, _, _ = load_game3()
    row_of = {int(f): k for k, f in enumerate(np.asarray(m3.frame))}
    truth = []
    for e in json.load(open(RAW3 + "events.json", encoding="utf-8"))["data"]:
        sub = e.get("subtypes")
        names = [sub.get("name", "")] if isinstance(sub, dict) else [s.get("name", "") for s in (sub or [])]
        if any("KICK OFF" in (s or "") for s in names) and int(e["start"]["frame"]) in row_of:
            truth.append(row_of[int(e["start"]["frame"])])
    yield "metrica:game3", m3, sorted(set(truth))


def skillcorner_sets():
    import skillcorner_eval as se
    import skillcorner_check as sk
    for path in sorted(p for p in glob.glob(os.path.join(se.SC_DIR, "*")) if os.path.isdir(p)
                       and os.path.exists(os.path.join(p, os.path.basename(p) + "_match.json"))):
        mid, pitch, team, pos, det, ball10, period10, dead10 = se.load(path)
        n = int((len(period10) - 1) * FPS / se.SC_FPS) + 1
        period = se.nearest(period10, n)
        _, _, _, _, events = sk.load_match(path)
        e = events[events["event_type"].eq("player_possession") & events["game_interruption_before"].notna()]
        with np.errstate(invalid="ignore"):
            center = np.hypot(e["x_start"].astype(float), e["y_start"].astype(float)) <= 3.0
        truth = sorted(set([int(f * FPS / se.SC_FPS) for f in e.loc[center, "frame_start"]] + period_starts(period)))
        for variant in ("all", "detected"):
            p10 = pos.copy()
            if variant == "detected":
                p10[~det] = np.nan
            players = se.upsample(p10, n).astype(np.float32)
            yield "skillcorner_{}:{}".format(variant, mid), fx.Match(players=players, team=team, ball=se.upsample(ball10, n),
                                                                      period=period, pitch=pitch), truth


def main():
    pd.set_option("display.width", 220)
    rows, errors, false_rows = [], [], []
    for gen in (dfl_sets, metrica_sets, skillcorner_sets):
        for name, match, truth in gen():
            found = [k for k, _ in ko.kickoffs(match)]
            hit = [t for t in truth if any(abs(k - t) <= 5 * FPS for k in found)]
            false = [k for k in found if not any(abs(k - t) <= 5 * FPS for t in truth)]
            rows.append({"set": name.split(":")[0], "true": len(truth), "found": len(hit), "detections": len(found),
                         "false": len(false)})
            for t in hit:
                k = min(found, key=lambda k: abs(k - t))
                errors.append({"set": name.split(":")[0], "error_s": (k - t) / FPS})
            false_rows += [{"data": name, "frame": k} for k in false]
            print("done", name, flush=True)
    t = pd.DataFrame(rows)
    print("\n== kickoffs found / false ==")
    print(t.groupby("set")[["true", "found", "detections", "false"]].sum().to_string())
    e = pd.DataFrame(errors)
    print("\n== kick time error (s), detection minus true ==")
    print(e.groupby("set")["error_s"].describe()[["count", "min", "25%", "50%", "75%", "max"]].round(2).to_string())
    if false_rows:
        print("\n== false detections ==")
        print(pd.DataFrame(false_rows).to_string(index=False))


if __name__ == '__main__':
    main()
