"""설정 정리(2026-11-17): 지금 설정(clean)과 그 전(before)을 같은 전체 파이프라인으로 비교.

    python src/eval_clean.py

before: 페널티킥 16명 하한 없음, 차는 순간 = 배치가 깨진 순간, 골키퍼 V4_ADOPTED(v9까지).
clean : 지금 설정(페널티킥 그대로, 골키퍼 EVIDENCE: pitch_check 제외).
데이터: DFL 7경기(원본, 쥠 구간 공 삭제, 날아오는 구간부터 공 삭제), Metrica 3경기, SkillCorner 20경기.
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
import compare_smoothing as cs  # noqa: E402
import keeper_hold as kh  # noqa: E402
from eval_dfl_current import run  # noqa: E402
from eval_goal_kickoff import sources  # noqa: E402
from eval_keeper_hold_size import keeper_hold_spans  # noqa: E402

FPS = CONFIG["fps"]
SETTINGS = {"before": ({"run_up_moment": "break", "min_visible": 0}, dict(kh.V4_ADOPTED)),
            "clean": ({"run_up_moment": "break", "min_visible": 0}, dict(kh.EVIDENCE))}


def score(match, score_, truth, variant):
    penalty, keeper = SETTINGS[variant]
    CONFIG["penalty"] = dict(penalty)
    kh.SETTING = dict(keeper)
    st = run(match, score_)
    return cs.score_one(st.astype(np.int8), truth)["alive_spell_f1"], float(st[np.asarray(match.period) > 0].mean())


def main():
    pd.set_option("display.width", 200)
    rows = []
    for name, match, s, label in sources():
        label = np.asarray(label).astype(bool)
        data, mid = name.split(":")[0], name.split(":")[1]
        truth = pd.DataFrame({"period": match.period, "dead": label.astype(np.int8)})
        conds = {data: match}
        if data == "dfl":
            spans = keeper_hold_spans(match, label)
            for cond, lead in (("dfl_hidden", 0), ("dfl_flight", FPS)):
                ball = np.asarray(match.ball, dtype=np.float64).copy()
                for a, b in spans:
                    ball[max(a - lead, 0):b] = np.nan
                conds[cond] = fx.Match(players=match.players, team=match.team, ball=ball, period=match.period,
                                       pitch=match.pitch)
        for cond, mm in conds.items():
            for variant in SETTINGS:
                f1, dead = score(mm, s, truth, variant)
                rows.append({"cond": cond, "match": mid, "variant": variant, "f1": f1, "dead_share": dead})
        print("done", name, flush=True)
    t = pd.DataFrame(rows)
    pv = t.pivot_table(index=["cond", "match"], columns="variant", values="f1")
    d = pv["clean"] - pv["before"]
    out = pd.DataFrame({"before": pv["before"].groupby("cond").mean(), "clean": pv["clean"].groupby("cond").mean(),
                        "diff": d.groupby("cond").mean(), "better": d.groupby("cond").apply(lambda x: int((x > 1e-9).sum())),
                        "worse": d.groupby("cond").apply(lambda x: int((x < -1e-9).sum()))})
    print("\n== clean vs before (in-play spell F1) ==")
    print(out.round(4).to_string())


if __name__ == "__main__":
    main()
