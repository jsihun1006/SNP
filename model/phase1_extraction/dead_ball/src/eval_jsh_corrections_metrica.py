"""JSH 브랜치의 'LightGBM v2 데드볼 보정(①~⑥ 규칙 결합)'을 Metrica 샘플 3경기로 외부 검증한다.

    python src/eval_jsh_corrections_metrica.py

노트북(JSH:model/영상추출/모델/dead_ball/LightGBM_deadball_corrections.ipynb)의 코드 셀 2·4·6을 git에서 그대로 읽어
실행한다(복사하지 않는다). 대회 적용과 같은 조합: VARIANTS['rules_combined'], COMPETITION_RULES.
비교 기준은 노트북의 aligned_baseline(= 지금 파이프라인의 비터비 + 라인 아웃 처리, 우리 eval과 같은 계산).
모델은 이 로컬의 data/external/dfl/model_all_v2.txt(현재 HSJ 모델). Metrica는 학습에 쓰지 않았다.
판정 기준(결과 보기 전에 정함, 2026-11-14): Metrica 3경기 평균이 높고 3경기 중 2경기 이상 개선이면 통과.
"""

import json
import os
import subprocess
import sys

import numpy as np
import pandas as pd

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = __import__("repo_root").ROOT
sys.path[:0] = [HERE, os.path.join(ROOT, "starter")]
NOTEBOOK = "JSH:model/영상추출/모델/dead_ball/LightGBM_deadball_corrections.ipynb"


def load_notebook_code(cells=(2, 4, 6)):
    raw = subprocess.run(["git", "show", NOTEBOOK], cwd=ROOT, capture_output=True, check=True).stdout
    nb = json.loads(raw.decode("utf-8"))
    namespace = {"__name__": "jsh_notebook"}
    old = os.getcwd()
    os.chdir(ROOT)   # 노트북은 현재 폴더에서 저장소 루트를 찾는다
    try:
        for i in cells:
            assert nb["cells"][i]["cell_type"] == "code"
            exec(compile("".join(nb["cells"][i]["source"]), "jsh_cell_{}".format(i), "exec"), namespace)
    finally:
        os.chdir(old)
    return namespace


def main():
    import lightgbm as lgb
    import compare_smoothing as cs
    import features as fx
    import final_dead_balls as fd
    import metrica_load as ml
    import train_dfl
    from config import CONFIG
    from eval_metrica3 import load_game3

    jsh = load_notebook_code()
    model = lgb.Booster(model_file=train_dfl.model_path(CONFIG["default_feature_set"]))
    games = []
    for g in ml.GAMES:
        tracking, events = ml.load_game(g)
        games.append(("game{}".format(g), ml.match_from_metrica(tracking),
                      ml.dead_label(tracking, ml.dead_spells(tracking, events))))
    m3, l3, _ = load_game3()
    games.append(("game3", m3, l3))

    rows = []
    for name, match, label in games:
        period = np.asarray(match.period).astype(np.int64)
        keep = period > 0
        match = fx.Match(players=match.players[keep], team=match.team, ball=match.ball[keep], period=period[keep],
                         pitch=match.pitch, frame=None)
        label = np.asarray(label)[keep].astype(np.int8)
        frame = fx.compute_features(match)
        score = model.predict(frame[model.feature_name()].astype(np.float32)).astype(np.float64)
        geometry = fd.ball_signals(match)
        crossings = fd.line_crossings(geometry)
        crossings = crossings[(crossings > 0) & (match.period[crossings] == match.period[crossings - 1])]
        signals = [jsh["movement_features"](match.players[h], match.team, match.ball[h], match.pitch)
                   for h in jsh["halves"](match.period)]
        game = dict(period=match.period, conditions={"original": (geometry, crossings, signals)})
        truth = pd.DataFrame({"period": match.period, "dead": label})
        base = jsh["aligned_baseline"](score, game, "original")
        corrected, _ = jsh["decode"](score, game, "original", jsh["VARIANTS"]["rules_combined"],
                                     jsh["COMPETITION_RULES"])
        only5, _ = jsh["decode"](score, game, "original", jsh["VARIANTS"]["5_false_dead"], jsh["COMPETITION_RULES"])
        for variant, state in (("baseline", base), ("rules_combined", corrected), ("5_false_dead_only", only5)):
            r = cs.score_one(state.astype(np.int8), truth)
            rows.append(dict(game=name, variant=variant, alive_spell_f1=r["alive_spell_f1"], dead_f1=r["dead_f1"],
                             pred_spells=r["pred_spells"], true_spells=r["true_spells"],
                             dead_share=float(state.mean())))
        print("done", name, flush=True)

    t = pd.DataFrame(rows)
    pd.set_option("display.width", 200)
    p = t.pivot(index="game", columns="variant", values="alive_spell_f1")
    print("\n== alive-spell F1 (Metrica, not used in training) ==")
    print(p.round(4).to_string())
    print("\nmean:", p.mean().round(4).to_dict())
    print("\ndetails:")
    print(t.round(4).to_string(index=False))
    for v in ("rules_combined", "5_false_dead_only"):
        d = p[v] - p["baseline"]
        ok = (p[v].mean() > p["baseline"].mean()) and int((d > 1e-9).sum()) >= 2
        print("{}: mean diff {:+.4f}, better {} / worse {} -> {}".format(
            v, d.mean(), int((d > 1e-9).sum()), int((d < -1e-9).sum()), "PASS" if ok else "not passed"))


if __name__ == '__main__':
    main()
