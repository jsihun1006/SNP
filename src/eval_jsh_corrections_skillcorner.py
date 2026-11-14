"""JSH 브랜치의 데드볼 보정(①~⑥ 규칙 결합, ⑤만)을 SkillCorner 20경기로 검증한다.

    python src/eval_jsh_corrections_skillcorner.py

SkillCorner는 데드볼 동안에도 화면에 잡힌 진짜 공을 기록한다. DFL(공 고정)·Metrica(공 삭제)와 달리 공 기반 규칙이
진짜 데드볼을 잘못 되돌리는 손해까지 보인다. 조건은 skillcorner_eval.py와 같다(25 fps 보간, 선수는 화면 밖 추정
위치 포함, 공은 검출된 프레임만, 정답은 phases of play 밖). 정답의 데드볼 시작이 실제보다 이른 점(추정)은
세 조건에 똑같이 적용된다.
판정 기준(결과 보기 전에 정함, 2026-11-14): 평균이 높고 악화 경기가 개선 경기보다 적으면 통과.
되돌린 프레임 중 실제 데드였던 비율(손해)도 함께 본다.
데이터: data/external/skillcorner/ (커밋 안 함). 출처: SkillCorner Open Data (MIT).
"""

import glob
import os
import sys

import numpy as np
import pandas as pd

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(HERE)
sys.path[:0] = [HERE, os.path.join(ROOT, "starter")]


def main():
    import lightgbm as lgb
    import compare_smoothing as cs
    import features as fx
    import final_dead_balls as fd
    import skillcorner_eval as se
    import train_dfl
    from config import CONFIG
    from eval_jsh_corrections_metrica import load_notebook_code

    jsh = load_notebook_code()
    model = lgb.Booster(model_file=train_dfl.model_path(CONFIG["default_feature_set"]))
    FPS = CONFIG["fps"]
    feat_dir = os.path.join(se.SC_DIR, "features_all")
    rows = []
    paths = sorted(p for p in glob.glob(os.path.join(se.SC_DIR, "*")) if os.path.isdir(p)
                   and os.path.exists(os.path.join(p, os.path.basename(p) + "_match.json")))
    for path in paths:
        mid, pitch, team, pos, det, ball10, period10, dead10 = se.load(path)
        n = int((len(period10) - 1) * FPS / se.SC_FPS) + 1
        period, dead, ball = se.nearest(period10, n), se.nearest(dead10, n), se.upsample(ball10, n)
        players = se.upsample(pos, n).astype(np.float32)
        cached = os.path.join(feat_dir, mid + ".parquet")
        full = fx.Match(players=players, team=team, ball=ball, period=period, pitch=pitch)
        frame = pd.read_parquet(cached) if os.path.exists(cached) else fx.compute_features(full)
        score_all = model.predict(frame[model.feature_name()].astype(np.float32)).astype(np.float64)
        keep = period > 0
        match = fx.Match(players=players[keep], team=team, ball=ball[keep], period=period[keep].astype(np.int64),
                         pitch=pitch)
        score, label = score_all[keep], dead[keep].astype(np.int8)
        geometry = fd.ball_signals(match)
        crossings = fd.line_crossings(geometry)
        crossings = crossings[(crossings > 0) & (match.period[crossings] == match.period[crossings - 1])]
        signals = [jsh["movement_features"](match.players[h], match.team, match.ball[h], match.pitch)
                   for h in jsh["halves"](match.period)]
        game = dict(period=match.period, conditions={"original": (geometry, crossings, signals)})
        truth = pd.DataFrame({"period": match.period, "dead": label})
        base = jsh["aligned_baseline"](score, game, "original")
        states = {"baseline": base}
        for v in ("rules_combined", "5_false_dead"):
            states[v], _ = jsh["decode"](score, game, "original", jsh["VARIANTS"][v], jsh["COMPETITION_RULES"])
        for v, state in states.items():
            r = cs.score_one(state.astype(np.int8), truth)
            to_alive = base & ~state           # 데드 -> 인플레이로 바뀐 프레임
            to_dead = ~base & state            # 인플레이 -> 데드로 바뀐 프레임
            rows.append(dict(match=mid, variant=v, alive_spell_f1=r["alive_spell_f1"], dead_f1=r["dead_f1"],
                             pred_spells=r["pred_spells"], true_spells=r["true_spells"],
                             to_alive_s=to_alive.sum() / FPS,
                             to_alive_truly_dead=float(label[to_alive].mean()) if to_alive.any() else np.nan,
                             to_dead_s=to_dead.sum() / FPS,
                             to_dead_truly_dead=float(label[to_dead].mean()) if to_dead.any() else np.nan))
        print("done", mid, flush=True)

    t = pd.DataFrame(rows)
    t.to_csv(os.path.join(se.SC_DIR, "jsh_corrections_results.csv"), index=False)
    pd.set_option("display.width", 220)
    p = t.pivot(index="match", columns="variant", values="alive_spell_f1")
    print("\n== alive-spell F1 per match ==")
    print(p.round(4).to_string())
    print("\n== mean ==")
    print(t.groupby("variant")[["alive_spell_f1", "dead_f1", "pred_spells", "true_spells", "to_alive_s",
                                "to_alive_truly_dead", "to_dead_s", "to_dead_truly_dead"]].mean().round(4).to_string())
    for v in ("rules_combined", "5_false_dead"):
        d = p[v] - p["baseline"]
        better, worse = int((d > 1e-9).sum()), int((d < -1e-9).sum())
        print("{}: mean diff {:+.4f}, better {} / worse {} -> {}".format(
            v, d.mean(), better, worse, "PASS" if d.mean() > 0 and worse < better else "not passed"))


if __name__ == '__main__':
    main()
