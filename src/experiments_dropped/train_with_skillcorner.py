"""실험: DFL 7경기 + SkillCorner 20경기로 학습하면 나아지는지 본다. Metrica 2경기는 학습에 쓰지 않는 외부 시험.

채택 안 함: Metrica 3경기에서 비중 맞춘 모델 0.811 대 지금 모델 0.809(1경기만 개선), DFL 0.869 대 0.878.

    python src/train_with_skillcorner.py

지금 모델 파일(data/external/dfl/model_all_v2.txt)은 건드리지 않는다. 실험 모델은 SkillCorner 폴더에 따로 저장한다.
비교(모두 인플레이 구간 F1, 모델만 / + 라인 아웃 처리):
  1. DFL: 경기 하나씩 빼고 학습(빼는 경기 = 평가). 학습 = 나머지 DFL 6경기(+ SkillCorner 20경기).
  2. Metrica 2경기: DFL 7경기(+ SkillCorner 20경기)로 학습.
  3. SkillCorner: 5경기씩 4묶음, 한 묶음씩 빼고 학습. 학습 = DFL 7경기(+ 나머지 SkillCorner 15경기).
학습 방식 세 가지: dfl_only(지금 방식), dfl_sc(행 그대로 합침), dfl_sc_balanced(DFL 전체와 SkillCorner 전체의
가중치 합이 같게). SkillCorner 선수는 화면 밖 추정 위치까지 포함(all)한 것, 정답은 skillcorner_eval.py와 같다.
DFL 학습 경기는 지금처럼 무작위 누락 시뮬레이션(rand), 평가 경기는 기본 시뮬레이션(sim).
채택 기준(미리 정함): DFL에서 손해가 없고 Metrica에서 좋아질 때만. 대회 데이터와 공개 정답 5분은 쓰지 않는다.
"""

import glob
import os
import sys
import time

import numpy as np
import pandas as pd

HERE = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))  # src/ (이 파일은 experiments_dropped/ 안)
ROOT = os.path.dirname(HERE)
sys.path[:0] = [HERE, os.path.join(ROOT, "starter")]
from config import CONFIG  # noqa: E402
import compare_smoothing as cs  # noqa: E402
import dfl_load  # noqa: E402
import features as fx  # noqa: E402
import final_dead_balls as fd  # noqa: E402
import metrica_load as ml  # noqa: E402
import skillcorner_eval as se  # noqa: E402
import smoothing  # noqa: E402
import train_dfl  # noqa: E402

FPS = CONFIG["fps"]
STRIDE = CONFIG["train_stride"]
SC_FEAT = os.path.join(se.SC_DIR, "features_all")
OUT = os.path.join(se.SC_DIR, "train_with_sc_results.csv")
VARIANTS = ("dfl_only", "dfl_sc", "dfl_sc_balanced")


def sc_ids():
    return sorted(os.path.basename(p) for p in glob.glob(os.path.join(se.SC_DIR, "*")) if os.path.isdir(p)
                  and os.path.exists(os.path.join(p, os.path.basename(p) + "_match.json")))


def sc_match(mid):
    """(features with period/dead, ball signals). 특징은 캐시한다."""
    mid_path = os.path.join(se.SC_DIR, mid)
    _, pitch, team, pos, det, ball10, period10, dead10 = se.load(mid_path)
    n = int((len(period10) - 1) * FPS / se.SC_FPS) + 1
    period, dead, ball = se.nearest(period10, n), se.nearest(dead10, n), se.upsample(ball10, n)
    sig = fd.ball_signals(fx.Match(players=np.zeros((n, 0, 2), np.float32), team=team[:0], ball=ball,
                                   period=period, pitch=pitch))
    path = os.path.join(SC_FEAT, mid + ".parquet")
    if os.path.exists(path):
        frame = pd.read_parquet(path)
    else:
        players = se.upsample(pos, n).astype(np.float32)
        frame = fx.compute_features(fx.Match(players=players, team=team, ball=ball, period=period, pitch=pitch))
        frame = frame[["period"] + train_dfl.model_columns(frame.columns)].astype(np.float32)
        frame["dead"] = dead.astype(np.int8)
        os.makedirs(SC_FEAT, exist_ok=True)
        frame.to_parquet(path)
    return frame, sig


def sc_train_rows(mid):
    frame, _ = sc_match(mid)
    frame = frame[frame["period"] > 0]
    return frame.iloc[::STRIDE]


def dfl_train_rows(i, m):
    return train_dfl.match_features(m, "rand", CONFIG["sim_seed"] + i).iloc[::STRIDE]


def fit(dfl_rows, sc_rows, variant):
    import lightgbm as lgb
    parts = list(dfl_rows) + (list(sc_rows) if variant != "dfl_only" else [])
    data = pd.concat(parts, ignore_index=True)
    columns = train_dfl.model_columns(data.columns)
    weight = None
    if variant == "dfl_sc_balanced":
        n_dfl = sum(len(f) for f in dfl_rows)
        n_sc = len(data) - n_dfl
        weight = np.concatenate([np.full(n_dfl, 0.5 * len(data) / n_dfl), np.full(n_sc, 0.5 * len(data) / n_sc)])
    params = dict(CONFIG["lgbm_params"])
    rounds = params.pop("n_estimators")
    params.update(objective="binary", bagging_fraction=params.pop("subsample"),
                  bagging_freq=params.pop("subsample_freq"), feature_fraction=params.pop("colsample_bytree"),
                  min_data_in_leaf=params.pop("min_child_samples"), seed=params.pop("random_state"))
    model = lgb.train(params, lgb.Dataset(data[columns], data["dead"], weight=weight), num_boost_round=rounds)
    return model, columns


def pipeline_scores(score, period, dead, sig):
    """모델만 / + 라인 아웃 처리(final_dead_balls.py와 같은 함수)의 인플레이 구간 F1."""
    choice = CONFIG["smooth_choice"]
    truth = pd.DataFrame({"period": period, "dead": np.asarray(dead).astype(np.int8)})
    base = smoothing.viterbi(score, period, choice["penalty"], choice["bias"]).astype(bool)
    crossings = fd.line_crossings(sig)
    free = np.zeros(len(score), dtype=bool)
    free[crossings] = True
    entry = smoothing.viterbi(score, period, choice["penalty"], choice["bias"], free_entry=free).astype(bool)
    forced, _ = fd.force_line_outs(entry, crossings, sig, score, period)
    graded = np.asarray(period) > 0
    return {"model_only": cs.score_one((base & graded).astype(np.int8), truth)["alive_spell_f1"],
            "plus_line_outs": cs.score_one((forced & graded).astype(np.int8), truth)["alive_spell_f1"]}


def main():
    pd.set_option("display.width", 200)
    started = time.time()
    ids = sc_ids()
    print("SkillCorner matches", len(ids), flush=True)
    sc_rows = {mid: sc_train_rows(mid) for mid in ids}
    dfl_rows = {m: dfl_train_rows(i, m) for i, m in enumerate(dfl_load.MATCHES)}
    print("training rows: DFL {}, SkillCorner {} ({:.0f}s)".format(
        sum(len(f) for f in dfl_rows.values()), sum(len(f) for f in sc_rows.values()), time.time() - started),
        flush=True)
    rows = []

    # 1. DFL 경기 하나씩 빼고
    for held in dfl_load.MATCHES:
        tracking, _, meta = dfl_load.load_match(held)
        sig = fd.ball_signals(fx.match_from_dfl(tracking, meta))
        test = train_dfl.match_features(held, "sim", CONFIG["sim_seed"])
        for v in VARIANTS:
            model, columns = fit([f for m, f in dfl_rows.items() if m != held], sc_rows.values(), v)
            score = model.predict(test[columns]).astype(np.float64)
            for stage, f1 in pipeline_scores(score, test["period"].to_numpy().astype(np.int64),
                                             test["dead"].to_numpy(), sig).items():
                rows.append(dict(test_set="dfl", match=held, variant=v, stage=stage, alive_spell_f1=f1))
        print("dfl", held, "done ({:.0f}s)".format(time.time() - started), flush=True)

    # 2. Metrica: DFL 7경기(+ SkillCorner 20경기)
    metrica = []
    for g in ml.GAMES:
        tracking, events = ml.load_game(g)
        spells = ml.dead_spells(tracking, events)
        match = ml.match_from_metrica(tracking)
        frame = fx.compute_features(match)
        metrica.append((g, frame, match.period, ml.dead_label(tracking, spells), fd.ball_signals(match)))
    for v in VARIANTS:
        model, columns = fit(dfl_rows.values(), sc_rows.values(), v)
        if v != "dfl_only":
            model.save_model(os.path.join(se.SC_DIR, "model_{}_v2.txt".format(v)))
        for g, frame, period, label, sig in metrica:
            score = model.predict(frame[columns].astype(np.float32)).astype(np.float64)
            for stage, f1 in pipeline_scores(score, period, label, sig).items():
                rows.append(dict(test_set="metrica", match="game{}".format(g), variant=v, stage=stage,
                                 alive_spell_f1=f1))
    print("metrica done ({:.0f}s)".format(time.time() - started), flush=True)

    # 3. SkillCorner: 5경기씩 묶음 하나씩 빼고
    folds = [ids[i::4] for i in range(4)]
    for k, held in enumerate(folds):
        tests = {mid: sc_match(mid) for mid in held}
        for v in VARIANTS:
            model, columns = fit(dfl_rows.values(), [f for m, f in sc_rows.items() if m not in held], v)
            for mid in held:
                frame, sig = tests[mid]
                score = model.predict(frame[columns]).astype(np.float64)
                for stage, f1 in pipeline_scores(score, frame["period"].to_numpy().astype(np.int64),
                                                 frame["dead"].to_numpy(), sig).items():
                    rows.append(dict(test_set="skillcorner", match=mid, variant=v, stage=stage, alive_spell_f1=f1))
        print("skillcorner fold", k, "done ({:.0f}s)".format(time.time() - started), flush=True)

    t = pd.DataFrame(rows)
    t.to_csv(OUT, index=False)
    print("\n== mean alive-spell F1 ==")
    print(t.pivot_table(index=["test_set", "stage"], columns="variant", values="alive_spell_f1").round(4).to_string())
    print("\n== per match vs dfl_only (plus line-outs): better / worse ==")
    p = t[t["stage"] == "plus_line_outs"].pivot_table(index=["test_set", "match"], columns="variant",
                                                       values="alive_spell_f1")
    for v in VARIANTS[1:]:
        d = (p[v] - p["dfl_only"]).groupby(level=0)
        print(v, {k: "{:+.4f} ({} / {})".format(g.mean(), int((g > 1e-9).sum()), int((g < -1e-9).sum()))
                  for k, g in d})
    print("\nper match (plus line-outs):")
    print(p.round(3).to_string())
    print("wrote", OUT, "({:.0f}s)".format(time.time() - started))


if __name__ == '__main__':
    main()
