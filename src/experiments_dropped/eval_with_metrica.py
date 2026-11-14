"""[채택하지 않은 실험] 결과: Metrica를 학습에 넣어도 좋아지지 않았다. 인플레이 구간 F1 Metrica 0.800(DFL만) → 0.790(섞기) / 0.794(비중 맞추기), DFL 0.845 → 0.843 / 0.838.

실험: Metrica 샘플 2경기를 DFL과 함께 학습에 넣으면 나은가.

    python src/eval_with_metrica.py

9경기(DFL 7 + Metrica 2)에서 경기 하나씩 빼고 나머지 8경기로 학습해, 빠진 경기를 평가한다.
학습 경기는 지금과 같이 경기마다 무작위 트래킹 노이즈를 넣는다(train_dfl.random_settings).
평가 경기: DFL은 기본 노이즈 버전(지금 DFL 평가와 같음), Metrica는 원본(지금 metrica_eval.py와 같음).

비교:
  dfl_only        지금 모델. DFL 평가는 경기 하나씩 빼고 학습한 확률(oof), Metrica 평가는 DFL 7경기 전체 모델
  mixed           8경기를 그대로 섞어 학습
  mixed_balanced  두 데이터셋의 학습 프레임 비중이 같도록 가중치(Metrica 경기가 적어서)
점수: 인플레이 구간 F1(공식 채점기, 인플레이 = H, 데드 = D)과 데드볼 F1. 평활화는 지금 설정(비터비 벌점 40)만.
주의: 두 데이터셋의 데드볼 정의가 다르다(DFL 공식 BallStatus, Metrica는 이벤트로 만든 라벨).
라이선스: Metrica 샘플은 정식 라이선스가 없고 README가 출처 표기를 요청한다. 데이터는 배포하지 않는다.
"""

import os
import sys
import time

import numpy as np
import pandas as pd

HERE = os.path.dirname(os.path.abspath(__file__))
HERE = os.path.dirname(HERE)  # src/ — 이 파일은 src/experiments_dropped/ 안에 있다
sys.path.insert(0, HERE)
from config import CONFIG  # noqa: E402
import compare_smoothing as cs  # noqa: E402
import dfl_load  # noqa: E402
import features as fx  # noqa: E402
import metrica_load as ml  # noqa: E402
import simulate  # noqa: E402
import smoothing  # noqa: E402
import train_dfl  # noqa: E402

FPS = CONFIG["fps"]
CACHE = os.path.join(ml.CACHE, "features_" + CONFIG["default_feature_set"])


def metrica_frame(game, kind, seed):
    """kind: 'raw' | 'rand'. 특징 + dead 라벨, 캐시한다."""
    os.makedirs(CACHE, exist_ok=True)
    path = os.path.join(CACHE, "game{}_{}_{}.parquet".format(game, kind, seed))
    if os.path.exists(path):
        return pd.read_parquet(path)
    tracking, events = ml.load_game(game)
    match = ml.match_from_metrica(tracking)
    if kind == "rand":
        match = simulate.simulate(match, seed=seed, **train_dfl.random_settings(np.random.default_rng(seed)))
    frame = fx.compute_features(match)
    frame = frame[["period"] + train_dfl.model_columns(frame.columns)].astype(np.float32)
    frame["dead"] = ml.dead_label(tracking, ml.dead_spells(tracking, events))
    frame.to_parquet(path)
    return frame


def train(parts, weights=None):
    """parts: 이미 train_stride로 솎아 낸 학습 표들(메모리를 아끼려고 읽을 때 솎는다)."""
    import lightgbm as lgb
    data = pd.concat(parts, ignore_index=True)
    w = None
    if weights is not None:
        w = np.concatenate([np.full(len(p), wt) for p, wt in zip(parts, weights)])
    columns = train_dfl.model_columns(data.columns)
    params = dict(CONFIG["lgbm_params"])
    rounds = params.pop("n_estimators")
    params.update(objective="binary", bagging_fraction=params.pop("subsample"),
                  bagging_freq=params.pop("subsample_freq"), feature_fraction=params.pop("colsample_bytree"),
                  min_data_in_leaf=params.pop("min_child_samples"), seed=params.pop("random_state"))
    model = lgb.train(params, lgb.Dataset(data[columns], data["dead"], weight=w), num_boost_round=rounds)
    return model, columns


def score(prob, frame):
    choice = CONFIG["smooth_choice"]
    period = frame["period"].to_numpy().astype(int)
    state = smoothing.viterbi(prob, period, choice["penalty"], choice["bias"])
    r = cs.score_one(state, pd.DataFrame({"period": period, "dead": frame["dead"].to_numpy()}))
    return r["alive_spell_f1"], r["dead_f1"], state.mean(), frame["dead"].mean()


def load_test(held):
    source, key = held
    if source == "dfl":
        return train_dfl.match_features(key, "sim", CONFIG["sim_seed"])
    return metrica_frame(key, "raw", 0)


def main():
    """메모리를 아끼려고 학습 표는 솎아서만 들고 있고, 평가 표는 그 경기를 평가할 때만 읽는다.
    경기마다 결과를 바로 CSV에 덧붙여서 중간에 멈춰도 그때까지의 결과가 남는다."""
    pd.set_option("display.width", 200)
    train_dfl.set_feature_set(CONFIG["default_feature_set"])
    seed0 = CONFIG["sim_seed"]
    stride = CONFIG["train_stride"]
    games = [("dfl", m) for m in dfl_load.MATCHES] + [("metrica", g) for g in ml.GAMES]
    train_parts = {}
    for i, (source, key) in enumerate(games):
        full = (train_dfl.match_features(key, "rand", seed0 + i) if source == "dfl"
                else metrica_frame(key, "rand", seed0 + i))
        train_parts[(source, key)] = full.iloc[::stride].copy()
        del full
    print("training features ready", flush=True)

    import lightgbm as lgb
    full_dfl = lgb.Booster(model_file=train_dfl.model_path())
    out_path = os.path.join(ml.CACHE, "eval_with_metrica.csv")
    done = set()
    if os.path.exists(out_path):
        prev = pd.read_csv(out_path, dtype={"match": str})
        done = set(zip(prev["source"], prev["match"]))
    for held in games:
        if (held[0], str(held[1])) in done:
            print("{} {} already done".format(*held), flush=True)
            continue
        started = time.time()
        test = load_test(held)
        rows = []
        if held[0] == "dfl":
            oof = pd.read_parquet(os.path.join(train_dfl.oof_dir(), "{}_sim.parquet".format(held[1])))
            prob = oof["score"].to_numpy(np.float64)
        else:
            prob = full_dfl.predict(test[full_dfl.feature_name()])
        rows.append(("dfl_only",) + score(prob, test))
        others = [g for g in games if g != held]
        parts = [train_parts[g] for g in others]
        n_dfl = sum(len(train_parts[g]) for g in others if g[0] == "dfl")
        n_met = sum(len(train_parts[g]) for g in others if g[0] == "metrica")
        balanced = [1.0 if g[0] == "dfl" else n_dfl / float(n_met) for g in others]
        for name, weights in (("mixed", None), ("mixed_balanced", balanced)):
            model, columns = train(parts, weights)
            rows.append((name,) + score(model.predict(test[columns]), test))
            del model
        del test
        frame = pd.DataFrame([dict(source=held[0], match=str(held[1]), variant=v, alive_spell_f1=a, dead_f1=d,
                                   pred_dead=p, true_dead=t) for v, a, d, p, t in rows])
        frame.to_csv(out_path, mode="a", header=not os.path.exists(out_path), index=False)
        print("{} {} done in {:.0f}s".format(*held, time.time() - started), flush=True)

    table = pd.read_csv(out_path, dtype={"match": str})
    print()
    print("== alive-spell F1 per held-out match ==")
    print(table.pivot_table(index=["source", "match"], columns="variant", values="alive_spell_f1").round(3).to_string())
    print()
    print("== mean by source ==")
    print(table.groupby(["source", "variant"])[["alive_spell_f1", "dead_f1", "pred_dead", "true_dead"]].mean()
          .round(3).to_string())


if __name__ == '__main__':
    main()
