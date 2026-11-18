"""DFL 7경기로 데드볼 LightGBM을 학습하고 leave-one-match-out으로 평가한다.

    python src/deadball/train_dfl.py --quick        # 1경기(J03WPY)만 빼고 빠르게 확인
    python src/deadball/train_dfl.py                # 7경기 전부 돌아가며 평가

입력은 선수 특징과 그 시간 문맥만 쓴다(공 특징 제외, config.py 참고).
학습 경기는 경기마다 누락·흔들림 설정을 무작위로 뽑아 시뮬레이션한다(대회 데이터의 누락 모양을 정확히
알 수 없어서 특정 모양에 기대지 않게 한다). 평가 경기는 기본 설정으로 시뮬레이션한 것과 원본 둘 다 본다.
특징은 data/external/dfl/features/에 parquet로 캐시한다.
"""

from pathlib import Path
import sys

_SRC = Path(__file__).resolve().parents[1]
sys.path[:0] = [str(_SRC), str(_SRC.parent)]

import argparse
import os
import time

import numpy as np
import pandas as pd

HERE = os.path.dirname(os.path.dirname(os.path.realpath(__file__)))
sys.path.insert(0, HERE)
from config import CONFIG  # noqa: E402
import dfl_labels  # noqa: E402
import dfl_load  # noqa: E402
import features as fx  # noqa: E402
import metrics  # noqa: E402
import simulate  # noqa: E402

FEATURE_SET = CONFIG["default_feature_set"]  # main()의 --features로 바꾼다


def set_feature_set(name):
    global FEATURE_SET
    FEATURE_SET = name


def feature_dir():
    base = os.path.join(CONFIG["dfl_cache"], "features")
    return base if FEATURE_SET == "v1" else os.path.join(base, FEATURE_SET)


def oof_dir():
    """경기 하나씩 빼고 학습한 모델의 평가 경기 확률(평활화 비교·검증 스크립트가 읽는다)."""
    base = os.path.join(CONFIG["dfl_cache"], "oof")
    return base if FEATURE_SET == "v1" else os.path.join(base, FEATURE_SET)


def model_path(name=None):
    name = name or FEATURE_SET
    filename = "model_all.txt" if name == "v1" else f"model_all_{name}.txt"
    return str(_SRC.parent / "artifacts/deadball/lightgbm/dfl_final/weights" / filename)


def player_feature_names(name=None):
    return CONFIG["player_features"] + CONFIG["feature_sets"][name or FEATURE_SET]


def model_columns(columns):
    """선수 특징과 그 문맥 열만 고른다."""
    keep = set(player_feature_names())
    return [c for c in columns if c.split("@")[0] in keep]


def random_settings(rng):
    ranges = CONFIG["sim_random_ranges"]
    return {name: float(rng.uniform(low, high)) for name, (low, high) in ranges.items()}


def match_features(match_id, kind, seed):
    """kind: 'raw' | 'sim'(기본 설정) | 'rand'(무작위 설정). 라벨 'dead'를 붙인 특징 DataFrame."""
    os.makedirs(feature_dir(), exist_ok=True)
    path = os.path.join(feature_dir(), "{}_{}_{}.parquet".format(match_id, kind, seed))
    if os.path.exists(path):
        return pd.read_parquet(path)
    tracking, _, meta = dfl_load.load_match(match_id)
    match = fx.match_from_dfl(tracking, meta)
    if kind == "sim":
        match = simulate.simulate(match, seed=seed)
    elif kind == "rand":
        match = simulate.simulate(match, seed=seed, **random_settings(np.random.default_rng(seed)))
    frame = fx.compute_features(match)
    frame = frame[["period"] + model_columns(frame.columns)].astype(np.float32)
    frame["dead"] = dfl_labels.dead_label(tracking)
    frame.to_parquet(path)
    return frame


def train(frames):
    import lightgbm as lgb
    stride = CONFIG["train_stride"]
    data = pd.concat([f.iloc[::stride] for f in frames], ignore_index=True)
    columns = model_columns(data.columns)
    params = dict(CONFIG["lgbm_params"])
    rounds = params.pop("n_estimators")
    params.update(objective="binary", bagging_fraction=params.pop("subsample"),
                  bagging_freq=params.pop("subsample_freq"), feature_fraction=params.pop("colsample_bytree"),
                  min_data_in_leaf=params.pop("min_child_samples"), seed=params.pop("random_state"))
    model = lgb.train(params, lgb.Dataset(data[columns], data["dead"]), num_boost_round=rounds)
    return model, columns


def evaluate(model, columns, frame, spells, save_as=None):
    score = model.predict(frame[columns])
    if save_as:
        os.makedirs(oof_dir(), exist_ok=True)
        pd.DataFrame({"period": frame["period"].astype(np.int8), "dead": frame["dead"],
                      "score": score.astype(np.float32)}).to_parquet(os.path.join(oof_dir(), save_as))
    pred = (score >= 0.5).astype(np.int8)
    out = metrics.frame_metrics(frame["dead"].to_numpy(), pred, score)
    seg = metrics.segment_metrics(spells, pred, frame["period"].to_numpy())
    found = seg.pop("found")
    out.update(seg)
    causes = spells.assign(found=found).groupby("cause")["found"].agg(["sum", "count"])
    return out, causes


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--quick", action="store_true", help="J03WPY만 평가 경기로")
    parser.add_argument("--features", default=CONFIG["default_feature_set"], choices=sorted(CONFIG["feature_sets"]))
    parser.add_argument("--save-final", action="store_true", help="끝나면 7경기 전부로 학습한 모델도 저장")
    args = parser.parse_args()
    set_feature_set(args.features)
    folds = ["J03WPY"] if args.quick else dfl_load.MATCHES
    pd.set_option("display.width", 200)

    rows, cause_tables, importance = [], [], []
    for held in folds:
        started = time.time()
        train_ids = [m for m in dfl_load.MATCHES if m != held]
        frames = [match_features(m, "rand", CONFIG["sim_seed"] + i) for i, m in enumerate(dfl_load.MATCHES)
                  if m in train_ids]
        model, columns = train(frames)
        tracking, events, _ = dfl_load.load_match(held)
        spells = dfl_labels.dead_spells(tracking, events)
        for kind in ("sim", "raw"):
            result, causes = evaluate(model, columns, match_features(held, kind, CONFIG["sim_seed"]), spells,
                                      save_as="{}_{}.parquet".format(held, kind))
            rows.append(dict(match=held, test=kind, **result))
            if kind == "sim":
                cause_tables.append(causes)
        importance.append(pd.Series(model.feature_importance("gain"), index=columns))
        print("{} done in {:.0f}s".format(held, time.time() - started), flush=True)

    table = pd.DataFrame(rows)
    shown = ["match", "test", "auc", "frame_acc", "dead_precision", "dead_recall", "dead_f1", "pred_dead_share",
             "true_dead_share", "spell_recall", "segment_precision", "pred_segments", "spells",
             "start_err_med", "start_abs_med", "end_err_med", "end_abs_med"]
    print("\n== leave-one-match-out ==")
    print(table[shown].round(3).to_string(index=False))
    print("\n== mean by test kind ==")
    print(table.groupby("test")[shown[2:]].mean().round(3).T.to_string())

    causes = pd.concat(cause_tables).groupby(level=0).sum()
    causes["recall"] = (causes["sum"] / causes["count"]).round(3)
    print("\n== spell recall by restart (simulated test matches) ==")
    print(causes.sort_values("count", ascending=False).to_string())

    gain = pd.concat(importance, axis=1).mean(axis=1)
    gain = (gain / gain.sum()).sort_values(ascending=False)
    print("\n== feature importance (gain share), top 20 ==")
    print(gain.head(20).round(4).to_string())
    base = gain.groupby(lambda c: c.split("@")[0]).sum().sort_values(ascending=False)
    print("\n== importance by base feature (all offsets summed) ==")
    print(base.round(4).to_string())

    if args.save_final:
        frames = [match_features(m, "rand", CONFIG["sim_seed"] + i) for i, m in enumerate(dfl_load.MATCHES)]
        model, columns = train(frames)
        output = Path(model_path())
        output.parent.mkdir(parents=True, exist_ok=True)
        model.save_model(str(output))
        print()
        print("saved {} ({} features, all 7 DFL matches)".format(model_path(), len(columns)))


if __name__ == '__main__':
    main()
