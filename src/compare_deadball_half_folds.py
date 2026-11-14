"""Compare v2 and the 370-column input using the same six dead-ball corrections.

Both models are fitted on 12 DFL halves and evaluated on two unseen halves
per fold. Run this script explicitly to train and write the scores.
"""

import argparse
import ast
import json
from pathlib import Path

import lightgbm as lgb
import numpy as np
import pandas as pd
from scipy.ndimage import maximum_filter1d, minimum_filter1d, uniform_filter1d
from scipy.special import logit

import compare_smoothing as scoring
import dfl_labels
import dfl_load
import features
import final_dead_balls as fd
import simulate
import smoothing
import train_dfl
from compare_derived_vif import feature_matrix
from config import CONFIG

ROOT = Path(__file__).resolve().parents[1]
NOTEBOOK = ROOT / "model/영상추출/모델/dead_ball/LightGBM_deadball_corrections.ipynb"
FPS = CONFIG["fps"]
PENALTY = CONFIG["smooth_choice"]["penalty"]
BIAS = CONFIG["smooth_choice"]["bias"]
RULES = (1, 2, 3, 4, 5, 6)
MODEL_NAMES = ("v2", "vif50_plus_current_mean_370")
METRICS = ["dead_spell_f1", "alive_spell_f1"]


def load_notebook_rules():
    """Reuse only the rule functions and the fixed competition settings."""
    notebook = json.loads(NOTEBOOK.read_text(encoding="utf-8"))
    required = {
        "halves", "shift", "spread", "extend_from", "peak_indices",
        "movement_features", "aligned_baseline", "viterbi_cost", "decode",
    }
    definitions = {}
    settings = None
    for cell in notebook["cells"]:
        if cell["cell_type"] != "code":
            continue
        tree = ast.parse("".join(cell["source"]))
        for node in tree.body:
            if isinstance(node, ast.FunctionDef) and node.name in required:
                definitions[node.name] = node
            if isinstance(node, ast.Assign) and any(
                isinstance(target, ast.Name) and target.id == "COMPETITION_RULES"
                for target in node.targets
            ):
                if not isinstance(node.value, ast.Call) or not isinstance(node.value.func, ast.Name) or node.value.func.id != "dict" or node.value.args:
                    raise ValueError("Unexpected COMPETITION_RULES syntax in notebook.")
                settings = {keyword.arg: ast.literal_eval(keyword.value) for keyword in node.value.keywords}
    if set(definitions) != required or set(settings or {}) != {"margin", "strength", "bonus"}:
        raise ValueError("Required rule functions or settings are missing from the notebook.")
    namespace = {
        "np": np, "features": features, "fd": fd, "smoothing": smoothing,
        "FPS": FPS, "PENALTY": PENALTY, "BIAS": BIAS, "logit": logit,
        "maximum_filter1d": maximum_filter1d, "minimum_filter1d": minimum_filter1d,
        "uniform_filter1d": uniform_filter1d,
    }
    exec(compile(ast.Module(body=list(definitions.values()), type_ignores=[]),
                 str(NOTEBOOK), "exec"), namespace)
    return namespace["movement_features"], namespace["decode"], settings


def load_folds(path):
    spec = json.loads(path.read_text(encoding="utf-8"))
    halves = {f"{match}_H{half}" for match in dfl_load.MATCHES for half in (1, 2)}
    folds = spec["folds"]
    if spec["split_unit"] != "match_half" or spec["n_splits"] != 7 or len(folds) != 7:
        raise ValueError("Expected the seven-fold half-level split.")
    all_test = []
    for expected, fold in enumerate(folds, start=1):
        train, test = fold["train"], fold["test"]
        if (fold["fold"] != expected or len(train) != 12 or len(test) != 2
                or set(train) & set(test) or set(train) | set(test) != halves):
            raise ValueError(f"Invalid partition in fold {expected}.")
        all_test.extend(test)
    if len(all_test) != len(set(all_test)) or set(all_test) != halves:
        raise ValueError("Each half must appear in exactly one test fold.")
    return spec


def slice_half(match, rows):
    return features.Match(
        players=match.players[rows], team=match.team, ball=match.ball[rows],
        period=match.period[rows], pitch=match.pitch, frame=match.frame[rows],
    )


def matrices(match):
    baseline = features.compute_features(match)
    v2 = baseline[train_dfl.model_columns(baseline.columns)].astype(np.float32)
    expanded = feature_matrix(match).astype(np.float32)
    if v2.shape[1] != 264 or expanded.shape[1] != 370:
        raise ValueError(f"Expected 264 and 370 inputs, got {v2.shape[1]} and {expanded.shape[1]}.")
    return dict(zip(MODEL_NAMES, (v2, expanded)))


def prepare_halves(movement_features, build_matrices=matrices):
    train_dfl.set_feature_set("v2")
    data = {}
    stride = CONFIG["train_stride"]
    seed = CONFIG["sim_seed"]
    for index, match_id in enumerate(dfl_load.MATCHES):
        tracking, _, meta = dfl_load.load_match(match_id)
        original = features.match_from_dfl(tracking, meta)
        truth = dfl_labels.dead_label(tracking)
        train_seed = seed + index
        settings = train_dfl.random_settings(np.random.default_rng(train_seed))
        training = simulate.simulate(original, seed=train_seed, **settings)
        testing = simulate.simulate(original, seed=seed)
        for period in (1, 2):
            rows = np.flatnonzero(original.period == period)
            half_id = f"{match_id}_H{period}"
            if not len(rows) or np.any(np.diff(rows) != 1) or np.any(np.diff(original.frame[rows]) != 1):
                raise ValueError(f"Expected contiguous 25 fps frames for {half_id}.")
            train_half, test_half = slice_half(training, rows), slice_half(testing, rows)
            train_rows = rows % stride == 0
            train = {name: frame.loc[train_rows].reset_index(drop=True)
                     for name, frame in build_matrices(train_half).items()}
            test = build_matrices(test_half)
            geo = fd.ball_signals(test_half)
            crossings = fd.line_crossings(geo)
            signal = movement_features(test_half.players, test_half.team, test_half.ball, test_half.pitch)
            data[half_id] = {
                "train": train, "train_dead": truth[rows][train_rows],
                "test": test, "test_dead": truth[rows],
                "game": {"period": test_half.period, "conditions": {"original": (geo, crossings, [signal])}},
            }
        print(f"Prepared {match_id}: both halves at {FPS} fps", flush=True)
    return data


def fit_model(parts, name, threads):
    columns = parts[0]["train"][name].columns.tolist()
    if any(part["train"][name].columns.tolist() != columns for part in parts):
        raise ValueError(f"Training columns differ for {name}.")
    frame = pd.concat([part["train"][name] for part in parts], ignore_index=True)
    target = np.concatenate([part["train_dead"] for part in parts])
    params = dict(CONFIG["lgbm_params"])
    rounds = params.pop("n_estimators")
    params.update(
        objective="binary", bagging_fraction=params.pop("subsample"),
        bagging_freq=params.pop("subsample_freq"), feature_fraction=params.pop("colsample_bytree"),
        min_data_in_leaf=params.pop("min_child_samples"), seed=params.pop("random_state"),
        num_threads=threads,
    )
    return lgb.train(params, lgb.Dataset(frame, label=target), num_boost_round=rounds)


def spell_f1(truth, state, period):
    """Score each class on its own maximal contiguous runs in one half."""
    pm = scoring.pm
    output = {}
    for name, gt, pred in (
        ("dead_spell_f1", truth, state), ("alive_spell_f1", ~truth, ~state),
    ):
        gt_codes = scoring.codes_of(~gt, period)
        pred_codes = scoring.codes_of(~pred, period)
        output[name] = pm.spell_scores(gt_codes, pred_codes, pm.valid_runs(gt_codes))["mean_f1"]
    return output


def evaluate_folds(spec, data, decode, settings, threads):
    by_half = []
    for fold in spec["folds"]:
        training = [data[half] for half in fold["train"]]
        for name in MODEL_NAMES:
            model = fit_model(training, name, threads)
            for half in fold["test"]:
                part = data[half]
                matrix = part["test"][name]
                if matrix.columns.tolist() != model.feature_name():
                    raise ValueError(f"Fit and evaluation columns differ for {name}.")
                probabilities = model.predict(matrix, num_threads=threads)
                game = part["game"]
                state, _ = decode(probabilities, game, "original", RULES, settings)
                metrics = spell_f1(part["test_dead"].astype(bool), state, game["period"])
                by_half.append({"fold": fold["fold"], "half_id": half, "model": name, **metrics})
        print(f"Scored fold {fold['fold']} of {spec['n_splits']}", flush=True)
    return pd.DataFrame(by_half)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--splits", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--dfl-cache", type=Path, default=Path(CONFIG["dfl_cache"]))
    parser.add_argument("--threads", type=int, default=6)
    args = parser.parse_args()
    if args.threads < 1:
        parser.error("--threads must be positive")
    if args.output.exists() and any(args.output.iterdir()):
        parser.error("--output must be new or empty")

    spec = load_folds(args.splits)
    movement, decode, settings = load_notebook_rules()
    args.dfl_cache.mkdir(parents=True, exist_ok=True)
    CONFIG["dfl_cache"] = str(args.dfl_cache)
    result = evaluate_folds(spec, prepare_halves(movement), decode, settings, args.threads)
    by_fold = result.groupby(["fold", "model"], sort=False)[METRICS].mean().reset_index()
    overall = by_fold.groupby("model", sort=False)[METRICS].mean().reindex(MODEL_NAMES)
    overall.loc["370_minus_v2"] = overall.loc[MODEL_NAMES[1]] - overall.loc[MODEL_NAMES[0]]
    args.output.mkdir(parents=True, exist_ok=True)
    result.to_csv(args.output / "by_half.csv", index=False)
    by_fold.to_csv(args.output / "by_fold.csv", index=False)
    overall.to_csv(args.output / "summary.csv", index_label="model")
    manifest = {
        "splits": spec, "models": dict(zip(MODEL_NAMES, (264, 370))),
        "iou_thresholds": list(scoring.pm.SPELL_IOUS), "rules": list(RULES),
        "rule_settings": settings, "training_stride": CONFIG["train_stride"],
        "evaluation_fps": FPS, "ball_condition": "original",
        "aggregation": "score each half independently, mean of halves per fold, then mean of folds",
        "feature_selection": "fixed 370-column schema, previously selected using all seven DFL matches",
    }
    (args.output / "run_manifest.json").write_text(json.dumps(manifest, indent=2) + "\n", encoding="utf-8")
    print(overall.round(5).to_string())


if __name__ == "__main__":
    main()
