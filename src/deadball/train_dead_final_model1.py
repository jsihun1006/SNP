"""Train and package the ball-restored dead-ball model with a Dead Final 1 attention initialization."""

from pathlib import Path
import sys

_SRC = Path(__file__).resolve().parents[1]
sys.path[:0] = [str(_SRC), str(_SRC.parent)]

import argparse
from collections import defaultdict
from importlib.metadata import version
import json
import shutil
from time import perf_counter

import numpy as np
import pandas as pd
import torch

import compare_transportmer_folds as transport
from config import CONFIG
from dead_final_artifacts import half_keys, read_json, sha, torch_weights, warm_weights, write_csv, write_json
import dfl_labels
import dfl_load
from metrics import frame_metrics
from transportmer_backbone import FFN, FPS, HEADS, WIDTH, WINDOW, STEP, TranSPORTmer

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
from games import asset_paths  # noqa: E402

ACTIVE = ROOT / "artifacts/deadball/transportmer/dead_final_1"
DOCS = ROOT / "experiments/deadball/transportmer/dead_final_model1"
DFL_CACHE = ROOT / "data/external/dfl"
DEFAULT_WORK = ROOT / "artifacts/deadball/transportmer/dead_final_model1_warm_v1"
DEFAULT_MEASUREMENT = ROOT / "artifacts/deadball/transportmer/dead_final_model1_warm_measure_J03WR9"
REUSABLE_FOLD = "J03WR9"
REUSE_DIR = ROOT / "artifacts/deadball/transportmer/transportmer_model1_holdout_J03WR9_train_loss"
INCUMBENT = "six_rules_line_out_start"
SHARED_SETTINGS = dict(motion_input=True, learning_rate=0.0002, ball_epochs=10, dead_epochs=8,
                       patience=3, batch=24, microbatch=8, stride=240)


def input_manifest(baseline, device):
    CONFIG["dfl_cache"] = str(DFL_CACHE)
    matches = list(dfl_load.MATCHES)
    training = read_json(baseline / "training/run_manifest.json")
    splits = {held: [match for match in matches if match != held] for held in matches}
    sources = {}
    for held, members in splits.items():
        if members != training["match_splits"][held]:
            raise ValueError(f"Unexpected split for {held}")
        weight = baseline / "weights" / f"{held}.pt"
        warm_weights(weight, members)
        sources[held] = sha(weight)
    dfl_hashes = {}
    for match in matches:
        for filename in dfl_load.cache_paths(match):
            path = Path(filename)
            key = str(path.relative_to(ROOT))
            dfl_hashes[key] = sha(path)
            if dfl_hashes[key] != training["input_sha256"][key]:
                raise ValueError(f"DFL cache differs from the incumbent evaluation: {key}")
    ref = read_json(REUSE_DIR / "manifest.json")
    ball = REUSE_DIR / "ball_restorer.pt"
    if (ref["splits"]["test_match"] != REUSABLE_FOLD
            or ref["splits"]["train_matches"] != splits[REUSABLE_FOLD]
            or sha(ball) != ref["ball_checkpoint"]["sha256"]
            or (ref["width"], ref["heads"], ref["ffn"], ref["tokens"]) != (WIDTH, HEADS, FFN, WINDOW)
            or ref["motion_input"] is not True
            or ref["ball_missing_target"] != transport.BALL_MISSING
            or (ref["ball_mask_mean_gap_seconds"], ref["ball_mask_seed_train"], ref["ball_mask_seed_test"]) != (6, 1200, 1400)):
        raise ValueError("Existing J03WR9 restorer does not match the requested split and model")
    TranSPORTmer("ball", motion_input=True).load_state_dict(torch_weights(ball), strict=True)
    _, _, rule_settings = transport.load_rules()
    code = [Path(__file__), Path(transport.__file__), ROOT / "src/transportmer_backbone.py",
            ROOT / "src/deadball/final_dead_balls.py", ROOT / "src/competition_data.py",
            ROOT / "src/smoothing.py", ROOT / "src/simulate.py", ROOT / "src/features.py",
            transport.prior.NOTEBOOK, ROOT / "evaluation/possession_metrics.py"]
    competition_inputs = asset_paths(source="tactical")
    comp_hash = {key: sha(Path(competition_inputs[key])) for key in ("atd_meta", "atd_raw")}
    baseline_scores = baseline / "comparison/by_match.csv"
    return dict(
        mode="model1_warm_attention", splits=splits, dfl_sha256=dfl_hashes, warm_sha256=sources,
        reuse_ball_sha256=sha(ball), baseline_scores_sha256=sha(baseline_scores), device=device,
        model=dict(width=WIDTH, heads=HEADS, ffn=FFN, window=WINDOW, step=STEP, fps=FPS),
        training=SHARED_SETTINGS, simulation_seed=CONFIG["sim_seed"],
        correction=rule_settings, competition_atd_sha256=comp_hash,
        score="mean of the seven match-level dead-ball Spell F1 values",
        source_sha256={str(path.relative_to(ROOT)): sha(path) for path in code},
        versions={name: version(name) for name in ("torch", "numpy", "pandas", "lightgbm", "pyarrow", "scipy")},
    )


def fit_models(parts, members, held, baseline, args):
    keys = half_keys(members)
    args.output.mkdir(parents=True, exist_ok=True)
    restorer = args.output / "ball_restorer.pt"
    classifier = args.output / "dead_best.pt"
    if not restorer.exists():
        if held == REUSABLE_FOLD:
            transport.save_stage(restorer, torch_weights(REUSE_DIR / "ball_restorer.pt"))
        else:
            ball, _ = transport.fit_epochs(parts, keys, "train", "ball", args, 701, False, args.ball_epochs)
            transport.save_stage(restorer, {key: value.cpu() for key, value in ball.state_dict().items()})
            del ball
            transport.release_device_cache(args)

    ball = TranSPORTmer("ball", motion_input=True).to(args.device)
    ball.load_state_dict(torch_weights(restorer), strict=True)
    dead = TranSPORTmer("dead", motion_input=True).to(args.device)
    if not classifier.exists() or not (args.output / "training.json").exists():
        for key in keys:
            parts[key]["restored_train"] = transport.complete(ball, parts[key]["train"], args)
        initial_holdout = held if held is not None else REUSABLE_FOLD
        initial_members = [match for match in dfl_load.MATCHES if match != initial_holdout]
        torch.manual_seed(801)
        initial_weights = warm_weights(baseline / "weights" / f"{initial_holdout}.pt", initial_members)
        dead, _ = transport.fit_epochs(parts, keys, "restored_train", "dead", args, 801, False,
                                       args.dead_epochs, select_training_loss=True,
                                       initial_weights=initial_weights)
        for key in keys:
            del parts[key]["restored_train"]
    else:
        dead.load_state_dict(torch_weights(classifier), strict=True)
    return ball, dead


def evaluate_fold(parts, held, members, baseline, args, decoder):
    done = args.output / "result.json"
    if done.is_file():
        result = read_json(done)
        for path, digest in result["sha256"].items():
            if sha(args.output / path) != digest:
                raise ValueError(f"Saved fold artifact changed: {held}/{path}")
        return result["scores"]

    start = perf_counter()
    ball, dead = fit_models(parts, members, held, baseline, args)
    move, decode, settings = decoder
    scores = []
    for key in half_keys([held]):
        part = parts[key]
        part["restored_test"] = transport.complete(ball, part["test"], args)
        scores.extend(transport.score_half(dead, part, "restored_test", args, None, move, decode, settings,
                                           half_id=key, condition="restored", event_mode="none"))
        del part["restored_test"]
    files = ["ball_restorer.pt", "dead_best.pt", "training.json"]
    files += [str(path.relative_to(args.output)) for path in sorted((args.output / "predictions").iterdir())]
    write_json(done, dict(scores=scores, wall_s=perf_counter() - start,
                          sha256={path: sha(args.output / path) for path in files}))
    return scores


def match_scores(folds):
    rows = []
    for held, folder in folds.items():
        states = defaultdict(list)
        for half, key in enumerate(half_keys([held]), 1):
            path = folder / "predictions" / f"{key}_restored_none.npz"
            with np.load(path, allow_pickle=False) as data:
                for column in ("truth", "before_rules", "after_rules"):
                    values = data[column]
                    if values.ndim != 1 or not np.isin(values, [0, 1]).all():
                        raise ValueError(f"Invalid {column} in {path}")
                    states[column].append(values.astype(bool))
                states["period"].append(np.full(len(data["truth"]), half))
        truth, period = (np.concatenate(states[key]) for key in ("truth", "period"))
        for stage in ("before_rules", "after_rules"):
            state = np.concatenate(states[stage])
            rows.append(dict(match_id=held, stage=stage, **transport.prior.spell_f1(truth, state, period),
                             dead_frame_f1=frame_metrics(truth, state)["dead_f1"]))
    return pd.DataFrame(rows)


def export_competition(out, weights, args, decoder, manifest):
    from competition_data import load_competition_match

    current_inputs = asset_paths(source="tactical")
    for name, digest in manifest["competition_atd_sha256"].items():
        if sha(Path(current_inputs[name])) != digest:
            raise ValueError(f"BAR–ATH input changed: {name}")
    weight_files = {"ball": weights / "ball_restorer.pt", "dead": weights / "dead_best.pt"}
    models = {}
    for task, path in weight_files.items():
        model = TranSPORTmer(task, motion_input=True).to(args.device)
        model.load_state_dict(torch_weights(path), strict=True)
        models[task] = model
    ball, dead = models["ball"], models["dead"]
    match, _ = load_competition_match()
    move, decode, settings = decoder
    frame = match.frame
    after, raw = [np.full(len(frame), -1, dtype=np.int8) for _ in range(2)]
    probability = np.full(len(frame), np.nan)
    observed = np.isfinite(match.ball).all(axis=1)
    for half in (1, 2):
        keep = match.period == half
        data = transport.arrays((match.players[keep] / match.pitch, match.ball[keep] / match.pitch, match.team))
        data = transport.complete(ball, data, args)
        prob = transport.predict(dead, data, args)[:, 0]
        game = transport.correction_game(data, match.pitch, move)
        state, _ = decode(prob, game, "original", transport.prior.RULES, settings)
        probability[keep], raw[keep], after[keep] = prob, prob >= 0.5, state
    valid = match.period > 0
    if (not np.array_equal(frame, np.arange(199_319))
            or not np.isin(after[valid], (0, 1)).all()
            or not np.isfinite(probability[valid]).all()):
        raise ValueError("Invalid BAR–ATH frame alignment or prediction")
    label = np.where(valid, np.where(after == 1, "D", "ALIVE"), "X")
    status = pd.DataFrame(dict(frame=frame, time_seconds=frame / FPS, half=match.period,
                               graded=valid.astype(int), ball_observed=observed.astype(int)))
    result = status.assign(dead_probability=probability, raw_dead=pd.array(raw, dtype="Int8"),
                           dead=pd.array(after, dtype="Int8"), state=label)
    result.loc[~valid, ["raw_dead", "dead"]] = pd.NA
    out.mkdir(parents=True, exist_ok=True)
    write_csv(out / "frame_status.csv", status)
    write_csv(out / "features.csv", result)
    write_json(out / "run_manifest.json", dict(input_sha256=manifest["competition_atd_sha256"],
                                               weight_sha256={task: sha(path) for task, path in weight_files.items()},
                                               output_sha256={name: sha(out / name) for name in ("frame_status.csv", "features.csv")},
                                               fps=FPS, device=args.device, frame_origin=0,
                                               coordinates="tactical ATD, metres internally", correction=settings))
    write_json(out / "validation.json", dict(total_frames=len(frame), processed_frames=int(valid.sum()),
                                             missing_frames=0, duplicated_frames=0, fps=FPS))


def package(work, manifest, folds, half_rows, comparison, summary, production):
    folder = work / "package"
    folder.mkdir(exist_ok=True)
    for name in ("weights", "predictions/dfl", "comparison", "training"):
        (folder / name).mkdir(parents=True, exist_ok=True)
    for match, fold in folds.items():
        for name in ("ball_restorer.pt", "dead_best.pt"):
            shutil.copyfile(fold / name, folder / "weights" / f"{match}_{name}")
        for source in (fold / "predictions").glob("*.csv"):
            shutil.copyfile(source, folder / "predictions/dfl" / source.name)
        shutil.copyfile(fold / "training.json", folder / "training" / f"{match}.json")
    for name in ("ball_restorer.pt", "dead_best.pt"):
        shutil.copyfile(production / name, folder / "weights" / name)
    shutil.copyfile(production / "training.json", folder / "training/production.json")
    write_csv(folder / "comparison/by_half.csv", pd.DataFrame(half_rows))
    write_csv(folder / "comparison/by_match.csv", comparison)
    write_csv(folder / "comparison/summary.csv", summary)
    shutil.copyfile(DOCS / "EXPERIMENT_PLAN.md", folder / "EXPERIMENT_PLAN.md")
    shutil.copyfile(DOCS / "MODEL_README.md", folder / "README.md")
    (folder / ".gitignore").write_text("/weights/\n/predictions/\n/training/\n/comparison/\n", encoding="utf-8")
    write_json(folder / "training/run_manifest.json", manifest)
    hashes = {str(path.relative_to(folder)): sha(path) for path in sorted(folder.rglob("*"))
              if path.is_file() and path != folder / "validation.json"}
    write_json(folder / "validation.json", dict(passed=True, folds=len(folds), dfl_halves=len(half_rows) // 2,
                                                artifact_sha256=hashes))
    return folder


def train(work, baseline, dev):
    manifest = input_manifest(baseline, dev)
    work.mkdir(parents=True, exist_ok=True)
    log = work / "experiment_manifest.json"
    if log.exists() and read_json(log) != manifest:
        raise ValueError(f"Inputs or settings changed. Choose a different --work-dir than {work}")
    write_json(log, manifest)
    args = argparse.Namespace(**SHARED_SETTINGS, device=dev, model_1=True, benchmark=False, export_csv=True)
    decoder = transport.load_rules()
    parts = transport.load_halves(work / "inputs", DFL_CACHE, prepare_dfl=True)
    folds, half_rows = {}, []
    for held in [REUSABLE_FOLD, *[match for match in dfl_load.MATCHES if match != REUSABLE_FOLD]]:
        args.output = folds[held] = work / "folds" / held
        half_rows.extend(evaluate_fold(parts, held, manifest["splits"][held], baseline, args, decoder))
        transport.release_device_cache(args)
        print(f"{held}: {pd.DataFrame(half_rows[-4:]).groupby('stage').dead_spell_f1.mean().to_dict()}", flush=True)

    rows = match_scores(folds)
    incumbent = pd.read_csv(baseline / "comparison/by_match.csv")
    incumbent = incumbent.loc[incumbent.variant.eq(INCUMBENT)].rename(columns={"variant": "stage"})
    if sorted(incumbent.match_id) != sorted(dfl_load.MATCHES) or len(rows) != 14:
        raise ValueError("Incomplete match-level comparison")
    combined = pd.concat([rows, incumbent], ignore_index=True)
    scores = ["dead_spell_f1", "alive_spell_f1", "dead_frame_f1"]
    summary = combined.groupby("stage", sort=False)[scores].mean().reset_index()
    write_csv(work / "by_half.csv", pd.DataFrame(half_rows))
    write_csv(work / "by_match.csv", combined)
    write_csv(work / "summary.csv", summary)
    print(summary.to_string(index=False), flush=True)
    values = summary.set_index("stage").dead_spell_f1
    qualifies = bool(values["after_rules"] > values[INCUMBENT])
    write_json(work / "selection.json", dict(promotable=qualifies, metric="dead_spell_f1",
                                             incumbent=values[INCUMBENT], candidate=values["after_rules"]))
    if qualifies:
        args.output = work / "production"
        fit_models(parts, list(dfl_load.MATCHES), None, baseline, args)
        transport.release_device_cache(args)
        export_competition(work / "package/predictions/BAR-ATH", args.output, args, decoder, manifest)
        package(work, manifest, folds, half_rows, combined, summary, args.output)
        print(f"Validated package: {work / 'package'}", flush=True)


def measure(work, baseline, dev):
    """Measure the one shared match using the saved cold-start run as the reference."""
    folded = input_manifest(baseline, dev)
    reference = read_json(REUSE_DIR / "manifest.json")
    for key, expected in (("max_epochs", SHARED_SETTINGS["dead_epochs"]),
                          ("dead_learning_rate", SHARED_SETTINGS["learning_rate"]),
                          ("batch", SHARED_SETTINGS["batch"]),
                          ("microbatch", SHARED_SETTINGS["microbatch"]),
                          ("window_stride_25fps", SHARED_SETTINGS["stride"])):
        if reference[key] != expected:
            raise ValueError(f"Cold-start comparison uses a different {key}")
    manifest = dict(mode="measure_only", held=REUSABLE_FOLD, inputs=folded,
                    reference={name: sha(REUSE_DIR / name) for name in ("manifest.json", "summary.csv", "training.json")})
    work.mkdir(parents=True, exist_ok=True)
    if (work / "manifest.json").exists() and read_json(work / "manifest.json") != manifest:
        raise ValueError(f"Inputs or settings changed. Use a different --work-dir than {work}")
    write_json(work / "manifest.json", manifest)
    parts = transport.load_halves(work / "inputs", DFL_CACHE, prepare_dfl=True)
    args = argparse.Namespace(**SHARED_SETTINGS, device=dev, model_1=True, benchmark=False, export_csv=True,
                              output=work / REUSABLE_FOLD)
    decoder = transport.load_rules()
    rows = evaluate_fold(parts, REUSABLE_FOLD, folded["splits"][REUSABLE_FOLD], baseline, args, decoder)
    warm = pd.DataFrame(rows).groupby("stage")[["dead_spell_f1", "alive_spell_f1"]].mean()
    cold = pd.read_csv(REUSE_DIR / "summary.csv").set_index("stage")[["dead_spell_f1", "alive_spell_f1"]]
    scores = pd.concat({"cold_start": cold, "warm_attention": warm}, names=["initialization"])
    reference_training = read_json(REUSE_DIR / "training.json")
    warm_training = read_json(args.output / "training.json")
    times = []
    goal = reference_training["best_train_bce"]
    for label, record in (("cold_start", reference_training), ("warm_attention", warm_training)):
        history = record["history"]
        hit = next((row for row in history if row["train_bce"] <= goal), None)
        times.append(dict(initialization=label, epochs=len(history), selected_epoch=record["selected_epoch"],
                          training_s=history[-1]["elapsed_s"], best_train_bce=record["best_train_bce"],
                          epoch_at_cold_best_train_bce=hit["epoch"] if hit else None,
                          seconds_at_cold_best_train_bce=hit["elapsed_s"] if hit else None))
    write_csv(work / "measurement.csv", scores.reset_index())
    write_csv(work / "timing.csv", pd.DataFrame(times))
    print(scores.to_string(), flush=True)
    print(pd.DataFrame(times).to_string(index=False), flush=True)


def promote(work):
    candidate = work / "package"
    selection = read_json(work / "selection.json")
    if not selection["promotable"]:
        raise ValueError("Candidate did not improve the incumbent dead-ball Spell F1")
    check = read_json(candidate / "validation.json")
    if not check["passed"] or (check["folds"], check["dfl_halves"]) != (7, 14):
        raise ValueError("Candidate package is incomplete")
    for name, digest in check["artifact_sha256"].items():
        if sha(candidate / name) != digest:
            raise ValueError(f"Candidate file changed after validation: {name}")
    scores = pd.read_csv(candidate / "comparison/summary.csv").set_index("stage").dead_spell_f1
    if not scores["after_rules"] > scores[INCUMBENT]:
        raise ValueError("Candidate package did not improve the incumbent dead-ball Spell F1")
    expected = read_json(candidate / "training/run_manifest.json")
    if sha(ACTIVE / "comparison/by_match.csv") != expected["baseline_scores_sha256"]:
        raise ValueError("The active model changed since evaluation")
    for match, digest in expected["warm_sha256"].items():
        if sha(ACTIVE / "weights" / f"{match}.pt") != digest:
            raise ValueError(f"The active initialization checkpoint changed: {match}")
    archive = work / "previous_dead_final"
    if archive.exists():
        raise FileExistsError(archive)
    ACTIVE.rename(archive)
    try:
        candidate.rename(ACTIVE)
    except OSError:
        archive.rename(ACTIVE)
        raise
    print(f"Installed: {ACTIVE}\nPrevious version: {archive}", flush=True)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("action", choices=("check", "measure", "train", "predict", "promote"))
    parser.add_argument("--work-dir", type=Path)
    parser.add_argument("--baseline", type=Path, default=ACTIVE, help="Original Dead Final 1 folder")
    parser.add_argument("--model-dir", type=Path, default=ACTIVE, help="For predict, a trained Model 1 package")
    parser.add_argument("--device", choices=("mps", "cuda", "cpu"), default="mps")
    opts = parser.parse_args()
    work = (opts.work_dir or (DEFAULT_MEASUREMENT if opts.action == "measure" else DEFAULT_WORK)).resolve()
    work_roots = (ROOT / "artifacts/deadball", ROOT / "data")
    if not any(work != base and work.is_relative_to(base) for base in work_roots):
        parser.error("--work-dir must be an experiment folder below artifacts/deadball or data")
    if opts.action == "promote":
        promote(work)
        return
    if ((opts.device == "mps" and not torch.backends.mps.is_available())
            or (opts.device == "cuda" and not torch.cuda.is_available())):
        parser.error(f"{opts.device} is unavailable in this Python environment")
    if opts.action == "predict":
        package_manifest = read_json(opts.model_dir / "training/run_manifest.json")
        if package_manifest["mode"] != "model1_warm_attention":
            raise ValueError("Predict requires a trained Model 1 package")
        out = work / "reproduced_BAR-ATH"
        if out.exists():
            raise FileExistsError(f"Use a fresh --work-dir: {out}")
        args = argparse.Namespace(**SHARED_SETTINGS, device=opts.device)
        decoder = transport.load_rules()
        if decoder[-1] != package_manifest["correction"]:
            raise ValueError("The rule settings changed since training")
        validation = read_json(opts.model_dir / "validation.json")
        for name, digest in package_manifest["source_sha256"].items():
            if sha(ROOT / name) != digest:
                raise ValueError(f"Prediction source changed since the run: {name}")
        for name in ("ball_restorer.pt", "dead_best.pt"):
            key = f"weights/{name}"
            if sha(opts.model_dir / key) != validation["artifact_sha256"][key]:
                raise ValueError(f"Production weights changed: {name}")
        export_competition(out, opts.model_dir / "weights", args, decoder, package_manifest)
    elif opts.action == "check":
        manifest = input_manifest(opts.baseline, opts.device)
        print(f"Preflight passed: {len(manifest['splits'])} folds, "
              f"{len(manifest['dfl_sha256'])} DFL caches, 7 warm initializers, "
              f"1 reusable restorer and 2 BAR–ATH ATD files. Device: {opts.device}.")
    elif opts.action == "measure":
        measure(work, opts.baseline, opts.device)
    else:
        train(work, opts.baseline, opts.device)


if __name__ == "__main__":
    main()
