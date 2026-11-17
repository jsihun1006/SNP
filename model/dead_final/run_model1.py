"""Run the Model 1 half-level seven-fold experiment with saved Dead Final 1 initialization."""

import argparse
from dataclasses import asdict
from importlib.metadata import version
from pathlib import Path
import sys

import numpy as np
import pandas as pd
import torch

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[1]
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(ROOT))

import compare_transportmer_folds as model1  # noqa: E402
import dfl_load  # noqa: E402
from metrics import frame_metrics  # noqa: E402
from dead_final_artifacts import half_keys, read_json, sha, warm_weights, write_csv, write_json  # noqa: E402
from transportmer_ifab_rules import (  # noqa: E402
    Rules, STAGES, apply_decoder, build_rules, constraints, observed_line_out_locks,
)

SPLITS = ROOT / "experiments/lightgbm_dfl/dfl_half_7fold.json"
DFL = ROOT / "data/external/dfl"
DEFAULT_OUTPUT = HERE / "model1_7fold"
STAGES_TO_SCORE = ("raw_dead", "six_rules", "line_out", "line_out_viterbi")
SPELL_METRICS = ("dead_spell_f1", "alive_spell_f1", "dead_frame_f1")
RULE_CONFIG = Rules()


def source_files():
    return [
        Path(__file__), Path(model1.__file__), Path(model1.prior.__file__), model1.prior.NOTEBOOK,
        ROOT / "src/dead_final_artifacts.py", ROOT / "src/transportmer_backbone.py",
        ROOT / "src/transportmer_ifab_rules.py",
        ROOT / "src/config.py", ROOT / "src/dfl_load.py", ROOT / "src/dfl_labels.py",
        ROOT / "src/features.py", ROOT / "src/final_dead_balls.py", ROOT / "src/smoothing.py",
        ROOT / "src/simulate.py", ROOT / "src/train_dfl.py", ROOT / "evaluation/possession_metrics.py", SPLITS,
    ]


def initialization_plan(spec, weights_dir):
    record = read_json(weights_dir / "initialization.json")
    training = record["match_splits"]
    initial_weights, provenance = {}, {}
    for fold in spec["folds"]:
        number = fold["fold"]
        test = set(fold["test"])
        validation = fold["train"][-1]
        # Prefer the least evaluation overlap, then the least inner validation overlap.
        def overlap(omitted):
            learned = set(half_keys(training[omitted]))
            return len(learned & test), validation in learned, omitted

        omitted = min(training, key=overlap)
        path = weights_dir / f"{omitted}.pt"
        digest = sha(path)
        if digest != record["weight_sha256_by_holdout"][omitted]:
            raise ValueError(f"Initial weight checksum mismatch: {path}")
        learned = set(half_keys(training[omitted]))
        with torch.random.fork_rng(devices=[]):
            torch.manual_seed(800 + number)
            initial_weights[number] = warm_weights(path, training[omitted])
        provenance[str(number)] = {
            "checkpoint": str(path), "sha256": digest, "checkpoint_excluded_match": omitted,
            "checkpoint_train_matches": training[omitted],
            "evaluation_halves_seen_by_initializer": sorted(test & learned),
            "validation_half": validation, "validation_half_seen_by_initializer": validation in learned,
        }
    return initial_weights, provenance


def ensure_dfl_cache(build):
    model1.CONFIG["dfl_cache"] = str(DFL)
    for match in dfl_load.MATCHES:
        if all(Path(path).is_file() for path in dfl_load.cache_paths(match)):
            continue
        if dfl_load.raw_files(match) is None:
            raise FileNotFoundError(f"Put {match}'s three DFL XML files in {DFL / 'raw'}")
        if build:
            dfl_load.load_match(match)
        else:
            print(f"{match}: local XML available; Parquet will be generated when the experiment runs.")


def score_stages(probability, restored_data, pitch, movement, decode, settings, six_rules):
    game = model1.correction_game(restored_data, pitch, movement)
    xy = restored_data["xy"] * np.asarray(pitch, dtype=np.float32)
    status = restored_data["status"]
    players = np.where((status[:, :-1] == 1)[..., None], xy[:, :-1], np.nan)
    observed_ball = np.where((status[:, -1] == 1)[:, None], xy[:, -1], np.nan)
    teams = restored_data["kind"][:-1]
    evidence = movement(players, teams, observed_ball, pitch)
    marks, _, _, _ = build_rules(observed_ball, players, teams, pitch, six_rules, evidence, {}, RULE_CONFIG)
    marks = observed_line_out_locks(marks)
    lock, _, _ = constraints(marks, len(probability), STAGES["all_rules"])
    locked = lock >= 0
    direct = np.where(locked, lock == 1, six_rules)
    viterbi = (apply_decoder(decode, probability, game, settings, lock, [], rules=model1.prior.RULES)
               if locked.any() else six_rules.copy())
    np.testing.assert_array_equal(viterbi[locked], direct[locked])
    return dict(raw_dead=probability >= .5, six_rules=six_rules, line_out=direct,
                line_out_viterbi=viterbi, observed_line_out_lock=locked,
                viterbi_only_change=direct != viterbi)


def score_half(model, half, field, args, events, movement, decode, settings, **key):
    rows = model1.score_half(model, half, field, args, events, movement, decode, settings, **key)
    stem = f"{key['half_id']}_restored_none"
    path = args.output / "predictions" / stem
    with np.load(path.with_suffix(".npz"), allow_pickle=False) as saved:
        probability, baseline = saved["dead_probability"], saved["after_rules"]
    stages = score_stages(probability, half[field], half["pitch"], movement, decode, settings, baseline)
    number = key["fold"]
    exposed = key["half_id"] in args.provenance[str(number)]["evaluation_halves_seen_by_initializer"]
    frame = pd.DataFrame(dict(
        fold=number, half_id=key["half_id"], frame_id=half["frame"], timestamp_s=half["timestamp"],
        truth_dead=half["dead"].astype(np.int8), dead_probability=probability,
        ball_observed=(half[field]["status"][:, -1] == 1).astype(np.int8),
        initializer_seen_evaluation_half=int(exposed),
        **{name: values.astype(np.int8) for name, values in stages.items()},
    ))
    write_csv(args.output / "comparison/predictions" / f"{key['half_id']}.csv", frame)
    return rows


def summarize(output, spec, provenance):
    comparison = output / "comparison"
    validation = []
    base = pd.read_csv(output / "by_half.csv").set_index(["half_id", "stage"])
    source_rows = []
    for fold in spec["folds"]:
        number = fold["fold"]
        for half in fold["test"]:
            path = comparison / "predictions" / f"{half}.csv"
            data = pd.read_csv(path)
            seen = half in provenance[str(number)]["evaluation_halves_seen_by_initializer"]
            if (data.empty or not data.half_id.eq(half).all() or not data.fold.eq(number).all()
                    or not data.initializer_seen_evaluation_half.eq(int(seen)).all()
                    or not np.all(np.diff(data.frame_id.to_numpy()) == 1)
                    or not data.dead_probability.between(0, 1).all()
                    or not np.isin(data[["truth_dead", *STAGES_TO_SCORE]], [0, 1]).all()):
                raise ValueError(f"Invalid prediction CSV {path}")
            truth = data.truth_dead.to_numpy(bool)
            period = np.ones(len(data), dtype=np.int8)
            for stage in STAGES_TO_SCORE:
                predicted = data[stage].to_numpy(bool)
                scores = model1.prior.spell_f1(truth, predicted, period)
                if stage in {"raw_dead", "six_rules"}:
                    original_stage = "before_rules" if stage == "raw_dead" else "after_rules"
                    np.testing.assert_allclose(list(scores.values()),
                                               base.loc[(half, original_stage), list(scores)].to_numpy(float),
                                               rtol=0, atol=1e-12)
                source_rows.append(dict(fold=number, half_id=half, stage=stage,
                                        initializer_seen_evaluation_half=bool(seen), frames=len(data), **scores,
                                        dead_frame_f1=frame_metrics(truth, predicted)["dead_f1"]))
            validation.append(dict(half_id=half, frames=len(data), output_sha256=sha(path)))
    per_half = pd.DataFrame(source_rows)
    if len(per_half) != 56 or per_half.duplicated(["half_id", "stage"]).any():
        raise ValueError("Expected result rows for four stages on each of fourteen unique halves.")
    per_fold = per_half.groupby(["fold", "stage"], sort=False)[list(SPELL_METRICS)].mean().reset_index()
    summary = per_fold.groupby("stage", sort=False)[list(SPELL_METRICS)].mean().reset_index()
    unseen = (per_half.loc[~per_half.initializer_seen_evaluation_half]
              .groupby("stage", sort=False)[list(SPELL_METRICS)].mean().reset_index())
    for name, tab in (("by_half", per_half), ("by_fold", per_fold), ("summary", summary),
                      ("initializer_unseen_halves", unseen)):
        write_csv(comparison / f"{name}.csv", tab)
    return summary, unseen, validation


def record_parameters(output, spec, provenance, metadata):
    rows = []
    for fold in spec["folds"]:
        number = fold["fold"]
        for task, stem in (("ball", f"ball_{700 + number}_plain_train"),
                           ("dead", f"dead_{800 + number}_plain_restored_train")):
            weight = output / f"{stem}.refit.pt"
            selected = torch.load(output / f"{stem}.tuned.pt", weights_only=True, map_location="cpu")
            timing = output / f"{stem}.timing.json"
            measured = read_json(timing) if timing.exists() else {}
            rows.append(dict(fold=number, task=task, selected_epoch=selected["epoch"],
                             weight_path=weight.name, weight_sha256=sha(weight),
                             selection_seconds=measured.get("selection_seconds"),
                             refit_seconds=measured.get("refit_seconds")))
    timing = pd.DataFrame(rows)
    write_csv(output / "comparison/fit_times.csv", timing)
    write_json(output / "parameters.json", {
        "training": read_json(output / "manifest.json"), "experiment": metadata,
        "initialization_by_fold": provenance, "weights": rows,
    })
    return timing


def write_report(output, summary, unseen, timing, n_seen):
    names = dict(raw_dead="원시 모델", six_rules="6보정", line_out="6보정 + 라인아웃",
                 line_out_viterbi="6보정 + 라인아웃 + 비터비 경로")
    lines = [
        "# TranSPORTmer Model 1 · 하프 7폴드 실측", "",
        f"기존 가중치의 사전 학습과 겹치는 평가 하프는 {n_seen}/14개다. 전체 점수와 초기 가중치에서 보지 않은 하프의 평균을 함께 표시한다.", "",
        "| 단계 | 데드볼 Spell F1 | 인플레이 Spell F1 | 데드볼 프레임 F1 | 초기 학습 미포함 하프의 데드볼 Spell F1 |",
        "|---|---:|---:|---:|---:|",
    ]
    reduced = unseen.set_index("stage").dead_spell_f1
    for row in summary.itertuples(index=False):
        other = f"{reduced[row.stage]:.4f}" if row.stage in reduced else "—"
        lines.append(f"| {names[row.stage]} | {row.dead_spell_f1:.4f} | {row.alive_spell_f1:.4f} | "
                     f"{row.dead_frame_f1:.4f} | {other} |")
    lines.extend(["", "## 단계별 학습 시간", "",
                  "| 단계 | 7폴드 에폭 선택 합계(초) | 7폴드 재학습 합계(초) |",
                  "|---|---:|---:|"])
    for task, group in timing.groupby("task", sort=False):
        totals = [group[key].sum(min_count=len(group)) for key in ("selection_seconds", "refit_seconds")]
        selection, refit = [f"{seconds:.1f}" if pd.notna(seconds) else "—" for seconds in totals]
        lines.append(f"| {task} | {selection} | {refit} |")
    lines.extend(["", "경기별 판정은 [예측 CSV](predictions/), 폴드별 시간은 [fit_times.csv](fit_times.csv)에서 확인할 수 있다."])
    (output / "comparison/RESULTS.md").write_text("\n".join(lines) + "\n", encoding="utf-8")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--check", action="store_true", help="Verify inputs and checkpoints without running training")
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--weights-dir", type=Path, default=HERE / "weights")
    parser.add_argument("--device", choices=("auto", "cuda", "mps", "cpu"), default="auto")
    parser.add_argument("--microbatch", type=int, help="GPU microbatch, default 24 on CUDA, 8 otherwise")
    args = parser.parse_args()
    device = args.device
    if device == "auto":
        device = "cuda" if torch.cuda.is_available() else "mps" if torch.backends.mps.is_available() else "cpu"
    if (device == "cuda" and not torch.cuda.is_available()) or (device == "mps" and not torch.backends.mps.is_available()):
        parser.error(f"{device} is unavailable in this Python environment")
    microbatch = (24 if device == "cuda" else 8) if args.microbatch is None else args.microbatch
    if not 1 <= microbatch <= 24:
        parser.error("--microbatch must be between 1 and 24")
    args.output = args.output.resolve()
    if args.output in {HERE, ROOT, args.weights_dir.resolve()}:
        parser.error("Choose an experiment output directory other than the repository, model or weights directory")

    spec = model1.prior.load_folds(SPLITS)
    weights_dir = args.weights_dir.resolve()
    initializers, provenance = initialization_plan(spec, weights_dir)
    model1.load_rules()
    ensure_dfl_cache(build=not args.check)
    n_seen = sum(len(row["evaluation_halves_seen_by_initializer"]) for row in provenance.values())
    print(f"Device {device} · seven folds · fourteen test halves · overlap with initial checkpoints {n_seen}/14",
          flush=True)
    if args.check:
        for fold, row in provenance.items():
            overlap = ", ".join(row["evaluation_halves_seen_by_initializer"]) or "none"
            print(f"Fold {fold}: {Path(row['checkpoint']).name}, evaluation overlap {overlap}")
        return

    output = args.output
    preparation = output / "experiment.json"
    if output.exists() and any(output.iterdir()) and not preparation.is_file():
        raise ValueError(f"Choose an empty output directory: {output}")
    hashes = [Path(path) for match in dfl_load.MATCHES for path in dfl_load.cache_paths(match)]
    metadata = dict(initialization="saved_dead_final_1_attention", evaluation_seen_by_initialization=n_seen,
                    device=device, microbatch=microbatch,
                    initialization_manifest_sha256=sha(weights_dir / "initialization.json"),
                    initial_checkpoints=provenance, corrections=list(STAGES_TO_SCORE),
                    locked_line_out_config=asdict(RULE_CONFIG),
                    code_sha256={str(path.relative_to(ROOT)): sha(path) for path in source_files()},
                    dfl_sha256={str(path.relative_to(ROOT)): sha(path) for path in hashes},
                    versions={name: version(name) for name in ("torch", "numpy", "pandas", "scipy", "pyarrow", "kloppy")})
    if (output / "manifest.json").exists() and read_json(output / "manifest.json").get("experiment") != metadata:
        raise ValueError(f"Existing output was generated from different files or settings: {output}")
    if preparation.exists() and read_json(preparation) != metadata:
        raise ValueError(f"Existing preparation used different files or settings: {output}")
    (output / "comparison/predictions").mkdir(parents=True, exist_ok=True)
    write_json(preparation, metadata)
    options = argparse.Namespace(
        output=output, inputs=output / "inputs", cache=DFL, splits=SPLITS, device=device,
        motion_input=True, model_1=True, export_csv=True, prepare_dfl=True,
        epochs=25, patience=4, batch=24, microbatch=microbatch, stride=240, learning_rate=.0002,
        holdout=None, validation_match=None, train_loss_only=False, benchmark=False, pilot=False,
        ball_weights=None, ball_selected_epochs=None, record_fit_timing=True, provenance=provenance,
    )
    torch.set_float32_matmul_precision("high")
    model1.run(options, score_evaluation_half=score_half,
               dead_initializers=initializers, experiment_metadata=metadata)
    summary, unseen, validation = summarize(output, spec, provenance)
    timing = record_parameters(output, spec, provenance, metadata)
    write_report(output, summary, unseen, timing, n_seen)
    write_json(output / "comparison/validation.json", dict(passed=True, folds=7, halves=validation,
                                                             initializer_seen_evaluation_halves=n_seen))
    print(summary.to_string(index=False))


if __name__ == "__main__":
    main()
