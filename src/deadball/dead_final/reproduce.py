"""Evaluate the saved tranSPORTmer_D OOF probabilities and package the raw model."""

import argparse
from dataclasses import asdict
from hashlib import file_digest
import json
from pathlib import Path
import shutil
import sys
from tempfile import TemporaryDirectory
from time import perf_counter

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(ROOT / "src"))

import compare_smoothing as scoring
import compare_transportmer_folds as rules
from config import CONFIG
import dfl_load
import features
from metrics import frame_metrics
from rerun_transportmer_d import DEFAULT_OUTPUT, NOTEBOOK
from rescore_transportmer_ball_location import metrics
from rescore_transportmer_ifab import apply_decoder, intervals, observed_line_out_locks
import simulate
from transportmer_ifab_rules import STAGES, Rules, build_rules, constraints


RAW_COLUMNS = [
    "match_id", "half", "frame", "source_frame_id", "timestamp_s", "time_seconds",
    "truth_dead", "dead_probability", "dead_raw", "model_sampled", "visible_players",
    "ball_x_m", "ball_y_m", "ball_observed",
]
VARIANTS = (
    "raw", "original_viterbi", "original_line_out", "six_rules",
    "six_rules_line_out_start", "six_rules_line_out_viterbi",
)
FOLD_FILES = ("model.pt", "predictions.csv", "training_history.json")


def sha256(path):
    with path.open("rb") as stream:
        return file_digest(stream, "sha256").hexdigest()


def decode_half(saved, match, half, movement, decode, settings, cfg):
    keep = match.period == half
    players, ball = match.players[keep], match.ball[keep]
    pitch = np.asarray(match.pitch)
    game = rules.correction_game(rules.arrays((players / pitch, ball / pitch, match.team)), pitch, movement)
    p = saved["dead_probability"].to_numpy()
    base, _ = decode(p, game, "original", rules.prior.RULES, settings)
    evidence = movement(players, match.team, ball, pitch)
    marks, _, _, _ = build_rules(ball, players, match.team, pitch, base, evidence, {}, cfg)
    marks = observed_line_out_locks(marks)
    lock, reason, conflict = constraints(marks, len(saved), STAGES["all_rules"])
    fixed = lock >= 0
    direct = np.where(fixed, lock == 1, base)
    final = apply_decoder(decode, p, game, settings, lock, []) if fixed.any() else base.copy()
    np.testing.assert_array_equal(final[fixed], direct[fixed])
    saved = saved.assign(
        six_rules=base.astype(np.int8), six_rules_line_out_start=direct.astype(np.int8),
        six_rules_line_out_viterbi=final.astype(np.int8), observed_line_out_lock=fixed,
        viterbi_only_change=direct != final,
    )
    labels = np.full(len(saved), "", dtype=object)
    reasons = labels.copy()
    for i, mark in enumerate(marks):
        labels[reason == i], reasons[reason == i] = mark.rule, mark.evidence
    saved["line_out_rule"], saved["line_out_evidence"] = labels, reasons
    outcomes = dict(zip(VARIANTS, (saved.dead_raw.to_numpy(dtype=bool),
                                 saved.dead_viterbi.to_numpy(dtype=bool), saved.after_rules.to_numpy(dtype=bool),
                                 base, direct, final), strict=True))
    context = saved.rename(columns={"source_frame_id": "frame_id"})
    status = np.where(conflict, "equal priority, lock deferred", np.where(fixed, "priority resolved", ""))
    changed = list(intervals(base != direct, context, VARIANTS[-2], base, direct, labels, reasons, status))
    path_labels = np.where(direct != final, "viterbi_context", "")
    path_reasons = np.where(direct != final, "sequence decoded with observed line-out constraints", "")
    changed.extend(intervals(direct != final, context, VARIANTS[-1],
                             direct, final, path_labels, path_reasons, status))
    truth = saved.truth_dead.to_numpy(dtype=bool)
    scores = []
    for variant, state in outcomes.items():
        score = metrics(truth, state, match.period[keep])
        score["dead_frame_f1"] = float(frame_metrics(truth, state)["dead_f1"])
        score["inplay_false_dead_seconds"] = score["inplay_false_dead_frames"] / cfg.fps
        scores.append({"match_id": saved.match_id.iloc[0], "half": half, "variant": variant, **score})
    assert sum(row["frames"] for row in changed) == (base != direct).sum() + (direct != final).sum()
    row_lookup = pd.Series(np.arange(len(context)), index=context.frame_id)
    for row in changed:
        a = row_lookup.loc[row["start_frame"]]
        state = outcomes[row["variant"]][a:a + row["frames"]]
        row["corrected_frames"] = int((state == truth[a:a + len(state)]).sum())
        row["new_error_frames"] = len(state) - row["corrected_frames"]
    return saved, outcomes, scores, changed


def package(source, dest, training, config, movement, decode, settings, cfg):
    for name in ("weights", "predictions", "comparison/predictions", "training/folds", "training/notebooks"):
        (dest / name).mkdir(parents=True, exist_ok=True)
    shutil.copyfile(source / "run_manifest.json", dest / "training" / "run_manifest.json")
    for notebook, key in (
        (NOTEBOOK, "notebook_sha256"),
        (NOTEBOOK.with_name("dfl_transportmer_joint.ipynb"), "joint_notebook_sha256"),
        (NOTEBOOK.with_name("dfl_transportmer.ipynb"), "base_notebook_sha256"),
    ):
        if sha256(notebook) != training[key]:
            raise ValueError(f"Training notebook checksum mismatch at {notebook}")
        shutil.copyfile(notebook, dest / "training" / "notebooks" / notebook.name)
    shutil.copyfile(rules.prior.NOTEBOOK, dest / "training" / "notebooks" / rules.prior.NOTEBOOK.name)
    saved_rows, by_half, all_changes = [], [], []
    started = perf_counter()
    for mid, train_matches in training["match_splits"].items():
        path = source / "matches" / mid
        fold = json.loads((path / "result.json").read_text())
        if set(train_matches) != set(dfl_load.MATCHES) - {mid} or fold["train_matches"] != train_matches:
            raise ValueError(f"Unexpected holdout split for {mid}")
        for name in FOLD_FILES:
            if sha256(path / name) != fold["artifacts_sha256"][name]:
                raise ValueError(f"Artifact checksum mismatch at {path / name}")
        for cache_path in map(Path, dfl_load.cache_paths(mid)):
            key = str(cache_path.relative_to(ROOT))
            if sha256(cache_path) != training["input_sha256"][key]:
                raise ValueError(f"DFL data checksum mismatch at {cache_path}")
        saved = pd.read_csv(path / "predictions.csv", float_precision="round_trip")
        tracking, _, meta = dfl_load.load_match(mid)
        original = features.match_from_dfl(tracking, meta)
        match = simulate.simulate(original, seed=training["simulation_seed"])
        np.testing.assert_array_equal(saved.source_frame_id, match.frame)
        np.testing.assert_array_equal(saved.half, match.period)
        np.testing.assert_array_equal(saved.visible_players, np.isfinite(match.players).all(axis=-1).sum(axis=1))
        np.testing.assert_allclose(saved[["ball_x_m", "ball_y_m"]], match.ball, rtol=0, atol=1e-10, equal_nan=True)
        assert saved.match_id.eq(mid).all() and saved.dead_probability.between(0, 1).all()
        np.testing.assert_array_equal(saved.dead_raw, saved.dead_probability >= config["raw_threshold"])
        saved["timestamp_s"] = tracking.timestamp_s.to_numpy()
        halves, outcomes = [], {key: [] for key in VARIANTS}
        for half in (1, 2):
            result, states, scores, changes = decode_half(
                saved.loc[saved.half == half], match, half, movement, decode, settings, cfg)
            for key, state in states.items():
                outcomes[key].append(state)
            halves.append(result)
            by_half.extend(scores)
            all_changes.extend(changes)
        np.testing.assert_array_equal(saved.truth_dead, rules.prior.dfl_labels.dead_label(tracking))
        frame = saved.rename(columns={"half": "period", "truth_dead": "dead"})
        for variant, states in outcomes.items():
            state = np.concatenate(states)
            stat = scoring.score_one(state, frame)
            spells = rules.prior.spell_f1(saved.truth_dead.to_numpy(dtype=bool), state, match.period)
            np.testing.assert_allclose(stat["alive_spell_f1"], spells["alive_spell_f1"], atol=1e-12)
            saved_rows.append({"match_id": mid, "variant": variant, "dead_frame_f1": stat["dead_f1"], **spells})
        pd.concat(halves).to_csv(dest / "comparison" / "predictions" / f"{mid}.csv", index=False)
        saved[RAW_COLUMNS].to_csv(dest / "predictions" / f"{mid}.csv", index=False)
        shutil.copyfile(path / "model.pt", dest / "weights" / f"{mid}.pt")
        fold_dir = dest / "training" / "folds" / mid
        fold_dir.mkdir()
        for filename in ("result.json", "training_history.json"):
            shutil.copyfile(path / filename, fold_dir / filename)
        print(f"{mid}: {len(saved):,} frames, raw and corrected CSVs exported", flush=True)

    detail = pd.DataFrame(saved_rows)
    half_detail = pd.DataFrame(by_half)
    summary = detail.groupby("variant", sort=False).agg(
        matches=("match_id", "size"), dead_frame_f1=("dead_frame_f1", "mean"),
        dead_spell_f1=("dead_spell_f1", "mean"), alive_spell_f1=("alive_spell_f1", "mean"))
    counts = [key for key in half_detail.columns if key not in ("match_id", "half", "variant")
              and not key.endswith("_f1")]
    summary = summary.join(half_detail.groupby("variant")[counts].sum()).reset_index()
    old = pd.read_csv(source / "model_summary.csv").set_index("stage")
    for variant, stage in zip(VARIANTS[:3], ("raw", "viterbi", "line_out"), strict=True):
        columns = ["dead_frame_f1", "alive_spell_f1"]
        np.testing.assert_allclose(summary.set_index("variant").loc[variant, columns].astype(float),
                                   old.loc[stage, columns].astype(float), rtol=0, atol=1e-12)
    for filename, table in (("summary.csv", summary), ("by_match.csv", detail), ("by_half.csv", half_detail),
                            ("changed_intervals.csv", pd.DataFrame(all_changes))):
        table.to_csv(dest / "comparison" / filename, index=False)
    config["comparison_seconds"] = perf_counter() - started
    config["weight_sha256_by_holdout"] = {mid: sha256(dest / "weights" / f"{mid}.pt")
                                          for mid in training["match_splits"]}
    (dest / "training" / "parameters.json").write_text(json.dumps(config, indent=2) + "\n")
    print(summary.to_string(index=False), flush=True)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--output", type=Path, default=Path(__file__).resolve().parent)
    args = parser.parse_args()
    source, output = args.source.resolve(), args.output.resolve()
    names = ("predictions", "weights", "comparison", "training")
    if any((output / name).exists() for name in names):
        parser.error("Choose a fresh output directory for the generated package files.")
    CONFIG["dfl_cache"] = str(ROOT / "data/external/dfl")
    training = json.loads((source / "run_manifest.json").read_text())
    if set(training["match_splits"]) != set(dfl_load.MATCHES) or training["viterbi"] != CONFIG["smooth_choice"]:
        raise ValueError("Expected the original seven match-level folds and Viterbi settings.")
    movement, decode, settings = rules.load_rules()
    cfg = Rules()
    config = {
        "model_name": "Dead Final 1",
        "selected_model": "tranSPORTmer_D raw",
        "raw_threshold": 0.5,
        "raw_postprocessing": [],
        "source_run": str(source),
        "training": training["config"],
        "training_device": training["device"],
        "training_versions": training["versions"],
        "simulation_seed": training["simulation_seed"],
        "simulation_config": {key: value for key, value in CONFIG.items() if key.startswith("sim_")},
        "comparison_variants": VARIANTS,
        "geometry_input": "25 fps observed DFL ball, seed-matched evaluation player simulation",
        "decoder_settings": {**settings, **training["viterbi"], "rule_ids": rules.prior.RULES},
        "rules_notebook_sha256": sha256(rules.prior.NOTEBOOK),
        "comparison_source_sha256": {key: sha256(ROOT / key) for key in (
            "src/deadball/dead_final/reproduce.py", "src/deadball/compare_transportmer_folds.py",
            "src/deadball/rescore_transportmer_ifab.py", "src/deadball/transportmer_ifab_rules.py", "src/simulate.py",
        )},
        "line_out_locks": {"rules": ["line_out", "pre_crossing_play"], "thresholds": asdict(cfg)},
        "iou_thresholds": scoring.pm.SPELL_IOUS,
    }
    output.mkdir(parents=True, exist_ok=True)
    with TemporaryDirectory(dir=output, prefix=".build-") as staging:
        package(source, Path(staging), training, config, movement, decode, settings, cfg)
        for name in names:
            (Path(staging) / name).rename(output / name)


if __name__ == "__main__":
    main()
