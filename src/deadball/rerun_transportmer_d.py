"""Repeat the original tranSPORTmer_D match holdouts and keep reusable artifacts."""

from pathlib import Path
import sys

_SRC = Path(__file__).resolve().parents[1]
sys.path[:0] = [str(_SRC), str(_SRC.parent)]

import argparse
import json
import math
import os
from time import perf_counter


ROOT = Path(__file__).resolve().parents[2]
NOTEBOOK = ROOT / "notebooks/deadball/tranSPORTmer_D.ipynb"
REFERENCE = ROOT / "artifacts/deadball/transportmer/transportmer_d_20261112_01"
DEFAULT_OUTPUT = ROOT / "artifacts/deadball/transportmer/transportmer_d_9067_rerun"


def load_experiment():
    os.chdir(ROOT)
    scope = {"__name__": "__transportmer_d__"}
    cells = {cell["id"]: cell for cell in json.loads(NOTEBOOK.read_text())["cells"]}
    for number in (2, 4):
        cell = cells[f"transportmer-d-{number}"]
        exec(compile("".join(cell["source"]), f"{NOTEBOOK}:{number}", "exec"), scope)
    return scope


def save_checkpoint(torch, path, payload):
    temp = path.with_suffix(".tmp")
    torch.save(payload, temp)
    temp.replace(path)


def export_csv(frame, path):
    temp = path.with_suffix(".tmp")
    frame.to_csv(temp, index=False)
    temp.replace(path)


def fit_fold(exp, output, held, train_matches, datasets, splits):
    torch, np = exp["torch"], exp["np"]
    model, opt = exp["new_joint"]()
    batch, patience = exp["BATCH"], exp["PATIENCE"]
    train = {key: np.concatenate([datasets[mid][key][splits[mid][0]] for mid in train_matches])
             for key in ("x", "valid", "target")}
    latest = output / "latest.pt"
    best_path = output / "model.pt"
    best_score, best_epoch, best, stale = -math.inf, 0, None, 0
    history = []
    identity = {"held_match": held, "train_matches": train_matches,
                "config": exp["config"], "architecture": "tranSPORTmer_D"}
    if latest.is_file():
        state = torch.load(latest, map_location="cpu", weights_only=True)
        if state["identity"] != identity:
            raise ValueError(f"Checkpoint split or configuration differs at {latest}")
        model.load_state_dict(state["model_state_dict"])
        opt.load_state_dict(state["optimizer_state_dict"])
        history = state["history"]
        best_score, best_epoch = state["best_score"], state["best_epoch"]
        best, stale = state["best_model_state_dict"], state["stale"]
        torch.random.set_rng_state(state["torch_rng_state"])
        if exp["DEVICE"].type == "mps":
            torch.mps.set_rng_state(state["device_rng_state"])
        print(f"{held} resuming after epoch {len(history)}", flush=True)
        del state

    print(f"{held} holdout | {len(train['x']):,} training windows | "
          f"{math.ceil(len(train['x']) / batch)} batches/epoch", flush=True)
    for epoch in range(len(history) + 1, exp["EPOCHS"] + 1):
        if stale >= patience:
            break
        started = perf_counter()
        model.train()
        order = np.random.default_rng(exp["SEED"] + epoch).permutation(len(train["x"]))
        losses = [exp["train_state_batch"](model, opt, train, order[i:i+batch])
                  for i in range(0, len(order), batch)]
        bce = float(torch.stack(losses).mean().item())
        if not math.isfinite(bce):
            raise ValueError(f"Nonfinite BCE at {held} epoch {epoch}")
        validation = None
        if epoch >= exp["MIN_EPOCHS"]:
            validation = exp["inner_f1"](model, datasets, splits, train_matches)
            if validation > best_score + exp["MIN_DELTA"]:
                best_score, best_epoch, best, stale = validation, epoch, exp["cpu_weights"](model), 0
            else:
                stale += 1
        seconds = perf_counter() - started
        history.append({"epoch": epoch, "bce": bce, "validation_alive_spell_f1": validation, "seconds": seconds})
        save_checkpoint(torch, latest, {
            "identity": identity, "model_state_dict": exp["cpu_weights"](model),
            "optimizer_state_dict": opt.state_dict(), "history": history, "best_score": best_score,
            "best_epoch": best_epoch, "best_model_state_dict": best, "stale": stale,
            "torch_rng_state": torch.random.get_rng_state(),
            "device_rng_state": torch.mps.get_rng_state() if exp["DEVICE"].type == "mps" else None,
        })
        print(f"{held} epoch {epoch:02}/{exp['EPOCHS']} | {seconds:.1f}s | BCE {bce:.4f} | "
              f"inner alive Spell F1 {validation} | best {best_epoch} | patience {stale}/{patience}", flush=True)

    if best is None:
        raise RuntimeError(f"No internally selected model for {held}")
    save_checkpoint(torch, best_path, {
        **identity, "model_state_dict": best, "best_epoch": best_epoch,
        "best_inner_alive_spell_f1": best_score, "stopped_epoch": len(history),
    })
    exp["save_json"](history, output / "training_history.json")
    return {"match": held, "train_matches": train_matches, "best_epoch": best_epoch,
            "stopped_epoch": len(history), "best_inner_alive_spell_f1": best_score,
            "training_s": sum(epoch["seconds"] for epoch in history)}


def evaluate(exp, output, held, evaluation):
    torch, np, pd = exp["torch"], exp["np"], exp["pd"]
    weights = torch.load(output / "model.pt", map_location="cpu", weights_only=True)
    model, _ = exp["new_joint"]()
    model.load_state_dict(weights["model_state_dict"], strict=True)
    probability, sampled = exp["predict"](exp["StateOutput"](model.eval()).eval(), evaluation)
    scores, viterbi, forced = exp["score_joint"](evaluation, probability)
    crossing = np.zeros(len(probability), dtype=np.int8)
    crossing[evaluation["crossings"]] = 1
    frame = np.arange(len(probability))
    predictions = pd.DataFrame({
        "match_id": held, "half": evaluation["period"], "frame": frame,
        "source_frame_id": evaluation["source_frame"], "time_seconds": frame / exp["FPS"],
        "truth_dead": evaluation["label"], "dead_probability": probability,
        "dead_raw": (probability >= 0.5).astype(np.int8), "dead_viterbi": viterbi, "after_rules": forced,
        "model_sampled": sampled.astype(np.int8), "visible_players": evaluation["visible_players"],
        "ball_x_m": evaluation["ball_m"][:, 0], "ball_y_m": evaluation["ball_m"][:, 1],
        "ball_observed": evaluation["geometry"]["seen"].astype(np.int8), "line_crossing": crossing,
    })
    reference = pd.read_csv(REFERENCE / "matches" / held / "features.csv")
    np.testing.assert_array_equal(predictions["source_frame_id"], reference["source_frame_id"])
    np.testing.assert_array_equal(predictions["frame"], reference["frame"])
    export_csv(predictions, output / "predictions.csv")
    return {"frames": len(frame), "scores": scores, "pitch_m": list(map(float, evaluation["pitch"])),
            "mean_abs_probability_delta_vs_original": float(np.mean(np.abs(probability - reference["dead_score"]))),
            "artifacts_sha256": {name: exp["digest"](output / name)
                                 for name in ("model.pt", "predictions.csv", "training_history.json")}}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--match", help="Run only this holdout. Later runs can fill in the other matches.")
    parser.add_argument("--device", choices=("mps", "cpu"), default="mps")
    args = parser.parse_args()
    output = args.output.resolve()
    if output == REFERENCE or REFERENCE in output.parents:
        parser.error("Choose a new directory outside the historical reference run")
    exp = load_experiment()
    torch, pd = exp["torch"], exp["pd"]
    if args.device == "mps" and not torch.backends.mps.is_available():
        parser.error("MPS is unavailable in this process")
    exp["DEVICE"] = torch.device(args.device)
    matches = exp["MATCHES"]
    if args.match and args.match not in matches:
        parser.error(f"Unknown match {args.match}. Available: {', '.join(matches)}")
    original = json.loads((REFERENCE / "run_manifest.json").read_text())
    digests = {"notebook_sha256": exp["digest"](NOTEBOOK),
               "joint_notebook_sha256": exp["digest"](exp["joint"]),
               "base_notebook_sha256": exp["digest"](exp["original"])}
    if any(original[name] != sha for name, sha in digests.items()):
        raise ValueError("Notebook files differ from the original tranSPORTmer_D run")
    started = perf_counter()
    datasets, evaluations, splits, inputs = {}, {}, {}, {}
    for match in matches:
        for path in map(Path, exp["dfl_load"].cache_paths(match)):
            inputs[str(path.relative_to(ROOT))] = exp["digest"](path)
        datasets[match], evaluations[match] = exp["prepare_joint"](match)
        splits[match] = exp["internal_split"](datasets[match])
        for key in ("ball", "ball_target"):
            del datasets[match][key]
    if inputs != original["input_sha256"]:
        raise ValueError("DFL cached inputs differ from the historical run")
    preparation = perf_counter() - started
    parameters = exp["verify_state_only"]()
    manifest = {
        "reference": str(REFERENCE.relative_to(ROOT)), **digests, "input_sha256": inputs,
        "config": {**exp["config"], "model_parameters": parameters, "ball_loss_weight": 0},
        "device": args.device, "amp": exp["AMP"], "fps": exp["FPS"],
        "viterbi": exp["VITERBI"], "simulation_seed": exp["CONFIG"]["sim_seed"],
        "match_splits": {m: [train for train in matches if train != m] for m in matches},
        "versions": {name: exp["version"](name) for name in original["versions"]},
        "runtime_source_sha256": {name: exp["digest"](ROOT / "src" / name) for name in (
            "rerun_transportmer_d.py", "config.py", "compare_smoothing.py", "dfl_labels.py", "dfl_load.py",
            "features.py", "final_dead_balls.py", "metrics.py", "simulate.py", "smoothing.py", "train_dfl.py")},
    }
    output.mkdir(parents=True, exist_ok=True)
    manifest_path = output / "run_manifest.json"
    if manifest_path.is_file() and json.loads(manifest_path.read_text()) != manifest:
        raise ValueError(f"Run configuration changed. Use a new output directory: {output}")
    exp["save_json"](manifest, manifest_path)
    print(f"{args.device} {manifest['versions']} | {parameters:,} parameters | preparation {preparation:.1f}s",
          flush=True)

    for held in (matches if args.match is None else (args.match,)):
        dest = output / "matches" / held
        dest.mkdir(parents=True, exist_ok=True)
        result_path = dest / "result.json"
        if not result_path.is_file():
            started = perf_counter()
            row = fit_fold(exp, dest, held, manifest["match_splits"][held], datasets, splits)
            row.update(evaluate(exp, dest, held, evaluations[held]))
            row["session_s"] = perf_counter() - started
            exp["save_json"](row, result_path)
            print(f"{held} saved | best {row['best_epoch']}/{row['stopped_epoch']} | "
                  f"line-out dead frame F1 {row['scores'][-1]['dead_f1']:.6f} | "
                  f"train {row['training_s'] / 60:.1f} min", flush=True)
        results = [json.loads(path.read_text()) for path in sorted(output.glob("matches/*/result.json"))]
        detail = pd.DataFrame([{"match": row["match"], "best_epoch": row["best_epoch"],
                                "stopped_epoch": row["stopped_epoch"], **scores}
                               for row in results for scores in row["scores"]])
        summary = detail.groupby("stage", sort=False).agg(
            matches=("match", "size"), dead_frame_f1=("dead_f1", "mean"),
            dead_recall=("dead_recall", "mean"), dead_precision=("dead_precision", "mean"),
            frame_acc=("frame_acc", "mean"), alive_spell_f1=("alive_spell_f1", "mean")).reset_index()
        export_csv(detail, output / "per_match.csv")
        export_csv(summary, output / "model_summary.csv")
    print(summary.to_string(index=False), flush=True)


if __name__ == "__main__":
    main()
