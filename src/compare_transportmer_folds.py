"""Seven-fold DFL evaluation of the four TranSPORTmer dead-ball conditions."""
import argparse
from contextlib import nullcontext
import hashlib
import json
import math
from pathlib import Path
from time import perf_counter

import numpy as np
import pandas as pd
import torch
from torch import nn

import compare_deadball_event_folds as events
import compare_deadball_half_folds as prior
import dfl_load
import features
import final_dead_balls as fd
import simulate
import train_dfl
from config import CONFIG

from transportmer_backbone import FFN, FPS, HEADS, STEP, WIDTH, WINDOW, Attention, TranSPORTmer

BALL_MISSING = 0.407
EVENT_MODES = ("none", "masked")
ROOT = Path(__file__).resolve().parents[1]


def arrays(half):
    pos, ball, kinds = half
    xy = np.concatenate((pos, ball[:, None]), axis=1).astype(np.float32)
    seen = np.isfinite(xy).all(axis=-1)
    return {"xy": np.where(seen[..., None], xy, 0), "status": seen.astype(np.uint8),
            "kind": np.r_[kinds, 2].astype(np.int64)}


def sample_ball(ball, rng):
    return simulate.ball_dropout(ball, rng, BALL_MISSING, 6.0).astype(np.float32)


def load_halves(directory, cache, prepare_dfl=False):
    CONFIG["dfl_cache"] = str(cache)
    parts = {}
    for index, match in enumerate(dfl_load.MATCHES):
        track, e, metadata = dfl_load.load_match(match)
        raw = features.match_from_dfl(track, metadata)
        test = simulate.simulate(raw, seed=CONFIG["sim_seed"])
        pitch = np.asarray(raw.pitch, dtype=np.float32)
        dead = prior.dfl_labels.dead_label(track)
        path = directory / f"{match}.npz"
        if prepare_dfl and not path.is_file():
            # Same DFL training simulator and seeds as the original input exporter.
            seed = CONFIG["sim_seed"] + index
            settings = train_dfl.random_settings(np.random.default_rng(seed))
            training = simulate.simulate(raw, seed=seed, **settings)
            observed = np.isfinite(training.players).all(axis=-1)
            pending = path.with_suffix(".tmp")
            directory.mkdir(parents=True, exist_ok=True)
            with pending.open("wb") as file:
                np.savez_compressed(file, positions=np.where(observed[..., None], training.players / pitch, 0),
                                    observed=observed, frame=raw.frame, period=raw.period,
                                    team=raw.team, dead=dead)
            pending.replace(path)
        with np.load(path) as npz:
            assert np.array_equal(npz["frame"], raw.frame) and np.array_equal(npz["period"], raw.period)
            assert np.array_equal(npz["team"], raw.team) and np.array_equal(npz["dead"], dead)
            for h in (1, 2):
                mask = raw.period == h
                truth = (raw.ball[mask] / pitch).astype(np.float32)
                train_ball = sample_ball(truth, np.random.default_rng(1200 + index*2 + h))
                eval_ball = sample_ball(truth, np.random.default_rng(1400 + index*2 + h))
                train_pos = np.where(npz["observed"][mask, :, None], npz["positions"][mask], np.nan)
                part = {
                    "train": arrays((train_pos, train_ball, raw.team)),
                    "test": arrays((test.players[mask] / pitch, eval_ball, raw.team)),
                    "truth_ball": truth, "dead": dead[mask],
                    "frame": raw.frame[mask], "timestamp": track["timestamp_s"].to_numpy()[mask],
                    "event": events.label_events(track["timestamp_s"].to_numpy()[mask], e.loc[e.period_id == h]),
                    "pitch": raw.pitch,
                }
                part["mask_fraction"] = float((~np.isfinite(eval_ball).all(axis=1)).mean())
                parts[f"{match}_H{h}"] = part
        print(f"DATA {match} ready", flush=True)
    return parts


def starts(length, stride, phase=0):
    last = length - 1 - STEP * (WINDOW - 1)
    return np.unique(np.r_[0, np.arange(phase, max(0, last), stride), max(0, last)])


def batch_tensor(part, frame_starts, device):
    indexes = frame_starts[:, None] + STEP * np.arange(WINDOW)[None]
    return (torch.as_tensor(part["xy"][indexes], device=device),
            torch.as_tensor(part["status"][indexes], device=device, dtype=torch.long),
            torch.as_tensor(part["kind"], device=device), indexes)


def encode_events(labels):
    labels = np.asarray(labels)
    onehot = labels[..., None] == np.arange(1, 7)
    return np.concatenate([onehot, np.ones_like(labels[..., None])], axis=-1).astype(np.float32)


def precision(args):
    if args.device == "cuda":
        return torch.autocast("cuda", dtype=torch.bfloat16)
    return nullcontext()


def release_device_cache(args):
    if args.device == "mps":
        torch.mps.synchronize()
        torch.mps.empty_cache()


def fit_epochs(parts, keys, field, task, args, seed, conditioned, epochs, validation=None,
               select_training_loss=False, initial_weights=None):
    torch.manual_seed(seed)
    rng = np.random.default_rng(seed)
    model = TranSPORTmer(task, use_events=conditioned, motion_input=args.motion_input).to(args.device)
    if initial_weights is not None:
        model.load_state_dict(initial_weights, strict=True)
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.learning_rate if task == "dead" else 0.001, eps=1e-4)
    begin = perf_counter()
    best_epoch, best_loss = 0, float("inf")
    history = []
    for epoch in range(epochs):
        model.train()
        measure = []
        measure_weights = []
        jobs = []
        for key in keys:
            indices = starts(len(parts[key]["dead"]), args.stride, phase=int(rng.integers(args.stride)))
            rng.shuffle(indices)
            jobs.extend((key, indices[i:i+args.batch]) for i in range(0, len(indices), args.batch))
        rng.shuffle(jobs)
        for step_number, (key, indices) in enumerate(jobs, 1):
            part = parts[key]
            ix = indices[:, None] + STEP * np.arange(WINDOW)[None]
            denominator = ix.size
            if task == "ball":
                hidden = (part[field]["status"][ix, -1] == 0) & np.isfinite(part["truth_ball"][ix]).all(axis=-1)
                denominator = max(1, int(hidden.sum()))
            optimizer.zero_grad(set_to_none=True)
            batch_loss = []
            for start in range(0, len(indices), args.microbatch):
                selected = indices[start:start+args.microbatch]
                xy, status, kind, ix = batch_tensor(part[field], selected, args.device)
                inputs = torch.as_tensor(encode_events(part["event"][ix]), device=args.device) if conditioned else None
                with precision(args):
                    logits = model(xy, status, kind, inputs).float()
                    if task == "ball":
                        truth = torch.as_tensor(part["truth_ball"][ix], device=args.device)
                        active = (status[:, :, -1] == 0) & torch.isfinite(truth).all(-1)
                        distance = torch.linalg.vector_norm((logits - truth.nan_to_num(0)) *
                                                            torch.tensor(part["pitch"], device=args.device), dim=-1)
                        loss = (distance * active).sum()
                    else:
                        truth = torch.as_tensor(part["dead"][ix], device=args.device, dtype=torch.float32)
                        loss = nn.functional.binary_cross_entropy_with_logits(
                            logits.squeeze(-1), truth, reduction="sum")
                    loss = loss / denominator
                loss.backward()
                batch_loss.append(loss.detach())
            nn.utils.clip_grad_norm_(model.parameters(), 5)
            optimizer.step()
            measure.append(torch.stack(batch_loss).sum())
            measure_weights.append(denominator)
            if args.benchmark and step_number == 8:
                if args.device == "mps":
                    torch.mps.synchronize()
                if args.device == "cuda":
                    torch.cuda.synchronize()
                print(f"BENCHMARK {task}: 8/{len(jobs)} batches in {perf_counter()-begin:.1f}s", flush=True)
                return model, 1
        batch_values = torch.stack(measure)
        if select_training_loss:
            weights = batch_values.new_tensor(measure_weights)
            training_loss = (batch_values * weights).sum().div(weights.sum()).item()
        else:
            training_loss = batch_values.mean().item()
        release_device_cache(args)
        if not np.isfinite(training_loss):
            raise ValueError(f"Non-finite {task} loss")
        if select_training_loss:
            if training_loss < best_loss - 1e-4:
                best_epoch, best_loss = epoch + 1, training_loss
                best_weights = {key: value.detach().cpu().clone() for key, value in model.state_dict().items()}
            history.append({"epoch": epoch + 1, "train_bce": training_loss, "elapsed_s": perf_counter() - begin})
            print(f"TRAIN {field} {task} epoch={epoch+1}/{epochs} train_bce={training_loss:.5f} "
                  f"best={best_epoch} elapsed={perf_counter()-begin:.0f}s", flush=True)
            if epoch + 1 - best_epoch >= args.patience:
                break
        elif validation is not None:
            errors = []
            for key in validation:
                check = parts[key]
                masked = np.zeros((len(check["dead"]), 7), dtype=np.float32) if conditioned else None
                prediction = predict(model, check[field], args, masked)
                if task == "ball":
                    valid = (check[field]["status"][:, -1] == 0) & np.isfinite(check["truth_ball"]).all(axis=1)
                    error = (prediction[valid] - check["truth_ball"][valid]) * np.asarray(check["pitch"])
                    errors.append(np.linalg.norm(error, axis=1))
                else:
                    prob = np.clip(prediction[:, 0].astype(float), 1e-7, 1 - 1e-7)
                    truth = check["dead"].astype(bool)
                    errors.append(-np.where(truth, np.log(prob), np.log1p(-prob)))
            errors = np.concatenate(errors)
            if not len(errors):
                raise ValueError("No masked ground truth in the ball validation data.")
            loss = float(errors.mean())
            if loss < best_loss - 1e-4:
                best_epoch, best_loss = epoch + 1, loss
                if task == "ball":
                    best_weights = {key: value.detach().cpu().clone() for key, value in model.state_dict().items()}
            print(f"VALID {field} {task} conditioned={conditioned} epoch={epoch+1}/{epochs} "
                  f"loss={loss:.5f} best={best_epoch} elapsed={perf_counter()-begin:.0f}s", flush=True)
            if epoch + 1 - best_epoch >= args.patience:
                break
        else:
            print(f"REFIT {field} {task} conditioned={conditioned} epoch={epoch+1}/{epochs} "
                  f"loss={training_loss:.5f} secs={perf_counter()-begin:.0f}", flush=True)
    if select_training_loss or (validation is not None and task == "ball"):
        model.load_state_dict(best_weights)
    if select_training_loss:
        save_stage(args.output / "dead_best.pt", best_weights)
        path = args.output / "training.json"
        pending = path.with_suffix(".tmp")
        pending.write_text(json.dumps({"selected_epoch": best_epoch, "best_train_bce": best_loss,
                                       "history": history}, indent=2) + "\n", encoding="utf-8")
        pending.replace(path)
    return model, best_epoch


def fit(parts, keys, field, task, args, seed, conditioned=False, tuning_field=None, on_tuned=None,
        initial_weights=None):
    # Fixed by the shuffled split; no evaluation half participates in stopping.
    validation = args.validation_halves or [keys[-1]]
    inner_train = [key for key in keys if key not in validation]
    stage = args.output / f"{task}_{seed}_{'event' if conditioned else 'plain'}_{field}"
    selection = stage.with_suffix(".tuned.pt")
    refit = stage.with_suffix(".refit.pt")
    timing_path = stage.with_suffix(".timing.json")
    record_timing = getattr(args, "record_fit_timing", False)
    timing = json.loads(timing_path.read_text()) if record_timing and timing_path.is_file() else {}

    def record(phase, began):
        if record_timing:
            timing.update(selected_epoch=best_epoch)
            timing[f"{phase}_seconds"] = perf_counter() - began
            pending = timing_path.with_suffix(".tmp")
            pending.write_text(json.dumps(timing, indent=2) + "\n", encoding="utf-8")
            pending.replace(timing_path)

    if args.model_1 and selection.exists():
        saved = torch.load(selection, map_location="cpu", weights_only=True)
        best_epoch = saved["epoch"]
        model = TranSPORTmer(task, use_events=conditioned, motion_input=args.motion_input).to(args.device)
        model.load_state_dict(saved["weights"])
    else:
        fixed = args.ball_selected_epochs if task == "ball" else None
        began = perf_counter()
        model, best_epoch = fit_epochs(
            parts, inner_train, tuning_field or field, task, args, seed, conditioned,
            fixed or args.epochs, validation=None if fixed else validation,
            initial_weights=initial_weights)
        best_epoch = fixed or best_epoch
        if args.model_1:
            save_stage(selection, {"epoch": best_epoch, "weights": model.state_dict()})
        record("selection", began)
    if on_tuned is not None:
        on_tuned(model)
    del model
    release_device_cache(args)
    print(f"STOP {field} {task} conditioned={conditioned} selected_epoch={best_epoch} "
          f"validation={validation}", flush=True)
    if args.model_1 and refit.exists():
        model = TranSPORTmer(task, use_events=conditioned, motion_input=args.motion_input).to(args.device)
        model.load_state_dict(torch.load(refit, map_location="cpu", weights_only=True))
    else:
        began = perf_counter()
        model, _ = fit_epochs(parts, keys, field, task, args, seed, conditioned, best_epoch,
                              initial_weights=initial_weights)
        if args.model_1:
            save_stage(refit, model.state_dict())
        record("refit", began)
    return model


def save_stage(path, values):
    pending = path.with_suffix(".tmp")
    torch.save(values, pending)
    pending.replace(path)


@torch.inference_mode()
def predict(model, data, args, events_input=None):
    model.eval()
    n = len(data["xy"])
    width = {"ball": 2, "dead": 1}[model.task]
    total = np.zeros((n, width), dtype=np.float32)
    counts = np.zeros(n, dtype=np.int32)
    # Keep each sampled sequence on the same 6.25 Hz grid; cover the tail.
    last = max(0, (n-1)//STEP - WINDOW + 1)
    boundaries = np.unique(np.r_[np.arange(0, last, WINDOW//2), last]) * STEP
    for begin in range(0, len(boundaries), args.microbatch):
        xy, status, kind, ix = batch_tensor(data, boundaries[begin:begin+args.microbatch], args.device)
        e = torch.as_tensor(events_input[ix], device=args.device) if events_input is not None else None
        with precision(args):
            y = model(xy, status, kind, e).float()
            if model.task == "dead":
                y = y.sigmoid()
        np.add.at(total, ix.ravel(), y.cpu().numpy().reshape(-1, width))
        np.add.at(counts, ix.ravel(), 1)
    known = np.flatnonzero(counts)
    if not np.isfinite(total).all():
        raise ValueError(f"Non-finite {model.task} prediction.")
    grid = np.arange(n)
    return np.column_stack([np.interp(grid, known, total[known, i]/counts[known]) for i in range(width)]).astype(np.float32)


def complete(model, data, args):
    recovered = predict(model, data, args)
    if not np.isfinite(recovered).all():
        raise ValueError("Non-finite ball restoration.")
    filled = {key: value.copy() for key, value in data.items()}
    missing = data["status"][:, -1] == 0
    filled["xy"][missing, -1] = recovered[missing]
    filled["status"][missing, -1] = 2
    assert np.array_equal(filled["xy"][~missing, -1], data["xy"][~missing, -1])
    return filled


def local_evidence(crossings, observed_geo, observed_restart):
    inside, outside = observed_geo["inside"], observed_geo["outside"]
    return np.array([bool(inside[max(0,c-FPS):c].any() and
                          (outside[c:c+FPS].any() or observed_restart[c:c+3*FPS].any()))
                     for c in crossings], dtype=bool)


def correction_game(data, pitch, movement):
    coords = data["xy"] * np.asarray(pitch, dtype=np.float32)
    players = np.where((data["status"][:, :-1] == 1)[..., None], coords[:, :-1], np.nan)
    teams, ball_source = data["kind"][:-1], data["status"][:, -1]
    period = np.ones(len(players), dtype=np.int8)
    observed_ball = np.where((ball_source == 1)[:, None], coords[:, -1], np.nan)
    input_ball = np.where((ball_source > 0)[:, None], coords[:, -1], np.nan)
    geometry = fd.ball_signals(features.Match(players, teams, input_ball, period, pitch))
    observed_geo = fd.ball_signals(features.Match(players, teams, observed_ball, period, pitch))
    observed_cross = fd.line_crossings(observed_geo)
    signals = movement(players, teams, input_ball, pitch)
    signals["weak_cross"] = np.array([], dtype=int)
    if (ball_source == 2).any():
        observed_signals = movement(players, teams, observed_ball, pitch)
        restart = observed_signals["restart"]
        inferred_cross = np.setdiff1d(fd.line_crossings(geometry), observed_cross)
        approved = local_evidence(inferred_cross, observed_geo, restart)
        signals["weak_cross"] = inferred_cross[~approved]
        crossings = np.union1d(observed_cross, inferred_cross[approved])
        for margin in (0.0, .25, .50):
            name = f"cross_{margin}"
            ids = signals[name]
            approved = local_evidence(ids, observed_geo, restart) | np.isin(ids, observed_signals[name])
            signals[name] = ids[approved]
            signals["weak_cross"] = np.union1d(signals["weak_cross"], ids[~approved])
    else:
        crossings = observed_cross
    # Preserve the original geometry for verified line-outs. The completed
    # geometry is used for supported crossings when the crossing was missing.
    for key, vector in observed_geo.items():
        if key != "speed":
            geometry[key] = np.where(ball_source == 1, vector, geometry[key])
    return {"period": period, "conditions": {"original": (geometry, crossings, [signals])}}


def load_rules():
    import inspect
    movement, decode, settings = prior.load_notebook_rules()
    # The function was compiled from the notebook, so read its definition there.
    notebook = json.loads(prior.NOTEBOOK.read_text(encoding="utf-8"))
    import ast
    node = next(n for c in notebook["cells"] if c["cell_type"] == "code"
                for n in ast.parse("".join(c["source"])).body
                if isinstance(n, ast.FunctionDef) and n.name == "decode")
    source = ast.unparse(node)
    needle = "        enter = np.full(n, PENALTY)"
    if source.count(needle) != 1:
        raise ValueError("Unable to inject weak crossing cost into notebook Viterbi.")
    source = source.replace(needle, needle + '\n        enter[sig["weak_cross"]] *= 0.7')
    namespace = dict(decode.__globals__)
    exec(compile(source, str(prior.NOTEBOOK), "exec"), namespace)
    return movement, namespace["decode"], settings


def report(output, rows):
    frame = pd.DataFrame(rows)
    keys = ["condition", "event_mode", "stage"]
    fold = frame.groupby([*keys, "fold"], sort=False)[prior.METRICS].mean().reset_index()
    summary = fold.groupby(keys, sort=False)[prior.METRICS].mean().reset_index()
    for name, table in [("by_half", frame), ("by_fold", fold), ("summary", summary)]:
        path = output / f"{name}.csv"
        tmp = path.with_suffix(".tmp")
        table.to_csv(tmp, index=False)
        tmp.replace(path)
    return summary


def score_half(model, half, field, args, event_values, movement, decode, settings, **key):
    data = half[field]
    game = correction_game(data, half["pitch"], movement)
    score = predict(model, data, args, event_values)[:, 0].astype(np.float64)
    truth = half["dead"].astype(bool)
    state = score >= .5
    decoded, _ = decode(score, game, "original", prior.RULES, settings)
    output = args.output / "predictions"
    output.mkdir(exist_ok=True)
    name = f'{key["half_id"]}_{key["condition"]}_{key["event_mode"]}.npz'
    path = output / name
    pending = path.with_suffix(".tmp")
    with pending.open("wb") as file:
        np.savez_compressed(file, dead_probability=score, truth=truth,
                            before_rules=state, after_rules=decoded)
    pending.replace(path)
    if args.model_1 or args.export_csv:
        match, period = key["half_id"].split("_H")
        csv = path.with_suffix(".csv")
        pending = csv.with_suffix(".csv.tmp")
        pd.DataFrame({
            "match_id": match, "half": int(period), "frame_id": half["frame"],
            "timestamp_s": half["timestamp"], "truth_dead": truth.astype(np.int8),
            "dead_probability": score, "before_rules": state.astype(np.int8),
            "after_rules": decoded.astype(np.int8),
        }).to_csv(pending, index=False)
        pending.replace(csv)
    changes = np.count_nonzero(state[:-1] != state[1:])
    print(f'DIAG {name} pstd={score.std():.5f} dead_share={state.mean():.4f} '
          f'prediction_runs={changes + 1}', flush=True)

    return [
        dict(**key, stage=stage, **prior.spell_f1(truth, labels, game["period"]))
        for stage, labels in (("before_rules", state), ("after_rules", decoded))
    ]


def run(args, *, score_evaluation_half=score_half, dead_initializers=None, experiment_metadata=None):
    args.validation_halves = []
    if args.holdout:
        training = [match for match in dfl_load.MATCHES if match != args.holdout]
        spec = {
            "split_unit": "match", "train_matches": training, "test_match": args.holdout,
            "folds": [{
                "fold": 1,
                "train": [f"{match}_H{half}" for match in training for half in (1, 2)],
                "test": [f"{args.holdout}_H{half}" for half in (1, 2)],
            }],
        }
        if not args.train_loss_only:
            validation = args.validation_match or ("J03WQQ" if args.holdout != "J03WQQ" else training[-1])
            args.validation_halves = [f"{validation}_H{half}" for half in (1, 2)]
            spec["inner_validation_match"] = validation
    else:
        spec = prior.load_folds(args.splits)
    movement, decode, settings = load_rules()
    parts = load_halves(args.inputs, args.cache, args.prepare_dfl)
    if args.benchmark:
        train = spec["folds"][0]["train"][:-1]
        for task in ("ball", "dead"):
            fit_epochs(parts, train, "train", task, args, 701 if task == "ball" else 801, False, 1)
        return
    args.output.mkdir(parents=True, exist_ok=True)
    if args.pilot:
        fold = spec["folds"][0]
        model = fit(parts, fold["train"], "train", "dead", args, 801)
        rows = [score_half(model, parts[key], "test", args, None, movement, decode, settings,
                           fold=1, half_id=key, condition="observed", event_mode="none")
                for key in fold["test"]]
        summary = report(args.output, [row for pair in rows for row in pair])
        print(summary.to_string(index=False), flush=True)
        return
    conditions = (("restored", "restored_train", "restored_test"),) if args.model_1 else (
        ("observed", "train", "test"), ("restored", "restored_train", "restored_test"))
    modes = ("none",) if args.model_1 else EVENT_MODES
    expected_rows = 2 * 2 * len(conditions) * len(modes)
    manifest = {
        "splits": spec, "width": WIDTH, "heads": HEADS, "ffn": FFN, "tokens": WINDOW,
        "hz": FPS/STEP, "max_epochs": args.epochs, "dead_learning_rate": args.learning_rate, "window_stride_25fps": args.stride, "batch": args.batch,
        "early_stopping": {"patience": args.patience, "min_delta": 1e-4,
                           "validation": "one training match" if args.holdout else "last training half",
                           "metrics": {"ball": "masked ADE", "dead": "binary log loss with event masking"},
                           "refit_all_training_halves": True,
                           "inner_restore_uses_only_inner_training": True},
        "motion_input": args.motion_input,
        "ball_missing_target": BALL_MISSING, "ball_mask_mean_gap_seconds": 6,
        "ball_mask_seed_train": 1200, "ball_mask_seed_test": 1400,
        "event_window_seconds_each_side": events.WINDOW_SECONDS,
        "event_modes": list(modes),
        "evaluation_fps": FPS, "rules": list(prior.RULES), "rule_settings": settings,
        "uncorroborated_restored_crossings_entry_cost_multiplier": 0.7,
        "weak_crossing_guard": "observed inside within 1s before plus observed outside within 1s after or observed restart within 3s",
    }
    if args.model_1 or args.device != "cuda" or args.export_csv or args.microbatch != args.batch:
        manifest.update(conditions=[c[0] for c in conditions], device=args.device,
                        torch_version=str(torch.__version__), prediction_csv=args.model_1 or args.export_csv,
                        microbatch=args.microbatch, preselected_ball_epochs=args.ball_selected_epochs)
    if args.train_loss_only:
        source = json.loads(args.ball_weights.with_name("manifest.json").read_text(encoding="utf-8"))
        for key in ("train_matches", "test_match"):
            if source["splits"][key] != spec[key]:
                raise ValueError(f"Ball checkpoint uses a different {key}")
        for key in ("motion_input", "width", "heads", "ffn", "tokens", "ball_missing_target",
                    "ball_mask_mean_gap_seconds", "ball_mask_seed_train", "ball_mask_seed_test"):
            if source[key] != manifest[key]:
                raise ValueError(f"Ball checkpoint uses a different {key}")
        manifest["ball_checkpoint"] = {
            "path": str(args.ball_weights.resolve()),
            "sha256": hashlib.sha256(args.ball_weights.read_bytes()).hexdigest(),
        }
        manifest["early_stopping"] = {"patience": args.patience, "min_delta": 1e-4,
                                      "metrics": {"dead": "training BCE"}}
    if experiment_metadata is not None:
        manifest["experiment"] = experiment_metadata
    file = args.output / "manifest.json"
    if file.exists() and json.loads(file.read_text(encoding="utf-8")) != manifest:
        raise ValueError("Existing output was produced with different settings.")
    file.write_text(json.dumps(manifest, indent=2) + "\n", encoding="utf-8")
    rows = []
    for fold in spec["folds"]:
        number, train, test = fold["fold"], fold["train"], fold["test"]
        done = args.output / f"fold_{number}.json"
        if done.exists():
            old = json.loads(done.read_text(encoding="utf-8"))
            if len(old) != expected_rows:
                raise ValueError(f"Incomplete scores in {done}")
            if args.model_1 or args.export_csv:
                for half in test:
                    for condition, _, _ in conditions:
                        for mode in modes:
                            csv = args.output / "predictions" / f"{half}_{condition}_{mode}.csv"
                            if not csv.is_file():
                                raise FileNotFoundError(f"Completed fold {number} is missing {csv}")
            rows.extend(old)
            print(f"FOLD {number} already complete", flush=True)
            continue
        start = perf_counter()
        def restore_inner(imputer):
            for key in train:
                parts[key]["restored_tuning"] = complete(imputer, parts[key]["train"], args)

        if args.train_loss_only:
            imputer = TranSPORTmer("ball", motion_input=args.motion_input).to(args.device)
            imputer.load_state_dict(torch.load(args.ball_weights, map_location="cpu", weights_only=True))
            print(f"REUSE ball checkpoint {args.ball_weights.name}", flush=True)
        else:
            imputer = fit(parts, train, "train", "ball", args, 700 + number, on_tuned=restore_inner)
        for key in train:
            parts[key]["restored_train"] = complete(imputer, parts[key]["train"], args)
        for key in test:
            parts[key]["restored_test"] = complete(imputer, parts[key]["test"], args)
        del imputer
        release_device_cache(args)
        fold_rows = []
        for condition, training, testing in conditions:
            seed = 800 + number
            tuning = "restored_tuning" if condition == "restored" else training
            if args.train_loss_only:
                baseline, _ = fit_epochs(parts, train, training, "dead", args, seed, False,
                                         args.epochs, select_training_loss=True)
            else:
                initializer = None if dead_initializers is None else dead_initializers[number]
                baseline = fit(parts, train, training, "dead", args, seed, tuning_field=tuning,
                               initial_weights=initializer)
            models = {"none": baseline}
            if "masked" in modes:
                models["masked"] = fit(parts, train, training, "dead", args, seed,
                                       conditioned=True, tuning_field=tuning)
            for key in test:
                half = parts[key]
                data = half[testing]
                masked = np.zeros((len(data["xy"]), 7), dtype=np.float32)
                for mode, model in models.items():
                    fold_rows.extend(score_evaluation_half(
                        model, half, testing, args, masked if mode == "masked" else None,
                        movement, decode, settings,
                        fold=number, half_id=key, condition=condition, event_mode=mode,
                    ))
            del baseline, models, model
            if number == 7:
                completed = [row for row in rows + fold_rows if row["condition"] == condition]
                summary = pd.DataFrame(completed).groupby(["event_mode", "stage"])[prior.METRICS].mean().reset_index()
                for mode in modes:
                    result = summary[(summary.event_mode == mode) & (summary.stage == "after_rules")].iloc[0]
                    print(f"RESULT {condition} {mode} dead={result.dead_spell_f1:.6f} alive={result.alive_spell_f1:.6f}", flush=True)
        if len(fold_rows) != expected_rows:
            raise ValueError(f"Expected {expected_rows} half/condition/stage rows.")
        tmp = done.with_suffix(".tmp")
        tmp.write_text(json.dumps(fold_rows, indent=2) + "\n", encoding="utf-8")
        tmp.replace(done)
        rows.extend(fold_rows)
        report(args.output, rows)
        for half in parts.values():
            half.pop("restored_train", None)
            half.pop("restored_test", None)
            half.pop("restored_tuning", None)
        print(f"EVALUATION {number}/{len(spec['folds'])} complete in {perf_counter()-start:.0f}s", flush=True)
    if len(rows) != len(spec["folds"]) * expected_rows:
        raise ValueError("Missing evaluation scores.")
    print(report(args.output, rows).to_string(index=False), flush=True)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    split = parser.add_mutually_exclusive_group(required=True)
    split.add_argument("--splits", type=Path)
    split.add_argument("--holdout", choices=dfl_load.MATCHES,
                       help="Evaluate both halves of one match, with the other six used for training")
    parser.add_argument("--validation-match", choices=dfl_load.MATCHES,
                        help="Select epochs using this training match before refitting on all six")
    parser.add_argument("--train-loss-only", action="store_true",
                        help="Train the holdout classifier once, selecting by training loss")
    parser.add_argument("--ball-weights", type=Path, help="Previously fitted ball restorer from the same split")
    parser.add_argument("--inputs", type=Path, default=ROOT / "data/model_inputs/transportmer")
    parser.add_argument("--cache", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--epochs", type=int, default=25)
    parser.add_argument("--patience", type=int, default=4)
    parser.add_argument("--learning-rate", type=float, default=0.0002)
    parser.add_argument("--motion-input", action="store_true", help="Add consecutive-coordinate displacement and its validity mask")
    parser.add_argument("--pilot", action="store_true", help="Run only fold 1 without event inputs or ball restoration")
    parser.add_argument("--model-1", action="store_true", help="Run only restored ball, no events, and export CSVs")
    parser.add_argument("--export-csv", action="store_true", help="Write frame-level CSVs next to existing NPZ files")
    parser.add_argument("--prepare-dfl", action="store_true", help="Build missing training inputs from local DFL data")
    parser.add_argument("--benchmark", action="store_true", help="Time eight training batches per task from fold 1")
    parser.add_argument("--device", choices=("cuda", "mps", "cpu"), default="cuda")
    parser.add_argument("--batch", type=int, default=24)
    parser.add_argument("--microbatch", type=int, help="Accumulate gradients for each logical batch")
    parser.add_argument("--ball-selected-epochs", type=int,
                        help="Reuse an epoch count already selected on the same training and validation split")
    parser.add_argument("--stride", type=int, default=240, help="Training-window stride in original frames")
    args = parser.parse_args()
    if args.microbatch is None:
        args.microbatch = args.batch
    if min(args.batch, args.epochs, args.stride, args.patience) < 1:
        parser.error("epochs, batch, stride and patience must be positive")
    if not math.isfinite(args.learning_rate) or args.learning_rate <= 0:
        parser.error("learning rate must be finite and positive")
    if args.device == "cuda" and not torch.cuda.is_available():
        parser.error("CUDA is unavailable")
    if args.device == "mps" and not torch.backends.mps.is_available():
        parser.error("MPS is unavailable")
    if args.pilot and (args.model_1 or args.benchmark):
        parser.error("--pilot is separate from --model-1 and --benchmark")
    if args.validation_match and (not args.holdout or args.validation_match == args.holdout):
        parser.error("--validation-match requires a different --holdout match")
    if not 1 <= args.microbatch <= args.batch:
        parser.error("--microbatch must be between 1 and --batch")
    if args.ball_selected_epochs is not None and not (
        args.model_1 and args.holdout and 1 <= args.ball_selected_epochs <= args.epochs
    ):
        parser.error("--ball-selected-epochs requires a model 1 holdout and a valid epoch count")
    if args.train_loss_only and (
        not args.model_1 or not args.holdout or not args.ball_weights or args.validation_match
        or args.ball_selected_epochs or args.benchmark
    ):
        parser.error("--train-loss-only requires --model-1, --holdout and --ball-weights")
    if args.ball_weights and not args.train_loss_only:
        parser.error("--ball-weights is for --train-loss-only")
    torch.set_float32_matmul_precision("high")
    run(args)


if __name__ == "__main__":
    main()
