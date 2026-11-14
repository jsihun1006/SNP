"""Seven-fold DFL evaluation of the four TranSPORTmer dead-ball conditions."""
import argparse
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
from config import CONFIG

WIDTH, HEADS, FFN = 128, 16, 512
WINDOW, STEP, FPS = 60, 4, 25
BALL_MISSING = 0.407
EVENT_MODES = ("none", "masked")
ROOT = Path(__file__).resolve().parents[1]


class Attention(nn.Module):
    def __init__(self):
        super().__init__()
        self.mha = nn.MultiheadAttention(WIDTH, HEADS, dropout=0.0, batch_first=True)
        self.norm_attention = nn.LayerNorm(WIDTH)
        self.norm_feed = nn.LayerNorm(WIDTH)
        self.feed = nn.Sequential(nn.Linear(WIDTH, FFN), nn.GELU(), nn.Linear(FFN, WIDTH))

    def forward(self, x, mask=None):
        keys = x
        if mask is not None:
            # A fully hidden trajectory has no valid key. A zero sentinel keeps
            # the attention finite while the query still carries its type and time.
            all_hidden = mask.all(dim=-1)
            keys = x.masked_fill(all_hidden[:, None, None], 0)
            mask = mask & ~(all_hidden[:, None] & (torch.arange(x.shape[1], device=x.device) == 0))
        y = self.mha(x, keys, keys, key_padding_mask=mask, need_weights=False)[0]
        x = self.norm_attention(x + y)
        return self.norm_feed(x + self.feed(x))


class TranSPORTmer(nn.Module):
    def __init__(self, task, use_events=False, motion_input=False):
        super().__init__()
        self.task = task
        self.motion_input = motion_input
        self.position = nn.Linear(5 if motion_input else 2, WIDTH)
        self.agent_type = nn.Embedding(4, WIDTH)
        self.source = nn.Embedding(3, WIDTH)
        self.event_input = nn.Linear(7, WIDTH) if use_events else None
        self.blocks = nn.ModuleList(Attention() for _ in range(6))
        self.readout = nn.Linear(WIDTH, {"ball": 2, "dead": 1}[task])
        dims = torch.exp(torch.arange(0, WIDTH, 2) * (-math.log(10000) / WIDTH))
        angle = torch.arange(WINDOW)[:, None] * dims
        self.register_buffer("time", torch.stack([angle.sin(), angle.cos()], -1).flatten(1))

    def forward(self, xy, status, kinds, event_values=None):
        batch, time, agents, _ = xy.shape
        coordinates = xy
        if self.motion_input:
            present = status > 0
            continuous = present[:, 1:] & present[:, :-1]
            # Displacements use the same slot and two consecutive observations.
            # Scale 6.25 Hz normalized displacements into a useful numerical range.
            change = 100 * (xy[:, 1:] - xy[:, :-1])
            change = torch.where(continuous[..., None], change, 0)
            change = torch.cat((torch.zeros_like(xy[:, :1]), change), dim=1)
            valid = torch.cat((torch.zeros_like(present[:, :1]), continuous), dim=1)
            centered = torch.where(present[..., None], 2 * xy - 1, 0)
            coordinates = torch.cat((centered, change, valid[..., None]), dim=-1)
        x = self.position(coordinates) + self.agent_type(kinds)[None, None] + self.source(status)
        if self.task != "ball":
            cls = self.agent_type.weight[3].expand(batch, time, 1, WIDTH)
            if self.event_input is not None:
                cls = cls + self.event_input(event_values).unsqueeze(2)
            x = torch.cat((x, cls), dim=2)
            status = torch.cat((status, torch.zeros_like(status[:, :, :1])), dim=2)
            agents += 1
        hidden = status == 0
        for stage in range(2):
            x = x + self.time[:time][None, :, None, :]
            x = x.transpose(1, 2).reshape(batch * agents, time, WIDTH)
            temporal = hidden.transpose(1, 2).reshape(batch * agents, time) if stage == 0 else None
            for index in (0, 1):
                x = self.blocks[stage * 3 + index](x, temporal)
            x = x.reshape(batch, agents, time, WIDTH).transpose(1, 2).reshape(batch * time, agents, WIDTH)
            social = hidden.reshape(batch * time, agents) if stage == 0 else None
            x = self.blocks[stage * 3 + 2](x, social).reshape(batch, time, agents, WIDTH)
        return self.readout(x[:, :, -1])


def arrays(half):
    pos, ball, kinds = half
    xy = np.concatenate((pos, ball[:, None]), axis=1).astype(np.float32)
    seen = np.isfinite(xy).all(axis=-1)
    return {"xy": np.where(seen[..., None], xy, 0), "status": seen.astype(np.uint8),
            "kind": np.r_[kinds, 2].astype(np.int64)}


def sample_ball(ball, rng):
    return simulate.ball_dropout(ball, rng, BALL_MISSING, 6.0).astype(np.float32)


def load_halves(directory, cache):
    CONFIG["dfl_cache"] = str(cache)
    parts = {}
    for index, match in enumerate(dfl_load.MATCHES):
        track, e, metadata = dfl_load.load_match(match)
        raw = features.match_from_dfl(track, metadata)
        test = simulate.simulate(raw, seed=CONFIG["sim_seed"])
        pitch = np.asarray(raw.pitch, dtype=np.float32)
        dead = prior.dfl_labels.dead_label(track)
        with np.load(directory / f"{match}.npz") as npz:
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


def fit_epochs(parts, keys, field, task, args, seed, conditioned, epochs, validation=None):
    torch.manual_seed(seed)
    rng = np.random.default_rng(seed)
    model = TranSPORTmer(task, use_events=conditioned, motion_input=args.motion_input).to(args.device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.learning_rate if task == "dead" else 0.001, eps=1e-4)
    begin = perf_counter()
    best_epoch, best_loss = 0, float("inf")
    for epoch in range(epochs):
        model.train()
        measure = []
        jobs = []
        for key in keys:
            indices = starts(len(parts[key]["dead"]), args.stride, phase=int(rng.integers(args.stride)))
            rng.shuffle(indices)
            jobs.extend((key, indices[i:i+args.batch]) for i in range(0, len(indices), args.batch))
        rng.shuffle(jobs)
        for key, indices in jobs:
            part = parts[key]
            xy, status, kind, ix = batch_tensor(part[field], indices, args.device)
            inputs = torch.as_tensor(encode_events(part["event"][ix]), device=args.device) if conditioned else None
            optimizer.zero_grad(set_to_none=True)
            with torch.autocast("cuda", dtype=torch.bfloat16):
                logits = model(xy, status, kind, inputs).float()
                if task == "ball":
                    truth = torch.as_tensor(part["truth_ball"][ix], device=args.device)
                    hidden = (status[:, :, -1] == 0) & torch.isfinite(truth).all(-1)
                    distance = torch.linalg.vector_norm((logits - truth.nan_to_num(0)) *
                                                        torch.tensor(part["pitch"], device=args.device), dim=-1)
                    loss = (distance * hidden).sum() / hidden.sum().clamp_min(1)
                else:
                    truth = torch.as_tensor(part["dead"][ix], device=args.device, dtype=torch.float32)
                    loss = nn.functional.binary_cross_entropy_with_logits(logits.squeeze(-1), truth)
            loss.backward()
            nn.utils.clip_grad_norm_(model.parameters(), 5)
            optimizer.step()
            measure.append(loss.detach())
        training_loss = torch.stack(measure).mean().item()
        if not np.isfinite(training_loss):
            raise ValueError(f"Non-finite {task} loss")
        if validation is not None:
            key = validation
            check = parts[key]
            masked = np.zeros((len(check["dead"]), 7), dtype=np.float32) if conditioned else None
            prediction = predict(model, check[field], args, masked)
            if task == "ball":
                valid = (check[field]["status"][:, -1] == 0) & np.isfinite(check["truth_ball"]).all(axis=1)
                if not valid.any():
                    raise ValueError("No masked ground truth in the ball validation half.")
                error = (prediction[valid] - check["truth_ball"][valid]) * np.asarray(check["pitch"])
                loss = float(np.linalg.norm(error, axis=1).mean())
            else:
                prob = np.clip(prediction[:, 0].astype(float), 1e-7, 1 - 1e-7)
                truth = check["dead"].astype(bool)
                loss = float(-np.where(truth, np.log(prob), np.log1p(-prob)).mean())
            if loss < best_loss - 1e-4:
                best_epoch, best_loss = epoch + 1, loss
                if task == "ball":
                    best_weights = {key: value.detach().cpu().clone() for key, value in model.state_dict().items()}
            print(f"VALID {field} {task} conditioned={conditioned} epoch={epoch+1}/{epochs} "
                  f"loss={loss:.5f} best={best_epoch} elapsed={perf_counter()-begin:.0f}s", flush=True)
            if epoch + 1 - best_epoch >= args.patience:
                break
        elif epoch + 1 == epochs:
            print(f"REFIT {field} {task} conditioned={conditioned} epochs={epochs} "
                  f"loss={training_loss:.5f} secs={perf_counter()-begin:.0f}", flush=True)
    if validation is not None and task == "ball":
        model.load_state_dict(best_weights)
    return model, best_epoch


def fit(parts, keys, field, task, args, seed, conditioned=False, tuning_field=None, on_tuned=None):
    # Fixed by the shuffled split; no evaluation half participates in stopping.
    validation = keys[-1]
    model, best_epoch = fit_epochs(parts, keys[:-1], tuning_field or field, task, args, seed, conditioned,
                                   args.epochs, validation=validation)
    if on_tuned is not None:
        on_tuned(model)
    del model
    print(f"STOP {field} {task} conditioned={conditioned} selected_epoch={best_epoch} "
          f"validation={validation}", flush=True)
    model, _ = fit_epochs(parts, keys, field, task, args, seed, conditioned, best_epoch)
    return model


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
    for begin in range(0, len(boundaries), args.batch):
        xy, status, kind, ix = batch_tensor(data, boundaries[begin:begin+args.batch], args.device)
        e = torch.as_tensor(events_input[ix], device=args.device) if events_input is not None else None
        with torch.autocast("cuda", dtype=torch.bfloat16):
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
    changes = np.count_nonzero(state[:-1] != state[1:])
    print(f'DIAG {name} pstd={score.std():.5f} dead_share={state.mean():.4f} '
          f'prediction_runs={changes + 1}', flush=True)

    return [
        dict(**key, stage=stage, **prior.spell_f1(truth, labels, game["period"]))
        for stage, labels in (("before_rules", state), ("after_rules", decoded))
    ]


def run(args):
    spec = prior.load_folds(args.splits)
    movement, decode, settings = load_rules()
    parts = load_halves(args.inputs, args.cache)
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
    manifest = {
        "splits": spec, "width": WIDTH, "heads": HEADS, "ffn": FFN, "tokens": WINDOW,
        "hz": FPS/STEP, "max_epochs": args.epochs, "dead_learning_rate": args.learning_rate, "window_stride_25fps": args.stride, "batch": args.batch,
        "early_stopping": {"patience": args.patience, "min_delta": 1e-4, "validation": "last training half",
                           "metrics": {"ball": "masked ADE", "dead": "binary log loss with event masking"},
                           "refit_all_training_halves": True,
                           "inner_restore_uses_only_inner_training": True},
        "motion_input": args.motion_input,
        "ball_missing_target": BALL_MISSING, "ball_mask_mean_gap_seconds": 6,
        "ball_mask_seed_train": 1200, "ball_mask_seed_test": 1400,
        "event_window_seconds_each_side": events.WINDOW_SECONDS,
        "event_modes": list(EVENT_MODES),
        "evaluation_fps": FPS, "rules": list(prior.RULES), "rule_settings": settings,
        "uncorroborated_restored_crossings_entry_cost_multiplier": 0.7,
        "weak_crossing_guard": "observed inside within 1s before plus observed outside within 1s after or observed restart within 3s",
    }
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
            rows.extend(old)
            print(f"FOLD {number} already complete", flush=True)
            continue
        start = perf_counter()
        def restore_inner(imputer):
            for key in train:
                parts[key]["restored_tuning"] = complete(imputer, parts[key]["train"], args)

        imputer = fit(parts, train, "train", "ball", args, 700 + number, on_tuned=restore_inner)
        for key in train:
            parts[key]["restored_train"] = complete(imputer, parts[key]["train"], args)
        for key in test:
            parts[key]["restored_test"] = complete(imputer, parts[key]["test"], args)
        del imputer
        fold_rows = []
        for condition, training, testing in (("observed", "train", "test"),
                                             ("restored", "restored_train", "restored_test")):
            seed = 800 + number
            tuning = "restored_tuning" if condition == "restored" else training
            baseline = fit(parts, train, training, "dead", args, seed, tuning_field=tuning)
            conditioned = fit(parts, train, training, "dead", args, seed,
                              conditioned=True, tuning_field=tuning)
            for key in test:
                half = parts[key]
                data = half[testing]
                masked = np.zeros((len(data["xy"]), 7), dtype=np.float32)
                for mode, model, event_values in (
                    ("none", baseline, None), ("masked", conditioned, masked),
                ):
                    fold_rows.extend(score_half(
                        model, half, testing, args, event_values, movement, decode, settings,
                        fold=number, half_id=key, condition=condition, event_mode=mode,
                    ))
            del baseline, conditioned
            if number == 7:
                completed = [row for row in rows + fold_rows if row["condition"] == condition]
                summary = pd.DataFrame(completed).groupby(["event_mode", "stage"])[prior.METRICS].mean().reset_index()
                for mode in EVENT_MODES:
                    result = summary[(summary.event_mode == mode) & (summary.stage == "after_rules")].iloc[0]
                    print(f"RESULT {condition} {mode} dead={result.dead_spell_f1:.6f} alive={result.alive_spell_f1:.6f}", flush=True)
        if len(fold_rows) != 16:
            raise ValueError("Expected 16 half/condition/stage rows.")
        tmp = done.with_suffix(".tmp")
        tmp.write_text(json.dumps(fold_rows, indent=2) + "\n", encoding="utf-8")
        tmp.replace(done)
        rows.extend(fold_rows)
        report(args.output, rows)
        for half in parts.values():
            half.pop("restored_train", None)
            half.pop("restored_test", None)
            half.pop("restored_tuning", None)
        print(f"FOLD {number}/7 complete in {perf_counter()-start:.0f}s", flush=True)
    if len(rows) != 112:
        raise ValueError("Expected seven complete folds.")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--splits", type=Path, required=True)
    parser.add_argument("--inputs", type=Path, default=ROOT / "data/model_inputs/transportmer")
    parser.add_argument("--cache", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--epochs", type=int, default=25)
    parser.add_argument("--patience", type=int, default=4)
    parser.add_argument("--learning-rate", type=float, default=0.0002)
    parser.add_argument("--motion-input", action="store_true", help="Add consecutive-coordinate displacement and its validity mask")
    parser.add_argument("--pilot", action="store_true", help="Run only fold 1 without event inputs or ball restoration")
    parser.add_argument("--batch", type=int, default=24)
    parser.add_argument("--stride", type=int, default=240, help="Training-window stride in original frames")
    args = parser.parse_args()
    if min(args.batch, args.epochs, args.stride, args.patience) < 1:
        parser.error("epochs, batch, stride and patience must be positive")
    if not math.isfinite(args.learning_rate) or args.learning_rate <= 0:
        parser.error("learning rate must be finite and positive")
    if not torch.cuda.is_available():
        parser.error("CUDA is required")
    args.device = "cuda"
    torch.set_float32_matmul_precision("high")
    run(args)


if __name__ == "__main__":
    main()
