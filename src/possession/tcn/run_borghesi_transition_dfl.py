"""Compare state and transition heads on shuffled DFL half folds."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys
from time import perf_counter

import numpy as np
import torch
from torch.nn import functional as F

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from run_borghesi_tcn_dfl import (
    CONTEXT_DILATIONS, DFL_FPS, INPUT_FPS, LEFT, RIGHT, ROOT, PossessionTCN, batches,
    dfl_load, draw_blocks, load_match, postprocess, references, scores, write_json,
)
from smoothing import viterbi
from transportmer_ha_metrics import metrics

EVENT_GAIN = (0.0, 0.5, 0.9, 1.0)


def make_folds(seed):
    halves = [(match, half) for match in sorted(dfl_load.MATCHES) for half in (1, 2)]
    order = np.random.default_rng(seed).permutation(len(halves))
    groups = [[halves[i] for i in order[start:start + 2]] for start in range(0, len(order), 2)]
    return [
        {
            "fold": i + 1,
            "test": test,
            "validation": groups[(i + 1) % len(groups)],
            "train": [half for j, group in enumerate(groups)
                      if j not in (i, (i + 1) % len(groups)) for half in group],
        }
        for i, test in enumerate(groups)
    ]


def event_targets(item):
    """Directions H→A and A→H between consecutive sampled target rows."""
    labels, rows = item.half.labels, item.sample_rows
    known = (labels == scores.H) | (labels == scores.A)
    event = np.zeros((len(labels), 3), dtype=np.int8)
    event[1:, 0] = (labels[:-1] == scores.H) & (labels[1:] == scores.A)
    event[1:, 1] = (labels[:-1] == scores.A) & (labels[1:] == scores.H)
    event[1:, 2] = ~(known[:-1] & known[1:])
    prefix = np.cumsum(event, axis=0, dtype=np.int32)
    interval = prefix[rows[1:]] - prefix[rows[:-1]]
    valid = (interval[:, 2] == 0) & (interval[:, :2].sum(axis=1) <= 1)
    target = np.full((len(rows), 2), -1, dtype=np.int8)
    target[1:] = np.where(valid[:, None], interval[:, :2], -1)
    return target


def batch_events(targets, refs, span, device):
    values = np.full((len(refs), span, 2), -1, dtype=np.int8)
    for row, (half, start) in zip(values, refs, strict=True):
        part = targets[half][start:start + span]
        row[:len(part)] = part
    return torch.from_numpy(values).to(device)


@torch.inference_mode()
def predict(model, halves, args):
    model.eval()
    channels = 3 if model.transitions is not None else 1
    refs = references(halves, args.span)
    sampled = [np.empty((len(half.sample_rows), channels), dtype=np.float32) for half in halves]
    for offset, (features, padding, _, sizes) in enumerate(
        batches(halves, refs, args.batch, args.span, args.device, window=model.context_frames)
    ):
        probabilities = model(features, padding)[..., 1:].sigmoid().cpu().numpy()
        group = refs[offset * args.batch:(offset + 1) * args.batch]
        for row, ((index, first), size) in enumerate(zip(group, sizes, strict=True)):
            sampled[index][first:first + size] = probabilities[row, :size]
    return [
        np.column_stack([
            np.interp(np.arange(len(item.half.frame)), item.sample_rows, channel)
            for channel in values.T
        ])
        for item, values in zip(halves, sampled, strict=True)
    ]


def decode(half, probability, penalty, gain=0.0):
    pred = np.full(len(half.frame), scores.D, dtype=np.int8)
    for start, stop in half.episodes:
        values = probability[start:stop]
        # The event head orders columns by H→A then A→H. Viterbi indexes by destination.
        switch_cost = penalty * (1.0 - gain * values[:, [2, 1]]) if gain else penalty
        pred[start:stop] = viterbi(values[:, 0], np.zeros(len(values), dtype=np.int8), switch_cost)
    return pred


def evaluate(halves, probabilities, condition, penalty, gain=0.0):
    if condition == "A":
        predicted = [postprocess(half, p[:, 0], "prob35")
                     for half, p in zip(halves, probabilities, strict=True)]
    else:
        predicted = [decode(half, p, penalty, gain)
                     for half, p in zip(halves, probabilities, strict=True)]
    result = metrics(halves, predicted)
    valid = [half.alive & (half.labels >= 0) for half in halves]
    result["ha_frame_accuracy"] = float(
        sum(int((half.labels[mask] == pred[mask]).sum())
            for half, pred, mask in zip(halves, predicted, valid, strict=True))
        / sum(int(mask.sum()) for mask in valid)
    )
    return result, predicted


def train(model, halves, validation, args, fold_seed, folder):
    optimizer = torch.optim.Adam(model.parameters(), lr=args.lr)
    torch.manual_seed(fold_seed)
    val_halves = [item.half for item in validation]
    labels = [event_targets(item) for item in halves] if model.transitions is not None else None
    frequencies = None
    if labels is not None:
        positive = np.sum([(values == 1).sum(0) for values in labels], axis=0)
        negative = np.sum([(values == 0).sum(0) for values in labels], axis=0)
        weight = torch.tensor(np.clip(negative / positive.clip(min=1), 1, 200),
                              dtype=torch.float32, device=args.device)
        frequencies = {"positive": positive.tolist(), "negative": negative.tolist(),
                       "pos_weight": weight.cpu().tolist()}
    history = []
    best_score = -1.0
    stale_checks = 0
    checkpoint = folder / f"{args.model}.pt"
    for epoch in range(1, args.epochs + 1):
        started = perf_counter()
        model.train()
        for group in optimizer.param_groups:
            group["lr"] = args.lr * args.decay ** (epoch - 1)
        refs = draw_blocks(halves, args.span, args.samples, np.random.default_rng([fold_seed, epoch]))
        train_loss = torch.zeros((), device=args.device)
        for step, (features, padding, states, _) in enumerate(
            batches(halves, refs, args.batch, args.span, args.device, window=model.context_frames)
        ):
            optimizer.zero_grad(set_to_none=True)
            logits = model(features, padding)
            loss = model.loss(logits, states)
            if labels is not None:
                group = refs[step * args.batch:(step + 1) * args.batch]
                target = batch_events(labels, group, args.span, args.device)
                event_loss = F.binary_cross_entropy_with_logits(
                    logits[..., 2:], target.clamp_min(0).float(), pos_weight=weight, reduction="none"
                )
                known = target >= 0
                loss = loss + args.event_loss * (event_loss * known).sum() / known.sum().clamp_min(1)
            loss.backward()
            optimizer.step()
            train_loss += loss.detach()
        record = {"epoch": epoch, "train_loss": float(train_loss / (step + 1)),
                  "lr": optimizer.param_groups[0]["lr"],
                  "train_seconds": perf_counter() - started}
        stop_early = False
        if epoch % args.validate_every == 0 or epoch == args.epochs:
            prediction = predict(model, validation, args)
            validation_scores, _ = evaluate(val_halves, prediction, "A", args.penalty)
            record["validation_prob35_spellF1"] = validation_scores["spellF1"]
            if validation_scores["spellF1"] > best_score:
                best_score = validation_scores["spellF1"]
                stale_checks = 0
                temporary = checkpoint.with_suffix(".tmp")
                torch.save({"epoch": epoch, "weights": model.state_dict()}, temporary)
                temporary.replace(checkpoint)
            else:
                stale_checks += 1
                stop_early = bool(args.patience and stale_checks >= args.patience and epoch < args.epochs)
            record["stale_validation_checks"] = stale_checks
            record["early_stop"] = stop_early
        history.append(record)
        write_json(folder / f"{args.model}_training.json", history)
        print(f"  epoch {epoch:02d}/{args.epochs} {json.dumps(record)}", flush=True)
        if stop_early:
            break
    selected = torch.load(checkpoint, map_location=args.device, weights_only=True)
    model.load_state_dict(selected["weights"])
    return selected["epoch"], history, frequencies


def choose_gain(validation, probabilities, penalty):
    rows = []
    for gain in EVENT_GAIN:
        result, _ = evaluate(validation, probabilities, "D", penalty, gain)
        rows.append({"gain": gain, **result})
    best = max(rows, key=lambda row: (row["spellF1"], row["short_matched"], -row["gain"]))
    return best["gain"], rows


def summarize(results):
    report = {}
    for condition in results[0]["conditions"]:
        rows = [fold["conditions"][condition] for fold in results]
        report[condition] = {key: float(np.mean([row[key] for row in rows]))
                             for key in ("spellF1", "ha_frame_accuracy")}
        summary = report[condition]
        for key in ("spurious_transitions", "missed_transitions", "matched_transitions",
                    "short_matched", "short_spells"):
            summary[key] = sum(row[key] for row in rows)
        summary["short_recall_iou_0_5"] = summary["short_matched"] / max(1, summary["short_spells"])
        matched = summary["matched_transitions"]
        summary["transition_mae_ms"] = (
            sum((row["transition_mae_ms"] or 0.0) * row["matched_transitions"] for row in rows)
            / matched if matched else None
        )
    return report


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model", choices=("state", "transition"), required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--cache", type=Path, default=ROOT / "data/external/dfl")
    parser.add_argument("--device", default="cpu")
    parser.add_argument("--threads", type=int, default=4)
    parser.add_argument("--seed", type=int, default=2027)
    parser.add_argument("--width", type=int, default=64)
    parser.add_argument("--batch", type=int, default=8)
    parser.add_argument("--span", type=int, default=128)
    parser.add_argument("--samples", type=int, default=24576)
    parser.add_argument("--epochs", type=int, default=12)
    parser.add_argument("--validate-every", type=int, default=3)
    parser.add_argument("--past-frames", type=int, default=LEFT)
    parser.add_argument("--future-frames", type=int, default=RIGHT)
    parser.add_argument("--patience", type=int, default=0, help="Consecutive validation checks without improvement. 0 disables early stopping.")
    parser.add_argument("--lr", type=float, default=0.001)
    parser.add_argument("--decay", type=float, default=0.7)
    parser.add_argument("--penalty", type=float, default=4.0)
    parser.add_argument("--event-loss", type=float, default=0.25)
    args = parser.parse_args()
    if min(args.width, args.batch, args.span, args.samples, args.epochs,
           args.validate_every, args.threads) < 1:
        parser.error("Width, batch, span, samples, epochs, validation interval and threads must be positive.")
    window = args.past_frames + 1 + args.future_frames
    if min(args.past_frames, args.future_frames, args.patience) < 0 or window not in CONTEXT_DILATIONS:
        parser.error("Use nonnegative past, future and patience, and a 32- or 64-frame context.")
    torch.set_num_threads(args.threads)
    if args.device.startswith("cuda") and not torch.cuda.is_available():
        parser.error("CUDA is unavailable. Select a GPU Colab runtime or pass --device cpu.")
    args.output.mkdir(parents=True, exist_ok=True)
    started = perf_counter()
    folds = make_folds(args.seed)
    settings = {key: value for key, value in vars(args).items()
                if key not in {"cache", "output", "threads", "device"}}
    settings.update(
        folds=json.loads(json.dumps(folds)), players="nearest5", window=window,
        dilations=list(CONTEXT_DILATIONS[window]),
        attention_heads_per_branch=1, fps=INPUT_FPS, evaluation_fps=DFL_FPS,
        deadball_mask="ground_truth",
        blocks_per_epoch=max(1, args.samples // args.span), event_gain_candidates=list(EVENT_GAIN),
    )
    manifest = args.output / "manifest.json"
    if manifest.exists():
        previous = json.loads(manifest.read_text())["settings"]
        previous.setdefault("patience", 0)
        previous.setdefault("dilations", list(CONTEXT_DILATIONS[previous["window"]]))
        if previous != settings:
            parser.error(f"Settings differ from {manifest}. Choose a new output folder.")
    else:
        write_json(manifest, {"settings": settings, "initial_device": args.device,
                              "initial_torch_version": str(torch.__version__)})
    existed = [(args.output / f"fold_{fold['fold']:02d}" / "result.json") for fold in folds]
    all_halves = {}
    if not all(path.is_file() for path in existed):
        all_halves = {(match, item.half.number): item for match in sorted(dfl_load.MATCHES)
                      for item in load_match(match, args.cache, past=args.past_frames, future=args.future_frames)}
    prepared = perf_counter() - started
    print(f"Prepared 14 halves in {prepared:.1f}s on {args.device}.", flush=True)
    results = []
    for fold, completed in zip(folds, existed, strict=True):
        if completed.is_file():
            results.append(json.loads(completed.read_text()))
            print(f"Fold {fold['fold']}/7 loaded from disk", flush=True)
            continue
        tick = perf_counter()
        folder = completed.parent
        folder.mkdir(exist_ok=True)
        print(f"Fold {fold['fold']}/7 validation={fold['validation']} test={fold['test']}", flush=True)
        training = [all_halves[key] for key in fold["train"]]
        validation = [all_halves[key] for key in fold["validation"]]
        torch.manual_seed(args.seed + fold["fold"])
        extended = args.model == "transition"
        model = PossessionTCN("shared", args.width, "nearest5", transitions=extended,
                              past=args.past_frames, future=args.future_frames).to(args.device)
        epoch, history, frequencies = train(model, training, validation, args, args.seed + fold["fold"], folder)
        gain, gain_rows = (0.0, [])
        if extended:
            gain, gain_rows = choose_gain([item.half for item in validation], predict(model, validation, args), args.penalty)
        testing = [all_halves[key] for key in fold["test"]]
        probabilities = predict(model, testing, args)
        test_halves = [item.half for item in testing]
        keys = ("C", "D") if extended else ("A", "B")
        condition_rows = {
            key: evaluate(test_halves, probabilities, key, args.penalty, gain if key == "D" else 0.0)
            for key in keys
        }
        for i, half in enumerate(test_halves):
            np.savez_compressed(folder / f"{half.key}.npz", frame=half.frame, time=half.time,
                                truth=half.labels, alive=half.alive, probability=probabilities[i],
                                **{key: condition_rows[key][1][i] for key in keys})
        result = {
            **fold, "conditions": {key: result for key, (result, _) in condition_rows.items()},
            "event_gain": gain, "gain_validation": gain_rows, "event_population": frequencies,
            "selected_epoch": epoch, "epochs_run": len(history),
            "early_stopped": len(history) < args.epochs,
            "training_seconds": sum(row["train_seconds"] for row in history),
            "seconds": perf_counter() - tick, "device": args.device,
        }
        write_json(completed, result)
        results.append(result)
        print(f"Fold {fold['fold']} finished in {result['seconds']:.1f}s "
              f"{json.dumps({key: result['conditions'][key]['spellF1'] for key in keys})}", flush=True)
        write_json(args.output / "results.json", {"folds": results, "summary": summarize(results)})
    report = {"folds": results, "summary": summarize(results), "prepared_seconds": prepared,
              "current_run_seconds": perf_counter() - started}
    write_json(args.output / "results.json", report)
    print(json.dumps(report["summary"], indent=2), flush=True)


if __name__ == "__main__":
    main()
