"""Train the Borghesi et al. (2023) TCN variants on local Sportec DFL."""

from __future__ import annotations

import argparse
from dataclasses import dataclass
import json
from pathlib import Path
import sys
from time import perf_counter

import numpy as np
import pandas as pd
import torch
from torch import nn
from torch.nn import functional as F

ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(ROOT))

import dfl_load
from evaluation import possession_metrics as scores
from features import match_from_dfl
from transportmer_ha_data import Half, augment_ball, contiguous_runs, seed_for, split_matches
from transportmer_ha_metrics import metrics, postprocess

WINDOW = 64
LEFT = 47
RIGHT = WINDOW - LEFT - 1
INPUT_FPS = 16
DFL_FPS = 25
VARIANTS = ("single", "shared", "separate")
CONTEXT_DILATIONS = {32: (1, 2, 4, 8, 8, 8), 64: (1, 2, 4, 8, 16, 32)}


@dataclass
class TCNHalf:
    half: Half
    sample_rows: np.ndarray
    padded_features: np.ndarray


def features_at_rate(xy, times, sample_rows, attributes, pitch):
    observed = np.isfinite(xy).all(axis=2)
    selected = xy[sample_rows]
    seen = observed[sample_rows]
    outside = seen & ((selected < 0) | (selected > 1)).any(axis=-1)
    invalid_prefix = np.cumsum(~observed, axis=0, dtype=np.int32)
    continuous = seen[1:] & (invalid_prefix[sample_rows[1:]] == invalid_prefix[sample_rows[:-1]])
    continuous &= seen[:-1]
    velocity = np.full_like(selected, -2)
    elapsed = np.diff(times[sample_rows]).astype(np.float32)
    change = np.diff(selected, axis=0) * np.asarray(pitch, dtype=np.float32) / elapsed[:, None, None]
    velocity[1:] = np.where(continuous[..., None], np.clip(change / 15.0, -3, 3), -2)
    position = np.where(seen[..., None], 2 * np.clip(selected, 0, 1) - 1, -2)
    flags = np.broadcast_to(attributes, (*position.shape[:2], attributes.shape[-1]))
    return np.concatenate((position, velocity, flags, outside[..., None], (~seen)[..., None]), axis=-1)


def load_match(match_id, cache, profile=None, seed=2027, *, past=LEFT, future=RIGHT):
    dfl_load.CONFIG["dfl_cache"] = str(cache)
    if not all(Path(path).is_file() for path in dfl_load.cache_paths(match_id)):
        raise FileNotFoundError(f"Local DFL cache missing for {match_id}: {cache}")
    tracking, _, meta = dfl_load.load_match(match_id)
    match = match_from_dfl(tracking, meta)
    positions = meta.set_index("player_id")["position"]
    player_ids = [col[:-2] for col in tracking.columns if col.endswith("_x") and col[:-2] in positions]
    goalkeeper = positions.loc[player_ids].str.contains("goalkeeper", case=False, na=False).to_numpy()
    # DFL Parquet contains no referee. Keep the referee flag as a zero channel.
    attributes = np.zeros((len(player_ids) + 1, 5), dtype=np.float32)
    attributes[0, 0] = 1
    attributes[1:, 2] = goalkeeper
    attributes[1:, 3] = match.team == 0
    attributes[1:, 4] = match.team == 1
    pitch = np.asarray(match.pitch, dtype=np.float32)
    result = []
    for number in (1, 2):
        rows = match.period == number
        frame = match.frame[rows]
        times = tracking.loc[rows, "timestamp_s"].to_numpy()
        if len(frame) < 2 or np.any(np.diff(frame) != 1) or np.any(np.diff(times) <= 0):
            raise ValueError(f"Expected a contiguous DFL half: {match_id} H{number}")
        alive = tracking.loc[rows, "ball_state"].to_numpy() == 1
        labels = tracking.loc[rows, "ball_owning_team"].map({"home": 0, "away": 1}).fillna(-1).to_numpy(np.int8)
        labels[~alive] = scores.D
        episodes = contiguous_runs(alive, frame)
        ball = match.ball[rows] / pitch
        augmentation = {}
        if profile is not None:
            ball, augmentation = augment_ball(ball, episodes, profile,
                np.random.default_rng(seed_for(seed, f"{match_id}:H{number}")), 0.25)
        elif not np.isfinite(ball).all():
            from scipy.interpolate import Akima1DInterpolator

            present = np.isfinite(ball).all(axis=1)
            if present.sum() >= 5:
                ball[~present] = Akima1DInterpolator(times[present], ball[present])(times[~present])
        xy = np.concatenate((ball[:, None], match.players[rows] / pitch), axis=1)
        grid = np.arange((len(frame) - 1) * INPUT_FPS // DFL_FPS + 1) / INPUT_FPS
        relative_time = times - times[0]
        upper = np.searchsorted(relative_time, grid).clip(1, len(times) - 1)
        lower = upper - 1
        sample_rows = np.where(grid - relative_time[lower] <= relative_time[upper] - grid, lower, upper)
        if np.any(np.diff(sample_rows) <= 0):
            raise ValueError(f"16 Hz resampling produced duplicate targets: {match_id} H{number}")
        features = features_at_rate(xy, times, sample_rows, attributes, pitch)
        padded = np.pad(features, ((past, future), (0, 0), (0, 0)), mode="edge")
        half = Half(match_id, number, frame, times, alive, labels, xy, np.r_[2, match.team],
                    match.pitch, episodes, augmentation)
        result.append(TCNHalf(half, sample_rows, padded.astype(np.float16)))
    return result


class TrajectoryEncoder(nn.Module):
    """Dense application of the last-output TCN to overlapping windows."""

    def __init__(self, width, dropout=0.10, *, dilations=CONTEXT_DILATIONS[WINDOW]):
        super().__init__()
        self.project = nn.Linear(11, width)
        self.first_norm = nn.BatchNorm1d(width)
        self.dilations = dilations
        self.temporal = nn.ModuleList(nn.Linear(2 * width, width) for _ in dilations)
        self.pointwise = nn.ModuleList(nn.Linear(width, width) for _ in dilations)
        self.last_norm = nn.BatchNorm1d(width)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x):
        x = F.relu(self.project(x))
        x = self.first_norm(x.reshape(-1, x.shape[-1])).reshape(x.shape)
        for dilation, temporal, pointwise in zip(self.dilations, self.temporal, self.pointwise, strict=True):
            x = torch.cat((x[:, :-dilation], x[:, dilation:]), dim=-1)
            x = F.relu(pointwise(self.dropout(F.relu(temporal(x)))))
        x = self.dropout(x)
        return self.last_norm(x.reshape(-1, x.shape[-1])).reshape(x.shape)


class BallAttentionHead(nn.Module):
    def __init__(self, width, outputs):
        super().__init__()
        self.query = nn.Linear(width, width)
        self.key = nn.Linear(width, width)
        self.value = nn.Linear(width, width)
        self.ffn = nn.Sequential(nn.Linear(width, 64), nn.ELU(), nn.Linear(64, 32), nn.ELU(), nn.Linear(32, outputs))
        self.scale = width ** -0.5

    def pool(self, embeddings, padding):
        query = self.query(embeddings[:, :, :1])
        attention = (query * self.key(embeddings)).sum(dim=-1) * self.scale
        weights = attention.masked_fill(padding, -torch.inf).softmax(dim=-1)
        return (self.value(embeddings) * weights[..., None]).sum(dim=-2)

    def forward(self, embeddings, padding):
        return self.ffn(self.pool(embeddings, padding))


class PossessionTCN(nn.Module):
    def __init__(self, variant, width, players="all", *, transitions=False, past=LEFT, future=RIGHT):
        super().__init__()
        if transitions and variant != "shared":
            raise ValueError("The transition head requires the shared TCN variant.")
        self.variant = variant
        self.players = players
        self.past = past
        self.context_frames = past + 1 + future
        dilations = CONTEXT_DILATIONS[self.context_frames]
        self.encoder = TrajectoryEncoder(width, dilations=dilations)
        self.first = BallAttentionHead(width, 3 if variant == "single" else 1)
        if variant != "single":
            self.possession = BallAttentionHead(width, 1)
        if variant == "separate":
            self.possession_encoder = TrajectoryEncoder(width, dilations=dilations)
        for module in self.modules():
            if isinstance(module, nn.Linear):
                nn.init.xavier_uniform_(module.weight)
                nn.init.zeros_(module.bias)
        self.transitions = None
        if transitions:
            self.transitions = nn.Linear(width, 2)
            nn.init.xavier_uniform_(self.transitions.weight)
            nn.init.zeros_(self.transitions.bias)

    def forward(self, value, padding):
        padding = padding[:, None]
        if self.players == "ball":
            value, padding = value[:, :, :1], padding[..., :1]
        elif self.players == "nearest5":
            outputs = value.shape[1] - self.context_frames + 1
            target_frame = value[:, self.past:self.past + outputs]
            xy = target_frame[..., :2]
            delta = xy[..., 1:, :] - xy[..., :1, :]
            distance = (105 * delta[..., 0]).square() + (68 * delta[..., 1]).square()
            unavailable = padding[..., 1:] | (target_frame[..., 1:, -1] > 0) | (target_frame[..., :1, -1] > 0)
            nearest = distance.masked_fill(unavailable, torch.inf).topk(min(5, distance.shape[-1]), largest=False).indices
            allowed = torch.zeros_like(unavailable).scatter_(-1, nearest, True)
            padding = torch.cat((torch.zeros_like(allowed[..., :1]), ~allowed | unavailable), dim=-1)
        encoded = self.encoder(value)
        first = self.first(encoded, padding)
        if self.variant == "single":
            return first
        if self.variant == "separate":
            encoded = self.possession_encoder(value)
        pooled = self.possession.pool(encoded, padding)
        outputs = (first, self.possession.ffn(pooled))
        if self.transitions is not None:
            outputs += (self.transitions(pooled),)
        return torch.cat(outputs, dim=-1)

    def loss(self, logits, targets):
        if self.variant == "single":
            return F.cross_entropy(logits.flatten(0, 1), targets.flatten(), ignore_index=-1)
        valid = targets >= 0
        possession = valid & (targets != scores.D)
        dead_loss = F.binary_cross_entropy_with_logits(logits[..., 0][valid], (targets[valid] == scores.D).float())
        ha_loss = F.binary_cross_entropy_with_logits(logits[..., 1][possession], targets[possession].float(),
                                                    reduction="sum") / possession.sum().clamp(min=1)
        return 0.5 * (dead_loss + ha_loss)

    def probabilities(self, logits):
        if self.variant == "single":
            return logits.softmax(-1)[..., scores.D], logits[..., :2].softmax(-1)[..., scores.A]
        return logits[..., 0].sigmoid(), logits[..., 1].sigmoid()


def batches(halves, references, batch_size, span, device, *, window=WINDOW):
    for offset in range(0, len(references), batch_size):
        group = references[offset:offset + batch_size]
        tracks = max(halves[index].padded_features.shape[1] for index, _ in group)
        value = np.zeros((len(group), span + window - 1, tracks, 11), dtype=np.float32)
        padding = np.ones((len(group), tracks), dtype=bool)
        target = np.full((len(group), span), -1, dtype=np.int64)
        sizes = []
        for j, (index, first) in enumerate(group):
            half = halves[index]
            size = min(span, len(half.sample_rows) - first)
            features = half.padded_features[first:first + size + window - 1]
            value[j, :len(features), :features.shape[1]] = features
            if len(features) < value.shape[1]:
                value[j, len(features):, :features.shape[1]] = features[-1]
            padding[j, :features.shape[1]] = False
            target[j, :size] = half.half.labels[half.sample_rows[first:first + size]]
            sizes.append(size)
        yield (torch.from_numpy(value).to(device), torch.from_numpy(padding).to(device),
               torch.from_numpy(target).to(device), sizes)


def references(halves, span):
    return [(i, first) for i, half in enumerate(halves) for first in range(0, len(half.sample_rows), span)]


def draw_blocks(halves, span, count, rng):
    range_sizes = np.array([max(1, len(half.sample_rows) - span + 1) for half in halves])
    chosen = rng.choice(len(halves), size=max(1, count // span), p=range_sizes / range_sizes.sum())
    return [(int(i), int(rng.integers(range_sizes[i]))) for i in chosen]


def decode_frame(variant, dead, away):
    team = (away >= 0.5).astype(np.int8)
    if variant == "single":
        p_home = (1 - dead) * (1 - away)
        p_away = (1 - dead) * away
        return np.argmax(np.stack((p_home, p_away, dead), axis=-1), axis=-1).astype(np.int8)
    return np.where(dead >= 0.5, scores.D, team).astype(np.int8)


def accuracy(labels, dead, away, variant):
    valid = labels >= 0
    alive = valid & (labels != scores.D)
    predicted = decode_frame(variant, dead, away)
    return {
        "global_accuracy": float((predicted[valid] == labels[valid]).mean()),
        "dead_alive_accuracy": float(((predicted[valid] == scores.D) == (labels[valid] == scores.D)).mean()),
        "ha_accuracy": float(((away[alive] >= 0.5) == labels[alive]).mean()),
        "scored_frames": int(valid.sum()),
        "inplay_scored_frames": int(alive.sum()),
    }


@torch.inference_mode()
def validation(model, halves, refs, args):
    model.eval()
    labels, dead, away = [], [], []
    for x, padding, target, sizes in batches(halves, refs, args.batch, args.span, args.device):
        p_dead, p_away = model.probabilities(model(x, padding))
        for i, size in enumerate(sizes):
            labels.append(target[i, :size].cpu().numpy())
            dead.append(p_dead[i, :size].cpu().numpy())
            away.append(p_away[i, :size].cpu().numpy())
    return accuracy(np.concatenate(labels), np.concatenate(dead), np.concatenate(away), model.variant)


def write_json(path, value):
    temporary = path.with_suffix(".tmp")
    temporary.write_text(json.dumps(value, indent=2, allow_nan=False) + "\n")
    temporary.replace(path)


def train(args, variant, development, val):
    torch.manual_seed(args.seed)
    model = PossessionTCN(variant, args.width, args.players).to(args.device)
    optimizer = torch.optim.Adam(model.parameters(), lr=args.lr)
    val_refs = draw_blocks(val, args.span, 5000, np.random.default_rng(args.seed))
    history = []
    best = -np.inf
    best_path = args.output / f"{variant}.pt"
    for epoch in range(args.epochs):
        tick = perf_counter()
        model.train()
        loss_sum = 0.0
        refs = draw_blocks(development, args.span, args.samples, np.random.default_rng(args.seed + epoch))
        rate = args.lr * args.decay ** epoch
        for group in optimizer.param_groups:
            group["lr"] = rate
        for x, padding, target, _ in batches(development, refs, args.batch, args.span, args.device):
            optimizer.zero_grad(set_to_none=True)
            loss = model.loss(model(x, padding), target)
            loss.backward()
            optimizer.step()
            loss_sum += loss.item()
        result = validation(model, val, val_refs, args)
        item = {"epoch": epoch + 1, "lr": rate, "seconds": perf_counter() - tick,
                "loss": loss_sum / ((len(refs) + args.batch - 1) // args.batch), **result}
        history.append(item)
        write_json(args.output / f"{variant}_training.json", history)
        if result["global_accuracy"] > best:
            best = result["global_accuracy"]
            torch.save({"epoch": epoch + 1, "weights": model.state_dict()}, best_path)
        print(f"{variant} epoch {epoch+1}/{args.epochs} loss={item['loss']:.4f} "
              f"val_DHA={result['global_accuracy']:.4f} val_HA={result['ha_accuracy']:.4f} "
              f"seconds={item['seconds']:.1f}", flush=True)
    return float(sum(item["seconds"] for item in history))


@torch.inference_mode()
def predict(model, halves, args):
    model.eval()
    prediction = [np.zeros((len(half.sample_rows), 2), dtype=np.float32) for half in halves]
    refs = references(halves, args.span)
    for offset, (x, padding, _, sizes) in enumerate(batches(halves, refs, args.batch, args.span, args.device)):
        probs = torch.stack(model.probabilities(model(x, padding)), dim=-1).cpu().numpy()
        for j, ((index, first), size) in enumerate(zip(refs[offset*args.batch:(offset+1)*args.batch], sizes, strict=True)):
            prediction[index][first:first+size] = probs[j, :size]
    result = []
    for half, probability in zip(halves, prediction, strict=True):
        row = np.arange(len(half.half.frame))
        result.append(np.column_stack([np.interp(row, half.sample_rows, value) for value in probability.T]))
    return result


def evaluate(model, halves, condition, args):
    started = perf_counter()
    originals = [item.half for item in halves]
    result = predict(model, halves, args)
    labels = np.concatenate([half.labels for half in originals])
    dead, away = np.concatenate(result).T
    raw = [postprocess(half, p[:, 1], "raw") for half, p in zip(originals, result, strict=True)]
    smooth = [postprocess(half, p[:, 1], "prob35") for half, p in zip(originals, result, strict=True)]
    joint = [decode_frame(model.variant, *p.T) for p in result]
    runs, offset = [], 0
    for half in originals:
        runs.extend((offset + start, offset + stop) for start, stop in contiguous_runs(half.labels >= 0, half.frame))
        offset += len(half.labels)
    joint_f1 = scores.spell_scores(labels, np.concatenate(joint), runs)["mean_f1"]
    for half, p, end_to_end, raw_team, smooth_team in zip(originals, result, joint, raw, smooth, strict=True):
        np.savez_compressed(args.output / f"{model.variant}_{condition}_{half.key}.npz", frame=half.frame,
                            p_dead=p[:, 0], p_away=p[:, 1], ground_truth=half.labels,
                            joint=end_to_end, fixed_dead_raw=raw_team, fixed_dead_prob35=smooth_team)
    return {
        "condition": condition,
        **accuracy(labels, dead, away, model.variant),
        "joint_spellF1": joint_f1,
        "raw_fixed_dead": metrics(originals, raw),
        "prob35_fixed_dead": metrics(originals, smooth),
        "evaluation_seconds": perf_counter() - started,
    }


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--cache", type=Path, default=ROOT / "data/external/dfl")
    parser.add_argument("--stress-profile", type=Path)
    parser.add_argument("--device", default="cpu")
    parser.add_argument("--threads", type=int, default=4)
    parser.add_argument("--seed", type=int, default=2027)
    parser.add_argument("--train-matches", type=int, choices=(4, 5), default=4)
    parser.add_argument("--width", type=int, default=24)
    parser.add_argument("--span", type=int, default=128)
    parser.add_argument("--batch", type=int, default=8, help="Dense temporal blocks per batch.")
    parser.add_argument("--samples", type=int, default=100000, help="Approximate training targets per epoch.")
    parser.add_argument("--epochs", type=int, default=12)
    parser.add_argument("--players", choices=("all", "nearest5", "ball"), default="all")
    parser.add_argument("--lr", type=float, default=0.001)
    parser.add_argument("--decay", type=float, default=0.7)
    parser.add_argument("--variants", nargs="+", choices=VARIANTS, default=VARIANTS)
    args = parser.parse_args()
    if min(args.width, args.span, args.batch, args.epochs, args.samples, args.threads) < 1:
        parser.error("Width, span, batch, epochs, samples, and threads must be positive.")
    torch.set_num_threads(args.threads)
    args.output.mkdir(parents=True, exist_ok=True)
    started = perf_counter()
    split = split_matches(args.seed)
    development = split["development"]
    train_ids = development[:args.train_matches]
    val_ids = development[args.train_matches:]
    test_id = split["holdout"]
    manifest = {name: str(value) if isinstance(value, Path) else value for name, value in vars(args).items()}
    manifest.update(train=train_ids, validation=val_ids, test=test_id, input_fps=INPUT_FPS, window=WINDOW,
                    past_frames=LEFT, future_frames=RIGHT, torch_version=torch.__version__,
                    per_epoch_targets=args.span * max(1, args.samples // args.span))
    write_json(args.output / "manifest.json", manifest)
    print(f"Train: {train_ids} | Validation: {val_ids} | Test: {test_id} | {args.device}", flush=True)
    train_halves = [half for match in train_ids for half in load_match(match, args.cache)]
    val_halves = [half for match in val_ids for half in load_match(match, args.cache)]
    print(f"Prepared train and validation data in {perf_counter()-started:.1f}s", flush=True)
    times = {variant: train(args, variant, train_halves, val_halves) for variant in args.variants}
    del train_halves, val_halves

    conditions = {"clean": load_match(test_id, args.cache)}
    if args.stress_profile:
        profile = json.loads(args.stress_profile.read_text())
        conditions["ball_missing"] = load_match(test_id, args.cache, profile, args.seed)
        write_json(args.output / "stress_augmentation.json",
                   {half.half.key: half.half.augmentation for half in conditions["ball_missing"]})
    report = []
    for variant in args.variants:
        model = PossessionTCN(variant, args.width, args.players).to(args.device)
        checkpoint = torch.load(args.output / f"{variant}.pt", map_location=args.device, weights_only=True)
        model.load_state_dict(checkpoint["weights"])
        for name, halves in conditions.items():
            report.append({"variant": variant, "selected_epoch": checkpoint["epoch"], "fit_seconds": times[variant],
                           **evaluate(model, halves, name, args)})
            write_json(args.output / "results.json", report)
            print(json.dumps(report[-1], ensure_ascii=False), flush=True)
    print(f"Total time {perf_counter()-started:.1f}s", flush=True)


if __name__ == "__main__":
    main()
