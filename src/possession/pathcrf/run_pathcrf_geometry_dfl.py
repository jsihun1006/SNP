"""Add ball proximity to the frozen PathCRF Viterbi transition scores on DFL."""

from __future__ import annotations

import argparse
import hashlib
from importlib import import_module
from pathlib import Path
import subprocess
import sys
from time import perf_counter
from unittest.mock import patch

import numpy as np
import pandas as pd
import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from run_pathcrf_dfl import (
    CHECKPOINT, ROOT, TRIAL, checkpoint_split, load_match, load_official_tracking,
    prepare_official_input, scores, team_predictions, write_json,
)
from transportmer_ha_data import contiguous_runs
from transportmer_ha_metrics import metrics

UPSTREAM_REV = "1b7c6f85a6b78c92e4543fde72ca2b86a933f997"
MAX_DISTANCE_M = 1.5
MIN_TEAM_MARGIN_M = 0.75
MAX_HEIGHT_M = 1.5
CONFIRM_FRAMES = 5
RADIUS_FRAMES = 10
WEIGHTS = (0.5, 1.0, 2.0, 4.0, 8.0)
MODES = ("xy", "xyz")


def contact_evidence(frame, distances, height, node_team):
    """Use the first frame of a sustained, unambiguous close approach as evidence."""
    home = distances[:, node_team == scores.H].min(axis=1)
    away = distances[:, node_team == scores.A].min(axis=1)
    nearest = distances.argmin(axis=1)
    with np.errstate(invalid="ignore"):
        close = (np.minimum(home, away) <= MAX_DISTANCE_M) & (np.abs(home - away) >= MIN_TEAM_MARGIN_M)
    close &= np.isfinite(home) & np.isfinite(away)
    low = np.isfinite(height) & (height >= 0) & (height <= MAX_HEIGHT_M)
    for mode, eligible in (("xy", close), ("xyz", close & low)):
        nodes = np.where(eligible, nearest, -1)
        cuts = np.r_[0, np.flatnonzero((np.diff(nodes) != 0) | (np.diff(frame) != 1)) + 1, len(frame)]
        candidates = [
            (int(start), int(nodes[start]))
            for start, stop in zip(cuts[:-1], cuts[1:], strict=True)
            if nodes[start] >= 0 and stop - start >= CONFIRM_FRAMES
        ]
        support = np.zeros(distances.shape, dtype=np.float32)
        segments = np.r_[0, np.cumsum(np.diff(frame) != 1)]
        for start, node in candidates:
            rows = np.arange(max(0, start-RADIUS_FRAMES), min(len(frame), start+RADIUS_FRAMES+1))
            rows = rows[segments[rows] == segments[start]]
            if mode == "xyz":
                # The delivery may be airborne just before reception. Only reward
                # receiving states on a low-ball observation.
                rows = rows[low[rows]]
            weight = 1 - np.abs(frame[rows]-frame[start]) / (RADIUS_FRAMES+1)
            support[rows, node] = np.maximum(support[rows, node], weight)
        yield mode, support, candidates


@torch.no_grad()
def decode_with_geometry(crf, emissions, embeddings, support_by_variant, node_team):
    """Decode each variant on the official sparse CRF graph."""
    inc = crf.inc_idx.clamp_min(0).cpu().numpy()
    allowed = crf.inc_mask.cpu().numpy()
    source = crf.comp_src.cpu().numpy()
    team = node_team[source]
    receiving = (team[:, None] >= 0) & (team[inc] >= 0) & (team[:, None] != team[inc])

    # The released CRF uses a single Linear on concatenated edge embeddings.
    # Its transition score is the sum of the old and new edge projections.
    old_weights, new_weights = crf.linear.weight.split(crf.edge_embed_dim, dim=1)
    old = torch.nn.functional.linear(embeddings[0], old_weights).squeeze(-1).cpu().numpy()
    new = torch.nn.functional.linear(embeddings[0], new_weights, crf.linear.bias).squeeze(-1).cpu().numpy()
    emission = emissions[0].cpu().numpy()
    steps, states = emission.shape
    variants = len(support_by_variant)
    previous = np.broadcast_to(emission[0], (variants, states)).copy()
    backpointers = np.empty((steps, variants, states), dtype=np.int16)
    members = np.arange(variants)[:, None, None]
    edges = np.arange(states)[None, :, None]
    support = support_by_variant[:, :, source]

    for step in range(1, steps):
        transition = old[step-1, inc] + new[step, :, None]
        value = previous[:, inc] + transition + support[:, step, :, None] * receiving
        value = np.where(allowed, value, crf.forbid_score)
        best = value.argmax(axis=2)[..., None]
        previous = emission[step] + value[members, edges, best].squeeze(-1)
        backpointers[step] = inc[edges, best].squeeze(-1)
    path = np.empty((variants, steps), dtype=np.int64)
    path[:, -1] = previous.argmax(axis=1)
    for step in range(steps-1, 0, -1):
        path[:, step-1] = backpointers[step, np.arange(variants), path[:, step]]
    return path


class GeometryInference:
    """Reuse the official encoder and window blending, intercepting episode decoding."""

    def __init__(self, upstream, tracking, scale, variants):
        self.upstream = upstream
        self.episode = upstream.inference_episode
        self.pad = upstream._pad_nodes
        self.tracking = tracking.set_index("frame_id", verify_integrity=True)
        self.scale = scale
        self.variants = variants
        self.chunks = {name: [] for name, _, _ in variants}
        self.candidates = {mode: [] for mode in MODES}
        self.checked_episodes = 0

    def infer_episode(self, *, model, ep_tracking, sample_freq, **kwargs):
        padded_order = None
        official_decode = model.crf.decode
        frame = ep_tracking.index.to_numpy()

        def capture_order(*args, **options):
            nonlocal padded_order
            padded, mask, padded_order = self.pad(*args, **options)
            return padded, mask, padded_order

        def decode(emissions, embeddings, *, last_node_idx):
            if padded_order is None or last_node_idx is not None:
                raise RuntimeError("Unexpected episode decode arguments from the pinned upstream")
            reference = official_decode(emissions, embeddings, last_node_idx=None)
            down = np.arange(0, len(frame), sample_freq)
            if down[-1] != len(frame)-1:
                down = np.append(down, len(frame)-1)
            if len(down) != emissions.shape[1]:
                raise ValueError("Geometry and PathCRF score lengths disagree")
            names = np.asarray(padded_order)
            node_team = np.array([
                scores.H if name.startswith("home_") else scores.A if name.startswith("away_") else -1
                for name in names
            ])
            raw = self.tracking.loc[frame]
            cols = [f"{name}_{axis}" for name in names for axis in ("x", "y")]
            players = raw.reindex(columns=cols).to_numpy().reshape(len(frame), len(names), 2)
            delta = (players - raw[["ball_x", "ball_y"]].to_numpy()[:, None, :]) * self.scale
            distances = np.linalg.norm(delta, axis=2)
            distances[~np.isfinite(distances)] = np.inf
            support = {}
            for mode, evidence, candidates in contact_evidence(frame, distances, raw["ball_z"].to_numpy(), node_team):
                support[mode] = evidence[down]
                self.candidates[mode].extend((int(frame[row]), int(node_team[node])) for row, node in candidates)
            no_bonus = np.zeros_like(support["xy"])
            variants = np.stack([no_bonus] + [support[mode] * weight for _, mode, weight in self.variants])
            paths = decode_with_geometry(model.crf, emissions, embeddings, variants, node_team)
            np.testing.assert_array_equal(paths[0], reference[0].cpu().numpy())
            self.checked_episodes += 1

            src, dst = (indices.cpu().numpy() for indices in (model.comp_src, model.comp_dst))
            for (name, _, _), path in zip(self.variants, paths[1:], strict=True):
                states = pd.DataFrame({"edge_src": names[src[path]], "edge_dst": names[dst[path]]}, index=frame[down])
                self.chunks[name].append(states.reindex(frame).ffill())
            return reference

        with patch.object(self.upstream, "_pad_nodes", capture_order), patch.object(model.crf, "decode", decode):
            return self.episode(model=model, ep_tracking=ep_tracking, sample_freq=sample_freq, **kwargs)

    def run(self, model, model_input, trial_args):
        with patch.object(self.upstream, "inference_episode", self.infer_episode):
            _, _, original, _ = self.upstream.inference(
                model, model_input, use_crf=True, ball_feature=False,
                evaluate=False, correct_episode_lasts=False,
                window_seconds=trial_args["window_seconds"], sample_freq=round(25 / trial_args["target_fps"]),
            )
        return {"pathcrf": original} | {name: pd.concat(parts, verify_integrity=True) for name, parts in self.chunks.items()}


def candidate_audit(halves, base, candidates):
    """Count candidates that contradict base and match any true transition in ±1s."""
    candidate_frames = dict(candidates)
    total = matched = 0
    for half, pred in zip(halves, base, strict=True):
        for start, stop in contiguous_runs(half.alive & (half.labels >= 0), half.frame):
            truth_changes = scores.transitions(half.labels, [(start, stop)])
            proposals = [
                (i, int(pred[i]), candidate_frames[frame])
                for i, frame in enumerate(half.frame[start+1:stop], start+1)
                if frame in candidate_frames and pred[i-1] == pred[i] == 1-candidate_frames[frame]
            ]
            matches, _, _ = scores.match_transitions(truth_changes, proposals, 25, typed="pair")
            total += len(proposals)
            matched += len(matches)
    return {"candidates": total, "matched": matched, "unmatched": total-matched,
            "precision": matched / total if total else None}


def run_match(match_id, args, upstream, model, trial_args, variants):
    from datatools.config import PITCH_X, PITCH_Y

    started = perf_counter()
    print(f"{match_id}: preparing tracking", flush=True)
    tracking = load_official_tracking(args.raw, match_id)
    model_input, raw_state = prepare_official_input(tracking)
    halves = [item.half for item in load_match(match_id, args.cache)]
    for half in halves:
        np.testing.assert_array_equal(raw_state.reindex(half.frame).eq("alive"), half.alive)
    if halves[0].pitch != halves[1].pitch:
        raise ValueError("Pitch dimensions differ between halves")
    scale = np.asarray(halves[0].pitch) / [PITCH_X, PITCH_Y]
    runner = GeometryInference(upstream, tracking, scale, variants)
    ready = perf_counter()
    print(f"{match_id}: prepared in {ready-started:.1f}s, running encoder and Viterbi variants", flush=True)
    decoded = runner.run(model, model_input, trial_args)
    inferred = perf_counter()
    predictions = {}
    all_scores = {}
    for name, edges in decoded.items():
        predictions[name] = [team_predictions(half, edges)[0] for half in halves]
        for half, pred in zip(halves, predictions[name], strict=True):
            np.testing.assert_array_equal(pred == scores.D, ~half.alive)
        all_scores[name] = metrics(halves, predictions[name])
        count = sum((half.alive & (half.labels >= 0)).sum() for half in halves)
        hits = sum(((half.labels == pred) & half.alive).sum() for half, pred in zip(halves, predictions[name], strict=True))
        all_scores[name]["ha_frame_accuracy"] = float(hits / count)
    height = tracking.loc[tracking["frame_id"].isin(np.concatenate([half.frame[half.alive] for half in halves])), "ball_z"]
    result = {
        "match": match_id, "metrics": all_scores,
        "candidate_audit": {mode: candidate_audit(halves, predictions["pathcrf"], runner.candidates[mode]) for mode in MODES},
        "height": {"inplay_frames": len(height), "observed": int(np.isfinite(height).sum()),
                   "above_threshold": int((height > MAX_HEIGHT_M).sum())},
        "zero_bonus_equivalence_checked_episodes": runner.checked_episodes,
        "prepare_seconds": ready-started, "inference_seconds": inferred-ready,
        "total_seconds": perf_counter()-started,
    }
    return result, halves, predictions


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--upstream", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--raw", type=Path, default=ROOT / "data/external/dfl/raw")
    parser.add_argument("--cache", type=Path, default=ROOT / "data/external/dfl")
    parser.add_argument("--threads", type=int, default=4)
    args = parser.parse_args()
    upstream_path = args.upstream.resolve()
    commit = subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=upstream_path, text=True).strip()
    if commit != UPSTREAM_REV:
        parser.error(f"The decoder adapter requires official PathCRF commit {UPSTREAM_REV}")
    sys.path.insert(0, str(upstream_path))
    upstream = import_module("inference")
    from models.utils import build_model, load_trial_args

    torch.set_num_threads(args.threads)
    trial_path = upstream_path / f"saved/{TRIAL}"
    trial_args = load_trial_args(str(trial_path))
    split = checkpoint_split(upstream_path, trial_args)
    checkpoint = trial_path / "model" / CHECKPOINT
    args.output.mkdir(parents=True, exist_ok=True)
    manifest = {
        "upstream_commit": commit, "checkpoint_trial": TRIAL, "checkpoint_name": CHECKPOINT,
        "checkpoint_sha256": hashlib.sha256(checkpoint.read_bytes()).hexdigest(), "seed": trial_args["seed"],
        "split": split, "device": "cpu", "threads": args.threads, "deadball_mask": "DFL ground truth",
        "geometry": {"max_distance_m": MAX_DISTANCE_M, "min_team_margin_m": MIN_TEAM_MARGIN_M,
                     "min_close_frames": CONFIRM_FRAMES, "radius_frames": RADIUS_FRAMES, "max_height_m": MAX_HEIGHT_M},
        "encoder_fps": trial_args["target_fps"], "geometry_and_evaluation_fps": 25,
        "weight_grid": WEIGHTS,
        "selection": "same positive weight for XY and XYZ; maximize mean validation spellF1; tie chooses smaller weight",
        "bonus": "triangular proximity evidence at receiving player in a legal inter-team CRF transition",
    }
    write_json(args.output / "manifest.json", manifest)
    model = build_model(trial_args, torch.device("cpu"))
    model.load_state_dict(torch.load(checkpoint, map_location="cpu", weights_only=True))
    grid = [(f"{mode}_{weight:g}", mode, weight) for weight in WEIGHTS for mode in MODES]
    validation, _, _ = run_match(split["validation"], args, upstream, model, trial_args, grid)
    mean_score = {weight: float(np.mean([validation["metrics"][f"{mode}_{weight:g}"]["spellF1"] for mode in MODES]))
                  for weight in WEIGHTS}
    chosen = max(WEIGHTS, key=lambda weight: (mean_score[weight], -weight))
    validation.update({"mean_spellF1_by_weight": mean_score, "selected_weight": chosen})
    write_json(args.output / "validation.json", validation)
    manifest["selected_weight"] = chosen
    write_json(args.output / "manifest.json", manifest)
    print(f"Validation selected common XY/XYZ weight {chosen:g}; evaluating {split['test']}", flush=True)

    test, halves, predictions = run_match(
        split["test"], args, upstream, model, trial_args, [(mode, mode, chosen) for mode in MODES],
    )
    test["selected_weight"] = chosen
    for i, half in enumerate(halves):
        pd.DataFrame({
            "frame": half.frame, "truth": half.labels, "inplay": half.alive,
            **{name: pred[i] for name, pred in predictions.items()},
        }).to_parquet(args.output / f"{half.key}.parquet", index=False)
    write_json(args.output / "results.json", test)
    for name, result in test["metrics"].items():
        print(f"{name}: spellF1 {result['spellF1']:.6f}, "
              f"fake {result['spurious_transitions']}, missed {result['missed_transitions']}", flush=True)


if __name__ == "__main__":
    main()
