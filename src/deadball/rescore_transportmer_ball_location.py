"""Apply existing restart-location rules to the saved Model 1 holdout scores."""

from pathlib import Path
import sys

_SRC = Path(__file__).resolve().parents[1]
sys.path[:0] = [str(_SRC), str(_SRC.parent)]

import argparse
import hashlib
import json
from types import SimpleNamespace

import numpy as np
import pandas as pd
import torch

import compare_transportmer_folds as run
import dfl_labels
import dfl_load
import features
from rescore_deadball_location_rules import (
    SNAP_RADIUS,
    preparation_at_restart_location,
    restart_candidates,
    snap_restarts,
)
from compare_deadball_rule_signals import BOUNDARY_CUT
import simulate
from config import CONFIG


def metrics(truth, state, period):
    pm = run.prior.scoring.pm
    target = np.where(truth, pm.H, pm.D)
    estimate = np.where(state, pm.H, pm.D)
    runs = pm.valid_runs(target)
    actual = pm.possession_spells(target, runs)
    predicted = pm.possession_spells(estimate, runs)
    _, _, misses = pm.match_spells(pm.spell_overlaps(actual, predicted), len(actual), len(predicted), 0.5)
    false = state & ~truth
    return {
        **run.prior.spell_f1(truth, state, period),
        "inplay_false_dead_frames": int(false.sum()),
        "inplay_false_dead_spans": len(features.runs_of(false)),
        "entirely_false_dead_spells": sum(not truth[a:b].any() for a, b in features.runs_of(state)),
        "missed_dead_iou_0_5": misses,
    }


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--cache", type=Path, default=run.ROOT / "data/external/dfl")
    parser.add_argument("--device", choices=("mps", "cpu", "cuda"))
    args = parser.parse_args()
    torch.set_num_threads(4)
    source = json.loads((args.source / "manifest.json").read_text())
    match = source["splits"]["test_match"]
    weights = args.source / "ball_restorer.pt"
    if hashlib.sha256(weights.read_bytes()).hexdigest() != source["ball_checkpoint"]["sha256"]:
        raise ValueError("Ball restoration checkpoint differs from the original score run.")
    device = args.device or source["device"]
    prediction_args = SimpleNamespace(device=device, microbatch=source["microbatch"])
    imputer = run.TranSPORTmer("ball", motion_input=source["motion_input"]).to(device)
    imputer.load_state_dict(torch.load(weights, weights_only=True, map_location="cpu"))
    imputer.eval()
    movement, decode, settings = run.load_rules()
    if settings != source["rule_settings"] or list(run.prior.RULES) != source["rules"]:
        raise ValueError("Decoder rule settings differ from the original score run.")
    settings["boundary_cut"] = BOUNDARY_CUT
    CONFIG["dfl_cache"] = str(args.cache)
    tracking, _, meta = dfl_load.load_match(match)
    raw = features.match_from_dfl(tracking, meta)
    simulated = simulate.simulate(raw, seed=CONFIG["sim_seed"])
    pitch = np.asarray(raw.pitch, dtype=np.float32)
    index = dfl_load.MATCHES.index(match)

    outputs, rows, edits = {}, [], []
    for half in (1, 2):
        keep = raw.period == half
        path = args.source / "predictions" / f"{match}_H{half}_restored_none.csv"
        saved = pd.read_csv(path, float_precision="round_trip")
        np.testing.assert_array_equal(saved.frame_id, raw.frame[keep])
        np.testing.assert_array_equal(saved.truth_dead, dfl_labels.dead_label(tracking)[keep])
        prob = saved.dead_probability.to_numpy()
        truth, baseline = (saved[col].to_numpy(dtype=bool) for col in ("truth_dead", "after_rules"))
        ball = run.sample_ball(raw.ball[keep] / pitch,
                               np.random.default_rng(source["ball_mask_seed_test"] + 2*index + half))
        players = simulated.players[keep]
        test = run.arrays((players / pitch, ball, raw.team))
        restored = run.complete(imputer, test, prediction_args)
        game = run.correction_game(restored, raw.pitch, movement)
        base, _ = decode(prob, game, "original", run.prior.RULES, settings)
        np.testing.assert_array_equal(base, baseline, err_msg=f"Saved six-rule result, H{half}")

        observed = ball * pitch
        old = movement(players, raw.team, observed, raw.pitch)
        _, location = restart_candidates(observed, raw.pitch, old)
        prep = preparation_at_restart_location(observed, raw.pitch, old)
        boundary = np.column_stack((np.zeros(len(ball)), location.astype(float)))
        location_snap, changes = snap_restarts(base, location)
        for before, after in changes:
            edits.append({"half": half, "before_frame": int(saved.frame_id.iloc[before]),
                          "after_frame": int(saved.frame_id.iloc[after]), "shift_ms": (after-before)*40})
        methods = {"six_rules": base, "location_snap": location_snap}
        for name, options in (
            ("location_viterbi", {"boundary": boundary}),
            ("location_preparation", {"location_preparation": prep}),
            ("location_both", {"boundary": boundary, "location_preparation": prep}),
        ):
            methods[name], _ = decode(prob, game, "original", run.prior.RULES, settings, **options)
        for name, state in methods.items():
            rows.append({"half": half, "variant": name, **metrics(truth, state, raw.period[keep])})
        outputs[half] = saved.assign(location_restart=location, location_preparation_candidate=prep,
                                     **{name: state.astype(np.int8) for name, state in methods.items()})
        print(f"H{half}: original mask reproduced, {int(location.sum())} location restart frames, "
              f"{len(changes)} relocated boundaries", flush=True)

    args.output.mkdir(parents=True, exist_ok=False)
    manifest = {"source": str(args.source.resolve()), "match": match, "device": device,
                "settings": settings, "snap_radius_frames": SNAP_RADIUS,
                "location_input": "observed frames of the original simulated input",
                "weights_used": "saved ball imputer for baseline geometry only"}
    (args.output / "manifest.json").write_text(json.dumps(manifest, indent=2) + "\n")
    by_half = pd.DataFrame(rows)
    by_half.to_csv(args.output / "by_half.csv", index=False)
    summary = by_half.drop(columns="half").groupby("variant", sort=False).agg(
        {col: "mean" if col.endswith("_f1") else "sum" for col in by_half if col not in ("half", "variant")})
    summary.to_csv(args.output / "summary.csv")
    pd.DataFrame(edits, columns=["half", "before_frame", "after_frame", "shift_ms"]).to_csv(
        args.output / "snap_edits.csv", index=False)
    for half, data in outputs.items():
        data.to_csv(args.output / f"{match}_H{half}_predictions.csv", index=False)
    print(summary.to_string())


if __name__ == "__main__":
    main()
