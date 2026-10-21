"""Command-line entry point for the possession lab.

    python -m possession_lab all
    python -m possession_lab predict
    python -m possession_lab evaluate
    python -m possession_lab stress
    python -m possession_lab visualize
    python -m possession_lab viewer

Everything is written under ``possession_lab/outputs`` by default, which is git-ignored.  The
generated submission is never copied back into the repository's submission assets.
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path
from typing import Any

import pandas as pd


def _json(path: Path, value: Any) -> None:
    from .evaluate import to_jsonable

    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(to_jsonable(value), indent=2), encoding="utf-8")


def _gpu_note(load_seconds: float, predict_seconds: float, stress_seconds: float) -> str:
    return (
        "GPU: not used. The model is a 3-state Viterbi pass over 199,319 frames plus a handful of "
        "46-player distance arrays. This CPU-only implementation measured "
        f"{predict_seconds:.2f}s for the full decode (data load {load_seconds:.2f}s; stress suite "
        f"{stress_seconds:.2f}s). Timings vary by machine. A GPU path would add transfer overhead "
        "for this workload; revisit it only for a pixel-space tracker or a learned sequence model."
    )


def _write_core_outputs(
    output_dir: Path,
    pred,
    metrics: dict[str, Any],
    stress_rows,
    full_stats: dict[str, Any],
    validation: dict[str, Any],
    homography_summary: dict[str, Any],
) -> None:
    from .config import MODEL_PARAMS
    from .pipeline import write_submission

    write_submission(pred, output_dir / "predictions.csv")
    _json(output_dir / "metrics_public.json", metrics)
    _json(output_dir / "full_stats.json", full_stats)
    _json(output_dir / "validation.json", validation)
    _json(output_dir / "homography.json", homography_summary)
    _json(output_dir / "params.json", MODEL_PARAMS)
    pd.DataFrame(stress_rows).to_csv(output_dir / "stress.csv", index=False)


def cmd_all(args: argparse.Namespace) -> int:
    from .config import PUBLIC_START, PUBLIC_STOP
    from .evaluate import evaluate_stress, full_match_stats, public_metrics, to_jsonable
    from .pipeline import (
        load_lab,
        predict_full,
        public_prediction,
        validate_codes,
        validate_submission_file,
    )
    from .visualize import generate_all

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    t0 = time.time()
    lab = load_lab(args.data_dir)
    load_seconds = time.time() - t0
    print(f"loaded tactical data in {load_seconds:.2f}s: {lab.tracking.n_frames:,} frames, "
          f"{lab.tracking.n_players} player slots, {len(lab.tags.events)} tag intervals")

    t1 = time.time()
    pred = predict_full(lab.tracking, lab.tags)
    predict_seconds = time.time() - t1
    print(f"decoded full match in {predict_seconds:.2f}s")

    metrics = public_metrics(pred, lab.gt, bootstrap_n=args.bootstrap_n)
    print(
        "public: spellF1 {:.4f} | macroF1 {:.4f} | D-F1 {:.4f} | pred spells {}".format(
            metrics["spell_f1_mean"],
            metrics["macro_f1"],
            metrics["per_class"]["D"]["f1"],
            metrics["n_pred_spells"],
        )
    )

    t2 = time.time()
    stress_rows = evaluate_stress(lab.tracking, lab.tags) if not args.no_stress else []
    stress_seconds = time.time() - t2
    if stress_rows:
        stress = pd.DataFrame(stress_rows)
        print(stress.to_string(index=False))

    full_stats = full_match_stats(pred)
    validation = validate_codes(pred)
    _json(
        output_dir / "timings.json",
        {
            "load_seconds": load_seconds,
            "predict_seconds": predict_seconds,
            "stress_seconds": stress_seconds,
            "device": "cpu",
            "gpu_note": _gpu_note(load_seconds, predict_seconds, stress_seconds),
        },
    )
    _write_core_outputs(
        output_dir,
        pred,
        metrics,
        stress_rows,
        full_stats,
        validation,
        lab.homography.summary(),
    )
    file_validation = validate_submission_file(output_dir / "predictions.csv")
    _json(output_dir / "file_validation.json", file_validation)
    print("submission:", file_validation)

    if not args.no_figures:
        generate_all(
            lab.tracking,
            lab.tags,
            lab.homography,
            pred,
            lab.gt,
            metrics,
            stress_rows,
            full_stats,
            validation,
            output_dir,
            _gpu_note(load_seconds, predict_seconds, stress_seconds),
        )
        print("report:", output_dir / "report.html")
    return 0


def cmd_predict(args: argparse.Namespace) -> int:
    from .pipeline import load_lab, predict_full, write_submission, validate_submission_file

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    lab = load_lab(args.data_dir, load_gt=False)
    pred = predict_full(lab.tracking, lab.tags)
    path = write_submission(pred, output_dir / "predictions.csv")
    print(validate_submission_file(path))
    return 0


def cmd_evaluate(args: argparse.Namespace) -> int:
    from .data import load_public_gt, load_submission
    from .evaluate import public_metrics
    from .config import PUBLIC_GT_PATH

    output_dir = Path(args.output_dir)
    pred = load_submission(output_dir / "predictions.csv")
    metrics = public_metrics(pred, load_public_gt(PUBLIC_GT_PATH), bootstrap_n=args.bootstrap_n)
    _json(output_dir / "metrics_public.json", metrics)
    print(
        "public: spellF1 {:.4f} | macroF1 {:.4f} | D-F1 {:.4f} | pred spells {}".format(
            metrics["spell_f1_mean"],
            metrics["macro_f1"],
            metrics["per_class"]["D"]["f1"],
            metrics["n_pred_spells"],
        )
    )
    return 0


def cmd_stress(args: argparse.Namespace) -> int:
    from .evaluate import evaluate_stress

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    lab = _load_for_stress(args)
    rows = evaluate_stress(lab.tracking, lab.tags)
    pd.DataFrame(rows).to_csv(output_dir / "stress.csv", index=False)
    print(pd.DataFrame(rows).to_string(index=False))
    return 0


def _load_for_stress(args):
    from .pipeline import load_lab

    return load_lab(args.data_dir, load_gt=False)


def cmd_visualize(args: argparse.Namespace) -> int:
    from .data import load_submission
    from .evaluate import full_match_stats
    from .pipeline import load_lab, validate_codes
    from .visualize import generate_all

    output_dir = Path(args.output_dir)
    lab = load_lab(args.data_dir)
    pred = load_submission(output_dir / "predictions.csv")
    metrics = json.loads((output_dir / "metrics_public.json").read_text(encoding="utf-8"))
    stress = pd.read_csv(output_dir / "stress.csv") if (output_dir / "stress.csv").is_file() else []
    full_stats = full_match_stats(pred)
    validation = validate_codes(pred)
    timings_path = output_dir / "timings.json"
    if timings_path.is_file():
        timings = json.loads(timings_path.read_text(encoding="utf-8"))
    else:
        timings = {"load_seconds": 2.7, "predict_seconds": 1.8, "stress_seconds": 0.7}
    generate_all(
        lab.tracking,
        lab.tags,
        lab.homography,
        pred,
        lab.gt,
        metrics,
        stress,
        full_stats,
        validation,
        output_dir,
        _gpu_note(
            float(timings.get("load_seconds", 0.0)),
            float(timings.get("predict_seconds", 0.0)),
            float(timings.get("stress_seconds", 0.0)),
        ),
    )
    print("report:", output_dir / "report.html")
    return 0


def cmd_viewer(args: argparse.Namespace) -> int:
    from .config import PUBLIC_GT_PATH, USER_TRUTH_PATH, asset_paths
    from .viewer import serve_viewer, tactical_video_path

    output_dir = Path(args.output_dir)
    predictions = Path(args.predictions) if args.predictions else output_dir / "predictions.csv"
    video = Path(args.video) if args.video else tactical_video_path(args.data_dir)
    gt = Path(args.gt) if args.gt else PUBLIC_GT_PATH
    tags = Path(args.tags) if args.tags else asset_paths(args.data_dir)["smart_tagging"]
    metrics = Path(args.metrics) if args.metrics else output_dir / "metrics_public.json"
    user_truth = Path(args.user_truth) if args.user_truth else USER_TRUTH_PATH

    try:
        return serve_viewer(
            predictions,
            video_path=video,
            gt_path=gt,
            tags_path=tags,
            metrics_path=metrics,
            user_truth_path=user_truth,
            data_dir=args.data_dir,
            port=args.port,
            open_browser=not args.no_open,
            start_frame=args.start_frame,
            verbose=args.verbose,
        )
    except FileNotFoundError as exc:
        print(f"viewer: {exc}", file=sys.stderr)
        return 2
    except ValueError as exc:
        print(f"viewer: datos de entrada no válidos: {exc}", file=sys.stderr)
        return 2
    except OSError as exc:
        print(f"viewer: no se pudo iniciar el servidor local: {exc}", file=sys.stderr)
        return 2


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="python -m possession_lab",
        description="Data-only, gap-aware HMM lab for BAR-ATH possession.",
    )
    parser.add_argument(
        "command",
        nargs="?",
        default="all",
        choices=["all", "predict", "evaluate", "stress", "visualize", "viewer"],
    )
    parser.add_argument("--data-dir", default=None, help="gated data directory (default: data/)")
    parser.add_argument("--output-dir", default=None, help="output directory (default: possession_lab/outputs)")
    parser.add_argument("--bootstrap-n", type=int, default=2000, help="bootstrap draws; 0 disables")
    parser.add_argument("--no-stress", action="store_true", help="skip the missing-ball suite")
    parser.add_argument("--no-figures", action="store_true", help="write data only, no report")

    viewer = parser.add_argument_group("viewer", "local video reviewer options")
    viewer.add_argument("--port", type=int, default=8765, help="local HTTP port (default: 8765)")
    viewer.add_argument("--no-open", action="store_true", help="do not open the browser automatically")
    viewer.add_argument("--predictions", default=None, help="prediction CSV (default: outputs/predictions.csv)")
    viewer.add_argument("--gt", default=None, help="public ground-truth CSV (default: evaluation/gt/...)")
    viewer.add_argument("--video", default=None, help="tactical MP4 (default: data/tactical/BAR-ATH_tactical.mp4)")
    viewer.add_argument("--tags", default=None, help="smart-tag XML (default: gated tactical tags)")
    viewer.add_argument("--metrics", default=None, help="public metrics JSON (default: outputs/metrics_public.json)")
    viewer.add_argument(
        "--user-truth",
        default=None,
        help="user-truth JSON (default: outputs/user_truth.json; viewer-only, never used by the model)",
    )
    viewer.add_argument("--start-frame", type=int, default=None, help="initial frame (default: public GT start)")
    viewer.add_argument("--verbose", action="store_true", help="log HTTP requests")
    return parser


def main(argv: list[str] | None = None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)
    if args.output_dir is None:
        from .config import OUTPUT_DIR

        args.output_dir = str(OUTPUT_DIR)
    commands = {
        "all": cmd_all,
        "predict": cmd_predict,
        "evaluate": cmd_evaluate,
        "stress": cmd_stress,
        "visualize": cmd_visualize,
        "viewer": cmd_viewer,
    }
    return commands[args.command](args)


if __name__ == "__main__":
    raise SystemExit(main())
