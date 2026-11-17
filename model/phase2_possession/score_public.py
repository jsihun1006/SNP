"""Score a Phase 2 prediction on the published 5 minutes with the official scorer (pipeline check only).

    python model/phase2_possession/score_public.py --model <model> --run-id submission

The public 5 minutes check frames, format and label meaning. No setting may be chosen, fitted or tuned on it.
Output: outputs/scores/<model>/<run_id>/score.json
"""
import argparse
import json
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[2]
sys.path[:0] = [str(ROOT), str(ROOT / "evaluation")]
import possession_metrics as pm  # noqa: E402
import run_evaluation as scorer  # noqa: E402


def main():
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--model", required=True)
    parser.add_argument("--run-id", required=True)
    args = parser.parse_args()
    prediction = ROOT / "outputs" / "phase2" / args.model / args.run_id / "prediction.csv"
    gt_path = ROOT / "evaluation" / "gt" / "BAR-ATH_possession_gt_public.csv"
    gt = pm.label_codes(pm.read_label_csv(str(gt_path)))
    result, reason = scorer.score_submission(str(prediction), gt, pm.TAUS, 1000)
    if reason:
        raise SystemExit(reason)
    dest = ROOT / "outputs" / "scores" / args.model / args.run_id
    dest.mkdir(parents=True, exist_ok=True)
    report = scorer.json_safe(result)
    report.update(submission=str(prediction), ground_truth=gt_path.name, note="public 5-minute pipeline check")
    (dest / "score.json").write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
    print("public 5-minute pipeline check written to {}".format(dest.relative_to(ROOT)))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
