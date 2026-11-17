"""Score prediction.csv on the published 5 minutes with the official scorer (pipeline check only).

    python model/phase2_possession/model1_dead_ball/evaluation.py                 # latest run
    python model/phase2_possession/model1_dead_ball/evaluation.py --run-id submission

The public 5 minutes are used only to check that frames, format and label meaning are right. No setting
in this repository was chosen, fitted or tuned on this score.
Output: outputs/scores/model1_dead_ball/<run_id>/score.json
"""
import argparse
import json
from pathlib import Path
import sys

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[2]
sys.path[:0] = [str(ROOT), str(ROOT / "evaluation"), str(HERE)]
import possession_metrics as pm  # noqa: E402
import run_evaluation as scorer  # noqa: E402
from validate import run_dir  # noqa: E402


def main():
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--run-id")
    args = parser.parse_args()
    out = run_dir(args.run_id)
    gt_path = ROOT / "evaluation" / "gt" / "BAR-ATH_possession_gt_public.csv"
    gt = pm.label_codes(pm.read_label_csv(str(gt_path)))
    result, reason = scorer.score_submission(str(out / "prediction.csv"), gt, pm.TAUS, 1000)
    if reason:
        raise SystemExit(reason)
    dest = ROOT / "outputs" / "scores" / "model1_dead_ball" / out.name
    dest.mkdir(parents=True, exist_ok=True)
    report = scorer.json_safe(result)
    report.update(submission=str(out / "prediction.csv"), ground_truth=gt_path.name, note="public 5-minute pipeline check")
    (dest / "score.json").write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
    print("spellF1 {:.4f}  macroF1 {:.4f}  accuracy {:.4f}  ->  {}".format(
        result["spell_f1_mean"], result["macro_f1"], result["accuracy"], dest.relative_to(ROOT)))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
