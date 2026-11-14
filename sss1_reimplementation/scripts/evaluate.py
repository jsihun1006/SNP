import argparse
import json
from pathlib import Path
from ..evaluation.official import evaluate


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--gt", type=Path, required=True)
    p.add_argument("--predictions", type=Path, required=True)
    p.add_argument("--output", type=Path, required=True)
    p.add_argument("--score-threshold", type=float, default=.25)
    p.add_argument("--detection-only", action="store_true")
    a = p.parse_args()
    report = evaluate(a.gt, a.predictions, a.score_threshold, not a.detection_only)
    # COCO -1 denotes an absent area bin; official NaN is retained as null with a clear note.
    import math
    def finite(value):
        if isinstance(value, dict): return {k:finite(v) for k,v in value.items()}
        if isinstance(value, list): return [finite(v) for v in value]
        if isinstance(value, float) and not math.isfinite(value): return None
        return value
    report["conventions"] = "AP in [0,1]; -1=unavailable COCO bin; null=undefined. No paper scores inserted."
    a.output.parent.mkdir(parents=True, exist_ok=True)
    a.output.write_text(json.dumps(finite(report), ensure_ascii=False, allow_nan=False, indent=2), encoding="utf-8")


if __name__ == "__main__":
    main()
