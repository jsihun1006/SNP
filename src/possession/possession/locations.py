"""Research workspace paths for the frozen possession inference pipeline."""
from pathlib import Path

PACKAGE = Path(__file__).resolve().parent
REPO = PACKAGE.parents[2]
ARTIFACT = REPO / "artifacts/possession/pathcrf/candidate25_dfl"
PARAMETERS = REPO / "experiments/possession/pathcrf/candidate25_dfl/inference_parameters.json"
PATHCRF = REPO / "vendor/pathcrf"
