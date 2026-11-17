"""Template for a new possession model that sits on top of the dead ball (copy this folder and rename it).

    python model/phase2_possession/<model>/possession.py --dead-run submission --run-id submission

Only predict_team() is model-specific. Everything else is shared (../overlay.py):
  - the dead ball is read from Phase 1 and always wins (D) inside the graded halves,
  - predict_team() must give 'H' or 'A' on every graded frame the dead ball calls in play
    ('' is allowed on dead or ungraded frames),
  - prediction.csv is written to outputs/phase2/<model>/<run_id>/ and checked with the organisers' validator.
Rules: train and tune on external data only; never read the ground truth or the video here.
"""

import argparse
from pathlib import Path
import sys

import numpy as np

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parent))
import overlay  # noqa: E402

NAME = HERE.name


def predict_team(match, features):
    """Team in possession per frame: array of N_FRAMES values 'H', 'A' or ''.

    match     features.Match of the provided match (overlay.load_match): players (n, slots, 2) m,
              team (slots,) 0 = H / 1 = A, ball (n, 2) m (NaN when missing), period (0 outside graded halves)
    features  Phase 1 dead-ball table: frame, graded, dead, dead_source (dead = 1 means D)
    """
    raise NotImplementedError("write the possession model here (see the docstring), or copy a model that has one")


def main():
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--dead-run", help="dead-ball run id (default: latest)")
    parser.add_argument("--run-id", help="run id of this prediction (default: same as the dead-ball run)")
    args = parser.parse_args()
    features, dead_run = overlay.load_dead_ball(args.dead_run)
    match, _ = overlay.load_match()
    team = np.asarray(predict_team(match, features), dtype=object)
    overlay.write_run(NAME, args.run_id or dead_run.name, team, features, dead_run, code=__file__)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
