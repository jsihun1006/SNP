# CLAUDE.md

Fork of the More than a Hack 2027 Phase 1 repo (ball possession, BAR-ATH). Task: one label per video
frame (`H` home/Barcelona, `A` away/Athletic, `D` dead ball, `X` abstain), 199,319 rows at 25 fps.
Ranked on `spellF1`, tie-break `macroF1`. Read README.md, CHALLENGE.md, DATA.md, EVALUATION.md,
SUBMISSION.md and RULES.md before changing anything; `evaluation/possession_metrics.py` is the
metric specification.

Team name: **VARcelona**. Final deliverables are named `VARcelona.csv` and `VARcelona.pdf` (or `.md`),
plus a public repo URL (see SUBMISSION.md).

## Git workflow
- Never commit or push to `main`. Work on a `feat/<topic>` branch and open a PR.
- Commit or open PRs only when the user asks.

## Do not touch
- `evaluation/` (scorer, validator, ground truth) and `games.py`. Both are the official contract.
- `data/` is gitignored match data under a licence. Never commit it, copy it elsewhere, or publish
  anything derived from it (frame-level exports, clips). Use `games.asset_paths()` for paths.
- Labels must come from an algorithm. No hand annotation, and no use of ground truth beyond the
  public 5 minutes.

## Environment
- System python has no pip/venv. Use `.venv/bin/python` (gitignored). Rebuild without sudo:
  `python3 -m venv --without-pip .venv`, bootstrap pip with get-pip.py, then
  `.venv/bin/pip install -r requirements-starter.txt` (kloppy is pinned to 3.18.0).
- Run scripts from the repo root, e.g. `.venv/bin/python evaluation/run_evaluation.py`.

## Layout
- `experiments/` : our work. One script per experiment, fixed parameters, one-line note of what
  changed in the docstring. `01_explore.py` is the Step 1 exploration.
- `evaluation/results/` : write submissions here. Only `baseline_*.csv` is tracked; other CSVs are
  gitignored on purpose.
- `starter/` : organiser loaders and the baseline. Reuse, do not edit.

## Workflow rules
- Score every change with `evaluation/run_evaluation.py`; report `spellF1`, `macroF1`, `poss` and
  `dur`, compared with the baselines.
- Run `evaluation/validate_submission.py <file>` on every output before calling it a submission.
- The public ground truth is 17 possessions (frames 11185-18684) and the ball is almost always
  visible there, so it does not exercise the missing-ball problem. Do not tune against it, and say
  when a gain is likely noise (spellF1 differences below ~0.04 are noise on the full match).
- Prefer football-sensible changes over ones that only move the public score.

## Data facts that matter
- ATD `frame_id - 1` = video frame. Coordinates are normalised 0-1; pitch is 105.3 x 68.0 m.
- 46 anonymous track slots that swap, so reason at team level, not per player.
- Ball missing on ~41% of graded frames, in long gaps (median ~11 s), concentrated in penalty areas
  and slow/still ball moments; short gaps are rare, so interpolation helps little.
- Graded frames: 11185-90651 and 110808-190197 only.
- Smart Tagging is inferred; `Team` is `N/A` for every throw-in and kick-off. Frame =
  `round(seconds * 25)`.
- Broadcast files are not frame-aligned to tactical; ignore them.

## Experiment backlog (not needed for possession yet)
- **Tactical camera motion:** is the tactical camera static or does it pan/zoom? Load the tactical
  homography (`starter/load_homography.py`), check `valid`, and measure how much the 3x3 matrices
  (or projected pitch corners in pixels) vary across graded frames. A varying matrix would mean the
  homography is useful for camera-motion cues; a constant one would mean it adds nothing beyond
  pixel conversion. Script: `experiments/00_homography_variation.py`.
- **Broadcast as a second ball source:** needs a broadcast-to-tactical frame alignment first (none
  is provided; only 65.8% of broadcast frames have a valid homography). Deferred.
