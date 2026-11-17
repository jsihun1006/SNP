# model1_dead_ball — H/A/D possession labels (Phase 2)

D comes from the Phase 1 dead-ball model; H/A comes from a fixed-cost decoder over the nearest team to the ball.

## Input

| input | where |
|---|---|
| Phase 1 `dead_ball` | `outputs/phase1/dead_ball/<run_id>/features.csv` (`frame, graded, dead, dead_source`) |
| provided ATD | `data/tactical/` — nearest player to the ball per frame (`starter/baseline.nearest_player_labels`) |

No ground truth is read when predicting.

## Decision

- A three-state (H/A/D) minimum-cost sequence decoder (Viterbi). Observation costs accumulate per second,
  switch costs apply once per change (`config.json`, fixed priors declared before any evaluation).
- **Inside the graded halves, frames the dead-ball model calls dead may only be D; every other frame may only
  be H or A.** Dead-ball boundaries therefore come from Phase 1; this decoder decides only H↔A changes inside an
  in-play stretch, so a single nearest-player flip does not become a possession change.
- Outside the graded halves the decoder runs unconstrained, and those frames are written as `X` (before kickoff,
  halftime, after the final whistle: never scored).

## Run

```bash
python model/phase2_possession/model1_dead_ball/possession.py --dead-run submission --run-id submission
python model/phase2_possession/model1_dead_ball/validate.py --run-id submission     # organisers' validator
python model/phase2_possession/model1_dead_ball/evaluation.py --run-id submission   # public 5-minute check
python model/phase2_possession/model1_dead_ball/test_model.py                       # synthetic tests
```

## Output: `outputs/phase2/model1_dead_ball/<run_id>/` (git-ignored)

`prediction.csv` (`frame,label`, 199,319 rows), `frame_diagnostics.csv`, `possession_share.csv`, `run_manifest.json`.
`evaluation.py` writes `outputs/scores/model1_dead_ball/<run_id>/score.json`.

## Limitations

- H/A is the team of the player nearest to the ball; while the ball is missing inside an in-play stretch the
  switch cost carries the surrounding team.
- Dead-ball boundary errors are inherited from Phase 1.
- The public 5-minute score is a pipeline check, never a selection criterion.
