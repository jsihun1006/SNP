# model2_template — a possession model on top of the dead ball (Phase 2)

Copy this folder to a new name (for example `model2_<name>`), write `predict_team()` in `possession.py`,
and run the submission with it:

```bash
python run_submission.py --possession model2_<name>
```

## Order

1. **Dead ball first** (Phase 1, `model/phase1_extraction/dead_ball`) → `outputs/phase1/dead_ball/<run_id>/features.csv`.
2. **Possession on top**: `predict_team()` gives `H` or `A` per frame; the shared overlay (`../overlay.py`) puts
   the dead ball over it.

| frame | label |
|---|---|
| graded, dead ball says dead | `D` (always; the possession model cannot override it) |
| graded, dead ball says in play | the model's `H`/`A` (**required on every such frame**) |
| outside the graded halves (before kickoff, halftime, after the final whistle) | `X` (never scored) |

## Input given to `predict_team(match, features)`

| argument | content |
|---|---|
| `match` | the provided match read exactly as the dead-ball step reads it (`overlay.load_match()`): `players` (n, 46, 2) m from the ATD, `team` per slot (0 = H, 1 = A), `ball` (n, 2) m — the **corrected ball** of the ball-tracking checkpoint on the Tactical frames it maps, ATD ball elsewhere (`CONFIG["competition_ball"]` of the dead-ball model), `period` (0 outside the graded halves), 0-based Tactical frames, 25 fps |
| `features` | the dead-ball table: `frame, graded, dead, dead_source` |

The checkpoint's player positions are identical to the ATD players, so `match` is the corrected tracking.

## Output (`outputs/phase2/<model>/<run_id>/`, git-ignored)

`prediction.csv` (`frame,label`, 199,319 rows, checked with the organisers' validator), `frame_diagnostics.csv`
(`frame, label, model_team, graded, dead, dead_source, label_source`), `possession_share.csv`, `run_manifest.json`.

## Rules

Train and tune on external data only. Do not read the ground truth or the video, and do not choose settings
on the public 5 minutes (`score_public.py` is a pipeline check only).
