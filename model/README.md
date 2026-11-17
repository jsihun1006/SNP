# Models

**Phase 1 — extract what possession needs from the provided data → Phase 2 — label every frame H/A/D/X.**

```text
model/
├── phase1_extraction/
│   ├── ball_tracking/      # retained tracking checkpoint -> tracking.csv (optional)
│   ├── tracking_alignment/ # tactical/broadcast frame alignment                   -> sync/final_tracking_alignment.csv
│   └── dead_ball/          # frame-level dead-ball detection                      -> features.csv
└── phase2_possession/
    ├── overlay.py          # shared: dead ball first (D), the possession model's H/A on top -> prediction.csv
    ├── score_public.py     # public 5-minute pipeline check for any Phase 2 model
    ├── model1_dead_ball/   # alternative; H/A from nearest team with a fixed-cost decoder -> prediction.csv
    ├── model2_pathcrf/     # submission default; 5fps PathCRF + 25fps specialist on top of the dead ball
    └── model2_template/    # copy to add a possession model: write predict_team(), the overlay does the rest
outputs/                    # everything the pipeline writes (git-ignored, derived from the match)
├── phase1/<model>/<run_id>/
├── phase2/<model>/<run_id>/
└── scores/<model>/<run_id>/
```

## Contract

Every model folder holds `README.md` (input → command → output → column meaning), an entry point
(`extract.py` in Phase 1, `possession.py` in Phase 2), `validate.py`, `config.json` and a pinned
`requirements.txt`. Phase 2 also has `evaluation.py`, which scores the published 5 minutes as a pipeline
check only.

| Phase 1 output (`outputs/phase1/<model>/<run_id>/`) | content |
|---|---|
| `features.csv` or `tracking.csv` | one row per frame (features) or per frame × object (tracking) |
| `frame_status.csv` | one row per processed frame |
| `run_manifest.json` | inputs, weights, settings, versions, hashes |
| `validation.json` | result of `validate.py` |

| Phase 2 output (`outputs/phase2/<model>/<run_id>/`) | content |
|---|---|
| `prediction.csv` | `frame,label` for all 199,319 frames (the submission format) |
| `frame_diagnostics.csv`, `possession_share.csv` | per-frame inputs and team shares |
| `run_manifest.json` | Phase 1 run used, settings, hashes |

- **Frames** are 0-based tactical-video frames; ATD `frame_id` = frame + 1. `time_seconds = frame / 25`.
- **Raw data** is read from `data/` (organiser layout, see `data/README.md`) and never copied into a model folder.
- **Private match files and frame-level exports are not committed.** `outputs/` and the submission CSV are git-ignored.

## Dead ball first, possession on top

`run_submission.py --possession <folder>` runs Phase 1 dead ball, then `model/phase2_possession/<folder>/possession.py`.
A new possession model only returns H/A per frame (`predict_team(match, features)`, see `model2_template/`);
`overlay.py` makes every graded frame the dead ball calls dead `D`, requires H/A on every other graded frame,
writes `X` on every frame outside the graded halves (never scored), then writes the outputs above and runs the organisers' validator. `overlay.load_match()` gives the model the match
exactly as the dead-ball step reads it (ATD players, corrected ball per the dead-ball `CONFIG["competition_ball"]`).

## ball_tracking status

The ball-tracking folder contains preserved original sources plus a retained-checkpoint export adapter.
Its tracking output uses Tactical frame0 and its frame-status output keeps all 199,319 frames. The
private accepted unified checkpoint is required. The module itself is not called by the submission runner;
the dead-ball step (`dead_ball/src/competition_data.py`) reads the ball of the same checkpoint directly.
