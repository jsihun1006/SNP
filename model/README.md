# Models

**Phase 1 — extract what possession needs from the provided data → Phase 2 — label every frame H/A/D/X.**

```text
model/
├── phase1_extraction/
│   ├── ball_tracking/      # retained tracking checkpoint -> tracking.csv (optional)
│   ├── tracking_alignment/ # tactical/broadcast frame alignment                   -> sync/final_tracking_alignment.csv
│   └── dead_ball/          # frame-level dead-ball detection                      -> features.csv
└── phase2_possession/
    └── model1_dead_ball/   # H/A/D decoder; D from dead_ball, H/A from nearest team -> prediction.csv
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

## ball_tracking status

The ball-tracking folder contains preserved original sources plus a retained-checkpoint export adapter.
Its tracking output uses Tactical frame0 and its frame-status output keeps all 199,319 frames. The
private accepted unified checkpoint is required. The module itself is not called by the submission runner;
the dead-ball step (`dead_ball/src/competition_data.py`) reads the ball of the same checkpoint directly.
