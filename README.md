# Team SNP — More than a Hack 2027, Phase 1 (ball possession)

Frame-level possession labels (`H`/`A`/`D`/`X`) for the BAR-ATH tactical video, produced by code from the
provided data. Technical document: [SNP.md](SNP.md).

## Reproduce the submission

1. Python 3.11 and the pinned packages:

   ```bash
   python -m pip install -r requirements.txt
   ```

2. Unpack the provided match bundle under `data/` exactly as described in [data/README.md](data/README.md)
   (`data/tactical/BAR-ATH_tactical_atd.txt`, `..._atd.xml`, `..._smart_tagging.xml`, ...). The match data is
   not in this repository and must not be added to it.

3. Run:

   ```bash
   python run_submission.py
   ```

   This writes `SNP.csv` (199,319 rows), checks it with `evaluation/validate_submission.py` and scores the
   published 5 minutes as a pipeline check. All intermediate files go to `outputs/` (git-ignored).

The dead-ball model weights are committed (`model/phase1_extraction/dead_ball/weights/`, trained on public
DFL data). The dead-ball step also reads the corrected ball of the team's ball-tracking checkpoint,
`outputs/unified_tracking_v1/tracking_all_frames.csv`, a private derived file that is not in this repository (see
[ball_tracking/README.md](model/phase1_extraction/ball_tracking/README.md)). Without it, set
`CONFIG["competition_ball"]["source"] = "atd"` in `model/phase1_extraction/dead_ball/src/config.py` to run
from the data bundle alone.

## Layout

```text
run_submission.py                 one command: data/ -> SNP.csv
SNP.md                            technical document
model/
├── phase1_extraction/
│   ├── ball_tracking/            retained tracking sources + checkpoint export (its ball feeds dead_ball)
│   ├── tracking_alignment/       tactical/broadcast frame alignment (not used by SNP.csv yet)
│   └── dead_ball/                frame-level dead-ball detection (DFL-trained LightGBM + football-law rules)
└── phase2_possession/
    ├── overlay.py                dead ball first (D), the possession model's H/A on top, X outside the graded halves
    ├── model1_dead_ball/         default: H/A from the nearest team with a fixed-cost decoder
    ├── model2_pathcrf/           5fps PathCRF + 25fps specialist (run_submission.py --possession model2_pathcrf)
    └── model2_template/          template for another possession model
games.py, evaluation/, starter/   organiser code used by the pipeline (scorer, validator, ATD loaders)
tests/                            synthetic tests for tracking_alignment (no match data)
```

Each model folder has its own README with inputs, command, outputs and column meanings; the contract
between phases is in [model/README.md](model/README.md).

## Rules followed

These statements describe the dead-ball and possession models. The corrected ball they read comes from
the ball-tracking checkpoint, whose path has its own review/model provenance (stored object-review
decisions, match-specific model heads), documented in its module README and not covered here.

- Possession and dead-ball labels come out of code; no frame label was set by hand.
- The dead-ball and possession models are trained and tuned on external data only (DFL/IDSSE, Metrica
  sample data, SkillCorner open data).
- The published 5 minutes are used only to check the pipeline, never to choose a setting.

## Licences and data sources

Organiser code (`games.py`, `evaluation/`, `starter/`) keeps its licence ([LICENSE](LICENSE)).
No external data is redistributed here; only the dead-ball model weights trained on DFL data are committed.

| external data | source | licence / terms | use |
|---|---|---|---|
| DFL/IDSSE open data, 7 matches | Bassek et al. 2025, *Scientific Data* 12, 195; figshare doi 10.6084/m9.figshare.28196177 | CC-BY 4.0 (attribution: DFL / Bassek et al.) | training + validation |
| SkillCorner open data, A-League 2024/25, 20 matches | github.com/SkillCorner/opendata | MIT (attribution requested) | validation only |
| Metrica Sports sample data, 3 matches | github.com/metrica-sports/sample-data | no licence file; the README asks users to acknowledge the source for public use | validation only |

Data: DFL Deutsche Fußball Liga / Bassek et al. (2025); SkillCorner; Metrica Sports.

## Optional ball-tracking module

`model/phase1_extraction/ball_tracking/` now contains the verified retained tracking sources and a
checkpoint-to-Phase-1 export/validation interface. It needs the private accepted unified checkpoint;
it is not a raw-data rebuild and is not called by `run_submission.py`. The dead-ball step reads the ball of
the same checkpoint directly. See the module README for its inputs and status.
