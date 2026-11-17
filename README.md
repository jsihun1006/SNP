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
DFL data). No other download is needed to regenerate the file.

## Layout

```text
run_submission.py                 one command: data/ -> SNP.csv
SNP.md                            technical document
model/
├── phase1_extraction/
│   ├── ball_tracking/            ball-tracking correction (placeholder, see its README)
│   └── dead_ball/                frame-level dead-ball detection (DFL-trained LightGBM + football-law rules)
└── phase2_possession/
    └── model1_dead_ball/         H/A/D decoder: D from dead_ball, H/A from the nearest team
games.py, evaluation/, starter/   organiser code used by the pipeline (scorer, validator, ATD loaders)
```

Each model folder has its own README with inputs, command, outputs and column meanings; the contract
between phases is in [model/README.md](model/README.md).

## Rules followed

- Labels come out of code. Nothing on the provided match was labelled or corrected by hand.
- Training and tuning use external data only (DFL/IDSSE, Metrica sample data, SkillCorner open data).
- The published 5 minutes are used only to check the pipeline, never to choose a setting.

## Licences

Organiser code (`games.py`, `evaluation/`, `starter/`) keeps its licence ([LICENSE](LICENSE)). External data
used for training/validation: DFL/IDSSE open data (CC-BY 4.0, Bassek et al. 2025), SkillCorner open data
(MIT), Metrica Sports sample data.
