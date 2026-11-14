# DFL LightGBM dead-ball experiments

Run from the repository root with Windows x64 and Python 3.10.22. Install the locked environment, then pass an empty cache directory and a new output directory.

```powershell
python -m venv .venv
.\.venv\Scripts\python.exe -m pip install -r experiments/lightgbm_dfl/requirements-lock.txt
.\.venv\Scripts\python.exe experiments/lightgbm_dfl/reproduce.py --cache data/external/dfl/repro_cache --output data/lightgbm_dfl_reproduction
```

The runner parses the checked-in DFL XML files and compares every output CSV with the corresponding file under `reference/`. It checks row order, labels and numbers with absolute tolerance 1e-12. A cache can be reused across runs, with a new output directory for each invocation.

## Experiment order

1. **264 versus 370 features.** Each fold fits both models on 12 halves and evaluates them on the two held-out halves. Training uses every fifth frame and evaluation uses all 25 fps frames.
2. **Events as probabilities.** The dead-ball model trains on the 370 tracking features plus six binary restart indicators. A separate multiclass model fits the same training halves and supplies the six event probabilities at evaluation.
3. **Events in training only.** The dead-ball model trains on the same binary indicators and receives six missing inputs at evaluation.

The split file fixes the shuffle and all seven folds. Restart labels use the DFL event timestamp ±0.5 s. All three runs use the fixed six-rule correction from the repository notebook and the official interval matcher. Dead-ball and in-play scores each average IoU thresholds 0.3, 0.4, 0.5, 0.6 and 0.7. Halves are scored independently, then averaged across halves and folds.

The 370-column schema uses the VIF 50 exclusion order in `lightgbm_vif50.json`, with the current home and away mean-speed columns restored. To regenerate the VIF selection from DFL tracking:

```powershell
.\.venv\Scripts\python.exe src/compare_derived_vif.py --cache data/external/dfl/repro_cache --report data/lightgbm_vif50_reproduction.json
.\.venv\Scripts\python.exe experiments/lightgbm_dfl/verify_vif.py data/lightgbm_vif50_reproduction.json
```

The selection was computed across DFL's seven matches. Each evaluation half also has its other half in training. These scores compare the feature and event configurations under that split.

| Configuration | Dead-ball spellF1 | In-play spellF1 |
|---|---:|---:|
| v2, 264 features | 0.89076 | 0.88524 |
| 370 features | 0.89635 | 0.88183 |
| 370 features plus event probabilities at evaluation | 0.87208 | 0.85664 |
| 370 features plus event indicators in training only | 0.89473 | 0.87750 |
