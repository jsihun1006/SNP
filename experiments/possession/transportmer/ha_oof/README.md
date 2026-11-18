# DFL in-play TranSPORTmer H/A

Run from the repository root in a Python 3.10+ environment with PyTorch, NumPy, pandas and PyArrow. The entry point uses the local DFL Parquet cache and the existing ATD and dead-ball files for an aggregate ball-availability profile.

```sh
python src/possession/transportmer/run_transportmer_ha.py --stage benchmark --output data/transportmer_ha/dfl_seed2027
python src/possession/transportmer/run_transportmer_ha.py --stage run --output data/transportmer_ha/dfl_seed2027
```

The default seed is 2027. Use `--device cpu`, `--device mps` or `--device cuda` to specify a device before creating the run directory. Use `--cache`, `--atd` and `--dead` to specify alternate local input paths. Use a fresh experiment directory for different settings or source. Run one process per directory. Repeating the run command resumes at the last completed epoch for each checkpoint and uses any completed per-half inference.

The six OOF matches are each predicted from a model trained on the other five development matches. The two halves of a match always travel together. Holdout preparation occurs after both B and C checkpoints are fixed. DFL ball status sets the in-play mask across conditions. The empirical gap-length histogram comes from half-bounded predicted in-play segments of the tactical MTAH ATD. The default interpolation assumption is 25 percent of synthetically affected frames, in addition to the targeted missing share. Each half has a fixed chronological draw reused across conditions. The realised augmentation is recorded separately.

The B and C comparison uses a shared five-epoch warm-up followed by the same number of optimizer steps. C draws half of each batch from persistent OOF error intervals, with uniform draws over available error types and intervals within a type. Co-occurring error tags are retained. The primary type determines its sampling group. Distances within two metres of the ball for a player from each team are a diagnostic for potential contests.

The primary score is official spellF1. The short-spell recall uses the official one-to-one match at temporal IoU 0.5 on ground-truth spells shorter than two seconds. H/A transition matches use the destination and origin, a 25-frame tolerance, and stay within one graded in-play run. The holdout comparison ends the experiment.

| Output | Meaning |
| --- | --- |
| `manifest.json` | Seed, match groups, settings and source hashes |
| `ball_profile.json` | Aggregate MTAH observations for DFL dropout |
| `development_halves.json`, `holdout_augmentation.json` | Per-half preparation and augmentation |
| `benchmark.json`, `remaining.json`, `timing.json` | Benchmark, updated ETA and measured phase timing |
| `oof_smoothing_by_match.csv`, `oof_smoothing_mean.csv` | Raw and smoothed OOF scores |
| `error_ledger.csv`, `oof_selection.json` | Framed OOF error intervals and frozen postprocessing |
| `locked_holdout_recipe.json`, `holdout_comparison.csv` | Completed B/C settings and DFL holdout scores |
| `oof_probabilities/`, `holdout_B/`, `holdout_C/` | Per-half probabilities and frame-aligned audits |
