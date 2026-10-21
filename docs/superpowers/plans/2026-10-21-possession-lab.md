# Possession Lab Implementation Plan

**Goal:** Implement the recommended data-only, gap-aware HMM for the BAR-ATH possession task in a
separate, reproducible package. Produce a full-match `frame,label` CSV, public-GT metrics, stress
tests, visual diagnostics, and an HTML report.

**Non-goals:** The MOSSE + GrayDiff + NanoTrack fusion, broadcast video indexing, hand-labelling,
and any use of private full-match ground truth.

## Files

- `possession_lab/__init__.py` — package version and labels
- `possession_lab/config.py` — match constants, model parameters, official paths
- `possession_lab/data.py` — ATD, smart-tag, homography loaders
- `possession_lab/features.py` — geometry, gap, touch, edge, tag features
- `possession_lab/model.py` — emissions, Viterbi, forward-backward confidence
- `possession_lab/evaluate.py` — official metrics, stress masks, full-match structure
- `possession_lab/pipeline.py` — inference, submission writing, full run
- `possession_lab/visualize.py` — static figures and HTML report
- `possession_lab/cli.py`, `__main__.py` — commands
- `possession_lab/tests/` — stdlib `unittest` tests
- `possession_lab/README.md` — usage, design, GPU note, limitations
- `possession_lab/outputs/` — generated only, ignored by git

## Tasks

1. Write failing tests for tracking parsing, tag parsing, feature geometry, HMM decode, metrics,
   full-match structure, and the small end-to-end pipeline.
2. Implement the loaders and feature extraction; make the data/feature tests pass.
3. Implement the gap-aware HMM and posterior confidence; make the model tests pass.
4. Implement evaluation, stress masks, and full-match statistics; make the evaluation tests pass.
5. Implement inference and submission writing; run the tests.
6. Run on the real tactical data. Verify public metrics against the reference
   (`spellF1 0.4966`, `macroF1 0.7709`, `D-F1 0.5916`).
7. Run clean/random/long-gap stress tests; compare with the reference.
8. Generate the figures and `outputs/report.html`.
9. Validate the full CSV with `evaluation/validate_submission.py`, inspect label counts and spell
   lengths, and record the final verification evidence.
10. Document CPU-vs-GPU reasoning and the skills used.

## Acceptance

- `python -m unittest discover -s possession_lab/tests -v` passes.
- `possession_lab/outputs/predictions.csv` has 199,319 rows, frames 0..199318, labels in
  `{H,A,D,X}`, and passes the official validator.
- Public metrics reproduce the reference within rounding.
- `possession_lab/outputs/report.html` embeds the metrics and all diagnostic figures.
- No match data or private ground truth is copied into the package.
