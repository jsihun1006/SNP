# model2_pathcrf — possession on top of the dead ball (Phase 2)

H/A from a 5fps PathCRF (public trial 140) refined by a 25fps specialist on uncertain stretches, with frozen
corrections. The model, its parameters, training split and DFL results are described in
[README_MODEL.md](README_MODEL.md) (original notes) and `parameters.json`. This folder adds the workflow that
runs it on the provided match on top of the Phase 1 dead ball.

**Status: wired, not yet run on the provided match.** `run_submission.py` still defaults to
`model1_dead_ball`; select this model with `--possession model2_pathcrf`.

## Run

```bash
python -m pip install -r model/phase2_possession/model2_pathcrf/requirements.txt
python run_submission.py --possession model2_pathcrf
# or, after a dead-ball run:
python model/phase2_possession/model2_pathcrf/possession.py --dead-run submission --run-id submission
python model/phase2_possession/model2_pathcrf/possession.py --dead-run submission --prepare-only   # input only
```

## Workflow

| step | what |
|---|---|
| 1. dead ball | `outputs/phase1/dead_ball/<run_id>/features.csv` from `model/phase1_extraction/dead_ball` |
| 2. input | the match as the dead-ball step reads it (`../overlay.load_match`): ATD players (46 anonymous track slots, never joined), corrected ball, 0-based Tactical frames → `input/match.npz` in the model's format (`README_MODEL.md`, "입력 NPZ"); `dead` = the dead-ball call plus every frame outside the graded halves |
| 3. model | `predict.py`, unchanged: frozen weights `weights/pathcrf_trial140.pt` and `weights/specialist_epoch29.pt` (SHA-256 checked against `parameters.json`); no training or search → `model_output.csv` |
| 4. overlay | `../overlay.py`: `D` wherever the dead ball says dead, the model's `H`/`A` on the other graded frames, `X` outside the graded halves → `prediction.csv`, checked with the organisers' validator |

## Fallback

The 5fps stage works on phases with a fixed set of player columns and skips phases with fewer than 100
in-play frames. The provided ATD has 46 track slots that start and stop through the match, so some in-play
frames may be left uncovered; `predict.py` then stops rather than invent a label. `--fallback model1` supplies
the H/A of `model1_dead_ball` for the same run id on those frames (run `model1_dead_ball/possession.py` first);
`--fallback none` (default) stops instead.

## Output (`outputs/phase2/model2_pathcrf/<run_id>/`, git-ignored)

`input/match.npz`, `input/dead.csv`, `model_output.csv` (the model's own H/A/D), then `prediction.csv`,
`frame_diagnostics.csv`, `possession_share.csv`, `run_manifest.json` from the overlay.

## Licences

`vendor/pathcrf/` and `weights/pathcrf_trial140.pt` come from github.com/hyunsungkim-ds/pathcrf
(revision in `vendor/pathcrf/SOURCE.json`), MPL-2.0 (`vendor/pathcrf/LICENSE`). The 25fps specialist was
trained on public DFL/IDSSE matches (split in `parameters.json`).
