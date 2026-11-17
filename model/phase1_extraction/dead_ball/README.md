# dead_ball — frame-level dead-ball detection (Phase 1)

Labels **every tactical-video frame as dead ball or not** from the provided ATD. It does not decide H/A.
Phase 2 (`model/phase2_possession/model1_dead_ball/`) takes `dead` as its D state.

## Method

```text
player features (team speed quantiles, shape, distance to the lines; no ball features)
-> LightGBM trained on 7 public DFL/IDSSE matches (no labels from the provided match)
-> 2-state Viterbi smoothing (switch cost 40)
   + slow build-up correction: stretches with a pass, a reception or a player carrying the ball are harder to enter
     dead from; a certified line crossing or a joint team stop makes entering dead cheaper; short
     crossing->restart pairs are kept (src/slow_buildup.py)
   + entering dead is free at a visible line crossing; a crossing seen 0.36 m+ outside for 2 frames also pushes the
     score towards dead from the crossing frame, and the last inside frame stays in play when the model sees play
-> a certain line-out stays dead until the ball moves again inside the pitch and the model calls play
-> a dead ball that ends within 5 s of a detected restart (ball leaves a player fast) ends at that restart
-> penalty kicks (src/penalty.py, detected from the player formation, Law 14): play restarts at the kick; if a kickoff
   follows (Law 8 formation) the goal holds dead until the kickoff (Law 10)
-> goalkeeper holding the ball is in play (Law 12): the ball vanishes next to the keeper in the box and
   reappears away from a restart spot (goal area for a goal kick, penalty mark for a penalty)
-> dead-ball starts snap to the line crossing
-> the second-half pre-kickoff window published in FAQ.md (frames 110808-113741) is D
```

- The ball is used for line crossings, restarts, penalty kicks and goalkeeper holds. **A missing ball is
  never taken as dead by itself.**
- On the provided match the ball is the corrected ball of the ball-tracking checkpoint
  (`outputs/unified_tracking_v1/tracking_all_frames.csv`, SHA-256 checked) on every Tactical frame it maps, and the ATD ball on the
  other frames; players always come from the ATD. `CONFIG["competition_ball"]["source"] = "atd"` uses the
  ATD ball everywhere. No setting was changed for the corrected ball.
- The model input has no ball features: DFL fixes the ball during dead balls and the Metrica samples
  delete it, so ball features would read the label off the training data.

## Evidence (external data with ground truth)

| validation (final pipeline) | in-play spell F1* |
|---|---|
| DFL 7 matches, leave-one-match-out (`src/eval_dfl_current.py`) | **0.905** |
| DFL with the ball deleted while the keeper holds it (as in the provided tracking) | 0.901 |
| DFL with the ball deleted from 1 s before the keeper receives it | 0.899 |
| Metrica sample, 3 matches (not used for training) | 0.832 |
| SkillCorner open data, 20 matches (not used for training; 10 fps upsampled) | 0.759 |

\* the official spell score with in play = H and dead = D: how well the dead balls split the match into
possessions. All numbers: `src/eval_clean.py`.

Every rule and constant was chosen on external data with criteria fixed before the run. The main
criterion is DFL (original and ball-deleted simulations); Metrica and SkillCorner are auxiliary.
Nothing was trained, tuned or validated on the provided match or its public 5 minutes.

## Run

From the repository root, with the provided data under `data/`:

```bash
python -m pip install -r model/phase1_extraction/dead_ball/requirements.txt
python model/phase1_extraction/dead_ball/extract.py --run-id submission     # pipeline + export
python model/phase1_extraction/dead_ball/extract.py --skip-pipeline          # export existing results only
python model/phase1_extraction/dead_ball/validate.py outputs/phase1/dead_ball/submission
```

## Weights

| item | value |
|---|---|
| file | `weights/model_all_v2.txt` (LightGBM text model, 2.8 MB) |
| SHA-256 | `18a46eb5dd1e89f1057edc163e04b5f33b178a5c63123d7b4b2078724683737e` |
| training data | Sportec Open DFL (IDSSE), 7 matches. Bassek et al. 2025, Scientific Data 12, 195. CC-BY 4.0 |
| labels | DFL's own ball status (alive/dead) in the tracking files, produced by DFL, not by us |
| settings | `src/config.py` (feature set v2, `lgbm_params`, `sim_seed` 7) |

To retrain: download the 7 matches × 3 raw files from figshare (doi 10.6084/m9.figshare.28196177) into
`data/external/dfl/raw/` (scripted downloads are blocked), then

```bash
python model/phase1_extraction/dead_ball/src/dfl_load.py all
python model/phase1_extraction/dead_ball/src/train_dfl.py --save-final
```

## Output: `outputs/phase1/dead_ball/<run_id>/` (git-ignored)

| file | content |
|---|---|
| `features.csv` | one row per frame: `frame,time_seconds,graded,dead_score,dead,dead_source` |
| `frame_status.csv` | one row per frame: `frame,time_seconds,processed,ball_seen` |
| `run_manifest.json` | inputs, weights, settings (incl. `slow_buildup`, `keeper_hold`), versions, hashes |
| `validation.json` | result of `validate.py` |

| column | meaning |
|---|---|
| `frame` | 0-based tactical-video frame (ATD `frame_id` = frame + 1) |
| `graded` | 1 inside the two graded halves |
| `dead_score` | frame-level dead-ball probability of the model before smoothing (0-1) |
| `dead` | final call, 1 = dead. Always 0 outside the graded halves |
| `dead_source` | `model`, `model+out_start`, `forced_line_out` or `public_kickoff`; empty when in play |

## Code (`src/`)

| role | files |
|---|---|
| run on the provided data | `apply_competition.py` (model probabilities), `final_dead_balls.py` (final calls), `config.py` (all settings), `features.py`, `smoothing.py`, `slow_buildup.py` (Viterbi with slow build-up, line-out and restart terms), `penalty.py` (penalty kicks), `kickoff.py`, `keeper_hold.py`, `competition_data.py`, `repo_root.py` |
| training | `train_dfl.py`, `dfl_load.py`, `dfl_labels.py`, `simulate.py`, `metrics.py` |
| validation on external data | `eval_*.py`, `metrica_eval.py`, `metrica_load.py`, `skillcorner_eval.py`, `skillcorner_check.py`, `compare_smoothing.py` |

Some functions keep options that are off in `config.py` (dropped variants such as `case3`, `no_exit` or the hard line-out lock). They stay so that the `eval_*.py` scripts can reproduce the comparisons that rejected them; the submitted path uses only the settings in `config.py`.

## Limitations

- Player movement is the main signal, so a slow build-up with the ball circulating can look like a short dead ball.
- 2-D tracking jitter near a line can create a false line-out (`dead_source = forced_line_out`).
- A short pass right after a whistle can look like a build-up and cut a running dead ball for a few seconds.
- After an ordinary goal with the ball out of sight, the celebration can briefly look like play (penalty goals are held to the kickoff).
- DFL numbers are optimistic for the provided match; the dead-ball end may be defined differently there.
