# Team SNP — technical document

More than a Hack 2027, Phase 1: frame-level ball possession (`H`/`A`/`D`/`X`) for the BAR–ATH tactical video.

## 1. Approach

**Dead ball first, possession on top.**

| step | what it does |
|---|---|
| input | players: the 46 Tactical ATD track slots (anonymous, never joined); ball: the team's corrected ball on the 155,923 Tactical frames its checkpoint maps, the ATD ball elsewhere |
| Phase 1 `dead_ball` | every frame dead or not: LightGBM on player movement (trained on public DFL matches) + Viterbi smoothing + rules from the Laws of the Game |
| Phase 2 `model1_dead_ball` | graded halves: `D` where the dead ball says dead, otherwise `H`/`A` from a fixed-cost decoder over the team nearest to the ball; outside the graded halves: `X` |

Under `spellF1` a dead ball ends one possession and starts the next, so dead-ball errors split or merge
possessions. The provided ball is missing on about 40% of frames, so "ball missing" is not a dead-ball signal;
player movement is (teams slow down and reshape after a whistle). With the dead ball fixed, the possession step only
decides which team has the ball inside each in-play stretch.

## 2. Assets used and dropped

| asset | use |
|---|---|
| Tactical ATD | players for both phases; ATD ball for H/A and for the dead-ball rules where the corrected ball has no frame |
| corrected ball (team `ball_tracking` checkpoint; see §3) | ball for the dead-ball rules (line crossings, restarts, penalty kicks, keeper holds) |
| smart tagging | graded-half boundaries only; tags are not labels, no setting was chosen on them |
| broadcast ATD, homography, video | only inside the ball-tracking path |
| public 5 minutes | pipeline check only, never used to choose a setting |

Tried and dropped (all judged on external data): a rule-based dead-ball detector from ball signals; interpolating
ball gaps; extrapolating line-outs through long gaps (19% precision on DFL); training on SkillCorner/Metrica with
DFL; holding dead from every goal to its restart (DFL 0.897 → 0.792); forcing dead at line crossings; a TranSPORTmer
dead-ball model; a PathCRF possession model (wired in `model2_pathcrf`, not used for `SNP.csv`).

## 3. Training data

| dataset | role | labels | who labelled | amount |
|---|---|---|---|---|
| DFL/IDSSE open data (Bassek et al. 2025; CC-BY 4.0) | dead-ball training + validation | ball alive/dead in the tracking | DFL | 7 matches |
| Metrica Sports sample data | validation only | dead balls from the event files | Metrica Sports | 3 matches |
| SkillCorner open data (MIT) | validation only | outside "phases of play" = dead | SkillCorner | 20 matches |

The possession step has no training: fixed costs declared before any evaluation. Nothing from the provided game was
used to train or tune the dead-ball or possession models.

**Corrected ball.** It comes from the team's ball-tracking path (`model/phase1_extraction/ball_tracking`): YOLO26X
ball detections in the broadcast video, projected with the provided homography and aligned to the Tactical frames,
then false-positive, precision and temporal corrections. That path uses stored object-review decisions and
classification heads specific to this match.

## 4. Possession rules

- **Dead ball (`D`)**: free entry at a visible line crossing, a certain line-out stays dead until the ball moves
  again inside the pitch (Law 9); a dead ball ends at a detected restart (ball leaves a player fast) within 5 s;
  passes, receptions and carries make a stretch harder to call dead; penalty kicks detected from the formation
  restart play at the kick, and a following kickoff formation holds dead after the goal (Laws 14, 8); a goalkeeper
  holding the ball is in play (Law 12); a missing ball is never dead by itself.
- **Team (`H`/`A`)**: the team nearest to the ball, decoded as a sequence (2.0 per second of disagreement, 1.2 per
  change of team), so a change must be sustained. Loose balls and challenges stay with the last team until the other
  team is consistently nearest; a missing ball keeps the surrounding team.
- **`X`**: every frame outside the graded halves (before kickoff, halftime, after the final whistle).

## 5. Results on external data (dead ball)

Official spell score with in play = `H` and dead = `D`:

| data | in-play spell F1 |
|---|---|
| DFL, 7 matches, leave-one-match-out | 0.905 |
| DFL, ball deleted while the keeper holds it | 0.901 |
| Metrica, 3 matches | 0.832 |
| SkillCorner, 20 matches | 0.759 |

## 6. Limitations

- A slow build-up can look like a short dead ball; 2-D jitter near a line can create a false line-out; a goalkeeper
  hold where the ball never disappears is not recognised.
- H/A uses only the nearest player; contested balls and fast passes between teams are the main errors.
- The corrected ball is a private derived file, so `SNP.csv` cannot be rebuilt from the data bundle alone, and its
  path is not label-free (§3).
- With more time: a possession model that uses ball trajectory and touches, and an end-to-end ball tracker without
  stored review decisions.

## 7. Reproduction

```bash
python -m pip install -r requirements.txt
python run_submission.py          # data/ -> SNP.csv, checked with evaluation/validate_submission.py
```

The provided bundle goes under `data/` (`data/README.md`). The corrected-ball checkpoint
(`outputs/unified_tracking_v1/tracking_all_frames.csv`, SHA-256 checked) is not in the repository; without it set
`CONFIG["competition_ball"]["source"] = "atd"` in `model/phase1_extraction/dead_ball/src/config.py` to run from the
bundle alone. Dead-ball weights are committed; retraining is in `model/phase1_extraction/dead_ball/README.md`.
