# Team SNP — technical document

More than a Hack 2027, Phase 1: frame-level ball possession (`H`/`A`/`D`/`X`) for the BAR–ATH tactical video.

## 1. Approach

**Dead ball first, possession on top.**

| step | what it does |
|---|---|
| input | players: the 46 Tactical ATD track slots (anonymous, never joined); ball: the team's corrected ball on the 155,923 Tactical frames its checkpoint maps, the ATD ball elsewhere |
| Phase 1 `dead_ball` | every frame dead or not: LightGBM on player movement (trained on public DFL matches) + Viterbi smoothing + rules from the Laws of the Game |
| Phase 2 `model2_pathcrf` | graded halves: `D` where the dead ball says dead, otherwise `H`/`A` from a 5fps PathCRF (public weights) refined by a 25fps model on uncertain stretches; outside the graded halves: `X` |

Under `spellF1` a dead ball ends one possession and starts the next, so dead-ball errors split or merge
possessions. The provided ball is missing on about 40% of frames, so "ball missing" is not a dead-ball signal;
player movement is (teams slow down and reshape after a whistle). With the dead ball fixed, the possession step only
decides which team has the ball inside each in-play stretch.

## 2. Assets used and dropped

| asset | use |
|---|---|
| Tactical ATD | players for both phases; ball where the corrected ball has no frame |
| corrected ball (team `ball_tracking` checkpoint; see §3) | ball for both phases on the frames it maps |
| smart tagging | graded-half boundaries only; tags are not labels, no setting was chosen on them |
| broadcast ATD, homography, video | only inside the ball-tracking path |
| public 5 minutes | pipeline check only, never used to choose a setting |

Tried and dropped (all judged on external data): a rule-based dead-ball detector from ball signals; interpolating
ball gaps; extrapolating line-outs through long gaps (19% precision on DFL); training on SkillCorner/Metrica with
DFL; holding dead from every goal to its restart (DFL 0.897 → 0.792); forcing dead at line crossings; a TranSPORTmer
dead-ball model; a nearest-team possession decoder (`model1_dead_ball`, kept as an alternative).

## 3. Training data

| dataset | role | labels | who labelled | amount |
|---|---|---|---|---|
| DFL/IDSSE open data (Bassek et al. 2025; CC-BY 4.0) | dead-ball training + validation | ball alive/dead in the tracking | DFL | 7 matches |
| Metrica Sports sample data | validation only | dead balls from the event files | Metrica Sports | 3 matches |
| SkillCorner open data (MIT) | validation only | outside "phases of play" = dead | SkillCorner | 20 matches |
| DFL/IDSSE open data (event data) | possession, 5fps PathCRF: public weights of Kim et al. (KDD 2026, MPL-2.0), trial 140 = held-out fold J03WR9 | ball possession and passes from the event data | DFL | 6 matches for training/validation |
| DFL/IDSSE open data (event data) | possession, 25fps model: trained by the team; corrections chosen on validation | same | DFL | 5 train (J03WMX, J03WN1, J03WOH, J03WOY, J03WPY), 1 validation (J03WQQ), 1 test (J03WR9) |

Nothing from the provided game was used to train or tune the dead-ball or possession models.

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
- **Team (`H`/`A`)**: a 5fps PathCRF decodes the ball's path between players (who holds it, who receives it) inside
  each in-play stretch; the dead-ball mask is fixed. Contested or uncertain stretches (low confidence, an opponent
  within 2.5 m, closing 0.65 m in 0.2 s) are re-scored by a 25fps model and mixed 1:1 with the 5fps scores; a change
  of team must gain at least 0.1 per changed frame; very short back-and-forth possessions are merged; final
  boundaries snap to an observed touch within 0.4 s. A missing ball is never a change of team by itself.
- **`X`**: every frame outside the graded halves (before kickoff, halftime, after the final whistle).

## 5. Results on external data (dead ball)

Official spell score with in play = `H` and dead = `D`:

| data | in-play spell F1 |
|---|---|
| DFL, 7 matches, leave-one-match-out | 0.905 |
| DFL, ball deleted while the keeper holds it | 0.901 |
| Metrica, 3 matches | 0.832 |
| SkillCorner, 20 matches | 0.759 |

Possession (H/A/D spell F1) on the held-out DFL match J03WR9 with the D mask fixed: 0.762 (`model2_pathcrf/README_MODEL.md`).

## 6. Limitations

- A slow build-up can look like a short dead ball; 2-D jitter near a line can create a false line-out; a goalkeeper
  hold where the ball never disappears is not recognised.
- The possession model was trained on player identities; the provided ATD has 46 anonymous track slots, so its
  player-set phases are shorter; contested balls and very short possessions are the main errors.
- The corrected ball is a private derived file, so `SNP.csv` cannot be rebuilt from the data bundle alone, and its
  path is not label-free (§3).
- With more time: possession trained on more public matches, and an end-to-end ball tracker without stored review
  decisions.

## 7. Reproduction

```bash
python -m pip install -r requirements.txt          # includes the possession model's requirements
python run_submission.py          # data/ -> SNP.csv (dead ball, then model2_pathcrf), checked with the validator
```

The provided bundle goes under `data/` (`data/README.md`). The corrected-ball checkpoint
(`outputs/unified_tracking_v1/tracking_all_frames.csv`, SHA-256 checked) is not in the repository; without it set
`CONFIG["competition_ball"]["source"] = "atd"` in `model/phase1_extraction/dead_ball/src/config.py` to run from the
bundle alone. Dead-ball and possession weights are committed; retraining of the dead ball is in
`model/phase1_extraction/dead_ball/README.md`. The possession model runs on CPU in tens of minutes (`--device cuda`
is faster).
