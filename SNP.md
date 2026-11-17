# Team SNP — technical document (draft)

> Draft. Sections marked **TODO (ball tracking)** and **TODO (possession)** are for the teammates who own
> those parts. The dead-ball sections are complete.

## 1. Approach

The task is split into what possession needs and the possession decision itself.

| step | component | what it does |
|---|---|---|
| Phase 1 | `ball_tracking` | retained tracking sources + checkpoint export; not used by the current `SNP.csv` |
| Phase 1 | `dead_ball` | label every frame dead ball or not, from player movement plus football-law rules |
| Phase 1 | `tracking_alignment` | map every tactical frame to the broadcast frame (ball trajectories + team layouts), so broadcast signals can be joined |
| Phase 2 | `model1_dead_ball` | label every frame H/A/D: D from `dead_ball`, H/A from the nearest team with a fixed-cost decoder |

**Why a separate dead-ball model.** Under `spellF1` a dead ball both ends a possession and starts a new
one, so missed or invented dead balls split or merge possessions. The provided ball is missing on about 40% of
graded frames, mostly during dead balls but also when a goalkeeper holds it, so "ball missing" is not a usable
dead-ball signal. Player movement is: after a whistle, teams slow down and reshape in recognisable ways. The
model learns this from public matches with ball-status labels, and a small set of rules taken from the Laws of
the Game handles what movement alone cannot see (line-outs, penalty kicks, goals, goalkeeper holds).

## 2. Assets used and dropped

| asset | use |
|---|---|
| tactical ATD (players, ball) | players: dead-ball features and rules, nearest team for H/A; ball: nearest team for H/A, and the dead-ball rules on frames the corrected ball does not cover |
| corrected ball (`ball_tracking` checkpoint) | the ball for the dead-ball rules (line crossings, restarts, penalty kicks, goalkeeper holds) on every Tactical frame the checkpoint maps |
| smart tagging | graded-half boundaries only (`games.half_video_blocks`); tags are not used as labels |
| homography, video | not read by the dead-ball/possession code; the ball-tracking path that produced the corrected ball used them (see its README) |
| broadcast ATD | `tracking_alignment` only (frame mapping to tactical); not used by the current possession pipeline |

Dropped for the dead ball (all measured on external data):

- a rule-based detector built from ball signals — many unsupported constants, worse than the model;
- interpolating ball gaps — short gaps barely exist in the provided track, so nothing to gain;
- line-outs inferred by extrapolating the ball through long gaps — 19% precision on DFL;
- training on SkillCorner or Metrica together with DFL — did not improve the fair held-out test;
- holding dead from every goal to its restart by locating the placed ball — DFL 0.897 → 0.792;
- several goalkeeper and slow build-up variants that failed their pre-set criteria (listed in `LESSONS.md` of the development branch).

## 3. Training and validation data

The training data below describe the dead-ball/possession models. None of those training datasets comes from the provided match. The corrected ball that the dead-ball rules read comes from the ball-tracking checkpoint, whose path contains stored object-review decisions and match-specific model heads; its provenance is documented in its README and is not covered by this statement.

| dataset | role | labels | who labelled | amount |
|---|---|---|---|---|
| DFL/IDSSE open data (Bassek et al. 2025, Scientific Data 12, 195; CC-BY 4.0) | training + main validation | ball status alive/dead in the tracking files | DFL (data provider) | 7 Bundesliga matches |
| Metrica Sports sample data (github.com/metrica-sports/sample-data; no licence file, source acknowledged as its README asks) | validation only | dead balls derived from the event files | Metrica Sports | 3 matches |
| SkillCorner open data (github.com/SkillCorner/opendata, A-League 2024/25; MIT) | validation only | outside "phases of play" = dead | SkillCorner | 20 matches |

Validation is leave-one-match-out on DFL. Because the provided tracking often loses the ball when the
goalkeeper holds it, DFL is also scored with the ball deleted during holds (and from 1 s before the keeper
receives it). Every rule and constant was adopted only if it met criteria fixed before the run on these data.

## 4. Possession rules implemented

**Dead ball (`dead_ball`)**

- Entering dead is free at a visible line crossing; a certain line-out stays dead until the ball moves again
  inside the pitch (Law 9). A crossing seen clearly outside (0.36 m, 2 frames) also pushes the score towards dead
  from the crossing frame. Forcing dead there instead was rejected: DFL records the dead ball up to 2 s after the
  crossing, and the forced frames left short in-play fragments that split possessions.
- A dead ball ends at a detected restart — the ball leaving a player fast — when one is seen within 5 s of the
  model's end.
- Slow build-up: stretches with a pass, a reception or a carried ball are harder to call dead.
- Penalty kick (Law 14): detected from the formation (keeper on the line, one kicker at the mark, nobody else
  in the box); play restarts at the kick. If a kickoff formation (Law 8) follows, the goal holds dead until the
  kickoff (Law 10).
- Goalkeeper holding the ball is in play (Law 12): the ball vanishes next to the keeper in the box during play
  and reappears away from a restart spot (goal area, penalty mark).
- The second-half pre-kickoff window published in `FAQ.md` is D.

**Possession (`model1_dead_ball`)** — **TODO (possession)** if the final possession model differs.

- H/A is the team of the player nearest to the ball; a change of team must be sustained (switch cost) rather
  than a single nearest-player flip.
- While the ball is missing inside an in-play stretch, the surrounding team keeps possession.
- Loose balls and challenges: possession stays with the last team until the other team is consistently nearest.

## 5. Results on external data (dead ball)

| data | in-play spell F1 |
|---|---|
| DFL, 7 matches, leave-one-match-out | 0.905 |
| DFL, ball deleted while the keeper holds it | 0.901 |
| DFL, ball deleted from 1 s before the keeper receives it | 0.899 |
| Metrica, 3 matches | 0.832 |
| SkillCorner, 20 matches | 0.759 |

In-play spell F1 = the official spell score with in play = H and dead = D.

## 6. Limitations

- Movement-based dead balls: a slow build-up can look like a short dead ball; a pass right after a whistle can
  cut a running dead ball for a few seconds.
- 2-D tracking jitter near a line can create a false line-out.
- After an ordinary goal with the ball out of sight, the celebration can briefly look like play.
- A goalkeeper hold where the ball never disappears and never rolls away is not recognised.
- H/A uses only the nearest player — **TODO (possession)**.
- With more time: learn restarts directly from public event data, and a possession model that uses ball
  trajectory rather than nearest player only.

## 7. Reproduction

```bash
python -m pip install -r requirements.txt
python run_submission.py          # data/ -> SNP.csv, validated with evaluation/validate_submission.py
```

Weights for the dead-ball model are in the repository; retraining instructions are in
`model/phase1_extraction/dead_ball/README.md`.

The dead-ball step reads the corrected ball from `outputs/unified_tracking_v1/tracking_all_frames.csv` (checked against its
SHA-256). That checkpoint is a private derived file: it is not in the repository and cannot be rebuilt from
the data bundle with this repository alone (see the ball-tracking README). Without it the run stops; setting
`CONFIG["competition_ball"]["source"] = "atd"` in `model/phase1_extraction/dead_ball/src/config.py` uses the
provided ATD ball instead and needs only the data bundle.

## Ball-tracking module

The ball-tracking module now preserves the retained V3 ball/Tactical-player source path and exports an
existing private unified checkpoint to the Phase 1 tracking contract. This export is a frame/format
adapter, not new detection, correction, training or possession inference. The dead-ball step reads the
ball of the same checkpoint directly (Tactical frames it maps). Its stored review references/model heads are
not covered by the external-training-only statement above; see
[ball_tracking/README.md](model/phase1_extraction/ball_tracking/README.md) for the exact scope.
