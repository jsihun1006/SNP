# Possession Lab — technical note

## Objective

Assign one label per tactical video frame: `H` (Barcelona possession), `A` (Atletic Club
possession), `D` (dead ball / out of play), or `X` (ungraded). The submission is 199,319 rows.

The model is a **data-only, gap-aware 3-state HMM**. It deliberately excludes the earlier
MOSSE + GrayDiff + NanoTrack video ensemble: the tactical tracking already provides a cleaner,
synchronised observations stream, and the public window is only a correctness check.

## Inputs

| Input | Use |
|---|---|
| `BAR-ATH_tactical_atd.txt/.xml` | Ball and 46 anonymous player slots, metres, 25 fps |
| `BAR-ATH_tactical_smart_tagging.xml` | 638 intervals, 14 codes, 6 types, team/type labels |
| `BAR-ATH_tactical_homography.csv` | Per-frame camera geometry; diagnostics and pitch-space validity |
| `evaluation/gt/...public.csv` | 7,500-frame public GT; evaluation only |

`video_frame = frame_id - 1`. Smart-tag intervals use
`[round(start*25), round(end*25)+1)`. The tactical homography is valid on 100% of frames (median
reprojection error 0.24 px, p99 4.81 px).

## Features

For each frame:

* `d_H`, `d_A`: distance from the ball to the nearest home/away tracked player.
* `near_team`, `near_distance`, `second_distance`.
* `ball_speed`, `ball_acceleration` (raw, only across consecutive observations).
* `team_speed`: mean speed of the tracked players, smoothed over one second.
* `gap`: frames since the last observed ball.
* `edge`, `last_edge`: distance of the ball (or last known ball) from the nearest pitch boundary.
* `out_of_play`: observed ball outside `[0,L] × [0,W]`.
* `last_touch`, `last_touch_age`: nearest player within 0.8 m.
* `d_H_last`, `d_A_last`: current team shape around the last observed ball.
* `tag_vote`, `tag_setpiece`, `tag_coverage`.
* `player_count`: visible player slots, used for the sparse-tracking guard.

## Emissions

At visible frames, with `σ = 3.5 m`, `d_dead = 7 m`:

```
q_H = exp(-d_H² / (2σ²)) + 1e-12
q_A = exp(-d_A² / (2σ²)) + 1e-12
q_D = sigmoid((min(d_H, d_A) - d_dead) / 1.5) + 1e-12
q_D += w_dead + w_motion * sigmoid((motion0 - team_speed) / motion_scale)
     + w_edge * sigmoid((edge0 - edge) / edge_scale)
q_D += w_out * out_of_play
q   = [q_H, q_A, q_D] / sum(q)
logB = log(q)
```

The validated model sets `w_out = 1`, `w_dead = 0.02`, `w_motion = 1`, `w_edge = 0.5`,
`motion0 = 1.6 m/s`, `motion_scale = 0.35 m/s`. The motion and edge terms are the dead-ball
evidence model: they do not look at the ball at all, which is what lets `D` separate a stoppage
where the ball is visible and a player is standing next to it. `w_dead` is the dead-ball base
rate; it keeps the `D` prior independent of the scale of the team Gaussians. The speed gate is
applied even when `gate = 0`:

```
α = 1 - sigmoid((ball_speed - gate) / 2)
logB = α * logB + (1 - α) * log(1/3)
```

so a stationary or unknown-speed ball is blended 50/50 with uniform. This is a deliberate
regulariser and is required to reproduce the reference metrics.

Loose-ball rule. A ball far from every player is not evidence of a stoppage: it is loose or in
flight. When `d_min > loose_R = 7 m` and the ball is not out of play, the emission is uniform and
the transition prior carries the possession. If the ball is also slower than
`loose_speed = 2 m/s`, the only possession evidence left is the team that touched it last, so that
team gets `touch_p = 0.7` of the emission mass, provided the touch is younger than
`touch_age = 125` frames (5 s). The remaining mass is split evenly over the other two states. This
removed the false `D` stretches where a rolling or in-flight ball sits 7–10 m from both teams.

Missing-ball emissions:

* `persist`: uniform for every missing frame.
* `hybrid` (recommended): uniform for `gap ≤ 120` frames (4.8 s); for longer holes use

```
m_H = exp(-d_H_last² / (2σ_miss²)) + 1e-6
m_A = exp(-d_A_last² / (2σ_miss²)) + 1e-6
m_D = 0.15 + sigmoid((5 - last_edge) / 1.5)
m_H *= 1 + max(tag_vote, 0)
m_A *= 1 + max(-tag_vote, 0)
m_D *= 1 + 0.2 * tag_setpiece
```

with `σ_miss = 6 m`. If fewer than 6 players are visible, the frame is treated as missing
evidence and the emission is uniform; this avoids reading team shape from a near-empty tracking
feed. The public window has at least 19 visible players, so this guard does not affect public or
stress scores.

## Transition and decode

```
T = [[.995,   .0005, .001],
     [.0005,  .995,  .001],
     [.01,    .01,   .98]]
```

Viterbi runs over each graded half with a leading 500-frame context pad. Frames outside
`(11,185, 90,652)` and `(110,808, 190,198)` are `X`. Forward–backward posterior confidence is
computed for diagnostics only. After decoding, a run shorter than one second that is flanked by
the same label is absorbed by it: a few frames of `D` (or of the other team) between two equal
labels are a tracking or ball-in-flight flicker, not a possession change. `X` is never changed.

## Local synchronized reviewer

`python -m possession_lab viewer` builds a compact JSON payload from `outputs/predictions.csv`, the
public GT, the smart-tag table, and `metrics_public.json`, then serves it from a local stdlib HTTP
server. The page is plain HTML/CSS/JS; no frontend build step or extra Python dependency is added.

* **Synchronization.** `frame i` is exactly row `i` of the submission. Browser seeking uses
  `currentTime = (i + 0.5) / 25`; the displayed frame is
  `clamp(floor(currentTime * 25 + 1e-6), 0, 199318)`. The video duration is checked against
  `199,319 / 25 = 7972.76 s` when metadata loads.
* **Payload.** Labels are compact `H/A/D/X` strings; prediction segments/spells cover both full
  graded halves, while GT segments/spells cover only the public window because the public CSV has
  `X` outside it. Changes, disagreement runs (grouped in the browser), 638 tag intervals, and
  public metrics are included.
* **Streaming.** `/video` implements a single-range `Range` parser and returns `206`, `200`, or
  `416` as appropriate. The 5.2 GB tactical MP4 has its `moov` atom at the end; the browser reads
  the tail range for metadata and then fetches playback ranges. Routes are whitelist-only:
  `/`, `/static/viewer.css`, `/static/viewer.js`, `/api/viewer-data`, `/video`, `/favicon.ico`.
* **Scope.** The UI has no label editing, correction export, or broadcast-video path. GT is shown
  only on non-`X` public frames. The page works entirely from `127.0.0.1`.

The reviewer is covered by `tests/test_viewer.py`: payload semantics, public-only GT behavior,
range parsing (`normal`, open, suffix, invalid), HTTP `206`/`416`/`HEAD`, API payload, and
unknown-route rejection.

## Validation

* Unit tests: 32 stdlib `unittest` tests, all passing.
* Public metrics for the current parameters: spell F1 0.6074, macro F1 0.8925,
  D-F1 0.8584, 10 predicted spells.
* Stress (HMM gap-aware, current parameters): clean 0.6074 / random 0.4615 / long 0.4000 spell F1.
* The files in `reports/data/` were produced by the pre-retune parameter set (`σ=3.0`,
  `p_dh=0.001`, no dead-ball evidence, no loose-ball rule) and are kept as history, not as an
  equality target.
* Submission: 199,319 rows, frames 0–199,318, official validator `OK`.

## Full-match reconciliation

A previous reference run reported 256 spells and H 41.4% / A 32.5% / D 26.1%; the pre-retune run
reported 255 spells and H 40.5% / A 32.5% / D 26.9%; the previous parameter set reported 239
spells and H 44.1% / A 34.6% / D 21.3%. The current run reports 186 spells and
H 42.3% / A 31.9% / D 25.8%. One part of the earlier difference is frames 89,905–90,651: the ATD
has no home players and only 1–3 away players, and the ball is missing. The old run's NaN
emissions effectively reset to H. The sparse-tracking guard treats the stretch as missing evidence
and predicts D, which the public and stress metrics are insensitive to. The rest is the retune:
wider team Gaussian, easier exits from D, the dead-ball evidence model, and the loose-ball rule
that returns a slow ball far from every player to the team that touched it last. This is a
full-match structural diagnostic, not a score.

## Reproducibility and privacy

* `python -m possession_lab all --bootstrap-n 2000` regenerates every output.
* `python -m unittest discover -s possession_lab/tests -v` runs the tests.
* `python evaluation/validate_submission.py possession_lab/outputs/predictions.csv` runs the
  official format gate.
* The gated `data/` bundle and generated `outputs/` are not committed.
* No private full-match GT and no match video frames are used.
* CPU is intentional: the 3-state Viterbi pass takes ~1.8 s; a GPU transfer would dominate.
