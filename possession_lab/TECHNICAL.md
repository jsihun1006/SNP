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
* `gap`: frames since the last observed ball.
* `edge`, `last_edge`: distance of the ball (or last known ball) from the nearest pitch boundary.
* `out_of_play`: observed ball outside `[0,L] × [0,W]`.
* `last_touch`, `last_touch_age`: nearest player within 0.8 m.
* `d_H_last`, `d_A_last`: current team shape around the last observed ball.
* `tag_vote`, `tag_setpiece`, `tag_coverage`.
* `player_count`: visible player slots, used for the sparse-tracking guard.

## Emissions

At visible frames, with `σ = 3 m`, `d_dead = 7 m`:

```
q_H = exp(-d_H² / (2σ²)) + 1e-12
q_A = exp(-d_A² / (2σ²)) + 1e-12
q_D = sigmoid((min(d_H, d_A) - d_dead) / 1.5) + 1e-12
q_D += w_out * out_of_play
q   = [q_H, q_A, q_D] / sum(q)
logB = log(q)
```

The validated model sets `w_out = 1`. The speed gate is applied even when `gate = 0`:

```
α = 1 - sigmoid((ball_speed - gate) / 2)
logB = α * logB + (1 - α) * log(1/3)
```

so a stationary or unknown-speed ball is blended 50/50 with uniform. This is a deliberate
regulariser and is required to reproduce the reference metrics.

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
     [.001,   .001,  .998]]
```

Viterbi runs over each graded half with a leading 500-frame context pad. Frames outside
`(11,185, 90,652)` and `(110,808, 190,198)` are `X`. Forward–backward posterior confidence is
computed for diagnostics only.

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
* **User truth (UT).** The browser can paint `H/A/D/X` on the current frame or an anchored range,
  from the buttons or globally from the keyboard. Each label press leaves an anchor at the end of
  the written span; run boundaries are drawn on both timelines and can be dragged to move a cut.
  `POST /api/user-truth` updates the UT array (paint, boundary move, or clear), writes
  `outputs/user_truth.json` atomically, and returns recomputed UT segments and official-scorer
  metrics over exactly the tagged frames. The UT is viewer-only: `features.py`, `model.py`, and
  `pipeline.py` never import or read it, and the fixed `predictions.csv` is never modified.
* **Streaming.** `/video` implements a single-range `Range` parser and returns `206`, `200`, or
  `416` as appropriate. The 5.2 GB tactical MP4 has its `moov` atom at the end; the browser reads
  the tail range for metadata and then fetches playback ranges. Routes are whitelist-only:
  `/`, `/static/viewer.css`, `/static/viewer.js`, `/api/viewer-data`, `/api/user-truth`, `/video`,
  `/favicon.ico`.
* **Scope.** The UI edits only the UT, has no correction export, and has no broadcast-video path.
  GT is shown only on non-`X` public frames. The page works entirely from `127.0.0.1`; the POST
  endpoint requires `application/json`.

The reviewer is covered by `tests/test_viewer.py`: payload semantics, public-only GT behavior,
UT scoring/persistence/editing, range parsing (`normal`, open, suffix, invalid), HTTP
`206`/`416`/`HEAD`, API payload, and unknown-route rejection.

## Validation

* Unit tests: 36 stdlib `unittest` tests (17 HMM + 19 viewer), all passing.
* Public prediction: bit-for-bit equal to `reports/data/final_window_codes.npy`.
* Public metrics: exact match to the reference (spell F1 0.4965517241, macro F1 0.7709254257,
  D-F1 0.5916230366, 12 predicted spells).
* Stress table: every row matches `reports/data/final_stress.csv` exactly.
* Submission: 199,319 rows, frames 0–199,318, official validator `OK`.

## Full-match reconciliation

A previous reference run reported 256 spells and H 41.4% / A 32.5% / D 26.1%. This run reports
255 spells and H 40.5% / A 32.5% / D 26.9%. The difference is frames 89,905–90,651: the ATD has no
home players and only 1–3 away players, and the ball is missing. The old run's NaN emissions
effectively reset to H. The sparse-tracking guard treats the stretch as missing evidence and
predicts D. This is a full-match structural diagnostic; the public and stress metrics are unchanged.

## Reproducibility and privacy

* `python -m possession_lab all --bootstrap-n 2000` regenerates every output.
* `python -m unittest discover -s possession_lab/tests -v` runs the tests.
* `python evaluation/validate_submission.py possession_lab/outputs/predictions.csv` runs the
  official format gate.
* The gated `data/` bundle and generated `outputs/` are not committed.
* No private full-match GT and no match video frames are used.
* CPU is intentional: the 3-state Viterbi pass takes ~1.8 s; a GPU transfer would dominate.
