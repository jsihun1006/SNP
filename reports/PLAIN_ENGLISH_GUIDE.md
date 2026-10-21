# Start Here — Plain-English Guide

**More than a Hack 2027 — Phase 1: Ball Possession**

This file explains, in simple terms:

- what you must deliver
- which video to use
- what the `.ipynb` notebooks are
- what the other data shows
- the pros and cons of each data source
- the pros and cons of each proposed approach
- what to do next

This guide is **not** one of the competition deliverables. It is a reading aid. The detailed research report is at `reports/possession_solution_report.md`.

---

## 1. The short version

You must hand in **three things**:

1. **A CSV file** with one label for every tactical-video frame.
2. **A public code repository** that can recreate that CSV.
3. **A short technical document** explaining what you built.

That is the whole submission.

The report I wrote earlier is for **choosing an approach**. It is not the submission itself.

---

## 2. The CSV file

The match has **199,319 tactical frames**.

Your file must have:

- header exactly: `frame,label`
- exactly **199,319 data rows**
- `frame` values **0 to 199318**, in order
- one label per row: **`H`, `A`, `D`, or `X`**

Example:

```csv
frame,label
0,X
1,X
...
11185,A
11186,A
...
199318,X
```

### The four labels

| Label | Meaning |
|---|---|
| `H` | Home team — Barcelona — has the ball; ball is in play |
| `A` | Away team — Athletic — has the ball; ball is in play |
| `D` | Dead ball — the ball is **not in play** |
| `X` | No answer / abstain |

**Important:** `D` does **not** mean “I cannot see the ball.”  
`D` means the ball is actually dead: out of play, whistle, stoppage, restart, etc.

### About `X`

- Before kickoff, halftime, and after the final whistle: `X` costs nothing.
- During graded live play: `X` is treated as “repeat your previous label.”
- In the middle of a possession, that is usually harmless.
- At a change of possession, it can cost you.

Simple rule:

> Use `X` when there is genuinely no play or you truly cannot answer. Otherwise predict `H`, `A`, or `D`.

### Validate before submitting

```bash
python evaluation/validate_submission.py yourteam.csv
```

If that fails, the scorer will reject the file.

---

## 3. How you are scored

The main score is **`spellF1`**.

Think of it as:

> Did you find the correct possessions, with roughly the correct start and end?

A **possession spell** is a continuous run of `H` or `A` while the ball is alive.  
`D` is a stoppage, not a possession.

The tie-break is **`macroF1`**:

> How many individual frames did you label correctly, averaged over `H`, `A`, and `D`?

So a model can have good frame accuracy but bad possession accuracy. That is why `spellF1` is the main score.

The public ground truth covers only a **5-minute window**:

- 7,500 frames
- 17 possessions
- 4 dead-ball spells

The rest of the match is not public. You get feedback only on that window.

---

## 4. Which video should you use?

**Use the tactical video.**

That is the reference timeline:

- tactical video = **199,319 frames**
- CSV row `0` = tactical frame `0`
- CSV row `199318` = tactical frame `199318`

If you use video at all, use the **tactical** video.

**Do not use the broadcast video for the submission timeline.**

The broadcast video:

- has **164,488 frames**
- has its own separate timeline
- has camera cuts and replays
- is **not synchronized** with the tactical video

It is optional extra input. If your CSV has about **164,000 rows**, you used the wrong video.

Simple rule:

> **Tactical video and tactical tracking define the frame numbers for your CSV.**  
> Broadcast video is optional extra information, not the submission clock.

You do not have to use video at all. You can build a prediction from the tactical tracking data, smart tags, and homography. But every frame number must still match the **tactical** video.

### Tactical vs broadcast — pros and cons

| Video | Pros | Cons |
|---|---|---|
| **Tactical** | Exact submission timeline; 199,319 tracking records; homography valid on 100% of frames; has smart tags | ~2 hours 13 minutes of video; video processing is slow; camera view is less “human-friendly” than TV |
| **Broadcast** | What a human analyst actually watches; useful for video-based methods; has its own tracking and homography | 164,488 frames on a separate timeline; cuts and replays; no smart tags; homography valid on only 65.8%; no mapping to tactical frames |

**Simple choice:** use tactical for the submission timeline. Use broadcast only if you have a specific video-based idea and a plan to map it back.

---

## 5. What are the `.ipynb` files?

They are **Jupyter notebooks** — interactive Python tutorials.

“`.ipynb`” means “interactive Python notebook.” Open them in Jupyter or VS Code and run them cell by cell.

There are three starter notebooks:

| Notebook | What it does |
|---|---|
| `starter/01_smart_tagging.ipynb` | Shows how to read and plot the tactical smart tags |
| `starter/02_tracking_atd.ipynb` | Shows how to load tracking data, plot players, and align frames |
| `starter/03_first_submission.ipynb` | Takes data in and produces a valid submission CSV |

**Start with `03_first_submission.ipynb`.** It is the practical one.

The notebooks are **learning aids**, not required deliverables. You can copy their code into a normal Python project.

---

## 6. The other data — what it shows

Besides the two videos, you get:

1. **Tracking data (ATD)**
2. **Smart Tagging**
3. **Homography**
4. **Metadata**

Simple summary:

> **Tracking = where. Smart tags = what kind of moment. Homography = camera map. Metadata = data dictionary.**

---

### 6.1 Tracking data (ATD)

**What it is:** automatic computer-vision output from Metrica.

For every frame it gives:

- ball position
- each visible player’s position
- each player’s speed
- team membership

Example, simplified:

```text
frame 50000:
  ball: x=0.35, y=0.46
  Track 7: x=0.32, y=0.44, speed=4.1 m/s, team=Barça
  Track 12: x=0.55, y=0.50, speed=3.8 m/s, team=Athletic
```

Coordinates are **0–1**, not metres. Multiply by the pitch size to get metres.

The obvious use:

> At this frame, which team has a player closest to the ball?

That is the basic possession signal.

#### Pros

- Objective, numerical, easy to process.
- Works without watching the video.
- Has the ball and player positions in the same coordinate system.
- Fast compared with video processing.
- Team membership is included.

#### Cons and catches

- The ball is only visible on **56.9% of all frames** and **59.3% of graded frames**.
- The ball disappears most often when it matters: crowded boxes, occlusions, aerial balls.
- There are **46 track slots** for 22 players.
- `Track 7` is **not shirt number 7** and **not a fixed person**.
- Slots appear, disappear, and swap identity.
- You get team membership, not player identity.
- Positions are automatic and noisy.
- Some player positions are missing too.

So do not build “passes between player 7 and player 12.” Build collective things like “nearest Barça player to the ball.”

---

### 6.2 Smart Tagging

**What it is:** Metrica’s automatic tactical annotation of the tactical video.

It gives **638 intervals** covering phases and set pieces, each with a code and tags.

Examples of codes:

- Build Up
- Progression
- Ball in Final Third
- Ball in the Box
- Set Pieces
- Attacking Transition
- Defensive Transition
- Goals

Examples of tags:

- `Team`: Barcelona / Athletic / N/A
- `Half`: 1st Half / 2nd Half
- `Type`: Throw-in / Corner / Goal Kick / Free Kick / Kick Off
- `Side`: Central / Left / Right
- `Direction of ball entry`
- `Max Players in the box`

#### Pros

- Free context that geometry cannot see.
- Helps detect restarts and set pieces.
- Strong prior for dead-ball moments.
- Every phase has a team tag, even if sometimes `N/A`.
- Useful when the ball is missing.

#### Cons and catches

- It is **automatic**, not ground truth. It contains false positives and false negatives.
- A tag interval is a **phase**, not a possession. Do not copy it directly as your answer.
- A Build Up can span one possession, stop short of one, or cover two.
- Coverage is **75.5% of public frames** and **70.1% of full-match frames**.
- The team vote is only **69.6% accurate** on `H/A` frames.
- **36 intervals are tagged `N/A`**, including **every throw-in** — exactly where team information would be useful.
- Box-entry tags are less reliable than build-up tags.

Use it as a **hint**, not as the answer.

---

### 6.3 Homography

**What it is:** a **3×3 matrix per frame** that converts:

> pitch position → pixel position in the video

or the reverse:

> pixel position in the video → pitch position

It is the bridge between the tracking data and the video image.

You use it if you want to:

- draw tracking positions onto the video
- check whether the tracker is correct
- build a video-based method
- project a pitch location into the camera view

#### Pros

- Connects tracking coordinates to actual video pixels.
- Useful for visual checks and overlays.
- Tactical homography is valid on **100%** of frames.
- Makes it possible to build a video-based or hybrid method.

#### Cons and catches

- It does **not** tell you who has possession.
- Broadcast homography is valid on only **65.8%** of frames.
- Always check the `valid` column before using a matrix.
- `reprojection_error` tells you how accurate the mapping is — lower is better.
- It is a camera mapping, not a label source.

---

### 6.4 Metadata

The tracking data comes as a pair:

- `atd.xml` — metadata
- `atd.txt` — raw tracking rows

The metadata includes things like:

- pitch dimensions: **105 × 68 m**
- frame rate: **25 fps**
- team information
- file structure

It does **not** include player names or reliable shirt numbers.

`kloppy` can read the XML/TXT pair directly. The starter scripts also show how to load it.

---

## 7. How the data lines up

This part is important:

| Source | How to convert to submission frame |
|---|---|
| Submission CSV | frame `0 … 199318` |
| Tracking ATD | `video_frame = frame_id - 1` |
| Smart Tagging | `video_frame = round(seconds × 25)` |
| Homography | already frame `0 … 199318` |

So:

- tracking frame IDs start at **1**
- submission frames start at **0**
- smart tags are in **seconds**
- homography is already in submission frames

Everything is on the **tactical video timeline**. Broadcast data is a separate timeline.

The graded periods are:

| Period | Frames | Graded? |
|---|---:|---|
| Before the match | 0–11,184 | No |
| First half | 11,185–90,651 | Yes |
| Halftime | 90,652–110,807 | No |
| Second half | 110,808–190,197 | Yes |
| After the final whistle | 190,198–199,318 | No |

---

## 8. How the data works together

A simple possession pipeline could be:

1. **Tracking** — find the ball and the nearest Barça/Athletic player.
2. **Smart tags** — tell the model “this is a set piece” or “this is a restart,” which helps with `D`.
3. **Homography** — optional: draw the result on the video or use camera information.
4. **Temporal model** — smooth the frame-by-frame decisions so possessions do not flicker.

Simple example:

> At frame 50,000, tracking says the ball is near a Barcelona player and the nearest Athletic player is far away. Smart tags say “Ball in Final Third, Team Barcelona.” Homography says where that moment appears in the video. The model outputs `H`.

But the final decision still comes from your algorithm. These files are **evidence**, not labels.

---

## 9. What you do **not** get

You do **not** get:

- player names
- reliable shirt numbers
- a ready-made possession label
- perfect ball tracking
- perfect smart tags
- a mapping from broadcast frames to tactical frames
- private full-match ground truth

So the data tells you **where things are and what kind of phase it is**, but it does not tell you the answer.

---

## 10. The proposed approaches — pros and cons

### 10.1 Gap-aware hybrid HMM — recommended

**What it proposes:**

Combine three things:

1. **Possession-zone evidence** — distance from the ball to the nearest player of each team.
2. **A three-state HMM** (`H`, `A`, `D`) with a strong self-transition prior, so labels do not flicker.
3. **Explicit missing-ball handling** — short gaps use persistence; long gaps use player proximity to the last known ball position, smart-tag context, and edge/out-of-play evidence.

**Public result:**

- `spellF1`: **0.4966**
- `macroF1`: **0.7709**
- `D-F1`: **0.5916**
- predicted spells: **12** vs 17 in the ground truth

**Stress result:**

- clean: **0.4966**
- random missing gaps: **0.5286**
- long missing gaps: **0.3520**

For comparison, the best public baseline gets:

- clean: **0.4485**
- random gaps: **0.2900**
- long gaps: **0.2452**

**Full-match structural check:**

- 256 H/A spells
- mean spell length 18.4 s
- H 41.4% / A 32.5% / D 26.1%

**Pros:**

- Best risk-adjusted result.
- Fast and explainable.
- Handles missing ball explicitly.
- Does not need a GPU or large labeled dataset.
- Works with the data you already have.
- Easy to reproduce.

**Cons:**

- Emissions are hand-designed.
- Under-segments some short possessions: 12 predicted spells vs 17.
- Public score is noisy because there are only 17 possessions.
- Needs careful tuning of gap rules and transition probabilities.
- Not as high a ceiling as a learned trajectory model.

**Verdict:** best first choice for this team and this deadline.

---

### 10.2 PathCRF / Ball Radar — highest ceiling

**What it proposes:**

Learn possession from **player trajectories**, even when the ball is missing.

- **PathCRF** models player trajectories as a dynamic graph and selects the most likely possession edge at each time step.
- **Ball Radar** uses a Set Transformer and hierarchical Bi-LSTM to infer the missing ball trajectory and player possession.

**Published results:**

- **PathCRF:** 69.64% possession-edge accuracy, 75.69% event F1, 22.88 s to decode a full match on an RTX 4090.
- **Ball Radar:** mean ball-position error below 3.7 m, 64.7% possession accuracy.

**Pros:**

- Highest ceiling.
- Directly addresses the missing-ball problem.
- Uses player context, not just the ball.
- Strong published evidence.
- Public code is available for both.

**Cons:**

- Needs large labeled tracking datasets.
- Needs GPU training and significant engineering.
- Integration risk is high.
- The published numbers are not on this challenge’s metric.
- Hard to finish well before the deadline without an ML specialist.

**Verdict:** good only if you have an ML engineer, a GPU, and access to external labeled data.

---

### 10.3 Rule-based possession zone / last touch — fastest fallback

**What it proposes:**

Assign the ball to the team of the nearest player inside a control radius. Smooth the result over time.

**Our last-touch result:**

- `spellF1`: **0.4364**
- `macroF1`: **0.5254**
- predicted spells: **16**

**Pros:**

- Very simple and interpretable.
- Fast to build.
- No training.
- Easy to explain in the technical document.

**Cons:**

- Depends on the ball being visible.
- Weak over long missing-ball gaps.
- Weak dead-ball detection.
- Contested touches are ambiguous.
- Lower ceiling than the HMM.

**Verdict:** good 48-hour fallback or feature layer, not the best final answer.

---

### 10.4 Learned TCN/CRF with external labels

**What it proposes:**

Train a temporal neural network on tracking sequences to predict possession.

**Published result:**

- Borghesi et al. report **86.2% possession accuracy** and **89.2% dead-alive accuracy** with enough labeled data.

**Our 5-minute experiment:**

- raw ML: **0.1269** spellF1
- ML + HMM smoothing: **0.2643**
- group-held-out CV: **0.0543 / 0.1086**

**Pros:**

- Strong published results.
- Can learn complex player context.
- Scales with more data.

**Cons:**

- Needs many labeled matches.
- The public 5-minute window is far too small.
- Our ML proxy overfits the window.
- Needs GPU and careful validation.

**Verdict:** only with external labeled data. Not viable with only the public window.

---

### 10.5 Pitch control + HMM

**What it proposes:**

Estimate which team controls each area of the pitch, then use that to decide who is likely to reach a loose ball.

**Pros:**

- Uses player positions only.
- Principled spatial model.
- Useful when the ball is missing.
- Can be combined with the HMM.

**Cons:**

- Needs calibration and compute.
- Unproven benefit on this exact metric.
- More complex than simple distance.
- Hard to tune before the deadline.

**Verdict:** optional feature, not the primary solution.

---

### 10.6 Video-only detection

**What it proposes:**

Detect players and the ball directly from video, track them, classify teams by jersey colour, and infer possession.

**Published result:**

- Guo et al. report 79.4% and 71.1% validation average precision for the detectors and **RMSE 4.87** for possession estimation on 20 full-match clips.

**Pros:**

- Independent of the provided tracking.
- Can use the broadcast video.
- Strong computer-vision literature.
- Useful if tracking quality is the main problem.

**Cons:**

- Broadcast has cuts, replays, and no sync.
- Needs homography, detection, tracking, team classification, and temporal logic.
- Heavy compute.
- Separate project from the possession model.
- Hard to finish and validate before the deadline.

**Verdict:** not the first move unless the team is a computer-vision team.

---

### 10.7 Smart-tag-only

**What it proposes:**

Use the smart tags directly to vote for the possessing team or dead ball.

**Our result:**

- `spellF1`: **0.2500**
- `macroF1`: **0.4893**
- 7 predicted spells

**Pros:**

- Very cheap.
- Adds phase context.
- Good for restarts and dead-ball moments.

**Cons:**

- Only 70–75% coverage.
- Team accuracy is 69.6%.
- A phase is not a possession.
- 36 intervals are `N/A`, including all throw-ins.
- Poor standalone score.

**Verdict:** prior only. Never the main model.

---

### 10.8 Ball imputation

**What it proposes:**

Reconstruct the missing ball trajectory from player trajectories, then apply a possession rule.

**Published evidence:**

- Ball Radar: below 3.7 m mean error, 64.7% possession accuracy.
- MIDAS: multi-agent trajectory imputation.

**Pros:**

- Directly addresses missing ball.
- Can improve short-gap features.
- Good published research.

**Cons:**

- Long gaps are ill-posed: after 20 seconds, many ball paths are plausible.
- Our imputation experiment gave no gain over smoothing.
- It can destroy dead-ball detection: in our check, a pure imputation variant had **D-F1 = 0**.
- Adds complexity and failure modes.

**Verdict:** short-gap preprocessing only.

---

## 11. Ranking

| Rank | Approach | Public evidence | Effort | Verdict |
|---:|---|---|---|---|
| 1 | **Gap-aware hybrid HMM** | 0.4966 spellF1, 0.7709 macroF1, 0.5916 D-F1 | Low | **Recommended** |
| 2 | **PathCRF / Ball Radar** | 69.64% edge accuracy, 75.69% event F1 | High | High-ceiling alternative |
| 3 | **Rule-based last touch** | 0.4364 spellF1 | Very low | Fallback |
| 4 | **Learned TCN/CRF** | 86.2% accuracy with labels | High | Needs external data |
| 5 | **Pitch control + HMM** | Unproven on this metric | Medium | Optional feature |
| 6 | **Video-only** | RMSE 4.87 on 20 matches | High | Separate project |
| 7 | **Smart tags only** | 0.2500 spellF1 | Very low | Prior only |
| 8 | **Ball imputation only** | No gain; D-F1 = 0 | Medium | Short gaps only |

---

## 12. The honest caveat

The public ground truth has only **17 possessions**.

That means the public score is noisy. The 95% bootstrap interval for the final HMM is **[0.303, 0.693]**. It overlaps the baselines.

So do not treat the public score as a leaderboard. The recommendation rests on:

1. the public score
2. the missing-ball stress results
3. the full-match structural check

That is why the gap-aware HMM is recommended over the simpler baseline, even though the public numbers are not decisive on their own.

---

## 13. Rules that can get you disqualified

- **Do not label the match by hand.** The labels must come from an algorithm.
- **Do not use the private full-match ground truth.** You only get the public 5-minute window.
- **Do not put the match data in your public repository.**
- **Do not submit a CSV your code cannot reproduce.**

Small amounts of hand-labelled data for training or validation are fine. Hand-writing the final answer is not.

---

## 14. What to do next

- [ ] Choose one approach.
- [ ] Build code that outputs `teamname.csv`.
- [ ] Run `python evaluation/validate_submission.py teamname.csv`.
- [ ] Push the code to a public repository.
- [ ] Write the short technical document.
- [ ] Submit the Google Form before **17 November 2026**.

The technical document should cover:

- what you built and why
- which data and assets you used
- your possession rules
- limitations
- how to reproduce the result

---

## 15. Key references

The full reference list is in `reports/possession_solution_report.md`. The most relevant sources are:

1. Link, D., & Hoernig, M. (2017). *Individual ball possession in soccer*. PLOS ONE. DOI: `10.1371/journal.pone.0179953`
2. Vidal-Codina, F., et al. (2022). *Automatic event detection in football using tracking data*. Sports Engineering. DOI: `10.1007/s12283-022-00381-6`
3. Borghesi, A., et al. (2023). *Using Temporal Convolutional Networks to estimate ball possession in soccer games*. Expert Systems with Applications. DOI: `10.1016/j.eswa.2023.119780`
4. Lang, S., et al. (2022). *Predicting the in-game status in soccer with machine learning using spatiotemporal player tracking data*. Scientific Reports. DOI: `10.1038/s41598-022-19948-1`
5. Kim, H., et al. (2023). *Ball Radar: Learning to Infer Ball Trajectories from Multi-Agent Trajectories*. KDD. DOI: `10.1145/3580305.3599779`
6. Kim, H., et al. (2026). *PathCRF: Ball-Free Possession Path Inference*. KDD. DOI: `10.1145/3770855.3818152`
7. Guo, et al. (2026). *Automatic estimation of football possession via improved YOLOv8 detection and DBSCAN-based team classification*. Sensors. DOI: `10.3390/s26041252`

Public code:

- Ball Radar: `https://github.com/hyunsungkim-ds/ballradar`
- PathCRF: `https://github.com/hyunsungkim-ds/pathcrf`

---

## 16. One-sentence summary

> Use the **tactical** video and tactical tracking for the CSV timeline, combine tracking with smart tags and optional homography, and — unless you have an ML team and external labels — start with the **gap-aware hybrid HMM**.
