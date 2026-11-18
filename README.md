# More than a Hack 2027 — Phase 1: Ball Possession

**Organised by [Metrica Sports](https://metrica-sports.com), [FC Barcelona](https://www.fcbarcelona.com)
and the [Mobile World Capital Barcelona](https://www.mwcbarcelona.com)**, at the Talent Arena in Barcelona.

A football data challenge on real professional match data, open to analysts, data scientists and
engineers. You are given one complete LaLiga match — two camera feeds, automatic tracking of every
player and the ball, automatic tactical tagging, and the camera-to-pitch geometry — and one
question to answer with it.

---

## The task

**For every frame of the match, say who has the ball.**

```
frame,label
11185,A        away in possession, ball alive
11186,A
...
18240,D        dead ball
...
18684,H        home in possession, ball alive
```

That is the whole task. One CSV, one row per video frame, 199,319 rows. It is scored automatically
against a private frame-by-frame ground truth, so there is a number at the end and no argument
about it — no jury, and no presentation to argue from.

The tracking is automatic and uncorrected: it has gaps and identity
switches, and the ball itself is invisible on 41% of the frames you are scored on.

---

## The challenge in one screen

| |                                                                                                                           |
|---|---------------------------------------------------------------------------------------------------------------------------|
| **Input** | one full match: tactical video, broadcast video, automatic tracking **with the ball** for both, Smart Tagging, homography |
| **Output** | `frame,label` CSV, 199,319 rows, labels `H` / `A` / `D` / `X`                                                             |
| **Ranked on** | `spellF1` — how many of the match's real possessions you recovered as whole possessions                                   |
| **Also reported** | `macroF1` — the per-frame view, and the first tie-break                                                                   |
| **Baseline to beat** | `spellF1` **0.2433**, in this repository                                                                                  |
| **Ground truth you get** | the first **5 minutes** of play — to check your pipeline, not to train on                                              |
| **Ground truth you are scored on** | the full match, **private**                                                                                               |

It is deliberately **not** a tracking challenge. The tracking, tagging and homography are given to
you so that nobody has to rebuild detection, re-identification and camera calibration before
getting to the actual question. See [CHALLENGE.md](docs/competition/CHALLENGE.md).

---

## Try it right now

The scorer, a 5-minute ground truth and four working baselines are in this repository. No
enrolment, no download, no data:

```bash
pip install -r requirements.txt
python evaluation/run_evaluation.py
```

Scored on the public 5 minutes, not on the full match -- which is why the order below is not the
order of the floor quoted above:

```
scoreboard  gt/BAR-ATH_possession_gt_public.csv  199319 frames  7500 graded  21 gt segments
ground truth has 17 possessions of mean 14.2 s  <- compare the poss and dur columns

rank  submission                macroF1  spellF1   poss     dur  
----  ------------------------  -------  -------  -----  ------  
   1  baseline_smooth125         0.5812   0.4242     16    18.5  
   2  baseline_smooth50          0.6207   0.3256     26    10.8  
   3  baseline_smooth25          0.6200   0.2868     36     7.6  
   4  baseline_raw               0.5972   0.0378    131     2.0  
```

Those four are **the same algorithm four times** — nearest tracked player to the ball, then a
majority vote over a window. Only the window changes. `macroF1` barely moves; `spellF1` moves by a
factor of eleven. That disagreement is the whole point of the metric, and
[EVALUATION.md](docs/competition/EVALUATION.md) explains it.

Once you have the data, regenerate them yourself and go further:

```bash
python starter/baseline.py                              # rebuild the four baselines
python evaluation/validate_submission.py <your>.csv     # before every submission
```

**Validate every file before you submit it.** `validate_submission.py` runs exactly the checks the
official scorer runs, and a file that fails them is dropped from the board with a one-line reason.
It takes two seconds and it is the single most common way a good entry scores nothing.

---

## 연구 작업

연구 저장소는 `jsihun1006/SNP`의 `JSH`에서 관리한다. 제출용 저장소는 `stanliy/SNP-MTAH2027`이다.

| 경로 | 내용 |
|---|---|
| [src/common](src/common) | DFL·Metrica 로더, 공통 좌표·특징 처리 |
| [src/deadball](src/deadball) | 데드볼 학습·추론·보정 |
| [src/possession](src/possession) | 팀 소유 모델과 전환 보정 |
| [src/tracking](src/tracking) | 볼·선수 검출, 좌표 처리, 영상 시간축 정렬 |
| [experiments](experiments/README.md) | 과제 / 모델 / 실행별 설정과 평가 지표 |
| [notebooks](notebooks) | 과제별 Colab·분석 노트북 |
| [docs](docs) | 대회 규정과 연구 기록 |
| [data](data/README.md) | 로컬 원본 경기 데이터와 전처리 캐시 |
| `artifacts/` | 로컬 가중치, 프레임별 예측, 로그와 생성 결과 |

각 모델의 README에 입력·실행·출력 경로를 기록한다. 이전 경로로 직접 실행하거나 가져오는 스크립트는 상대 링크로 원본 코드를 참조한다. `mp4/BAR-ATH_tactical-001.mp4`는 표준 입력명 `data/tactical/BAR-ATH_tactical.mp4`를 참조한다. `data/public` 아래의 동일한 공개 평가 자료 6개는 `evaluation/`의 원본을 참조한다.

- YOLO11n과 football YOLOv8x는 공통 검출·검사 코드를 쓰고 모델마다 클래스, 임계값, 가중치를 지정한다. [검출 모델 안내](src/tracking/detection/README.md)
- Dead Final 1은 동일 가중치 7쌍을 공유한다. 후보 25fps 모델은 학습 때의 `best.pt`를 추론에서도 사용한다. [Dead Final 1 기록](experiments/deadball/transportmer/dead_final_1/README.md) · [소유 모델 안내](src/possession/possession/README.md)
- SHA-256이 같은 과거 실행 입력·평가 자료 46개는 APFS에서 저장 공간을 공유하도록 정리했다. 각 경로의 파일은 개별적으로 갱신할 수 있다. 중복 목록과 이동 전후 경로는 로컬 `artifacts/organization/duplicate-storage.json`, `artifacts/organization/moves.json`에 있다.

Git에는 코드·설정·문서·결과 집계와 공식 평가 예제만 둔다. 기존 DFL 압축본과 모델 가중치는 기존 작업 폴더에 있으므로, 새 체크아웃에서 학습·추론하려면 각 README에 적힌 입력과 가중치도 준비해야 한다. [데이터 준비](data/README.md)

---

## Getting the data

The match assets are released to enrolled participants who have accepted the terms of use.

1. Enrol: **[More than A Hack 2027](https://landing.mobileworldcapital.com/more-than-a-hack-2027)**
2. Read and sign the terms and conditions in the registration form
3. You receive a download link.
4. Unpack it into `./data`, or point `MTAH_DATA_DIR` at wherever you put it.

Full description of every asset and how the frames line up: **[DATA.md](docs/competition/DATA.md)**.

---

## Read these, in this order

| | |
|---|---|
| **[CHALLENGE.md](docs/competition/CHALLENGE.md)** | the problem, the four labels, what counts as possession |
| **[DATA.md](docs/competition/DATA.md)** | every provided asset, its format, and the frame alignment |
| **[EVALUATION.md](docs/competition/EVALUATION.md)** | how you are scored, and why — read before optimising anything |
| **[SUBMISSION.md](docs/competition/SUBMISSION.md)** | the file contract, the three deliverables, how to submit |
| **[RULES.md](docs/competition/RULES.md)** | what you may and may not do |
| **[FAQ.md](docs/competition/FAQ.md)** | |
| **[OVERVIEW.md](docs/competition/OVERVIEW.md)** | the competition as a whole — context, why the 2027 format changed, the calendar |


---

## What is in this repository

```
games.py                    per-match constants, asset paths and the frame alignment
evaluation/
  possession_metrics.py     the metric library — every number on the board is defined here
  run_evaluation.py         the scorer. This exact file produces the official result
  validate_submission.py    check a file before you send it
  gt/                       the public 5-minute ground truth
  results/                  the four baselines; put your submissions here
starter/
  baseline.py               the reference baseline, numpy only
  load_atd.py               read the tracking and line it up with video frames
  load_smart_tagging.py     read the tactical tagging as video-frame intervals
  load_homography.py        read the homography; project pitch <-> pixels
  utils.py                  pitch plotting for the notebooks
  01_smart_tagging.ipynb    guided tour of the tactical tagging
  02_tracking_atd.ipynb     guided tour of the tracking data, with plots
  03_first_submission.ipynb data in, scored submission out — start here
data/                       unpack the downloaded match assets here. Empty in a fresh clone
```

The scorer you run locally is the scorer that decides the competition. If it scores on your
machine, it scores the same way on ours.

---

## Requirements

**Python 3.8 or newer.** The evaluation needs `numpy` and `pandas` and nothing else; it is tested
under 3.8 and 3.11 and produces identical output under both. `starter/baseline.py` needs nothing
more either — you can reach a scored submission without installing anything else.

The notebooks additionally want `kloppy` and `matplotlib` (`requirements-starter.txt`), and
**Python 3.9+ is recommended for those**: on Windows, installing Jupyter under 3.8 hits a
`pywinpty` build failure that has nothing to do with this repository.

---

## Timeline

| |                |
|---|----------------|
| registration opens | **October 1, 2026** |
| data released | **October 20, 2026** |
| submission deadline | **November 17, 2026** |
| results announced | **Before November 26, 2026**      |

---

## Questions

Open an issue for anything about the data, the metrics or the documentation — corrections are
genuinely useful, and a fix that helps you helps everyone equally. For enrolment and the terms of
use: **hackathon.talentarena@mobileworldcapital.com**.

## Licence and citation

The code and documentation in this repository are released under the MIT Licence — see
[LICENSE](LICENSE). **The match data is not covered by it**: it is separately licensed for this
competition only, under the terms you accept at enrolment.

```bibtex
@misc{mtah2027,
  title  = {More than a Hack 2027: Phase 1 - Ball Possession},
  author = {Metrica Sports and FC Barcelona Innovation Hub},
  year   = {2027},
  url    = {https://github.com/metrica-sports/more-than-a-hack-2027}
}
```
