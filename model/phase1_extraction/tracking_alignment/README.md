# tracking_alignment — tactical/broadcast frame alignment (Phase 1)

Aligns the provided **broadcast** ATD to the **tactical** ATD frame by frame, half by half, so that any
broadcast-derived signal can be joined to the tactical frame index that the submission uses.
It does not detect anything in the video and does not predict possession. Author: team member KKJ.

## Method

- Half 1 starts at the official tactical start (`games.py`). Half 2 starts at the earliest Smart Tagging
  `Half = 2nd Half` interval.
- For each half the broadcast offset is searched independently from **forward ball trajectories and team-wise
  player layouts** (`find_joint_tracking_match.py`, `find_layout_match.py`); the next ball window is a separate
  consistency check. Mirrored pitch orientations are tried (`tracking_utils.FLIPS`).
- After each anchor the mapping assumes constant timing to the official half end. No manual anchors and no
  ground truth are used; ATD files are never modified and missing coordinates stay missing.

## Run

```bash
python -m pip install -r model/phase1_extraction/tracking_alignment/requirements.txt
python model/phase1_extraction/tracking_alignment/final_tracking_alignment.py
python -m unittest discover -s tests -v          # synthetic tests, no match data
```

Needs `data/tactical/` and `data/broadcast/` ATD TXT/XML and the tactical Smart Tagging XML.

## Output: `outputs/phase1/tracking_alignment/sync/` (git-ignored)

- `final_tracking_alignment.csv` — one row per tactical frame: tactical and broadcast frame indices, half,
  status, offset, orientation index, initial labels (`X` outside tagged halves, blank inside)
- `final_tracking_alignment_report.json` — anchors, search scores, consistency checks

## Files

| file | role |
|---|---|
| `final_tracking_alignment.py` | entry point: per-half anchors and the frame table |
| `find_joint_tracking_match.py`, `find_layout_match.py`, `align_half_starts.py` | offset search |
| `tracking_utils.py` | ATD I/O, team mapping, masks, orientations |
| `compare_tracking_player.py` | side-by-side MP4 export of both sources using the frame table (optional) |
| `HALF_ALIGNMENT.md` | detailed alignment rules (written before the move; the movement-vector and team-motion viewers it mentions are not part of this submission) |

## Status in the submission

The current possession pipeline (`run_submission.py`) reads the tactical ATD only, so this alignment does not
change `SNP.csv`. It is included so that broadcast-based components can be joined on the tactical frame index.
Limitations: a constant offset per half is assumed after the anchor; full-half synchronisation is not proven.
