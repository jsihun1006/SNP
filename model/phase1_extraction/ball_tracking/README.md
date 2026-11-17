# ball_tracking — verified tracking sources and checkpoint export

This Phase 1 module contains the **49 verified sources used to produce the retained V3 ball + Tactical
player tracking**, plus a submission-layout export/validation interface. The interface reads the existing
retained tracking checkpoint and re-indexes it on the Tactical timeline; it does not run a new detector,
change the ball coordinates, smooth a trajectory or infer possession.

**Status:** a checkpoint is required. This module is **not called by `run_submission.py`**, so adding it
does not change the current `SNP.csv`. The Phase 2 decoder still reads the provided Tactical ATD. The
historical sources are preserved for provenance; a fresh raw-video rebuild of their private input/model/
review dependencies has not been integrated into this entry point.

## Layout

```text
extract.py                     checkpoint -> Phase 1 tracking export
validate.py                    saved export/source preservation checks
config.json                    source identity, timeline, units and output settings
requirements.txt               pinned dependencies for this adapter
CODE_FILES.txt                 the original verified 49-source inventory
CODE_MANIFEST.json             historical source roles, inputs, outputs and hashes
MODULE_INTEGRATION.json        relocation + adapter provenance
starter/, games.py             retained source originals (byte-for-byte)
yolo26_football_detection/      retained actual YOLO26X detector source
docs/                          confirmed method and historical input audit
```

## Inputs

Keep private assets outside the model directory, under the repository root:

- `outputs/unified_tracking_v1/tracking_all_frames.csv` — the accepted Broadcast-timeline checkpoint,
  164,487 frames. The default configuration checks its recorded SHA256
  `bdd017ba3f8335028a951631768780af570e71bf064c7bb850572af5419b0608`.
- `outputs/unified_tracking_v1/roster.json` — the original Tactical slot/team metadata, not verified
  player identities or shirt numbers.

These are derived private inputs, not part of the official raw bundle and not committed here. A clean
clone with only the official raw assets cannot run this checkpoint adapter until the accepted checkpoint
is supplied privately. The adapter does not silently fall back to another tracking source.

## Install and run

From the repository root, Python 3.11:

```bash
python -m pip install -r model/phase1_extraction/ball_tracking/requirements.txt
python model/phase1_extraction/ball_tracking/extract.py --run-id retained_v3
python model/phase1_extraction/ball_tracking/validate.py --run-id retained_v3
```

`--config` selects another explicitly configured source identity. `--input-csv` and `--roster` select
private input locations; the configured source integrity checks still apply. The default repository root
is resolved from this module's location, and `--repo-root` supports a separate private workspace.
Existing output directories are refused rather than overwritten. See `--help` for the supported options.

## Output contract

`outputs/phase1/ball_tracking/<run_id>/` (git-ignored):

| file | content |
|---|---|
| `tracking.csv` | one row for each available player/ball XY pair, on the Tactical frame index |
| `frame_status.csv` | all 199,319 Tactical frames, including unmatched/no-position frames |
| `run_manifest.json` | exact inputs, hashes, settings, output counts and source semantics |
| `validation.json` | independent saved-output/source checks |

Core tracking columns are
`frame,time_seconds,track_id,id_kind,object_type,team,x_norm,y_norm,x_m,y_m,confidence,observed`.
Additional columns retain the Broadcast source frame, prediction flag, source name and speed channel.

- `frame` is Tactical video frame0; `time_seconds = frame/25`. The source checkpoint's `frame` is
  Broadcast frame0 and is retained separately. Its explicit `tactical_frame` mapping is used.
- Unmapped Broadcast rows have no Tactical time assigned. They are counted in the manifest and are not
  pasted onto a guessed frame. Unmatched Tactical frames remain present in `frame_status.csv`.
- Player identifiers are anonymous tracking slots. `team` is H/A for players, not an inferred possession
  label for the ball.
- Coordinates/speed come directly from the accepted record. Metres use 105×68m. Off-pitch values remain
  off pitch; missing XY pairs do not become tracking points.
- `confidence` is blank when absent from the source. It is not fabricated from source availability.
- `observed` means a recorded non-predicted position. A supplied ATD record is not claimed to be a fresh
  direct YOLO observation. The original `source` and `is_prediction` distinguish these cases.
- `s_kph` is the recorded speed channel, not height and not a speed recomputed after coordinate changes.

## Retained method

The historical path starts from the user-specified corrected SNP half exports, applies original-H
projection/merge, recorded false-positive decisions, actual X-only detections, precision and two-ended
correction, Tactical ball references, accepted V3 local repair, then Tactical player fusion.

- [Confirmed pipeline](docs/PIPELINE.md)
- [Tracking technical note](docs/TECHNICAL.md)
- [Historical input audit](docs/INPUTS.md)

The discarded line/H repair branch, stronger V4, separate raw-ATD baseline, old partial-fill output and
3D/long-pass/goalpost experiments are excluded. Whole modules remain where their shared functions are
required; this is not permission to run their excluded standalone alternatives.

The 49 retained sources keep their original local paths/ROOT assumptions and original bytes. They are
not represented as a relocated raw-data end-to-end runner. The new adapter and validator use repository-
root input/output paths and do not import or execute those historical correction scripts.

## Verification and data handling

Tests for the adapter use synthetic inputs only; no match labels or public possession GT are read.
The original final CSV/ATD/XML/roster integrity was checked during collection. `VERIFICATION.json` records
that historical audit, including the original workspace's LFS-pointer state. A clean submission checkout
instead contains no private ATD/video inputs at all. Neither state is a successful fresh pipeline rerun.

The historical path contains stored object-review references, fixed decisions and game-specific model
heads. Their provenance/competition-rule eligibility is not certified by moving the source files. No new
labels, model training or inference are performed by this export adapter. Do not publish match inputs,
frame-level exports, crops, candidate banks or private feature/review assets in this repository.
