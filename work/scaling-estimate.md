# Frozen scaling and accuracy assessment

Assessment date: 2026-11-15, Asia/Seoul. This report does not launch full correction and is based on the already completed 4,500-frame pilot.

## Runtime evidence

The pilot covers 180 seconds of source footage at 25fps. Artifact timestamps suggest about **150.4 seconds of automatic processing**, excluding implementation, waiting and visual review. These are approximate phase durations, not instrumented wall timers.

| Phase | Timestamp evidence (KST) | Approximate seconds |
|---|---|---:|
| Preparation, source reading, camera motion | prepare script created 12:04:55.415; candidates written 12:05:16.042 | 20.7 |
| First association | correction script created 12:07:13.519; first corrected file 12:07:25.508 | 12.0 |
| Targeted re-detection | recovery script created 12:08:24.264; enriched candidates written 12:09:26.182 | 61.9 |
| Final association | correction script modified 12:10:02.623; final corrected file 12:10:22.113 | up to 19.5 |
| Coordinate export and master comparison video | export script created 12:14:23.222; final report written 12:14:53.837 | 30.6 |
| Preview encoding and verification | verification script created 12:16:12.164; verification written 12:16:17.852 | 5.7 |

The time between these phases includes reasoning, implementation, review and other work, so summing the entire user-visible pilot elapsed time would overestimate automatic runtime. Conversely, treating file creation as process start introduces some uncertainty. The planning ranges below allow for this and for differences in recovery frequency and candidate counts.

| Scope | Frames | Linear automatic correction estimate | Planning estimate |
|---|---:|---:|---|
| Existing first-half candidates | 79,467 | 44.3 minutes | 40–70 minutes |
| Both configured halves | 155,923 | 86.9 minutes | about 1.3–2.3 hours for correction |
| Entire MP4, including non-playing footage | 164,487 | 91.6 minutes | about 1.5–2.5 hours for correction |

Only the first-half original candidates are known to be available. At the observed approximately 7fps baseline detector speed, creating the missing second-half candidates adds **about 3.03 hours**. Both configured halves therefore require approximately **4.3–5.4 hours** including that missing detection stage and automatic correction. Processing every MP4 frame requires 85,020 additional candidate frames, about **3.37 hours** at 7fps, before correction.

The estimate assumes a recovery frequency near the pilot's 1,092/4,500 = **24.27%**. The pilot was deliberately selected as a difficult window; it is not a representative random sample. Automatic runtime does not include broader ground-truth annotation, model retraining or full visual verification.

## Accuracy evidence

- 3,130/4,500 frames contain an accepted observation: **69.56% acceptance coverage**, not precision or recall.
- 155 accepted frames use newly re-detected candidates. These 155 frames have not been independently labeled, and their count is not a measured gain in correct detections.
- 1,370 frames remain provisional or unresolved; this includes both genuinely unobservable footage and unresolved misses.
- Nineteen exact frames received sparse visual review. **All 19 final coordinates are identical to the original highest-confidence candidate in those same frames (0px difference).** Thus this reviewed subset demonstrates no improvement relative to a max-score baseline; it only supports some selected observations.
- No complete, independent ball-location ground truth is available for this pilot. Neither overall precision/recall nor their change has been measured. No numeric accuracy uplift is supported.

The fair next measurement is a temporally separate, labeled set of clips containing wide shots, close-ups, cuts/replays, airborne balls, player occlusions and dead balls. Evaluate the original candidate detector, an explicit original highest-score selector, and the correction on the **same labeled frames**. Report visible-ball recall, selection precision, localization error, false tracks per minute and uncertain-frame coverage separately. Labels used to tune the correction must not also serve as the held-out test set.

Full visual review alone requires at least the source playback duration (52.98 minutes for this first half) to see every frame once. Pauses, zooming and annotation can extend this to hours; there is no measured annotation rate supporting a tighter estimate.

## Deployment limits

The pilot scripts are not a ready full-match pipeline: they contain fixed ranges, output paths and an exact-frame visual override for frame 20,119. Generalize these and use bounded chunks or shots before scaling. Candidate dictionaries and duplicated exports can create high RAM usage when loaded for a full match. Do not carry the one-frame review override to other frames or treat ATD as ground truth.

Machine-readable companion: `work/scaling-estimate.json`.
