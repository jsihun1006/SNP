"""Convert MAIN's private broadcast half labels into two portable frame ranges.

This helper does not run alignment or infer kickoff from images. Run MAIN's
final_tracking_alignment.py locally first, then pass its broadcast label CSV.
Only interval metadata is exported; per-frame labels and coordinates stay local.
"""
import argparse
import csv
from decimal import Decimal, InvalidOperation, ROUND_HALF_UP
import json
import os
from pathlib import Path
import uuid


TAGS = {"1st Half": "first_half", "2nd Half": "second_half"}


def _fps(value):
    try:
        result = Decimal(str(value))
    except InvalidOperation as exc:
        raise ValueError("FPS must be a positive finite number") from exc
    if not result.is_finite() or result <= 0:
        raise ValueError("FPS must be a positive finite number")
    return result


def _timestamp(frame, fps):
    # Nine fractional digits keep round-trip error far below one frame, even
    # for fractional FPS. At 25 FPS the output is exact to two decimal places.
    value = (Decimal(frame) / fps).quantize(Decimal("0.000000001"), rounding=ROUND_HALF_UP)
    whole = int(value)
    hours, remainder = divmod(whole, 3600)
    minutes, seconds = divmod(remainder, 60)
    fraction = format(value - whole, ".9f").split(".")[1].rstrip("0").ljust(2, "0")
    return f"{hours:02}:{minutes:02}:{seconds:02}.{fraction}"


def ranges_from_labels(labels, *, fps=25, video_frames=None):
    """Validate a complete label timeline and return two half-open intervals.

    ``video_frames`` may trim trailing ATD label rows that exceed decoded MP4
    coverage. A half that lies wholly outside that coverage is rejected. A
    clipped active half is explicitly recorded, never silently extended.
    """
    labels = Path(labels).resolve(strict=True)
    before = labels.stat()
    fps_value = _fps(fps)
    if video_frames is not None:
        if isinstance(video_frames, bool) or not isinstance(video_frames, int) or video_frames < 1:
            raise ValueError("video_frames must be a positive integer")
    blocks = []
    seen = set()
    active = ""
    first = None
    count = 0
    with labels.open(newline="", encoding="utf-8-sig") as handle:
        reader = csv.DictReader(handle)
        if not {"frame", "half_tag", "label"}.issubset(reader.fieldnames or []):
            raise ValueError("Expected MAIN broadcast labels with frame, half_tag, label columns")
        for row in reader:
            try:
                frame = int(row["frame"])
            except (TypeError, ValueError) as exc:
                raise ValueError("Label frames must be integer source-video indices") from exc
            if frame != count:
                raise ValueError("Label frames must start at zero and be unique and contiguous")
            tag = (row["half_tag"] or "").strip()
            label = (row["label"] or "").strip().upper()
            if tag not in ("", *TAGS):
                raise ValueError("Unknown half tag")
            if label not in ("", "H", "A", "D", "X"):
                raise ValueError("Unknown frame label")
            if tag and label == "X":
                raise ValueError("A tagged half contains an excluded X frame")
            if not tag and label != "X":
                raise ValueError("Untagged frames must be excluded with X")
            if tag != active:
                if active:
                    blocks.append(dict(tag=active, start_frame=first, end_frame=frame))
                if tag:
                    if tag in seen:
                        raise ValueError("Each half must have exactly one contiguous block")
                    seen.add(tag)
                    first = frame
                active = tag
            count += 1
    if active:
        blocks.append(dict(tag=active, start_frame=first, end_frame=count))
    if [block["tag"] for block in blocks] != list(TAGS):
        raise ValueError("Expected exactly one first-half block followed by one second-half block")
    if video_frames is not None and video_frames > count:
        raise ValueError("The supplied label table does not cover the decoded video")
    limit = count if video_frames is None else video_frames
    jobs = []
    clipped = []
    for block in blocks:
        first, last = block["start_frame"], min(block["end_frame"], limit)
        if not first < last:
            raise ValueError("A required half lies outside decoded video coverage")
        if last < block["end_frame"]:
            clipped.append(TAGS[block["tag"]])
        jobs.append(dict(part=TAGS[block["tag"]], start_frame=first, end_frame=last,
                         start=_timestamp(first, fps_value), end=_timestamp(last, fps_value)))
    if jobs[0]["end_frame"] > jobs[1]["start_frame"]:
        raise ValueError("Half ranges overlap or reverse")
    after = labels.stat()
    if (before.st_size, before.st_mtime_ns) != (after.st_size, after.st_mtime_ns):
        raise ValueError("The input label file changed during validation")
    result = dict(source="main final_tracking_alignment", fps=float(fps_value),
                  interval="start inclusive, end exclusive", source_frame_indices="zero-based original broadcast timeline",
                  input_label_rows=count, ignored_trailing_label_rows=max(0, count - limit),
                  clipped_part_ids=clipped, alignment_verified_full_half=False,
                  broadcast_mp4_alignment_verified=False,
                  alignment_assumption="per-half automatically selected anchor followed by constant timing offset; inspect video boundaries before production",
                  exports_private_per_frame_labels=False, jobs=jobs)
    if video_frames is not None:
        result["video_frames"] = video_frames
    return result


def main():
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--labels", required=True, help="MAIN final_tracking_broadcast_labels.csv")
    parser.add_argument("--fps", default="25")
    parser.add_argument("--out", required=True, help="Derived range metadata JSON, e.g. jobs.local.json")
    parser.add_argument("--video-frames", type=int, default=None, help="Decoded MP4 frame count, if known")
    args = parser.parse_args()
    output = Path(args.out).resolve()
    if output.suffix.lower() != ".json" or output == Path(args.labels).resolve():
        raise ValueError("Use a separate .json output file")
    result = ranges_from_labels(args.labels, fps=args.fps, video_frames=args.video_frames)
    output.parent.mkdir(parents=True, exist_ok=True)
    temporary = output.with_name(output.name + ".tmp-" + uuid.uuid4().hex)
    try:
        with temporary.open("w", encoding="utf-8") as handle:
            json.dump(result, handle, ensure_ascii=False, indent=2, allow_nan=False)
            handle.write("\n")
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(temporary, output)
    finally:
        if temporary.exists():
            temporary.unlink()
    print(json.dumps(dict(output=str(output), jobs=result["jobs"],
                          ignored_trailing_label_rows=result["ignored_trailing_label_rows"]), ensure_ascii=False))


if __name__ == "__main__":
    main()
