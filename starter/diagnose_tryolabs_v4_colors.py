"""Inspect existing cached tracks for colour hypotheses, never manual truth.

Only complete numeric chunk directories are read. No SQLite/tracker/model writes.
The broad search bins are NOT final goalkeeper kit ranges or team assignments.
"""
import argparse
from collections import defaultdict
import gzip
import json
from pathlib import Path

import cv2
import numpy as np

from run_tryolabs_classification_v3 import FrameReader, VIDEO, caption
from tryolabs_uniform_v2 import crop_region
from tryolabs_v4_data import ROOT, ASSETS, OUTPUT, write_json, protected_snapshot, verify_preserved

SEARCH = {"teal": ([75, 55, 45], [110, 255, 255]),
          "orange": ([0, 100, 90], [25, 255, 255]),
          "green": ([35, 70, 45], [74, 255, 255])}


def discover(out, cache, prepared, stride=25):
    out.mkdir(parents=True, exist_ok=False)
    before = protected_snapshot()
    write_json(out / "preservation_before.json", before)
    reader = FrameReader(VIDEO)
    frames = []; excluded = []; pixels = defaultdict(list); candidates = []
    for folder in sorted((cache / "chunks").iterdir()):
        if not folder.name[0].isdigit():
            excluded.append({"path": str(folder.relative_to(ROOT)), "reason": "INCOMPLETE_OR_NONNUMERIC_CHUNK"})
            continue
        with gzip.open(folder / "tracking.jsonl.gz", "rt", encoding="utf-8") as f:
            for line in f:
                r = json.loads(line)
                if r["frame0"] % stride == 0:
                    frames.append(dict(frame0=r["frame0"], sequence=f'full:{r["scene_id"]}', objects=r["players"]))
    for line in prepared.read_text(encoding="utf-8").splitlines():
        r = json.loads(line)
        if r["frame0"] % 5 == 0 and r["sequence"].startswith("legacy:"):
            frames.append(r)
    rng = np.random.default_rng(4)
    try:
        for j, r in enumerate(frames):
            image = reader.read(r["frame0"])
            for obj in r["objects"]:
                if obj["age"] != 0:
                    continue
                torso, _ = crop_region(image, obj["points"])
                shorts, _ = crop_region(image, obj["points"], (.48, .72))
                if torso.size == 0 or torso.shape[0] * torso.shape[1] < 128:
                    continue
                hsv = cv2.cvtColor(torso, cv2.COLOR_BGR2HSV)
                sh = cv2.cvtColor(shorts, cv2.COLOR_BGR2HSV) if shorts.size else np.zeros((1, 1, 3), np.uint8)
                ratios = {k: float(cv2.inRange(hsv, tuple(lo), tuple(hi)).mean() / 255) for k, (lo, hi) in SEARCH.items()}
                name = max(ratios, key=ratios.get)
                if ratios[name] < .30:
                    continue
                mask = cv2.inRange(hsv, tuple(SEARCH[name][0]), tuple(SEARCH[name][1])).astype(bool)
                pp = hsv[mask]
                pixels[name].append(pp[rng.choice(len(pp), min(128, len(pp)), replace=False)])
                candidates.append(dict(sequence=r["sequence"], frame0=r["frame0"], track_id=obj["track_id"],
                    points=obj["points"], search_color=name, torso_support=ratios,
                    shorts_support=float(cv2.inRange(sh, tuple(SEARCH[name][0]), tuple(SEARCH[name][1])).mean() / 255),
                    hsv_q10_q50_q90=np.quantile(pp, [.1, .5, .9], axis=0).tolist(),
                    actual_play_state="UNVERIFIED", role="UNCONFIRMED_COLOR_CANDIDATE"))
            if j % 100 == 0:
                print("COLOR_DISCOVERY", j, r["frame0"], len(candidates), flush=True)
        np.savez_compressed(out / "candidate_pixels.npz", **{k: np.concatenate(v) for k, v in pixels.items()})
        with (out / "candidates.jsonl").open("x", encoding="utf-8") as f:
            for obj in candidates:
                f.write(json.dumps(obj) + "\n")
        groups = defaultdict(list)
        for obj in candidates:
            groups[obj["search_color"], obj["sequence"], obj["track_id"]].append(obj)
        stats = []
        for (color, seq, tid), rows in groups.items():
            stats.append(dict(search_color=color, sequence=seq, track_id=tid, observations=len(rows),
                median_torso_support=float(np.median([r["torso_support"][color] for r in rows])),
                median_shorts_support=float(np.median([r["shorts_support"] for r in rows])),
                first_frame0=min(r["frame0"] for r in rows), last_frame0=max(r["frame0"] for r in rows)))
        write_json(out / "track_candidates.json", sorted(stats, key=lambda r: (-r["observations"], r["search_color"])))
        for color in SEARCH:
            groups_color = [(key, rows) for key, rows in groups.items() if key[0] == color]
            groups_color.sort(key=lambda item: -len(item[1]))
            strips = []
            for key, rows in groups_color[:15]:
                strip = np.full((200, 1200, 3), 22, np.uint8)
                for column, obj in enumerate([rows[0], rows[len(rows) // 2], rows[-1]]):
                    im = reader.read(obj["frame0"]); thumb = cv2.resize(im, (300, 169))
                    pp = np.rint(np.asarray(obj["points"]) * [300/im.shape[1],169/im.shape[0]]).astype(int)
                    cv2.rectangle(thumb, tuple(pp[0]), tuple(pp[1]), (0, 255, 255), 1)
                    strip[22:191, column*400:column*400+300] = thumb
                    roi, _ = crop_region(im, obj["points"])
                    strip[22:182,column*400+305:column*400+395] = cv2.resize(roi,(90,160),interpolation=cv2.INTER_NEAREST)
                    caption(strip, f'{key} f{obj["frame0"]}', (column*400+2,16), .34)
                strips.append(strip)
            if strips:
                cv2.imwrite(str(out / f"{color}_candidates.jpg"), np.concatenate(strips))
        write_json(out / "discovery.json", dict(frames_sampled=len(frames), candidates=len(candidates),
            search_hypotheses=SEARCH, excluded_chunks=excluded, manual_labels=False, detector_tracker_calls=0))
        write_json(out / "verification.json", verify_preserved(before))
    finally:
        reader.close()


if __name__ == "__main__":
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--output", type=Path, default=OUTPUT / "discovery")
    p.add_argument("--cache", type=Path, default=ASSETS / "full_match_v2/main")
    p.add_argument("--prepared", type=Path, default=ASSETS / "classification_v3/validated/input_frames.jsonl")
    args = p.parse_args()
    if not args.output.resolve().is_relative_to(OUTPUT.resolve()):
        p.error("Use a new output under classification_v4")
    discover(args.output, args.cache, args.prepared)
