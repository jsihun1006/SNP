"""Replay the unchanged 750-frame detector cache with a separate HSV classifier.

All private outputs remain under data/derived/tryolabs_broadcast/classification_v2.
The original runner, upstream modules, weights and first experiment are read-only.
"""

import argparse
from collections import Counter
import csv
import json
from pathlib import Path
import subprocess
import sys
import time

import cv2
import numpy as np
import pandas as pd

import run_tryolabs_broadcast as legacy
from tryolabs_uniform_v2 import UniformClassifier, TemporalClassifier, metadata, select_candidate

ROOT = legacy.ROOT
BASE = legacy.ASSETS
OUT = BASE / "classification_v2"
VIDEO = ROOT / "data/broadcast/BAR-ATH_broadcast-001.mp4"
POLICY = ROOT / "starter/configs/tryolabs_classification_v2_policy.json"
DEFAULT_CONFIG = ROOT / "starter/configs/tryolabs_classification_v2_hsv.json"


def roi(frame, points, part="torso"):
    """Same torso geometry as upstream; lower body is a separate role check."""
    (x1, y1), (x2, y2) = np.asarray(points, dtype=float)
    xa, xb = int(x1 + .15*(x2-x1)), int(x1 + .85*(x2-x1))
    low, high = (.15, .50) if part == "torso" else (.60, .95)
    ya, yb = int(y1+low*(y2-y1)), int(y1+high*(y2-y1))
    h, w = frame.shape[:2]
    return frame[max(0, ya):min(h, yb), max(0, xa):min(w, xb)]


def extract(out):
    if (out/"track_observations.jsonl").exists() or (out/"torso_pixel_samples.npz").exists():
        raise FileExistsError("Extraction exists; choose a new output subdirectory")
    policy = json.loads(POLICY.read_text())["calibration"]
    u = legacy.import_upstream()
    rows = list(csv.DictReader((BASE / "preview/frames.csv").open()))
    cached = [json.loads(line) for line in (BASE / "preview/detections.jsonl").read_text().splitlines()]
    assert len(rows) == len(cached) == 750
    cap = cv2.VideoCapture(str(VIDEO))
    previous_segment = None
    samples, records, observations, mismatches = [], [], [], []
    rng = np.random.default_rng(policy["seed"])
    sample_offset = 0
    for i, (old, raw) in enumerate(zip(rows, cached)):
        frame0 = int(old["source_frame0"])
        assert raw["source_frame0"] == frame0
        if old["segment"] != previous_segment:
            state = legacy.new_state(u)
            cap.set(cv2.CAP_PROP_POS_FRAMES, frame0)
            previous_segment = old["segment"]
        ok, frame = cap.read()
        assert ok and round(cap.get(cv2.CAP_PROP_POS_FRAMES)) == frame0+1
        people = u["Converter"].DataFrame_to_Detections(pd.DataFrame(raw["persons"]))
        balls = u["Converter"].DataFrame_to_Detections(pd.DataFrame(raw["balls"]))
        for d in people+balls:
            d.data["source_frame0"] = frame0
        transform = u["update_motion_estimator"](state["motion"], people+balls, frame)
        pts = state["player_tracker"].update(detections=people, coord_transformations=transform)
        bts = state["ball_tracker"].update(detections=balls, coord_transformations=transform)
        pdets = u["Converter"].TrackedObjects_to_Detections(pts)
        bdets = u["Converter"].TrackedObjects_to_Detections(bts)
        pdets = state["classifier"].predict_from_detections(pdets, frame)
        players = u["Player"].from_detections(pdets, state["teams"])
        ball = u["get_main_ball"](bdets)
        state["match"].update(players, ball if players else u["Ball"](None))
        match = state["match"]
        actual = [match.team_possession.name, state["teams"][0].possession, state["teams"][1].possession, match.possession_counter]
        expected = [old["legacy_owner"], int(old["legacy_blue_frames"]), int(old["legacy_light_frames"]), int(old["legacy_counter"])]
        if actual != expected:
            mismatches.append(dict(frame0=frame0, actual=actual, expected=expected))
        objs = []
        for d in pdets:
            age = frame0-d.data["source_frame0"]
            record = dict(segment=int(old["segment"]), frame0=frame0, track_id=d.data["id"],
                          points=d.points.tolist(), age=age, legacy_class=d.data["classification"],
                          detector_confidence=float(d.data["p"]))
            objs.append(record)
            if age:
                continue
            crop = roi(frame, d.points)
            if min(crop.shape[:2], default=0) < 2:
                continue
            pixels = cv2.cvtColor(crop, cv2.COLOR_BGR2HSV).reshape(-1, 3)
            selection = pixels[rng.choice(len(pixels), min(policy["maximum_pixels_per_object_observation"], len(pixels)), replace=False)]
            samples.append(selection)
            records.append({**record, "sample_start":sample_offset, "sample_count":len(selection),
                            "torso_width":crop.shape[1], "torso_height":crop.shape[0],
                            "hsv_q10_q50_q90":np.quantile(pixels, [.1, .5, .9], axis=0).round(2).tolist()})
            sample_offset += len(selection)
        observations.append(dict(segment=int(old["segment"]), frame0=frame0, output_frame0=i, objects=objs,
                                 main_ball=None if ball.detection is None else dict(points=ball.detection.points.tolist(),
                                 age=frame0-ball.detection.data["source_frame0"])))
        if i % 125 == 0:
            print("EXTRACT", i, frame0, flush=True)
    cap.release()
    for name, rr in [("track_observations.jsonl", observations), ("pixel_sample_sources.jsonl", records)]:
        with (out/name).open("x") as f:
            for r in rr:
                f.write(json.dumps(r)+"\n")
    pixels = np.concatenate(samples)
    np.savez_compressed(out/"torso_pixel_samples.npz", hsv=pixels)
    (out/"legacy_replay_verification.json").write_text(json.dumps(dict(frames=750, mismatches=mismatches), indent=2))
    print("REPLAY", len(rows), "mismatches", len(mismatches), "pixels", len(pixels), flush=True)
    if mismatches:
        raise ValueError("Legacy replay differs from frozen counters; see verification log")


def cluster_pixels(out):
    if (out/"pixel_clusters.json").exists():
        raise FileExistsError("Pixel clusters already exist")
    policy = json.loads(POLICY.read_text())["calibration"]
    pixels = np.load(out/"torso_pixel_samples.npz")["hsv"]
    rng = np.random.default_rng(policy["seed"])
    indices = rng.choice(len(pixels), min(policy["maximum_kmeans_pixels"], len(pixels)), replace=False)
    sampled = pixels[indices].astype(np.float32)
    angle = sampled[:, 0]*np.pi/90
    sat, value = sampled[:, 1]/255, sampled[:, 2]/255
    features = np.column_stack([sat*np.cos(angle), sat*np.sin(angle), value]).astype(np.float32)
    cv2.setRNGSeed(policy["seed"])
    _, labels, centers = cv2.kmeans(features, policy["clusters"], None, (cv2.TERM_CRITERIA_EPS+cv2.TERM_CRITERIA_MAX_ITER, 100, .001), 3, cv2.KMEANS_PP_CENTERS)
    clusters = []
    for j in range(len(centers)):
        pp = sampled[labels.ravel() == j]
        clusters.append(dict(cluster=j, count=len(pp), hsv_quantiles=np.quantile(pp, [.02, .1, .5, .9, .98], axis=0).round(2).tolist(),
                             bgr=cv2.cvtColor(np.uint8([[np.median(pp, axis=0)]]), cv2.COLOR_HSV2BGR)[0,0].tolist()))
    (out/"pixel_clusters.json").write_text(json.dumps(clusters, indent=2))
    for c in clusters:
        print(c)


def calibrate(out):
    if (out/"hsv_config.json").exists():
        raise FileExistsError("Measured HSV config already exists")
    policy = json.loads(POLICY.read_text())
    clusters = json.loads((out/"pixel_clusters.json").read_text())
    def choose(predicate):
        candidates = [c for c in clusters if predicate(*c["hsv_quantiles"][2])]
        if len(candidates) != 1:
            raise ValueError(f"Unresolved automatic color component: {[c['cluster'] for c in candidates]}")
        return candidates[0]
    # Broad semantic seed hypotheses only. Classification boxes below come from
    # measured quantiles, not these seed gates; no object is hand-labelled.
    components = {
        "navy": [(choose(lambda h,s,v:100 <= h <= 135 and s >= 90 and v < 190), 0, 4)],
        "burgundy": [(choose(lambda h,s,v:155 <= h <= 179 and s >= 140 and v >= 100), 0, 4),
                     (choose(lambda h,s,v:135 <= h <= 155 and s >= 100 and v < 170), 1, 3)],
        "lilac": [(choose(lambda h,s,v:125 <= h <= 155 and 50 <= s <= 130 and v >= 200), 0, 4),
                  (choose(lambda h,s,v:125 <= h <= 155 and 20 <= s <= 120 and 123 <= v < 200), 1, 3)],
        "yellow": [(choose(lambda h,s,v:20 <= h <= 45 and s >= 180 and v >= 180), 0, 4)],
    }
    bounds = {}
    for name, cs in components.items():
        bounds[name] = [dict(lower=np.floor(c["hsv_quantiles"][lo]).astype(int).tolist(),
                            upper=np.ceil(c["hsv_quantiles"][hi]).astype(int).tolist(),
                            cluster=c["cluster"], quantiles=[.02 if lo == 0 else .1, .98 if hi == 4 else .9]) for c,lo,hi in cs]
    dark = choose(lambda h,s,v:v < 100 and s < 90)
    config = dict(schema_version=1, hsv_components=bounds, quality=policy["quality"], temporal=policy["temporal"],
                  lower_dark_value_max=int(dark["hsv_quantiles"][3][2]),
                  minimum_referee_lower_dark_fraction=.1,
                  decision={label:dict(minimum_support=.15, minimum_margin=.02) for label in ["TEAM_A", "TEAM_B", "REFEREE"]})
    classifier = UniformClassifier(config)
    cap = cv2.VideoCapture(str(VIDEO))
    anchors = {label:[] for label in config["decision"]}
    track_distributions = {}
    previous_segment = None
    observations = [json.loads(x) for x in (out/"track_observations.jsonl").read_text().splitlines()]
    fractions = []
    for record in observations:
        if record["segment"] != previous_segment:
            cap.set(cv2.CAP_PROP_POS_FRAMES, record["frame0"])
            previous_segment = record["segment"]
        ok, frame = cap.read()
        assert ok
        for obj in record["objects"]:
            if obj["age"]:
                continue
            f = classifier.features(frame, obj["points"])
            ranked = sorted(f["scores"], key=lambda k:(-f["scores"][k], k))
            winner, second = ranked[:2]
            top, runner = f["scores"][winner], f["scores"][second]
            a = policy["calibration"]
            if f["torso_pixels"] >= policy["quality"]["minimum_torso_pixels"] and top >= a["anchor_minimum_support"] and top >= a["anchor_winner_runner_up_ratio"]*runner:
                anchors[winner].append(dict(support=top, margin=top-runner, lower_dark=f["lower_dark_fraction"],
                                            frame0=record["frame0"], segment=record["segment"], track_id=obj["track_id"]))
            fractions.append(dict(segment=record["segment"], frame0=record["frame0"], track_id=obj["track_id"], **f))
            key = f'{record["segment"]}:{obj["track_id"]}'
            track_distributions.setdefault(key, []).append(f["components"])
    cap.release()
    q = policy["calibration"]["decision_quantile"]
    for label, values in anchors.items():
        if len(values) < 10:
            raise ValueError(f"Insufficient automatic color anchors for {label}")
        config["decision"][label] = dict(minimum_support=float(np.quantile([a["support"] for a in values], q)),
                                       minimum_margin=max(policy["calibration"]["minimum_decision_margin_floor"], float(np.quantile([a["margin"] for a in values], q))))
    config["minimum_referee_lower_dark_fraction"] = max(policy["calibration"]["minimum_lower_dark_floor"], float(np.quantile([a["lower_dark"] for a in anchors["REFEREE"]], q)))
    config["provenance"] = dict(method="Unlabelled torso-pixel clustering and automatic dominant-color anchors; no accuracy calibration",
                                policy_sha256=legacy.digest(POLICY), pixel_samples_sha256=legacy.digest(out/"torso_pixel_samples.npz"),
                                anchors={k:len(v) for k,v in anchors.items()}, calibration_frames=750, held_out=False)
    with (out/"hsv_config.json").open("x") as f:
        json.dump(config, f, indent=2)
    with (out/"calibration_support.jsonl").open("x") as f:
        for r in fractions:
            f.write(json.dumps(r)+"\n")
    stats = {key:dict(observations=len(values), component_q10_q50_q90={name:np.quantile([v[name] for v in values], [.1,.5,.9]).tolist() for name in bounds}) for key,values in track_distributions.items()}
    (out/"track_color_distributions.json").write_text(json.dumps(stats, indent=2))
    (out/"automatic_anchors.json").write_text(json.dumps(anchors, indent=2))
    print("CALIBRATION", json.dumps(config), flush=True)


def objects_to_players(u, objects, teams):
    from norfair import Detection
    result = []
    for obj in objects:
        if not (obj["eligible"] and obj["role"] == "PLAYER" and obj["team"] in ("TEAM_A", "TEAM_B") and obj["age"] == 0):
            continue
        detection = Detection(points=np.asarray(obj["points"]), data={"id":obj["track_id"], "classification":obj["team"]})
        result.extend(u["Player"].from_detections([detection], teams))
    return result


def safe_match_update(u, match, players, ball):
    """Keep upstream Match.update and 45px/20 thresholds; guard min([])."""
    match.update(players, ball if players else u["Ball"](None))


def draw_comparison(old_image, frame, objects, raw, old, row):
    from types import SimpleNamespace
    canvas = np.full((930,1920,3), 22, np.uint8)
    canvas[45:585,:960] = old_image[42:582,960:1920]
    image = cv2.resize(frame, (960,540))
    colors = {"TEAM_A":(255,140,80), "TEAM_B":(235,160,245), "REFEREE":(0,255,255), "UNKNOWN":(155,155,155)}
    for o in objects:
        label = f'{o["label"]} | ID: {o["track_id"]}'
        if o["label"] == "REFEREE":
            label += " | EXCLUDED"
        legacy.box(image, SimpleNamespace(points=np.asarray(o["points"])), colors[o["label"]], label, bool(o["age"]))
        if o["label"] == "UNKNOWN":
            x, y = np.rint(np.asarray(o["points"])[1]*.5).astype(int)
            legacy.caption(image, o["reason"], (max(0,x-45),min(535,y+9)), colors["UNKNOWN"], .25)
    for b in raw["balls"]:
        d = SimpleNamespace(points=np.array([[b["xmin"],b["ymin"]],[b["xmax"],b["ymax"]]]))
        legacy.box(image,d,(80,255,80),f'ball? {b["confidence"]:.2f}')
        cv2.circle(image, tuple(np.rint(d.points.mean(axis=0)*.5).astype(int)), 12, (80,255,80), 1)
    canvas[45:585,960:] = image
    legacy.caption(canvas,"BEFORE: frozen BLUE / LIGHT / OTHER",(16,30))
    legacy.caption(canvas,"V2: TEAM / ROLE / ABSTENTION (same detections)",(976,30))
    legacy.caption(canvas,f'Segment {row["segment"]} | source {row["source_frame0"]} | file {row["source_frame0"]/25:.2f}s | output {row["output_frame0"]}',(16,617))
    legacy.caption(canvas,f'Frozen legacy owner {old["legacy_owner"]} | BLUE {int(old["legacy_blue_frames"])/25:.2f}s / LIGHT {int(old["legacy_light_frames"])/25:.2f}s',(16,653),(100,190,255))
    legacy.caption(canvas,f'V2 historical counter: {row["v2_history_owner"] or "UNINITIALIZED"} | A {row["v2_history_a_frames"]/25:.2f}s / B {row["v2_history_b_frames"]/25:.2f}s | not current evidence',(16,686),(100,190,255))
    legacy.caption(canvas,f'Current: {row["evidence_status"]} | candidate ID {row["candidate_id"]} | {row["candidate_team"]} | run {row["fresh_run"]}/20',(16,720),(100,230,180))
    legacy.caption(canvas,f'Reason {row["evidence_reason"]} | referees EXCLUDED {row["referees_excluded"]} | eligible players {row["eligible_players"]} | ball candidates {row["ball_candidates"]}',(16,754))
    counts = Counter(o["label"] for o in objects)
    legacy.caption(canvas,f'V2 classes {dict(counts)} | UNKNOWN blocks possession; goalkeeper role unassigned',(16,788),scale=.59)
    legacy.caption(canvas,"TEAM_A: navy + burgundy | TEAM_B: lilac | REFEREE: yellow torso + dark lower body | green ball boxes remain unverified",(16,823),scale=.58)
    legacy.caption(canvas,"CSV/JSONL: component support, margin, raw/stable labels, reasons, eligibility; classification_confidence = null",(16,858),scale=.6)
    legacy.caption(canvas,"Unlabelled calibration and demo share 750 frames. No accuracy / precision / recall / F1; 45px and 20-frame rule preserved.",(16,901),scale=.59)
    return canvas


def count_changes(rows, field, *, by_track=False):
    previous, count = {}, 0
    for row in rows:
        key = (row["segment"], row["track_id"]) if by_track else row["segment"]
        value = row[field]
        old = previous.get(key)
        if old is not None and row["source_frame0"] == old[0]+1 and value != old[1]:
            count += 1
        previous[key] = row["source_frame0"], value
    return count


def transition_details(objects):
    previous = {}
    known, unknown = 0, 0
    for obj in objects:
        key = obj["segment"],obj["track_id"]
        old = previous.get(key)
        if old and obj["source_frame0"] == old[0]+1 and obj["label"] != old[1]:
            if "UNKNOWN" in (obj["label"],old[1]):
                unknown += 1
            else:
                known += 1
        previous[key] = obj["source_frame0"],obj["label"]
    return dict(direct_known_class_changes=known,unknown_boundary_changes=unknown)


def run_comparison(out, config_path):
    if any((out/name).exists() for name in ["comparison.mp4","objects.jsonl","frames.csv","summary.json"]):
        raise FileExistsError("V2 comparison already exists; use a new --output subdirectory")
    observations_path = out/"track_observations.jsonl"
    if not observations_path.exists():
        observations_path = OUT/"track_observations.jsonl"
    observations = [json.loads(x) for x in observations_path.read_text().splitlines()]
    old_rows = list(csv.DictReader((BASE/"preview/frames.csv").open()))
    cache = [json.loads(x) for x in (BASE/"preview/detections.jsonl").read_text().splitlines()]
    config = json.loads(config_path.read_text())
    classifier = UniformClassifier(config)
    u = legacy.import_upstream()
    from norfair import Detection
    cap = cv2.VideoCapture(str(VIDEO))
    old_cap = cv2.VideoCapture(str(BASE/"preview/broadcast_tryolabs.mp4"))
    import imageio_ffmpeg
    command = [imageio_ffmpeg.get_ffmpeg_exe(), "-nostdin", "-v", "error", "-n", "-f", "rawvideo", "-pix_fmt", "bgr24", "-s", "1920x930", "-r", "25", "-i", "pipe:0", "-an", "-c:v", "libx264", "-preset", "veryfast", "-crf", "21", "-pix_fmt", "yuv420p", "-movflags", "+faststart", str(out/"comparison.mp4")]
    frames, objects_log = [], []
    previous_segment = None
    start = time.monotonic()
    log = (out/"encoder.log").open("x")
    encoder = subprocess.Popen(command, stdin=subprocess.PIPE, stderr=log)
    try:
        for old, raw, obs in zip(old_rows, cache, observations):
            frame0, segment = obs["frame0"], obs["segment"]
            assert frame0 == int(old["source_frame0"]) == raw["source_frame0"]
            if segment != previous_segment:
                cap.set(cv2.CAP_PROP_POS_FRAMES, frame0)
                temporal = TemporalClassifier(config["temporal"])
                teams = [u["Team"](name="TEAM_A",abbreviation="TMA"),u["Team"](name="TEAM_B",abbreviation="TMB")]
                match = u["Match"](teams[0],teams[1],fps=25)
                match.team_possession = None  # Separate history starts without an arbitrary owner.
                evidence_run = legacy.EvidenceRun()
                previous_segment = segment
            ok, frame = cap.read(); old_ok, old_image = old_cap.read()
            assert ok and old_ok and round(cap.get(cv2.CAP_PROP_POS_FRAMES)) == frame0+1
            objects = []
            for obj in obs["objects"]:
                others = [o["points"] for o in obs["objects"] if o["track_id"] != obj["track_id"] and o["age"] == 0]
                raw_class = classifier.classify(frame, obj["points"], others)
                decision = temporal.update(obj["track_id"], frame0, raw_class, fresh=obj["age"] == 0)
                record = {**obj, **decision, "source_frame0":frame0, "legacy_eligible":obj["legacy_class"] in ("BLUE","LIGHT") and obj["age"] == 0}
                objects.append(record)
                objects_log.append(record)
            ball_points = None if len(raw["balls"]) != 1 else np.array([[raw["balls"][0]["xmin"],raw["balls"][0]["ymin"]],[raw["balls"][0]["xmax"],raw["balls"][0]["ymax"]]])
            candidate, status, reason = select_candidate(objects, ball_points, ball_count=len(raw["balls"]))
            # Upstream first finds nearest among all people, even OTHER; preserve
            # that ordering when measuring the old instantaneous candidate.
            nearest_all = None
            if ball_points is not None:
                center = ball_points.mean(axis=0)
                def distance(o):
                    (x1,_),(x2,y2) = np.array(o["points"])
                    return min(np.linalg.norm(center-[x1,y2]),np.linalg.norm(center-[x2,y2]))
                fresh = [o for o in objects if o["age"] == 0]
                nearest_all = min(fresh,key=distance) if fresh else None
            old_candidate = None
            if nearest_all is not None and distance(nearest_all) <= 45 and nearest_all["legacy_class"] in ("BLUE","LIGHT"):
                old_candidate = nearest_all["track_id"]
            valid_players = objects_to_players(u, objects, teams)
            main_ball = obs["main_ball"]
            # Keep original ball tracking and selected main ball. Freshness and
            # uniqueness gate only whether it is evidence for this update.
            tracked_ball = u["Ball"](None)
            if main_ball is not None and main_ball["age"] == 0 and len(raw["balls"]) == 1:
                tracked_ball = u["Ball"](Detection(points=np.array(main_ball["points"]),data={}))
            safe_match_update(u,match,valid_players,tracked_ball)
            fresh_team = evidence_run.update(None if candidate is None else candidate["team"])
            if candidate is not None:
                status = "PROXIMITY_20_FRAME_EVIDENCE" if fresh_team else "PROXIMITY_PENDING"
            row = dict(segment=segment, source_frame0=frame0, output_frame0=len(frames),
                       ball_candidates=len(raw["balls"]), ball_evidence_insufficient=len(raw["balls"]) != 1,
                       referees_excluded=sum(o["role"] == "REFEREE" for o in objects),
                       referee_nearest_excluded=bool(nearest_all is not None and nearest_all["role"] == "REFEREE" and distance(nearest_all) <= 45),
                       eligible_players=len(valid_players), legacy_candidate_id=old_candidate,
                       candidate_id=None if candidate is None else candidate["track_id"],
                       candidate_team=None if candidate is None else candidate["team"],
                       evidence_status=status,evidence_reason=reason,fresh_run=evidence_run.count,
                       legacy_owner=old["legacy_owner"],legacy_blue_frames=int(old["legacy_blue_frames"]),legacy_light_frames=int(old["legacy_light_frames"]),
                       v2_history_owner=None if match.team_possession is None else match.team_possession.name,
                       v2_history_a_frames=teams[0].possession,v2_history_b_frames=teams[1].possession)
            canvas = draw_comparison(old_image,frame,objects,raw,old,row)
            encoder.stdin.write(canvas.tobytes())
            frames.append(row)
            if len(frames)%250 in (1,126,0):
                cv2.imwrite(str(out/f"segment_{segment}_frame_{frame0}.jpg"),canvas)
            if len(frames)%125 == 0:
                print("V2",len(frames),frame0,dict(Counter(o["label"] for o in objects)),flush=True)
    finally:
        cap.release(); old_cap.release(); encoder.stdin.close(); encoder.wait(); log.close()
        with (out/"objects.jsonl").open("x") as f:
            for obj in objects_log:
                f.write(json.dumps(obj)+"\n")
        if frames:
            with (out/"frames.csv").open("x") as f:
                w = csv.DictWriter(f,fieldnames=list(frames[0]));w.writeheader();w.writerows(frames)
    if encoder.returncode or len(frames) != 750:
        raise RuntimeError("Incomplete comparison; inspect logs")
    object_fields = ["segment","source_frame0","track_id","label","team","role","classification_confidence","classification_source","reason","raw_label","legacy_class","age","eligible","score_margin","lower_dark_fraction"]
    with (out/"objects.csv").open("x") as f:
        w = csv.DictWriter(f,fieldnames=object_fields,extrasaction="ignore");w.writeheader();w.writerows(objects_log)
    video = cv2.VideoCapture(str(out/"comparison.mp4"))
    decoded = 0
    while video.read()[0]:
        decoded += 1
    fps,width,height = video.get(5),video.get(3),video.get(4)
    video.release()
    assert decoded == 750 and fps == 25 and (width,height) == (1920,930)
    segments = []
    for segment in (1,2,3):
        rr = [r for r in frames if r["segment"] == segment]
        oo = [o for o in objects_log if o["segment"] == segment]
        class_counts = Counter(o["label"] for o in oo)
        old_counts = Counter(o["legacy_class"] for o in oo)
        segments.append(dict(segment=segment,frames=len(rr),objects=len(oo),
                             source_start_frame0=rr[0]["source_frame0"],source_end_frame0=rr[-1]["source_frame0"],
                             legacy_classes=dict(old_counts),v2_classes=dict(class_counts),
                             unknown_fraction=class_counts["UNKNOWN"]/max(1,len(oo)),
                             legacy_other_fraction=old_counts["OTHER"]/max(1,len(oo)),
                             legacy_track_class_changes=count_changes(oo,"legacy_class",by_track=True),
                             v2_track_class_changes=count_changes(oo,"label",by_track=True),
                             v2_raw_track_class_changes=count_changes(oo,"raw_label",by_track=True),
                             v2_transition_details=transition_details(oo),
                             referee_exclusion_frames=sum(r["referees_excluded"] > 0 for r in rr),
                             referee_nearest_excluded_frames=sum(r["referee_nearest_excluded"] for r in rr),
                             legacy_candidate_changes=count_changes(rr,"legacy_candidate_id"),v2_candidate_changes=count_changes(rr,"candidate_id"),
                             candidate_different_frames=sum(r["legacy_candidate_id"] != r["candidate_id"] for r in rr),
                             ball_evidence_insufficient_frames=sum(r["ball_evidence_insufficient"] for r in rr),
                             ball_evidence_insufficient_fraction=sum(r["ball_evidence_insufficient"] for r in rr)/len(rr),
                             evidence_statuses=dict(Counter(r["evidence_status"] for r in rr))))
    summary = dict(frames=len(frames),objects=len(objects_log),output_decoded_frames=decoded,output_fps=fps,
                   elapsed_seconds=time.monotonic()-start,segments=segments,
                   total_v2_classes=dict(Counter(o["label"] for o in objects_log)),
                   unknown_fraction=sum(o["label"] == "UNKNOWN" for o in objects_log)/len(objects_log),
                   unknown_reasons=dict(Counter(o["reason"] for o in objects_log if o["label"] == "UNKNOWN")),
                   raw_person_detections=sum(len(r["persons"]) for r in cache),
                   candidate_reference="Nearest of all fresh legacy tracks, then known BLUE/LIGHT; v2 nearest eligible PLAYER after exclusions. Both use unique raw ball and 45px. Changes include None transitions; segments never connected.",
                   config_path=str(config_path),config_sha256=legacy.digest(config_path),
                   detector_cache_sha256=legacy.digest(BASE/"preview/detections.jsonl"),
                   unchanged_ball_detection=True,legacy_counters_preserved=all(
                   (r["legacy_owner"],r["legacy_blue_frames"],r["legacy_light_frames"]) == (old["legacy_owner"],int(old["legacy_blue_frames"]),int(old["legacy_light_frames"])) for r,old in zip(frames,old_rows)),
                   classification_confidence=None,scope="Unlabelled calibration/demo; no accuracy claims; team appearance labels are not official H/A identities")
    (out/"summary.json").write_text(json.dumps(summary,indent=2))
    print(json.dumps(summary,indent=2),flush=True)


def verify_preservation(out):
    before = json.loads((OUT/"preservation_before.json").read_text())
    changed = [path for path,record in before["files"].items() if not (ROOT/path).is_file() or legacy.digest(ROOT/path) != record["sha256"]]
    st = VIDEO.stat()
    video_unchanged = (st.st_size,st.st_mtime_ns) == (before["video"]["size"],before["video"]["mtime_ns"])
    objects = [json.loads(x) for x in (out/"objects.jsonl").read_text().splitlines()]
    invalid = [o for o in objects if o["eligible"] and (o["role"] != "PLAYER" or o["team"] not in ("TEAM_A","TEAM_B") or o["age"] != 0)]
    old = list(csv.DictReader((BASE/"preview/frames.csv").open()))
    new = list(csv.DictReader((out/"frames.csv").open()))
    assert len(old) == len(new) == 750
    preserved_counters = all(all(a[k] == b[k] for k in ["source_frame0","legacy_owner","legacy_blue_frames","legacy_light_frames"]) for a,b in zip(old,new))
    by_frame = {}
    for o in objects:
        by_frame.setdefault(o["source_frame0"],{})[o["track_id"]] = o
    selected_nonplayers = []
    for row in new:
        if row["candidate_id"]:
            selected = by_frame[int(row["source_frame0"])][int(row["candidate_id"])]
            if selected["role"] != "PLAYER" or not selected["eligible"]:
                selected_nonplayers.append(int(row["source_frame0"]))
    verification = dict(protected_files=len(before["files"]),changed_files=changed,source_video_size_mtime_unchanged=video_unchanged,
                        invalid_eligible_objects=len(invalid),selected_nonplayer_frames=selected_nonplayers,
                        preserved_frozen_counter_frames=750 if preserved_counters else 0,
                        all_classification_confidence_null=all(o["classification_confidence"] is None for o in objects),
                        measured_and_default_configs_equal=json.loads((OUT/"hsv_config.json").read_text()) == json.loads(DEFAULT_CONFIG.read_text()))
    assert not changed and video_unchanged and not invalid and not selected_nonplayers and preserved_counters
    with (out/"verification.json").open("x") as f:
        json.dump(verification,f,indent=2)
    print("VERIFIED",json.dumps(verification),flush=True)


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--extract", action="store_true")
    p.add_argument("--cluster", action="store_true")
    p.add_argument("--calibrate", action="store_true")
    p.add_argument("--run", action="store_true")
    p.add_argument("--verify", action="store_true")
    p.add_argument("--config", type=Path)
    p.add_argument("--output", type=Path, default=OUT)
    args = p.parse_args()
    if not any([args.extract,args.cluster,args.calibrate,args.run,args.verify]):
        p.error("Select --run or one or more preparation stages")
    out = args.output.resolve()
    if not out.is_relative_to(OUT.resolve()):
        p.error("Output must be under the separate classification_v2 directory")
    out.mkdir(parents=True, exist_ok=True)
    if args.extract:
        extract(out)
    if args.cluster:
        cluster_pixels(out)
    if args.calibrate:
        calibrate(out)
    if args.run:
        config_path = args.config or (out/"hsv_config.json" if args.calibrate else DEFAULT_CONFIG)
        run_comparison(out, config_path.resolve())
    if args.verify:
        verify_preservation(out)


if __name__ == "__main__":
    main()
