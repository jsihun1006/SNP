"""Conservative HSV appearance evidence with separate team and role metadata.

No model inference, labels, match accumulation or upstream monkeypatching here.
HSV bounds are loaded from measured pixel clusters, never guessed in this module.
Scores are pixel support fractions, not calibrated correctness probabilities.
"""

from collections import Counter, deque

import cv2
import numpy as np

LABELS = ("TEAM_A", "TEAM_B", "REFEREE", "UNKNOWN")
TEAMS = ("TEAM_A", "TEAM_B", "UNKNOWN")
ROLES = ("PLAYER", "REFEREE", "GOALKEEPER", "UNKNOWN")


def crop_region(frame, points, vertical=(.15, .50)):
    (x1, y1), (x2, y2) = np.asarray(points, dtype=float)
    xa, xb = int(x1+.15*(x2-x1)), int(x1+.85*(x2-x1))
    ya, yb = int(y1+vertical[0]*(y2-y1)), int(y1+vertical[1]*(y2-y1))
    h, w = frame.shape[:2]
    return frame[max(0, ya):min(h, yb), max(0, xa):min(w, xb)], (xa, ya, xb, yb)


def mask_range(hsv, ranges):
    result = np.zeros(hsv.shape[:2], bool)
    for bounds in ranges:
        lo, hi = np.asarray(bounds["lower"]), np.asarray(bounds["upper"])
        result |= ((hsv >= lo) & (hsv <= hi)).all(axis=2)
    return result


def metadata(label, **extras):
    return dict(label=label, team=label if label in ("TEAM_A", "TEAM_B") else (None if label == "REFEREE" else "UNKNOWN"),
                role="REFEREE" if label == "REFEREE" else ("PLAYER" if label in ("TEAM_A", "TEAM_B") else "UNKNOWN"),
                classification_confidence=None, classification_source="HSV", **extras)


class UniformClassifier:
    def __init__(self, config):
        self.config = config

    def features(self, frame, points):
        torso, rectangle = crop_region(frame, points)
        lower, _ = crop_region(frame, points, (.60, .95))
        if torso.size == 0:
            return dict(scores={k:0. for k in LABELS[:-1]}, components={}, torso_pixels=0,
                        torso_width=0, torso_height=0, lower_dark_fraction=0., rectangle=rectangle)
        hsv = cv2.cvtColor(torso, cv2.COLOR_BGR2HSV)
        masks = {name:mask_range(hsv, ranges) for name, ranges in self.config["hsv_components"].items()}
        support = {name:float(mask.mean()) for name, mask in masks.items()}
        navy = masks["navy"]
        burgundy = masks["burgundy"]
        dual_color = min(support["navy"], support["burgundy"]) >= self.config["quality"]["minimum_secondary_color_fraction"]
        scores = dict(TEAM_A=float((navy | burgundy).mean()) if dual_color else 0.,
                      TEAM_B=support["lilac"], REFEREE=support["yellow"])
        dark = 0. if lower.size == 0 else float((cv2.cvtColor(lower, cv2.COLOR_BGR2HSV)[:,:,2] <= self.config["lower_dark_value_max"]).mean())
        return dict(scores=scores, components=support, torso_pixels=hsv.shape[0]*hsv.shape[1],
                    torso_width=hsv.shape[1], torso_height=hsv.shape[0], lower_dark_fraction=dark, rectangle=rectangle)

    def classify(self, frame, points, other_points=()):
        f = self.features(frame, points)
        q = self.config["quality"]
        reason = None
        if f["torso_width"] < q["minimum_torso_width"] or f["torso_height"] < q["minimum_torso_height"] or f["torso_pixels"] < q["minimum_torso_pixels"]:
            reason = "TOO_SMALL_OR_EMPTY"
        xa, ya, xb, yb = f["rectangle"]
        area = max(1, (xb-xa)*(yb-ya))
        overlap = 0.
        for other in other_points:
            (ox1, oy1), (ox2, oy2) = np.asarray(other)
            overlap = max(overlap, max(0, min(xb, ox2)-max(xa, ox1))*max(0, min(yb, oy2)-max(ya, oy1))/area)
        if overlap > q["maximum_torso_overlap"]:
            reason = reason or "TORSO_OCCLUDED"
        ranked = sorted(f["scores"], key=lambda k:(-f["scores"][k], k))
        winner, second = ranked[:2]
        top, margin = f["scores"][winner], f["scores"][winner]-f["scores"][second]
        thresholds = self.config["decision"][winner]
        if top <= 0 or top < thresholds["minimum_support"]:
            reason = reason or "LOW_COLOR_SUPPORT"
        elif margin < thresholds["minimum_margin"]:
            reason = reason or "AMBIGUOUS_COLORS"
        if winner == "REFEREE" and f["lower_dark_fraction"] < self.config["minimum_referee_lower_dark_fraction"]:
            reason = reason or "YELLOW_WITHOUT_DARK_LOWER_BODY"
        label = "UNKNOWN" if reason else winner
        return metadata(label, reason=reason or "HSV_APPEARANCE", **f, score_margin=margin,
                        torso_overlap=float(overlap), eligible=label in ("TEAM_A", "TEAM_B"))


class TemporalClassifier:
    """Fresh observations only; UNKNOWN never silently becomes a known player.

    A conflicting team observation abstains while retaining stable memory.
    Referee appearance vetoes player eligibility immediately. Long gaps reset IDs.
    """

    def __init__(self, policy):
        self.policy = policy
        self.tracks = {}

    def update(self, track_id, frame0, raw, *, fresh=True):
        result = dict(raw, raw_label=raw["label"], track_id=track_id,
                      classification_source="HSV_TEMPORAL")
        if not fresh:
            return dict(result, **metadata("UNKNOWN"), raw_label=raw["label"], reason="STALE_TRACK", eligible=False)
        state = self.tracks.get(track_id)
        if state is None or frame0-state["frame0"] > self.policy["reset_gap_frames"]:
            state = dict(frame0=frame0, votes=deque(maxlen=self.policy["window"]), stable="UNKNOWN")
            self.tracks[track_id] = state
        if frame0 <= state["frame0"] and state["votes"]:
            raise ValueError("Duplicate or reversed temporal observation")
        state["frame0"] = frame0
        state["votes"].append(raw["label"])
        counts = Counter(state["votes"])
        label, votes = sorted(counts.items(), key=lambda kv:(-kv[1], kv[0]))[0]
        if label != "UNKNOWN" and votes >= self.policy["minimum_votes"] and votes/len(state["votes"]) >= self.policy["majority_fraction"]:
            state["stable"] = label
        result["stable_memory"] = state["stable"]
        result["votes"] = dict(counts)
        if raw["label"] == "REFEREE":
            # Do not wait for inertia to exclude a referee near the ball.
            return dict(result, eligible=False)
        if raw["label"] == "UNKNOWN":
            return dict(result, eligible=False)
        if raw["label"] != state["stable"]:
            reason = "TEMPORAL_WARMUP" if state["stable"] == "UNKNOWN" else "TEMPORAL_CONFLICT"
            result.update(metadata("UNKNOWN"), reason=reason, eligible=False)
        return result


def select_candidate(objects, ball_points, *, ball_count=1, ball_age=0):
    """Only fresh, known-team PLAYERs can be proximity candidates (45px)."""
    eligible = [o for o in objects if o.get("role") == "PLAYER" and o.get("team") in ("TEAM_A", "TEAM_B")
                and o.get("eligible", False) and o.get("age", 0) == 0]
    if ball_count != 1 or ball_points is None or ball_age != 0:
        return None, "INSUFFICIENT_EVIDENCE", "NO_UNIQUE_FRESH_BALL"
    if not eligible:
        return None, "INSUFFICIENT_EVIDENCE", "NO_ELIGIBLE_PLAYER"
    center = np.asarray(ball_points).mean(axis=0)
    def distance(o):
        (x1, _), (x2, y2) = np.asarray(o["points"])
        return min(np.linalg.norm(center-[x1, y2]), np.linalg.norm(center-[x2, y2]))
    closest = min(eligible, key=lambda o:(distance(o), o["track_id"]))
    if distance(closest) > 45:
        return None, "INSUFFICIENT_EVIDENCE", "DISTANCE_GT_45PX"
    return dict(closest, distance_px=float(distance(closest))), "PROXIMITY_EVIDENCE", "NEAREST_ELIGIBLE_PLAYER"
