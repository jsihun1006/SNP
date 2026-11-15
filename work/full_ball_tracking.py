"""Select observed ball candidates, retaining uncertain alternatives explicitly.

No model, image decoder, interpolation or manual frame override is used here.
``selected_observation`` is an automated choice, never a confirmed annotation.
Heuristic thresholds are screening rules, not calibrated probabilities.
"""
from __future__ import annotations

import math
import numpy as np


VERSION = "full_half_observed_ball_v1"
IDENTITY = np.eye(2, 3)


def _context(context, name, index, default):
    try:
        return context[name][index]
    except (KeyError, IndexError, TypeError):
        return default


def _finite_number(value, default=None):
    try:
        number = float(value)
        return number if math.isfinite(number) else default
    except (TypeError, ValueError):
        return default


def _point(candidate):
    try:
        point = np.asarray(candidate["point"], dtype=float)
        return point if point.shape == (2,) and np.isfinite(point).all() else None
    except (KeyError, TypeError, ValueError):
        return None


def _diameter(candidate):
    try:
        box = np.asarray(candidate["bbox_xyxy"], dtype=float)
        if box.shape == (4,) and np.isfinite(box).all():
            return math.hypot(max(0., box[2] - box[0]), max(0., box[3] - box[1]))
    except (KeyError, TypeError, ValueError):
        pass
    return 12.


def evidence(candidate, prior, h_error):
    """Return an observation score and auditable, non-probabilistic features."""
    point = _point(candidate)
    score = _finite_number(candidate.get("confidence"), -1.)
    if point is None or not 0 <= score <= 1:
        return -1.e6, {"invalid_candidate": True, "prior_support": 0.,
                       "soft_field_mark_penalty": 0., "background_penalty": 0.}
    error = _finite_number(h_error)
    h_quality = error is not None and 0 <= error <= 2
    prior = np.asarray(prior, dtype=float)
    distance = _finite_number(candidate.get("atd_distance_px"))
    if distance is None and prior.shape == (2,) and np.isfinite(prior).all():
        distance = float(np.linalg.norm(point - prior))
    sigma = max(25., min(70., _diameter(candidate) * 2))
    support = math.exp(-.5 * (distance / sigma) ** 2) if distance is not None and h_quality else 0.
    spot = _finite_number(candidate.get("penalty_spot_distance_m"), 1.e9)
    mark_penalty = 1.65 * math.exp(-.5 * (spot / .9) ** 2) * (1 - support) if h_quality else 0.
    green = min(1., max(0., _finite_number(candidate.get("green_fraction"), .5)))
    pitch = candidate.get("pitch")
    off_field = False
    if pitch is not None and h_quality:
        pitch = np.asarray(pitch, dtype=float)
        off_field = bool(pitch.shape == (2,) and np.isfinite(pitch).all() and
                         (pitch[0] < -.025 or pitch[0] > 1.025 or
                          pitch[1] < -.025 or pitch[1] > 1.025))
    border = bool(point[1] < 190 or point[1] > 1040)
    background_penalty = .8 * (1 - green) if border or off_field else 0.
    nonfield_suspect = bool((border or off_field) and green < .35)
    unary = 2.8 * score - .7 + 1.1 * support - mark_penalty - background_penalty
    return unary, dict(prior_support=support, atd_distance_px=distance,
                       prior_quality_usable=bool(h_quality and distance is not None),
                       soft_field_mark_penalty=mark_penalty,
                       background_penalty=background_penalty,
                       nonfield_suspected=nonfield_suspect,
                       off_field_ground_projection=off_field,
                       invalid_candidate=False)


def _warp(context, index):
    matrix = np.asarray(_context(context, "warps", index, IDENTITY), dtype=float)
    return matrix if matrix.shape == (2, 3) and np.isfinite(matrix).all() else IDENTITY


def _is_cut(context, index):
    return bool(_context(context, "cuts", index, False))


def _motion_valid(context, index):
    return bool(_context(context, "motion_valid", index, True))


def _transition(previous, current, context, index):
    costs = np.zeros((len(previous) + 1, len(current) + 1), dtype=float)
    if _is_cut(context, index):
        return costs
    costs[1:, 0] = -.20
    costs[0, 1:] = -.25
    warp = _warp(context, index)
    reliable = _motion_valid(context, index)
    for p, old in enumerate(previous):
        old_point = _point(old)
        if old_point is None:
            costs[p + 1, 1:] = -2.
            continue
        predicted = warp @ np.r_[old_point, 1.]
        for q, new in enumerate(current):
            point = _point(new)
            if point is None:
                costs[p + 1, q + 1] = -2.
                continue
            scale = max(24., min(90., _diameter(new) * 1.5))
            residual = float(np.linalg.norm(predicted - point)) / scale
            # Missing motion evidence is not a camera cut. Soften geometry;
            # never treat an unreliable identity warp as precise compensation.
            penalty = .65 * min(7., residual ** 2)
            costs[p + 1, q + 1] = (.18 - penalty) if reliable else (.04 - .2 * penalty)
    return costs


def _consistent(records, selected, context, first, last):
    """Real observed endpoint agreement within at most two same-shot frames."""
    if not 0 <= first < last < len(records) or last - first > 2:
        return False
    if selected[first] < 0 or selected[last] < 0:
        return False
    if records[last]["frame"] - records[first]["frame"] != last - first:
        return False
    if any(_is_cut(context, k) or not _motion_valid(context, k)
           for k in range(first + 1, last + 1)):
        return False
    old = records[first]["candidates"][selected[first]]
    new = records[last]["candidates"][selected[last]]
    predicted, target = _point(old), _point(new)
    if predicted is None or target is None:
        return False
    for k in range(first + 1, last + 1):
        predicted = _warp(context, k) @ np.r_[predicted, 1.]
    allowance = max(24., min(90., max(_diameter(old), _diameter(new)) * 1.5))
    allowance *= last - first
    return bool(np.linalg.norm(predicted - target) <= allowance)


def _same_location(left, right):
    a, b = _point(left), _point(right)
    return bool(a is not None and b is not None and
                np.linalg.norm(a - b) <= max(12., .5 * (_diameter(left) + _diameter(right))))


def solve(records, context):
    """Return one decision per input frame, with separate uncertain coordinates.

    ``records`` have ``frame`` and immutable ``candidates`` entries. Optional
    context arrays are ``warps`` (previous -> current image pixels),
    ``motion_valid``, ``cuts`` (hard cuts only), ``atd_pixel`` and ``h_error``.
    Record order must be contiguous. Candidate index zero is a valid state.
    """
    n = len(records)
    if not n:
        return []
    if any(records[i]["frame"] != records[0]["frame"] + i for i in range(n)):
        raise ValueError("Input source frames must be contiguous and ordered")
    unaries, features, forwards, transitions, backs = [], [], [], [], []
    previous_values = np.array([0.])
    previous_candidates = []
    for i, record in enumerate(records):
        candidates = record["candidates"]
        info = [evidence(c, _context(context, "atd_pixel", i, [np.nan, np.nan]),
                         _context(context, "h_error", i, np.nan)) for c in candidates]
        unary = np.array([0.] + [v[0] for v in info])
        transition = (_transition(previous_candidates, candidates, context, i)
                      if i else np.zeros((1, len(candidates) + 1)))
        values = previous_values[:, None] + transition
        pred = np.argmax(values, axis=0)
        current = values[pred, np.arange(len(unary))] + unary
        current -= np.max(current)
        unaries.append(unary)
        features.append([v[1] for v in info])
        forwards.append(current)
        transitions.append(transition)
        backs.append(pred)
        previous_values, previous_candidates = current, candidates

    selected = np.empty(n, dtype=int)
    state = int(np.argmax(previous_values))
    for i in range(n - 1, -1, -1):
        selected[i] = state - 1
        state = int(backs[i][state])
    futures = [None] * n
    futures[-1] = np.zeros_like(unaries[-1])
    for i in range(n - 2, -1, -1):
        future = np.max(transitions[i + 1] + unaries[i + 1][None, :] +
                        futures[i + 1][None, :], axis=1)
        futures[i] = future - np.max(future)

    decisions = []
    for i, record in enumerate(records):
        candidates = record["candidates"]
        marginal = forwards[i] + futures[i]
        marginal -= np.max(marginal)
        index = int(selected[i])
        path_observed = index >= 0
        # Keep a reviewable detected alternative even when the best path is
        # unknown. It remains provisional, never an invented missing position.
        if index < 0 and candidates:
            order = np.argsort(marginal[1:])[::-1]
            index = next((int(j) for j in order if not features[i][int(j)]["invalid_candidate"]), -1)
        item = dict(source_frame=int(record["frame"]), local_frame=i,
                    candidate_index=index, status="unresolved", pixel_x=None, pixel_y=None,
                    provisional_pixel_x=None, provisional_pixel_y=None,
                    candidate_confidence=None, evidence=None, origin=None,
                    needs_review=True, visual_reviewed=False, manually_verified=False,
                    algorithm_version=VERSION, review_candidates=[])
        if index < 0:
            decisions.append(item)
            continue
        candidate = candidates[index]
        feat = dict(features[i][index])
        if feat["invalid_candidate"]:
            item["candidate_index"] = -1
            decisions.append(item)
            continue
        point = _point(candidate)
        score = float(candidate["confidence"])
        left = any(_consistent(records, selected, context, j, i) for j in range(max(0, i - 2), i))
        right = any(_consistent(records, selected, context, i, j) for j in range(i + 1, min(n, i + 3)))
        bidirectional = bool(left and right)
        near_cut = any(_is_cut(context, k) for k in range(max(1, i - 2), min(n, i + 3)))
        motion_uncertain = bool(i and not _motion_valid(context, i))
        alternatives = [j for j, other in enumerate(candidates)
                        if j != index and not features[i][j]["invalid_candidate"] and
                        not _same_location(candidate, other)]
        alternative_gap = (float(marginal[index + 1] - max(marginal[j + 1] for j in alternatives))
                           if alternatives else None)
        ambiguous = bool(alternative_gap is not None and alternative_gap < .30)
        null_margin = float(marginal[index + 1] - marginal[0])
        prior_supported = bool(score >= .18 and feat["prior_support"] >= .65)
        temporal_supported = bool(score >= .22 and bidirectional)
        model_supported = bool(score >= .60)
        atd_conflict = bool(feat["prior_quality_usable"] and feat["atd_distance_px"] > 120)
        suspicious_mark = bool(feat["soft_field_mark_penalty"] >= .8 and not prior_supported)
        suspicious_background = bool(feat["nonfield_suspected"] and not prior_supported)
        # ATD disagreement alone is not a hard rejection: a clear, coherent
        # observed ball may disagree with automatic ATD (especially airborne).
        unresolved_conflict = bool(atd_conflict and not (score >= .80 and bidirectional))
        accepted = bool(path_observed and (prior_supported or temporal_supported or model_supported)
                        and not ambiguous and not suspicious_mark and not suspicious_background
                        and not unresolved_conflict and null_margin >= -.000001)
        needs_review = bool(not accepted or near_cut or motion_uncertain or atd_conflict)
        feat.update(temporal_left=bool(left), temporal_right=bool(right),
                    bidirectional_geometry=bidirectional, near_camera_cut=bool(near_cut),
                    camera_motion_uncertain=motion_uncertain, atd_conflict=atd_conflict,
                    competing_candidate=ambiguous, alternative_path_gap=alternative_gap,
                    observed_vs_unknown_path_gap=null_margin,
                    path_selected_observation=bool(path_observed))
        item.update(candidate_confidence=score, evidence=feat,
                    origin=candidate.get("origin", "original_detector"),
                    needs_review=needs_review, near_camera_cut=bool(near_cut))
        if accepted:
            item.update(status="selected_observation", pixel_x=float(point[0]), pixel_y=float(point[1]))
        else:
            item.update(status="review_candidate", provisional_pixel_x=float(point[0]),
                        provisional_pixel_y=float(point[1]))
        ranked = sorted((j for j in range(len(candidates)) if not features[i][j]["invalid_candidate"]),
                        key=lambda j: float(marginal[j + 1]), reverse=True)
        for j in ranked[:5]:
            other = candidates[j]
            other_point = _point(other)
            item["review_candidates"].append(dict(candidate_index=j,
                pixel_x=float(other_point[0]), pixel_y=float(other_point[1]),
                confidence=float(other["confidence"]), path_score_gap=float(-marginal[j + 1]),
                origin=other.get("origin", "original_detector")))
        item["alternative_candidates_total"] = len(ranked)
        decisions.append(item)
    return decisions


def _self_test():
    def candidate(x, y=500., score=.8, **extra):
        return dict(point=[x, y], bbox_xyxy=[x - 4, y - 4, x + 4, y + 4],
                    confidence=score, green_fraction=.8, **extra)

    def records(values):
        return [dict(frame=100 + i, candidates=row) for i, row in enumerate(values)]

    ctx = dict(warps=np.tile(IDENTITY, (3, 1, 1)), cuts=np.zeros(3, bool),
               motion_valid=np.ones(3, bool), h_error=np.full(3, np.nan),
               atd_pixel=np.full((3, 2), np.nan))
    result = solve(records([[candidate(100)], [candidate(105, score=.22)], [candidate(110)]]), ctx)
    assert result[1]["status"] == "selected_observation" and result[1]["candidate_index"] == 0
    assert result[1]["evidence"]["bidirectional_geometry"]
    assert result[1]["pixel_x"] == 105.
    gap = solve(records([[candidate(100)], [], [candidate(110)]]), ctx)
    assert gap[1]["status"] == "unresolved" and gap[1]["pixel_x"] is None
    distant = solve(records([[candidate(100)], [candidate(900, score=.22)], [candidate(110)]]), ctx)
    assert distant[1]["status"] == "review_candidate" and distant[1]["pixel_x"] is None
    assert not distant[1]["evidence"]["bidirectional_geometry"]
    cut_context = dict(ctx, cuts=np.array([False, True, False]))
    cut = solve(records([[candidate(100)], [candidate(900)], [candidate(905)]]), cut_context)
    assert cut[1]["status"] == "selected_observation" and cut[1]["needs_review"]
    assert not cut[1]["evidence"]["temporal_left"]
    weak_context = dict(ctx, motion_valid=np.array([True, False, True]))
    weak = solve(records([[candidate(100)], [candidate(105, score=.22)], [candidate(110)]]), weak_context)
    assert not weak[1]["evidence"]["bidirectional_geometry"]
    ambiguous = solve(records([[candidate(100), candidate(900)] for _ in range(3)]), ctx)
    assert all(row["status"] == "review_candidate" for row in ambiguous)
    assert len(ambiguous[1]["review_candidates"]) == 2
    prior_context = dict(ctx, h_error=np.zeros(3), atd_pixel=np.array([[100, 500.]] * 3))
    prior = solve(records([[candidate(100, score=.18)] for _ in range(3)]), prior_context)
    assert prior[1]["status"] == "selected_observation"
    # Camera translation, not raw pixel displacement, supports the low-score point.
    pan_context = dict(ctx, warps=np.array([IDENTITY, [[1, 0, 200], [0, 1, 0]],
                                          [[1, 0, 200], [0, 1, 0]]]))
    pan = solve(records([[candidate(100)], [candidate(305, score=.22)], [candidate(510)]]), pan_context)
    assert pan[1]["status"] == "selected_observation"
    assert all(row["visual_reviewed"] is False for row in pan)
    assert solve([], {}) == []
    print("Passed CPU checks: observed index0, low-score temporal recovery, empty gap, distant mismatch, "
          "camera cut, unreliable motion, ambiguity pool, weak prior and compensated camera pan.")


if __name__ == "__main__":
    _self_test()
