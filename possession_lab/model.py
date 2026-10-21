"""A three-state, gap-aware hidden Markov model for possession.

States: H (home), A (away), D (dead ball / out of play).  The emissions are deliberately simple:

* when the ball is visible, a Gaussian in the distance to each team's nearest player;
* D gets a sigmoid on the nearest-player distance plus an out-of-play bonus;
* when the ball is missing, short gaps are uniform (the transition prior carries the state);
* long gaps use the team shape around the last known ball position, the last known edge, and a
  weak smart-tag vote;
* a ball far from every player is treated as loose/in flight: uniform, or the last-touch team when
  the ball is also slow, so distance alone cannot vote a live ball dead.

The Viterbi pass is CPU-only.  It is a 199k-step, 3-state dynamic program; a GPU would spend more
time copying arrays than computing, so the package does not pretend otherwise.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import numpy as np

from .config import FPS, MODEL_PARAMS, X


@dataclass(frozen=True)
class ModelParams:
    sigma: float = 3.5
    d_dead: float = 7.0
    dead_scale: float = 1.5
    edge0: float = 5.0
    edge_scale: float = 1.5
    w_edge: float = 0.0
    w_gap: float = 0.0
    gap0: float = 100.0
    gap_scale: float = 25.0
    w_out: float = 1.0
    w_dead: float = 0.0
    w_motion: float = 0.0
    motion0: float = 1.6
    motion_scale: float = 0.35
    w_tag: float = 0.0
    w_sp: float = 0.0
    w_lt: float = 0.0
    lt_R: float = 0.8
    p_stay: float = 0.995
    p_ha: float = 0.0005
    p_hd: float = 0.001
    p_dh: float = 0.01
    gate: float = 0.0
    loose_R: float = 7.0
    loose_speed: float = 2.0
    touch_age: int = 125
    touch_p: float = 0.7
    miss_mode: str = "hybrid"
    sigma_miss: float = 6.0
    gap_thresh: float = 120.0
    min_players: int = 6

    @classmethod
    def from_mapping(cls, values: dict[str, Any] | None = None) -> "ModelParams":
        if values is None:
            return cls(**MODEL_PARAMS)
        merged = dict(MODEL_PARAMS)
        merged.update(values)
        return cls(**merged)


DEFAULT_PARAMS = ModelParams.from_mapping()


@dataclass(frozen=True)
class DecodeResult:
    codes: np.ndarray
    posterior: np.ndarray
    confidence: np.ndarray
    entropy: np.ndarray


def _sigmoid(x: np.ndarray | float) -> np.ndarray:
    return 1.0 / (1.0 + np.exp(-x))


def _log_normalize(values: np.ndarray) -> np.ndarray:
    values = np.asarray(values, dtype=np.float64)
    normalizer = values.sum(axis=1, keepdims=True)
    normalizer = np.where(normalizer > 0, normalizer, 1.0)
    return np.log(values / normalizer)


def dead_ball_evidence(features: Any, params: ModelParams = DEFAULT_PARAMS) -> np.ndarray:
    """Additive support for the D state from stoppage evidence.

    This is deliberately separate from the team-distance emission.  Players walk or stand still
    during a stoppage, and the ball sits near a touchline for throw-ins, corners and goal kicks.
    ``w_dead`` is a base-rate offset: it keeps the dead-ball prior independent of the scale of the
    team Gaussians.
    """
    edge = np.where(np.isfinite(features.edge), features.edge, features.last_edge)
    edge = np.where(np.isfinite(edge), edge, 20.0)
    motion = _sigmoid(
        (params.motion0 - np.asarray(features.team_speed, dtype=np.float64)) / params.motion_scale
    )
    return (
        params.w_dead
        + params.w_motion * motion
        + params.w_edge * _sigmoid((params.edge0 - edge) / params.edge_scale)
    )


def emission_log_probs(features: Any, params: ModelParams = DEFAULT_PARAMS) -> np.ndarray:
    """Return an ``(n, 3)`` array of log emission probabilities."""
    n = len(np.asarray(features.present))
    if n == 0:
        return np.zeros((0, 3), dtype=np.float64)

    d_home = np.asarray(features.d_home, dtype=np.float64)
    d_away = np.asarray(features.d_away, dtype=np.float64)
    present = np.asarray(features.present, dtype=bool)
    last_edge = np.asarray(features.last_edge, dtype=np.float64)
    gap = np.asarray(features.gap, dtype=np.float64)
    out_of_play = np.asarray(features.out_of_play, dtype=bool)

    finite_home = np.isfinite(d_home)
    finite_away = np.isfinite(d_away)
    q_home = np.where(
        finite_home,
        np.exp(-(d_home * d_home) / (2.0 * params.sigma * params.sigma)),
        1e-12,
    ) + 1e-12
    q_away = np.where(
        finite_away,
        np.exp(-(d_away * d_away) / (2.0 * params.sigma * params.sigma)),
        1e-12,
    ) + 1e-12

    d_min = np.minimum(d_home, d_away)
    finite_min = np.isfinite(d_min)
    q_dead = np.where(
        finite_min,
        _sigmoid((d_min - params.d_dead) / params.dead_scale),
        0.0,
    ) + 1e-12

    q_dead += dead_ball_evidence(features, params)
    if params.w_gap:
        q_dead += params.w_gap * _sigmoid((gap - params.gap0) / params.gap_scale)
    q_dead += params.w_out * out_of_play

    log_emission = _log_normalize(np.stack([q_home, q_away, q_dead], axis=1))

    # Speed gate.  The reference model applies this even when ``gate`` is zero: a stationary or
    # unknown ball speed then gets a 50/50 blend with the uniform emission.  That is not a no-op,
    # so keep the operation unconditional to reproduce the validated configuration.
    speed = np.nan_to_num(np.asarray(features.ball_speed, dtype=np.float64), nan=0.0)
    alpha = 1.0 - _sigmoid((speed - params.gate) / 2.0)
    alpha = np.where(present, alpha, 0.0)
    log_emission = alpha[:, None] * log_emission + (1.0 - alpha[:, None]) * np.log(1.0 / 3.0)

    log_emission[~present] = np.log(1.0 / 3.0)
    log_emission[out_of_play] = np.log(np.array([1e-6, 1e-6, 1.0 - 2e-6]))

    if params.w_tag:
        vote = np.asarray(features.tag_vote, dtype=np.float64)
        log_emission[:, 0] += params.w_tag * np.maximum(vote, 0.0)
        log_emission[:, 1] += params.w_tag * np.maximum(-vote, 0.0)
    if params.w_sp:
        log_emission[:, 2] += params.w_sp * np.asarray(features.tag_setpiece, dtype=np.float64)

    if params.w_lt:
        touch = (
            present
            & (np.asarray(features.last_touch) >= 0)
            & (np.asarray(features.near_distance, dtype=np.float64) <= params.lt_R)
        )
        touch_team = np.asarray(features.last_touch, dtype=np.int8)
        log_emission[:, 0] += params.w_lt * touch * (touch_team == 0)
        log_emission[:, 1] += params.w_lt * touch * (touch_team == 1)

    # Missing-ball emissions.  The hybrid mode is the recommendation: keep short holes uniform so
    # the transition prior does the work, and switch to the geometric prior only for long holes.
    missing = ~present
    if missing.any():
        if params.miss_mode == "persist":
            log_emission[missing] = np.log(1.0 / 3.0)
        else:
            missing_q = np.ones((n, 3), dtype=np.float64)
            if params.miss_mode in {"prox", "both", "hybrid", "hybrid_prox"}:
                d_home_last = np.nan_to_num(
                    np.asarray(features.d_home_last, dtype=np.float64), nan=1e6
                )
                d_away_last = np.nan_to_num(
                    np.asarray(features.d_away_last, dtype=np.float64), nan=1e6
                )
                missing_q[:, 0] = (
                    np.exp(-(d_home_last * d_home_last) / (2.0 * params.sigma_miss * params.sigma_miss))
                    + 1e-6
                )
                missing_q[:, 1] = (
                    np.exp(-(d_away_last * d_away_last) / (2.0 * params.sigma_miss * params.sigma_miss))
                    + 1e-6
                )
                last_edge_safe = np.where(np.isfinite(last_edge), last_edge, 20.0)
                missing_q[:, 2] = 0.15 + _sigmoid((params.edge0 - last_edge_safe) / 1.5)
            if params.miss_mode in {"tag", "both", "hybrid"}:
                vote = np.asarray(features.tag_vote, dtype=np.float64)
                missing_q[:, 0] *= 1.0 + np.maximum(vote, 0.0)
                missing_q[:, 1] *= 1.0 + np.maximum(-vote, 0.0)
                missing_q[:, 2] *= 1.0 + 0.2 * np.asarray(features.tag_setpiece, dtype=np.float64)
            missing_q = missing_q / missing_q.sum(axis=1, keepdims=True)
            log_emission[missing] = np.log(missing_q[missing])
            if params.miss_mode in {"hybrid", "hybrid_prox"}:
                short = missing & (gap <= params.gap_thresh)
                log_emission[short] = np.log(1.0 / 3.0)

    # If the tracking feed itself is almost empty, do not read team shape from three lonely
    # away slots.  Treat the frame as missing evidence and let the transition prior persist.
    if params.min_players > 0 and hasattr(features, "player_count"):
        sparse = missing & (np.asarray(features.player_count) < params.min_players)
        log_emission[sparse] = np.log(1.0 / 3.0)

    # A ball that is far from every player is loose or in flight, not evidence of a dead ball.
    # Let the prior carry the possession instead of letting the distance term vote `D`.  When
    # that loose ball is also slow, its only possession evidence is the team that touched it
    # last, so give that team the mass and leave the rest for the prior.
    if params.loose_R > 0:
        loose = present & (d_min > params.loose_R) & ~out_of_play
        log_emission[loose] = np.log(1.0 / 3.0)
        if params.touch_p > 0:
            last_touch = np.asarray(features.last_touch, dtype=np.int8)
            recent = loose & (speed < params.loose_speed) & (last_touch >= 0)
            recent = recent & (
                np.asarray(features.last_touch_age, dtype=np.int64) <= params.touch_age
            )
            if recent.any():
                touched = last_touch[recent]
                q = np.full((touched.size, 3), (1.0 - params.touch_p) / 2.0)
                q[np.arange(touched.size), touched] = params.touch_p
                log_emission[recent] = np.log(q)

    return log_emission


def transition_matrix(params: ModelParams = DEFAULT_PARAMS) -> np.ndarray:
    return np.array(
        [
            [params.p_stay, params.p_ha, params.p_hd],
            [params.p_ha, params.p_stay, params.p_hd],
            [params.p_dh, params.p_dh, 1.0 - 2.0 * params.p_dh],
        ],
        dtype=np.float64,
    )


def viterbi(log_emission: np.ndarray, params: ModelParams = DEFAULT_PARAMS) -> np.ndarray:
    """Standard Viterbi for three states, using log-space arithmetic."""
    log_emission = np.asarray(log_emission, dtype=np.float64)
    n = log_emission.shape[0]
    if n == 0:
        return np.zeros(0, dtype=np.int8)
    log_transition = np.log(transition_matrix(params))
    back = np.zeros((n, 3), dtype=np.int8)
    score = np.log(np.array([1.0 / 3.0, 1.0 / 3.0, 1.0 / 3.0])) + log_emission[0]
    for t in range(1, n):
        candidates = score[:, None] + log_transition
        back[t] = np.argmax(candidates, axis=0)
        score = np.max(candidates, axis=0) + log_emission[t]
    states = np.zeros(n, dtype=np.int8)
    states[-1] = int(np.argmax(score))
    for t in range(n - 1, 0, -1):
        states[t - 1] = back[t, states[t]]
    return states


def merge_isolated_runs(codes: np.ndarray, min_frames: int = FPS) -> np.ndarray:
    """Absorb brief label flips that return to the same label.

    Viterbi can drop a possession for a few frames when the ball is between players, in flight,
    or briefly untracked.  A run shorter than ``min_frames`` whose neighbours agree is a flicker,
    not a new possession, so the neighbours take it back.  ``X`` frames are never changed.
    """
    out = np.asarray(codes, dtype=np.int8).copy()
    if min_frames <= 1 or out.size < 3:
        return out
    starts = np.flatnonzero(np.r_[True, out[1:] != out[:-1]])
    stops = np.r_[starts[1:], out.size]
    for start, stop in zip(starts, stops):
        if (
            stop - start < min_frames
            and 0 < start
            and stop < out.size
            and out[start] != X
            and out[start - 1] == out[stop] != X
        ):
            out[start:stop] = out[start - 1]
    return out


def _logsumexp(values: np.ndarray, axis: int) -> np.ndarray:
    maximum = np.max(values, axis=axis, keepdims=True)
    return maximum + np.log(np.sum(np.exp(values - maximum), axis=axis, keepdims=True))


def forward_backward(
    log_emission: np.ndarray, params: ModelParams = DEFAULT_PARAMS
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Posterior, per-frame confidence, and entropy for diagnostics.

    These are not used for the submission labels; they are how the report highlights uncertain
    gaps.  The forward-backward recursion is in log space so long missing runs stay stable.
    """
    log_emission = np.asarray(log_emission, dtype=np.float64)
    n = log_emission.shape[0]
    if n == 0:
        return np.zeros((0, 3)), np.zeros(0), np.zeros(0)
    log_transition = np.log(transition_matrix(params))
    log_start = np.log(np.array([1.0 / 3.0, 1.0 / 3.0, 1.0 / 3.0]))

    alpha = np.empty((n, 3), dtype=np.float64)
    alpha[0] = log_start + log_emission[0]
    for t in range(1, n):
        alpha[t] = log_emission[t] + _logsumexp(alpha[t - 1][:, None] + log_transition, axis=0)[0]

    beta = np.zeros((n, 3), dtype=np.float64)
    for t in range(n - 2, -1, -1):
        beta[t] = _logsumexp(
            log_transition + log_emission[t + 1][None, :] + beta[t + 1][None, :],
            axis=1,
        )[:, 0]

    log_posterior = alpha + beta
    log_posterior -= _logsumexp(log_posterior, axis=1)
    posterior = np.exp(log_posterior)
    confidence = np.max(posterior, axis=1)
    entropy = -np.sum(np.where(posterior > 0, posterior * np.log(posterior), 0.0), axis=1)
    return posterior, confidence, entropy


def decode(
    features: Any,
    params: ModelParams = DEFAULT_PARAMS,
    *,
    with_confidence: bool = True,
) -> DecodeResult:
    log_emission = emission_log_probs(features, params)
    codes = viterbi(log_emission, params)
    if with_confidence:
        posterior, confidence, entropy = forward_backward(log_emission, params)
    else:
        posterior = np.zeros((len(features), 3), dtype=np.float64)
        confidence = np.zeros(len(features), dtype=np.float64)
        entropy = np.zeros(len(features), dtype=np.float64)
    return DecodeResult(codes=codes, posterior=posterior, confidence=confidence, entropy=entropy)
