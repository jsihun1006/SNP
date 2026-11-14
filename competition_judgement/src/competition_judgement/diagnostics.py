"""Fixed perturbation sensitivity on Validation only, without labels or tuning."""
from dataclasses import replace

import numpy as np
import pandas as pd

from .quality import observation_row
from .detector import compute_evidence, apply_policy
from .artifacts import summarize

CONDITIONS = (("jitter", .1), ("jitter", .5), ("jitter", 1.),
              ("dropout", .1), ("dropout", .2), ("dropout", .3),
              ("composition", .2), ("home_shift", 1.), ("permutation", 0.), ("team_swap", 0.))


def perturb(frame, kind, strength, seed=2027, active=True):
    """Copy player data only. Same random draw scaled across perturbation levels."""
    rng = np.random.default_rng(np.random.SeedSequence([seed, frame.frame]))
    xy = frame.xy_m.copy()
    teams = frame.team_codes.copy()
    flags = {k: getattr(frame,k).copy() for k in ("parse_missing", "parse_partial", "parse_invalid")}
    observed = np.isfinite(xy).all(axis=1)
    if kind == "jitter":
        xy[observed] += rng.normal(size=xy.shape)[observed]*strength
    elif kind in ("dropout", "composition"):
        remove = observed & (rng.random(len(xy)) < strength) & (active or kind == "dropout")
        xy[remove] = np.nan
        flags["parse_missing"][remove] = True
        flags["parse_partial"][remove] = False
        flags["parse_invalid"][remove] = False
    elif kind == "home_shift" and active:
        xy[observed & (teams == 0), 0] += strength
    elif kind == "permutation":
        order = rng.permutation(len(xy))
        xy, teams = xy[order], teams[order]
        flags = {k: v[order] for k,v in flags.items()}
    elif kind == "team_swap":
        teams = 1-teams
    elif kind not in ("clean", "home_shift"):
        raise ValueError("Unknown synthetic perturbation")
    return replace(frame, xy_m=xy, team_codes=teams, **flags)


def validation_supports(manifest):
    """Twelve fixed, separated 151-frame supports; never inspect score to choose."""
    supports = []
    for b in manifest.blocks:
        if b.split != "validation":
            continue
        if b.last-b.first < 150:
            raise ValueError("Validation block too short for diagnostic context")
        for start in np.linspace(b.first, b.last-150, 6, dtype=int):
            supports.append((int(start), int(start+150)))
    return supports


def compare_frames(clean, changed):
    if not np.array_equal(clean.frame, changed.frame):
        raise ValueError("Perturbations require identical frame support")
    joint = np.isfinite(clean.candidate_score) & np.isfinite(changed.candidate_score)
    delta = np.abs(clean.loc[joint,"candidate_score"].to_numpy()-changed.loc[joint,"candidate_score"].to_numpy())
    known = (clean.candidate_state != "insufficient_observation") & (changed.candidate_state != "insufficient_observation")
    q = summarize(changed)
    q.update(joint_score_frames=int(joint.sum()),
             score_abs_change_p50=float(np.median(delta)) if len(delta) else None,
             score_abs_change_p95=float(np.quantile(delta,.95)) if len(delta) else None,
             candidate_noncandidate_flips=int(((clean.candidate_state != changed.candidate_state) & known).sum()),
             entered_insufficient=int(((clean.candidate_state != "insufficient_observation") &
                                       (changed.candidate_state == "insufficient_observation")).sum()),
             left_insufficient=int(((clean.candidate_state == "insufficient_observation") &
                                    (changed.candidate_state != "insufficient_observation")).sum()))
    return q


def robustness(store, manifest, fitted, min_run=5):
    supports = validation_supports(manifest)
    clean = []
    # Each support has its own warmup for all conditions, no state crosses samples.
    for first,last in supports:
        observations = [store.frame(store.position(f)) for f in range(first,last+1)]
        table = pd.DataFrame([observation_row(f,store.pitch) for f in observations])
        clean.append(apply_policy(compute_evidence(table,manifest), fitted,min_run))
    joined = pd.concat(clean,ignore_index=True)
    report = dict(seed=2027, supports=[list(x) for x in supports], frames=len(joined),
                  role="validation", sample_selection="six evenly spaced 151-frame supports per Validation block",
                  warmup="independent 25-frame causal history and run gate per support",
                  full_validation=False, threshold_refit=False, clean=summarize(joined), conditions={})
    for kind, strength in CONDITIONS:
        changed = []
        for first,last in supports:
            rows = []
            for offset, frame_id in enumerate(range(first,last+1)):
                f = store.frame(store.position(frame_id))
                p = perturb(f,kind,strength,active=65 <= offset <= 85)
                rows.append(observation_row(p,store.pitch))
            changed.append(apply_policy(compute_evidence(pd.DataFrame(rows),manifest),fitted,min_run))
        report["conditions"][f"{kind}_{strength:g}"] = compare_frames(joined,pd.concat(changed,ignore_index=True))
    return report
