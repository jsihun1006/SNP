"""Local CSV/JSON summaries. No labels, servers, telemetry or remote clients."""
from collections import Counter
import json

import numpy as np
import pandas as pd


def json_write(path, value):
    path.write_text(json.dumps(value, ensure_ascii=False, indent=2, allow_nan=False)+"\n")


def segments(table):
    """Partition ALL rows by contiguous state, quality, half and split."""
    result = []
    if not len(table):
        return pd.DataFrame(columns=["first_frame", "last_frame", "n_frames", "duration_s",
                                     "candidate_state", "quality_tier", "period", "split"])
    key_cols = ["candidate_state", "quality_tier", "period", "split"]
    key = table[key_cols].astype(str).agg("|".join, axis=1).to_numpy()
    ids = table.frame.to_numpy()
    starts = np.flatnonzero(np.r_[True, (key[1:] != key[:-1]) | (np.diff(ids) != 1)])
    ends = np.r_[starts[1:]-1, len(table)-1]
    for a, b in zip(starts, ends):
        first, last = table.iloc[a], table.iloc[b]
        values = dict(first_frame=int(first.frame), last_frame=int(last.frame), n_frames=int(b-a+1),
                      duration_s=float((b-a+1)/25), **{k: first[k] for k in key_cols})
        score = table.candidate_score.iloc[a:b+1].to_numpy()
        values["mean_candidate_score"] = float(np.nanmean(score)) if np.isfinite(score).any() else np.nan
        result.append(values)
    return pd.DataFrame(result)


def summarize(table):
    n = len(table)
    c = table.candidate_state == "candidate"
    eligible = table.candidate_state != "insufficient_observation"
    s = segments(table)
    durations = s.loc[s.candidate_state == "candidate", "duration_s"].to_numpy()
    return dict(frames=n, strict_quality_frames=int(table.strict_quality_valid.sum()),
                metric_valid_frames=int(table.metric_valid.sum()),
                eligible_frames=int(eligible.sum()),
                eligible_fraction=float(eligible.mean()) if n else None,
                states={state: int((table.candidate_state == state).sum()) for state in
                        ("candidate", "not_candidate", "insufficient_observation")},
                candidate_fraction_all=float(c.mean()) if n else None,
                candidate_fraction_eligible=float(c.sum()/eligible.sum()) if eligible.sum() else None,
                temporal_available_frames=int(table.temporal_available.sum()),
                spatial_fallback_frames=int((eligible & ~table.temporal_available).sum()),
                candidate_segments=len(durations),
                candidate_duration_s=dict(total=float(durations.sum()),
                                          mean=float(durations.mean()) if len(durations) else None,
                                          median=float(np.median(durations)) if len(durations) else None,
                                          p95=float(np.quantile(durations,.95)) if len(durations) else None,
                                          max=float(durations.max()) if len(durations) else None),
                quality_tiers={str(k): int(v) for k, v in Counter(table.quality_tier).items()})


def attach_zones(table, zones_path):
    """Reuse legacy 1fps zone data; NaN at unsampled frames, never forward-fill."""
    z = pd.read_csv(zones_path, usecols=["frame", "period", "zone", "n_players", "mixed_index", "density_per_m2"])
    if z.duplicated(["frame", "zone"]).any() or not z.zone.between(0,47).all():
        raise ValueError("Invalid 48-zone sample schema")
    if len(z) and (not z.groupby("frame").size().eq(48).all() or
                   not np.array_equal(table.period.to_numpy()[z.frame.to_numpy()], z.period)):
        raise ValueError("Zone sample alignment mismatch")
    out = table.copy()
    groups = z.groupby("frame")
    summary = groups.agg(zone_max_mixing=("mixed_index", "max"),
                         zone_max_density_per_m2=("density_per_m2", "max"),
                         zone_total_inside=("n_players", "sum"), zone_max_players=("n_players", "max"))
    summary["zone_mixed_cells"] = groups.mixed_index.apply(lambda x: int((x > 0).sum()))
    for col in summary:
        out[col] = out.frame.map(summary[col])
    out["zone_concentration"] = out.zone_max_players/out.zone_total_inside.replace(0, np.nan)
    out["zone_sample_available"] = out.frame.isin(summary.index)
    return out
