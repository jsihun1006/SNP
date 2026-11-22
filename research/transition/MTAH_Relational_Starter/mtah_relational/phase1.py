"""Streaming local-only QA, relational features and a fixed-rule candidate index."""
from __future__ import annotations

from collections import Counter
from dataclasses import asdict
from pathlib import Path
import csv
import json
import os
import platform
import sys
import tempfile

import numpy as np

from .candidate_baseline import CausalContestedIndex
from .data_quality import ObservationPolicy, assess_observations, iter_player_frames, read_metadata
from .features import geometry_features
from .spatial_schema import (FEATURE_NAMES, SCHEMA_VERSION, TARGET_NAMES,
                             TEAM_STRUCTURE_NAMES, symmetric_summary)
from .zones import zone_metrics


class QualityTotals:
    def __init__(self):
        self.frames = 0
        self.sums = Counter()
        self.frames_with = Counter()
        self.observed = Counter()
        self.home = Counter()
        self.away = Counter()

    def add(self, quality, index):
        self.frames += 1
        self.observed[quality["raw_observed"]] += 1
        self.home[quality["n_home_valid"]] += 1
        self.away[quality["n_away_valid"]] += 1
        for key, value in quality.items():
            self.sums[key] += int(value)
            self.frames_with[key] += int(bool(value))
        self.sums["index_valid"] += int(index["index_valid"])
        self.sums["temporal_available"] += int(index["temporal_available"])

    def report(self, slots):
        denom = self.frames * slots
        return dict(frames=self.frames,
                    mean_observed_players=sum(n*k for n, k in self.observed.items())/self.frames if self.frames else None,
                    missing_slot_fraction=self.sums["missing_slots"]/denom if denom else None,
                    totals=dict(self.sums), frames_with=dict(self.frames_with),
                    observed_players_histogram=dict(sorted(self.observed.items())),
                    valid_home_count_histogram=dict(sorted(self.home.items())),
                    valid_away_count_histogram=dict(sorted(self.away.items())))


def _json(path, value):
    path.write_text(json.dumps(value, indent=2, ensure_ascii=False, allow_nan=False)+"\n", encoding="utf-8")


def run_phase1(*, meta_path, raw_path, output_dir, pitch, fps, n_frames, blocks,
               expected_slots=46, policy=ObservationPolicy(), zone_stride=25,
               save_nodes=False, progress=False):
    """Atomic new directory; no existing outputs overwritten, no network calls.

    Entire raw file is checked before outputs are published. Nodes are optional
    local-only memmaps. Normalized x/y -> official metre pitch; speed is unused.
    """
    if fps != 25:
        raise ValueError("This tactical pipeline requires the official 25 FPS")
    if not isinstance(zone_stride, int) or zone_stride < 0:
        raise ValueError("zone_stride must be a nonnegative integer (0 disables zones)")
    output = Path(output_dir).resolve()
    if output.exists():
        raise FileExistsError("Output directory already exists; choose a new local directory")
    meta = read_metadata(meta_path, expected_fps=fps, expected_slots=expected_slots)
    output.parent.mkdir(parents=True, exist_ok=True)
    groups = {key: QualityTotals() for key in ("all", "official_halves", "half_1", "half_2", "outside_halves")}
    indexer = CausalContestedIndex(fps=int(fps))
    zone_rows = 0
    with tempfile.TemporaryDirectory(prefix=".relational_phase1_", dir=output.parent) as temp:
        dest = Path(temp)
        nodes = None
        observed = None
        if save_nodes:
            nodes = np.lib.format.open_memmap(dest/"source_slot_xy_m.npy", mode="w+", dtype="float32",
                                              shape=(n_frames, len(meta.team_codes), 2))
            observed = np.lib.format.open_memmap(dest/"observed.npy", mode="w+", dtype="bool",
                                                 shape=(n_frames, len(meta.team_codes)))
            np.save(dest/"team_codes.npy", meta.team_codes, allow_pickle=False)
        feature_writer = None
        zone_writer = None
        zone_previous = None
        with (dest/"features.csv").open("w", newline="", encoding="utf-8") as features_file, \
                (dest/"symmetric_features.csv").open("w", newline="", encoding="utf-8") as sym_file, \
                (dest/"zones_48.csv").open("w", newline="", encoding="utf-8") as zones_file:
            sym_names = ["frame", "period"] + list(FEATURE_NAMES) + [f"valid_{n}" for n in FEATURE_NAMES]
            sym_writer = csv.DictWriter(sym_file, fieldnames=sym_names)
            sym_writer.writeheader()
            for frame in iter_player_frames(raw_path, meta, pitch, n_frames, blocks):
                xy, q = assess_observations(frame, pitch, policy)
                f = geometry_features(xy[frame.team_codes == 0], xy[frame.team_codes == 1])
                index = indexer.step(frame.frame, frame.period, f, q["quality_valid"])
                row = dict(frame=frame.frame, period=frame.period, time_s=frame.frame/fps, **f, **q, **index)
                if feature_writer is None:
                    feature_writer = csv.DictWriter(features_file, fieldnames=list(row))
                    feature_writer.writeheader()
                feature_writer.writerow(row)
                values, valid = symmetric_summary(f, pitch)
                sym_writer.writerow(dict(frame=frame.frame, period=frame.period,
                                          **dict(zip(FEATURE_NAMES, values)),
                                          **{f"valid_{n}": bool(v) for n, v in zip(FEATURE_NAMES, valid)}))
                keys = ("all", "official_halves", f"half_{frame.period}") if frame.period else ("all", "outside_halves")
                for key in keys:
                    groups[key].add(q, index)
                if nodes is not None:
                    nodes[frame.frame] = xy
                    observed[frame.frame] = np.isfinite(xy).all(axis=1)
                if zone_stride and frame.period and frame.frame % zone_stride == 0:
                    zones = zone_metrics(xy, frame.team_codes, pitch)
                    reliable_delta = (zone_previous is not None and zone_previous[1] == frame.period
                                      and frame.frame-zone_previous[0] == zone_stride
                                      and q["quality_valid"] and zone_previous[2])
                    for zone in zones:
                        z = zone["zone"]
                        delta = ((zone["density_per_m2"]-zone_previous[3][z])*fps/zone_stride
                                 if reliable_delta else np.nan)
                        zone_row = dict(frame=frame.frame, period=frame.period, quality_valid=q["quality_valid"],
                                        **zone, density_change_per_m2_s=delta)
                        if zone_writer is None:
                            zone_writer = csv.DictWriter(zones_file, fieldnames=list(zone_row))
                            zone_writer.writeheader()
                        zone_writer.writerow(zone_row)
                        zone_rows += 1
                    zone_previous = (frame.frame, frame.period, q["quality_valid"],
                                     [z["density_per_m2"] for z in zones])
                if progress and (frame.frame+1) % 25000 == 0:
                    print(f"Validated {frame.frame+1:,} player-only frames", flush=True)
        if nodes is not None:
            nodes.flush()
            observed.flush()
            del nodes, observed
        report = dict(schema_version=SCHEMA_VERSION, complete=True,
                      source="tactical ATD players only", source_slots_are_person_ids=False,
                      alignment=dict(records=n_frames, strict_contiguous_ids=True,
                                     first_atd_frame_id=1, last_atd_frame_id=n_frames,
                                     first_video_frame=0, last_video_frame=n_frames-1,
                                     mapping="frame=frame_id-1", fps_expected=fps, fps_metadata=meta.fps,
                                     video_fps_independently_measured=False),
                      metadata=dict(team_mapping_source=meta.mapping_source, slot_order_source=meta.slot_order_source,
                                    home_code=0, away_code=1, n_slots=len(meta.team_codes),
                                    n_home_source_slots=int((meta.team_codes == 0).sum()),
                                    n_away_source_slots=int((meta.team_codes == 1).sum()),
                                    xml_pitch_m=meta.xml_pitch, used_pitch_m=list(pitch),
                                    coordinate_units=meta.coordinate_units, speed_used=False),
                      policy=asdict(policy), official_half_blocks=[list(b) for b in blocks],
                      quality={k: v.report(len(meta.team_codes)) for k, v in groups.items()},
                      zones=dict(columns=8, rows=6, stride_frames=zone_stride, rows_written=zone_rows,
                                 global_distances_zone_gated=False),
                      inputs_excluded=["ball", "speed", "smart_tagging", "broadcast", "public_gt"],
                      index=dict(kind="fixed_rule_spatial_contact_and_causal_approach", probability=False,
                                 trained=False, temporal_lag_frames=int(fps), startup="spatial-only if reliable"),
                      runtime=dict(python=sys.version.split()[0], interpreter=sys.executable,
                                   numpy=np.__version__, platform=platform.system()))
        _json(dest/"quality_report.json", report)
        _json(dest/"schema.json", dict(version=SCHEMA_VERSION, symmetric_features=list(FEATURE_NAMES),
                                       future_reconstruction_targets=list(TARGET_NAMES),
                                       team_structure_names=list(TEAM_STRUCTURE_NAMES),
                                       team_structure_order=["H", "A"],
                                       masks="natural feature validity != quality eligibility != artificial span mask",
                                       neural_model_implemented=False))
        (dest/"LOCAL_ONLY.txt").write_text(
            "Licensed match-derived data. Keep local; do not upload/push to public GitHub or cloud LLMs.\n"
            "No manual labels, Smart Tagging, ball data or GT used. Index is not contest probability.\n",
            encoding="utf-8")
        if output.exists():
            raise FileExistsError("Output appeared during extraction; refusing overwrite")
        os.rename(dest, output)
    return report
