"""Command-line extraction of team and cross-team geometry from MTAH tactical ATD.

From the official root use PYTHONPATH="$PWD/MTAH_Relational_Starter:$PWD".
"""
from __future__ import annotations

import argparse
import os
from pathlib import Path

import numpy as np

from .features import geometry_features, add_temporal_features
from .data_quality import (ObservationPolicy, iter_player_frames, parse_players,
                           read_metadata)


def _official():
    try:
        import games
    except ImportError as exc:
        raise RuntimeError(
            "Run from the official root with "
            "PYTHONPATH=\"$PWD/MTAH_Relational_Starter:$PWD\"."
        ) from exc
    return games, None


def _meta_team_codes(path: str) -> np.ndarray:
    return read_metadata(path, expected_fps=None, expected_slots=None,
                         require_explicit_teams=False).team_codes


def _parse_player_groups(groups: str, n_slots: int, pitch: tuple[float, float]):
    return parse_players(groups, n_slots, pitch)[0].astype(np.float32)


def _period(frame: int, blocks) -> int:
    for half, first, last in blocks:
        if first <= frame <= last:
            return half
    return 0


def run_extract(start: int, count: int | None, out: str, save_nodes: bool) -> None:
    import pandas as pd
    games, _ = _official()
    paths = games.asset_paths(source="tactical")
    for key in ("atd_meta", "atd_raw"):
        if not os.path.isfile(paths[key]):
            raise FileNotFoundError(paths[key])
    codes = _meta_team_codes(paths["atd_meta"])
    n_frames = games.source_frames("tactical")
    if start < 0 or start >= n_frames:
        raise ValueError(f"--start must be within 0..{n_frames-1}")
    if count is not None and count <= 0:
        raise ValueError("--count must be positive")
    end = n_frames if count is None else min(n_frames, start + count)
    pitch = games.GAMES[games.GAME]["pitch"]
    blocks = games.half_video_blocks(games.GAME)
    rows = []
    node_xy = []
    frame_ids = []
    node_period = []
    meta = read_metadata(paths["atd_meta"], expected_fps=None, expected_slots=None,
                         require_explicit_teams=False)
    for observation in iter_player_frames(paths["atd_raw"], meta, pitch, n_frames, blocks):
        frame = observation.frame
        if not start <= frame < end:
            continue
        xy = observation.xy_m
        features = geometry_features(xy[codes == 0], xy[codes == 1])
        features["frame"] = frame
        features["period"] = _period(frame, blocks)
        rows.append(features)
        if save_nodes:
            node_xy.append(xy)
            frame_ids.append(frame)
            node_period.append(features["period"])
    if len(rows) != end-start:
        raise RuntimeError(f"Expected {end-start} rows, got {len(rows)}. Check ATD continuity.")
    df = pd.DataFrame(rows)
    df = df[["frame", "period"] + [c for c in df if c not in ("frame", "period")]]
    df = add_temporal_features(df)
    output = Path(out)
    output.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(output, index=False)
    print(f"Wrote {len(df):,} feature rows: {output}")
    print(f"Coverage: {len(df.query('period > 0')):,} rows during halves; "
          f"low-coverage flag: {int(df.low_coverage.sum()):,}")
    if save_nodes:
        node_path = output.with_name("raw_nodes.npz")
        np.savez_compressed(node_path,
            frame=np.asarray(frame_ids, dtype=np.int32),
            period=np.asarray(node_period, dtype=np.int8),
            xy_m=np.stack(node_xy),
            observed=np.isfinite(np.stack(node_xy)).all(axis=-1),
            team_codes=codes)
        print(f"Wrote raw node positions: {node_path}")
    print("Note: these are unlabeled geometry features; no contested labels inferred.")


def run_check():
    games, _ = _official()
    paths = games.asset_paths(source="tactical")
    game = games.GAMES[games.GAME]
    meta = read_metadata(paths["atd_meta"], expected_fps=game["fps"])
    count = sum(1 for _ in iter_player_frames(paths["atd_raw"], meta, game["pitch"],
                                             games.source_frames("tactical"),
                                             games.half_video_blocks(games.GAME)))
    print(f"Strict player-only alignment passed: {count:,} records; FPS {meta.fps}; "
          "frame=frame_id-1; explicit Local/Visiting mapping.")


def run_plot(source: str, output: str):
    import pandas as pd
    import matplotlib.pyplot as plt
    df = pd.read_csv(source)
    fig, axes = plt.subplots(3, 1, figsize=(12, 8), sharex=True)
    x = df["frame"] / 25.0 / 60.0
    for ax, name, title in zip(
        axes,
        ["cross_pairs_3m", "max_cross_cluster_3m", "centroid_sep_m"],
        ["Opponent pairs <= 3m", "Largest cross-team proximity cluster", "Team centroid separation (m)"]
    ):
        ax.plot(x, df[name].values, linewidth=0.8)
        ax.set_ylabel(title)
        ax.grid(alpha=.25)
    axes[-1].set_xlabel("Tactical video minute")
    fig.tight_layout()
    dest = Path(output)
    dest.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(dest, dpi=140)
    plt.close(fig)
    print(f"Saved diagnostics: {dest}")


def main():
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    sub = parser.add_subparsers(dest="command", required=True)
    sub.add_parser("check", help="Check ATD/tactical frame alignment")
    phase1 = sub.add_parser("phase1", help="Player-only integrity, relations and fixed-rule candidate index")
    phase1.add_argument("--out-dir", default="data/derived/relational_phase1")
    phase1.add_argument("--zone-stride", type=int, default=25, help="Auxiliary zone sampling in frames; 0 disables")
    phase1.add_argument("--save-nodes", action="store_true", help="Optional LOCAL ONLY source-slot tensors")
    phase1.add_argument("--bounds-margin-m", type=float, default=2.0)
    phase1.add_argument("--duplicate-radius-m", type=float, default=0.2)
    phase1.add_argument("--min-team-players", type=int, default=8)
    extract = sub.add_parser("extract", help="Extract geometry / graph summary features")
    extract.add_argument("--start", type=int, default=11185)
    extract.add_argument("--count", type=int, default=5000)
    extract.add_argument("--full", action="store_true", help="Process all 199319 frames")
    extract.add_argument("--out", default="data/derived/relational_features.csv")
    extract.add_argument("--save-nodes", action="store_true")
    plot = sub.add_parser("plot", help="Plot features produced by extract")
    plot.add_argument("--input", default="data/derived/relational_features.csv")
    plot.add_argument("--out", default="data/derived/feature_diagnostics.png")
    args = parser.parse_args()
    if args.command == "check":
        run_check()
    elif args.command == "phase1":
        from .phase1 import run_phase1
        games, _ = _official()
        paths = games.asset_paths(source="tactical")
        game = games.GAMES[games.GAME]
        policy = ObservationPolicy(bounds_margin_m=args.bounds_margin_m,
                                   duplicate_radius_m=args.duplicate_radius_m,
                                   min_team_players=args.min_team_players)
        report = run_phase1(meta_path=paths["atd_meta"], raw_path=paths["atd_raw"],
                            output_dir=args.out_dir, pitch=game["pitch"], fps=game["fps"],
                            n_frames=games.source_frames("tactical"),
                            blocks=games.half_video_blocks(games.GAME), policy=policy,
                            zone_stride=args.zone_stride, save_nodes=args.save_nodes, progress=True)
        print(f"Completed {report['alignment']['records']:,} local-only frames: {args.out_dir}")
        print("Fixed-rule index only. No contest probabilities or H/A/D labels; no training.")
    elif args.command == "extract":
        run_extract(0 if args.full else args.start, None if args.full else args.count,
                    args.out, args.save_nodes)
    else:
        run_plot(args.input, args.out)


if __name__ == "__main__":
    main()
