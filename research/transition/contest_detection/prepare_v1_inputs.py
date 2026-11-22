"""Generate v1 caches, chronological split and Phase1 features from original ATD.

Uses the preserved generators unchanged. Does not download/create raw ATD,
train models, run candidate inference or overwrite an existing output.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys
import tempfile

ROOT = Path(__file__).resolve().parents[1]
for directory in (ROOT, ROOT / "MTAH_Relational_Starter"):
    if str(directory) not in sys.path:
        sys.path.insert(0, str(directory))

from games import GAME, GAMES, asset_paths, half_video_blocks
from mtah_relational.data_quality import ObservationPolicy
from mtah_relational.frame_store import build_frame_cache
from mtah_relational.phase1 import run_phase1
from mtah_relational.splits import make_split_manifest

CACHE_FILES = (
    "cache.json", "frame_ids.npy", "periods.npy", "xy_m.npy", "team_codes.npy",
    "parse_missing.npy", "parse_partial.npy", "parse_invalid.npy",
)
FEATURE_FILES = ("features.csv", "quality_report.json", "zones_48.csv")


def prepare_inputs(*, meta_path, raw_path, cache_dir, split_path, features_dir,
                   pitch, fps, n_frames, blocks, only="all", expected_slots=46,
                   policy=ObservationPolicy(), progress=True):
    """Call original generators. Explicit dimensions permit tiny synthetic tests."""
    if only not in ("all", "cache", "split", "features"):
        raise ValueError("Choose all, cache, split or features")
    if fps != 25:
        raise ValueError("The preserved v1 generators require 25 FPS")
    paths = dict(cache=Path(cache_dir).resolve(), split=Path(split_path).resolve(),
                 features=Path(features_dir).resolve())
    selected = list(paths) if only == "all" else [only]
    conflicts = [str(paths[key]) for key in selected if paths[key].exists() or paths[key].is_symlink()]
    if conflicts:
        raise FileExistsError("Outputs already exist; choose new paths or --only: " + ", ".join(conflicts))
    # Reject overlapping outputs before a generator creates either directory.
    for index, left in enumerate(selected):
        for right in selected[index + 1:]:
            a, b = paths[left], paths[right]
            if a.is_relative_to(b) or b.is_relative_to(a):
                raise ValueError("Output paths must be distinct and not nested")
    originals = [Path(meta_path).resolve(), Path(raw_path).resolve()]
    before = {}
    if any(key in selected for key in ("cache", "features")):
        missing = [str(path) for path in originals if not path.is_file()]
        if missing:
            raise FileNotFoundError("Original ATD inputs required: " + ", ".join(missing))
        before = {path: (path.stat().st_size, path.stat().st_mtime_ns) for path in originals}
        if any(path == dest or path.is_relative_to(dest) for path in originals for dest in paths.values()):
            raise ValueError("Original ATD must remain outside generated output paths")
    generated = {}
    if "cache" in selected:
        print("Creating player-only raw_cache with build_frame_cache...", flush=True)
        store = build_frame_cache(meta_path=originals[0], raw_path=originals[1],
                                  output_dir=paths["cache"], pitch=pitch, n_frames=n_frames,
                                  blocks=blocks, expected_slots=expected_slots, progress=progress)
        generated["cache"] = dict(path=str(paths["cache"]), required_files=list(CACHE_FILES),
                                  source_fingerprint=store.fingerprint)
        del store
    if "split" in selected:
        print("Creating split_manifest.json with make_split_manifest...", flush=True)
        manifest = make_split_manifest(blocks)
        target = paths["split"]
        target.parent.mkdir(parents=True, exist_ok=True)
        with tempfile.TemporaryDirectory(prefix=".v1_split_", dir=target.parent) as temporary:
            staged = Path(temporary) / "split_manifest.json"
            manifest.save(staged)
            with target.open("xb") as output:
                output.write(staged.read_bytes())
        generated["split"] = dict(path=str(target), fingerprint=manifest.fingerprint)
    if "features" in selected:
        print("Creating features/quality/zones with run_phase1 (zone_stride=25)...", flush=True)
        run_phase1(meta_path=originals[0], raw_path=originals[1], output_dir=paths["features"],
                   pitch=pitch, fps=fps, n_frames=n_frames, blocks=blocks,
                   expected_slots=expected_slots, policy=policy, zone_stride=25,
                   save_nodes=False, progress=progress)
        generated["features"] = dict(path=str(paths["features"]), required_files=list(FEATURE_FILES),
                                     additional_original_outputs=["symmetric_features.csv", "schema.json", "LOCAL_ONLY.txt"])
    if any((path.stat().st_size, path.stat().st_mtime_ns) != stat for path, stat in before.items()):
        raise RuntimeError("Original ATD size/mtime changed during preparation")
    return dict(generated=generated, original_inputs_preserved=True,
                training_performed=False, candidate_inference_performed=False)


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--repo-root", type=Path, default=ROOT)
    parser.add_argument("--data-dir", type=Path, help="Root containing tactical/BAR-ATH_tactical_atd.xml and .txt")
    parser.add_argument("--cache-dir", type=Path)
    parser.add_argument("--split-manifest", type=Path)
    parser.add_argument("--features-dir", type=Path)
    parser.add_argument("--only", choices=("all", "cache", "split", "features"), default="all")
    parser.add_argument("--plan", action="store_true", help="List inputs/outputs without reading tracking or generating files")
    args = parser.parse_args(argv)
    root = args.repo_root.resolve()
    source_paths = asset_paths(data_dir=str(args.data_dir.resolve()) if args.data_dir else None)
    temporal = root / "data/derived/relational_temporal_v1_20261109"
    cache = args.cache_dir or temporal / "raw_cache"
    split = args.split_manifest or temporal / "split_manifest.json"
    features = args.features_dir or root / "data/derived/relational_phase1_v1_20261109"
    if args.plan:
        result = dict(only=args.only, inputs={key: dict(path=str(Path(source_paths[key]).resolve()),
                      exists=Path(source_paths[key]).is_file()) for key in ("atd_meta", "atd_raw")},
                      outputs={key: dict(path=str(Path(value).resolve()), exists=Path(value).exists())
                               for key, value in dict(cache=cache, split=split, features=features).items()},
                      creates_raw_atd=False, writes_performed=False)
    else:
        game = GAMES[GAME]
        try:
            result = prepare_inputs(meta_path=source_paths["atd_meta"], raw_path=source_paths["atd_raw"],
                                    cache_dir=cache, split_path=split, features_dir=features,
                                    pitch=game["pitch"], fps=game["fps"], n_frames=game["n_video_frames"],
                                    blocks=half_video_blocks(GAME), only=args.only)
        except (OSError, ValueError) as error:
            parser.exit(2, str(error) + "\n")
    print(json.dumps(result, indent=2, ensure_ascii=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
