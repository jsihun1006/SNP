"""Stage 0/1 only. Run from repository root with the two local packages on PYTHONPATH."""
import argparse
from dataclasses import asdict
import hashlib
import json
import os
from pathlib import Path
import platform
import sys
import tempfile
import time

import numpy as np
import pandas as pd

from games import GAME, GAMES, asset_paths, half_video_blocks
from mtah_relational.frame_store import open_frame_cache
from mtah_relational.phase1 import run_phase1
from mtah_relational.splits import SplitManifest, make_split_manifest
from mtah_relational.data_quality import read_metadata
from .quality import read_features, observation_row, policy_dict
from .detector import compute_evidence, fit_threshold, apply_policy, fingerprint
from .artifacts import json_write, segments, summarize, attach_zones
from .diagnostics import robustness


def sha(path):
    h = hashlib.sha256()
    with Path(path).open("rb") as f:
        for chunk in iter(lambda: f.read(1024*1024), b""):
            h.update(chunk)
    return h.hexdigest()


def validate_config(config):
    c = config["candidate_rule"]
    if (c["kind"] != "train_only_quantile" or c["fit_on"] != "unique_train_raw_frames" or
        c["fit_quality"] != "existing_strict_valid" or c["bridge_unknown_gap_frames"] != 0 or
        config["time"]["fps"] != 25 or config["time"]["cross_half_history"] or
        config["quality"]["primary"] != "existing_strict_valid" or
        config["features"]["global_pair_radii_m"] != [3,5,10] or config["features"]["zone_grid"] != [8,6]):
        raise ValueError("Unsupported Stage 1 policy; do not silently change semantics")
    if c["min_consecutive_valid_frames"] < 1 or not 0 < c["quantile"] < 1:
        raise ValueError("Invalid threshold/run policy")


def preservation_snapshot(root):
    files = {}
    for folder in ("starter", "evaluation", "MTAH_Relational_Starter", "docs"):
        for p in (root/folder).rglob("*"):
            if p.is_file() and "__pycache__" not in p.parts and p.suffix != ".pyc":
                files[str(p)] = sha(p)
    for p in (root/"games.py",root/".gitignore"):
        files[str(p)] = sha(p)
    artifacts = {str(p): [p.stat().st_size, p.stat().st_mtime_ns]
                 for folder in ("data/tactical", "data/broadcast", "data/derived")
                 for p in (root/folder).rglob("*") if p.is_file()}
    return dict(files=files,artifacts=artifacts)


def verify_preserved(snapshot):
    if any(not Path(p).is_file() or sha(p) != h for p,h in snapshot["files"].items()):
        raise RuntimeError("Protected source/baseline file changed")
    if any(not Path(p).is_file() or [Path(p).stat().st_size, Path(p).stat().st_mtime_ns] != s
           for p,s in snapshot["artifacts"].items()):
        raise RuntimeError("Existing source data or artifact changed")
    return dict(protected_files_unchanged=len(snapshot["files"]),
                previous_data_artifacts_unchanged=len(snapshot["artifacts"]))


def baseline_validation(root):
    from evaluation import possession_metrics as pm
    result = {}
    for path in sorted((root/"evaluation/results").glob("baseline_*.csv")):
        table = pm.read_label_csv(path)
        error = pm.validate_submission(table, GAMES[GAME]["n_video_frames"])
        if error:
            raise ValueError(f"Existing baseline format invalid: {path.name}: {error}")
        result[path.name] = dict(rows=len(table), valid=True, sha256=sha(path))
    if len(result) != 4:
        raise ValueError("Expected four existing submission baselines")
    return result


def execute(*, root, output, features, cache, split_path, config_path):
    start = time.perf_counter()
    root, output = Path(root).resolve(), Path(output).resolve()
    if not output.is_relative_to(root/"data/derived/competition_judgement"):
        raise ValueError("Actual-match outputs must stay under data/derived/competition_judgement")
    if output.exists():
        raise FileExistsError("Refusing to overwrite an existing run")
    config = json.loads(Path(config_path).read_text())
    validate_config(config)
    snapshot = preservation_snapshot(root)
    baselines = baseline_validation(root)
    blocks, game = half_video_blocks(GAME), GAMES[GAME]
    manifest = SplitManifest.load(split_path)
    expected = make_split_manifest(blocks)
    if manifest.fingerprint != expected.fingerprint:
        raise ValueError("Existing raw-time split does not match published half contracts")
    paths = asset_paths()
    metadata = read_metadata(paths["atd_meta"])
    store = open_frame_cache(cache)
    cache_info = json.loads((Path(cache)/"cache.json").read_text())
    source_hash = {"metadata":sha(paths["atd_meta"]), "raw":sha(paths["atd_raw"])}
    if cache_info["source_sha256"] != source_hash or not np.array_equal(store.team_codes,metadata.team_codes):
        raise ValueError("Player-only cache provenance mismatch")
    if len(store) != game["n_video_frames"] or tuple(store.pitch) != tuple(game["pitch"]):
        raise ValueError("Official full-frame/pitch contract mismatch")
    output.parent.mkdir(parents=True,exist_ok=True)
    with tempfile.TemporaryDirectory(prefix=".stage1_",dir=output.parent) as temp:
        dest = Path(temp)
        json_write(dest/"preregistered_policy.json",config)
        features = Path(features)
        if not features.exists():
            generated = dest/"player_features"
            run_phase1(meta_path=paths["atd_meta"],raw_path=paths["atd_raw"],output_dir=generated,
                       pitch=game["pitch"],fps=game["fps"],n_frames=game["n_video_frames"],blocks=blocks)
            features = generated/"features.csv"
        quality_report = json.loads((features.parent/"quality_report.json").read_text())
        if quality_report["policy"] != policy_dict():
            raise ValueError("Existing feature quality policy differs")
        table = read_features(features,game["n_video_frames"],blocks)
        if not np.array_equal(store.frame_ids,table.frame) or not np.array_equal(store.periods,table.period):
            raise ValueError("Raw cache and feature frame/period axes differ")
        # Fixed independent provenance spots, without selecting by candidate score.
        for row in np.linspace(0,len(store)-1,31,dtype=int):
            f = observation_row(store.frame(row),store.pitch)
            for key,value in f.items():
                if not np.isclose(table.iloc[row][key],value,equal_nan=True):
                    raise ValueError("Raw-cache/Phase1 geometry mismatch")
        print("Stage 0 passed: source contracts, baseline format and protected-file snapshot",flush=True)
        evidence = compute_evidence(table,manifest)
        old = pd.read_csv(features,usecols=["contested_index","temporal_available"])
        if not np.array_equal(evidence.baseline_score.to_numpy(),old.contested_index.to_numpy(),equal_nan=True):
            # CSV decimal parsing may differ by the last bit after recomputing exp().
            if not np.allclose(evidence.baseline_score,old.contested_index,atol=1e-14,rtol=0,equal_nan=True):
                raise ValueError("Unmodified strict baseline was not reproduced")
        if not np.array_equal(evidence.baseline_temporal_available,old.temporal_available):
            raise ValueError("Strict temporal availability changed")
        fitted = fit_threshold(evidence,config["candidate_rule"]["quantile"])
        main = apply_policy(evidence,fitted,config["candidate_rule"]["min_consecutive_valid_frames"])
        zone_path = features.parent/"zones_48.csv"
        if not zone_path.exists() or zone_path.stat().st_size == 0:
            raise ValueError("Existing auxiliary zone samples required; regenerate Phase1 with zone_stride=25")
        main = attach_zones(main,zone_path)
        summary = dict(all=summarize(main),official=summarize(main.loc[main.in_official_block]),
                       outside=summarize(main.loc[~main.in_official_block]),
                       by_split={s:summarize(main.loc[main.split==s]) for s in ("train","validation","test")},
                       by_period={str(p):summarize(main.loc[main.period==p]) for p in (1,2)},
                       by_quality={str(q):summarize(g) for q,g in main.loc[main.in_official_block].groupby("quality_tier")},
                       by_count_change={str(k):summarize(g) for k,g in main.loc[main.in_official_block & main.count_change_available].groupby("count_change_prev_frame")})
        sensitivity = []
        for quantile in config["sensitivity_only"]["quantiles"]:
            fit = fit_threshold(evidence,quantile)
            for minimum in config["sensitivity_only"]["minimum_run_frames"]:
                variant = apply_policy(evidence,fit,minimum)
                sensitivity.append(dict(**fit,min_run=minimum,official=summarize(variant.loc[variant.in_official_block]),
                                        validation=summarize(variant.loc[variant.split=="validation"])))
        print("Full-frame candidate pass complete; running fixed Validation perturbations",flush=True)
        noise = robustness(store,manifest,fitted,config["candidate_rule"]["min_consecutive_valid_frames"])
        # All full-match rows are deployment output; no Test result chooses any rule.
        resolved = dict(config=config,config_sha256=fingerprint(config),fit=fitted,
                        observation_policy=policy_dict(),split_manifest=asdict(manifest),
                        split_fingerprint=manifest.fingerprint,source_sha256=source_hash,
                        phase1_feature_sha256=sha(features),causal=True,
                        startup="spatial fallback; split/half/gap reset",
                        persistence="causal confirmation; no retroactive relabelling; four-frame onset delay",
                        secondary_policy="metric coverage only; relaxed candidate policy deferred to Stage2")
        report = dict(version="candidate-stage1-v1",summary=summary,threshold_sensitivity=sensitivity,
                      robustness=noise,baseline_reproduced=True,baseline_atol=1e-14,
                      split_reset_changed_score_frames=int((~np.isclose(evidence.candidate_score,evidence.baseline_score,
                                                                       atol=1e-14,equal_nan=True)).sum()),
                      baseline_validation=baselines,quality_policy=policy_dict(),
                      probability=False,labels_used=False,test_used_for_selection=False,
                      threshold_fitted_once_on_train=True,window_counting=False,
                      audit=dict(official_frames=int(main.in_official_block.sum()),
                                 outside_frames=int((~main.in_official_block).sum()),
                                 frame_rows=len(main),alignment_valid=np.array_equal(main.frame,np.arange(len(store))),
                                 geometry_cache_spot_checks=31,zone_sample_frames=int(main.zone_sample_available.sum())),
                      environment=dict(python=sys.version.split()[0],numpy=np.__version__,pandas=pd.__version__,
                                       platform=platform.platform(),neural_framework_used=False),
                      runtime_s=time.perf_counter()-start)
        report["audit"]["alignment_valid"] = bool(report["audit"]["alignment_valid"])
        json_write(dest/"policy_resolved.json",resolved)
        json_write(dest/"quality_and_sensitivity.json",report)
        main.to_csv(dest/"candidate_frames.csv",index=False,na_rep="NaN")
        seg = segments(main)
        seg.to_csv(dest/"candidate_segments.csv",index=False,na_rep="NaN")
        assert int(seg.n_frames.sum()) == len(main)
        schema = json.loads((root/"competition_judgement/schemas/candidate_frame_columns.json").read_text())
        assert all(c["name"] in main for c in schema["columns"] if c["required"])
        assert set(main.candidate_state).issubset({"candidate","not_candidate","insufficient_observation"})
        (dest/"LOCAL_ONLY.txt").write_text("Licensed match derivatives. Local only: no upload, publication or cloud LLM.\nCandidate evidence is not a probability or H/A/D/X label.\n")
        preservation = verify_preserved(snapshot)
        json_write(dest/"verification.json",dict(**preservation,baseline_validation=baselines,
                   candidate_csv_sha256=sha(dest/"candidate_frames.csv"),candidate_rows=len(main),
                   source_code_sha256={str(p.relative_to(root)):sha(p) for p in
                                       (root/"competition_judgement/src").rglob("*.py")},
                   split_fingerprint=manifest.fingerprint,source_data_unchanged=True))
        if output.exists():
            raise FileExistsError("Output appeared during run")
        os.rename(dest,output)
    print(json.dumps(dict(output=str(output),rows=len(main),official=summary["official"],threshold=fitted["threshold"]),ensure_ascii=False))
    return report


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--out-dir",required=True)
    p.add_argument("--features",default="data/derived/relational_phase1_v1_20261109/features.csv")
    p.add_argument("--cache",default="data/derived/relational_temporal_v1_20261109/raw_cache")
    p.add_argument("--split-manifest",default="data/derived/relational_temporal_v1_20261109/split_manifest.json")
    p.add_argument("--config",default="competition_judgement/configs/initial_policy.json")
    a = p.parse_args()
    execute(root=Path.cwd(),output=a.out_dir,features=a.features,cache=a.cache,
            split_path=a.split_manifest,config_path=a.config)


if __name__ == "__main__":
    main()
