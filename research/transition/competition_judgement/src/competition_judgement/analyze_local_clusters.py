"""Local-only v2-1 extraction and fixed descriptive comparisons, no new decisions."""
import argparse
import csv
import json
import os
from pathlib import Path
import platform
import sys
import tempfile
import time

import numpy as np
import pandas as pd

from games import GAME,GAMES,half_video_blocks,asset_paths
from mtah_relational.features import max_bipartite_cluster_size
from mtah_relational.frame_store import open_frame_cache
from mtah_relational.splits import SplitManifest,make_split_manifest
from .cli import sha,preservation_snapshot,verify_preserved
from .artifacts import json_write
from .local_clusters import build_mixed_graph
from .local_features import component_features,extract_local_frame
from .local_diagnostics import distribution,run_local_robustness

V1_COLUMNS = ["frame","period","split","n_home","n_away","strict_quality_valid","metric_valid",
              "quality_tier","candidate_score","candidate_state"]


def validate_local_config(c):
    if c["radii_m"] != [3.,5.] or c["primary_radius_m"] != 3. or c["fit_required"] is not False:
        raise ValueError("v2-1 uses separate fixed 3m/5m views and does not fit")
    for key in ("proximity_decay_m","compact_max_diameter_over_radius","compact_max_mean_radius_over_radius"):
        if not np.isfinite(c[key]) or c[key] <= 0:
            raise ValueError("Invalid physical scale")
    if not 0 < c["descriptive_strong_contact_min"] <= 1 or c["small_max_players"] < 2:
        raise ValueError("Invalid descriptive case configuration")


def extract_store(store,v1,config,cluster_path,progress=False):
    if len(store) != len(v1) or not np.array_equal(store.frame_ids,v1.frame) or not np.array_equal(store.periods,v1.period):
        raise ValueError("v1 and raw cache frame/period axes differ")
    # Schema from synthetic two-observation graph, including the all-empty case.
    dummy=build_mixed_graph(np.array([[0.,0.],[1.,0.]]),np.array([0,1]))
    names=["frame","period","quality_tier","strict_quality_valid",*component_features(dummy,dummy.components[3.][0],config)]
    frames=[]
    checks=set(np.linspace(0,len(store)-1,min(31,len(store)),dtype=int))
    parity=0
    with Path(cluster_path).open("w",newline="") as f:
        writer=csv.DictWriter(f,fieldnames=names)
        writer.writeheader()
        for i in range(len(store)):
            summary,rows,graph=extract_local_frame(store.frame(i),store.pitch,config)
            old=v1.iloc[i]
            if (summary["n_home"] != old.n_home or summary["n_away"] != old.n_away or
                summary["strict_quality_valid"] != old.strict_quality_valid or
                summary["local_metric_available"] != old.metric_valid):
                raise ValueError("v1 geometry/quality contract mismatch")
            if i in checks:
                for r in (3.,5.):
                    mask=graph.distance_m[np.ix_(graph.team==0,graph.team==1)] <= r
                    expected=max_bipartite_cluster_size(mask)
                    actual=max((len(c.nodes) for c in graph.components[r]),default=0)
                    assert actual==expected
                    parity+=1
            summary.update(split=old.split,in_official_block=summary["period"] in (1,2),
                           v1_candidate_score=old.candidate_score,v1_candidate_state=old.candidate_state)
            frames.append(summary)
            writer.writerows(rows)
            if progress and (i+1)%25000==0:
                print(f"Local 3m/5m features: {i+1:,} / {len(store):,} frames",flush=True)
    return pd.DataFrame(frames),parity


def descriptive_cases(table,threshold,config):
    out=table.copy()
    ids=out.frame.to_numpy();period=out.period.to_numpy();role=out.split.to_numpy()
    available=out.local_metric_available.to_numpy()
    prior=np.r_[False,(np.diff(ids)==1) & (period[1:]==period[:-1]) & (period[1:]>0) &
                 (role[1:]==role[:-1]) & available[1:] & available[:-1]]
    countchange=np.r_[False,(np.diff(out.n_home)!=0) | (np.diff(out.n_away)!=0)]
    out["case4_comparison_available"]=prior
    out["observed_count_change"]=prior & countchange
    for radius in (3,5):
        out[f"case1_v1_negative_local_strong_{radius}m"]=(out.v1_candidate_state.eq("not_candidate") &
                                                            out[f"has_strong_compact_relation_{radius}m"].eq(True))
        out[f"case2_v1_high_extended_component_{radius}m"]=(out.v1_candidate_score.ge(threshold) &
                                                             out[f"max_cluster_diameter_m_{radius}m"].gt(2*radius))
        out[f"case3_multiple_components_{radius}m"]=out[f"n_local_clusters_{radius}m"].ge(2)
        jump=(out[f"n_local_clusters_{radius}m"].diff().abs().ge(1) |
              out[f"max_local_cluster_size_{radius}m"].diff().abs().ge(config["case4_size_jump_min"]) |
              out[f"n_cross_edges_{radius}m"].diff().abs().ge(config["case4_edge_jump_min"]))
        out[f"case4_count_change_structure_jump_{radius}m"]=prior & countchange & jump.to_numpy()
    return out


def summarize_frames(table):
    result=dict(frames=len(table),local_available=int(table.local_metric_available.sum()),
                strict_valid=int(table.strict_quality_valid.sum()),
                low_but_available=int((table.local_metric_available & ~table.strict_quality_valid).sum()),
                local_undefined=int((~table.local_metric_available).sum()))
    for radius in (3,5):
        values=table[f"n_local_clusters_{radius}m"]
        result[f"radius_{radius}m"]=dict(frames_with_cluster=int(values.gt(0).sum()),
             observed_no_cluster=int(values.eq(0).sum()),frames_multiple=int(values.ge(2).sum()),
             multiple_fraction_available=float(values.ge(2).sum()/table.local_metric_available.sum()) if table.local_metric_available.any() else None,
             n_clusters_histogram={str(int(k)):int(v) for k,v in values.value_counts().sort_index().items()},
             max_strength=distribution(table[f"max_local_contact_strength_{radius}m"]),
             max_size=distribution(table[f"max_local_cluster_size_{radius}m"]))
    return result


def summarize_clusters(path):
    cols=["period","radius_m","quality_tier","n_players","cluster_radius_m","cluster_max_pairwise_distance_m",
          "cross_pair_density","contact_strength","compact_mixed_cluster","same_node_set_across_radii"]
    clusters=pd.read_csv(path,usecols=cols)
    def summarize(t):
        return dict(cluster_observations=len(t),size_histogram={str(int(k)):int(v) for k,v in t.n_players.value_counts().sort_index().items()},
                    compact=int(t.compact_mixed_cluster.sum()),same_node_set_at_other_radius=int(t.same_node_set_across_radii.sum()),
                    distributions={c:distribution(t[c]) for c in ("n_players","cluster_radius_m","cluster_max_pairwise_distance_m","cross_pair_density","contact_strength")})
    return dict(all_by_radius={str(int(r)):summarize(t) for r,t in clusters.groupby("radius_m")},
                official_by_radius={str(int(r)):summarize(t) for r,t in clusters.loc[clusters.period>0].groupby("radius_m")},
                official_by_quality_radius={f"{q}_{int(r)}":summarize(t) for (q,r),t in
                                            clusters.loc[clusters.period>0].groupby(["quality_tier","radius_m"])})


def summarize_cases(table):
    official=table.loc[table.in_official_block]
    result={}
    for name in table.columns:
        if not name.startswith(("case1_","case2_","case3_","case4_count_")):
            continue
        selected=official.loc[official[name]]
        result[name]=dict(frames=len(selected),fraction_official=len(selected)/len(official) if len(official) else None,
                          quality_tiers={str(k):int(v) for k,v in selected.quality_tier.value_counts().items()},
                          distributions={c:distribution(selected[c]) for c in ("n_local_clusters_3m","n_local_clusters_5m",
                          "max_local_cluster_size_3m","max_cluster_diameter_m_3m","max_local_contact_strength_3m",
                          "max_local_cluster_size_5m","max_cluster_diameter_m_5m","max_local_contact_strength_5m","v1_candidate_score")})
    return result


def execute_local(*,root,output,cache,v1_dir,config_path,split_path):
    start=time.perf_counter()
    root,output=Path(root).resolve(),Path(output).resolve()
    if not output.is_relative_to(root/"data/derived/competition_judgement"):
        raise ValueError("Licensed derivatives must stay under data/derived/competition_judgement")
    if output.exists():
        raise FileExistsError("Existing output preserved; choose a new run")
    config=json.loads(Path(config_path).read_text());validate_local_config(config)
    snapshot=preservation_snapshot(root)
    # Include v1 and v2 source, configuration and documents in runtime preservation.
    snapshot["files"].update({str(p):sha(p) for p in (root/"competition_judgement").rglob("*")
                             if p.is_file() and "__pycache__" not in p.parts and p.name!=".DS_Store"})
    v1_dir=Path(v1_dir)
    previous=json.loads((v1_dir/"verification.json").read_text())
    v1_hash=sha(v1_dir/"candidate_frames.csv")
    if previous["candidate_csv_sha256"] != v1_hash:
        raise ValueError("v1 candidate file differs from its recorded immutable output")
    fitted=json.loads((v1_dir/"policy_resolved.json").read_text())["fit"]
    v1=pd.read_csv(v1_dir/"candidate_frames.csv",usecols=V1_COLUMNS)
    store=open_frame_cache(cache)
    if len(store)!=GAMES[GAME]["n_video_frames"] or not np.array_equal(store.frame_ids,np.arange(len(store))):
        raise ValueError("Expected full 199,319-frame raw cache")
    if tuple(store.pitch)!=GAMES[GAME]["pitch"]:
        raise ValueError("Official pitch mismatch")
    source=json.loads((Path(cache)/"cache.json").read_text())["source_sha256"]
    paths=asset_paths()
    if source!={"metadata":sha(paths["atd_meta"]),"raw":sha(paths["atd_raw"])}:
        raise ValueError("Raw-cache source checksum mismatch")
    manifest=SplitManifest.load(split_path)
    if manifest.fingerprint!=make_split_manifest(half_video_blocks(GAME)).fingerprint:
        raise ValueError("Official chronological split mismatch")
    expected_period=np.zeros(len(store),int)
    for period,first,last in half_video_blocks(GAME):expected_period[first:last+1]=period
    if not np.array_equal(store.periods,expected_period) or any(manifest.split_of(int(f))!=(r if r!="outside" else None) for f,r in zip(v1.frame,v1.split)):
        raise ValueError("Official period/split alignment mismatch")
    output.parent.mkdir(parents=True,exist_ok=True)
    with tempfile.TemporaryDirectory(prefix=".local_v2_1_",dir=output.parent) as temp:
        dest=Path(temp);json_write(dest/"preregistered_config.json",config)
        frames,parity=extract_store(store,v1,config,dest/"local_clusters.csv",progress=True)
        frames=descriptive_cases(frames,fitted["threshold"],config)
        frames.to_csv(dest/"local_frame_features.csv",index=False,na_rep="NaN")
        print("Full-frame extraction complete; fixed Validation robustness next",flush=True)
        noise=run_local_robustness(store,manifest,fitted,config)
        report=dict(version=config["version"],config=config,
                    frames=dict(all=summarize_frames(frames),official=summarize_frames(frames.loc[frames.in_official_block]),
                                official_by_quality={str(q):summarize_frames(g) for q,g in frames.loc[frames.in_official_block].groupby("quality_tier")}),
                    clusters=summarize_clusters(dest/"local_clusters.csv"),cases=summarize_cases(frames),robustness=noise,
                    source=dict(raw_sha256=source,v1_candidate_sha256=v1_hash,split_fingerprint=manifest.fingerprint),
                    environment=dict(python=sys.version.split()[0],numpy=np.__version__,pandas=pd.__version__,platform=platform.platform(),device="cpu"),
                    runtime_s=time.perf_counter()-start,maximum_component_parity_checks=parity,
                    notes="Full-match deployment features include Test-time blocks; no feature selection, fitting or tuning on Test. No new candidate state.")
        json_write(dest/"local_cluster_report.json",report)
        (dest/"LOCAL_ONLY.txt").write_text("Licensed match-derived spatial descriptors; keep local, no uploads or push.\nNo contest labels, ball, tagging or GT used.\n")
        assert np.array_equal(frames.frame,np.arange(len(store)))
        assert np.array_equal(frames.v1_candidate_state,v1.candidate_state)
        np.testing.assert_array_equal(frames.v1_candidate_score.to_numpy(),v1.candidate_score.to_numpy())
        assert sha(v1_dir/"candidate_frames.csv")==v1_hash
        verification=dict(**verify_preserved(snapshot),v1_output_unchanged=True,v1_candidate_sha256=v1_hash,
                          frame_rows=len(frames),frame_columns=len(frames.columns),source_code_sha256={str(p.relative_to(root)):sha(p)
                            for p in (root/"competition_judgement/src").rglob("*.py")},
                          output_sha256={p.name:sha(p) for p in (dest/"local_frame_features.csv",dest/"local_clusters.csv")})
        json_write(dest/"verification.json",verification)
        if output.exists():raise FileExistsError("Output appeared during extraction")
        os.rename(dest,output)
    print(json.dumps(dict(output=str(output),official=report["frames"]["official"],runtime_s=report["runtime_s"])))
    return report


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument("--out-dir",required=True)
    p.add_argument("--cache",default="data/derived/relational_temporal_v1_20261109/raw_cache")
    p.add_argument("--v1-dir",default="data/derived/competition_judgement/stage1_v1_final_20261109")
    p.add_argument("--split-manifest",default="data/derived/relational_temporal_v1_20261109/split_manifest.json")
    p.add_argument("--config",default="competition_judgement/configs/local_clusters_v2_1.json")
    a=p.parse_args()
    execute_local(root=Path.cwd(),output=a.out_dir,cache=a.cache,v1_dir=a.v1_dir,
                  config_path=a.config,split_path=a.split_manifest)


if __name__=="__main__":main()
