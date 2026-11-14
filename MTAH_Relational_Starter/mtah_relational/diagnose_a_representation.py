"""Local, split-contained representation diagnostics; no labels/Test/ball."""
import argparse
import csv
from pathlib import Path
import json
import os
os.environ.setdefault("PYTORCH_ENABLE_MPS_FALLBACK", "1")

import numpy as np
import pandas as pd
import torch

from .a_diagnostic_models import covariance_diagnostics
from .corruption import SpanMasker
from .diagnose_a_training import write_json, restore_variant
from .model_a import load_checkpoint
from .model_a_data import FeatureWindowDataset, build_feature_cache, sampled_indices, batch_arrays
from .reconstruction import ReconstructionMetrics
from .reconstruction_baselines import reconstruct_baselines
from .spatial_schema import FEATURE_NAMES, TARGET_NAMES
from .temporal_cli import open_temporal_dataset
from .train_model_a import select_device, set_seed


class NoArtificialMask:
    def sample(self, length, **routing):
        return np.zeros(length,dtype=bool)


def summary(a):
    a=np.asarray(a)
    return dict(n=len(a),mean=float(a.mean()) if len(a) else None,
                median=float(np.median(a)) if len(a) else None,
                p95=float(np.percentile(a,95)) if len(a) else None)


def correlation(a,b):
    a,b=np.asarray(a),np.asarray(b)
    valid=np.isfinite(a)&np.isfinite(b)
    if valid.sum()<3 or np.std(a[valid])==0 or np.std(b[valid])==0:return None
    return float(np.corrcoef(a[valid],b[valid])[0,1])


def spearman(a,b):
    return correlation(pd.Series(a).rank().to_numpy(),pd.Series(b).rank().to_numpy())


def center_embeddings(model,base,cache,positions,device,*,quality_zero=False):
    base.masker=NoArtificialMask()
    data=FeatureWindowDataset(base,cache)
    a=np.empty((len(positions),64),dtype=np.float32)
    model.eval()
    with torch.inference_mode():
        for first in range(0,len(positions),128):
            x,pad,_,_=batch_arrays(data,positions[first:first+128],with_targets=False)
            if quality_zero:x[...,48:52]=0
            _,h=model(torch.as_tensor(x,device=device),torch.as_tensor(pad,device=device))
            a[first:first+len(x)]=h[:,75].cpu().numpy()
    return a


def analyze_embeddings(train,validation,train_rows,val_rows,base,cache):
    good=np.asarray(cache.quality_valid[val_rows]);train_good=np.asarray(cache.quality_valid[train_rows])
    frames=base.source.frame_ids[val_rows]
    feature=np.asarray(cache.input_values[val_rows,:13],dtype=float)
    valid=np.asarray(cache.input_valid[val_rows,:13])
    quality=np.asarray(cache.quality[val_rows])
    centered=np.asarray(train[train_good],dtype=float)
    mean=centered.mean(axis=0)
    eigen,basis=np.linalg.eigh(np.cov(centered,rowvar=False));basis=basis[:,::-1]
    # PCA is fitted on TRAIN only; validation uses a fixed mean and rotation.
    score=(np.asarray(validation,dtype=float)-mean)@basis
    projected_var=score[good].var(axis=0)
    denom=projected_var.sum()
    ratios=projected_var/denom if denom>0 else np.zeros(64)
    pca=dict(fit_role="train_quality_valid_only",fit_frames=int(train_good.sum()),
             validation_explained_variance_along_train_components=ratios.tolist(),
             validation_cumulative_variance_along_train_components=np.cumsum(ratios).tolist())
    corrs={}
    for k,name in enumerate(TARGET_NAMES):
        select=good&valid[:,k]
        corrs[name]=[correlation(score[select,j],feature[select,k]) for j in range(5)]
    quality_corr={name:[correlation(score[:,j],quality[:,k]) for j in range(5)]
                  for k,name in enumerate(("min_team_count","max_team_count","duplicate_fraction","outside_fraction"))}
    consecutive=np.diff(frames)==1
    delta=np.linalg.norm(np.diff(validation,axis=0),axis=1)
    df=np.diff(feature,axis=0)
    pair_good=consecutive&good[1:]&good[:-1]&valid[1:].all(axis=1)&valid[:-1].all(axis=1)
    count_change=(np.rint(11*quality[1:,:2])!=np.rint(11*quality[:-1,:2])).any(axis=1)
    masks=dict(good_to_good=consecutive&good[:-1]&good[1:],bad_to_bad=consecutive&~good[:-1]&~good[1:],
               good_to_bad=consecutive&good[:-1]&~good[1:],bad_to_good=consecutive&~good[:-1]&good[1:])
    changes={k:summary(delta[m]) for k,m in masks.items()}
    relation=dict(embedding_delta_vs_feature_delta_norm=spearman(delta[pair_good],np.linalg.norm(df[pair_good],axis=1)),
                  by_feature={name:spearman(delta[pair_good],np.abs(df[pair_good,k])) for k,name in enumerate(TARGET_NAMES)},
                  unchanged_counts=summary(delta[pair_good&~count_change]),
                  changed_counts=summary(delta[pair_good&count_change]))
    covariance={name:covariance_diagnostics(validation[m]) for name,m in
                (("all",np.ones(len(validation),dtype=bool)),("quality_valid",good),("quality_bad",~good)) if m.sum()>1}
    return dict(covariance=covariance,train_quality_covariance=covariance_diagnostics(train[train_good]),
                pca=pca,feature_correlations_first5_pcs=corrs,quality_correlations_first5_pcs=quality_corr,
                temporal_delta_per_video_frame=changes,geometry_change_relations=relation,
                analysis_scope="full-context Train/Validation centers only; no deployment boundary mixing; not causal/contest labels")


def feature_dynamics(base,cache):
    rows=base.index.records["center_row"]
    ids=base.source.frame_ids[rows]
    good=np.asarray(cache.quality_valid[rows]);x=np.asarray(cache.input_values[rows,:13],dtype=float)
    counts=np.rint(np.asarray(cache.quality[rows,:2])*11)
    adjacent=(np.diff(ids)==1)&good[:-1]&good[1:]
    changed=(counts[1:]!=counts[:-1]).any(axis=1)
    triple=adjacent[:-1]&adjacent[1:]
    stable_triple=triple&~changed[:-1]&~changed[1:]
    delta=np.diff(x,axis=0);curve=np.diff(x,n=2,axis=0)
    result={}
    for k,name in enumerate(TARGET_NAMES):
        result[name]=dict(absolute_delta=summary(np.abs(delta[adjacent,k])),
                         stable_count_delta=summary(np.abs(delta[adjacent&~changed,k])),
                         changed_count_delta=summary(np.abs(delta[adjacent&changed,k])),
                         stable_count_curvature=summary(np.abs(curve[stable_triple,k])),
                         exact_zero_delta_fraction=float((delta[adjacent,k]==0).mean()),
                         train_scale=float(base.normalizer.scale[k]))
    return result


def interpolation_strata(base,cache,positions,seed):
    report={}
    for label,bounds in (("8-13",(8,13)),("25",(25,25)),("50",(50,50))):
        base.masker=SpanMasker(seed,*bounds);data=FeatureWindowDataset(base,cache)
        metrics={k:ReconstructionMetrics() for k in ("unchanged_counts","changed_counts")}
        groups={k:0 for k in metrics}
        for first in range(0,len(positions),64):
            x,pad,target,windows=batch_arrays(data,positions[first:first+64])
            pred,_=reconstruct_baselines(x,pad)
            changes=[]
            for w in windows:
                t=np.flatnonzero(w.artificial_mask)
                # Diagnostic-only clean counts; never passed to reconstruction.
                times=w.routing.frame_ids[max(0,t[0]-1):min(151,t[-1]+2)]
                rows=np.searchsorted(base.source.frame_ids,times)
                counts=np.rint(11*np.asarray(cache.quality[rows,:2]))
                changes.append(bool((np.diff(counts,axis=0)!=0).any()))
            changes=np.asarray(changes)
            for key,choose in (("changed_counts",changes),("unchanged_counts",~changes)):
                metrics[key].update(pred["interpolation"],target.values,target.observed,
                                    target.quality_weight*choose[:,None,None],target.artificial_mask)
                groups[key]+=int(choose.sum())
        report[label]={k:dict(windows=groups[k],**v.report(TARGET_NAMES,base.normalizer.scale[:13])) for k,v in metrics.items()}
    return report


def run(diagnostic_dir,temporal,cache_path,original_path):
    folder=Path(diagnostic_dir);training=json.loads((folder/"training_report.json").read_text())
    output=folder/"representation"
    if output.exists():raise FileExistsError("Representation diagnostics already exist")
    output.mkdir()
    config=training["config"];set_seed(config["seed"],config["cpu_threads"])
    device,_=select_device(config["device"])
    train_base=open_temporal_dataset(temporal,split="train")
    val_base=open_temporal_dataset(temporal,split="validation")
    cache=build_feature_cache(train_base,cache_path)
    train_pos=sampled_indices(train_base.index,config["representation_train_stride"])
    val_pos=np.arange(len(val_base))
    train_rows=train_base.index.records["center_row"][train_pos];val_rows=val_base.index.records["center_row"]
    # Always analyze the original and strongest Validation challenger, adopted or not.
    kinds=("original",training["decision"]["best_candidate"])
    reports={}
    for kind in kinds:
        model,_=load_checkpoint(original_path,device) if kind=="original" else restore_variant(folder/kind/"best.pt",device)
        train_embedding=center_embeddings(model,train_base,cache,train_pos,device)
        val_embedding=center_embeddings(model,val_base,cache,val_pos,device)
        reports[kind]=analyze_embeddings(train_embedding,val_embedding,train_rows,val_rows,val_base,cache)
        # Fixed, role-only subsample; quality input ablation is diagnostic, no refit.
        subset=val_pos[::25]
        ablated=center_embeddings(model,val_base,cache,subset,device,quality_zero=True)
        reports[kind]["quality_channel_ablation_shift"]=summary(np.linalg.norm(ablated-val_embedding[::25],axis=1))
        np.save(output/f"{kind}_validation_embeddings.npy",val_embedding,allow_pickle=False)
        np.save(output/f"{kind}_train_sample_embeddings.npy",train_embedding,allow_pickle=False)
        print(f"Representation {kind}: validation quality rank={reports[kind]['covariance']['quality_valid']['effective_rank']:.3f}",flush=True)
    np.save(output/"validation_center_frames.npy",val_base.index.records["center_frame"],allow_pickle=False)
    report=dict(representations=reports,feature_dynamics=feature_dynamics(val_base,cache),
                interpolation_count_strata=interpolation_strata(val_base,cache,sampled_indices(val_base.index,25),config["seed"]),
                test_used=False,train_centers=len(train_pos),validation_centers=len(val_pos))
    write_json(output/"report.json",report)
    with (output/"pca_feature_correlations.csv").open("w",newline="") as handle:
        writer=csv.writer(handle);writer.writerow(["model","feature","pc1","pc2","pc3","pc4","pc5"])
        for kind,rep in reports.items():
            for name,corrs in rep["feature_correlations_first5_pcs"].items():writer.writerow([kind,name,*corrs])
    plot(folder,training,report)
    return report


def plot(folder,training,report):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    figs=folder/"figures";figs.mkdir(exist_ok=True)
    fig,axes=plt.subplots(1,2,figsize=(11,4))
    for kind,rep in report["representations"].items():
        good=rep["covariance"]["quality_valid"]
        axes[0].plot(np.arange(1,65),good["cumulative_explained_variance"],label=kind)
        axes[1].plot(np.arange(64),good["dimension_variance"],label=kind)
    axes[0].set(xlabel="PCs (descriptive validation covariance)",ylabel="Cumulative explained variance")
    axes[1].set(xlabel="Embedding dimension",ylabel="Variance",yscale="log")
    for ax in axes:ax.legend();ax.grid(alpha=.2)
    fig.tight_layout();fig.savefig(figs/"pca_variance.png",dpi=170);plt.close(fig)
    fig,axes=plt.subplots(1,2,figsize=(12,4))
    for kind,metrics in training["validation"].items():
        axes[0].plot(("8-13","25","50"),[metrics[m]["metrics"]["bilstm"]["mae_standardized"] for m in ("8-13","25","50")],marker="o",label=kind)
        axes[1].plot(("8-13","25","50"),[metrics[m]["metrics"]["bilstm"]["huber"] for m in ("8-13","25","50")],marker="o",label=kind)
    for ax,key in zip(axes,("mae_standardized","huber")):
        ax.plot(("8-13","25","50"),[training["validation"]["original"][m]["metrics"]["interpolation"][key] for m in ("8-13","25","50")],marker="x",linestyle="--",color="black",label="interpolation")
        ax.set(xlabel="Consecutive masked frames",ylabel=f"Validation {key}");ax.legend(fontsize=7);ax.grid(alpha=.2)
    fig.tight_layout();fig.savefig(figs/"validation_comparison.png",dpi=170);plt.close(fig)


def main():
    p=argparse.ArgumentParser(description=__doc__);p.add_argument("--diagnostic-dir",required=True)
    p.add_argument("--data",default="data/derived/relational_temporal_v1_20261109")
    p.add_argument("--cache",default="data/derived/model_a_feature_cache_v1")
    p.add_argument("--original",default="data/derived/model_a_run_v2_20261109/best.pt")
    a=p.parse_args();run(a.diagnostic_dir,a.data,a.cache,a.original)


if __name__=="__main__":main()
