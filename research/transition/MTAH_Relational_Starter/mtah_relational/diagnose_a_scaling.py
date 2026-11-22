"""Train-only scale/discrete component audit; read-only, no fitting or tuning."""
import argparse
import json
from pathlib import Path

import numpy as np

from .diagnose_a_training import write_json
from .model_a_data import build_feature_cache
from .temporal_cli import open_temporal_dataset


def audit(temporal,cache_path,diagnostic_dir):
    base=open_temporal_dataset(temporal,split="train")
    cache=build_feature_cache(base,cache_path)
    select=np.zeros(len(base.source),dtype=bool)
    for block in base.index.blocks:
        if block.split=="train":
            select|=(base.source.frame_ids>=block.first)&(base.source.frame_ids<=block.last)
    rows=np.flatnonzero(select&cache.quality_valid)
    # Existing schema: component reference count = max component size /22.
    component=np.rint((cache.input_values[rows,17]*base.normalizer.scale[17]+base.normalizer.median[17])*22).astype(int)
    counts=np.rint(11*cache.quality[rows,:2]).sum(axis=1).astype(int)
    values,frequency=np.unique(component,return_counts=True)
    n,f=np.unique(counts,return_counts=True)
    q25,q75=np.percentile(component/counts,[25,75])
    stats=dict(train_quality_frames=len(rows),component_size_counts=dict(zip(values.astype(str),frequency.tolist())),
               train_cluster_fraction_median=float(base.normalizer.median[5]),
               train_cluster_fraction_iqr=float(base.normalizer.scale[5]),
               reconstructed_q25=float(q25),reconstructed_q75=float(q75),
               component2_fraction=float((component==2).mean()),
               total_observed_count_histogram=dict(zip(n.astype(str),f.tolist())),train_only=True,scaler_refitted=False)
    report=json.loads((Path(diagnostic_dir)/"training_report.json").read_text())
    share={}
    for name,result in report["validation"].items():
        metrics=result["8-13"]["metrics"]["bilstm"]
        share[name]=metrics["per_feature"]["cross_cluster_fraction"]["huber"]/(metrics["huber"]*13)
    metrics=report["validation"]["original"]["8-13"]["metrics"]["interpolation"]
    share["interpolation"]=metrics["per_feature"]["cross_cluster_fraction"]["huber"]/(metrics["huber"]*13)
    stats["validation_short_huber_cluster_share"]=share
    write_json(Path(diagnostic_dir)/"scaling_diagnostic.json",stats)
    return stats


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument("--data",default="data/derived/relational_temporal_v1_20261109")
    p.add_argument("--cache",default="data/derived/model_a_feature_cache_v1")
    p.add_argument("--diagnostic-dir",required=True)
    a=p.parse_args();r=audit(a.data,a.cache,a.diagnostic_dir)
    print(f"Train frames={r['train_quality_frames']}; cluster IQR={r['train_cluster_fraction_iqr']:.9f}; no scaler refit")


if __name__=="__main__":main()
