"""Read-only paired A20 strata and exact-graph-input geometry controls."""
import argparse
import json
import os
os.environ.setdefault('PYTORCH_ENABLE_MPS_FALLBACK','1')
from pathlib import Path

import numpy as np
import torch

from .corruption import SpanMasker
from .diagnose_a_training import restore_variant,write_json
from .diagnose_b_same_frame import FrameStudy,role_rows,frame_scores,score
from .features import geometry_features
from .model_a_data import FeatureWindowDataset,build_feature_cache,sampled_indices,batch_arrays
from .spatial_schema import symmetric_summary
from .temporal_cli import open_temporal_dataset
from .train_model_a import set_seed,select_device
from .same_frame_graph import fit_ridge,predict_ridge


def run(diagnostic_dir,data_dir,target_cache_path):
    root=Path(diagnostic_dir);report=json.loads((root/'report.json').read_text())
    if (root/'a20_paired_comparison.json').exists():raise FileExistsError('Paired analysis already exists')
    config=report['config'];set_seed(config['seed'],config['cpu_threads']);device,_=select_device(config['device'])
    base=open_temporal_dataset(data_dir,split='train');cache=build_feature_cache(base,target_cache_path)
    study=FrameStudy(base,cache)
    val=open_temporal_dataset(data_dir,split='validation');val.masker=SpanMasker(config['seed'])
    positions=sampled_indices(val.index,25);dataset=FeatureWindowDataset(val,cache)
    model,_=restore_variant('data/derived/a_diagnostic_once_20261109/continued_short/best.pt',device)
    model.eval().requires_grad_(False)
    predictions=[];rows_all=[];changes=[];context_good=[]
    with torch.inference_mode():
        for first in range(0,len(positions),64):
            selected=positions[first:first+64]
            x,pad,_,windows=batch_arrays(dataset,selected)
            prediction,_=model(torch.as_tensor(x,device=device),torch.as_tensor(pad,device=device));prediction=prediction.cpu().numpy()
            for j,w in enumerate(windows):
                rows=np.searchsorted(base.source.frame_ids,w.routing.frame_ids);mask=w.artificial_mask
                counts=np.stack([study.record(r)['counts'] for r in rows])
                changed=bool((np.diff(counts,axis=0)!=0).any());good=bool(cache.quality_valid[rows].all())
                predictions.append(prediction[j,mask]);rows_all.extend(rows[mask])
                changes.extend([changed]*int(mask.sum()));context_good.extend([good]*int(mask.sum()))
    rows=np.asarray(rows_all);prediction=np.concatenate(predictions)
    scores=frame_scores(study,rows,prediction);values,observed,weight=study.targets(rows);scale=base.normalizer.scale[:13]
    good=cache.quality_valid[rows];changes=np.asarray(changes);context_good=np.asarray(context_good)
    for label,select in (('strict_window_count_stable',~changes),('strict_window_count_changed',changes),
                         ('strict_entire_context_quality_valid',context_good),('strict_context_has_quality_bad',~context_good)):
        scores[label]=dict(target_frame_occurrences=int((select&good).sum()),metrics=score(prediction,values,observed,weight*select[:,None],scale))
    old=json.loads(Path('data/derived/model_b_v1_20261109/report.json').read_text())['comparison']['8-13']['A20']
    assert np.isclose(scores['strict_all']['metrics']['mae_standardized'],old['mae_standardized'],atol=1e-5)
    write_json(root/'a20_paired_comparison.json',scores)
    # Exact existing node2 values, not full-precision source coordinates.
    selected=role_rows(base,'validation',config['validation_raw_stride']);prediction=[]
    for row in selected:
        g=study.graph(row);xy=g.x.astype(float)*np.asarray(g.pitch)
        v,valid=symmetric_summary(geometry_features(xy[g.team==0],xy[g.team==1]),g.pitch)
        prediction.append(base.normalizer.transform(v[:13],valid[:13]))
    write_json(root/'graph_input_geometry_control.json',dict(metrics=frame_scores(study,selected,np.stack(prediction)),
        interpretation='Target-definition algorithm on exact float32 graph-node inputs. Tautological information-availability control, not a neural model.'))
    plot(root)
    print('Frozen A20 matched strata and exact graph-input control completed',flush=True)


def plot(root):
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    root=Path(root);folder=root/'figures';folder.mkdir(exist_ok=True)
    training=json.loads((root/'report.json').read_text());paired=json.loads((root/'paired_comparison.json').read_text())['models']
    paired['A20_masked']=json.loads((root/'a20_paired_comparison.json').read_text())
    names=['A20_masked','B20_masked','B20_unmasked','frozen64','frozen96','same_frame64','count_readout','count_distance_readout']
    fig,axes=plt.subplots(1,2,figsize=(13,5))
    for ax,exclude in zip(axes,(False,True)):
        values=[paired[n]['strict_all']['metrics']['excluding_cluster']['mae_standardized'] if exclude else paired[n]['strict_all']['metrics']['mae_standardized'] for n in names]
        ax.barh(names,values);ax.invert_yaxis();ax.set(xlabel='Standardized MAE',title='Without cluster feature' if exclude else 'All 13 features')
        ax.grid(axis='x',alpha=.2)
    fig.suptitle('Same Validation target occurrences; pointwise models see target-time coordinates')
    fig.tight_layout();fig.savefig(folder/'paired_reconstruction.png',dpi=170);plt.close(fig)
    features=list(paired['B20_masked']['strict_all']['metrics']['per_feature'])
    matrix=np.array([[paired[n]['strict_all']['metrics']['per_feature'][f]['mae_standardized'] for f in features] for n in names])
    fig,ax=plt.subplots(figsize=(13,6));im=ax.imshow(np.log10(np.maximum(matrix,1e-5)),aspect='auto',cmap='magma')
    ax.set_xticks(np.arange(len(features)),features,rotation=55,ha='right');ax.set_yticks(np.arange(len(names)),names)
    fig.colorbar(im,ax=ax,label='log10 standardized MAE');ax.set_title('Paired feature reconstruction; lower is better')
    fig.tight_layout();fig.savefig(folder/'feature_mae.png',dpi=170);plt.close(fig)
    fig,ax=plt.subplots(figsize=(11,4))
    for name,model in training['models'].items():
        ax.plot([r['epoch'] for r in model['history']],[r['validation_huber'] for r in model['history']],marker='o',label=name)
    ax.set(xlabel='Diagnostic epoch',ylabel='Same-frame Validation Huber');ax.legend(fontsize=8);ax.grid(alpha=.2)
    fig.tight_layout();fig.savefig(folder/'learning_curve.png',dpi=170);plt.close(fig)
    fig,ax=plt.subplots(figsize=(11,5))
    strata=['strict_window_count_stable','strict_window_count_changed','strict_entire_context_quality_valid','strict_context_has_quality_bad']
    x=np.arange(len(strata));show=['A20_masked','B20_masked','same_frame64','count_distance_readout']
    for offset,name in enumerate(show):
        vals=[paired[name][s]['metrics']['excluding_cluster']['mae_standardized'] for s in strata]
        ax.bar(x+(offset-1.5)*.19,vals,.18,label=name)
    ax.set_xticks(x,['Count stable\nwhole window','Count changes\nwhole window','Whole context\nquality valid','Context contains\nquality bad'])
    ax.set(ylabel='MAE excluding cluster',title='Strict quality-valid targets, same paired strata');ax.legend(fontsize=8);ax.grid(axis='y',alpha=.2)
    fig.tight_layout();fig.savefig(folder/'count_quality_strata.png',dpi=170);plt.close(fig)


def readout_probes(diagnostic_dir,data_dir,target_cache_path):
    """Fixed-alpha closed-form readability checks, no neural model training.

    Ridge uses squared error and Train-only input standardization, so this is
    an information-access probe, not a matched replacement for the Huber heads.
    """
    root=Path(diagnostic_dir);config=json.loads((root/'config.json').read_text())
    if (root/'readout_probes.json').exists():raise FileExistsError('Readout probes already exist')
    base=open_temporal_dataset(data_dir,split='train');cache=build_feature_cache(base,target_cache_path)
    train=role_rows(base,'train',config['train_raw_stride']);val=role_rows(base,'validation',config['validation_raw_stride'])
    rows=np.load(root/'frozen_features/feature_rows.npy',allow_pickle=False)
    lookup={int(row):i for i,row in enumerate(rows)}
    tr=np.array([lookup[int(r)] for r in train]);va=np.array([lookup[int(r)] for r in val])
    train_good=np.asarray(cache.quality_valid[train]);val_good=np.asarray(cache.quality_valid[val])
    z=np.load(root/'frozen_features/frozen64.npy',mmap_mode='r',allow_pickle=False)
    raw=np.load(root/'frozen_features/frozen96.npy',mmap_mode='r',allow_pickle=False)
    statistics=np.load(root/'frozen_features/input_statistics.npy',mmap_mode='r',allow_pickle=False)
    variants={'ridge64':z,'ridge96':raw,'ridge64_count':np.c_[z,statistics[:,:5]],
              'ridge64_count_distance':np.c_[z,statistics]}
    target=np.asarray(cache.target_values[train]);results={}
    for name,x in variants.items():
        fit=fit_ridge(x[tr[train_good]],target[train_good],config['ridge_alpha'])
        prediction=predict_ridge(fit,x[va])
        values=np.asarray(cache.target_values[val]);observed=np.asarray(cache.target_observed[val])
        weight=np.broadcast_to(val_good[:,None],values.shape)
        results[name]=score(prediction,values,observed,weight,base.normalizer.scale[:13])
    density=np.cumsum(statistics[va,5:8].astype(float),axis=1)
    density=(density-base.normalizer.median[:3])/base.normalizer.scale[:3]
    density_mae=np.abs(density-cache.target_values[val,:3])[val_good].mean(0)
    np.save(root/'frozen_features/feature_frame_ids.npy',base.source.frame_ids[rows],allow_pickle=False)
    report=dict(models=results,ridge_alpha=config['ridge_alpha'],train_quality_frames=int(train_good.sum()),
                validation_quality_frames=int(val_good.sum()),analytic_histogram_density_mae=density_mae.tolist(),
                fit_role='train_only',test_used=False,
                objective_difference='Ridge squared loss with Train input standardization; neural heads used Huber without stats standardization',
                interpretation='Fixed post-training readability diagnostic; no alpha search, no B20/temporal update. Extra histogram overlaps density targets.')
    write_json(root/'readout_probes.json',report)
    print('Fixed readout probes:',{k:v['excluding_cluster']['mae_standardized'] for k,v in results.items()})
    return report


def main():
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--diagnostic-dir',required=True)
    p.add_argument('--data',default='data/derived/relational_temporal_v1_20261109')
    p.add_argument('--target-cache',default='data/derived/model_a_feature_cache_v1')
    p.add_argument('--readout-probes-only',action='store_true')
    a=p.parse_args()
    if a.readout_probes_only:readout_probes(a.diagnostic_dir,a.data,a.target_cache)
    else:run(a.diagnostic_dir,a.data,a.target_cache)


if __name__=='__main__':main()
