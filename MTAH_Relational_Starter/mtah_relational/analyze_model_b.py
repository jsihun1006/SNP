"""Split-contained A/B embeddings, ordered H/A exports and offline reports."""
import argparse
import json
import os
os.environ.setdefault('PYTORCH_ENABLE_MPS_FALLBACK','1')
from pathlib import Path
import time

import numpy as np
import torch

from .a_diagnostic_models import covariance_diagnostics
from .diagnose_a_representation import NoArtificialMask,center_embeddings
from .diagnose_a_training import write_json,restore_variant
from .graph_windows import VisibleGraphCache,GraphWindowDataset,graph_batch_arrays
from .model_a import load_checkpoint
from .model_a_data import build_feature_cache,sampled_indices,FeatureWindowDataset,batch_arrays
from .model_b import load_b_checkpoint
from .temporal_cli import open_temporal_dataset
from .train_model_a import set_seed,select_device
from .train_model_b import evaluate_b
from .corruption import SpanMasker


def run(run_dir,data_dir,target_cache_path):
    root=Path(run_dir);output=root/'representations'
    if output.exists():raise FileExistsError('B representation output exists')
    output.mkdir()
    set_seed(2027,4);device,_=select_device('auto')
    base=open_temporal_dataset(data_dir,split='validation');base.masker=NoArtificialMask()
    cache=build_feature_cache(base,target_cache_path)
    positions=np.arange(len(base));rows=base.index.records['center_row']
    geometry=VisibleGraphCache(base);dataset=GraphWindowDataset(base,geometry,cache)
    quality=np.asarray(cache.quality_valid[rows])
    np.save(output/'center_frames.npy',base.index.records['center_frame'],allow_pickle=False)
    np.save(output/'quality_valid.npy',quality,allow_pickle=False)
    models=dict(A12=load_checkpoint('data/derived/model_a_run_v2_20261109/best.pt',device)[0],
                A20=restore_variant('data/derived/a_diagnostic_once_20261109/continued_short/best.pt',device)[0],
                B12=load_b_checkpoint(root/'b12_best.pt',device)[0],
                B20=load_b_checkpoint(root/'b20_best.pt',device)[0])
    reports={}
    for name,model in models.items():
        model.eval();started=time.perf_counter()
        if name.startswith('A'):
            embedding=center_embeddings(model,base,cache,positions,device)
        else:
            embedding=np.empty((len(base),64),dtype=np.float32)
            scene=np.empty_like(embedding);teams=np.empty((len(base),2,32),dtype=np.float32)
            team_valid=np.zeros((len(base),2),dtype=bool)
            with torch.inference_mode():
                for first in range(0,len(positions),64):
                    batch,_,_=graph_batch_arrays(dataset,positions[first:first+64],with_targets=False)
                    out=model(batch.to(device));n=len(batch.mapping)
                    embedding[first:first+n]=out.embedding[:,75].cpu().numpy()
                    scene[first:first+n]=out.graph_embedding[:,75].cpu().numpy()
                    teams[first:first+n]=out.team_embedding[:,75].cpu().numpy()
                    team_valid[first:first+n]=out.team_observed[:,75].cpu().numpy()
            np.save(output/f'{name}_graph.npy',scene,allow_pickle=False)
            np.save(output/f'{name}_teams.npy',teams,allow_pickle=False)
            np.save(output/f'{name}_team_observed.npy',team_valid,allow_pickle=False)
        total_seconds=time.perf_counter()-started
        np.save(output/f'{name}_temporal.npy',embedding,allow_pickle=False)
        cov={label:covariance_diagnostics(embedding[choose]) for label,choose in
             (('all',np.ones(len(base),dtype=bool)),('quality_valid',quality),('quality_bad',~quality))}
        # Warm, same 64 full-context consecutive centers. Inputs already packed.
        chosen=positions[:64]
        if name.startswith('B'):x=graph_batch_arrays(dataset,chosen,with_targets=False)[0].to(device)
        else:
            data=FeatureWindowDataset(base,cache);a,pad,_,_=batch_arrays(data,chosen,with_targets=False)
            x=(torch.as_tensor(a,device=device),torch.as_tensor(pad,device=device))
        timings=[]
        with torch.inference_mode():
            for _ in range(6):
                if str(device)=='mps':torch.mps.synchronize()
                start=time.perf_counter()
                if name.startswith('B'):model(x)
                else:model(*x)
                if str(device)=='mps':torch.mps.synchronize()
                timings.append(time.perf_counter()-start)
        reports[name]=dict(covariance=cov,full_validation_inference_seconds=total_seconds,
                           center_windows=len(base),milliseconds_per_center=total_seconds/len(base)*1000,
                           warm_64_window_forward_seconds=timings[1:],
                           warm_forward_seconds_mean=float(np.mean(timings[1:])),
                           timing_caveat='Shared local geometry cache warms across models; end-to-end times are not controlled hardware rankings')
        print(f"{name} representation quality rank={cov['quality_valid']['effective_rank']:.3f}; seconds={total_seconds:.2f}",flush=True)
    report=dict(models=reports,validation_windows=len(base),quality_valid_windows=int(quality.sum()),
                test_used=False,context='full151 within Validation blocks, unmasked, center index75',
                temporal_shape=[len(base),64],team_shape=[len(base),2,32])
    write_json(output/'report.json',report)
    # Terminal checkpoints expose the exact 12/20-epoch outcomes as well as best models.
    terminal={}
    for budget in (12,20):
        model,meta=load_b_checkpoint(root/f'b{budget}_terminal.pt',device)
        terminal[f'B{budget}']=dict(epoch=meta['epoch'],updates=meta['updates'],target_cells=meta['target_cells'],masks={})
        for label,bounds in (('8-13',(8,13)),('25',(25,25)),('50',(50,50))):
            base.masker=SpanMasker(2027,*bounds)
            terminal[f'B{budget}']['masks'][label]=evaluate_b(model,dataset,sampled_indices(base.index,25),64,device)['metrics']
    write_json(root/'terminal_comparison.json',terminal)
    plot(root,report)
    return report


def plot(root,rep):
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    training=json.loads((root/'report.json').read_text());figures=root/'figures';figures.mkdir(exist_ok=True)
    fig,axes=plt.subplots(1,2,figsize=(12,4));labels=('8-13','25','50')
    for name in ('interpolation','A12','A20','B12','B20'):
        for ax,metric in zip(axes,('mae_standardized','huber')):
            ax.plot(labels,[training['comparison'][m][name][metric] for m in labels],marker='o',label=name)
            ax.set(xlabel='Masked consecutive frames',ylabel=f'Validation {metric}');ax.grid(alpha=.2)
    for ax in axes:ax.legend()
    fig.tight_layout();fig.savefig(figures/'reconstruction_comparison.png',dpi=170);plt.close(fig)
    fig,axes=plt.subplots(1,2,figsize=(12,4))
    for name,report in rep['models'].items():
        q=report['covariance']['quality_valid']
        axes[0].plot(np.arange(1,65),q['cumulative_explained_variance'],label=f"{name}, rank {q['effective_rank']:.2f}")
        axes[1].plot(q['dimension_variance'],label=name)
    axes[0].set(xlabel='Principal components',ylabel='Cumulative explained variance')
    axes[1].set(xlabel='Temporal dimension',ylabel='Variance',yscale='log')
    for ax in axes:ax.legend();ax.grid(alpha=.2)
    fig.tight_layout();fig.savefig(figures/'embedding_variance.png',dpi=170);plt.close(fig)
    fig,ax=plt.subplots(figsize=(8,4))
    ax.plot([x['epoch'] for x in training['history']],[x['train_huber'] for x in training['history']],label='Train (changing masks)')
    ax.plot([x['epoch'] for x in training['history']],[x['validation_huber'] for x in training['history']],label='Validation fixed short mask')
    ax.axvline(12,color='gray',linestyle='--',label='Adam reset after 12 epochs')
    ax.set(xlabel='Epoch',ylabel='Weighted standardized Huber');ax.legend();ax.grid(alpha=.2)
    fig.tight_layout();fig.savefig(figures/'learning_curve.png',dpi=170);plt.close(fig)


def main():
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--run-dir',required=True)
    p.add_argument('--data',default='data/derived/relational_temporal_v1_20261109')
    p.add_argument('--target-cache',default='data/derived/model_a_feature_cache_v1')
    a=p.parse_args();run(a.run_dir,a.data,a.target_cache)


if __name__=='__main__':main()
