"""Local B12/B20 budgets. Short-mask Validation selects; Test remains locked."""
import argparse
import csv
import json
import os
os.environ.setdefault("PYTORCH_ENABLE_MPS_FALLBACK","1")
from pathlib import Path
import platform
import resource
import shutil
import time

import numpy as np
import torch

from .corruption import SpanMasker
from .diagnose_a_training import write_json
from .graph_windows import VisibleGraphCache,GraphWindowDataset,graph_batch_arrays,grouped_batches
from .model_a import load_checkpoint
from .model_a_data import build_feature_cache,sampled_indices,FeatureWindowDataset
from .model_b import GraphBiLSTM,save_b_checkpoint,load_b_checkpoint
from .reconstruction import masked_huber,ReconstructionMetrics
from .spatial_schema import TARGET_NAMES
from .temporal_cli import open_temporal_dataset
from .train_model_a import set_seed,select_device,evaluate as evaluate_a,_code_versions


def enrich_metrics(metrics):
    features=metrics['per_feature']
    total_huber=sum(v['huber']*v['weight_sum'] for v in features.values() if v['huber'] is not None)
    for name,v in features.items():
        v['huber_contribution_fraction']=v['huber']*v['weight_sum']/total_huber if total_huber and v['huber'] is not None else 0.
    keep=[v for name,v in features.items() if name!='cross_cluster_fraction' and v['weight_sum']>0]
    weight=sum(v['weight_sum'] for v in keep)
    metrics['excluding_cluster']=dict(mae_standardized=sum(v['mae_standardized']*v['weight_sum'] for v in keep)/weight if weight else None,
                                    huber=sum(v['huber']*v['weight_sum'] for v in keep)/weight if weight else None)
    metrics['feature_macro_mae']=float(np.mean([v['mae_standardized'] for v in features.values() if v['mae_standardized'] is not None]))
    return metrics


def evaluate_b(model,dataset,positions,batch_size,device):
    model.eval();metrics=ReconstructionMetrics();started=time.perf_counter()
    with torch.inference_mode():
        for first in range(0,len(positions),batch_size):
            batch,target,_=graph_batch_arrays(dataset,positions[first:first+batch_size])
            out=model(batch.to(device))
            metrics.update(out.prediction.cpu().numpy(),target.values,target.observed,target.quality_weight,target.artificial_mask)
    return dict(windows=len(positions),metrics=enrich_metrics(metrics.report(TARGET_NAMES,dataset.base.normalizer.scale[:13])),
                seconds=time.perf_counter()-started)


def run(data_dir,target_cache_path,out_dir,config,robust_config,*,
        a12_path='data/derived/model_a_run_v2_20261109/best.pt',
        a20_path='data/derived/a_diagnostic_once_20261109/continued_short/best.pt'):
    output=Path(out_dir)
    if output.exists():raise FileExistsError("Choose a new local B run; old artifacts are protected")
    output.mkdir(parents=True)
    write_json(output/'config.json',config);write_json(output/'robustness_config.json',robust_config)
    (output/'LOCAL_ONLY.txt').write_text('Local match graph/model/embeddings only. No upload, GT, Smart Tagging, ball, Test or push.\n')
    set_seed(config['seed'],config['cpu_threads'])
    base=open_temporal_dataset(data_dir,split='train');base.masker=SpanMasker(seed=config['seed'])
    target_cache=build_feature_cache(base,target_cache_path)
    geometry=VisibleGraphCache(base,capacity=config['graph_cache_capacity'],k=config['graph_k'],radius=config['cross_radius_m'])
    train=GraphWindowDataset(base,geometry,target_cache)
    positions=sampled_indices(base.index,config['train_stride'])
    val_base=open_temporal_dataset(data_dir,split='validation');val_base.masker=SpanMasker(seed=config['seed'])
    validation=GraphWindowDataset(val_base,geometry,target_cache)
    val_positions=sampled_indices(val_base.index,config['validation_stride'])
    device,device_info=select_device(config['device'])
    set_seed(config['seed'],config['cpu_threads'])
    model=GraphBiLSTM(graph_chunk=config['graph_chunk'],checkpoint_graph=config['checkpoint_graph']).to(device)
    optimizer=torch.optim.Adam(model.parameters(),lr=config['learning_rate'])
    import torch_geometric
    environment=dict(device=device_info,torch=str(torch.__version__),pyg=torch_geometric.__version__,numpy=np.__version__,
                     platform=platform.platform(),code=_code_versions(),backend='native dense GAT, validated against installed PyG forward/backward')
    write_json(output/'environment.json',environment)
    metadata=dict(config=config,source_fingerprint=base.source.fingerprint,split_fingerprint=base.normalizer.manifest_fingerprint)
    history=[];best=float('inf');best_epoch=0;total_updates=total_cells=0;max_mps_driver=0.
    started_all=time.perf_counter()
    for epoch in range(config['epochs']):
        if epoch==config['reset_optimizer_after_epoch']:
            optimizer=torch.optim.Adam(model.parameters(),lr=config['learning_rate'])
        model.train();train.set_epoch(epoch)
        batches=grouped_batches(base.index,positions,batch_size=config['batch_size'],group_size=config['group_size'],seed=config['seed'],epoch=epoch)
        summed=weights=0.;updates=skipped=0;epoch_cells=0;unique_graphs=0;started=time.perf_counter()
        for step,indices in enumerate(batches):
            batch,target,_=graph_batch_arrays(train,indices)
            if not target.loss_mask.any():skipped+=1;continue
            optimizer.zero_grad(set_to_none=True)
            graph_input=batch.to(device);out=model(graph_input)
            loss=masked_huber(out.prediction,torch.as_tensor(target.values,device=device),
                 torch.as_tensor(target.observed,device=device),torch.as_tensor(target.quality_weight,device=device),
                 torch.as_tensor(target.artificial_mask,device=device),config['feature_weights'],config['huber_delta'])
            if loss.loss is None:skipped+=1;continue
            loss.loss.backward();torch.nn.utils.clip_grad_norm_(model.parameters(),config['gradient_clip'],error_if_nonfinite=True)
            optimizer.step()
            summed+=float(loss.loss.detach().cpu())*loss.weight_sum;weights+=loss.weight_sum
            updates+=1;epoch_cells+=loss.valid_cells;unique_graphs+=len(batch.x)
            if str(device)=='mps':max_mps_driver=max(max_mps_driver,torch.mps.driver_allocated_memory())
            if step and step%100==0:print(f'epoch {epoch+1} step {step}/{len(batches)}',flush=True)
        val=evaluate_b(model,validation,val_positions,config['batch_size'],device)
        value=val['metrics']['huber'];total_updates+=updates;total_cells+=epoch_cells
        row=dict(epoch=epoch+1,train_huber=summed/weights if weights else None,validation_huber=value,
                 validation_mae=val['metrics']['mae_standardized'],optimizer_updates=updates,skipped_batches=skipped,
                 eligible_target_cells=epoch_cells,train_target_weight=weights,cumulative_updates=total_updates,
                 cumulative_target_cells=total_cells,encoded_unique_graphs_across_updates=unique_graphs,
                 seconds=time.perf_counter()-started)
        history.append(row)
        if value<best:
            best,best_epoch=value,epoch+1
            save_b_checkpoint(output/'best.pt',model,epoch=best_epoch,updates=total_updates,target_cells=total_cells,
                              validation_huber=best,**metadata)
        with (output/'history.csv').open('w',newline='') as f:
            writer=csv.DictWriter(f,fieldnames=list(row));writer.writeheader();writer.writerows(history)
        print(f"B epoch {epoch+1}: train={row['train_huber']}, val={value:.6f}, updates={updates}, skipped={skipped}, seconds={row['seconds']:.1f}",flush=True)
        if epoch+1 in config['budget_checkpoints']:
            save_b_checkpoint(output/f'b{epoch+1}_terminal.pt',model,epoch=epoch+1,updates=total_updates,target_cells=total_cells,
                              validation_huber=value,**metadata)
            shutil.copyfile(output/'best.pt',output/f'b{epoch+1}_best.pt')
    seconds=time.perf_counter()-started_all
    comparison={}
    bmodels={f'B{budget}':load_b_checkpoint(output/f'b{budget}_best.pt',device) for budget in config['budget_checkpoints']}
    a12,_=load_checkpoint(a12_path,device)
    from .diagnose_a_training import restore_variant
    a20,_=restore_variant(a20_path,device)
    for label,bounds in (('8-13',(8,13)),('25',(25,25)),('50',(50,50))):
        vbase=open_temporal_dataset(data_dir,split='validation');vbase.masker=SpanMasker(config['seed'],*bounds)
        graph_data=GraphWindowDataset(vbase,geometry,target_cache);feature_data=FeatureWindowDataset(vbase,target_cache)
        a=evaluate_a(a12,feature_data,val_positions,config['batch_size'],device)
        by_model={name:enrich_metrics(a['metrics'][key]) for name,key in (('A12','bilstm'),('interpolation','interpolation'))}
        a=evaluate_a(a20,feature_data,val_positions,config['batch_size'],device,baselines=False)
        by_model['A20']=enrich_metrics(a['metrics']['bilstm'])
        for name,(bm,_) in bmodels.items():by_model[name]=evaluate_b(bm,graph_data,val_positions,config['batch_size'],device)['metrics']
        assert len({m['valid_cells'] for m in by_model.values()})==1
        comparison[label]=by_model
    os_peak=resource.getrusage(resource.RUSAGE_SELF).ru_maxrss
    if platform.system()!='Darwin':os_peak*=1024
    report=dict(version='model-b-v1',parameters=sum(p.numel() for p in model.parameters()),environment=environment,
                config=config,train_windows=len(positions),validation_windows=len(val_positions),training_seconds=seconds,
                history=history,best_short_epoch=best_epoch,selected_budget_epochs={name:meta['epoch'] for name,(_,meta) in bmodels.items()},
                comparison=comparison,geometry_cache=dict(frames=len(geometry.frames),hits=geometry.hits,misses=geometry.misses),
                memory=dict(os_process_peak_rss_mb=os_peak/1e6,max_post_step_mps_driver_mb=max_mps_driver/1e6,
                            note='OS peak RSS and maximum observed post-step MPS driver allocation; not exact GPU peak'),
                test_evaluated=False,scaler_refitted=False,early_stopping=False,
                selection='short Validation only; fixed 12/20 epoch budgets; long masks evaluated after selection',
                batching_caveat='same windows/masks/targets as A; shuffled local groups change Adam minibatch composition',
                graph_fusion_clustering_possession=False)
    write_json(output/'report.json',report)
    with (output/'feature_metrics.csv').open('w',newline='') as f:
        fields=['mask','model','feature','mae_standardized','mae_schema_units','huber','valid_cells','weight_sum','huber_contribution_fraction']
        writer=csv.DictWriter(f,fieldnames=fields);writer.writeheader()
        for mask,models in comparison.items():
            for name,m in models.items():
                for feature,values in m['per_feature'].items():writer.writerow(dict(mask=mask,model=name,feature=feature,**values))
    print('B12/B20 training and Validation comparison complete',flush=True)
    return report


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--data',default='data/derived/relational_temporal_v1_20261109')
    p.add_argument('--target-cache',default='data/derived/model_a_feature_cache_v1')
    p.add_argument('--out-dir',required=True)
    p.add_argument('--config',default='MTAH_Relational_Starter/configs/model_b_small.json')
    p.add_argument('--robustness-config',default='MTAH_Relational_Starter/configs/model_b_robustness.json')
    a=p.parse_args();run(a.data,a.target_cache,a.out_dir,json.loads(Path(a.config).read_text()),json.loads(Path(a.robustness_config).read_text()))


if __name__=='__main__':main()
