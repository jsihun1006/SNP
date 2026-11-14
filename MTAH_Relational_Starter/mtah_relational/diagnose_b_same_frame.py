"""One bounded same-frame diagnosis; original B20 and Test remain frozen."""
import argparse
import copy
import csv
import hashlib
import json
import os
os.environ.setdefault('PYTORCH_ENABLE_MPS_FALLBACK','1')
from pathlib import Path
import time

import numpy as np
import torch

from .corruption import SpanMasker
from .diagnose_a_representation import NoArtificialMask
from .diagnose_a_training import write_json
from .dynamic_graph import build_graph
from .features import geometry_features
from .graph_windows import VisibleGraphCache,GraphWindowDataset,graph_batch_arrays
from .model_a_data import build_feature_cache,sampled_indices
from .model_b import load_b_checkpoint
from .reconstruction import ReconstructionMetrics
from .same_frame_graph import (SameFrameReconstructor,graph_statistics,node_pool96,pack_graphs,same_frame_huber,
                               fit_ridge,predict_ridge,regression_report,STAT_NAMES)
from .spatial_schema import TARGET_NAMES,symmetric_summary
from .temporal_cli import open_temporal_dataset
from .train_model_a import set_seed,select_device,EarlyStopping,_code_versions
from .train_model_b import enrich_metrics


def role_rows(base,role,stride=1):
    if role not in ('train','validation'):raise ValueError('Test is locked')
    selected=[]
    for block in base.index.blocks:
        if block.split==role:
            rows=np.flatnonzero((base.source.frame_ids>=block.first)&(base.source.frame_ids<=block.last))
            assert (base.source.periods[rows]==block.period).all()
            selected.append(rows[::stride])
    return np.concatenate(selected)


class FrameStudy:
    def __init__(self,base,target_cache):
        self.base,self.cache=base,target_cache
        self.geometry=VisibleGraphCache(base)
        self.roles=np.full(len(base.source),-1,np.int8)
        for i,role in enumerate(('train','validation')):self.roles[role_rows(base,role)]=i
        self.record_cache={};self.feature_lookup={}

    def graph(self,row):
        if self.roles[int(row)]<0:raise ValueError('Out-of-scope raw frame (Test/period0)')
        return self.geometry.get_visible(int(row))

    def record(self,row):
        row=int(row)
        if row in self.record_cache:return self.record_cache[row]
        g=self.graph(row);counts=np.array([(g.team==i).sum() for i in (0,1)])
        valid_prev=row>0 and self.roles[row-1]==self.roles[row] \
            and self.base.source.frame_ids[row]-self.base.source.frame_ids[row-1]==1 \
            and self.base.source.periods[row]==self.base.source.periods[row-1]
        changed=-1;previous_quality=False
        if valid_prev:
            prev=self.graph(row-1);old=np.array([(prev.team==i).sum() for i in (0,1)])
            changed=int((old!=counts).any());previous_quality=bool(self.cache.quality_valid[row-1])
        xy=g.x.astype(float)*np.asarray(g.pitch)
        ordered=np.full((2,2),np.nan)
        for side in (0,1):
            p=xy[g.team==side]
            if len(p)>=2:
                extent=np.percentile(p,90,axis=0)-np.percentile(p,10,axis=0)
                ordered[side]=extent[[1,0]]/np.asarray(g.pitch)[[1,0]]
        result=dict(counts=counts,changed=changed,previous_quality=previous_quality,
                    statistics=graph_statistics(g),ordered_width_length=ordered)
        self.record_cache[row]=result
        return result

    def targets(self,rows,quality_mode='strict'):
        rows=np.asarray(rows)
        observed=np.asarray(self.cache.target_observed[rows]).copy()
        q=np.asarray(self.cache.quality_valid[rows])
        select=q if quality_mode=='strict' else ~q
        weight=np.broadcast_to(select[:,None],observed.shape).astype(np.float32).copy()
        weight*=self.base.normalizer.counts[:13]>0
        return np.asarray(self.cache.target_values[rows]),observed,weight


def extract_frozen(study,encoder,rows,device,output):
    encoder.eval();encoder.requires_grad_(False)
    z=np.empty((len(rows),64),np.float32);raw=np.empty((len(rows),96),np.float32)
    teams=np.empty((len(rows),2,32),np.float32)
    stats=np.empty((len(rows),15),np.float32)
    with torch.inference_mode():
        for first in range(0,len(rows),512):
            selected=rows[first:first+512];graphs=[study.graph(row) for row in selected]
            args=pack_graphs(graphs,device);pool,team=node_pool96(encoder,*args)
            latent=encoder.readout(pool);n=len(selected)
            z[first:first+n]=latent.cpu().numpy();raw[first:first+n]=pool.cpu().numpy();teams[first:first+n]=team.cpu().numpy()
            stats[first:first+n]=np.stack([study.record(row)['statistics'] for row in selected])
    for name,array in (('frozen64',z),('frozen96',raw),('frozen_teams',teams),('input_statistics',stats)):
        np.save(output/f'{name}.npy',array,allow_pickle=False)
    study.feature_lookup={int(row):i for i,row in enumerate(rows)}
    study.frozen={'frozen64':z,'frozen96':raw};study.frozen_teams=teams;study.stats=stats
    np.save(output/'feature_rows.npy',rows,allow_pickle=False)


def predict(model,study,rows,device):
    locations=np.asarray([study.feature_lookup[int(r)] for r in rows])
    if model.mode.startswith('frozen'):
        return model(frozen_features=torch.as_tensor(study.frozen[model.mode][locations],device=device))
    args=pack_graphs([study.graph(row) for row in rows],device)
    statistics=torch.as_tensor(study.stats[locations],device=device) if model.extra else None
    return model(*args,statistics=statistics)


def evaluate_frame(model,study,rows,device,batch_size=256):
    model.eval();predictions=[]
    with torch.inference_mode():
        for first in range(0,len(rows),batch_size):predictions.append(predict(model,study,rows[first:first+batch_size],device).cpu().numpy())
    prediction=np.concatenate(predictions)
    report=frame_scores(study,rows,prediction)
    return report,prediction


def score(prediction,values,observed,weight,scale):
    m=ReconstructionMetrics();m.update(prediction,values,observed,weight,np.ones(len(prediction),bool))
    report=m.report(TARGET_NAMES,scale)
    if not report['valid_cells']:
        for v in report['per_feature'].values():v['huber_contribution_fraction']=0.
        report['excluding_cluster']=dict(mae_standardized=None,huber=None);report['feature_macro_mae']=None
        return report
    return enrich_metrics(report)


def frame_scores(study,rows,prediction):
    values,observed,weight=study.targets(rows)
    good=np.asarray(study.cache.quality_valid[rows]);records=[study.record(r) for r in rows]
    changed=np.asarray([r['changed'] for r in records]);previous=np.asarray([r['previous_quality'] for r in records])
    scale=study.base.normalizer.scale[:13]
    result={}
    for name,select in (('strict_all',good),('strict_count_stable',good&previous&(changed==0)),
                        ('strict_count_changed',good&previous&(changed==1))):
        result[name]=dict(frames=int(select.sum()),metrics=score(prediction,values,observed,weight*select[:,None],scale))
    v,o,w=study.targets(rows,'low_quality')
    result['low_quality_math_valid']=dict(frames=int((~good).sum()),metrics=score(prediction,v,o,w,scale),
        caveat='Untrusted geometry targets; excluded from training/selection, not tracking accuracy')
    return result


def save_frame_model(path,model,**metadata):
    torch.save(dict(version='b-same-frame-v1',mode=model.mode,metadata=metadata,
        state={k:v.detach().cpu().clone() for k,v in model.state_dict().items()}),path)


def load_frame_model(path,reference_encoder,device='cpu'):
    state=torch.load(path,map_location='cpu',weights_only=True)
    if state.get('version')!='b-same-frame-v1':raise ValueError('Invalid same-frame checkpoint')
    model=SameFrameReconstructor(reference_encoder,state['mode'])
    model.load_state_dict(state['state'],strict=True)
    return model.to(device),state['metadata']


def collision_audit(encoder,device):
    xy=np.array([[10.,10.],[18.,10.],[10.,12.],[18.,12.]])
    team=np.array([0,0,1,1])
    graphs=[build_graph(xy,team),build_graph(np.repeat(xy,2,axis=0),np.repeat(team,2))]
    with torch.inference_mode():
        raw,_=node_pool96(encoder,*pack_graphs(graphs,device));latent=encoder.readout(raw)
    targets=[]
    for x,t in ((xy,team),(np.repeat(xy,2,axis=0),np.repeat(team,2))):
        v,_=symmetric_summary(geometry_features(x[t==0],x[t==1]),(105.3,68.));targets.append(v[:13])
    return dict(node_counts=[len(g.x) for g in graphs],directed_edge_counts=[int(g.adjacency.sum()) for g in graphs],
                max_pool96_difference=float((raw[0]-raw[1]).abs().max().cpu()),
                max_readout64_difference=float((latent[0]-latent[1]).abs().max().cpu()),
                length_sum_targets=[float(t[9]) for t in targets],
                count_statistics=[graph_statistics(g)[:5].tolist() for g in graphs],
                caveat='Synthetic coincident-duplicate counterexample outside strict quality policy; not a claim of real valid-frame collisions')


def auxiliary_probes(study,train_rows,val_rows,alpha):
    tr=np.asarray([study.feature_lookup[int(r)] for r in train_rows]);va=np.asarray([study.feature_lookup[int(r)] for r in val_rows])
    tr=tr[study.cache.quality_valid[train_rows]];va=va[study.cache.quality_valid[val_rows]]
    result={}
    for kind in ('frozen64','frozen96'):
        fit=fit_ridge(study.frozen[kind][tr],study.stats[tr],alpha)
        result[kind]=regression_report(predict_ridge(fit,study.frozen[kind][va]),study.stats[va],STAT_NAMES)
    # A shared decoder of each ordered team32 preserves team-swap equivariance.
    tr_target=np.stack([study.record(r)['ordered_width_length'] for r in train_rows if study.cache.quality_valid[r]])
    va_target=np.stack([study.record(r)['ordered_width_length'] for r in val_rows if study.cache.quality_valid[r]])
    fit=fit_ridge(study.frozen_teams[tr].reshape(-1,32),tr_target.reshape(-1,2),alpha)
    pred=predict_ridge(fit,study.frozen_teams[va].reshape(-1,32)).reshape(-1,2,2)
    scale=np.array([68.,105.3])
    result['ordered_team32']={side:regression_report(pred[:,i]*scale,va_target[:,i]*scale,('width_p90_p10_m','length_p90_p10_m')) for i,side in enumerate(('home','away'))}
    return dict(results=result,fit_role='train_quality_valid_only',alpha=alpha,train_frames=len(tr),validation_frames=len(va),
                caveat='Linear recoverability, not proof of absence of nonlinear information; histogram inputs overlap density targets')


def deterministic_geometry_control(study,rows):
    from .data_quality import assess_observations
    predictions=[]
    for row in rows:
        frame=study.base.source.frame(int(row))
        xy,_=assess_observations(frame,study.base.source.pitch,study.base.policy)
        good=np.isfinite(xy).all(1);xy,team=xy[good],frame.team_codes[good]
        values,valid=symmetric_summary(geometry_features(xy[team==0],xy[team==1]),study.base.source.pitch)
        predictions.append(study.base.normalizer.transform(values[:13],valid[:13]))
    return dict(metrics=frame_scores(study,rows,np.stack(predictions)),
                interpretation='Deterministic target-definition identity control; pipeline check, not learned reconstruction or an independent baseline')


def paired_rows(val_base,positions):
    selected=[]
    for i in positions:
        routing,rows=val_base._routing(int(i));mask=val_base._mask(routing)
        selected.extend(rows[mask])
    return np.asarray(selected,dtype=np.int64)


def paired_temporal_audit(reference,models,study,val_base,positions,device):
    source=study.base.source;frames=paired_rows(val_base,positions);scale=study.base.normalizer.scale[:13]
    direct={name:[] for name in models};masked=[];unmasked=[];ordered_rows=[];window_changes=[];window_good=[]
    geometry=VisibleGraphCache(val_base);data=GraphWindowDataset(val_base,geometry,study.cache)
    for first in range(0,len(positions),32):
        chosen=positions[first:first+32]
        val_base.masker=SpanMasker(2027)
        batch,target,windows=graph_batch_arrays(data,chosen)
        with torch.inference_mode():a=reference(batch.to(device)).prediction.cpu().numpy()
        val_base.masker=NoArtificialMask()
        full,_,_=graph_batch_arrays(data,chosen,with_targets=False)
        with torch.inference_mode():b=reference(full.to(device)).prediction.cpu().numpy()
        for j,w in enumerate(windows):
            mask=w.artificial_mask;rows=np.searchsorted(source.frame_ids,w.routing.frame_ids)
            observed_counts=np.stack([study.record(r)['counts'] for r in rows])
            changed=bool((np.diff(observed_counts,axis=0)!=0).any())
            quality=bool(study.cache.quality_valid[rows].all())
            masked.append(a[j,mask]);unmasked.append(b[j,mask]);ordered_rows.extend(rows[mask])
            window_changes.extend([changed]*int(mask.sum()));window_good.extend([quality]*int(mask.sum()))
    rows=np.asarray(ordered_rows);assert np.array_equal(rows,frames)
    predictions={'B20_masked':np.concatenate(masked),'B20_unmasked':np.concatenate(unmasked)}
    for name,model in models.items():
        with torch.inference_mode():
            predictions[name]=np.concatenate([predict(model,study,rows[first:first+256],device).cpu().numpy() for first in range(0,len(rows),256)])
    values,observed,weight=study.targets(rows);good=study.cache.quality_valid[rows]
    changes=np.asarray(window_changes);window_good=np.asarray(window_good)
    result={}
    for name,prediction in predictions.items():
        result[name]=frame_scores(study,rows,prediction)
        for label,select in (('strict_window_count_stable',~changes),('strict_window_count_changed',changes),
                             ('strict_entire_context_quality_valid',window_good),('strict_context_has_quality_bad',~window_good)):
            result[name][label]=dict(target_frame_occurrences=int((select&good).sum()),
                metrics=score(prediction,values,observed,weight*select[:,None],scale))
    return dict(models=result,window_count=len(positions),target_frame_occurrences=len(rows),
                distinct_target_frames=len(np.unique(rows)),
                caveat='Same target occurrences, different information: pointwise models see target-time coordinates; masked B20 cannot. Unmasked B20 visible predictions were not its training objective.')


def run(data_dir,target_cache_path,b20_path,out_dir,config):
    output=Path(out_dir)
    if output.exists():raise FileExistsError('Choose a new diagnostic output')
    output.mkdir(parents=True);write_json(output/'config.json',config)
    (output/'LOCAL_ONLY.txt').write_text('Local same-frame B diagnosis. Original B20 frozen. No Test, GT, Smart Tagging, ball, Fusion, decoder or push.\n')
    set_seed(config['seed'],config['cpu_threads']);device,info=select_device(config['device'])
    base=open_temporal_dataset(data_dir,split='train');val_base=open_temporal_dataset(data_dir,split='validation')
    val_base.masker=SpanMasker(config['seed']);positions=sampled_indices(val_base.index,25)
    cache=build_feature_cache(base,target_cache_path);study=FrameStudy(base,cache)
    reference,metadata=load_b_checkpoint(b20_path,device);reference.eval();reference.requires_grad_(False)
    original_hash=hashlib.sha256(Path(b20_path).read_bytes()).hexdigest()
    initial_reference={k:v.detach().cpu().clone() for k,v in reference.state_dict().items()}
    train_rows=role_rows(base,'train',config['train_raw_stride']);val_rows=role_rows(base,'validation',config['validation_raw_stride'])
    paired=paired_rows(val_base,positions)
    all_rows=np.unique(np.r_[train_rows,val_rows,paired])
    frozen_dir=output/'frozen_features';frozen_dir.mkdir()
    extract_frozen(study,reference.graph,all_rows,device,frozen_dir)
    write_json(output/'collision.json',collision_audit(reference.graph,device))
    write_json(output/'auxiliary_probes.json',auxiliary_probes(study,train_rows,val_rows,config['ridge_alpha']))
    write_json(output/'deterministic_control.json',deterministic_geometry_control(study,val_rows))
    write_json(output/'environment.json',dict(device=info,torch=str(torch.__version__),numpy=np.__version__,code=_code_versions(),
        original_b20_sha256=original_hash,original_selected_epoch=metadata['epoch']))
    reports={};models={};started_all=time.perf_counter()
    for mode in config['variants']:
        set_seed(config['seed'],config['cpu_threads'])
        model=SameFrameReconstructor(reference.graph.cpu(),mode,config['seed']).to(device)
        reference.to(device)
        optimizer=torch.optim.Adam((p for p in model.parameters() if p.requires_grad),lr=config['learning_rate'])
        stopper=EarlyStopping(config['patience'],config['min_delta']);history=[];best_epoch=0;updates=cells=0
        folder=output/mode;folder.mkdir();started=time.perf_counter()
        for epoch in range(config['epochs']):
            model.train();order=np.random.default_rng(config['seed']+epoch).permutation(train_rows)
            total=weight_sum=0.;epoch_updates=epoch_cells=0
            for first in range(0,len(order),config['batch_size']):
                rows=order[first:first+config['batch_size']];values,observed,weight=study.targets(rows)
                if not (observed&(weight>0)).any():continue
                optimizer.zero_grad(set_to_none=True)
                prediction=predict(model,study,rows,device)
                loss=same_frame_huber(prediction,torch.as_tensor(values,device=device),torch.as_tensor(observed,device=device),torch.as_tensor(weight,device=device))
                if loss.loss is None:continue
                loss.loss.backward();torch.nn.utils.clip_grad_norm_((p for p in model.parameters() if p.requires_grad),config['gradient_clip'],error_if_nonfinite=True)
                optimizer.step();total+=float(loss.loss.detach().cpu())*loss.weight_sum;weight_sum+=loss.weight_sum
                epoch_updates+=1;epoch_cells+=loss.valid_cells
            validation,_=evaluate_frame(model,study,val_rows,device)
            value=validation['strict_all']['metrics']['huber'];improved,stop=stopper.step(value)
            updates+=epoch_updates;cells+=epoch_cells
            row=dict(epoch=epoch+1,train_huber=total/weight_sum if weight_sum else None,validation_huber=value,
                     validation_mae=validation['strict_all']['metrics']['mae_standardized'],updates=epoch_updates,target_cells=epoch_cells)
            history.append(row)
            if improved:
                best_epoch=epoch+1;save_frame_model(folder/'best.pt',model,epoch=best_epoch,config=config)
            with (folder/'history.csv').open('w',newline='') as f:
                w=csv.DictWriter(f,fieldnames=list(row));w.writeheader();w.writerows(history)
            print(f"{mode} {epoch+1}/{config['epochs']}: val Huber={value:.6f}, MAE={row['validation_mae']:.6f}",flush=True)
            if stop:break
        model,_=load_frame_model(folder/'best.pt',reference.graph.cpu(),device);reference.to(device);model.eval()
        validation,pred=evaluate_frame(model,study,val_rows,device)
        train_evaluation,_=evaluate_frame(model,study,train_rows,device)
        report=dict(best_epoch=best_epoch,epochs_completed=len(history),optimizer_updates=updates,target_cells=cells,
                    seconds=time.perf_counter()-started,train=train_evaluation,validation=validation,
                    trainable_parameters=sum(p.numel() for p in model.parameters() if p.requires_grad),
                    total_parameters=sum(p.numel() for p in model.parameters()),history=history)
        reports[mode]=report;models[mode]=model
        write_json(folder/'report.json',report);np.save(folder/'validation_predictions.npy',pred,allow_pickle=False)
    reference.to(device)
    comparison=paired_temporal_audit(reference,models,study,val_base,positions,device)
    write_json(output/'paired_comparison.json',comparison)
    assert hashlib.sha256(Path(b20_path).read_bytes()).hexdigest()==original_hash
    assert all(torch.equal(v.detach().cpu(),initial_reference[k]) for k,v in reference.state_dict().items())
    assert all(not p.requires_grad and p.grad is None for p in reference.parameters())
    report=dict(version='b-same-frame-diagnostic-v1',config=config,device=info,train_raw_frames=len(train_rows),
                validation_raw_frames=len(val_rows),train_quality_frames=int(cache.quality_valid[train_rows].sum()),
                validation_quality_frames=int(cache.quality_valid[val_rows].sum()),models=reports,
                diagnostic_training_seconds=time.perf_counter()-started_all,
                original_b20_frozen=True,original_sha256=original_hash,reference_selected_epoch=metadata['epoch'],
                test_used=False,scaler_refitted=False,temporal_training=False,artificial_input_masking=False)
    write_json(output/'report.json',report)
    with (output/'feature_metrics.csv').open('w',newline='') as f:
        fields=['model','stratum','feature','mae_standardized','mae_schema_units','huber','valid_cells','weight_sum','huber_contribution_fraction']
        w=csv.DictWriter(f,fieldnames=fields);w.writeheader()
        for mode,r in reports.items():
            for label,s in r['validation'].items():
                for name,values in s['metrics']['per_feature'].items():w.writerow(dict(model=mode,stratum=label,feature=name,**values))
    print('Frozen-B20 same-frame diagnostic complete',flush=True)
    return report


def main():
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--out-dir',required=True)
    p.add_argument('--data',default='data/derived/relational_temporal_v1_20261109')
    p.add_argument('--target-cache',default='data/derived/model_a_feature_cache_v1')
    p.add_argument('--b20',default='data/derived/model_b_v1_20261109/b20_best.pt')
    p.add_argument('--config',default='MTAH_Relational_Starter/configs/b_same_frame_diagnostic.json')
    a=p.parse_args();run(a.data,a.target_cache,a.b20,a.out_dir,json.loads(Path(a.config).read_text()))


if __name__=='__main__':main()
