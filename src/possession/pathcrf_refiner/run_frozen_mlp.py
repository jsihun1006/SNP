"""Fast frozen-encoder residual MLP with candidate-entry/exit supervision."""
from __future__ import annotations
import argparse,json,math,csv
from pathlib import Path
from time import perf_counter
import numpy as np
import pandas as pd
import torch
from torch import nn
from torch.nn import functional as F
from scipy.special import log_softmax

from inputs import ROOT,N_FEATURES,load_match,make_features
from refiner import Refiner
from run import collate,inference_windows,digest,write_json,save_checkpoint,sync
from run_candidate25 import candidate_masks
from correct_5fps import native_features,report
from correct_5fps_v2 import error_summary
from correct_candidate25 import base_logs,decode_setting,snapped
from transportmer_ha_data import contiguous_runs


class ResidualMLP(nn.Module):
    def __init__(self,width):
        super().__init__()
        self.net=nn.Sequential(nn.LayerNorm(width),nn.Linear(width,64),nn.GELU(),nn.Linear(64,6))
        nn.init.zeros_(self.net[-1].weight);nn.init.zeros_(self.net[-1].bias)
    def forward(self,x,raw,state_allowed,edge_allowed):
        out=self.net(x)
        gate=out[:,5].sigmoid()
        delta=2*out[:,:5].tanh()*gate[:,None]
        delta=torch.cat((delta[:,:2]*state_allowed[:,None],delta[:,2:]*edge_allowed[:,None]),1)
        return raw+delta,out[:,5]


def cache_match(args,config,match,encoder,device,training):
    started=perf_counter()
    halves=load_match(args,match,torch.device('cpu'))
    masks=candidate_masks(halves,config)
    folder=args.feature_cache/match;folder.mkdir(parents=True,exist_ok=True)
    identity={'checkpoint':digest(args.run/'best.pt'),'training_augmentation':training,
              'feature_code':digest(Path(__file__)),'config':config,'encoder_device':str(device)}
    record=folder/'identity.json'
    if record.exists() and json.loads(record.read_text())['identity']==identity:
        stored=[]
        for h,mask in zip(halves,masks,strict=True):
            with np.load(folder/f'{h.key}.npz') as z:
                np.testing.assert_array_equal(z['frame'],h.frame);np.testing.assert_array_equal(z['mask'],mask)
                stored.append((z['scene'].copy(),z['raw'].copy()))
        print(f'{match}: frozen feature cache reused',flush=True)
        return halves,masks,stored
    for h in halves:make_features(h,config,training=training)
    refs,_=inference_windows(halves,masks,config)
    sums=[np.zeros((len(h.frame),128),np.float32) for h in halves]
    raw_sums=[np.zeros((len(h.frame),5),np.float32) for h in halves]
    count=[np.zeros(len(h.frame),np.float32) for h in halves]
    capture={}
    handle=encoder.scene.register_forward_hook(lambda m,x,y:capture.update(scene=y.detach()))
    encoder.eval()
    with torch.inference_mode():
        for first in range(0,len(refs),args.encoder_batch):
            batch=refs[first:first+args.encoder_batch]
            x,seen,_,_=collate(halves,batch,config)
            state,change=encoder(x.to(device),seen.to(device))
            output=torch.cat((state,change),-1).float().cpu().numpy()
            scene=capture.pop('scene').float().cpu().numpy()
            for ref,e,r in zip(batch,scene,output,strict=True):
                n=ref.stop-ref.start;w=np.minimum(np.arange(1,n+1),np.arange(n,0,-1))
                sums[ref.half][ref.start:ref.stop]+=e[:n]*w[:,None]
                raw_sums[ref.half][ref.start:ref.stop]+=r[:n]*w[:,None]
                count[ref.half][ref.start:ref.stop]+=w
    handle.remove();stored=[]
    reference=args.logit_cache/f'{match}.npz'
    ref_cache=np.load(reference) if not training and reference.exists() else None
    for i,h in enumerate(halves):
        scene=np.divide(sums[i],count[i][:,None],out=np.zeros_like(sums[i]),where=count[i][:,None]>0).astype(np.float16)
        raw=np.divide(raw_sums[i],count[i][:,None],out=np.full_like(raw_sums[i],np.nan),where=count[i][:,None]>0)
        if ref_cache is not None:
            np.testing.assert_array_equal(ref_cache[f'{h.key}_frame'],h.frame)
            # Retain the exactly reproduced FP32 base scores; zero residual preserves the prior result.
            raw=ref_cache[f'{h.key}_logits'].copy()
        if not np.isfinite(raw[masks[i]]).all():raise AssertionError('Missing frozen scores in candidate.')
        np.savez_compressed(folder/f'{h.key}.npz',frame=h.frame,mask=masks[i],scene=scene,raw=raw)
        stored.append((scene,raw));h.features=None
    if ref_cache is not None:ref_cache.close()
    elapsed=perf_counter()-started
    write_json(record,{'identity':identity,'windows':len(refs),'seconds':elapsed})
    print(f'{match}: cached {len(refs)} frozen windows in {elapsed:.2f}s on {device}',flush=True)
    return halves,masks,stored


def frame_features(h,mask,scene,raw,post):
    prev=np.zeros_like(scene)
    preceding=np.zeros(len(mask),bool)
    for a,b in contiguous_runs(h.coarse['base']!=2,h.frame):
        prev[a]=scene[a];prev[a+1:b]=scene[a:b-1]
        preceding[a+1:b]=mask[a:b-1]
    eligible=mask|preceding
    blend=np.zeros(len(mask),np.float32)
    for a,b in contiguous_runs(mask,h.frame):
        n=b-a
        blend[a:b]=post['blend']*np.minimum(1,np.minimum(np.arange(1,n+1),np.arange(n,0,-1))/(post['edge_frames']+1))
    original5=base_logs(h,2.0).astype(np.float32)
    values=np.c_[scene.astype(np.float32),prev.astype(np.float32),original5,np.nan_to_num(raw),
                 h.signals[:,0],mask,preceding].astype(np.float32)
    rows=np.flatnonzero(eligible&np.isfinite(raw).all(axis=1))
    y=np.full(len(mask),-1,np.int64)
    valid=mask&np.isin(h.labels,(0,1));y[valid]=h.labels[valid]
    gate_y=np.full(len(mask),-1,np.float32)
    gate_y[valid]=(h.coarse['base'][valid]!=h.labels[valid]).astype(np.float32)
    change_y=np.full(len(mask),-1,np.int64)
    boundary=np.zeros(len(mask),np.float32)
    for a,b in contiguous_runs(h.coarse['base']!=2,h.frame):
        for t in range(a+1,b):
            if not (mask[t] or mask[t-1]):continue
            left=int(h.labels[t-1]) if mask[t-1] else int(h.coarse['base'][t-1])
            right=int(h.labels[t]) if mask[t] else int(h.coarse['base'][t])
            if left in (0,1) and right in (0,1):
                change_y[t]=0 if left==right else 2-right
                boundary[t]=float(mask[t]!=mask[t-1])
    return {'x':values[rows],'raw':raw[rows].astype(np.float32),'y':y[rows],'change':change_y[rows],
            'gate_y':gate_y[rows],'blend':blend[rows],'base_log':original5[rows],
            'state_allowed':mask[rows].astype(np.float32),'edge_allowed':eligible[rows].astype(np.float32),
            'preserve':(valid&(h.coarse['base']==h.labels))[rows].astype(np.float32),
            'boundary':boundary[rows],'rows':rows}


def tensors(data,device):return {k:torch.from_numpy(v).to(device) for k,v in data.items() if k!='rows'}


def objective(head,b,weights):
    raw,gate=head(b['x'],b['raw'],b['state_allowed'],b['edge_allowed'])
    specialist=F.log_softmax(raw[:,:2]/.5,dim=1)
    mixed=(1-b['blend'][:,None])*b['base_log']+b['blend'][:,None]*specialist
    known=b['y']>=0
    state=F.cross_entropy(mixed[known],b['y'][known]) if known.any() else mixed.sum()*0
    valid=b['change']>=0
    if valid.any():
        each=F.cross_entropy(raw[valid,2:]/.5,b['change'][valid],reduction='none')
        w=weights[b['change'][valid]]*(1+b['boundary'][valid])
        transition=(each*w).sum()/w.sum().clamp_min(1)
    else:transition=raw.sum()*0
    preserve=b['preserve']>0
    if preserve.any():
        p=b['base_log'][preserve].exp()
        kl=F.kl_div(F.log_softmax(mixed[preserve],dim=1),p,reduction='batchmean')
    else:kl=state*0
    gate_valid=b['gate_y']>=0
    gate_loss=F.binary_cross_entropy_with_logits(gate[gate_valid],b['gate_y'][gate_valid]) if gate_valid.any() else state*0
    # The original CRF/Viterbi decoder is retained. MLP fitting avoids recurrent CRF backprop for speed.
    return state+transition+.1*kl+.05*gate_loss


@torch.inference_mode()
def evaluate(head,bundles,post,calibration,device,batch):
    head.eval();halves=[];masks=[];logits=[];info={}
    for h,mask,cache,data in bundles:
        values=cache[1].copy()
        for first in range(0,len(data['rows']),batch):
            last=first+batch
            b=tensors({k:v[first:last] for k,v in data.items() if k!='rows'},device)
            output,_=head(b['x'],b['raw'],b['state_allowed'],b['edge_allowed'])
            values[data['rows'][first:last]]=output.float().cpu().numpy()
        halves.append(h);masks.append(mask);logits.append(values)
        info[h.key]=native_features(h)
    preds=decode_setting(halves,{'original':masks},logits,calibration,post)
    preds=snapped(halves,preds,info,masks,post['snap_radius_frames'])
    for h,mask,p in zip(halves,masks,preds,strict=True):
        np.testing.assert_array_equal(p[~mask],h.coarse['base'][~mask])
        np.testing.assert_array_equal(p==2,h.coarse['base']==2)
    return report(halves,preds),preds,logits


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--run',type=Path,default=ROOT/'artifacts/possession/pathcrf/candidate25_dfl/historical_run')
    parser.add_argument('--output',type=Path,default=ROOT/'artifacts/possession/pathcrf/candidate25_dfl/historical_run/frozen_mlp_boundary_v1')
    parser.add_argument('--feature-cache',type=Path,default=ROOT/'data/transition_final/frozen_mlp_features_v1')
    parser.add_argument('--logit-cache',type=Path,default=ROOT/'data/transition_final/candidate25_epoch29_logits')
    parser.add_argument('--device',choices=('mps','cpu'),default='mps')
    parser.add_argument('--encoder-batch',type=int,default=32)
    parser.add_argument('--frame-batch',type=int,default=8192)
    parser.add_argument('--epochs',type=int,default=40)
    parser.add_argument('--patience',type=int,default=5)
    parser.add_argument('--upstream',type=Path,default=Path('/private/tmp/pathcrf-official-dfl'))
    parser.add_argument('--cache',type=Path,default=ROOT/'data/external/dfl')
    parser.add_argument('--raw',type=Path,default=ROOT/'data/external/dfl/raw')
    parser.add_argument('--sources',type=Path,default=ROOT/'data/transition_final/source')
    parser.add_argument('--official-prepared',type=Path,default=ROOT/'data/pathcrf_dfl/transition_head_v1/prepared')
    parser.add_argument('--dead-scene',type=Path,default=ROOT/'data/pathcrf_dfl/scene_two_models_v1')
    parser.add_argument('--frozen-cache',type=Path,default=ROOT/'data/transition_final/frozen_7a9ebd31eba97ba5')
    args=parser.parse_args();torch.set_num_threads(4);torch.manual_seed(100)
    if args.device=='mps' and not torch.backends.mps.is_available():raise RuntimeError('MPS GPU unavailable in this process')
    device=torch.device(args.device);args.output.mkdir(parents=True,exist_ok=True)
    started=perf_counter()
    source=json.loads((args.run/'manifest.json').read_text());config=source['parameters']
    selected=json.loads((args.run/'correction_epoch29/selected.json').read_text())
    post,calibration=selected['parameters'],selected['calibration']
    sha=digest(args.run/'best.pt')
    manifest={'experiment':'frozen epoch29 encoder + boundary-aware residual MLP','split':source['split'],
              'checkpoint_sha256':sha,'seed':100,'encoder_device':str(device),'encoder_batch':args.encoder_batch,
              'frame_batch':args.frame_batch,'epochs':args.epochs,'early_stop':args.patience,
              'hidden_width':64,'residual_bound':2.,'zero_initialized_residual':True,'encoder_frozen':True,
              'selection':'fixed-postprocessing validation Spell F1, then short recall; epoch0 is eligible',
              'boundary_training':'candidate CE plus entry/exit transition CE conditioned on frozen 5fps anchors',
              'objective':'state CE + weighted transition CE + .1 preservation KL + .05 repair-gate BCE; no CRF backprop',
              'gate_supervision':'five training matches only; in-sample frozen 5fps errors, no new OOF fitting',
              'postprocessing_fixed':post,'calibration_fixed':calibration,
              'test_status':'previously inspected holdout, retrospective comparison'}
    write_json(args.output/'manifest.json',manifest)
    encoder=Refiner(config,N_FEATURES).to(device)
    encoder.load_state_dict(torch.load(args.run/'best.pt',map_location=device,weights_only=True))
    encoder.eval().requires_grad_(False)
    training=[];cache_seconds=[]
    for match in source['split']['train']:
        tick=perf_counter();halves,masks,stored=cache_match(args,config,match,encoder,device,True)
        for h,mask,cache in zip(halves,masks,stored,strict=True):
            data=frame_features(h,mask,*cache,post);training.append(data)
            h.xy=None
        cache_seconds.append(perf_counter()-tick)
        print(f'Frozen cache stage: {len(cache_seconds)}/5 train matches complete',flush=True)
    val_halves,val_masks,val_stored=cache_match(args,config,source['split']['validation'],encoder,device,False)
    validation=[(h,m,c,frame_features(h,m,*c,post)) for h,m,c in zip(val_halves,val_masks,val_stored,strict=True)]
    arrays={k:np.concatenate([d[k] for d in training]) for k in training[0] if k!='rows'}
    del training
    width=arrays['x'].shape[1]
    count=np.bincount(arrays['change'][arrays['change']>=0],minlength=3)
    weights_np=np.sqrt(count[0]/np.maximum(1,count)).clip(1,8).astype(np.float32)
    boundary_count=int(arrays['boundary'].sum())
    print(f'MLP training examples={len(arrays["x"]):,}; entry/exit boundary examples={boundary_count}; params input={width}',flush=True)
    # Measure small-head updates on CPU and GPU; feature extraction always uses the requested GPU.
    probe={k:v[:args.frame_batch] for k,v in arrays.items()}
    benchmarks={}
    for target in (torch.device('cpu'),device):
        if str(target) in benchmarks:continue
        head=ResidualMLP(width).to(target);opt=torch.optim.AdamW(head.parameters(),lr=1e-3,weight_decay=1e-4)
        b=tensors(probe,target);w=torch.from_numpy(weights_np).to(target);times=[]
        for step in range(4):
            sync(target);tick=perf_counter();opt.zero_grad(set_to_none=True)
            loss=objective(head,b,w);loss.backward();opt.step();sync(target)
            if step:times.append(perf_counter()-tick)
        benchmarks[str(target)]=float(np.median(times))
    head_device=torch.device(min(benchmarks,key=benchmarks.get))
    write_json(args.output/'benchmark.json',{'head_update_seconds':benchmarks,'chosen_head_device':str(head_device),
               'encoder_device':str(device),'train_examples':len(arrays['x']),'boundary_examples':boundary_count,
               'frames_per_update':args.frame_batch,'feature_cache_seconds':perf_counter()-started})
    print(f'Head benchmark seconds/update: {benchmarks}; fastest={head_device}',flush=True)
    torch.manual_seed(100);head=ResidualMLP(width).to(head_device)
    optimizer=torch.optim.AdamW(head.parameters(),lr=1e-3,weight_decay=1e-4)
    tensor_data=tensors(arrays,head_device);del arrays
    weight=torch.from_numpy(weights_np).to(head_device)
    baseline,_,_=evaluate(head,validation,post,calibration,head_device,args.frame_batch)
    expected=selected['validation_metrics']
    if not np.isclose(baseline['spellF1'],expected['spellF1']):raise AssertionError('Zero-init baseline does not reproduce prior correction')
    best=(baseline['spellF1'],baseline['short_recall_iou_0_5']);bad=0;history=[];best_epoch=0
    save_checkpoint(args.output/'best_mlp.pt',head.state_dict())
    write_json(args.output/'best_validation.json',{'epoch':0,'metrics':baseline})
    size=len(tensor_data['x']);steps=math.ceil(size/args.frame_batch)
    projected=benchmarks[str(head_device)]*steps
    print(f'Estimated head-only training per epoch {projected:.2f}s + validation/decoding; max {args.epochs} epochs',flush=True)
    for epoch in range(1,args.epochs+1):
        tick=perf_counter();head.train();order=torch.randperm(size,device=head_device);losses=[]
        for first in range(0,size,args.frame_batch):
            idx=order[first:first+args.frame_batch];b={k:v[idx] for k,v in tensor_data.items()}
            optimizer.zero_grad(set_to_none=True);loss=objective(head,b,weight)
            if not torch.isfinite(loss):raise FloatingPointError('Nonfinite MLP objective')
            loss.backward();torch.nn.utils.clip_grad_norm_(head.parameters(),5);optimizer.step();losses.append(float(loss.detach().cpu()))
        quality,pred,_=evaluate(head,validation,post,calibration,head_device,args.frame_batch)
        key=(quality['spellF1'],quality['short_recall_iou_0_5'])
        if key>best:
            best,bad,best_epoch=key,0,epoch;save_checkpoint(args.output/'best_mlp.pt',head.state_dict())
            write_json(args.output/'best_validation.json',{'epoch':epoch,'metrics':quality})
        else:bad+=1
        if bad and bad%2==0:
            for group in optimizer.param_groups:group['lr']=max(1e-5,group['lr']*.5)
        row={'epoch':epoch,'loss':float(np.mean(losses)),'validation_spellF1':key[0],
             'short_recall':key[1],'bad_epochs':bad,'seconds':perf_counter()-tick,'lr':optimizer.param_groups[0]['lr']}
        history.append(row);pd.DataFrame(history).to_csv(args.output/'training_history.csv',index=False)
        save_checkpoint(args.output/'last_mlp.pt',{'model':head.state_dict(),'optimizer':optimizer.state_dict(),
                        'epoch':epoch,'history':history,'torch_rng':torch.get_rng_state(),'best':best,'best_epoch':best_epoch})
        print(json.dumps(row),flush=True)
        if epoch==1:
            print(f'Measured epoch {row["seconds"]:.1f}s; maximum remaining {(args.epochs-epoch)*row["seconds"]/60:.1f}min; early-stop patience={args.patience}',flush=True)
        if bad>=args.patience:break
    # Test labels/cache are first accessed after the validation checkpoint is frozen.
    head.load_state_dict(torch.load(args.output/'best_mlp.pt',map_location=head_device,weights_only=True))
    test_halves,test_masks,test_stored=cache_match(args,config,source['split']['test'],encoder,device,False)
    test_bundles=[(h,m,c,frame_features(h,m,*c,post)) for h,m,c in zip(test_halves,test_masks,test_stored,strict=True)]
    result,preds,_=evaluate(head,test_bundles,post,calibration,head_device,args.frame_batch)
    before=[];raw=[]
    for h in test_halves:
        d=pd.read_parquet(args.run/'correction_epoch29'/f'{h.key}.parquet')
        np.testing.assert_array_equal(d.frame,h.frame);before.append(d.corrected_selected.to_numpy());raw.append(d.epoch29_saved.to_numpy())
    conditions={'epoch29_raw':raw,'epoch29_corrected':before,'frozen_mlp_boundary':preds}
    rows=[]
    for name,p in conditions.items():
        totals={}
        for h,q in zip(test_halves,p,strict=True):
            for k,v in error_summary(h,q).items():totals[k]=totals.get(k,0)+v
        rows.append({'condition':name,**report(test_halves,p),**totals})
    for i,h in enumerate(test_halves):
        data={'match':h.match,'half':h.number,'frame':h.frame,'seconds':h.time,'truth':h.labels,
              'dead':h.coarse['base']==2,'candidate':test_masks[i],'base_5fps':h.coarse['base']}
        for name,p in conditions.items():data[name]=p[i]
        pd.DataFrame(data).to_parquet(args.output/f'{h.key}.parquet',index=False)
        pd.DataFrame(data).to_csv(args.output/f'{h.key}.csv',index=False,encoding='utf-8-sig')
    pd.DataFrame(rows).to_csv(args.output/'holdout_comparison.csv',index=False)
    assert digest(args.run/'best.pt')==sha
    write_json(args.output/'results.json',{'metrics':rows,'best_epoch':best_epoch,'completed_epochs':len(history),
               'baseline_validation':baseline,'best_validation':list(best),'checkpoint_frozen':True,
               'total_seconds':perf_counter()-started,'encoder_device':str(device),'head_device':str(head_device),
               'head_parameters':sum(p.numel() for p in head.parameters()),'boundary_examples':boundary_count})
    print(pd.DataFrame(rows).to_string(index=False),flush=True)
    print(f'Completed in {perf_counter()-started:.1f}s; best residual epoch={best_epoch}',flush=True)


if __name__=='__main__':main()
