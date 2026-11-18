"""Fixed epoch-29 hybrid corrections, selected on validation before holdout scoring."""
from __future__ import annotations
import argparse
from itertools import product
import json
from pathlib import Path
from time import perf_counter
from types import SimpleNamespace
import numpy as np
import pandas as pd
import torch
from scipy.special import log_softmax, expit

from inputs import ROOT,N_FEATURES
from refiner import Refiner,decode,TRANSITION_IDS
from run import digest,write_json
from correct_5fps import report,snap
from correct_5fps_v2 import error_summary
from candidate25_scores import load_scores
from transportmer_ha_data import contiguous_runs


def base_logs(half,temperature):
    margin=np.clip(half.signals[:,0],0,.98)
    p=np.where(half.coarse['base']==1,.5+.49*margin,.5-.49*margin)
    raw=np.log(np.clip(np.c_[1-p,p],1e-6,1))
    return log_softmax(raw/temperature,axis=1)


def calibrate(halves,gates,logits):
    trials=[]
    for t5,t25 in product((.5,1.,2.,4.),repeat=2):
        nll5=nll25=count=0
        for h,mask,raw in zip(halves,gates['original'],logits,strict=True):
            valid=mask&np.isin(h.labels,(0,1))
            y=h.labels[valid]
            l5=base_logs(h,t5)[valid]
            l25=log_softmax(raw[valid,:2]/t25,axis=1)
            nll5-=l5[np.arange(len(y)),y].sum();nll25-=l25[np.arange(len(y)),y].sum();count+=len(y)
        trials.append({'base_temperature':t5,'specialist_temperature':t25,'nll':float((nll5+nll25)/max(1,count))})
    return min(trials,key=lambda r:r['nll']),trials


def path_score(path,state,change,left,right,right_change):
    pair=np.asarray(TRANSITION_IDS)
    value=state[np.arange(len(path)),path].sum()
    if len(path)>1:value+=change[np.arange(1,len(path)),pair[path[:-1],path[1:]]].sum()
    if left is not None:value+=change[0,pair[left,path[0]]]
    if right is not None:value+=right_change[pair[path[-1],right]]
    return float(value)


def pair_prune(path,state,change,left,right,right_change,minimum):
    if not minimum:return path
    result=path.copy()
    cuts=np.r_[0,np.flatnonzero(np.diff(result))+1,len(result)]
    # Assess an entire entry/exit pair; never delete solely because a spell is short.
    for j in range(1,len(cuts)-2):
        a,b=cuts[j:j+2]
        if b-a>=50 or result[a-1]!=result[b]:continue
        proposal=result.copy();proposal[a:b]=result[a-1]
        gain=path_score(result,state,change,left,right,right_change)-path_score(proposal,state,change,left,right,right_change)
        if gain/max(1,b-a)<minimum:result=proposal
    return result


def decode_setting(halves,gates,logits,calibration,setting):
    predictions=[h.coarse['base'].copy() for h in halves]
    masks=gates[setting['gate']]
    for index,(h,mask,raw) in enumerate(zip(halves,masks,logits,strict=True)):
        l5=base_logs(h,calibration['base_temperature'])
        l25=log_softmax(raw[:,:2]/calibration['specialist_temperature'],axis=1)
        transition=(raw[:,2:]-raw[:,2,None])/calibration['specialist_temperature']
        transition[:,1:]-=setting['switch_cost']
        for first,stop in contiguous_runs(h.coarse['base']!=2,h.frame):
            for local_a,local_b in contiguous_runs(mask[first:stop],h.frame[first:stop]):
                a,b=first+local_a,first+local_b
                if not np.isfinite(raw[a:b]).all():raise AssertionError('Missing scores in candidate patch')
                n=b-a
                w=np.full(n,setting['blend'],float)
                edge=setting['edge_frames']
                if edge:w*=np.minimum(1,np.minimum(np.arange(1,n+1),np.arange(n,0,-1))/(edge+1))
                state=(1-w[:,None])*l5[a:b]+w[:,None]*l25[a:b]
                change=transition[a:b].copy()
                left=int(h.coarse['base'][a-1]) if a>first else None
                right=int(h.coarse['base'][b]) if b<stop else None
                boundary=transition[b] if right is not None else change[-1]
                decoded=decode(state,change,left,right,boundary)
                original=h.coarse['base'][a:b]
                gain=path_score(decoded,state,change,left,right,boundary)-path_score(original,state,change,left,right,boundary)
                changed=int((decoded!=original).sum())
                if changed and gain/changed<setting['minimum_gain']:decoded=original.copy()
                if setting.get('join_threshold',0):
                    pair=np.asarray(TRANSITION_IDS)
                    weak=False
                    if left is not None and decoded[0]!=left and original[0]==left:
                        p=np.exp(log_softmax(raw[a,2:]))[pair[left,decoded[0]]]
                        weak|=p<setting['join_threshold']
                    if right is not None and decoded[-1]!=right and original[-1]==right:
                        p=np.exp(log_softmax(raw[b,2:]))[pair[decoded[-1],right]]
                        weak|=p<setting['join_threshold']
                    if weak:decoded=original.copy()
                decoded=pair_prune(decoded,state,change,left,right,boundary,setting.get('minimum_pair_gain',0))
                predictions[index][a:b]=decoded
    return predictions


def original_decode(halves,masks,logits):
    result=[h.coarse['base'].copy() for h in halves]
    for i,(h,mask,raw) in enumerate(zip(halves,masks,logits,strict=True)):
        for first,stop in contiguous_runs(h.coarse['base']!=2,h.frame):
            for aa,bb in contiguous_runs(mask[first:stop],h.frame[first:stop]):
                a,b=first+aa,first+bb
                result[i][a:b]=decode(raw[a:b,:2],raw[a:b,2:],int(h.coarse['base'][a-1]) if a>first else None,
                                      int(h.coarse['base'][b]) if b<stop else None,raw[b,2:] if b<stop else None)
    return result


def snapped(halves,predictions,info,masks,radius):
    result=[]
    for h,p,mask in zip(halves,predictions,masks,strict=True):
        q=snap(h,p,info[h.key],radius)
        q[~mask]=h.coarse['base'][~mask]
        result.append(q)
    return result


def quality(metrics,index):return metrics['spellF1'],metrics['short_recall_iou_0_5'] or 0,-index


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--run',type=Path,default=ROOT/'artifacts/possession/pathcrf/candidate25_dfl/historical_run')
    parser.add_argument('--output',type=Path,default=ROOT/'artifacts/possession/pathcrf/candidate25_dfl/historical_run/correction_epoch29')
    parser.add_argument('--logit-cache',type=Path,default=ROOT/'data/transition_final/candidate25_epoch29_logits')
    parser.add_argument('--device',choices=('cpu','mps'),default='cpu')
    parser.add_argument('--microbatch',type=int,default=16)
    parser.add_argument('--upstream',type=Path,default=Path('/private/tmp/pathcrf-official-dfl'))
    parser.add_argument('--cache',type=Path,default=ROOT/'data/external/dfl')
    parser.add_argument('--raw',type=Path,default=ROOT/'data/external/dfl/raw')
    parser.add_argument('--sources',type=Path,default=ROOT/'data/transition_final/source')
    parser.add_argument('--official-prepared',type=Path,default=ROOT/'data/pathcrf_dfl/transition_head_v1/prepared')
    parser.add_argument('--dead-scene',type=Path,default=ROOT/'data/pathcrf_dfl/scene_two_models_v1')
    parser.add_argument('--frozen-cache',type=Path,default=ROOT/'data/transition_final/frozen_7a9ebd31eba97ba5')
    args=parser.parse_args();args.output.mkdir(parents=True,exist_ok=True)
    started=perf_counter();torch.set_num_threads(4)
    manifest=json.loads((args.run/'manifest.json').read_text());config=manifest['parameters']
    chosen_epoch=json.loads((args.run/'best_validation.json').read_text())['epoch']
    if chosen_epoch!=29:raise ValueError('This correction run requires saved epoch 29.')
    sha=digest(args.run/'best.pt')
    write_json(args.output/'manifest.json',{'epoch':29,'checkpoint_sha256':sha,'source_run':str(args.run),
               'split':manifest['split'],'D_mask':'identical saved predicted D','new_model_training':False,
               'selection':'validation Spell F1, short recall, simpler setting; frozen before test is opened',
               'test_status':'previously inspected holdout, retrospective diagnostic',
               'not_applied':'short-spell extra training and checkpoint reselection require changing frozen weights',
               'inference_precision':'local FP32; saved Colab FP16 hard predictions retained as reference'})
    device=torch.device(args.device)
    model=Refiner(config,N_FEATURES).to(device)
    model.load_state_dict(torch.load(args.run/'best.pt',map_location=device,weights_only=True))
    validation,info,gates,logits=load_scores(args,config,manifest['split']['validation'],model,device)
    calibration,cal_grid=calibrate(validation,gates,logits)
    local_raw=original_decode(validation,gates['original'],logits)
    base=[h.coarse['base'].copy() for h in validation]
    original_scores=report(validation,local_raw)
    print('Validation raw epoch29 local:',original_scores,flush=True)
    settings=[dict(zip(('gate','blend','edge_frames','switch_cost','minimum_gain'),x))
              for x in product(('original','wide','touch'),(.5,.75,1.),(0,10),(-.5,0.,.5),(0.,.1))]
    results=[];best_score=None;best_pred=None;best_setting=None
    tick=perf_counter()
    for i,s in enumerate(settings):
        pred=decode_setting(validation,gates,logits,calibration,s)
        m=report(validation,pred);results.append({'parameters':s,'metrics':m})
        if best_score is None or quality(m,i)>best_score:
            best_score=quality(m,i);best_pred=pred;best_setting=s
        if (i+1)%36==0:print(f'Validation decoding: {i+1}/{len(settings)} settings',flush=True)
    # The unchanged original hybrid is also a valid candidate.
    if quality(original_scores,-1)>best_score:
        best_setting={'source':'original'};best_pred=local_raw
    post=[];post_preds=[]
    for join,paired,radius in product((0.,.3,.6),(0.,.05,.15),(0,5,10)):
        setting={**best_setting,'join_threshold':join,'minimum_pair_gain':paired,'snap_radius_frames':radius}
        if best_setting.get('source')=='original':
            # Apply acceptance/seam/pair corrections using the original raw score scale.
            start_pred=local_raw
        else:start_pred=decode_setting(validation,gates,logits,calibration,setting)
        gate_name=setting.get('gate','original')
        pred=snapped(validation,start_pred,info,gates[gate_name],radius)
        post.append({'parameters':setting,'metrics':report(validation,pred)})
        post_preds.append(pred)
    pick=max(range(len(post)),key=lambda i:quality(post[i]['metrics'],i))
    selected=post[pick]['parameters']
    selected_metrics=post[pick]['metrics']
    # Retain unmodified specialist if all changes lose validation performance.
    if quality(original_scores,-1)>quality(post[pick]['metrics'],pick):
        selected={'source':'original','snap_radius_frames':0}
        selected_metrics=original_scores
    snap_trials=[]
    for radius in (0,5,10):
        pred=snapped(validation,local_raw,info,gates['original'],radius)
        snap_trials.append(report(validation,pred))
    snap_pick=max(range(3),key=lambda i:quality(snap_trials[i],i))
    snap_radius=(0,5,10)[snap_pick]
    write_json(args.output/'selected.json',{'parameters':selected,'calibration':calibration,
               'validation_metrics':selected_metrics,'snap_only_radius':snap_radius})
    write_json(args.output/'validation.json',{'calibration_grid':cal_grid,'raw':original_scores,'grid':results,'post_grid':post})
    validation_select_seconds=perf_counter()-tick
    print('Frozen before test:',selected,'calibration:',calibration,flush=True)
    # Only now access the evaluation match and its ground truth.
    test,test_info,test_gates,test_logits=load_scores(args,config,manifest['split']['test'],model,device)
    saved=[]
    for h in test:
        table=pd.read_parquet(args.run/f'estimated_D_primary_{h.key}.parquet')
        np.testing.assert_array_equal(table.frame,h.frame);np.testing.assert_array_equal(table.truth,h.labels)
        np.testing.assert_array_equal(table.base,h.coarse['base']);np.testing.assert_array_equal(table.candidate,test_gates['original'][len(saved)])
        saved.append(table.refined.to_numpy())
    local=original_decode(test,test_gates['original'],test_logits)
    final=local if selected.get('source')=='original' else decode_setting(test,test_gates,test_logits,calibration,selected)
    final_masks=test_gates[selected.get('gate','original')]
    final=snapped(test,final,test_info,final_masks,selected.get('snap_radius_frames',0))
    predictions={'base_5fps':[h.coarse['base'].copy() for h in test], 'epoch29_saved':saved,
                 'epoch29_local_precision_check':local,'corrected_selected':final}
    # Predeclared reference: touch snapping alone, with the radius chosen on validation.
    predictions['snap_only']=snapped(test,saved,test_info,test_gates['original'],snap_radius)
    rows=[]
    for name,pred in predictions.items():
        error={}
        for h,p in zip(test,pred,strict=True):
            np.testing.assert_array_equal(p==2,h.coarse['base']==2)
            for k,v in error_summary(h,p).items():error[k]=error.get(k,0)+v
        rows.append({'condition':name,**report(test,pred),**error})
    for i,h in enumerate(test):
        np.testing.assert_array_equal(final[i][~final_masks[i]],h.coarse['base'][~final_masks[i]])
        data={'match':h.match,'half':h.number,'frame':h.frame,'seconds':h.time,'truth':h.labels,
              'dead':h.coarse['base']==2,'candidate_original':test_gates['original'][i],'candidate_corrected':final_masks[i]}
        for name,pred in predictions.items():data[name]=pred[i]
        pd.DataFrame(data).to_parquet(args.output/f'{h.key}.parquet',index=False)
        pd.DataFrame(data).to_csv(args.output/f'{h.key}.csv',index=False,encoding='utf-8-sig')
    pd.DataFrame(rows).to_csv(args.output/'holdout_comparison.csv',index=False,encoding='utf-8-sig')
    pd.DataFrame([{'parameter':k,'value':v} for k,v in selected.items()]).to_csv(args.output/'parameters.csv',index=False)
    changed=sum(int((a!=b).sum()) for a,b in zip(saved,local,strict=True))
    assert digest(args.run/'best.pt')==sha
    write_json(args.output/'results.json',{'metrics':rows,'selected':selected,'calibration':calibration,
               'snap_only_radius':snap_radius,'local_vs_saved_prediction_differences':changed,
               'checkpoint_unchanged':True,'D_masks_identical':True,'total_seconds':perf_counter()-started,
               'validation_selection_seconds':validation_select_seconds})
    print(pd.DataFrame(rows).to_string(index=False),flush=True)
    print('Precision differences:',changed,'Total seconds:',perf_counter()-started,flush=True)


if __name__=='__main__':main()
