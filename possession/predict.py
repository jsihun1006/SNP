"""possession: fixed 5fps PathCRF + epoch-29 candidate 25fps + frozen corrections."""
from __future__ import annotations
import argparse,hashlib,json,sys
from pathlib import Path
from types import SimpleNamespace
from time import perf_counter

PACKAGE=Path(__file__).resolve().parent
sys.path.insert(0,str(PACKAGE/'code'))

import numpy as np
import pandas as pd
import torch
from inputs import Half,HA,FEATURES,contention,coarse_predictions,gate,make_features
from network_25fps import Refiner,decode
from runtime import collate,inference_windows
from signals import native_features
from postprocess import decode_setting,snapped


def sha(path):
    with path.open('rb') as f:return hashlib.file_digest(f,'sha256').hexdigest()


def check_weights(parameters):
    for item in parameters['weights'].values():
        file=PACKAGE/item['file']
        if not file.is_file():raise FileNotFoundError(f'Missing local model weight: {file}')
        if sha(file)!=item['sha256']:raise ValueError(f'Weight checksum differs: {file}')


def external_dead(path,frames):
    data=pd.read_csv(path)
    np.testing.assert_array_equal(data.frame.to_numpy(),frames)
    if 'dead' in data:
        if not data.dead.isin([0,1,False,True]).all():raise ValueError('dead must be 0 or 1')
        return data.dead.to_numpy(bool)
    if 'label' in data and data.label.isin(['H','A','D']).all():return data.label.eq('D').to_numpy()
    raise ValueError('Dead-mask CSV needs frame,dead (0/1) or frame,label (H/A/D).')


def load_array_match(args,parameters):
    """Submission input: no team-possession ground truth is consumed."""
    config=parameters['network_25fps']
    with np.load(args.input,allow_pickle=False) as z:
        players=z['players'].astype(np.float32);ball=z['ball'].astype(np.float32);team=z['team'].astype(np.int8)
        frame=z['frame'].astype(np.int64);period=z['half'].astype(np.int8);seconds=z['seconds'].astype(np.float64)
        source_frame=z['source_frame'].astype(np.int64) if 'source_frame' in z else frame.copy()
        keepers=z['goalkeeper'].astype(np.int8) if 'goalkeeper' in z else np.full(len(team),-1,np.int8)
        if 'player_names' in z:names=z['player_names'].astype(str).tolist()
        else:names=[f'{"home" if t==0 else "away"}_{i+1}' for i,t in enumerate(team)]
        if args.dead:dead=external_dead(args.dead,frame)
        elif 'dead' in z:
            if not np.isin(z['dead'],[0,1,False,True]).all():raise ValueError('dead must contain only 0/1')
            dead=z['dead'].astype(bool)
        else:raise ValueError('Supply --dead CSV or the NPZ dead array from the existing D model.')
        fallback=z['fallback_ha'].astype(np.int8) if 'fallback_ha' in z else None
        pitch=z['pitch'].astype(float) if 'pitch' in z else np.array([105.,68.])
    n=len(frame)
    if n<1 or len(np.unique(source_frame))!=n:raise ValueError("Nonempty unique source-frame axis required")
    if fallback is not None and (fallback.shape!=(n,) or not np.isin(fallback[~dead],[0,1]).all()):raise ValueError("Invalid fallback_ha")
    np.testing.assert_array_equal(frame,np.arange(n))
    if players.shape!=(n,len(team),2) or ball.shape!=(n,2):raise ValueError('Incorrect tracking array shapes')
    if len(period)!=n or len(seconds)!=n or len(dead)!=n or len(source_frame)!=n:raise ValueError('Axis length mismatch')
    if not np.isin(team,[0,1]).all() or not np.isin(period,[0,1,2]).all():raise ValueError('team=0/1 and half=0/1/2 required')
    if len(set(names))!=len(names):raise ValueError('Player names must be unique tracks')
    for name,t in zip(names,team,strict=True):
        if not name.startswith('home_' if t==0 else 'away_'):raise ValueError('Player names must start with home_/away_')
    if (~dead & (period==0)).any():raise ValueError('Frames outside the two playing halves must have dead=1')
    scale=np.array([105.,68.])/pitch
    players*=scale;ball*=scale
    columns={'frame_id':source_frame,'period_id':period,'timestamp':seconds,'ball_x':ball[:,0],'ball_y':ball[:,1]}
    for i,name in enumerate(names):
        columns[name+'_x']=players[:,i,0];columns[name+'_y']=players[:,i,1]
    tracking=pd.DataFrame(columns)
    descriptors=[];rows=[]
    for half in (1,2):
        indices=np.flatnonzero(period==half)
        if not len(indices):continue
        if (np.diff(source_frame[indices])<=0).any() or (np.diff(seconds[indices])<=0).any():raise ValueError('Invalid half axes')
        descriptors.append((half,source_frame[indices]));rows.append(indices)
    if args.base_cache and args.base_cache.is_file():
        coarse=[]
        with np.load(args.base_cache,allow_pickle=False) as saved:
            for (number,axis),idx in zip(descriptors,rows,strict=True):
                np.testing.assert_array_equal(saved[f'h{number}_frame'],axis)
                base=saved[f'h{number}_base'].copy()
                np.testing.assert_array_equal(base==2,dead[idx])
                coarse.append((base,saved[f'h{number}_margin'].copy(),saved[f'h{number}_carrier'].copy(),0))
    else:
        # Missing 5fps paths may use an explicitly supplied H/A fallback. Never silently invent H.
        supplied=[fallback[idx] if fallback is not None else np.full(len(idx),-1,np.int8) for idx in rows]
        coarse=coarse_predictions(tracking,descriptors,[~dead[idx] for idx in rows],supplied,
                                  args.upstream,torch.device('cpu'),None)
        if fallback is None and any(item[3] for item in coarse):
            raise ValueError('5fps could not cover some in-play frames. Supply an explicit fallback_ha array or aligned base-cache.')
    halves=[]
    for (number,axis),idx,(base,margin,carrier,missing) in zip(descriptors,rows,coarse,strict=True):
        xy=np.concatenate((players[idx],ball[idx,None,:]),axis=1)
        teams=np.r_[team,np.int8(2)];roles=np.r_[keepers,np.int8(-1)]
        distance,closing=contention(xy,names+['ball'],teams,carrier,axis,base!=2)
        halves.append(Half(f'input_H{number}','input',number,axis,seconds[idx],np.full(len(idx),-1,np.int8),
                           xy,teams,roles,dead[idx],{'base':base,'fallback':int(missing)},np.c_[margin,distance,closing]))
    return halves,rows,frame


@torch.inference_mode()
def specialist_scores(model,halves,masks,config,device,batch_size):
    refs,patches=inference_windows(halves,masks,config)
    sums=[np.zeros((len(h.frame),5),np.float32) for h in halves]
    counts=[np.zeros(len(h.frame),np.float32) for h in halves]
    for first in range(0,len(refs),batch_size):
        selected=refs[first:first+batch_size]
        x,seen,_,_=collate(halves,selected,config)
        state,change=model(x.to(device),seen.to(device))
        output=torch.cat((state.float(),change.float()),-1).cpu().numpy()
        if not np.isfinite(output).all():raise FloatingPointError('Nonfinite 25fps output')
        for ref,value in zip(selected,output,strict=True):
            size=ref.stop-ref.start;weight=np.minimum(np.arange(1,size+1),np.arange(size,0,-1))
            sums[ref.half][ref.start:ref.stop]+=value[:size]*weight[:,None]
            counts[ref.half][ref.start:ref.stop]+=weight
    logits=[]
    for i,h in enumerate(halves):
        required=masks[i].copy()
        for index,a,b,first,stop in patches:
            if index==i and b<stop:required[b]=True
        if not (counts[i][required]>0).all():raise AssertionError('Candidate inference coverage hole')
        logits.append(np.divide(sums[i],counts[i][:,None],out=np.full_like(sums[i],np.nan),where=counts[i][:,None]>0))
    return logits




def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--input',type=Path,required=True,help='Whole-video tracking NPZ; no possession labels needed')
    parser.add_argument('--dead',type=Path,help='Aligned frame,dead CSV for --input')
    parser.add_argument('--base-cache',type=Path,help='Optional aligned frozen-5fps NPZ or DFL cache directory')
    parser.add_argument('--output',type=Path,default=PACKAGE/'outputs/possession.csv')
    parser.add_argument('--device',choices=('auto','cpu','mps','cuda'),default='cpu')
    parser.add_argument('--batch',type=int,default=16)
    parser.add_argument('--threads',type=int,default=4)
    args=parser.parse_args();started=perf_counter();torch.set_num_threads(args.threads)
    parameters=json.loads((PACKAGE/'parameters.json').read_text());check_weights(parameters)
    config=parameters['network_25fps'];args.upstream=PACKAGE/'vendor/pathcrf'
    sys.path.insert(0,str(args.upstream))
    if args.device=='auto':args.device='cuda' if torch.cuda.is_available() else 'mps' if torch.backends.mps.is_available() else 'cpu'
    device=torch.device(args.device)
    halves,rows,output_frame=load_array_match(args,parameters)
    masks=[gate(h,config['gate_threshold'],config['gate_margin_frames']) for h in halves]
    info={h.key:native_features(h) for h in halves}
    for half in halves:make_features(half,config,training=False)
    model=Refiner(config,14).to(device).eval()
    model.load_state_dict(torch.load(PACKAGE/parameters['weights']['specialist']['file'],map_location=device,weights_only=True))
    logits=specialist_scores(model,halves,masks,config,device,args.batch)
    final=decode_setting(halves,{'original':masks},logits,parameters['calibration'],parameters['correction'])
    final=snapped(halves,final,info,masks,parameters['correction']['snap_radius_frames'])
    combined=np.full(len(output_frame),2,np.int8)
    for h,p,mask,idx in zip(halves,final,masks,rows,strict=True):
        np.testing.assert_array_equal(p[~mask],h.coarse['base'][~mask])
        np.testing.assert_array_equal(p==2,h.coarse['base']==2)
        combined[idx]=p
    args.output.parent.mkdir(parents=True,exist_ok=True)
    pd.DataFrame({'frame':output_frame,'label':np.array(['H','A','D'])[combined]}).to_csv(args.output,index=False)
    print(f'possession: wrote {len(output_frame):,} frames to {args.output}; {perf_counter()-started:.1f}s')


if __name__=='__main__':main()
