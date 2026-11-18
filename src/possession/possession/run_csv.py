"""Aligned tracking CSV + existing SNP predictions -> frozen possession models -> CSV.

Tracking CSV may use a broadcast frame axis: tactical_frame is the explicit join.
Unmapped frames and the supplied D mask are preserved, never guessed or shifted.
"""
from __future__ import annotations
import argparse
import hashlib
import json
import os
from pathlib import Path
import sys
from time import perf_counter
import xml.etree.ElementTree as ET

import numpy as np
import pandas as pd
import torch

from locations import ARTIFACT, PACKAGE, PARAMETERS, PATHCRF
sys.path.insert(0, str(PACKAGE))
import predict


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--tracking', type=Path, required=True)
    parser.add_argument('--snp', type=Path, required=True)
    parser.add_argument('--metadata', type=Path, required=True, help='ATD metadata XML for team membership')
    parser.add_argument('--output', type=Path, default=ARTIFACT/'inference/SNP_final.csv')
    parser.add_argument('--device', choices=('auto','mps','cpu','cuda'), default='auto')
    parser.add_argument('--batch', type=int, default=32)
    parser.add_argument('--threads', type=int, default=min(8,os.cpu_count() or 4))
    args = parser.parse_args()
    start = perf_counter()
    torch.set_num_threads(args.threads)
    torch.set_num_interop_threads(2)
    if args.device == 'auto':
        args.device = 'cuda' if torch.cuda.is_available() else 'mps' if torch.backends.mps.is_available() else 'cpu'
    device = torch.device(args.device)
    if args.device == 'mps':
        torch.backends.mha.set_fastpath_enabled(False)
    print(f'Device={device}; CPU threads={args.threads}; batch={args.batch}', flush=True)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    cache = args.output.parent/'snp_cache'
    cache.mkdir(exist_ok=True)
    snp = pd.read_csv(args.snp)
    if list(snp.columns) != ['frame','label'] or not snp.label.isin(['H','A','D']).all():
        raise ValueError('SNP must have frame,label with H/A/D predictions')
    np.testing.assert_array_equal(snp.frame, np.arange(len(snp)))
    tracking = pd.read_csv(args.tracking, usecols=lambda c: c in ('frame','tactical_frame','alignment_half')
                           or c.endswith(('_x_m','_y_m')))
    valid = tracking.tactical_frame.between(0,len(snp)-1) & tracking.alignment_half.isin([1,2])
    tracking = tracking.loc[valid].sort_values('tactical_frame')
    index = tracking.tactical_frame.to_numpy(np.int64)
    if len(np.unique(index)) != len(index): raise ValueError('Duplicate tactical_frame mappings')
    root = ET.parse(args.metadata).getroot()
    team_ids = [e.attrib['id'] for e in root.iter() if e.tag.split('}')[-1]=='Team']
    teams = {p.attrib['id']: team_ids.index(p.attrib['teamId']) for p in root.iter()
             if p.tag.split('}')[-1]=='Player'}
    ids = sorted(int(c.split('_')[1]) for c in tracking if c.startswith('player_') and c.endswith('_x_m'))
    ids = [i for i in ids if tracking[f'player_{i:02}_x_m'].notna().any()]
    n = len(snp)
    players = np.full((n,len(ids),2),np.nan,np.float32)
    ball = np.full((n,2),np.nan,np.float32)
    for j,i in enumerate(ids):
        players[index,j] = tracking[[f'player_{i:02}_x_m',f'player_{i:02}_y_m']].to_numpy(np.float32)
    ball[index] = tracking[['ball_x_m','ball_y_m']].to_numpy(np.float32)
    half = np.zeros(n,np.int8)
    half[index] = tracking.alignment_half.to_numpy(np.int8)
    covered = half > 0
    # The model sees only explicitly mapped frames; merge untouched SNP frames back later.
    dead = snp.label.eq('D').to_numpy() | ~covered
    team = np.array([teams[str(i)] for i in ids],np.int8)
    if not np.isin(team,[0,1]).all(): raise ValueError('Expected two teams in metadata')
    names = np.array([f'{"home" if t==0 else "away"}_{i}' for t,i in zip(team,ids)])
    # Pitch columns are already metres on a 105.3 x 68m pitch. The model rescales to 105 x 68.
    input_path = cache/'tracking.npz'
    np.savez(input_path,players=players,ball=ball,team=team,player_names=names,
             frame=np.arange(n),source_frame=np.arange(n),half=half,seconds=np.arange(n)/25.,
             dead=dead,pitch=np.array([105.3,68.]),fallback_ha=snp.label.map({'H':0,'A':1,'D':0}).to_numpy(np.int8))
    parameters=json.loads(PARAMETERS.read_text())
    predict.check_weights(parameters)
    config=parameters['network_25fps']
    sys.path.insert(0,str(PATHCRF))
    model_args=argparse.Namespace(input=input_path,dead=None,base_cache=None,upstream=PATHCRF)
    halves,rows,frames=predict.load_array_match(model_args,parameters)
    coarse_seconds=perf_counter()-start
    base_fields={}
    for h in halves:
        for key,value in [('frame',h.frame),('base',h.coarse['base']),('margin',h.signals[:,0])]:
            base_fields[f'h{h.number}_{key}']=value
    np.savez(cache/'base5_diagnostics.npz',**base_fields)
    print('5fps complete: '+str({h.key:{'inplay':int((~h.dead_prediction).sum()),'fallback':h.coarse['fallback']} for h in halves}),flush=True)
    masks=[predict.gate(h,config['gate_threshold'],config['gate_margin_frames']) for h in halves]
    info={h.key:predict.native_features(h) for h in halves}
    for h in halves: predict.make_features(h,config,training=False)
    refs,_=predict.inference_windows(halves,masks,config)
    print(f'25fps starting: {len(refs)} windows; candidate frames={sum(int(m.sum()) for m in masks)}; preparation={perf_counter()-start:.1f}s',flush=True)
    model=predict.Refiner(config,14).to(device).eval()
    model.load_state_dict(torch.load(ARTIFACT/parameters['weights']['specialist']['file'],map_location=device,weights_only=True))
    logits=predict.specialist_scores(model,halves,masks,config,device,args.batch)
    np.savez(cache/'specialist_scores.npz',**{h.key:l for h,l in zip(halves,logits)})
    result=predict.decode_setting(halves,{'original':masks},logits,parameters['calibration'],parameters['correction'])
    result=predict.snapped(halves,result,info,masks,parameters['correction']['snap_radius_frames'])
    final=snp.label.to_numpy().copy()
    for h,p,mask,idx in zip(halves,result,masks,rows,strict=True):
        np.testing.assert_array_equal(p[~mask],h.coarse['base'][~mask])
        np.testing.assert_array_equal(p==2,h.dead_prediction)
        final[idx]=np.array(['H','A','D'])[p]
    np.testing.assert_array_equal(final=='D',snp.label.eq('D'))
    np.testing.assert_array_equal(final[~covered],snp.label.to_numpy()[~covered])
    pd.DataFrame({'frame':frames,'label':final}).to_csv(args.output,index=False)
    report={'output':str(args.output),'device':str(device),'batch':args.batch,'threads':args.threads,
            'frames':n,'mapped_frames':int(covered.sum()),'unmapped_inplay_preserved':int((~covered&snp.label.ne('D')).sum()),
            'five_fps_seconds_including_preparation':coarse_seconds,
            'five_fps_fallback_frames':sum(h.coarse['fallback'] for h in halves),
            'candidate_frames':sum(int(m.sum()) for m in masks),'specialist_windows':len(refs),
            'changed_ha_frames':int((final!=snp.label.to_numpy()).sum()),'dead_mask_unchanged':True,
            'total_seconds':perf_counter()-start,'labels':pd.Series(final).value_counts().to_dict(),
            'parameters':parameters,'inputs':{}}
    for key,path in [('tracking',args.tracking),('snp',args.snp),('metadata',args.metadata)]:
        with path.open('rb') as f: digest=hashlib.file_digest(f,'sha256').hexdigest()
        report['inputs'][key]={'path':str(path),'sha256':digest}
    args.output.with_suffix('.json').write_text(json.dumps(report,indent=2,ensure_ascii=False))
    print(json.dumps({k:v for k,v in report.items() if k not in ('parameters','inputs')},ensure_ascii=False),flush=True)


if __name__=='__main__': main()
