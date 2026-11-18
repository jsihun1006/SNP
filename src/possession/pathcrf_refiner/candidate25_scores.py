"""Read-only epoch-29 inference scores, preserving the original window averages."""
from pathlib import Path
import json
from time import perf_counter
import numpy as np
import torch
from scipy.ndimage import maximum_filter1d
from inputs import load_match, make_features, gate
from run import collate, inference_windows, digest, write_json
from correct_5fps import native_features
from transportmer_ha_data import contiguous_runs


def gate_options(halves, config, info):
    result={'original':[], 'wide':[], 'touch':[]}
    for half in halves:
        raw=gate(half,config['gate_threshold'],config['gate_margin_frames'])
        wide=gate(half,config['gate_threshold'],25)
        scene=info[half.key]
        anchors=scene['touch'] & (scene['contest'] | (half.signals[:,0] <= .65))
        extra=np.zeros(len(raw),bool)
        for a,b in contiguous_runs(half.coarse['base']!=2,half.frame):
            extra[a:b]=maximum_filter1d(anchors[a:b].astype(np.uint8),size=21,mode='constant')>0
        result['original'].append(raw)
        result['wide'].append(wide)
        result['touch'].append(raw|extra)
    return result


@torch.inference_mode()
def load_scores(args, config, match, model, device):
    started=perf_counter()
    halves=load_match(args,match,torch.device('cpu'))
    info={h.key:native_features(h) for h in halves}
    gates=gate_options(halves,config,info)
    args.logit_cache.mkdir(parents=True,exist_ok=True)
    path=args.logit_cache/f'{match}.npz'
    identity={'checkpoint':digest(args.run/'best.pt'),'device':str(device),'torch':str(torch.__version__),
              'feature_config':config,'score_code':digest(Path(__file__)),'microbatch':args.microbatch}
    metadata=path.with_suffix('.json')
    if path.exists() and metadata.exists() and json.loads(metadata.read_text())['identity']==identity:
        with np.load(path) as cache:
            logits=[]
            for h in halves:
                np.testing.assert_array_equal(cache[f'{h.key}_frame'],h.frame)
                logits.append(cache[f'{h.key}_logits'].copy())
        print(f'{match}: reused logits cache',flush=True)
        return halves,info,gates,logits
    for half in halves:make_features(half,config,training=False)
    primary_refs,_=inference_windows(halves,gates['original'],config)
    union=[np.logical_or.reduce([gates[k][i] for k in gates]) for i in range(len(halves))]
    expanded_refs,_=inference_windows(halves,union,config)
    sums=[np.zeros((len(h.frame),5),np.float32) for h in halves]
    counts=[np.zeros(len(h.frame),np.float32) for h in halves]
    model.eval()
    def accumulate(refs,allow):
        for first in range(0,len(refs),args.microbatch):
            batch=refs[first:first+args.microbatch]
            x,seen,_,_=collate(halves,batch,config)
            state,change=model(x.to(device),seen.to(device))
            out=torch.cat((state.float(),change.float()),-1).cpu().numpy()
            if not np.isfinite(out).all():raise FloatingPointError('Nonfinite inference scores')
            for ref,values in zip(batch,out,strict=True):
                n=ref.stop-ref.start
                blend=np.minimum(np.arange(1,n+1),np.arange(n,0,-1))
                if allow is not None:blend=blend*allow[ref.half][ref.start:ref.stop]
                sums[ref.half][ref.start:ref.stop]+=values[:n]*blend[:,None]
                counts[ref.half][ref.start:ref.stop]+=blend
            if first==0 or first+args.microbatch>=len(refs):
                print(f'{match}: inference {min(first+args.microbatch,len(refs))}/{len(refs)} windows',flush=True)
    accumulate(primary_refs,None)
    holes=[c==0 for c in counts]
    extras=[r for r in expanded_refs if (holes[r.half][r.start:r.stop]&union[r.half][r.start:r.stop]).any()]
    accumulate(extras,holes)
    logits=[];saved={}
    for i,half in enumerate(halves):
        if not (counts[i][union[i]]>0).all():raise AssertionError('Expanded gate score coverage hole')
        values=np.divide(sums[i],counts[i][:,None],out=np.full_like(sums[i],np.nan),where=counts[i][:,None]>0)
        logits.append(values)
        saved[f'{half.key}_frame']=half.frame
        saved[f'{half.key}_logits']=values
        half.features=None
    np.savez_compressed(path,**saved)
    write_json(metadata,{'identity':identity,'original_windows':len(primary_refs),'additional_windows':len(extras),
                         'seconds':perf_counter()-started,'precision':'FP32 CPU; original Colab hard labels kept separately'})
    print(f'{match}: cached scores in {perf_counter()-started:.1f}s',flush=True)
    return halves,info,gates,logits
