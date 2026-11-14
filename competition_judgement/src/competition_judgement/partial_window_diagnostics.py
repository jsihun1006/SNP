"""Hypothetical support predicates ONLY; production v2-2 is never modified."""
import numpy as np


def run_statistics(mask):
    """True-run lengths, per-window maxima/counts and offset histograms."""
    mask=np.asarray(mask,dtype=bool)
    if mask.ndim!=2:raise ValueError('Expected windows by offsets')
    before=np.pad(mask,((0,0),(1,0)))[:,:-1]
    after=np.pad(mask,((0,0),(0,1)))[:,1:]
    sr,sc=np.nonzero(mask&~before);er,ec=np.nonzero(mask&~after)
    assert np.array_equal(sr,er)
    lengths=ec-sc+1
    maximum=np.zeros(len(mask),int);np.maximum.at(maximum,sr,lengths)
    counts=np.bincount(sr,minlength=len(mask))
    return lengths,maximum,counts


def policy_predicates(valid,config,*,current_strict=None,count_changed=None,center_relation=None):
    valid=np.asarray(valid,dtype=bool);n,t=valid.shape;c=t//2
    if t!=101:raise ValueError('Stage A fixed 101-frame window')
    _,maximum,_=run_statistics(~valid)
    gates=dict(none=np.ones(n,bool),center_band_full=valid[:,c-10:c+11].all(1))
    for limit in config['gap_limits']:gates[f'max_gap_{limit}']=maximum<=limit
    if current_strict is not None:gates['current_strict']=np.asarray(current_strict,bool)
    if count_changed is not None:gates['constant_global_count']=~np.asarray(count_changed,bool)
    if center_relation is not None:gates['center_relation_valid']=np.asarray(center_relation,bool)
    result={}
    for fraction in config['partial_fractions']:
        minimum=int(np.ceil(c*fraction-1e-12))
        base=valid[:,c]&(valid[:,:c].sum(1)>=minimum)&(valid[:,c+1:].sum(1)>=minimum)
        for gate,condition in gates.items():result[(int(round(fraction*100)),gate)]=base&condition
    return result
