import numpy as np
from scipy.special import log_softmax
from data_utils import contiguous_runs
from network_25fps import decode, TRANSITION_IDS
from signals import snap

def base_logs(half,temperature):
    margin=np.clip(half.signals[:,0],0,.98)
    p=np.where(half.coarse['base']==1,.5+.49*margin,.5-.49*margin)
    raw=np.log(np.clip(np.c_[1-p,p],1e-6,1))
    return log_softmax(raw/temperature,axis=1)


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




def snapped(halves,predictions,info,masks,radius):
    result=[]
    for h,p,mask in zip(halves,predictions,masks,strict=True):
        q=snap(h,p,info[h.key],radius)
        q[~mask]=h.coarse['base'][~mask]
        result.append(q)
    return result
