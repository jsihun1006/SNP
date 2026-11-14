"""Diagnostic same-frame graph models. No temporal state or input masking.

The original B20 is read-only. Trainable variants own independent encoder copies.
Added statistics are explicit diagnostic inputs, never a claimed new B/C model.
"""
import copy

import numpy as np
import torch
from torch import nn
import torch.nn.functional as F

from .model_b import dense_edge_attributes
from .reconstruction import masked_huber


COUNT_NAMES=('min_team_count_over11','max_team_count_over11','same_nonself_edges_over242',
             'cross_edges_over242','max_nonself_degree_over22')
RELATION_NAMES=('distance_bin_0_3','distance_bin_3_5','distance_bin_5_10','distance_bin_10_20',
                'distance_bin_over20','cross_distance_mean_overdiag','cross_distance_std_overdiag',
                'cross_distance_q10_overdiag','cross_distance_q50_overdiag','cross_distance_q90_overdiag')
STAT_NAMES=COUNT_NAMES+RELATION_NAMES
MODES=('frozen64','frozen96','same_frame64','count_readout','count_distance_readout')


def graph_statistics(graph):
    """Only node coordinates/team and topology; no target/scaler/quality access.

    The first three histogram cumulative sums reproduce cross-density targets.
    This overlap is intentional and must be reported as privileged diagnostics.
    """
    team=graph.team;n=len(team);same=team[:,None]==team[None,:]
    nonself=graph.adjacency.copy();np.fill_diagonal(nonself,False)
    nh,na=int((team==0).sum()),int((team==1).sum())
    counts=[min(nh,na)/11,max(nh,na)/11,float((nonself&same).sum())/242,
            float((nonself&~same).sum())/242,float(nonself.sum(0).max())/22 if n else 0.]
    xy=graph.x.astype(float)*np.asarray(graph.pitch)
    d=np.linalg.norm(xy[team==0,None,:]-xy[None,team==1,:],axis=-1).ravel()
    if len(d):
        cdf=[float((d<=r).mean()) for r in (3,5,10,20)]
        histogram=np.diff([0.,*cdf,1.])
        distribution=np.r_[d.mean(),d.std(),np.quantile(d,[.1,.5,.9])]/np.linalg.norm(graph.pitch)
    else:histogram=np.zeros(5);distribution=np.zeros(5)
    return np.asarray([*counts,*histogram,*distribution],dtype=np.float32)


def pack_graphs(graphs,device='cpu'):
    size=max([len(g.x) for g in graphs]+[1])
    x=np.zeros((len(graphs),size,2),np.float32)
    team=np.full((len(graphs),size),-1,np.int64)
    adjacency=np.zeros((len(graphs),size,size),bool)
    for i,g in enumerate(graphs):
        n=len(g.x);x[i,:n]=g.x;team[i,:n]=g.team;adjacency[i,:n,:n]=g.adjacency
    return tuple(torch.as_tensor(a,device=device) for a in (x,team,adjacency))


def node_pool96(encoder,x,team,adjacency):
    """Same GAT/LN/ELU and pooling as B, exposing the pre-readout96 vector."""
    valid=team>=0;attr=dense_edge_attributes(x,team)
    h=F.elu(encoder.norm1(encoder.gat1(x,adjacency,attr,valid))).masked_fill(~valid[...,None],0)
    h=F.elu(encoder.norm2(encoder.gat2(h,adjacency,attr,valid))).masked_fill(~valid[...,None],0)
    pooled=[]
    for side in (0,1):
        mask=team==side
        pooled.append((h*mask[...,None]).sum(1)/mask.sum(1).clamp_min(1)[:,None])
    maximum=h.masked_fill(~valid[...,None],-1e9).max(1).values
    maximum=torch.where(valid.any(1)[:,None],maximum,0)
    home,away=pooled
    return torch.cat((home+away,(home-away).abs(),maximum),-1),torch.stack(pooled,1)


class SameFrameReconstructor(nn.Module):
    def __init__(self,reference_encoder,mode,head_seed=2027):
        super().__init__()
        if mode not in MODES:raise ValueError('Unknown diagnostic variant')
        self.mode=mode
        self.encoder=copy.deepcopy(reference_encoder)
        frozen=mode.startswith('frozen')
        self.encoder.requires_grad_(not frozen)
        self.extra=5 if mode=='count_readout' else 15 if mode=='count_distance_readout' else 0
        if self.extra:
            old=self.encoder.readout[0]
            expanded=nn.Linear(96+self.extra,64)
            with torch.no_grad():
                expanded.weight.zero_();expanded.weight[:,:96].copy_(old.weight);expanded.bias.copy_(old.bias)
            self.encoder.readout[0]=expanded
        # Identical initialization for all four 64-input heads.
        with torch.random.fork_rng():
            torch.manual_seed(head_seed)
            width=96 if mode=='frozen96' else 64
            self.head=nn.Sequential(nn.Linear(width,64),nn.ELU(),nn.Linear(64,13))

    def forward(self,x=None,team=None,adjacency=None,statistics=None,*,frozen_features=None):
        if frozen_features is not None:
            if not self.mode.startswith('frozen'):raise ValueError('Trainable encoders cannot use detached embeddings')
            latent=frozen_features
        else:
            raw,_=node_pool96(self.encoder,x,team,adjacency)
            latent=raw if self.mode=='frozen96' else self.encoder.readout(
                torch.cat((raw,statistics[:,:self.extra]),-1) if self.extra else raw)
        prediction=self.head(latent)
        if not torch.isfinite(prediction).all():raise FloatingPointError('Nonfinite same-frame prediction')
        return prediction


def same_frame_huber(prediction,target,observed,quality):
    # Legacy Huber helper accepts a row selector. Here every visible frame is
    # selected for loss; this is NOT artificial corruption of the model input.
    return masked_huber(prediction,target,observed,quality,
                        torch.ones(prediction.shape[:-1],device=prediction.device,dtype=torch.bool))


def fit_ridge(x,y,alpha=.001):
    """Small frozen-information probe. All mean/std/coefficients fitted on Train."""
    x,y=np.asarray(x,dtype=float),np.asarray(y,dtype=float)
    xm,ym=x.mean(0),y.mean(0);xs=x.std(0);xs=np.where(xs>1e-8,xs,1.)
    a=(x-xm)/xs
    coef=np.linalg.solve(a.T@a+alpha*np.eye(a.shape[1]),a.T@(y-ym))
    return dict(mean_x=xm,scale_x=xs,mean_y=ym,coefficient=coef)


def predict_ridge(model,x):return (np.asarray(x)-model['mean_x'])/model['scale_x']@model['coefficient']+model['mean_y']


def regression_report(prediction,target,names):
    result={}
    for k,name in enumerate(names):
        y=target[:,k];error=prediction[:,k]-y;variance=float(np.sum((y-y.mean())**2))
        result[name]=dict(mae=float(np.abs(error).mean()),r2=float(1-np.sum(error**2)/variance) if variance>1e-15 else None)
    return result
