"""Dense execution of edge-aware GAT mathematics, then pooling and Bi-LSTM.

No graph label/clock/slot/engineered feature or clean target enters forward.
Dense masks compute precisely the sparse neighbor set from DynamicGraphBuilder.
"""
from dataclasses import dataclass

import torch
from torch import nn
import torch.nn.functional as F
from torch.utils.checkpoint import checkpoint


def dense_edge_attributes(x,team,pitch=(105.3,68.)):
    # [G,source,destination,8]; direction is destination - source.
    delta=x[:,None,:,:]-x[:,:,None,:]
    distance=torch.linalg.vector_norm(delta*x.new_tensor(pitch),dim=-1)
    opposite=(team[:,:,None]!=team[:,None,:]) & (team[:,:,None]>=0)&(team[:,None,:]>=0)
    same=(team[:,:,None]==team[:,None,:]) & (team[:,:,None]>=0)&(team[:,None,:]>=0)
    return torch.cat((delta,(distance/10)[...,None],same[...,None],opposite[...,None],
                      *[(opposite&(distance<=r+1e-5))[...,None] for r in (3,5,10)]),dim=-1)


class DenseEdgeGAT(nn.Module):
    """GATConv(heads=4,out=32,concat=False,edge_dim=8,dropout=0).

    Edge projection and edge attention are algebraically contracted before the
    data projection, avoiding an unnecessary [G,N,N,4,32] intermediate.
    """
    def __init__(self,in_channels):
        super().__init__()
        self.lin=nn.Linear(in_channels,128,bias=False)
        self.lin_edge=nn.Linear(8,128,bias=False)
        self.att_src=nn.Parameter(torch.empty(4,32));self.att_dst=nn.Parameter(torch.empty(4,32))
        self.att_edge=nn.Parameter(torch.empty(4,32));self.bias=nn.Parameter(torch.zeros(32))
        for p in (self.lin.weight,self.lin_edge.weight,self.att_src,self.att_dst,self.att_edge):nn.init.xavier_uniform_(p)

    def forward(self,x,adjacency,edge_attr,node_valid):
        h=self.lin(x).reshape(*x.shape[:-1],4,32)
        src=(h*self.att_src).sum(-1);dst=(h*self.att_dst).sum(-1)
        edge_kernel=torch.einsum('hco,hc->oh',self.lin_edge.weight.reshape(4,32,8),self.att_edge)
        scores=src[:,:,None,:]+dst[:,None,:,:]+edge_attr@edge_kernel
        scores=F.leaky_relu(scores,.2).masked_fill(~adjacency[...,None],-1e9)
        # Incoming source normalization for each destination/head.
        alpha=scores.softmax(dim=1)*adjacency[...,None]
        out=torch.einsum('gijh,gihc->gjhc',alpha,h).mean(dim=-2)+self.bias
        return out.masked_fill(~node_valid[...,None],0)


class GraphEncoder(nn.Module):
    def __init__(self):
        super().__init__()
        self.gat1=DenseEdgeGAT(2);self.gat2=DenseEdgeGAT(32)
        self.norm1=nn.LayerNorm(32);self.norm2=nn.LayerNorm(32)
        self.readout=nn.Sequential(nn.Linear(96,64),nn.LayerNorm(64))

    def forward(self,x,team,adjacency):
        valid=team>=0
        attr=dense_edge_attributes(x,team)
        h=F.elu(self.norm1(self.gat1(x,adjacency,attr,valid))).masked_fill(~valid[...,None],0)
        h=F.elu(self.norm2(self.gat2(h,adjacency,attr,valid))).masked_fill(~valid[...,None],0)
        pooled=[];observed=[]
        for side in (0,1):
            mask=team==side;count=mask.sum(1)
            pooled.append((h*mask[...,None]).sum(1)/count.clamp_min(1)[:,None])
            observed.append(count>0)
        maximum=h.masked_fill(~valid[...,None],-1e9).max(dim=1).values
        maximum=torch.where(valid.any(1)[:,None],maximum,0)
        home,away=pooled
        symmetric=self.readout(torch.cat((home+away,(home-away).abs(),maximum),dim=-1))
        return symmetric,torch.stack((home,away),dim=1),torch.stack(observed,dim=1)


@dataclass(frozen=True)
class BOutput:
    prediction: torch.Tensor
    embedding: torch.Tensor
    graph_embedding: torch.Tensor
    team_embedding: torch.Tensor        # [B,T,2,32], H then A
    team_observed: torch.Tensor         # hidden/pad both False


class GraphBiLSTM(nn.Module):
    def __init__(self,*,graph_chunk=512,checkpoint_graph=True):
        super().__init__()
        self.graph=GraphEncoder()
        self.mask_token=nn.Parameter(torch.randn(64)*.02)
        self.temporal=nn.LSTM(64,32,batch_first=True,bidirectional=True)
        self.reconstruction=nn.Sequential(nn.Linear(64,64),nn.ELU(),nn.Linear(64,13))
        self.graph_chunk,self.checkpoint_graph=graph_chunk,checkpoint_graph

    def forward(self,batch):
        if not torch.isfinite(batch.x).all():raise FloatingPointError("Nonfinite graph node input")
        if ((batch.mapping>=0)&(batch.artificial|batch.padding)).any() \
                or ((batch.mapping<0)&~(batch.artificial|batch.padding)).any():
            raise ValueError("Graph mapping exposes hidden data or loses visible observations")
        parts=[];teams=[];observed=[]
        for first in range(0,len(batch.x),self.graph_chunk):
            args=(batch.x[first:first+self.graph_chunk],batch.team[first:first+self.graph_chunk],batch.adjacency[first:first+self.graph_chunk])
            if self.training and self.checkpoint_graph:
                out=checkpoint(self.graph,*args,use_reentrant=False,preserve_rng_state=False)
            else:out=self.graph(*args)
            parts.append(out[0]);teams.append(out[1]);observed.append(out[2])
        shape=batch.mapping.shape
        if parts:
            pooled=torch.cat(parts);tp=torch.cat(teams);op=torch.cat(observed)
            index=batch.mapping.clamp_min(0)
            sequence=pooled[index].masked_fill((batch.mapping<0)[...,None],0)
            team_output=tp[index].masked_fill((batch.mapping<0)[...,None,None],0)
            obs=op[index] & (batch.mapping>=0)[...,None]
        else:
            sequence=self.mask_token.new_zeros((*shape,64))
            team_output=self.mask_token.new_zeros((*shape,2,32))
            obs=torch.zeros((*shape,2),device=sequence.device,dtype=torch.bool)
        sequence=torch.where(batch.artificial[...,None],self.mask_token,sequence)
        sequence=sequence.masked_fill(batch.padding[...,None],0)
        if not batch.padding.any():embedding,_=self.temporal(sequence)
        else:
            embedding=sequence*0
            for b in range(len(sequence)):
                times=torch.nonzero(~batch.padding[b]).flatten()
                if not len(times):continue
                left,right=int(times[0]),int(times[-1])+1
                if right-left!=len(times):raise ValueError("Padding cannot bridge record gaps")
                embedding[b,left:right]=self.temporal(sequence[b:b+1,left:right])[0][0]
        prediction=self.reconstruction(embedding).masked_fill(batch.padding[...,None],0)
        embedding=embedding.masked_fill(batch.padding[...,None],0)
        if not torch.isfinite(prediction).all() or not torch.isfinite(embedding).all():
            raise FloatingPointError("Nonfinite B-model output")
        return BOutput(prediction,embedding,sequence,team_output,obs)


def save_b_checkpoint(path,model,**metadata):
    torch.save(dict(version="model-b-v1",model_state={k:v.detach().cpu().clone() for k,v in model.state_dict().items()},
                    metadata=metadata,graph_chunk=model.graph_chunk,checkpoint_graph=model.checkpoint_graph),path)


def load_b_checkpoint(path,device="cpu"):
    stored=torch.load(path,map_location="cpu",weights_only=True)
    if stored.get("version")!="model-b-v1":raise ValueError("Unknown B checkpoint")
    model=GraphBiLSTM(graph_chunk=stored["graph_chunk"],checkpoint_graph=stored["checkpoint_graph"])
    model.load_state_dict(stored["model_state"],strict=True)
    return model.to(device),stored["metadata"]
