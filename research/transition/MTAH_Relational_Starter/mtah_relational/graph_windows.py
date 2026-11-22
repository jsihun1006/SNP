"""Raw-first graph windows with a visible-query-only, unlearned geometry cache."""
from collections import OrderedDict
from dataclasses import dataclass

import numpy as np
import torch

from .data_quality import assess_observations
from .dynamic_graph import build_graph
from .model_a_data import FeatureWindowDataset
from .windows import collate_targets


class VisibleGraphCache:
    def __init__(self,base,*,capacity=160000,k=3,radius=10.):
        self.base,self.capacity,self.k,self.radius=base,capacity,k,radius
        self.frames=OrderedDict();self.hits=self.misses=0

    def get_visible(self,row):
        row=int(row)
        if row in self.frames:
            self.hits+=1;self.frames.move_to_end(row);return self.frames[row]
        self.misses+=1
        frame=self.base.source.frame(row)
        xy,_=assess_observations(frame,self.base.source.pitch,self.base.policy)
        keep=np.isfinite(xy).all(axis=1)
        graph=build_graph(xy[keep],frame.team_codes[keep],self.base.source.pitch,k=self.k,radius=self.radius)
        self.frames[row]=graph
        if len(self.frames)>self.capacity:self.frames.popitem(last=False)
        return graph


@dataclass(frozen=True)
class GraphWindow:
    graphs: tuple
    artificial_mask: np.ndarray
    padding_mask: np.ndarray
    routing: object


class GraphWindowDataset:
    def __init__(self,base,geometry_cache,target_cache):
        self.base,self.geometry_cache=base,geometry_cache
        self.loss_accessor=FeatureWindowDataset(base,target_cache)

    def __len__(self):return len(self.base)
    def set_epoch(self,epoch):self.base.set_epoch(epoch)

    def __getitem__(self,item):
        routing,rows=self.base._routing(item)
        hidden=self.base._mask(routing);pad=routing.frame_ids<0
        graphs=[None]*len(hidden)
        # Decide raw-time corruption BEFORE any graph/node/topology query.
        for t,row in zip(np.flatnonzero(~pad),rows):
            if not hidden[t]:graphs[t]=self.geometry_cache.get_visible(row)
        return GraphWindow(tuple(graphs),hidden,pad,routing)

    def get_targets(self,item,*,window):
        return self.loss_accessor.get_targets(item,window=window)


@dataclass(frozen=True)
class GraphBatch:
    x: torch.Tensor                # unique visible graphs [G,Nmax,2]
    team: torch.Tensor             # -1 is padding, not a fake node
    adjacency: torch.Tensor        # [G,Nmax,Nmax], source,destination
    mapping: torch.Tensor          # [B,T], local graph index; hidden/pad=-1
    artificial: torch.Tensor
    padding: torch.Tensor

    def to(self,device):
        return GraphBatch(*(getattr(self,n).to(device) for n in self.__dataclass_fields__))


def pack_windows(windows):
    b,t=len(windows),len(windows[0].graphs)
    mapping=np.full((b,t),-1,dtype=np.int64)
    unique=[];lookup={}
    for i,w in enumerate(windows):
        for j,g in enumerate(w.graphs):
            if w.artificial_mask[j] or w.padding_mask[j]:
                if g is not None:raise AssertionError("Hidden/padded graph was exposed")
                continue
            if g is None:raise AssertionError("Visible graph is missing (empty graph must be explicit)")
            # Shared immutable observation object, not learned output caching.
            key=id(g)
            if key not in lookup:lookup[key]=len(unique);unique.append(g)
            mapping[i,j]=lookup[key]
    size=max([len(g.x) for g in unique]+[1])
    x=np.zeros((len(unique),size,2),dtype=np.float32)
    team=np.full((len(unique),size),-1,dtype=np.int64)
    adjacency=np.zeros((len(unique),size,size),dtype=bool)
    for i,g in enumerate(unique):
        n=len(g.x);x[i,:n]=g.x;team[i,:n]=g.team;adjacency[i,:n,:n]=g.adjacency
    return GraphBatch(torch.from_numpy(x),torch.from_numpy(team),torch.from_numpy(adjacency),torch.from_numpy(mapping),
                      torch.from_numpy(np.stack([w.artificial_mask for w in windows])),
                      torch.from_numpy(np.stack([w.padding_mask for w in windows])))


def graph_batch_arrays(dataset,indices,*,with_targets=True):
    windows=[dataset[int(i)] for i in indices]
    target=collate_targets([dataset.get_targets(int(i),window=w) for i,w in zip(indices,windows)]) if with_targets else None
    return pack_windows(windows),target,windows


def grouped_batches(index,positions,*,batch_size=64,group_size=16,seed=2027,epoch=0):
    """Shuffle local groups, retain all windows/masks, reset RNN per window.

    Four independent local 16-window groups fit the default 64-window batch.
    Reusing a frame's differentiable encoding within a batch is mathematically
    identical to evaluating that deterministic encoder repeatedly. This changes
    Adam minibatch order relative to A's fully shuffled individual windows.
    """
    if batch_size%group_size or group_size<1:raise ValueError("Invalid graph batching groups")
    groups=[];positions=np.asarray(positions)
    for block in np.unique(index.records['block_index'][positions]):
        local=positions[index.records['block_index'][positions]==block]
        groups.extend(local[first:first+group_size] for first in range(0,len(local),group_size))
    order=np.random.default_rng(seed+epoch).permutation(len(groups))
    per_batch=batch_size//group_size
    return [np.concatenate([groups[int(i)] for i in order[first:first+per_batch]])
            for first in range(0,len(order),per_batch)]
