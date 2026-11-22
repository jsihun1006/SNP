"""Frame-local observation graphs. No identities, zones, targets or time features."""
from dataclasses import dataclass

import numpy as np


@dataclass(frozen=True)
class GraphFrame:
    x: np.ndarray                  # normalized xy [N,2]
    team: np.ndarray               # [N], 0=H,1=A
    adjacency: np.ndarray          # [source,destination], exactly one self loop
    pitch: tuple = (105.3,68.)

    @property
    def edge_index(self):
        return np.stack(np.nonzero(self.adjacency)).astype(np.int64)

    @property
    def edge_attr(self):
        src,dst=self.edge_index
        delta=self.x[dst].astype(float)-self.x[src]
        distance=np.linalg.norm(delta*np.asarray(self.pitch),axis=1)
        opposite=self.team[src]!=self.team[dst]
        return np.column_stack([delta,distance/10,~opposite,opposite,
                                *[opposite&(distance<=r+1e-5) for r in (3,5,10)]]).astype(np.float32)


def build_graph(xy_m,team,pitch=(105.3,68.),*,k=3,radius=10.,tie_tolerance=1e-8):
    """kth-distance ties are all included; fallback-opponent ties likewise.

    Edges are bidirectional, deduplicated by a boolean adjacency matrix. Self
    loops are added once here (not automatically by the attention operator).
    Nonfinite observations are absent nodes. Finite bounds policy is upstream.
    """
    xy=np.asarray(xy_m,dtype=float).reshape(-1,2)
    team=np.asarray(team)
    if team.shape!=(len(xy),) or not np.isin(team,[0,1]).all() or k<0 or radius<=0 \
            or len(pitch)!=2 or not np.isfinite(pitch).all() or min(pitch)<=0:
        raise ValueError("Invalid graph observations/settings")
    keep=np.isfinite(xy).all(axis=1)
    xy,team=xy[keep],team[keep].astype(np.int8)
    n=len(xy);adj=np.eye(n,dtype=bool)
    distance=np.linalg.norm(xy[:,None]-xy[None,:],axis=-1)
    for i in range(n):
        mates=np.flatnonzero((team==team[i])&(np.arange(n)!=i))
        if k and len(mates):
            cutoff=np.partition(distance[i,mates],min(k,len(mates))-1)[min(k,len(mates))-1]
            adj[i,mates[distance[i,mates]<=cutoff+tie_tolerance]]=True
        others=np.flatnonzero(team!=team[i])
        if len(others):
            cutoff=max(radius,float(distance[i,others].min()))
            adj[i,others[distance[i,others]<=cutoff+tie_tolerance]]=True
    adj|=adj.T
    x=(xy/np.asarray(pitch)).astype(np.float32)
    for a in (x,team,adj):a.setflags(write=False)
    return GraphFrame(x,team,adj,tuple(pitch))
