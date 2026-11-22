"""Frame-local cache and support routing; no slot differences or identity links."""
from pathlib import Path

import numpy as np
from .quality import POLICY


class TemporalSource:
    def __init__(self,store,frame_table,manifest,config,cache_dir=None,clusters=None):
        self.ids,self.periods,self.pitch=store.frame_ids,store.periods,store.pitch
        self.config=config
        if not np.array_equal(self.ids,frame_table.frame) or not np.array_equal(self.periods,frame_table.period):
            raise ValueError("Temporal source frame/period axis mismatch")
        self.roles=np.array([manifest.split_of(int(f)) or "outside" for f in self.ids])
        self.strict=frame_table.strict_quality_valid.to_numpy(dtype=bool)
        self.nh=frame_table.n_home.to_numpy(dtype=int);self.na=frame_table.n_away.to_numpy(dtype=int)
        self.v2_strength=np.column_stack([frame_table[f"max_local_contact_strength_{r}m"] for r in (3,5)])
        xy=np.asarray(store.xy_m).copy()
        finite=np.isfinite(xy).all(axis=-1)
        excluded=((xy < -POLICY.bounds_margin_m).any(axis=-1) | (xy > np.asarray(self.pitch)+POLICY.bounds_margin_m).any(axis=-1)) & finite
        xy[~finite|excluded]=np.nan
        self.h=xy[:,store.team_codes==0];self.a=xy[:,store.team_codes==1]
        if not np.array_equal(np.isfinite(self.h).all(-1).sum(1),self.nh) or not np.array_equal(np.isfinite(self.a).all(-1).sum(1),self.na):
            raise ValueError("QA source counts disagree with v2-1")
        n,h,a=len(self.ids),self.h.shape[1],self.a.shape[1]
        def allocate(name,dtype):
            if cache_dir is None:return np.empty((n,h,a),dtype=dtype)
            dest=Path(cache_dir);dest.mkdir(parents=True,exist_ok=True)
            return np.lib.format.open_memmap(dest/f"{name}.npy",mode="w+",dtype=dtype,shape=(n,h,a))
        self.kernel=allocate("pair_kernel","float32")
        self.hard3=allocate("pair_3m","bool");self.hard5=allocate("pair_5m","bool")
        for start in range(0,n,2048):
            sl=slice(start,start+2048)
            d=np.linalg.norm(self.h[sl,:,None]-self.a[sl,None,:],axis=-1)
            self.hard3[sl]=d<=3.;self.hard5[sl]=d<=5.
            self.kernel[sl]=np.where(np.isfinite(d),np.exp(-d/config["kernel_decay_m"]),0)
        changes=np.r_[True,(np.diff(self.ids)!=1)|(np.diff(self.periods)!=0)|(self.roles[1:]!=self.roles[:-1])]
        starts=np.flatnonzero(changes);ends=np.r_[starts[1:]-1,n-1]
        self.first=np.empty(n,int);self.last=np.empty(n,int)
        for first,last in zip(starts,ends):self.first[first:last+1]=first;self.last[first:last+1]=last
        outside=self.periods==0
        self.first[outside]=np.flatnonzero(outside);self.last[outside]=np.flatnonzero(outside)
        self.component_xy=self.component_size=self.component_rows=None
        if clusters is not None:
            k=max(1,int(clusters.cluster_id.max()+1)) if len(clusters) else 1
            self.component_xy=np.full((n,2,k,2),np.nan,dtype=np.float32)
            self.component_size=np.zeros((n,2,k),dtype=np.int16)
            self.component_rows=np.full((n,2,k),-1,dtype=np.int64)
            ci=np.searchsorted(self.ids,clusters.frame.to_numpy())
            if (ci>=n).any() or not np.array_equal(self.ids[ci],clusters.frame):raise ValueError("Invalid cluster routing")
            ri=(clusters.radius_m.to_numpy()==5).astype(int);ki=clusters.cluster_id.to_numpy(dtype=int)
            self.component_xy[ci,ri,ki]=clusters[['cluster_centroid_x','cluster_centroid_y']]
            self.component_size[ci,ri,ki]=clusters.n_players
            self.component_rows[ci,ri,ki]=np.arange(len(clusters))

    def support(self,rows):
        radius=self.config["window_radius_frames"]
        offsets=np.arange(-radius,radius+1)
        ideal=rows[:,None]+offsets
        valid=(ideal>=self.first[rows,None])&(ideal<=self.last[rows,None])&(self.periods[rows,None]>0)
        # Invalid support gathers the CENTER sentinel, never another block's data.
        index=np.where(valid,ideal,rows[:,None])
        return offsets,index,valid

    def anchor_series(self,rows,centers,radii):
        offsets,index,records=self.support(rows)
        h=self.h[index];a=self.a[index]
        hi=np.sum((h-centers[:,None,None,:])**2,axis=-1)<=radii[:,None,None]**2
        ai=np.sum((a-centers[:,None,None,:])**2,axis=-1)<=radii[:,None,None]**2
        nh,na=hi.sum(-1),ai.sum(-1)
        observed=records & (nh>0)&(na>0)
        pairs=hi[:,:,:,None]&ai[:,:,None,:]
        k=np.where(pairs,self.kernel[index],0)
        maximum=k.max(axis=(-1,-2)) if k.shape[-1] and k.shape[-2] else np.zeros(index.shape)
        mean=np.divide(k.sum(axis=(-1,-2)),nh*na,out=np.full(index.shape,np.nan),where=(nh*na)>0)
        hard={}
        for r,mask in ((3,self.hard3),(5,self.hard5)):
            edges=pairs&mask[index]
            hard[f"edges_{r}m"]=edges.sum(axis=(-1,-2)).astype(float)
            hard[f"contact_size_{r}m"]=edges.any(-1).sum(-1)+edges.any(-2).sum(-1)
        same=(self.nh[index]==self.nh[rows,None])&(self.na[index]==self.na[rows,None])
        reliable=observed&self.strict[index]&self.strict[rows,None]&same
        distance=-self.config["kernel_decay_m"]*np.log(np.maximum(maximum,1e-30))
        distance[~observed]=np.nan
        return dict(offsets=offsets,index=index,record_support=records,observed=observed,reliable=reliable,
                    soft=maximum.astype(float),soft_pair_mean=mean,nearest_m=distance,
                    anchor_n_home=nh,anchor_n_away=na,global_count_same=same,
                    strict=self.strict[index],**hard)

    def nearby_components(self,index,centers,radii,radius_codes):
        if self.component_xy is None:return None
        ci=self.component_xy[index,radius_codes[:,None]]
        size=self.component_size[index,radius_codes[:,None]]
        inside=(np.sum((ci-centers[:,None,None,:])**2,axis=-1)<=radii[:,None,None]**2)&(size>0)
        return dict(nearby_cluster_count=inside.sum(-1),nearby_cluster_max_size=np.where(inside,size,0).max(-1))
