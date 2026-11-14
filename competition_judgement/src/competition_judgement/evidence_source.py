"""Read frozen player/pair/cluster caches, query fixed spatial regions."""
import numpy as np
import pandas as pd
from .missingness_cache import ReadOnlyAnchorCache

CLUSTER_COLUMNS=['frame','period','radius_m','cluster_id','n_players','n_home','n_away',
    'strict_quality_valid','quality_tier','cluster_centroid_x','cluster_centroid_y','cluster_radius_m',
    'cluster_max_pairwise_distance_m','n_cross_edges','cross_pair_density','contact_strength']
MATCH_COLUMNS=['frame','radius_m','cluster_id']+[f'match_{d}_{k}' for d in ('prev','next') for k in
    ('status','frame','cluster_id','candidate_count')]


class EvidenceSource(ReadOnlyAnchorCache):
    def __init__(self,root,frames,manifest,clusters,matching,config):
        super().__init__(root,frames,manifest)
        self.evidence_config=config;self.clusters=clusters.reset_index(drop=True)
        self.flat_xy=clusters[['cluster_centroid_x','cluster_centroid_y']].to_numpy()
        self.flat_nh=clusters.n_home.to_numpy();self.flat_na=clusters.n_away.to_numpy()
        self.flat_D=clusters.cluster_max_pairwise_distance_m.to_numpy();self.flat_R=clusters.cluster_radius_m.to_numpy()
        self.flat_density=clusters.cross_pair_density.to_numpy()
        key=['frame','radius_m','cluster_id']
        if not np.array_equal(clusters[key].to_numpy(),matching[key].to_numpy()):raise ValueError('Frozen cluster/matching keys differ')
        if clusters.duplicated(key).any():raise ValueError('Duplicate cluster keys')
        self.cluster_frame=np.searchsorted(self.ids,clusters.frame.to_numpy())
        k=int(clusters.cluster_id.max())+1;n=len(self.ids)
        self.component_rows=np.full((n,2,k),-1,np.int32)
        ci=self.cluster_frame;ri=(clusters.radius_m.to_numpy()==5).astype(int);ki=clusters.cluster_id.to_numpy(int)
        self.component_rows[ci,ri,ki]=np.arange(len(clusters))
        self.component_xy=np.full((n,2,k,2),np.nan,np.float32)
        self.component_xy[ci,ri,ki]=clusters[['cluster_centroid_x','cluster_centroid_y']]
        self.component_radius=np.full((n,2,k),np.nan,np.float32);self.component_diameter=self.component_radius.copy()
        self.component_radius[ci,ri,ki]=clusters.cluster_radius_m
        self.component_diameter[ci,ri,ki]=clusters.cluster_max_pairwise_distance_m
        self.prev_status=matching.match_prev_status.to_numpy();self.next_status=matching.match_next_status.to_numpy()
        self.prev_candidates=matching.match_prev_candidate_count.to_numpy(float);self.next_candidates=matching.match_next_candidate_count.to_numpy(float)
        for direction in ('prev','next'):
            status=matching[f'match_{direction}_status'].to_numpy();target=np.full(len(clusters),-1,int)
            good=status=='reciprocal_unique_geometry'
            frame=np.searchsorted(self.ids,matching.loc[good,f'match_{direction}_frame'].to_numpy())
            ids=matching.loc[good,f'match_{direction}_cluster_id'].to_numpy(int)
            target[good]=self.component_rows[frame,ri[good],ids]
            if (target[good]<0).any():raise ValueError('Unresolved frozen geometric correspondence')
            setattr(self,direction,target)

    def geometry_path(self,selected):
        """Ephemeral +/-10-frame corridor, NEVER a permanent group/player ID."""
        n=len(selected);path=np.full((n,21),-1,int);path[:,10]=selected
        reason=np.full(n,'ok',object)
        for sign,links,statuses in ((-1,self.prev,self.prev_status),(1,self.next,self.next_status)):
            previous=selected.copy();alive=np.ones(n,bool)
            for step in range(1,11):
                safe=np.maximum(previous,0);target=links[safe]
                ok=alive&(previous>=0)&(target>=0)
                label=statuses[safe]
                reason=np.where((reason=='ok')&~ok,np.where(alive,label,'incomplete_geometry_corridor'),reason)
                path[:,10+sign*step]=np.where(ok,target,-1);previous=np.where(ok,target,-1);alive=ok
        known=(path>=0).all(1);safe=np.maximum(path,0)
        rows=self.cluster_frame[safe];center=self.cluster_frame[selected]
        correct=(self.ids[rows]==self.ids[center,None]+np.arange(-10,11)).all(1)
        same_period=(self.periods[rows]==self.periods[center,None]).all(1)
        same_role=(self.roles[rows]==self.roles[center,None]).all(1)
        strict=self.strict[rows].all(1)
        same_global=((self.nh[rows]==self.nh[center,None])&(self.na[rows]==self.na[center,None])).all(1)
        ch=self.flat_nh[safe];ca=self.flat_na[safe]
        same_composition=((ch==ch[:,10,None])&(ca==ca[:,10,None])).all(1)
        valid=known&correct&same_period&same_role&strict&same_global&same_composition
        reason=np.where((reason=='ok')&~same_composition,'matched_component_composition_changed',reason)
        reason=np.where((reason=='ok')&~valid,'path_boundary_qc_or_count_unknown',reason)
        xy=self.flat_xy[safe]
        motion=np.linalg.norm(xy[:,-1]-xy[:,0],axis=1)
        D=np.ptp(self.flat_D[safe],axis=1)
        R=np.ptp(self.flat_R[safe],axis=1)
        density=np.ptp(self.flat_density[safe],axis=1)
        scale=self.evidence_config['scales']
        score=np.clip(motion/scale['motion_m'],0,1)*np.exp(-D/scale['motion_diameter_range_m']-R/scale['motion_radius_range_m']-density/scale['density'])
        out=dict(path_valid=valid,path_reason=reason,path_score=np.where(valid,score,np.nan),
            path_motion_m=motion,path_diameter_range_m=D,path_radius_range_m=R,path_density_range=density)
        for name in ('path_motion_m','path_diameter_range_m','path_radius_range_m','path_density_range'):out[name]=np.where(valid,out[name],np.nan)
        return out

    def query(self,selected):
        selected=np.asarray(selected,int);ch=self.clusters.iloc[selected]
        rows=self.cluster_frame[selected];centers=ch[['cluster_centroid_x','cluster_centroid_y']].to_numpy()
        radii=np.maximum(self.config['anchor_min_radius_m'],ch.cluster_max_pairwise_distance_m.to_numpy()+self.config['anchor_diameter_margin_m'])
        s=self.anchor_series(rows,centers,radii);index=s['index'];ri=(ch.radius_m.to_numpy()==5).astype(int)
        cxy=self.component_xy[index,ri[:,None]]
        inside=(np.sum((cxy-centers[:,None,None,:])**2,axis=-1)<=radii[:,None,None]**2)
        inside&=self.component_rows[index,ri[:,None]]>=0
        s['nearby_count']=inside.sum(-1).astype(float)
        for name,data in [('component_radius',self.component_radius),('component_diameter',self.component_diameter)]:
            vals=np.max(np.where(inside,data[index,ri[:,None]],-np.inf),axis=-1)
            s[name]=np.where(inside.any(-1),vals,np.nan)
        s['global_n_home']=self.nh[index];s['global_n_away']=self.na[index]
        s.update(self.geometry_path(selected))
        s['prev_status']=self.prev_status[selected];s['next_status']=self.next_status[selected]
        s['prev_candidate_count']=self.prev_candidates[selected];s['next_candidate_count']=self.next_candidates[selected]
        return s
