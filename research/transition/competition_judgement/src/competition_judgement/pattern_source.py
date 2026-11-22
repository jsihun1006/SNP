"""Immutable Stage B source with explicit path support and fixed-probe access."""
import numpy as np
import pandas as pd
from .evidence_source import EvidenceSource
from .temporal_matching import adjacent_matches


class PatternSource(EvidenceSource):
    @classmethod
    def from_memory(cls,base,clusters,legacy):
        """Adapt synthetic/perturbed old TemporalSource without patching old code."""
        obj=object.__new__(cls);obj.__dict__.update(base.__dict__)
        obj.evidence_config=legacy;obj.clusters=clusters.reset_index(drop=True)
        c=obj.clusters;obj.cluster_frame=np.searchsorted(obj.ids,c.frame.to_numpy())
        obj.flat_xy=c[['cluster_centroid_x','cluster_centroid_y']].to_numpy()
        obj.flat_nh=c.n_home.to_numpy();obj.flat_na=c.n_away.to_numpy()
        obj.flat_D=c.cluster_max_pairwise_distance_m.to_numpy();obj.flat_R=c.cluster_radius_m.to_numpy()
        obj.flat_density=c.cross_pair_density.to_numpy()
        obj.component_radius=np.full(obj.component_rows.shape,np.nan,np.float32);obj.component_diameter=obj.component_radius.copy()
        ri=(c.radius_m.to_numpy()==5).astype(int);ki=c.cluster_id.to_numpy(int);ci=obj.cluster_frame
        obj.component_radius[ci,ri,ki]=c.cluster_radius_m;obj.component_diameter[ci,ri,ki]=c.cluster_max_pairwise_distance_m
        match=adjacent_matches(base,c,base.config)
        obj.prev_status=match.match_prev_status.to_numpy();obj.next_status=match.match_next_status.to_numpy()
        obj.prev_candidates=match.match_prev_candidate_count.to_numpy(float);obj.next_candidates=match.match_next_candidate_count.to_numpy(float)
        for direction in ('prev','next'):
            status=match[f'match_{direction}_status'].to_numpy();good=status=='reciprocal_unique_geometry'
            target=np.full(len(c),-1,int)
            target[good]=obj.component_rows[np.searchsorted(obj.ids,match.loc[good,f'match_{direction}_frame']),ri[good],match.loc[good,f'match_{direction}_cluster_id'].to_numpy(int)]
            setattr(obj,direction,target)
        return obj

    def path_metadata(self,selected,centers,radii):
        selected=np.asarray(selected,int);n=len(selected);path=np.full((n,21),-1,int);path[:,10]=selected
        for sign,links in ((-1,self.prev),(1,self.next)):
            prior=selected.copy()
            for step in range(1,11):
                prior=np.where(prior>=0,links[np.maximum(prior,0)],-1)
                path[:,10+sign*step]=prior
        known=path>=0;safe=np.maximum(path,0);row=self.cluster_frame[safe];centerrow=self.cluster_frame[np.maximum(selected,0)]
        support=known&(self.ids[row]==self.ids[centerrow,None]+np.arange(-10,11))
        support&=(self.periods[row]==self.periods[centerrow,None])&(self.roles[row]==self.roles[centerrow,None])
        support&=self.strict[row]&(self.nh[row]==self.nh[centerrow,None])&(self.na[row]==self.na[centerrow,None])
        support&=(self.flat_nh[safe]==self.flat_nh[np.maximum(selected,0),None])&(self.flat_na[safe]==self.flat_na[np.maximum(selected,0),None])
        support&=(selected>=0)[:,None]
        inside=(np.sum((self.flat_xy[safe]-centers[:,None])**2,axis=-1)<=radii[:,None]**2)&known
        return dict(path_support=support,path_inside_anchor=inside.all(1)&support.all(1))

    def query(self,selected):
        s=super().query(selected);ch=self.clusters.iloc[selected]
        centers=ch[['cluster_centroid_x','cluster_centroid_y']].to_numpy()
        radii=np.maximum(6,ch.cluster_max_pairwise_distance_m.to_numpy()+3)
        s.update(self.path_metadata(selected,centers,radii));return s

    def query_fixed(self,frames,centers,radii,radius_codes,mapped):
        """Same clean physical probes; absent bridge NEVER forces a group identity."""
        rows=np.searchsorted(self.ids,frames);s=self.anchor_series(rows,centers,radii);index=s['index']
        cxy=self.component_xy[index,radius_codes[:,None]]
        inside=(np.sum((cxy-centers[:,None,None,:])**2,axis=-1)<=radii[:,None,None]**2)&(self.component_rows[index,radius_codes[:,None]]>=0)
        s['nearby_count']=inside.sum(-1).astype(float)
        for name,vals in [('component_radius',self.component_radius),('component_diameter',self.component_diameter)]:
            maximum=np.max(np.where(inside,vals[index,radius_codes[:,None]],-np.inf),axis=-1)
            s[name]=np.where(inside.any(-1),maximum,np.nan)
        s['global_n_home']=self.nh[index];s['global_n_away']=self.na[index]
        safe=np.maximum(mapped,0);s.update(self.geometry_path(safe));unknown=mapped<0
        s['path_valid']&=~unknown
        s['path_reason']=np.where(unknown,'probe_mapping_unknown',s['path_reason'])
        for key in ('path_score','path_motion_m','path_diameter_range_m','path_radius_range_m','path_density_range'):
            s[key]=np.where(s['path_valid'],s[key],np.nan)
        for direction in ('prev','next'):
            s[f'{direction}_status']=np.where(unknown,'probe_mapping_unknown',getattr(self,f'{direction}_status')[safe])
            s[f'{direction}_candidate_count']=np.where(unknown,np.nan,getattr(self,f'{direction}_candidates')[safe])
        s.update(self.path_metadata(mapped,centers,radii))
        return s
